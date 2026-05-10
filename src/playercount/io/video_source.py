"""Async video source — PyAV-backed frame producer with a thread bridge.

PyAV is a Pythonic wrapper around ffmpeg/libav. It gives us:

* In-process decode without piping ffmpeg over stdin (lower latency, no
  process management).
* Direct access to packet timestamps so we can emit accurate ``timestamp_s``
  per frame even with variable frame rates.

The decode loop itself is **synchronous C code** — there is no asyncio-aware
ffmpeg in the wild. We bridge to async with the standard pattern:

* A worker submitted to a single-thread executor runs ``_decode_loop``
  synchronously: opens the container, iterates frames, converts each to BGR.
* Each decoded frame is published to an :class:`asyncio.Queue` via
  ``loop.call_soon_threadsafe(queue.put_nowait, ...)`` (with a backpressure
  fallback when the queue is full — see :func:`_thread_put`).
* The :meth:`frames` async generator awaits the queue and yields tuples,
  staying on the event loop.
* On consumer-side cancellation (``GeneratorExit`` / ``CancelledError``), we
  set a :class:`threading.Event` to tell the worker to stop, drain the queue
  to release any backpressured worker, and let the worker exit naturally.

This is the only correct shape — an ``async def`` cannot ``yield`` from
inside an executor thread. Earlier docstrings in this file described that
impossible pattern; the audit caught it.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator
from concurrent.futures import Future
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np


@runtime_checkable
class VideoSource(Protocol):
    """An async iterator over decoded frames with metadata."""

    fps: float
    width: int
    height: int

    def frames(
        self, stride: int = 1
    ) -> AsyncIterator[tuple[int, float, np.ndarray]]:
        """Yield ``(frame_index, timestamp_s, bgr_ndarray)`` tuples."""
        ...

    async def close(self) -> None:
        """Release the underlying container/stream."""
        ...


class PyAvVideoSource:
    """PyAV-backed source with a producer-thread + asyncio.Queue bridge.

    Usage::

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            src = PyAvVideoSource(path, executor=pool)
            async for idx, ts, frame in src.frames(stride=1):
                ...
            await src.close()

    The executor must allow at least one worker; one is the right number for
    a single video source (PyAV decode is single-threaded per stream).
    """

    # Bounded; small because each item is a ~6 MB BGR frame at 1080p.
    _QUEUE_MAX = 4

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._path = Path(path)
        self._executor = executor
        self._container: object | None = None  # av.container.InputContainer once open
        self._stream: object | None = None  # av.video.stream.VideoStream
        # Populated by _open()
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        # Lifecycle state
        self._opened = False
        self._cancel_signal = threading.Event()
        self._decode_future: Future[None] | None = None

    # -- lifecycle -----------------------------------------------------------

    def _open(self) -> None:
        """Synchronously open the container and probe the video stream metadata.

        Called lazily inside the executor on the first ``frames()`` call so
        construction itself is cheap and never blocks the event loop.
        """
        import av  # type: ignore[import-not-found]

        if self._opened:
            return
        if not self._path.is_file():
            raise FileNotFoundError(f"video not found: {self._path}")
        self._container = av.open(str(self._path))
        streams = self._container.streams.video  # type: ignore[attr-defined]
        if not streams:
            raise RuntimeError(f"no video stream in {self._path}")
        self._stream = streams[0]
        avg_rate = self._stream.average_rate  # type: ignore[attr-defined]
        # average_rate is a Fraction; coerce safely with a fallback.
        self.fps = float(avg_rate) if avg_rate else 25.0
        cc = self._stream.codec_context  # type: ignore[attr-defined]
        self.width = int(cc.width)
        self.height = int(cc.height)
        self._opened = True

    def _close_sync(self) -> None:
        if self._container is not None:
            with suppress(Exception):
                self._container.close()  # type: ignore[attr-defined]
            self._container = None
        self._stream = None
        self._opened = False

    async def close(self) -> None:
        """Close the underlying container off-thread and stop any decode worker."""
        self._cancel_signal.set()
        if self._decode_future is not None:
            # Don't block forever — workers should exit promptly because of the
            # cancel signal. Future.result() honours the timeout.
            with suppress(Exception):
                await asyncio.get_running_loop().run_in_executor(
                    self._executor, self._decode_future.result, 2.0
                )
            self._decode_future = None
        if self._container is not None:
            await asyncio.get_running_loop().run_in_executor(
                self._executor, self._close_sync
            )

    # -- metadata (synchronous, cheap) --------------------------------------

    @staticmethod
    def probe(path: str | os.PathLike[str]) -> tuple[float, int, int]:
        """Read ``(fps, width, height)`` synchronously without starting decode.

        Lets callers (e.g. the CLI's annotator setup) size a VideoWriter
        without spinning up the decode worker.
        """
        import av  # type: ignore[import-not-found]

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"video not found: {p}")
        container = av.open(str(p))
        try:
            streams = container.streams.video
            if not streams:
                raise RuntimeError(f"no video stream in {p}")
            stream = streams[0]
            fps = float(stream.average_rate) if stream.average_rate else 25.0
            cc = stream.codec_context
            return fps, int(cc.width), int(cc.height)
        finally:
            container.close()

    # -- iteration -----------------------------------------------------------

    async def frames(
        self, stride: int = 1
    ) -> AsyncIterator[tuple[int, float, np.ndarray]]:
        """Yield ``(frame_index, timestamp_s, bgr_ndarray)`` tuples.

        Decoding runs in the executor; the generator awaits a bounded queue
        so backpressure from a full downstream pipeline stalls decoding
        cleanly.
        """
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[int, float, np.ndarray] | None] = asyncio.Queue(
            maxsize=self._QUEUE_MAX
        )
        # Reset cancel state for a fresh iteration
        self._cancel_signal = threading.Event()

        # Submit the decode worker. It runs to completion (or until cancel)
        # and pushes a None sentinel when it's done.
        self._decode_future = self._executor.submit(
            self._decode_loop, queue, stride, self._cancel_signal, loop
        )

        try:
            while True:
                item = await queue.get()
                if item is None:
                    # Either EOF or worker stopped after a cancel signal.
                    # Surface any worker exception by awaiting the future.
                    if self._decode_future.done():
                        exc = self._decode_future.exception()
                        if exc is not None:
                            raise exc
                    break
                yield item
        except (GeneratorExit, asyncio.CancelledError):
            # Consumer disengaged. Tell the worker to stop, drain the queue
            # (so a worker blocked in put_nowait can return), and let it exit.
            self._cancel_signal.set()
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            raise

    # -- decode worker (runs on executor thread) -----------------------------

    def _decode_loop(
        self,
        queue: asyncio.Queue[tuple[int, float, np.ndarray] | None],
        stride: int,
        cancel_signal: threading.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Synchronous decode loop. Pushes frames onto the asyncio queue."""
        try:
            self._open()
            assert self._container is not None
            i = 0
            for frame in self._container.decode(video=0):  # type: ignore[attr-defined]
                if cancel_signal.is_set():
                    break
                if i % stride != 0:
                    i += 1
                    continue
                bgr = frame.to_ndarray(format="bgr24")
                if frame.pts is not None and frame.time_base is not None:
                    ts = float(frame.pts * frame.time_base)
                else:
                    ts = i / max(self.fps, 1.0)
                _thread_put(loop, queue, (i, ts, bgr), cancel_signal)
                i += 1
        finally:
            # Push EOF so the consumer's await queue.get() resolves.
            _thread_put(loop, queue, None, cancel_signal)


# ---------------------------------------------------------------------------
# Thread → asyncio.Queue bridge with backpressure
# ---------------------------------------------------------------------------


def _thread_put(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[tuple[int, float, np.ndarray] | None],
    item: tuple[int, float, np.ndarray] | None,
    cancel_signal: threading.Event,
) -> None:
    """Put an item on an asyncio.Queue from a worker thread.

    We can't ``await`` on the worker thread, but we also don't want to
    silently drop frames if the queue is full. Use
    ``run_coroutine_threadsafe(queue.put(...), loop).result()`` so the worker
    blocks on disk read speed when downstream is slow — that's the
    backpressure that bounds RAM.

    ``cancel_signal`` is polled every wakeup so we don't hang forever if the
    consumer disengaged while we were blocked.
    """
    fut = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
    while not fut.done():
        if cancel_signal.is_set():
            fut.cancel()
            return
        try:
            fut.result(timeout=0.1)
            return
        except TimeoutError:
            continue
        except Exception:
            # Loop closed or queue gone. Either way, abort the put.
            return


__all__ = ["PyAvVideoSource", "VideoSource"]
