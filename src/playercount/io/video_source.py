"""PyAV-backed async video source.

PyAV decode is synchronous C code, so :class:`PyAvVideoSource` runs the
decode loop in a worker thread and bridges to the event loop through a
bounded :class:`asyncio.Queue`. The :meth:`PyAvVideoSource.frames` async
generator awaits the queue and yields ``(frame_index, timestamp_s, bgr)``
tuples.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Protocol, runtime_checkable

import av  # type: ignore[import-not-found]
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

    The executor must allow at least one worker; one is sufficient since
    PyAV decode is single-threaded per stream.
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
        self._container: object | None = None
        self._stream: object | None = None
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self._opened = False
        self._cancel_signal = threading.Event()
        self._decode_future: Future[None] | None = None

    # -- lifecycle -----------------------------------------------------------

    def _open(self) -> None:
        """Open the container and populate ``fps`` / ``width`` / ``height``.

        Called lazily inside the executor so construction never blocks the
        event loop.
        """
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
        """Stop the decode worker and close the container off-thread."""
        self._cancel_signal.set()
        if self._decode_future is not None:
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
        """Return ``(fps, width, height)`` without starting decode.

        Useful for sizing a downstream :class:`cv2.VideoWriter` before the
        decode worker is spun up.
        """
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

        Decoding runs on the executor; the generator awaits a bounded queue,
        so a slow consumer applies backpressure to the decode worker rather
        than buffering frames in memory.
        """
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[int, float, np.ndarray] | None] = asyncio.Queue(
            maxsize=self._QUEUE_MAX
        )
        self._cancel_signal = threading.Event()

        self._decode_future = self._executor.submit(
            self._decode_loop, queue, stride, self._cancel_signal, loop
        )

        try:
            while True:
                item = await queue.get()
                if item is None:
                    # End of stream or cancelled worker; surface any worker
                    # exception by inspecting the future.
                    if self._decode_future.done():
                        exc = self._decode_future.exception()
                        if exc is not None:
                            raise exc
                    break
                yield item
        except (GeneratorExit, asyncio.CancelledError):
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
        """Iterate decoded frames and publish them onto ``queue``."""
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
    """Put ``item`` on ``queue`` from a worker thread, blocking on backpressure.

    Uses :func:`asyncio.run_coroutine_threadsafe` so the worker waits when the
    queue is full instead of dropping frames. ``cancel_signal`` is polled on
    each wakeup to allow prompt teardown.
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
            # Loop closed or queue gone — abort.
            return


__all__ = ["PyAvVideoSource", "VideoSource"]
