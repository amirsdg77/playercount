"""Pipeline orchestrator: queues, executors, lifecycle, cancellation.

Two surfaces:

* :meth:`PipelineRunner.run` drains the entire video into the configured sink.
* :meth:`PipelineRunner.stream` yields :class:`FrameResult` items as they
  are produced (used by the NDJSON streaming endpoint).

Stages run as siblings inside an :class:`asyncio.TaskGroup`; any failure
cancels the rest. Detection, embedding, and decode each get their own
:class:`ThreadPoolExecutor` so a backlog in one cannot starve the others.
Bounded :class:`asyncio.Queue` instances between stages provide the
backpressure that bounds memory.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from playercount.aggregation import TrackState
from playercount.pipeline.stages import (
    aggregate_stage,
    decode_stage,
    detect_track_stage,
    embed_assign_stage,
    sink_stage,
)
from playercount.schemas import FrameResult

if TYPE_CHECKING:
    from playercount.config import Settings
    from playercount.detection import Detector
    from playercount.io import ResultSink, VideoSource
    from playercount.io.sinks import AnnotatedVideoSink
    from playercount.team_id import TeamClassifier
    from playercount.tracking import Tracker


# ---------------------------------------------------------------------------
# Components container
# ---------------------------------------------------------------------------


@dataclass
class PipelineComponents:
    """The five collaborators a :class:`PipelineRunner` needs."""

    source: VideoSource
    detector: Detector
    tracker: Tracker
    classifier: TeamClassifier
    sink: ResultSink


# ---------------------------------------------------------------------------
# Queue depths
# ---------------------------------------------------------------------------

# Raw 1080p BGR ≈ 6 MB/frame; depth 8 caps decode-side memory at ~50 MB.
_Q_DECODE_TO_DETECT = 8
# Per-frame after the detect stage flushes its batch.
_Q_DETECT_TO_EMBED = 8
# Per-frame after embedding; embedding is typically the bottleneck.
_Q_EMBED_TO_AGG = 8
# Final results are tiny (~1 KB each); generous buffer for streaming.
_Q_AGG_TO_SINK = 16


def _default_executor_workers(settings: Settings) -> int:
    """Worker count for the detect and embed pools.

    Default ``max(2, cpu_count // 2)``. Single CUDA context per process means
    real GPU parallelism is 1; the extra workers exist to overlap host
    pre/post-processing with the previous batch's GPU work.
    """
    if settings.executor_max_workers is not None:
        return settings.executor_max_workers
    return max(2, (os.cpu_count() or 1) // 2)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class PipelineRunner:
    """Owns queues, executors, and the async lifecycle of one analysis."""

    def __init__(
        self,
        settings: Settings,
        components: PipelineComponents,
        *,
        annotator: AnnotatedVideoSink | None = None,
        detect_executor: ThreadPoolExecutor | None = None,
        embed_executor: ThreadPoolExecutor | None = None,
        decode_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._settings = settings
        self._components = components
        self._annotator = annotator
        # Caller-supplied executors stay under the caller's ownership; we only
        # shut down the pools we built ourselves.
        self._owns_detect_executor = detect_executor is None
        self._owns_embed_executor = embed_executor is None
        self._owns_decode_executor = decode_executor is None
        self._detect_executor = detect_executor
        self._embed_executor = embed_executor
        self._decode_executor = decode_executor
        self._closed = False

    # -- lifecycle -----------------------------------------------------------

    def _ensure_executors(self) -> tuple[ThreadPoolExecutor, ThreadPoolExecutor, ThreadPoolExecutor]:
        """Lazily build the three pools we don't already own."""
        n = _default_executor_workers(self._settings)
        if self._detect_executor is None:
            self._detect_executor = ThreadPoolExecutor(
                max_workers=n, thread_name_prefix="playercount-detect"
            )
        if self._embed_executor is None:
            self._embed_executor = ThreadPoolExecutor(
                max_workers=n, thread_name_prefix="playercount-embed"
            )
        if self._decode_executor is None:
            # PyAV decode is single-threaded per source.
            self._decode_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="playercount-decode"
            )
        return self._detect_executor, self._embed_executor, self._decode_executor

    @property
    def detect_executor(self) -> ThreadPoolExecutor:
        if self._detect_executor is None:
            self._ensure_executors()
        assert self._detect_executor is not None
        return self._detect_executor

    @property
    def decode_executor(self) -> ThreadPoolExecutor:
        if self._decode_executor is None:
            self._ensure_executors()
        assert self._decode_executor is not None
        return self._decode_executor

    async def aclose(self) -> None:
        """Release source, sink, annotator, and any pools we own. Idempotent."""
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):
            await self._components.source.close()
        with suppress(Exception):
            await self._components.sink.aclose()
        if self._annotator is not None:
            with suppress(Exception):
                await self._annotator.aclose()
        for owned, pool in (
            (self._owns_detect_executor, self._detect_executor),
            (self._owns_embed_executor, self._embed_executor),
            (self._owns_decode_executor, self._decode_executor),
        ):
            if owned and pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)

    async def __aenter__(self) -> PipelineRunner:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- run-to-completion ---------------------------------------------------

    async def run(self) -> None:
        """Drive the full pipeline to completion.

        Returns when the sink has consumed the last frame. On any stage
        error, :meth:`aclose` is called and the first underlying exception is
        re-raised (the wrapping ``ExceptionGroup`` is unwrapped for callers).
        """
        det_exec, emb_exec, _ = self._ensure_executors()
        c = self._components
        s = self._settings

        q_decode: asyncio.Queue[Any] = asyncio.Queue(maxsize=_Q_DECODE_TO_DETECT)
        q_track: asyncio.Queue[Any] = asyncio.Queue(maxsize=_Q_DETECT_TO_EMBED)
        q_assign: asyncio.Queue[Any] = asyncio.Queue(maxsize=_Q_EMBED_TO_AGG)
        q_results: asyncio.Queue[Any] = asyncio.Queue(maxsize=_Q_AGG_TO_SINK)

        state = TrackState(window=s.track_window)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(
                    decode_stage(c.source, q_decode, stride=s.sampling_stride),
                    name="decode",
                )
                tg.create_task(
                    detect_track_stage(
                        q_decode,
                        q_track,
                        detector=c.detector,
                        tracker=c.tracker,
                        executor=det_exec,
                        batch_size=s.yolo_batch_size,
                    ),
                    name="detect_track",
                )
                tg.create_task(
                    embed_assign_stage(
                        q_track,
                        q_assign,
                        classifier=c.classifier,
                        executor=emb_exec,
                        keep_frames=self._annotator is not None,
                    ),
                    name="embed_assign",
                )
                tg.create_task(
                    aggregate_stage(q_assign, q_results, state=state),
                    name="aggregate",
                )
                tg.create_task(
                    sink_stage(q_results, c.sink, annotator=self._annotator),
                    name="sink",
                )
        except* Exception as eg:
            raise eg.exceptions[0] from None
        finally:
            await self.aclose()

    # -- streaming -----------------------------------------------------------

    async def stream(self) -> AsyncIterator[FrameResult]:
        """Yield :class:`FrameResult` items as they are produced.

        Used by the streaming endpoint to avoid buffering the whole video.
        Annotation is not supported in this mode.
        """
        det_exec, emb_exec, _ = self._ensure_executors()
        s = self._settings

        q_decode: asyncio.Queue[Any] = asyncio.Queue(maxsize=_Q_DECODE_TO_DETECT)
        q_track: asyncio.Queue[Any] = asyncio.Queue(maxsize=_Q_DETECT_TO_EMBED)
        q_assign: asyncio.Queue[Any] = asyncio.Queue(maxsize=_Q_EMBED_TO_AGG)
        q_results: asyncio.Queue[Any] = asyncio.Queue(maxsize=_Q_AGG_TO_SINK)

        state = TrackState(window=s.track_window)

        producer = asyncio.create_task(
            self._drive_until_results(
                q_decode, q_track, q_assign, q_results, state, det_exec, emb_exec
            ),
            name="pipeline_driver",
        )

        try:
            while True:
                item = await q_results.get()
                if item is None:
                    break
                # Aggregator emits (FrameResult, frame|None, tracks); in stream
                # mode keep_frames=False so frame is None.
                result, _frame, _tracks = item
                yield result
            await producer  # surface any error
        except BaseException:
            producer.cancel()
            with suppress(BaseException):
                await producer
            raise
        finally:
            await self.aclose()

    async def _drive_until_results(
        self,
        q_decode: asyncio.Queue[Any],
        q_track: asyncio.Queue[Any],
        q_assign: asyncio.Queue[Any],
        q_results: asyncio.Queue[Any],
        state: TrackState,
        det_exec: ThreadPoolExecutor,
        emb_exec: ThreadPoolExecutor,
    ) -> None:
        """TaskGroup wiring for :meth:`stream` (no sink stage; caller drains q_results)."""
        c = self._components
        s = self._settings
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(decode_stage(c.source, q_decode, stride=s.sampling_stride))
                tg.create_task(
                    detect_track_stage(
                        q_decode,
                        q_track,
                        detector=c.detector,
                        tracker=c.tracker,
                        executor=det_exec,
                        batch_size=s.yolo_batch_size,
                    )
                )
                tg.create_task(
                    embed_assign_stage(
                        q_track,
                        q_assign,
                        classifier=c.classifier,
                        executor=emb_exec,
                        keep_frames=False,
                    )
                )
                tg.create_task(aggregate_stage(q_assign, q_results, state=state))
        except* Exception as eg:
            raise eg.exceptions[0] from None


__all__ = ["PipelineComponents", "PipelineRunner"]
