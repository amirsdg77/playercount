"""Pipeline orchestrator — owns queues, executors, lifecycle, and cancellation.

This is the file that turns a pile of stage coroutines into a *system*.
It is written to be readable end-to-end so the design conversation in the
interview can lean on it directly. Two surfaces:

* :meth:`PipelineRunner.run` — drains the entire video, returns when the sink
  has consumed the last frame.
* :meth:`PipelineRunner.stream` — yields :class:`FrameResult` objects as they
  are produced. Exists to feed the NDJSON streaming endpoint with no extra
  buffering.

Concurrency notes:

* Stages run as siblings inside an :class:`asyncio.TaskGroup` — any failure
  cancels the rest. We catch and re-raise the *first* underlying exception
  rather than letting the ``ExceptionGroup`` propagate raw, since callers
  (FastAPI, the CLI) are not written against that idiom yet.
* **Three executors** instead of one shared pool. Detector and embedder run
  on separate :class:`ThreadPoolExecutor` instances so a backlog in one
  cannot starve the other; PyAV decode runs on its own single-thread pool
  because there is exactly one video source per pipeline. Audit finding #6.
* Bounded :class:`asyncio.Queue` instances between every pair of stages —
  when a queue fills, ``await out_q.put(...)`` blocks the upstream stage.
  That is the backpressure that keeps RAM bounded if the GPU is the
  bottleneck.
* Optional :class:`AnnotatedVideoSink` plumbed as a side-channel to the sink
  stage. When provided, the embed stage forwards the original frame ndarray
  through the queues; otherwise frames are dropped after embedding to save
  memory pressure on the downstream queues.
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
    """The five collaborators a :class:`PipelineRunner` needs.

    Constructed by the dependency-injection layer (CLI builds them inline,
    the API builds them from the :class:`playercount.models.registry.ModelRegistry`).
    """

    source: VideoSource
    detector: Detector
    tracker: Tracker
    classifier: TeamClassifier
    sink: ResultSink


# ---------------------------------------------------------------------------
# Queue depths — extracted as constants so they are visible at the top of the
# file and easy to grep / tweak. Defaults trade memory for throughput slack.
# ---------------------------------------------------------------------------

# Q1: raw 1080p BGR ≈ 6 MB/frame; 8 → ~50 MB ceiling.
_Q_DECODE_TO_DETECT = 8
# Q2: items here are *batches* of yolo_batch_size frames; tighter cap keeps
# decoding from running too far ahead of inference.
_Q_DETECT_TO_EMBED = 8  # per-frame after the detect stage flushes
# Q3: per-frame post-detect payload (tracks + crops); embedding is the typical
# bottleneck so we keep this small.
_Q_EMBED_TO_AGG = 8
# Q4: results are tiny (~1 KB each); a generous buffer keeps NDJSON streaming
# smooth even under intermittent client backpressure.
_Q_AGG_TO_SINK = 16


def _default_executor_workers(settings: Settings) -> int:
    """Per-pool worker count.

    Each of the detector and embedder pools gets ``max(2, cpu_count // 2)``
    workers. Single CUDA context per process means real GPU parallelism is 1;
    the extra workers exist to overlap host pre/post (resize, BGR→RGB, tensor
    copy) with the previous batch's GPU work. Settings can override.
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
        # If callers pre-build executors, we don't own them and won't shut
        # them down on aclose; the lifecycle stays with the caller.
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
            # PyAV decode is single-threaded per source; one worker is enough.
            self._decode_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="playercount-decode"
            )
        return self._detect_executor, self._embed_executor, self._decode_executor

    @property
    def detect_executor(self) -> ThreadPoolExecutor:
        """Public accessor; makes the helper sources reuse the same pool."""
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
        """Release the source, the sink, the annotator (if any), and our pools."""
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

        Returns when the sink has consumed the last frame and every stage has
        exited cleanly. On any stage error, :meth:`aclose` is called and the
        first underlying exception is re-raised.
        """
        det_exec, emb_exec, _ = self._ensure_executors()
        c = self._components
        s = self._settings

        # Stage queues are heterogeneous payload-wise; ``Any`` here lets the
        # five stage signatures stay aligned without invariance gymnastics.
        # See ``stages.StageQueue`` for the rationale.
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
            # Unwrap to the first underlying error so callers don't have to
            # know about ExceptionGroup.
            raise eg.exceptions[0] from None
        finally:
            await self.aclose()

    # -- streaming -----------------------------------------------------------

    async def stream(self) -> AsyncIterator[FrameResult]:
        """Yield :class:`FrameResult` items as they are produced.

        Used by the ``POST /analyze/stream`` endpoint to feed an NDJSON
        StreamingResponse without buffering the whole video in memory.

        Annotation is **not supported** in stream mode — there is no place to
        write an MP4 alongside an HTTP response.
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
                # Aggregator now emits (FrameResult, frame|None, tracks).
                # In stream mode we keep_frames=False so frame is None.
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
        """Same TaskGroup wiring as :meth:`run`, factored for the streaming case.

        Note we deliberately do **not** spawn the sink stage here — the caller
        consumes ``q_results`` directly. We *do* still need the aggregator to
        drain into ``q_results``.
        """
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
