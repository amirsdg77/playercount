"""Async pipeline stages, each a coroutine bound to one role.

Stages communicate via :class:`asyncio.Queue` instances created by
:class:`playercount.pipeline.runner.PipelineRunner`. Each stage:

* ``await`` s its input queue,
* dispatches CPU/GPU-heavy work to a :class:`ThreadPoolExecutor` via
  ``loop.run_in_executor`` (CUDA forwards and PyAV decode release the GIL),
* ``await`` s its output queue (which gives us backpressure when downstream
  stalls — bounded queues are the whole point),
* propagates a sentinel (``None``) on its output queue when its input is
  exhausted, so the stage downstream of it knows to drain and exit cleanly.
* On cancellation, the ``finally`` block uses :func:`_emit_eof`'s
  cancellation-aware non-blocking path so the TaskGroup teardown never
  deadlocks on a full downstream queue.

**Sentinel-once invariant.** Every stage sets a local ``_eof_sent`` flag
*after* the in-loop ``return`` path that consumes the upstream EOF and emits
its own. The ``finally`` block then only emits if ``_eof_sent`` is False —
preventing a double EOF if the body raised after a successful in-loop emit.

Why this shape: a single CUDA context per process bounds true GPU parallelism
to one. Adding more processes multiplies VRAM without raising throughput; the
correct scaling axis is *more replicas*. Within one replica, pipelining
overlaps decode (I/O), inference (GPU), and post-processing (CPU) so the
critical-path latency drops to roughly ``max(stage_i)`` instead of their sum.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from playercount.aggregation import build_frame_result

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from playercount.aggregation import TrackState
    from playercount.detection import Detector
    from playercount.io import ResultSink, VideoSource
    from playercount.schemas import Detection, Track
    from playercount.team_id import TeamClassifier
    from playercount.tracking import Tracker

# Stages communicate via heterogeneous payloads (raw frames, batches of
# tracks, FrameResult, …). asyncio.Queue is invariant in its element type, so
# a strictly-typed queue per stage forces variance gymnastics at the runner.
# Each stage's *real* expected payload shape is documented inline; the alias
# below is the practical lingua franca between them.
StageQueue = "asyncio.Queue[Any]"


# Sentinel placed on every output queue when the stage's input is exhausted.
# Using a singleton ``object()`` over ``None`` would be safer but ``None`` is
# the convention in stdlib examples and keeps queue typing simpler.
EOF: None = None


async def _emit_eof(out_q: asyncio.Queue[Any]) -> None:
    """Push EOF, but never block when the surrounding task has been cancelled.

    Stage ``finally`` blocks call this on the way out. There are two cases:

    * **Normal end-of-stream.** The source is exhausted, downstream is still
      consuming. We need to ``await put`` so the EOF actually lands and the
      downstream stage exits cleanly.
    * **Cancellation mid-flight.** The TaskGroup told us to stop. Downstream
      is also cancelling and the queue may be full forever; ``await put``
      would deadlock the whole TaskGroup teardown. Use ``put_nowait`` and
      drop the EOF — the cancellation propagates naturally without it.

    We tell the cases apart by checking the current task's cancellation state.
    """
    task = asyncio.current_task()
    if task is not None and task.cancelling() > 0:
        with suppress(asyncio.QueueFull):
            out_q.put_nowait(EOF)
        return
    try:
        await out_q.put(EOF)
    except asyncio.CancelledError:
        # Cancelled while waiting on a full queue → drop EOF (see above).
        with suppress(asyncio.QueueFull):
            out_q.put_nowait(EOF)
        raise


# ---------------------------------------------------------------------------
# Stage 1: decode
# ---------------------------------------------------------------------------


async def decode_stage(
    source: VideoSource,
    out_q: asyncio.Queue[Any],  # tuple[int, float, np.ndarray] | None
    *,
    stride: int = 1,
) -> None:
    """Pull frames from ``source`` and put them on ``out_q``.

    Yields ``(frame_index, timestamp_s, bgr_ndarray)`` tuples. When the source
    is exhausted, places exactly one ``None`` sentinel on ``out_q`` and returns.
    """
    eof_sent = False
    try:
        async for idx, ts, frame in source.frames(stride=stride):
            await out_q.put((idx, ts, frame))
        await _emit_eof(out_q)
        eof_sent = True
    finally:
        if not eof_sent:
            await _emit_eof(out_q)


# ---------------------------------------------------------------------------
# Stage 2: detect + track  (single stage — tracking must be inline with detect
# to keep ID continuity per frame ordering)
# ---------------------------------------------------------------------------


async def detect_track_stage(
    in_q: asyncio.Queue[Any],  # tuple[int, float, np.ndarray] | None
    out_q: asyncio.Queue[Any],  # tuple[int, float, np.ndarray, list[Track]] | None
    *,
    detector: Detector,
    tracker: Tracker,
    executor: ThreadPoolExecutor,
    batch_size: int = 8,
) -> None:
    """Batch frames, run YOLO in the executor, update the tracker, emit tracks.

    Why combine detect + track in one stage: the tracker is stateful and must
    consume detections in *frame order*, one frame at a time. If detection ran
    in a separate stage from tracking, we'd either need a per-batch ordering
    barrier or we'd risk out-of-order tracker updates.
    """
    loop = asyncio.get_running_loop()
    batch: list[tuple[int, float, np.ndarray]] = []
    eof_sent = False

    async def flush() -> None:
        if not batch:
            return
        frames = [b[2] for b in batch]
        # YOLO forward in the executor; CUDA releases the GIL.
        det_lists: list[list[Detection]] = await loop.run_in_executor(
            executor, detector.infer, frames
        )
        # Tracker update is cheap CPU and *must* be in frame order; do it here
        # on the event-loop thread (microseconds) to keep ID continuity.
        for (idx, ts, frame), dets in zip(batch, det_lists, strict=True):
            tracks = tracker.update(idx, dets)
            await out_q.put((idx, ts, frame, tracks))
        batch.clear()

    try:
        while True:
            item = await in_q.get()
            if item is EOF:
                await flush()
                await _emit_eof(out_q)
                eof_sent = True
                return
            batch.append(item)
            if len(batch) >= batch_size:
                await flush()
    finally:
        if not eof_sent:
            await _emit_eof(out_q)


# ---------------------------------------------------------------------------
# Stage 3: embed + team-assign
# ---------------------------------------------------------------------------


async def embed_assign_stage(
    in_q: asyncio.Queue[Any],  # tuple[int, float, np.ndarray, list[Track]] | None
    out_q: asyncio.Queue[Any],  # tuple[int, float, np.ndarray | None, list[Track]] | None
    *,
    classifier: TeamClassifier,
    executor: ThreadPoolExecutor,
    keep_frames: bool = False,
) -> None:
    """Cut crops out of each frame, run the team classifier, fill ``track.team_id``.

    Crops for *all* tracks in a frame are passed to the classifier in one
    batched call so the SigLIP forward sees a single tensor. The classifier
    decides per track whether to actually assign (skips referees and balls).

    By default the frame ndarray is dropped from the tuple before the next
    queue — the aggregator does not need it, and dropping the ~6 MB payload
    here materially lowers memory pressure on Q3. When ``keep_frames=True``
    (used by the annotator side-channel), the frame is forwarded; the
    aggregator stage will pass it through to the sink.

    Tracks with degenerate crops (zero area after clamping) are still
    forwarded but receive ``team_id=None`` — the sliding-window stabiliser
    treats them as no-vote frames.
    """
    loop = asyncio.get_running_loop()
    eof_sent = False

    try:
        while True:
            item = await in_q.get()
            if item is EOF:
                await out_q.put(EOF)
                eof_sent = True
                return

            idx, ts, frame, tracks = item
            if tracks:
                raw_crops = [_crop_bgr(frame, t) for t in tracks]
                # Filter to (track, crop) pairs with valid crops; classifier
                # only sees real pixels.
                valid_pairs = [
                    (t, c) for t, c in zip(tracks, raw_crops, strict=True) if c is not None
                ]
                if valid_pairs:
                    valid_tracks = [p[0] for p in valid_pairs]
                    valid_crops = [p[1] for p in valid_pairs]
                    assignments = await loop.run_in_executor(
                        executor, classifier.assign, valid_tracks, valid_crops
                    )
                    assignments_by_id = {a.track_id: a.team_id for a in assignments}
                else:
                    assignments_by_id = {}
                # Fill team_id back into tracks; refs/ball/invalid-crop stay None.
                for t in tracks:
                    t.team_id = assignments_by_id.get(t.track_id)
            payload = (idx, ts, frame if keep_frames else None, tracks)
            await out_q.put(payload)
    finally:
        if not eof_sent:
            await _emit_eof(out_q)


def _crop_bgr(frame: np.ndarray, track: Track) -> np.ndarray | None:
    """Cut ``track.detection.bbox`` out of ``frame`` with int rounding + clamping.

    Returns ``None`` if the bbox is degenerate (zero area after clamping to
    the frame). Callers (the embed stage) skip degenerate crops rather than
    feeding empty arrays into SigLIP.
    """
    h, w = frame.shape[:2]
    bb = track.detection.bbox
    x1 = max(0, int(round(bb.x1)))
    y1 = max(0, int(round(bb.y1)))
    x2 = min(w, int(round(bb.x2)))
    y2 = min(h, int(round(bb.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Stage 4: aggregate
# ---------------------------------------------------------------------------


async def aggregate_stage(
    in_q: asyncio.Queue[Any],  # tuple[int, float, np.ndarray | None, list[Track]] | None
    out_q: asyncio.Queue[Any],  # tuple[FrameResult, np.ndarray | None, list[Track]] | None
    *,
    state: TrackState,
) -> None:
    """Reduce tracks → :class:`FrameResult` using the sliding-window stabilizer.

    Pure-Python, ``O(num_tracks_in_frame)``, microseconds — no executor needed.

    The output payload is a triple ``(FrameResult, frame_bgr | None, tracks)``
    so the sink stage can route results to a plain :class:`ResultSink` (using
    only the FrameResult) and optionally to an :class:`AnnotatedVideoSink`
    (using all three). Plumbing the frame through here, when present, is
    cheaper than an out-of-band ring buffer keyed by frame_index.
    """
    eof_sent = False
    try:
        while True:
            item = await in_q.get()
            if item is EOF:
                await out_q.put(EOF)
                eof_sent = True
                return
            idx, ts, frame, tracks = item
            result = build_frame_result(idx, ts, tracks, state)
            await out_q.put((result, frame, tracks))
    finally:
        if not eof_sent:
            await _emit_eof(out_q)


# ---------------------------------------------------------------------------
# Stage 5: sink
# ---------------------------------------------------------------------------


async def sink_stage(
    in_q: asyncio.Queue[Any],  # tuple[FrameResult, np.ndarray | None, list[Track]] | None
    sink: ResultSink,
    annotator: Any | None = None,  # AnnotatedVideoSink | None — avoid import cycle
) -> None:
    """Drain results into the sink (and optionally the annotator) until EOF."""
    while True:
        item = await in_q.get()
        if item is EOF:
            return
        result, frame, tracks = item
        await sink.write(result)
        if annotator is not None and frame is not None:
            await annotator.write_with_frame(result, frame, tracks)


__all__ = [
    "EOF",
    "aggregate_stage",
    "decode_stage",
    "detect_track_stage",
    "embed_assign_stage",
    "sink_stage",
]
