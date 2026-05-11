"""Async pipeline stages.

Each stage is a coroutine that consumes an input :class:`asyncio.Queue` and
produces to an output queue, with CPU/GPU work dispatched to a
:class:`ThreadPoolExecutor`. Stages signal end-of-stream with a single
``None`` sentinel.
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

# Stage queues carry heterogeneous payloads; ``Any`` keeps the five stage
# signatures aligned without invariance gymnastics. Real shapes are documented
# inline at each ``in_q``/``out_q`` parameter.
StageQueue = "asyncio.Queue[Any]"

# End-of-stream sentinel pushed on every output queue.
EOF: None = None


async def _emit_eof(out_q: asyncio.Queue[Any]) -> None:
    """Push the EOF sentinel.

    On normal end-of-stream uses an awaiting ``put`` so downstream sees it.
    When the surrounding task is being cancelled, falls back to ``put_nowait``
    so a full downstream queue cannot deadlock TaskGroup teardown.
    """
    task = asyncio.current_task()
    if task is not None and task.cancelling() > 0:
        with suppress(asyncio.QueueFull):
            out_q.put_nowait(EOF)
        return
    try:
        await out_q.put(EOF)
    except asyncio.CancelledError:
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
    """Pull frames from ``source`` and put ``(frame_index, ts_s, bgr)`` on ``out_q``.

    Emits exactly one EOF sentinel when the source is exhausted.
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
# Stage 2: detect + track
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
    """Batch frames, run detection on the executor, update the tracker, emit tracks.

    Detection runs off-thread; tracker.update runs in-line so updates stay in
    frame order (the tracker is stateful and not order-tolerant).
    """
    loop = asyncio.get_running_loop()
    batch: list[tuple[int, float, np.ndarray]] = []
    eof_sent = False

    async def flush() -> None:
        if not batch:
            return
        frames = [b[2] for b in batch]
        det_lists: list[list[Detection]] = await loop.run_in_executor(
            executor, detector.infer, frames
        )
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
    """Crop each tracked detection, batch-classify teams, fill ``track.team_id``.

    All crops in a frame are sent to the classifier in one batched call.
    Referees and balls are skipped by the classifier; degenerate crops
    (zero area) are dropped before the call and their tracks keep ``team_id=None``.

    If ``keep_frames`` is True, the original BGR frame is forwarded on the
    output queue (used by the annotator side-channel); otherwise it is
    replaced with ``None`` to free memory.
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
                for t in tracks:
                    t.team_id = assignments_by_id.get(t.track_id)
            payload = (idx, ts, frame if keep_frames else None, tracks)
            await out_q.put(payload)
    finally:
        if not eof_sent:
            await _emit_eof(out_q)


def _crop_bgr(frame: np.ndarray, track: Track) -> np.ndarray | None:
    """Return the BGR sub-array for ``track.detection.bbox``, or ``None`` if degenerate."""
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
    """Reduce per-frame tracks to :class:`FrameResult` via the sliding-window stabiliser.

    Output is ``(FrameResult, frame_bgr | None, tracks)`` so the sink can
    route either to a plain :class:`ResultSink` or to an annotator that
    needs the original frame.
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
    annotator: Any | None = None,  # AnnotatedVideoSink | None
) -> None:
    """Drain results into ``sink`` (and ``annotator`` if provided) until EOF."""
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
