"""Tracker protocol + ByteTrack implementation backed by ``supervision``.

Why tracking matters for the count: without stable ids, a player who is briefly
mis-detected for one frame would drop in and out of the count, producing
flicker. Tracking turns the per-frame question "how many detections are
visible?" into the more useful "how many distinct active tracks are visible?",
which is robust to single-frame misses.

Tracker state is per-stream and is **not** safe to share between concurrent
analyses — :class:`playercount.pipeline.runner.PipelineRunner` constructs one
tracker per request.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from playercount.schemas import CLASS_NAMES, BBox, Detection, Track


@runtime_checkable
class Tracker(Protocol):
    """Stateful per-stream tracker. Single-threaded use only."""

    def update(self, frame_idx: int, dets: list[Detection]) -> list[Track]:
        """Consume one frame's detections; return tracks (with ids) for the same frame."""
        ...

    def reset(self) -> None:
        """Clear internal state (e.g. on stream end / video boundary)."""
        ...


class ByteTrackTracker:
    """ByteTrack via ``supervision.ByteTrack``.

    We pick ByteTrack over BoT-SORT for the PoC: it's faster and ID switches on
    short occlusions are *cheap* for our use case because the team-classifier
    re-snaps the team label on the new track. BoT-SORT would only matter if we
    cared about long-term identity (jersey number, possession, scouting clips).
    """

    def __init__(
        self,
        *,
        frame_rate: float,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ) -> None:
        self._frame_rate = frame_rate
        self._track_activation_threshold = track_activation_threshold
        self._lost_track_buffer = lost_track_buffer
        self._minimum_matching_threshold = minimum_matching_threshold
        self._impl: Any = None
        self.reset()

    def update(self, frame_idx: int, dets: list[Detection]) -> list[Track]:
        """Translate detections → ``sv.Detections`` → ByteTrack update → :class:`Track` list."""
        # Lazy import — supervision pulls in opencv etc. at import time.
        import supervision as sv  # type: ignore[import-not-found]

        if not dets:
            # ByteTrack still needs to be ticked so its lost-buffer counters
            # advance; pass an empty Detections object.
            self._impl.update_with_detections(sv.Detections.empty())
            return []

        xyxy = np.array([d.bbox.as_xyxy() for d in dets], dtype=np.float32)
        confidence = np.array([d.score for d in dets], dtype=np.float32)
        class_id = np.array([d.class_id for d in dets], dtype=int)

        sv_dets = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self._impl.update_with_detections(sv_dets)

        out: list[Track] = []
        if len(tracked) == 0:
            return out

        for i in range(len(tracked)):
            tid = tracked.tracker_id[i] if tracked.tracker_id is not None else None
            if tid is None:
                continue
            x1, y1, x2, y2 = tracked.xyxy[i].tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            cls = int(tracked.class_id[i]) if tracked.class_id is not None else 0
            score = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            cls_name = CLASS_NAMES.get(cls)
            if cls_name is None:
                continue
            out.append(
                Track(
                    track_id=int(tid),
                    detection=Detection(
                        bbox=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                        score=max(0.0, min(1.0, score)),
                        class_id=cls,  # type: ignore[arg-type]
                        class_name=cls_name,
                    ),
                )
            )
        return out

    def reset(self) -> None:
        """Reinitialize the underlying tracker."""
        import supervision as sv  # type: ignore[import-not-found]

        self._impl = sv.ByteTrack(
            track_activation_threshold=self._track_activation_threshold,
            lost_track_buffer=self._lost_track_buffer,
            minimum_matching_threshold=self._minimum_matching_threshold,
            frame_rate=int(self._frame_rate),
        )


__all__ = ["ByteTrackTracker", "Tracker"]
