"""Stabilization + counting layer that turns raw tracks into per-frame counts."""

from __future__ import annotations

from playercount.aggregation.frame_counter import build_frame_result
from playercount.aggregation.track_state import TrackState

__all__ = ["TrackState", "build_frame_result"]
