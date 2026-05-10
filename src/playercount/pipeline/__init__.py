"""Async streaming pipeline that wires the ML stages together."""

from __future__ import annotations

from playercount.pipeline.runner import PipelineComponents, PipelineRunner
from playercount.pipeline.stages import (
    aggregate_stage,
    decode_stage,
    detect_track_stage,
    embed_assign_stage,
    sink_stage,
)

__all__ = [
    "PipelineComponents",
    "PipelineRunner",
    "aggregate_stage",
    "decode_stage",
    "detect_track_stage",
    "embed_assign_stage",
    "sink_stage",
]
