"""Video sources and result sinks."""

from __future__ import annotations

from playercount.io.sinks import (
    AnnotatedVideoSink,
    JsonSink,
    NdjsonSink,
    ResultSink,
)
from playercount.io.video_source import PyAvVideoSource, VideoSource

__all__ = [
    "AnnotatedVideoSink",
    "JsonSink",
    "NdjsonSink",
    "PyAvVideoSource",
    "ResultSink",
    "VideoSource",
]
