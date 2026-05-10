"""Tiny end-to-end smoke check used in CI and local sanity testing.

Builds a :class:`PipelineRunner` over an in-memory fake source so neither the
sample video nor the model weights are required. Useful for catching wiring
regressions that the unit suite might miss.

Usage::

    python scripts/smoke_run.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from playercount.config import Settings
from playercount.io.sinks import NdjsonSink
from playercount.pipeline.runner import PipelineComponents, PipelineRunner
from playercount.schemas import (
    BBox,
    Detection,
    FrameResult,
    TeamAssignment,
    Track,
)

# ---------------------------------------------------------------------------
# Inline fakes — same shape as tests/unit/test_runner.py but standalone so the
# script does not depend on the test package being importable.
# ---------------------------------------------------------------------------


class _FakeSource:
    fps: float = 10.0
    width: int = 16
    height: int = 16

    def __init__(self, num_frames: int) -> None:
        self._num = num_frames

    async def frames(self, stride: int = 1) -> AsyncIterator[tuple[int, float, np.ndarray]]:
        for i in range(0, self._num, stride):
            await asyncio.sleep(0)
            yield i, i / self.fps, np.zeros((self.height, self.width, 3), dtype=np.uint8)

    async def close(self) -> None:
        return None


class _FakeDetector:
    def infer(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        return [
            [
                Detection(
                    bbox=BBox(x1=0, y1=0, x2=10, y2=10),
                    score=0.9,
                    class_id=0,
                    class_name="player",
                )
            ]
            for _ in frames_bgr
        ]

    def warm(self) -> None:
        return None


class _FakeTracker:
    def update(self, frame_idx: int, dets: list[Detection]) -> list[Track]:
        return [Track(track_id=1, detection=d) for d in dets]

    def reset(self) -> None:
        return None


class _FakeClassifier:
    def needs_calibration(self) -> bool:
        return False

    def calibrate(self, crops: list[np.ndarray]) -> None:
        return None

    def assign(
        self, tracks: list[Track], crops: list[np.ndarray]
    ) -> list[TeamAssignment]:
        return [
            TeamAssignment(track_id=t.track_id, team_id=0, confidence=0.95) for t in tracks
        ]


class _PrintSink:
    def __init__(self) -> None:
        self.n = 0

    async def write(self, frame: FrameResult) -> None:
        self.n += 1
        if frame.frame_index < 3:
            print(f"  frame {frame.frame_index}: a={frame.team_a_count} b={frame.team_b_count}")

    async def aclose(self) -> None:
        print(f"  total frames sunk: {self.n}")


async def _amain() -> None:
    settings = Settings()
    components = PipelineComponents(
        source=_FakeSource(num_frames=30),
        detector=_FakeDetector(),
        tracker=_FakeTracker(),
        classifier=_FakeClassifier(),
        sink=_PrintSink(),
    )
    print("[smoke] running pipeline with 30 fake frames...")
    async with PipelineRunner(settings, components) as runner:
        await runner.run()
    print("[smoke] OK")

    # Also exercise the NDJSON sink path so a full happy round-trip is covered.
    sink = NdjsonSink()
    components = PipelineComponents(
        source=_FakeSource(num_frames=5),
        detector=_FakeDetector(),
        tracker=_FakeTracker(),
        classifier=_FakeClassifier(),
        sink=sink,
    )
    async with PipelineRunner(settings, components) as runner:
        await runner.run()
    text = sink.getvalue()
    assert text.count("\n") == 5, text
    print("[smoke] ndjson sink OK")


if __name__ == "__main__":
    asyncio.run(_amain())
