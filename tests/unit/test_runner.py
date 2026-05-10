"""End-to-end test of the async stage wiring with fake collaborators.

We avoid YOLO/SigLIP entirely — fakes produce synthetic detections so we can
prove that:

* The full :class:`PipelineRunner` graph drains a video to a sink.
* Sentinel propagation works: each stage sees its EOF and emits one downstream.
* Cancellation does not hang (we cancel mid-stream and assert the awaitable
  resolves quickly).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from playercount.detection import Detector
from playercount.io import ResultSink, VideoSource
from playercount.io.sinks import NdjsonSink
from playercount.pipeline.runner import PipelineComponents, PipelineRunner
from playercount.schemas import (
    BBox,
    Detection,
    FrameResult,
    TeamAssignment,
    Track,
)
from playercount.team_id import TeamClassifier
from playercount.tracking import Tracker

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSource:
    """A tiny in-memory source yielding ``num_frames`` blank BGR frames."""

    def __init__(self, num_frames: int = 8, fps: float = 10.0) -> None:
        self.fps = fps
        self.width = 16
        self.height = 16
        self._num_frames = num_frames
        self._closed = False

    async def frames(
        self, stride: int = 1
    ) -> AsyncIterator[tuple[int, float, np.ndarray]]:
        for i in range(0, self._num_frames, stride):
            await asyncio.sleep(0)  # cooperative yield to the loop
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            yield i, i / self.fps, frame

    async def close(self) -> None:
        self._closed = True


class _FakeDetector:
    """Returns one player detection per frame for the same arbitrary id."""

    def infer(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        out: list[list[Detection]] = []
        for _ in frames_bgr:
            out.append(
                [
                    Detection(
                        bbox=BBox(x1=0, y1=0, x2=10, y2=10),
                        score=0.9,
                        class_id=0,
                        class_name="player",
                    )
                ]
            )
        return out

    def warm(self) -> None:
        return None


class _FakeTracker:
    """Trivial tracker: assigns track_id=1 to every detection."""

    def update(self, frame_idx: int, dets: list[Detection]) -> list[Track]:
        return [Track(track_id=1, detection=d) for d in dets]

    def reset(self) -> None:
        return None


class _FakeClassifier:
    """Always assigns team 0 to track id 1."""

    def needs_calibration(self) -> bool:
        return False

    def calibrate(self, crops: list[np.ndarray]) -> None:
        return None

    def assign(
        self, tracks: list[Track], crops: list[np.ndarray]
    ) -> list[TeamAssignment]:
        return [
            TeamAssignment(track_id=t.track_id, team_id=0, confidence=0.99)
            for t in tracks
        ]


class _RecordingSink:
    def __init__(self) -> None:
        self.frames: list[FrameResult] = []
        self.closed = False

    async def write(self, frame: FrameResult) -> None:
        self.frames.append(frame)

    async def aclose(self) -> None:
        self.closed = True


class _BlockingSink:
    """Sink that blocks forever — used to force cancellation paths."""

    def __init__(self) -> None:
        self.closed = False
        self._gate = asyncio.Event()

    async def write(self, frame: FrameResult) -> None:
        await self._gate.wait()  # never resolves

    async def aclose(self) -> None:
        self.closed = True
        self._gate.set()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_components(
    *,
    source: VideoSource,
    sink: ResultSink,
    detector: Detector | None = None,
    tracker: Tracker | None = None,
    classifier: TeamClassifier | None = None,
) -> PipelineComponents:
    return PipelineComponents(
        source=source,
        detector=detector or _FakeDetector(),
        tracker=tracker or _FakeTracker(),
        classifier=classifier or _FakeClassifier(),
        sink=sink,
    )


@pytest.mark.asyncio
async def test_runner_drains_to_sink(settings):
    src = _FakeSource(num_frames=12)
    sink = _RecordingSink()
    runner = PipelineRunner(settings, _make_components(source=src, sink=sink))
    await runner.run()
    assert len(sink.frames) == 12
    # team_a counts should converge to 1 once the majority window has filled.
    assert sink.frames[-1].team_a_count == 1
    assert sink.closed is True


@pytest.mark.asyncio
async def test_runner_writes_to_real_ndjson_sink(tmp_path, settings):
    src = _FakeSource(num_frames=4)
    out = tmp_path / "out.ndjson"
    sink = NdjsonSink(out)
    runner = PipelineRunner(settings, _make_components(source=src, sink=sink))
    await runner.run()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4


@pytest.mark.asyncio
async def test_runner_streams(settings):
    src = _FakeSource(num_frames=6)
    sink = _RecordingSink()  # ignored in stream mode
    runner = PipelineRunner(settings, _make_components(source=src, sink=sink))
    out: list[FrameResult] = []
    async for frame in runner.stream():
        out.append(frame)
    assert len(out) == 6


@pytest.mark.asyncio
async def test_runner_cancellation_resolves(settings):
    """Cancelling a runner mid-stream must not hang."""
    src = _FakeSource(num_frames=1000)
    sink = _BlockingSink()
    runner = PipelineRunner(settings, _make_components(source=src, sink=sink))

    task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.05)  # let stages spin up
    task.cancel()

    # If aclose / cancellation is broken, this would hang. Asserting a 2 s
    # timeout buys safety margin in CI.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)
    assert sink.closed is True


@pytest.mark.asyncio
async def test_runner_aclose_idempotent(settings):
    src = _FakeSource(num_frames=2)
    sink = _RecordingSink()
    runner = PipelineRunner(settings, _make_components(source=src, sink=sink))
    await runner.run()
    await runner.aclose()
    await runner.aclose()  # no-op the second time
