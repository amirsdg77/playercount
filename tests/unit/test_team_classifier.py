"""Tests for the orchestration logic in EmbeddingTeamClassifier.

The deep ML bodies are tested via integration; here we verify the
orchestration contract — argument validation, length checks, and the
fail-loud-when-uncalibrated invariant added in audit fix A11.
"""

from __future__ import annotations

import numpy as np
import pytest

from playercount.schemas import BBox, Detection, Track
from playercount.team_id import (
    EmbeddingTeamClassifier,
    SigLipEmbedder,
    TeamClassifier,
    TeamClusterer,
)


def _player_track(track_id: int) -> Track:
    return Track(
        track_id=track_id,
        detection=Detection(
            bbox=BBox(x1=0, y1=0, x2=1, y2=1),
            score=0.9,
            class_id=0,
            class_name="player",
        ),
    )


def test_protocol_runtime_check():
    embedder = SigLipEmbedder()
    clusterer = TeamClusterer()
    cls = EmbeddingTeamClassifier(embedder=embedder, clusterer=clusterer)
    assert isinstance(cls, TeamClassifier)


def test_raises_before_calibration():
    """Audit fix A11: silently returning [] before calibration was a
    correctness foot-gun (counts collapsed without warning). The pipeline
    must call calibrate() before run() now."""
    embedder = SigLipEmbedder()
    clusterer = TeamClusterer()
    cls = EmbeddingTeamClassifier(embedder=embedder, clusterer=clusterer)
    assert cls.needs_calibration() is True
    with pytest.raises(RuntimeError, match="not calibrated"):
        cls.assign([_player_track(1)], [np.zeros((10, 10, 3), dtype=np.uint8)])


def test_assign_validates_lengths():
    embedder = SigLipEmbedder()
    clusterer = TeamClusterer()
    cls = EmbeddingTeamClassifier(embedder=embedder, clusterer=clusterer)
    with pytest.raises(ValueError, match="!="):
        cls.assign([_player_track(1), _player_track(2)], [np.zeros((4, 4, 3), dtype=np.uint8)])


def test_calibrate_rejects_empty_input():
    embedder = SigLipEmbedder()
    clusterer = TeamClusterer()
    cls = EmbeddingTeamClassifier(embedder=embedder, clusterer=clusterer)
    with pytest.raises(ValueError):
        cls.calibrate([])
