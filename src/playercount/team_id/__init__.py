"""Unsupervised team identification: embedding → cluster → assign."""

from __future__ import annotations

from playercount.team_id.clusterer import TeamClusterer
from playercount.team_id.embedder import SigLipEmbedder
from playercount.team_id.team_classifier import (
    EmbeddingTeamClassifier,
    HsvTeamClassifier,
    TeamClassifier,
)

__all__ = [
    "EmbeddingTeamClassifier",
    "HsvTeamClassifier",
    "SigLipEmbedder",
    "TeamClassifier",
    "TeamClusterer",
]
