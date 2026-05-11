"""Team-classification strategies behind the :class:`TeamClassifier` protocol.

Two implementations:

* :class:`EmbeddingTeamClassifier` — SigLIP embeddings → UMAP → KMeans(k=2).
  Robust to lighting, no labels needed. Goalkeepers are excluded from the
  fit and snapped to the nearer team centroid at predict time.
* :class:`HsvTeamClassifier` — CPU-only fallback. Torso-region HSV histogram
  + KMeans(k=2). Brittle when team kits share luminance.

Referees and the ball never get a team label.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import cv2  # type: ignore[import-not-found]
import numpy as np
from sklearn.cluster import KMeans

from playercount.constants import CLS_GOALKEEPER, CLS_PLAYER, TEAM_A, TEAM_B
from playercount.schemas import TeamAssignment, Track

if TYPE_CHECKING:
    from playercount.team_id.clusterer import TeamClusterer
    from playercount.team_id.embedder import SigLipEmbedder


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TeamClassifier(Protocol):
    """Maps tracks to team assignments. Stateful — may require calibration."""

    def assign(self, tracks: list[Track], crops: list[np.ndarray]) -> list[TeamAssignment]:
        """Return a :class:`TeamAssignment` per track that received a label.

        ``crops[i]`` is the BGR pixel crop of ``tracks[i].detection.bbox``.
        Referees and balls are skipped.
        """
        ...

    def needs_calibration(self) -> bool:
        """True if :meth:`calibrate` must be called before :meth:`assign` works."""
        ...

    def calibrate(self, crops: list[np.ndarray]) -> None:
        """Fit internal models on a representative sample of player crops."""
        ...


# ---------------------------------------------------------------------------
# Primary: SigLIP + UMAP + KMeans
# ---------------------------------------------------------------------------


class EmbeddingTeamClassifier:
    """Unsupervised classifier: SigLIP image embeddings + UMAP + KMeans(k=2).

    Calibration is a one-shot fit on a buffer of player-only crops;
    goalkeepers and referees are excluded so the two KMeans centroids
    represent the two outfield kits. At assign time, players are labelled by
    KMeans on their UMAP-reduced embeddings, goalkeepers are snapped to the
    nearer team centroid, and referees/balls receive no assignment.
    """

    def __init__(self, embedder: SigLipEmbedder, clusterer: TeamClusterer) -> None:
        self._embedder = embedder
        self._clusterer = clusterer
        # threading.Lock — assign() runs on a ThreadPoolExecutor worker, so an
        # asyncio.Lock would raise at acquire time. The lock guards calibrate()
        # against an in-flight assign() that would otherwise see a half-fitted
        # clusterer.
        self._fit_lock = threading.Lock()

    def needs_calibration(self) -> bool:
        return not self._clusterer.is_fit

    def calibrate(self, crops: list[np.ndarray]) -> None:
        """Embed ``crops`` (player-only) and fit the clusterer."""
        if not crops:
            raise ValueError("calibrate requires a non-empty crop list")
        embeddings = self._embedder.embed(crops)
        with self._fit_lock:
            self._clusterer.fit(embeddings)

    def assign(self, tracks: list[Track], crops: list[np.ndarray]) -> list[TeamAssignment]:
        if len(tracks) != len(crops):
            raise ValueError(f"len(tracks)={len(tracks)} != len(crops)={len(crops)}")
        if self.needs_calibration():
            raise RuntimeError(
                "EmbeddingTeamClassifier is not calibrated. "
                "Call calibrate(crops) once before pipeline.run()."
            )
        with self._fit_lock:
            if self.needs_calibration():
                raise RuntimeError("clusterer became unfit between checks")
        return self._assign_impl(tracks, crops)

    def _assign_impl(
        self, tracks: list[Track], crops: list[np.ndarray]
    ) -> list[TeamAssignment]:
        idx_player: list[int] = []
        idx_gk: list[int] = []
        for i, t in enumerate(tracks):
            cls = t.detection.class_id
            if cls == CLS_PLAYER:
                idx_player.append(i)
            elif cls == CLS_GOALKEEPER:
                idx_gk.append(i)

        if not idx_player and not idx_gk:
            return []

        ordered_idx = idx_player + idx_gk
        ordered_crops = [crops[i] for i in ordered_idx]
        embeddings = self._embedder.embed(ordered_crops)
        if embeddings.shape[0] != len(ordered_idx):
            raise RuntimeError(
                f"embedder returned {embeddings.shape[0]} rows for "
                f"{len(ordered_idx)} crops"
            )

        n_players = len(idx_player)
        emb_p = (
            embeddings[:n_players]
            if n_players
            else np.zeros((0, embeddings.shape[1]), dtype=embeddings.dtype)
        )
        emb_gk = (
            embeddings[n_players:]
            if idx_gk
            else np.zeros((0, embeddings.shape[1]), dtype=embeddings.dtype)
        )

        out: list[TeamAssignment] = []

        if n_players:
            labels_p = self._clusterer.predict(emb_p)
            reduced_p = self._clusterer.transform(emb_p)
            centroids = self._clusterer.centroids()
            dists_p = _pairwise_distances(reduced_p, centroids)
            conf_p = _softmax_confidence(dists_p, labels_p)
            for local, i in enumerate(idx_player):
                team_id = int(labels_p[local])
                if team_id not in (TEAM_A, TEAM_B):
                    continue
                out.append(
                    TeamAssignment(
                        track_id=tracks[i].track_id,
                        team_id=team_id,  # type: ignore[arg-type]
                        confidence=float(conf_p[local]),
                    )
                )

        if idx_gk:
            reduced_gk = self._clusterer.transform(emb_gk)
            centroids = self._clusterer.centroids()
            dists_gk = _pairwise_distances(reduced_gk, centroids)
            labels_gk = np.argmin(dists_gk, axis=1)
            conf_gk = _softmax_confidence(dists_gk, labels_gk)
            for local, i in enumerate(idx_gk):
                team_id = int(labels_gk[local])
                if team_id not in (TEAM_A, TEAM_B):
                    continue
                out.append(
                    TeamAssignment(
                        track_id=tracks[i].track_id,
                        team_id=team_id,  # type: ignore[arg-type]
                        confidence=float(conf_gk[local]),
                    )
                )

        return out


# ---------------------------------------------------------------------------
# Fallback: torso-HSV histogram + KMeans
# ---------------------------------------------------------------------------


class HsvTeamClassifier:
    """CPU-only fallback. Torso-ROI HSV histogram + per-stream KMeans(k=2).

    Brittle when both kits have similar hue/luminance.
    """

    def __init__(
        self,
        *,
        torso_box: tuple[float, float, float, float] = (0.2, 0.2, 0.8, 0.6),
        n_bins_h: int = 16,
        n_bins_s: int = 16,
        random_state: int = 42,
    ) -> None:
        self._torso_box = torso_box
        self._n_bins_h = n_bins_h
        self._n_bins_s = n_bins_s
        self._random_state = random_state
        self._kmeans: object | None = None

    def needs_calibration(self) -> bool:
        return self._kmeans is None

    def calibrate(self, crops: list[np.ndarray]) -> None:
        if not crops:
            raise ValueError("calibrate requires a non-empty crop list")
        feats = np.stack([self._features(c) for c in crops], axis=0)
        self._kmeans = KMeans(
            n_clusters=2, n_init=10, random_state=self._random_state
        ).fit(feats)

    def assign(self, tracks: list[Track], crops: list[np.ndarray]) -> list[TeamAssignment]:
        if len(tracks) != len(crops):
            raise ValueError(f"len(tracks)={len(tracks)} != len(crops)={len(crops)}")
        if self._kmeans is None:
            raise RuntimeError("HsvTeamClassifier not calibrated; call calibrate() first")

        idx_player: list[int] = []
        idx_gk: list[int] = []
        for i, t in enumerate(tracks):
            cls = t.detection.class_id
            if cls == CLS_PLAYER:
                idx_player.append(i)
            elif cls == CLS_GOALKEEPER:
                idx_gk.append(i)

        if not idx_player and not idx_gk:
            return []

        ordered = idx_player + idx_gk
        feats = np.stack([self._features(crops[i]) for i in ordered], axis=0)
        labels = self._kmeans.predict(feats)  # type: ignore[attr-defined]
        centroids = self._kmeans.cluster_centers_  # type: ignore[attr-defined]
        dists = _pairwise_distances(feats, centroids)
        conf = _softmax_confidence(dists, labels)

        out: list[TeamAssignment] = []
        for local, i in enumerate(ordered):
            team_id = int(labels[local])
            if team_id not in (TEAM_A, TEAM_B):
                continue
            out.append(
                TeamAssignment(
                    track_id=tracks[i].track_id,
                    team_id=team_id,  # type: ignore[arg-type]
                    confidence=float(conf[local]),
                )
            )
        return out

    # -- helpers -------------------------------------------------------------

    # Grass-mask thresholds in OpenCV's HSV (H ∈ [0, 179], S ∈ [0, 255]).
    _GRASS_H_LO = 35
    _GRASS_H_HI = 85
    _GRASS_S_MIN = 80

    def _features(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Per-crop H-S histogram on the torso ROI with grass masked out."""
        empty = np.zeros(self._n_bins_h * self._n_bins_s, dtype=np.float32)
        if crop_bgr.size == 0:
            return empty
        h, w = crop_bgr.shape[:2]
        x1f, y1f, x2f, y2f = self._torso_box
        x1 = max(0, int(round(x1f * w)))
        y1 = max(0, int(round(y1f * h)))
        x2 = min(w, int(round(x2f * w)))
        y2 = min(h, int(round(y2f * h)))
        if x2 <= x1 or y2 <= y1:
            return empty
        roi = crop_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, _ = cv2.split(hsv)
        grass = (
            (h_chan >= self._GRASS_H_LO)
            & (h_chan <= self._GRASS_H_HI)
            & (s_chan >= self._GRASS_S_MIN)
        )
        keep = ~grass
        if not keep.any():
            return empty
        hist, _, _ = np.histogram2d(
            h_chan[keep],
            s_chan[keep],
            bins=[self._n_bins_h, self._n_bins_s],
            range=[[0, 180], [0, 256]],
        )
        total = hist.sum()
        if total > 0:
            hist = hist / total
        return hist.astype(np.float32).flatten()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _pairwise_distances(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """L2 distance from each row of ``points`` to each row of ``centroids``."""
    diff = points[:, None, :] - centroids[None, :, :]
    out: np.ndarray = np.linalg.norm(diff, axis=2)
    return out


def _softmax_confidence(dists: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """softmax(-dists) selected at the column for each row's label.

    Calibrated confidence proxy. ``dists.shape == (N, K)``.
    """
    if dists.size == 0:
        return np.zeros((0,), dtype=np.float32)
    z = -dists
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    rows = np.arange(probs.shape[0])
    out: np.ndarray = probs[rows, labels].astype(np.float32)
    return out


__all__ = [
    "EmbeddingTeamClassifier",
    "HsvTeamClassifier",
    "TeamClassifier",
]
