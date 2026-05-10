"""Team-classification strategies.

The pipeline only knows about :class:`TeamClassifier` — a small Protocol.
Two implementations are provided:

* :class:`EmbeddingTeamClassifier` — primary path. SigLIP embeddings → UMAP →
  KMeans(k=2). Robust to lighting, no labels needed, the recipe Roboflow ship
  for the "sports" example. Goalkeepers (different kit) are excluded from the
  fit and snapped at predict time to the nearer team centroid.
* :class:`HsvTeamClassifier` — CPU-only fallback. Torso-region HSV histogram +
  KMeans(k=2). Deterministic, dependency-light, sub-millisecond per crop.
  Brittle when team kits have similar luminance.

Referees (class id 2) are filtered before any clustering — they get
``team_id=None`` and never participate in the count.

The :class:`TeamClassifier` protocol is the swap point: configuration toggles
which implementation the registry constructs, and nothing else has to change.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from playercount.schemas import TeamAssignment, Track

if TYPE_CHECKING:
    from playercount.team_id.clusterer import TeamClusterer
    from playercount.team_id.embedder import SigLipEmbedder


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TeamClassifier(Protocol):
    """Maps tracks → team assignments. Implementations may be stateful (require calibration)."""

    def assign(self, tracks: list[Track], crops: list[np.ndarray]) -> list[TeamAssignment]:
        """Return one :class:`TeamAssignment` per track that received a team label.

        ``crops[i]`` is the BGR pixel crop of ``tracks[i].detection.bbox``.
        Tracks of class ``referee`` or ``ball`` are skipped (no assignment).
        """
        ...

    def needs_calibration(self) -> bool:
        """Return ``True`` if :meth:`calibrate` must be called before :meth:`assign` works."""
        ...

    def calibrate(self, crops: list[np.ndarray]) -> None:
        """Fit any internal models on a representative sample of player crops."""
        ...


# ---------------------------------------------------------------------------
# Primary: SigLIP + UMAP + KMeans
# ---------------------------------------------------------------------------

# Class-id constants mirror playercount.schemas.CLASS_NAMES (kept inline to
# avoid a runtime import cycle when the classifier is used in hot paths).
_CLS_PLAYER = 0
_CLS_GOALKEEPER = 1
_CLS_REFEREE = 2
_CLS_BALL = 3


class EmbeddingTeamClassifier:
    """Primary unsupervised classifier: SigLIP image embeddings + UMAP + KMeans(k=2).

    Calibration is a one-shot fit on a buffer of *player-only* crops collected
    from the first ~150 detected frames; goalkeepers and referees are excluded
    from the buffer so the two KMeans centroids represent the two outfield kits.

    At assign-time:

    * Players → KMeans label.
    * Goalkeepers → distance from GK embedding to each team centroid (in UMAP
      space) → argmin → that team's id.
    * Referees / ball → no assignment returned.
    """

    def __init__(self, embedder: SigLipEmbedder, clusterer: TeamClusterer) -> None:
        self._embedder = embedder
        self._clusterer = clusterer
        # threading.Lock (not asyncio.Lock!) — assign() runs on a worker
        # thread via run_in_executor; an asyncio.Lock would be a RuntimeError
        # there. The lock guards calibrate()'s clusterer.fit() so an in-flight
        # assign() never sees a half-fitted clusterer if calibration happens
        # mid-stream. Both threads are CPython under GIL → cheap.
        self._fit_lock = threading.Lock()

    # -- protocol methods ----------------------------------------------------

    def needs_calibration(self) -> bool:
        return not self._clusterer.is_fit

    def calibrate(self, crops: list[np.ndarray]) -> None:
        """Embed ``crops`` and fit the underlying clusterer.

        Caller is expected to hand in *player-only* crops; goalkeepers/referees
        should be filtered upstream by the calibration script.
        """
        if not crops:
            raise ValueError("calibrate requires a non-empty crop list")
        embeddings = self._embedder.embed(crops)
        with self._fit_lock:
            self._clusterer.fit(embeddings)

    def assign(self, tracks: list[Track], crops: list[np.ndarray]) -> list[TeamAssignment]:
        if len(tracks) != len(crops):
            raise ValueError(f"len(tracks)={len(tracks)} != len(crops)={len(crops)}")
        if self.needs_calibration():
            # Fail loud rather than silently disabling counting (audit #14).
            # The runner is expected to handle calibration before pipeline
            # start; if we get here the precondition was missed.
            raise RuntimeError(
                "EmbeddingTeamClassifier is not calibrated. "
                "Call calibrate(crops) once before pipeline.run()."
            )

        # Snapshot under the lock so we don't race a concurrent calibrate().
        # The actual heavy work (embedding + predict) happens outside the
        # lock so calibration can complete promptly.
        with self._fit_lock:
            if self.needs_calibration():
                raise RuntimeError("clusterer became unfit between checks")
        return self._assign_impl(tracks, crops)

    # -- implementation ------------------------------------------------------

    def _assign_impl(
        self, tracks: list[Track], crops: list[np.ndarray]
    ) -> list[TeamAssignment]:
        """Embed all classifiable crops in one batch, then label players + GKs."""
        # Partition by class. Track refs/balls are dropped — no assignment.
        idx_player: list[int] = []
        idx_gk: list[int] = []
        for i, t in enumerate(tracks):
            cls = t.detection.class_id
            if cls == _CLS_PLAYER:
                idx_player.append(i)
            elif cls == _CLS_GOALKEEPER:
                idx_gk.append(i)
            # _CLS_REFEREE and _CLS_BALL → skipped silently.

        if not idx_player and not idx_gk:
            return []

        # One batched forward through SigLIP for all classifiable crops.
        ordered_idx = idx_player + idx_gk
        ordered_crops = [crops[i] for i in ordered_idx]
        embeddings = self._embedder.embed(ordered_crops)
        if embeddings.shape[0] != len(ordered_idx):
            # Defensive — embedder must return one row per input crop.
            raise RuntimeError(
                f"embedder returned {embeddings.shape[0]} rows for "
                f"{len(ordered_idx)} crops"
            )

        n_players = len(idx_player)
        emb_p = embeddings[:n_players] if n_players else np.zeros((0, embeddings.shape[1]), dtype=embeddings.dtype)
        emb_gk = embeddings[n_players:] if idx_gk else np.zeros((0, embeddings.shape[1]), dtype=embeddings.dtype)

        out: list[TeamAssignment] = []

        # --- Players: KMeans predict on UMAP-reduced embeddings ---
        if n_players:
            labels_p = self._clusterer.predict(emb_p)  # shape (n_players,)
            # Confidence proxy = softmax over (-distance to each centroid)
            # in UMAP space.
            reduced_p = self._clusterer.transform(emb_p)
            centroids = self._clusterer.centroids()
            dists_p = _pairwise_distances(reduced_p, centroids)
            conf_p = _softmax_confidence(dists_p, labels_p)
            for local, i in enumerate(idx_player):
                team_id = int(labels_p[local])
                # Clip to {0,1} defensively in case KMeans ever returned otherwise.
                if team_id not in (0, 1):
                    continue
                out.append(
                    TeamAssignment(
                        track_id=tracks[i].track_id,
                        team_id=team_id,  # type: ignore[arg-type]
                        confidence=float(conf_p[local]),
                    )
                )

        # --- Goalkeepers: snap to nearer centroid ---
        if idx_gk:
            reduced_gk = self._clusterer.transform(emb_gk)
            centroids = self._clusterer.centroids()
            dists_gk = _pairwise_distances(reduced_gk, centroids)
            labels_gk = np.argmin(dists_gk, axis=1)
            conf_gk = _softmax_confidence(dists_gk, labels_gk)
            for local, i in enumerate(idx_gk):
                team_id = int(labels_gk[local])
                if team_id not in (0, 1):
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
    """CPU-only classical fallback. Torso-ROI HSV histogram + per-stream KMeans.

    Useful when:

    * The deployment target lacks a GPU.
    * Determinism matters more than accuracy on edge-case kits.
    * SigLIP weights are unavailable.

    Limitations: brittle when both kits have similar luminance/hue
    (e.g. red-vs-orange).
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
        from sklearn.cluster import KMeans

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
            if cls == _CLS_PLAYER:
                idx_player.append(i)
            elif cls == _CLS_GOALKEEPER:
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
            if team_id not in (0, 1):
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

    def _features(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Per-crop H-S histogram on the torso ROI with grass masked out."""
        import cv2  # type: ignore[import-not-found]

        if crop_bgr.size == 0:
            return np.zeros(self._n_bins_h * self._n_bins_s, dtype=np.float32)
        h, w = crop_bgr.shape[:2]
        x1f, y1f, x2f, y2f = self._torso_box
        x1 = max(0, int(round(x1f * w)))
        y1 = max(0, int(round(y1f * h)))
        x2 = min(w, int(round(x2f * w)))
        y2 = min(h, int(round(y2f * h)))
        if x2 <= x1 or y2 <= y1:
            return np.zeros(self._n_bins_h * self._n_bins_s, dtype=np.float32)
        roi = crop_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, _ = cv2.split(hsv)
        # Mask out green (grass): H in [35, 85], S >= 80 in OpenCV's 0-179/0-255.
        grass = (h_chan >= 35) & (h_chan <= 85) & (s_chan >= 80)
        keep = ~grass
        if not keep.any():
            return np.zeros(self._n_bins_h * self._n_bins_s, dtype=np.float32)
        h_vals = h_chan[keep]
        s_vals = s_chan[keep]
        hist, _, _ = np.histogram2d(
            h_vals, s_vals,
            bins=[self._n_bins_h, self._n_bins_s],
            range=[[0, 180], [0, 256]],
        )
        # L1-normalise so brightness doesn't bias the cluster.
        total = hist.sum()
        if total > 0:
            hist = hist / total
        return hist.astype(np.float32).flatten()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _pairwise_distances(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """L2 distance from each row of ``points`` to each row of ``centroids``."""
    # (N, 1, D) - (1, K, D) → (N, K, D); norm along D → (N, K)
    diff = points[:, None, :] - centroids[None, :, :]
    out: np.ndarray = np.linalg.norm(diff, axis=2)
    return out


def _softmax_confidence(dists: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """softmax(-dists) and pick the column corresponding to each row's label.

    Used as a calibrated confidence proxy. `dists.shape == (N, K)`.
    """
    if dists.size == 0:
        return np.zeros((0,), dtype=np.float32)
    # Numerical stability: subtract row-min before exponentiating.
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
