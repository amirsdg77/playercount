"""UMAP + KMeans clusterer that turns SigLIP embeddings into team labels.

The fit is one-shot per match: we pull ~150 player crops, embed them, fit
UMAP(3) + KMeans(2), and persist the result via joblib. Inference is just
``umap.transform`` + ``kmeans.predict``.

Goalkeepers wear a different kit so they pollute the KMeans fit if included —
the team classifier excludes them at fit time and snaps them to the nearer of
the two team centroids at predict time (using :meth:`transform` to bring the
GK embedding into UMAP space, then computing distances to :meth:`centroids`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class TeamClusterer:
    """UMAP(n_components=3) → KMeans(n_clusters=2). Persistable via joblib."""

    def __init__(
        self,
        *,
        n_components: int = 3,
        random_state: int = 42,
    ) -> None:
        self._n_components = n_components
        self._random_state = random_state
        self._reducer: Any = None  # umap.UMAP once fit
        self._kmeans: Any = None  # sklearn.cluster.KMeans once fit

    # -- properties ----------------------------------------------------------

    @property
    def is_fit(self) -> bool:
        return self._reducer is not None and self._kmeans is not None

    @property
    def n_components(self) -> int:
        return self._n_components

    # -- training / inference ------------------------------------------------

    def fit(self, embeddings: np.ndarray) -> None:
        """Fit UMAP then KMeans on a (N, D) embeddings matrix."""
        if embeddings.ndim != 2:
            raise ValueError(f"expected 2D embeddings, got shape {embeddings.shape}")
        n = embeddings.shape[0]
        if n < max(2, self._n_components + 1):
            raise ValueError(
                f"need at least {self._n_components + 1} embeddings to fit, got {n}"
            )

        import umap  # type: ignore[import-not-found]
        from sklearn.cluster import KMeans

        # n_neighbors must be <= n_samples - 1; clamp to keep the fit valid
        # on small calibration sets.
        n_neighbors = max(2, min(15, n - 1))
        self._reducer = umap.UMAP(
            n_components=self._n_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=self._random_state,
        ).fit(embeddings)
        reduced = self._reducer.transform(embeddings)
        self._kmeans = KMeans(
            n_clusters=2, n_init=10, random_state=self._random_state
        ).fit(reduced)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Return shape ``(N,)`` int labels in {0, 1}. Requires :pyattr:`is_fit`."""
        if not self.is_fit:
            raise RuntimeError("TeamClusterer.predict called before fit()")
        if embeddings.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        reduced = self._reducer.transform(embeddings)
        labels: np.ndarray = self._kmeans.predict(reduced).astype(np.int64)
        return labels

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings into UMAP-reduced space."""
        if not self.is_fit:
            raise RuntimeError("TeamClusterer.transform called before fit()")
        if embeddings.shape[0] == 0:
            return np.zeros((0, self._n_components), dtype=np.float32)
        result: np.ndarray = np.asarray(self._reducer.transform(embeddings))
        return result

    def centroids(self) -> np.ndarray:
        """KMeans centroids in UMAP-reduced space, shape ``(2, n_components)``."""
        if not self.is_fit:
            raise RuntimeError("TeamClusterer.centroids called before fit()")
        return np.asarray(self._kmeans.cluster_centers_)

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist ``(reducer, kmeans, meta)`` to a single joblib file."""
        if not self.is_fit:
            raise RuntimeError("TeamClusterer.save called before fit()")
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "reducer": self._reducer,
                "kmeans": self._kmeans,
                "n_components": self._n_components,
                "random_state": self._random_state,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> TeamClusterer:
        """Restore a fitted clusterer from a joblib file."""
        import joblib

        payload = joblib.load(Path(path))
        instance = cls(
            n_components=int(payload.get("n_components", 3)),
            random_state=int(payload.get("random_state", 42)),
        )
        instance._reducer = payload["reducer"]
        instance._kmeans = payload["kmeans"]
        return instance


__all__ = ["TeamClusterer"]
