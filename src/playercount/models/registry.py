"""Lazy, thread-safe singleton container for the heavy ML models.

Why this exists:

* YOLO and SigLIP weights are tens to hundreds of megabytes — loading them
  once per request would dominate latency.
* The first request to FastAPI may race the lifespan warmer; both might try
  to build the same model simultaneously. Naive lazy init would load weights
  twice and waste VRAM.
* Tests need to inject fake models without touching the global state.

Design:

* One :class:`ModelRegistry` per process. The FastAPI app stores it in
  :attr:`fastapi.FastAPI.state` and dependency-injects it into routes.
* Per-model getters use double-checked locking with a :class:`threading.Lock`
  (cheap when the fast path is taken — Python only acquires the lock if the
  attribute is still ``None``).
* :meth:`warm` is called from the FastAPI ``lifespan`` so the first real
  request never pays the load cost.

The registry holds *concrete* implementation classes (``YoloDetector``,
``SigLipEmbedder``). The pipeline still talks to the abstract
``Detector``/``TeamClassifier`` protocols — the registry is the only place
that knows the concrete types.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from playercount.detection import YoloDetector
from playercount.team_id import (
    EmbeddingTeamClassifier,
    SigLipEmbedder,
    TeamClassifier,
    TeamClusterer,
)

if TYPE_CHECKING:
    from playercount.config import Settings


class ModelRegistry:
    """Holds the loaded YOLO detector + SigLIP team classifier for the process.

    Construction is cheap; :meth:`warm` (or the first :meth:`detector` /
    :meth:`team_classifier` call) does the actual loading.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._detector: YoloDetector | None = None
        self._embedder: SigLipEmbedder | None = None
        self._classifier: TeamClassifier | None = None
        self._lock = threading.Lock()
        # Resolved torch device string (e.g. "cuda:0", "cpu"), populated by
        # the detector/embedder when they warm. Different from
        # settings.device, which may be "auto".
        self._resolved_device: str | None = None
        # is_ready becomes True only after warm() completes without raising.
        # Distinct from models_loaded (which only checks attributes are set):
        # a partially-warmed registry where YOLO loaded but SigLIP failed has
        # models_loaded=False and is_ready=False, but a successful registry
        # has both True.
        self._is_ready = False
        self._warm_error: str | None = None

    # -- accessors -----------------------------------------------------------

    def detector(self) -> YoloDetector:
        """Return the lazily-initialized detector. Thread-safe."""
        if self._detector is not None:  # fast path: no lock needed
            return self._detector
        with self._lock:
            if self._detector is None:  # double-checked
                det = YoloDetector(
                    weights=self._settings.yolo_weights,
                    device=self._settings.device,
                    conf=self._settings.det_conf,
                    iou=self._settings.det_iou,
                    batch_size=self._settings.yolo_batch_size,
                    registry=self,
                )
                det.warm()
                self._detector = det
            return self._detector

    def embedder(self) -> SigLipEmbedder:
        """Return the lazily-initialized SigLIP embedder. Thread-safe."""
        if self._embedder is not None:
            return self._embedder
        with self._lock:
            if self._embedder is None:
                emb = SigLipEmbedder(
                    model_id=self._settings.siglip_model_id,
                    device=self._settings.device,
                    batch_size=self._settings.siglip_batch_size,
                )
                emb.warm()
                self._embedder = emb
            return self._embedder

    def team_classifier(self) -> TeamClassifier:
        """Return the team classifier (constructed once from embedder + clusterer)."""
        if self._classifier is not None:
            return self._classifier
        with self._lock:
            if self._classifier is None:
                clusterer = self._maybe_load_clusterer()
                self._classifier = EmbeddingTeamClassifier(
                    embedder=self.embedder(),
                    clusterer=clusterer,
                )
            return self._classifier

    # -- introspection -------------------------------------------------------

    def device(self) -> str:
        """Resolved torch device label, falling back to the configured value.

        Returns the actual device string (e.g. ``"cuda:0"`` or ``"cpu"``) once
        a model has been warmed; before warming, returns the configured value
        (which may be ``"auto"``).
        """
        if self._resolved_device is not None:
            return self._resolved_device
        return self._settings.device

    @property
    def models_loaded(self) -> bool:
        return self._detector is not None and self._embedder is not None

    @property
    def is_ready(self) -> bool:
        """``True`` only after :meth:`warm` completes without raising.

        Use for the readiness probe; ``models_loaded`` is just a structural
        check that the attributes are set, which is *necessary but not
        sufficient* for the service to actually answer requests.
        """
        return self._is_ready

    @property
    def warm_error(self) -> str | None:
        """Last exception message from :meth:`warm`, or ``None`` if it succeeded."""
        return self._warm_error

    def set_resolved_device(self, device: str) -> None:
        """Set by ``YoloDetector.warm()`` once the model lands on a torch device."""
        self._resolved_device = device

    # -- lifecycle -----------------------------------------------------------

    def warm(self) -> None:
        """Pre-load every model. Called from the FastAPI ``lifespan``.

        On any exception, ``is_ready`` stays ``False`` and ``warm_error``
        records the message; the lifespan should propagate the exception so
        the container fails fast in production.
        """
        try:
            self.detector()
            self.embedder()
            self.team_classifier()
        except Exception as exc:
            self._warm_error = repr(exc)
            raise
        self._is_ready = True

    # -- helpers -------------------------------------------------------------

    def _maybe_load_clusterer(self) -> TeamClusterer:
        """Load a persisted clusterer if present, otherwise an unfit one.

        An unfit clusterer makes :meth:`EmbeddingTeamClassifier.needs_calibration`
        return ``True`` — the pipeline buffers crops until calibration is run.
        """
        path = self._settings.teams_clusterer_path
        if path.is_file():
            return TeamClusterer.load(path)
        return TeamClusterer(
            n_components=self._settings.umap_components,
            random_state=self._settings.kmeans_random_state,
        )


__all__ = ["ModelRegistry"]
