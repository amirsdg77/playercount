"""SigLIP image-embedding wrapper.

Maps player crops to L2-normalised embedding vectors. The image tower runs
on whichever device the registry chose; output lives on the unit sphere so
downstream UMAP / KMeans operate on cosine geometry.

This wrapper is synchronous (one ``embed`` call per stage invocation) and
is invoked from a :class:`concurrent.futures.ThreadPoolExecutor` worker.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch  # type: ignore[import-not-found]
from torch.nn import functional as torchfunc  # type: ignore[import-not-found]
from transformers import AutoImageProcessor, AutoModel  # type: ignore[import-not-found]


class SigLipEmbedder:
    """Batched SigLIP image-tower wrapper around a HuggingFace transformers model."""

    # SigLIP-base hidden size; fallback for empty-input shape.
    _DEFAULT_DIM = 768

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        *,
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None
        self._processor: Any = None
        self._resolved_device: str = "cpu"
        self._embed_dim: int = self._DEFAULT_DIM

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def warm(self) -> None:
        """Load weights + image processor; run a dummy forward to JIT kernels."""
        if self._device == "auto":
            self._resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._resolved_device = self._device

        # AutoImageProcessor (not AutoProcessor) skips loading the text
        # tokeniser, which would pull `sentencepiece` in as a hard dependency.
        self._processor = AutoImageProcessor.from_pretrained(self._model_id)  # type: ignore[no-untyped-call]
        self._model = (
            AutoModel.from_pretrained(self._model_id).to(self._resolved_device).eval()
        )

        dummy = np.zeros((8, 8, 3), dtype=np.uint8)
        with torch.inference_mode():
            inputs = self._processor(images=[dummy], return_tensors="pt").to(
                self._resolved_device
            )
            feats = _extract_image_features(self._model.get_image_features(**inputs))
            self._embed_dim = int(feats.shape[-1])

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Return ``(N, D)`` float32 L2-normalised embeddings.

        Crops are raw BGR ndarrays of arbitrary shape; the processor handles
        resize + normalise. Empty input returns shape ``(0, D)``.
        """
        if not crops_bgr:
            return np.zeros((0, self._embed_dim), dtype=np.float32)
        if self._model is None or self._processor is None:
            raise RuntimeError("SigLipEmbedder.embed called before warm()")

        # ``[:, :, ::-1]`` produces a negative-stride view that torch rejects;
        # materialise with .copy() so the processor sees a contiguous array.
        rgb_crops = [c[:, :, ::-1].copy() for c in crops_bgr]

        all_chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(rgb_crops), self._batch_size):
                batch = rgb_crops[start : start + self._batch_size]
                inputs = self._processor(images=batch, return_tensors="pt").to(
                    self._resolved_device
                )
                feats = _extract_image_features(self._model.get_image_features(**inputs))
                feats = torchfunc.normalize(feats, dim=-1)
                all_chunks.append(feats.detach().cpu().numpy().astype(np.float32))

        return (
            np.concatenate(all_chunks, axis=0)
            if all_chunks
            else np.zeros((0, self._embed_dim), dtype=np.float32)
        )


def _extract_image_features(out: Any) -> Any:
    """Normalise across transformers 4.x (returns tensor) and 5.x (returns object)."""
    if hasattr(out, "shape"):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state.mean(dim=1)
    raise RuntimeError(
        f"unexpected get_image_features return type: {type(out).__name__}"
    )


__all__ = ["SigLipEmbedder"]
