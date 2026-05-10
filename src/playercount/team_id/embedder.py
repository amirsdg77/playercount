"""SigLIP image-embedding wrapper used to vectorize player crops.

We use SigLIP because it produces semantically rich, lighting-robust embeddings
that cluster well on jersey *appearance* without needing labels. The image
tower runs on whichever device the registry chose; embeddings are L2-normalized
so downstream UMAP / KMeans operate on the unit sphere.

This wrapper is sync (one ``embed`` call per stage invocation) — it is invoked
from a :class:`concurrent.futures.ThreadPoolExecutor` worker.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SigLipEmbedder:
    """Batched SigLIP image-tower wrapper around a HuggingFace transformers model."""

    # SigLIP-base hidden size. Used as a fallback for empty-input shape.
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
        """Load weights + image processor; run a tiny dummy batch to JIT kernels."""
        # Lazy heavy imports.
        import torch  # type: ignore[import-not-found]
        from transformers import AutoImageProcessor, AutoModel  # type: ignore[import-not-found]

        if self._device == "auto":
            self._resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._resolved_device = self._device

        # AutoImageProcessor (vs AutoProcessor) skips loading the text tokeniser
        # — we never touch text, and SiglipTokenizer pulls in `sentencepiece`
        # as a hard dependency we'd rather avoid for the PoC.
        self._processor = AutoImageProcessor.from_pretrained(self._model_id)  # type: ignore[no-untyped-call]
        self._model = AutoModel.from_pretrained(self._model_id).to(self._resolved_device).eval()

        # Dummy forward to JIT/compile and pin memory.
        dummy = np.zeros((8, 8, 3), dtype=np.uint8)  # tiny — processor will resize
        with torch.inference_mode():
            inputs = self._processor(images=[dummy], return_tensors="pt").to(self._resolved_device)
            feats = _extract_image_features(self._model.get_image_features(**inputs))
            self._embed_dim = int(feats.shape[-1])

    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Return (N, D) float32 L2-normalized embeddings for ``len(crops_bgr)`` crops.

        Crops are raw BGR ndarrays of arbitrary shape (the processor handles
        resize + normalize). Empty input returns shape ``(0, D)``.
        """
        if not crops_bgr:
            return np.zeros((0, self._embed_dim), dtype=np.float32)
        if self._model is None or self._processor is None:
            raise RuntimeError("SigLipEmbedder.embed called before warm()")

        import torch  # type: ignore[import-not-found]
        from torch.nn import functional as torchfunc  # type: ignore[import-not-found]

        # Convert BGR → RGB once; the processor expects HxWx3 uint8 RGB.
        # ``[:, :, ::-1]`` produces a negative-stride view that torch.from_numpy
        # rejects in transformers 5.x — we materialise with .copy() so the
        # processor sees a contiguous array.
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

        return np.concatenate(all_chunks, axis=0) if all_chunks else np.zeros(
            (0, self._embed_dim), dtype=np.float32
        )


def _extract_image_features(out: Any) -> Any:
    """Normalise across transformers versions.

    transformers 4.x: ``get_image_features`` returns a plain tensor.
    transformers 5.x: it returns a ``BaseModelOutputWithPooling`` whose
    pooled embedding is at ``.pooler_output`` (shape ``(N, D)``). We accept
    either.
    """
    if hasattr(out, "shape"):
        return out  # tensor
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        # Fallback: mean-pool over patches.
        return out.last_hidden_state.mean(dim=1)
    raise RuntimeError(
        f"unexpected get_image_features return type: {type(out).__name__}"
    )


__all__ = ["SigLipEmbedder"]
