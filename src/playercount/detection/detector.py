"""Detector protocol and the Ultralytics YOLO implementation.

The pipeline only talks to :class:`Detector`; this file is the single place
that imports :mod:`ultralytics`. Class IDs from the weights file are remapped
to the project's schema (``{player: 0, goalkeeper: 1, referee: 2, ball: 3}``)
by name, so the rest of the pipeline is independent of which dataset trained
the weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch  # type: ignore[import-not-found]
from ultralytics import YOLO  # type: ignore[import-not-found,attr-defined]

from playercount.constants import CLS_PLAYER
from playercount.schemas import CLASS_NAMES, BBox, Detection


@runtime_checkable
class Detector(Protocol):
    """Batched object detector. Implementations must be thread-safe."""

    def infer(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        """Run inference on a batch of BGR frames; one ``Detection`` list per input frame."""
        ...

    def warm(self) -> None:
        """Optional pre-load / JIT warmup."""
        ...


# Inverse of CLASS_NAMES — schema id by name; used to build the remap table.
_NAME_TO_SCHEMA_ID: dict[str, int] = {v: k for k, v in CLASS_NAMES.items()}


class YoloDetector:
    """Ultralytics YOLOv8 wrapper configured for the project's class schema.

    Construction is cheap; :meth:`warm` loads the weights and runs a dummy
    forward pass. :meth:`infer` is invoked from a
    :class:`concurrent.futures.ThreadPoolExecutor` worker.
    """

    def __init__(
        self,
        weights: Path,
        *,
        device: str = "auto",
        conf: float = 0.35,
        iou: float = 0.5,
        batch_size: int = 8,
        imgsz: int = 1280,
        registry: Any | None = None,
    ) -> None:
        self._weights = Path(weights)
        self._device = device
        self._conf = conf
        self._iou = iou
        self._batch_size = batch_size
        self._imgsz = imgsz
        self._registry = registry  # back-ref so we can publish resolved device
        self._model: Any = None
        self._resolved_device: str = "cpu"
        # Maps weight-file class id → schema class id; populated on warm.
        # Classes absent from the map are dropped at inference time.
        self._class_remap: dict[int, int] = {}

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

    # -- lifecycle -----------------------------------------------------------

    def warm(self) -> None:
        """Load weights, build the class remap, and JIT-compile with a dummy forward."""
        if not self._weights.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found at {self._weights}. "
                "Run `python scripts/download_weights.py`."
            )

        if self._device == "auto":
            self._resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._resolved_device = self._device
        if self._registry is not None and hasattr(self._registry, "set_resolved_device"):
            self._registry.set_resolved_device(self._resolved_device)

        self._model = YOLO(str(self._weights))
        names = getattr(self._model, "names", None) or {}
        self._class_remap = {}
        for wid, name in names.items():
            schema_id = _NAME_TO_SCHEMA_ID.get(str(name).lower())
            if schema_id is not None:
                self._class_remap[int(wid)] = int(schema_id)

        # COCO fallback: when the weights only know `person`, map it to the
        # schema's `player` so the pipeline still runs end-to-end (counts
        # will lump referees into the player class).
        if not self._class_remap and "person" in {str(n).lower() for n in names.values()}:
            for wid, name in names.items():
                if str(name).lower() == "person":
                    self._class_remap[int(wid)] = CLS_PLAYER

        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._resolved_device,
            verbose=False,
        )

    # -- inference -----------------------------------------------------------

    def infer(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        """Run YOLO over a batch of BGR frames; return per-frame detection lists."""
        if not frames_bgr:
            return []
        if self._model is None:
            raise RuntimeError("YoloDetector.infer called before warm()")

        results = self._model.predict(
            frames_bgr,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._resolved_device,
            verbose=False,
        )

        out: list[list[Detection]] = []
        for r in results:
            per_frame: list[Detection] = []
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                out.append(per_frame)
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(boxes)):
                schema_cls = self._class_remap.get(int(clses[i]))
                if schema_cls is None:
                    continue
                x1, y1, x2, y2 = xyxy[i].tolist()
                # The model can emit degenerate boxes on tiny detections;
                # skip them rather than tripping the BBox validator.
                if x2 <= x1 or y2 <= y1:
                    continue
                per_frame.append(self._to_detection((x1, y1, x2, y2), float(scores[i]), schema_cls))
            out.append(per_frame)
        return out

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _to_detection(
        xyxy: tuple[float, float, float, float], score: float, cls: int
    ) -> Detection:
        """Build a :class:`Detection` from raw YOLO outputs (class already remapped)."""
        if cls not in CLASS_NAMES:
            raise ValueError(f"unexpected schema class id {cls}")
        x1, y1, x2, y2 = xyxy
        return Detection(
            bbox=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
            score=max(0.0, min(1.0, float(score))),
            class_id=cls,  # type: ignore[arg-type]
            class_name=CLASS_NAMES[cls],
        )


__all__ = ["Detector", "YoloDetector"]
