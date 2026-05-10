"""Detector protocol and the Ultralytics YOLO implementation.

The pipeline never imports :mod:`ultralytics` directly — it goes through
:class:`Detector`. That keeps the async wiring testable with cheap fakes and
lets us swap YOLOv8 → YOLO11 / RT-DETR without touching anything else.

Class-id remapping is the subtle bit. Roboflow's
``football-players-detection-3zvbc`` weights list classes alphabetically
(``["ball", "goalkeeper", "player", "referee"]``) → ids ``{ball: 0,
goalkeeper: 1, player: 2, referee: 3}``. Our schema (and downstream
counter) uses ``{player: 0, goalkeeper: 1, referee: 2, ball: 3}``. We
build a remap table by name when the model loads so the rest of the
pipeline never has to think about which dataset trained the weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from playercount.schemas import CLASS_NAMES, BBox, Detection

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class Detector(Protocol):
    """A batched object detector. Implementations must be safe to call from a worker thread."""

    def infer(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        """Run inference on a batch of BGR frames.

        Returns one list of :class:`Detection` per input frame, in the same order.
        """
        ...

    def warm(self) -> None:
        """Optional pre-load / JIT warmup. May be a no-op."""
        ...


# Inverse of CLASS_NAMES — schema id by name. Used to build the remap table.
_NAME_TO_SCHEMA_ID: dict[str, int] = {v: k for k, v in CLASS_NAMES.items()}


class YoloDetector:
    """Ultralytics YOLOv8 wrapper configured for the Roboflow soccer schema.

    Construction loads the weights once (during :meth:`warm`); ``infer`` is
    the per-batch hot path. The CUDA forward releases the GIL so this
    object is invoked from a :class:`concurrent.futures.ThreadPoolExecutor`
    worker.
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
        self._model: Any = None  # ultralytics.YOLO once loaded
        self._resolved_device: str = "cpu"
        # Maps weight-file class id → schema class id; populated on warm.
        # If a class is missing in the weights, it's just absent from the map
        # and detections for that class are dropped.
        self._class_remap: dict[int, int] = {}

    # -- properties ---------------------------------------------------------

    @property
    def resolved_device(self) -> str:
        return self._resolved_device

    # -- lifecycle -----------------------------------------------------------

    def warm(self) -> None:
        """Load weights and run a dummy forward pass so the first ``infer`` is fast."""
        # Lazy ultralytics import — heavy dep; we don't pay for it in unit tests.
        import numpy as np
        import torch  # type: ignore[import-not-found]
        from ultralytics import YOLO  # type: ignore[import-not-found,attr-defined]

        if not self._weights.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found at {self._weights}. "
                "Run `python scripts/download_weights.py`."
            )

        # Resolve device.
        if self._device == "auto":
            self._resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._resolved_device = self._device
        if self._registry is not None and hasattr(self._registry, "set_resolved_device"):
            self._registry.set_resolved_device(self._resolved_device)

        self._model = YOLO(str(self._weights))
        # ultralytics models hold a names dict like {0: "ball", 1: "goalkeeper", ...}.
        # Build the remap (weight id → schema id).
        names = getattr(self._model, "names", None) or {}
        self._class_remap = {}
        for wid, name in names.items():
            schema_id = _NAME_TO_SCHEMA_ID.get(str(name).lower())
            if schema_id is not None:
                self._class_remap[int(wid)] = int(schema_id)

        # COCO fallback: when only `person` matches and no soccer classes do
        # (i.e. no "player" name in the weights), treat all `person`
        # detections as players. This is an explicit graceful-degradation
        # path so the pipeline stays runnable even without soccer-tuned
        # weights — counts will lump referees in with players.
        if not self._class_remap and "person" in {str(n).lower() for n in names.values()}:
            for wid, name in names.items():
                if str(name).lower() == "person":
                    self._class_remap[int(wid)] = 0  # → schema "player"

        # Dummy forward to JIT/compile kernels and lay out CUDA memory.
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

        # ultralytics expects BGR ndarrays as a list — happy path.
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
                weight_cls = int(clses[i])
                schema_cls = self._class_remap.get(weight_cls)
                if schema_cls is None:
                    continue  # class not in our schema — drop
                x1, y1, x2, y2 = xyxy[i].tolist()
                # Defensive: the model can emit boxes with x1==x2 on tiny
                # detections (rare). Skip them so the BBox validator doesn't raise.
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
        """Build a :class:`Detection` from raw YOLO outputs (already remapped)."""
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
