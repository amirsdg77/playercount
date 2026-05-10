"""Fetch model weights into ``models/``.

Usage::

    python scripts/download_weights.py
    PLAYERCOUNT_WEIGHTS_DIR=/some/cache python scripts/download_weights.py

What it pulls (in order, first hit wins):

1. **HuggingFace Hub mirror** — no API key required. We try a public mirror
   of soccer-tuned YOLOv8 weights (e.g. ``keremberke/yolov8m-football-detection``).
2. **Roboflow** — if ``ROBOFLOW_API_KEY`` is set, pulls the canonical
   ``football-players-detection-3zvbc`` weights via the SDK.
3. **Direct URL** — if ``PLAYERCOUNT_YOLO_WEIGHTS_URL`` is set.
4. **Ultralytics COCO fallback** — if all of the above fail and
   ``--allow-coco-fallback`` is passed (or env ``PLAYERCOUNT_ALLOW_COCO=1``),
   downloads vanilla ``yolov8m.pt``. Counts will be less accurate (referees
   become "person") but the pipeline runs end-to-end.

SigLIP weights are pulled lazily by the HuggingFace transformers cache on
first use; this script does not pre-download them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Resolve the target dir from env or the package default.
WEIGHTS_DIR = Path(os.environ.get("PLAYERCOUNT_WEIGHTS_DIR", "./models")).expanduser()
YOLO_FILENAME = "yolov8m-soccer.pt"
ROBOFLOW_PROJECT = "roboflow-jvuqo/football-players-detection-3zvbc"

# Public HF mirrors known to ship soccer-tuned YOLOv8 weights.
# The first one that resolves wins. Each entry is (repo_id, filename).
HF_CANDIDATES: list[tuple[str, str]] = [
    ("keremberke/yolov8m-football-detection", "best.pt"),
    ("keremberke/yolov8n-football-detection", "best.pt"),
]


def _ensure_dir() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def _have_yolo() -> bool:
    return (WEIGHTS_DIR / YOLO_FILENAME).is_file()


def _try_huggingface() -> bool:
    """Try downloading a public HF mirror of soccer-tuned YOLOv8 weights."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError:
        print("[weights] huggingface_hub not installed; skipping HF path", file=sys.stderr)
        return False
    target = WEIGHTS_DIR / YOLO_FILENAME
    for repo_id, filename in HF_CANDIDATES:
        print(f"[weights] trying HuggingFace {repo_id}/{filename}…")
        try:
            local = hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as exc:
            print(f"[weights]   {repo_id} failed: {exc}", file=sys.stderr)
            continue
        shutil.copy(local, target)
        print(f"[weights] OK: copied to {target}")
        return True
    return False


def _try_roboflow() -> bool:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        return False
    try:
        from roboflow import Roboflow  # type: ignore[import-not-found]
    except ImportError:
        print(
            "[weights] roboflow SDK not installed; "
            "install with `pip install roboflow` or set PLAYERCOUNT_YOLO_WEIGHTS_URL.",
            file=sys.stderr,
        )
        return False
    print(f"[weights] downloading {ROBOFLOW_PROJECT} via Roboflow SDK…")
    rf = Roboflow(api_key=api_key)
    workspace, project = ROBOFLOW_PROJECT.split("/", 1)
    version = rf.workspace(workspace).project(project).version(1)
    dataset = version.download(model_format="yolov8", location=str(WEIGHTS_DIR))
    print(f"[weights] roboflow download finished at {dataset.location}")
    src = Path(dataset.location) / "weights.pt"
    if src.is_file():
        src.replace(WEIGHTS_DIR / YOLO_FILENAME)
    return _have_yolo()


def _try_url() -> bool:
    url = os.environ.get("PLAYERCOUNT_YOLO_WEIGHTS_URL")
    if not url:
        return False
    import urllib.request

    target = WEIGHTS_DIR / YOLO_FILENAME
    print(f"[weights] downloading from {url} -> {target}")
    urllib.request.urlretrieve(url, target)
    return target.is_file()


def _try_coco_fallback() -> bool:
    """Download vanilla COCO yolov8m.pt as a last-resort fallback.

    Counts will be less accurate (referees and crowd both surface as "person"),
    but the pipeline at least runs end-to-end. Class remap will collapse all
    ``person`` detections to schema id 0 (player) — see detector.py.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError:
        print("[weights] ultralytics not installed; skipping COCO fallback", file=sys.stderr)
        return False
    print("[weights] falling back to vanilla COCO yolov8m.pt …")
    # Ultralytics auto-downloads to its cache; we copy the file out.
    YOLO("yolov8m.pt")  # triggers the download into ultralytics cache
    # Find the cached file.
    candidates = [
        Path.home() / "AppData" / "Roaming" / "Ultralytics" / "yolov8m.pt",  # Windows
        Path.home() / ".config" / "Ultralytics" / "yolov8m.pt",  # Linux
        Path.cwd() / "yolov8m.pt",  # current dir (newer ultralytics)
    ]
    for cand in candidates:
        if cand.is_file():
            shutil.copy(cand, WEIGHTS_DIR / YOLO_FILENAME)
            print(f"[weights] OK: COCO weights copied to {WEIGHTS_DIR / YOLO_FILENAME}")
            return True
    # Last resort: ultralytics stashes it in CWD by default
    cwd_pt = Path.cwd() / "yolov8m.pt"
    if cwd_pt.is_file():
        shutil.copy(cwd_pt, WEIGHTS_DIR / YOLO_FILENAME)
        return True
    print("[weights] could not locate cached COCO weights after download", file=sys.stderr)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-coco-fallback",
        action="store_true",
        default=os.environ.get("PLAYERCOUNT_ALLOW_COCO", "") in {"1", "true", "yes"},
        help="If soccer-tuned weights cannot be fetched, use vanilla COCO yolov8m.pt.",
    )
    args = parser.parse_args()

    _ensure_dir()
    if _have_yolo():
        print(f"[weights] {YOLO_FILENAME} already present at {WEIGHTS_DIR}")
        return 0

    for attempt in (_try_huggingface, _try_roboflow, _try_url):
        if attempt():
            return 0

    if args.allow_coco_fallback and _try_coco_fallback():
        return 0

    print(
        "\n[weights] could not fetch YOLO weights automatically.\n"
        "Choose one of:\n"
        "  1. (Easiest) re-run with `--allow-coco-fallback` to use vanilla yolov8m.pt.\n"
        "  2. Export ROBOFLOW_API_KEY and re-run "
        "(see https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc).\n"
        "  3. Export PLAYERCOUNT_YOLO_WEIGHTS_URL with a direct download URL.\n"
        "  4. Place the file manually at "
        f"{WEIGHTS_DIR / YOLO_FILENAME}.\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
