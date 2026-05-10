"""Runtime configuration: defaults → YAML overlay → environment variables.

Resolution order (last wins):

1. Class defaults on :class:`Settings`.
2. Values from the YAML file at ``$PLAYERCOUNT_CONFIG_FILE`` (or argument to
   :func:`load_settings`), under the nested keys that match attribute names.
3. ``PLAYERCOUNT_*`` environment variables.

Algorithmic tunables (thresholds, batch sizes, queue depths) typically live in
the YAML so they're version-controlled per experiment; deployment knobs (paths,
device, log level) live in env vars so they can be overridden per container.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
Device = Literal["cuda", "cpu", "auto"]


class Settings(BaseSettings):
    """All runtime configuration for the pipeline and API service."""

    model_config = SettingsConfigDict(
        env_prefix="PLAYERCOUNT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- paths ----
    yolo_weights: Path = Path("./models/yolov8m-soccer.pt")
    siglip_model_id: str = "google/siglip-base-patch16-224"
    weights_dir: Path = Path("./models")
    config_file: Path | None = None
    teams_clusterer_path: Path = Path("./models/teams.joblib")

    # ---- device / perf ----
    device: Device = "auto"
    yolo_batch_size: int = Field(default=8, ge=1, le=64)
    siglip_batch_size: int = Field(default=64, ge=1, le=256)
    sampling_stride: int = Field(default=1, ge=1)
    queue_maxsize: int = Field(default=8, ge=1, le=64)
    executor_max_workers: int | None = Field(
        default=None,
        description="Override for the ThreadPoolExecutor; default uses min(8, cpu+4).",
    )

    # ---- detection thresholds ----
    det_conf: float = Field(default=0.35, ge=0.0, le=1.0)
    det_iou: float = Field(default=0.5, ge=0.0, le=1.0)

    # ---- aggregation ----
    track_window: int = Field(default=30, ge=1)
    cluster_min_samples: int = Field(default=60, ge=10)
    umap_components: int = Field(default=3, ge=2, le=10)
    kmeans_random_state: int = 42

    # ---- logging ----
    log_level: LogLevel = "INFO"
    log_json: bool = True

    @field_validator("yolo_weights", "weights_dir", "teams_clusterer_path", mode="before")
    @classmethod
    def _expand_path(cls, v: object) -> object:
        if isinstance(v, str):
            return Path(v).expanduser()
        return v


# ---------------------------------------------------------------------------
# YAML overlay
# ---------------------------------------------------------------------------


def _flatten_yaml(data: dict[str, Any], parent: str = "") -> dict[str, Any]:
    """Flatten ``{detection: {conf: 0.3}}`` into ``{"det_conf": 0.3}``-style keys.

    The yaml file uses logical groupings for human readability; we map them onto
    the flat attribute names on :class:`Settings`. Unknown groups/keys are
    ignored — :class:`Settings` itself drops extras silently
    (``extra="ignore"``).
    """
    aliases: dict[str, str] = {
        "detection.conf": "det_conf",
        "detection.iou": "det_iou",
        "detection.yolo_batch_size": "yolo_batch_size",
        "team_id.siglip_batch_size": "siglip_batch_size",
        "team_id.cluster_min_samples": "cluster_min_samples",
        "team_id.umap_components": "umap_components",
        "team_id.kmeans_random_state": "kmeans_random_state",
        "aggregation.track_window": "track_window",
        "pipeline.queue_maxsize": "queue_maxsize",
        "pipeline.sampling_stride": "sampling_stride",
        "pipeline.executor_max_workers": "executor_max_workers",
    }
    out: dict[str, Any] = {}
    for k, v in data.items():
        path = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten_yaml(v, path))
        else:
            mapped = aliases.get(path, k)
            out[mapped] = v
    return out


def load_settings(yaml_path: Path | str | None = None) -> Settings:
    """Build :class:`Settings`, layering YAML below env vars.

    Resolution: defaults → YAML (if any) → env (env wins). The YAML file path
    can be passed explicitly, or supplied via ``PLAYERCOUNT_CONFIG_FILE``.
    """
    yaml_overrides: dict[str, Any] = {}

    candidate = (
        Path(yaml_path).expanduser()
        if yaml_path is not None
        else Settings().config_file  # respects env-overridden default
    )
    if candidate is not None and Path(candidate).is_file():
        with Path(candidate).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{candidate}: top-level YAML must be a mapping")
        yaml_overrides = _flatten_yaml(raw)

    # Re-instantiating after passing yaml_overrides as kwargs lets pydantic-settings
    # reapply env-var precedence on top of them — the standard pydantic merge rule
    # keeps explicit kwargs from being clobbered, but env-derived fields *are*
    # taken first when the kwarg is not provided. To get "env beats yaml" we
    # therefore strip yaml keys that env actually set.
    env_set = {k for k in Settings.model_fields if f"PLAYERCOUNT_{k.upper()}" in _env_keys()}
    yaml_overrides = {k: v for k, v in yaml_overrides.items() if k not in env_set}
    return Settings(**yaml_overrides)


def _env_keys() -> set[str]:
    """Cheap helper kept separate so tests can monkey-patch ``os.environ``."""
    import os

    return set(os.environ.keys())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached :class:`Settings` instance.

    Used by FastAPI dependency injection; tests should override via FastAPI's
    ``app.dependency_overrides`` rather than mutating this cache.
    """
    return load_settings()


__all__ = ["Device", "LogLevel", "Settings", "get_settings", "load_settings"]
