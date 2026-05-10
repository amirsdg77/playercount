"""Settings: env > yaml > defaults; validators; path expansion."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from playercount.config import Settings, load_settings


def test_defaults_are_sensible():
    s = Settings()
    assert s.det_conf == 0.35
    assert s.yolo_batch_size == 8
    assert s.queue_maxsize == 8
    assert s.track_window == 30
    assert s.log_level == "INFO"


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("PLAYERCOUNT_DET_CONF", "0.7")
    monkeypatch.setenv("PLAYERCOUNT_LOG_LEVEL", "DEBUG")
    s = Settings()
    assert s.det_conf == pytest.approx(0.7)
    assert s.log_level == "DEBUG"


def test_yaml_overrides_default(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            detection:
              conf: 0.42
              iou: 0.55
              yolo_batch_size: 16
            aggregation:
              track_window: 45
            pipeline:
              queue_maxsize: 12
            """
        )
    )
    s = load_settings(cfg)
    assert s.det_conf == pytest.approx(0.42)
    assert s.det_iou == pytest.approx(0.55)
    assert s.yolo_batch_size == 16
    assert s.track_window == 45
    assert s.queue_maxsize == 12


def test_env_beats_yaml(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("detection:\n  conf: 0.42\n")
    monkeypatch.setenv("PLAYERCOUNT_DET_CONF", "0.99")
    s = load_settings(cfg)
    assert s.det_conf == pytest.approx(0.99)


def test_invalid_conf_rejected():
    with pytest.raises(ValidationError):
        Settings(det_conf=1.5)


def test_invalid_stride_rejected():
    with pytest.raises(ValidationError):
        Settings(sampling_stride=0)


def test_path_expansion_for_tilde():
    home = os.path.expanduser("~")
    s = Settings(yolo_weights="~/weights.pt")
    assert str(s.yolo_weights).startswith(home)


def test_unknown_yaml_keys_are_ignored(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("unknown:\n  thing: 42\ndet_conf: 0.6\n")
    s = load_settings(cfg)
    assert s.det_conf == pytest.approx(0.6)
