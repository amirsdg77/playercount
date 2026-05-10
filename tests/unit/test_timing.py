"""StageTimer + Counters."""

from __future__ import annotations

import time

from playercount.utils.timing import Counters, StageTimer


def test_stage_timer_records_elapsed():
    c = Counters()
    with StageTimer("decode", c):
        time.sleep(0.005)
    snap = c.snapshot()
    hist = snap["histograms"]["decode"]  # type: ignore[index]
    assert hist["count"] == 1
    assert hist["total_s"] >= 0.005


def test_counter_inc():
    c = Counters()
    c.inc("frames")
    c.inc("frames", 4)
    assert c.snapshot()["counters"] == {"frames": 5}  # type: ignore[index]


def test_render_prometheus_smoke():
    c = Counters()
    c.inc("frames", 3)
    with StageTimer("decode", c):
        pass
    text = c.render_prometheus()
    assert "playercount_frames" in text
    assert "playercount_decode_seconds" in text
    assert "# TYPE" in text


def test_time_context_manager():
    c = Counters()
    with c.time("embed"):
        time.sleep(0.001)
    assert c.snapshot()["histograms"]["embed"]["count"] == 1  # type: ignore[index]
