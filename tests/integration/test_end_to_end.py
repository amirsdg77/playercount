"""End-to-end integration tests against the real ``data/sample.mp4`` and weights.

Skipped by default — run with ``pytest -m slow`` after::

    make download-weights
    cp ../sample.mp4 data/

These tests assume the ML stage bodies have been implemented; until then they
will fail fast with a NotImplementedError, which is exactly what we want
(integration tests are the exit gate from skeleton → working PoC).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

SAMPLE = Path("data/sample.mp4")
WEIGHTS = Path("models/yolov8m-soccer.pt")


def _skip_if_missing() -> None:
    if not SAMPLE.is_file():
        pytest.skip(f"sample video missing: {SAMPLE}")
    if not WEIGHTS.is_file():
        pytest.skip(f"weights missing: {WEIGHTS}")


@pytest.mark.asyncio
async def test_end_to_end_summary(client):
    """POST /analyze (summary) returns counts with non-zero team means."""
    _skip_if_missing()
    with SAMPLE.open("rb") as fh:
        files = {"file": ("sample.mp4", fh, "video/mp4")}
        r = await client.post("/analyze", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["frames_processed"] > 0
    assert body["summary"]["team_a_mean"] > 0
    assert body["summary"]["team_b_mean"] > 0


@pytest.mark.asyncio
async def test_stream_endpoint_yields_ndjson_lines(client):
    """/analyze/stream produces well-formed NDJSON."""
    _skip_if_missing()
    with SAMPLE.open("rb") as fh:
        files = {"file": ("sample.mp4", fh, "video/mp4")}
        async with client.stream("POST", "/analyze/stream", files=files) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("application/x-ndjson")
            count = 0
            async for line in r.aiter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                assert "frame_index" in payload
                assert "team_a_count" in payload
                count += 1
            assert count >= 100
