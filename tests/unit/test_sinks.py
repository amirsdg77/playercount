"""JSON / NDJSON sinks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from playercount.io.sinks import JsonSink, NdjsonSink, ndjson_stream
from playercount.schemas import FrameResult


def _frame(i: int) -> FrameResult:
    return FrameResult(
        frame_index=i,
        timestamp_s=i * 0.1,
        team_a_count=10,
        team_b_count=11,
        referee_count=2,
        goalkeeper_a_count=1,
        goalkeeper_b_count=1,
    )


# ---------------------------------------------------------------------------
# JsonSink
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_sink_writes_one_doc(tmp_path: Path):
    out = tmp_path / "out.json"
    sink = JsonSink(out)
    for i in range(3):
        await sink.write(_frame(i))
    await sink.aclose()
    payload = json.loads(out.read_text())
    assert isinstance(payload, list)
    assert len(payload) == 3
    assert payload[0]["frame_index"] == 0
    assert payload[2]["team_b_count"] == 11


@pytest.mark.asyncio
async def test_json_sink_aclose_idempotent(tmp_path: Path):
    sink = JsonSink(tmp_path / "out.json")
    await sink.aclose()
    await sink.aclose()  # must not raise


@pytest.mark.asyncio
async def test_json_sink_write_after_close_raises(tmp_path: Path):
    sink = JsonSink(tmp_path / "out.json")
    await sink.aclose()
    with pytest.raises(RuntimeError):
        await sink.write(_frame(0))


# ---------------------------------------------------------------------------
# NdjsonSink
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ndjson_sink_one_line_per_frame(tmp_path: Path):
    out = tmp_path / "out.ndjson"
    sink = NdjsonSink(out)
    for i in range(5):
        await sink.write(_frame(i))
    await sink.aclose()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    parsed = [json.loads(line) for line in lines]
    assert [p["frame_index"] for p in parsed] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_ndjson_sink_in_memory_getvalue():
    sink = NdjsonSink()
    await sink.write(_frame(0))
    await sink.write(_frame(1))
    text = sink.getvalue()
    assert text.count("\n") == 2
    first = json.loads(text.splitlines()[0])
    assert first["frame_index"] == 0


@pytest.mark.asyncio
async def test_ndjson_stream_helper_yields_bytes():
    async def src():
        for i in range(2):
            yield _frame(i)

    chunks = []
    async for c in ndjson_stream(src()):
        chunks.append(c)
    assert all(isinstance(c, bytes) for c in chunks)
    text = b"".join(chunks).decode("utf-8")
    assert text.count("\n") == 2
