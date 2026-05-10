"""HTTP-level tests against the FastAPI app via httpx.AsyncClient."""

from __future__ import annotations

import json

import pytest

from playercount import __version__


@pytest.mark.asyncio
async def test_healthz(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["models_loaded"] is False  # fake registry says so
    assert body["ready"] is True  # fake registry is_ready=True
    assert "device" in body


@pytest.mark.asyncio
async def test_readyz_returns_503_when_not_ready(client, app):
    """If the registry hasn't warmed, /readyz returns 503 with a descriptive payload."""
    # Flip the fake registry's readiness off.
    fake = app.state.registry
    fake.is_ready = False
    fake.warm_error = "torch.cuda.OutOfMemoryError: simulated for test"
    try:
        r = await client.get("/readyz")
        assert r.status_code == 503
        detail = r.json()["detail"]
        assert detail["status"] == "not_ready"
        assert "simulated" in detail["warm_error"]
    finally:
        # Reset for other tests in the same client lifetime.
        fake.is_ready = True
        fake.warm_error = None


@pytest.mark.asyncio
async def test_readyz_ok_when_ready(client):
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["ready"] is True


@pytest.mark.asyncio
async def test_version_endpoint(client):
    r = await client.get("/version")
    assert r.status_code == 200
    assert r.json() == {"name": "playercount", "version": __version__}


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    r = await client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")


@pytest.mark.asyncio
async def test_analyze_requires_input(client):
    """Empty JSON body returns 400 (no source available)."""
    r = await client.post(
        "/analyze",
        content=b"",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400
    assert "video_uri" in r.json()["detail"] or "file" in r.json()["detail"]


@pytest.mark.asyncio
async def test_analyze_rejects_bad_uri_scheme(client):
    r = await client.post("/analyze", json={"video_uri": "ftp://x/y.mp4"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any("video_uri" in str(d) for d in detail)


@pytest.mark.asyncio
async def test_analyze_blocks_when_registry_not_ready(client, app):
    """If the registry isn't warmed, /analyze refuses with 503 — never tries to run."""
    fake = app.state.registry
    fake.is_ready = False
    fake.warm_error = "fake warm failure"
    try:
        r = await client.post("/analyze", json={"video_uri": "https://example.com/x.mp4"})
        assert r.status_code == 503
        assert r.json()["detail"]["status"] == "not_ready"
    finally:
        fake.is_ready = True
        fake.warm_error = None


@pytest.mark.asyncio
async def test_analyze_stream_requires_input(client):
    r = await client.post(
        "/analyze/stream",
        content=b"",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_unsupported_content_type_returns_415(client):
    r = await client.post(
        "/analyze",
        content=b"hello",
        headers={"content-type": "text/plain"},
    )
    assert r.status_code == 415


@pytest.mark.asyncio
async def test_openapi_schema_is_well_formed(client):
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    schema = json.loads(r.content)
    paths = schema["paths"]
    for path in ["/healthz", "/readyz", "/version", "/metrics", "/analyze", "/analyze/stream"]:
        assert path in paths, f"missing path: {path}"
