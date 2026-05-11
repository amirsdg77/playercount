"""HTTP routes for the playercount service.

Endpoints:

* ``GET  /healthz``         — liveness probe (process is up).
* ``GET  /readyz``          — readiness probe (models warmed successfully).
* ``GET  /version``         — package version.
* ``GET  /metrics``         — Prometheus text-format counters/timings.
* ``POST /analyze``         — runs the pipeline to completion, returns JSON.
* ``POST /analyze/stream``  — runs the pipeline streaming, returns NDJSON.

The two analyze endpoints accept either a JSON body with ``video_uri`` (the
server fetches), or a ``multipart/form-data`` upload with a ``file`` part. The
pipeline plumbing is the same — only the :class:`VideoSource` differs.

A subtlety the audit caught: ``await request.body()`` consumes the request
stream, which then makes any ``UploadFile`` parameter bind to ``None`` even on
a real upload. We can't have both a manual JSON parse *and* a FastAPI-managed
``UploadFile`` extractor. Fix: peek the content type first, then either parse
JSON via ``request.body()`` or call ``request.form()`` to extract the file —
never both on the same request.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from playercount import __version__
from playercount.api.dependencies import (
    get_counters,
    get_registry,
    get_settings,
)
from playercount.io.sinks import ndjson_stream
from playercount.io.video_source import PyAvVideoSource
from playercount.pipeline.runner import PipelineComponents, PipelineRunner
from playercount.schemas import (
    AnalyzeRequest,
    FrameCount,
    FrameResult,
    HealthResponse,
    VersionResponse,
    VideoCounts,
)
from playercount.tracking import ByteTrackTracker

if TYPE_CHECKING:
    from playercount.config import Settings
    from playercount.models import ModelRegistry
    from playercount.utils import Counters


router = APIRouter()


# ---------------------------------------------------------------------------
# Liveness / introspection
# ---------------------------------------------------------------------------


@router.get("/healthz", response_model=HealthResponse, tags=["meta"])
async def healthz(
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_registry),
) -> HealthResponse:
    """Liveness — the process is up."""
    return HealthResponse(
        status="ok",
        version=__version__,
        device=registry.device(),
        models_loaded=registry.models_loaded,
        ready=registry.is_ready,
        warm_error=registry.warm_error,
    )


@router.get("/readyz", response_model=HealthResponse, tags=["meta"])
async def readyz(
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_registry),
) -> HealthResponse:
    """Readiness — the models are warmed and ready to serve.

    Returns 503 with a descriptive ``warm_error`` if warmup hasn't completed
    or failed. Container orchestrators should wire this to the readiness
    probe (vs ``/healthz`` which only confirms liveness).
    """
    if not registry.is_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "warm_error": registry.warm_error,
                "device": registry.device(),
            },
        )
    return HealthResponse(
        status="ok",
        version=__version__,
        device=registry.device(),
        models_loaded=registry.models_loaded,
        ready=True,
        warm_error=None,
    )


@router.get("/version", response_model=VersionResponse, tags=["meta"])
async def version() -> VersionResponse:
    return VersionResponse(name="playercount", version=__version__)


@router.get("/metrics", response_class=Response, tags=["meta"])
async def metrics(counters: Counters = Depends(get_counters)) -> Response:
    """Prometheus text-format metrics."""
    return Response(
        content=counters.render_prometheus(),
        media_type="text/plain; version=0.0.4",
    )


# ---------------------------------------------------------------------------
# Analyze input parsing — JSON body OR multipart upload, never both
# ---------------------------------------------------------------------------


async def _parse_analyze_input(
    request: Request,
) -> tuple[AnalyzeRequest | None, StarletteUploadFile | None]:
    """Parse the request once and return either the JSON body, or the upload.

    Why this is its own helper (audit #3): the original code called
    ``await request.body()`` *and* declared ``file: UploadFile`` on the route
    signature. ``body()`` consumes the multipart stream, so the upload always
    bound to ``None``. Fixed by branching on content-type *before* reading
    anything: JSON path uses ``body()``; multipart path uses ``form()``.
    """
    ctype = (request.headers.get("content-type") or "").lower()

    if "multipart/form-data" in ctype:
        # Read the multipart form. Starlette returns one UploadFile per file
        # part. We only honour a part literally named "file".
        form = await request.form()
        file_part = form.get("file")
        if isinstance(file_part, StarletteUploadFile):
            return None, file_part
        return None, None

    if "json" in ctype or ctype == "":
        raw = await request.body()
        if not raw:
            return None, None
        try:
            return AnalyzeRequest.model_validate_json(raw), None
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=exc.errors(include_url=False, include_context=False),
            ) from exc

    # Other content types are not supported.
    raise HTTPException(
        status_code=415,
        detail=f"Unsupported content type: {ctype!r}; use application/json or multipart/form-data",
    )


async def _materialise_source(
    parsed: AnalyzeRequest | None,
    upload: StarletteUploadFile | None,
    runner_decode_executor: Any,
) -> tuple[PyAvVideoSource, Path | None]:
    """Resolve the input to a local file and wrap it in a :class:`PyAvVideoSource`.

    Returns ``(source, tempfile_path)`` — the second element is the temp
    file we created (if any) and must clean up after the pipeline finishes.
    """
    tmp_path: Path | None = None
    if upload is not None:
        # Write the upload to a temp file. PyAV needs a path it can seek; we
        # don't try to wrap UploadFile.file as a stream because PyAV's
        # iterator-based decode is not happy with chunked uploads.
        suffix = Path(upload.filename or "upload.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = await upload.read(1 << 20)
                if not chunk:
                    break
                tmp.write(chunk)
    elif parsed is not None and parsed.video_uri is not None:
        scheme = parsed.video_uri.scheme
        uri = str(parsed.video_uri)
        if scheme == "file":
            local = Path(parsed.video_uri.path or "")
            if not local.is_file():
                raise HTTPException(404, f"file not found: {local}")
            return PyAvVideoSource(local, executor=runner_decode_executor), None
        if scheme in ("http", "https"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp_path = Path(tmp.name)
                async with (
                    httpx.AsyncClient(timeout=60) as client,
                    client.stream("GET", uri) as response,
                ):
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        tmp.write(chunk)
        else:  # pragma: no cover - validator already rejects unsupported schemes
            raise HTTPException(400, f"unsupported scheme: {scheme}")
    else:
        raise HTTPException(
            status_code=400,
            detail="provide either a JSON body with video_uri or a multipart 'file' upload",
        )

    assert tmp_path is not None
    return PyAvVideoSource(tmp_path, executor=runner_decode_executor), tmp_path


# ---------------------------------------------------------------------------
# Sinks used internally by the routes
# ---------------------------------------------------------------------------


class _NullSink:
    """Placeholder sink — never used because routes inject their own."""

    async def write(self, frame: FrameResult) -> None:
        return None

    async def aclose(self) -> None:
        return None


class _CollectingSink:
    """In-memory sink that the synchronous /analyze endpoint drains into."""

    def __init__(self) -> None:
        self.frames: list[FrameResult] = []

    async def write(self, frame: FrameResult) -> None:
        self.frames.append(frame)

    async def aclose(self) -> None:
        return None


def _build_video_counts(
    frames: list[FrameResult], fps: float, duration_s: float
) -> VideoCounts:
    """Aggregate per-frame results into the wire response."""
    counts = [
        FrameCount(
            frame_index=f.frame_index,
            timestamp_s=f.timestamp_s,
            team_a=f.team_a_count,
            team_b=f.team_b_count,
            referees=f.referee_count,
        )
        for f in frames
    ]
    if frames:
        ta = [f.team_a_count for f in frames]
        tb = [f.team_b_count for f in frames]
        rf = [f.referee_count for f in frames]
        summary = {
            "team_a_mean": sum(ta) / len(ta),
            "team_a_max": float(max(ta)),
            "team_b_mean": sum(tb) / len(tb),
            "team_b_max": float(max(tb)),
            "referee_mean": sum(rf) / len(rf),
        }
    else:
        summary = {
            "team_a_mean": 0.0, "team_a_max": 0.0,
            "team_b_mean": 0.0, "team_b_max": 0.0,
            "referee_mean": 0.0,
        }

    return VideoCounts(
        fps=fps if fps > 0 else 25.0,
        duration_s=duration_s if duration_s > 0 else 0.001,
        frames_processed=len(frames),
        counts=counts,
        summary=summary,
    )


def _ensure_ready(registry: ModelRegistry) -> None:
    """Block the analyze routes if the registry never warmed."""
    if not registry.is_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "warm_error": registry.warm_error,
            },
        )


# ---------------------------------------------------------------------------
# Analyze (synchronous — full result in one response)
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=VideoCounts, tags=["analyze"])
async def analyze(
    request: Request,
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_registry),
) -> VideoCounts:
    """Run the full pipeline and return :class:`VideoCounts`."""
    _ensure_ready(registry)
    parsed, upload = await _parse_analyze_input(request)

    # One-worker decode pool so the source can publish frames.
    decode_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playercount-decode-api")
    source, tmp_path = await _materialise_source(parsed, upload, decode_pool)

    sink = _CollectingSink()
    components = PipelineComponents(
        source=source,
        detector=registry.detector(),
        tracker=ByteTrackTracker(frame_rate=source.fps or 25.0),
        classifier=registry.team_classifier(),
        sink=sink,
    )
    try:
        async with PipelineRunner(
            settings, components, decode_executor=decode_pool
        ) as runner:
            await runner.run()
    finally:
        decode_pool.shutdown(wait=False)
        if tmp_path is not None:
            with suppress(Exception):
                tmp_path.unlink(missing_ok=True)

    duration_s = (sink.frames[-1].timestamp_s - sink.frames[0].timestamp_s) if sink.frames else 0.0
    return _build_video_counts(sink.frames, fps=source.fps, duration_s=max(duration_s, 0.001))


# ---------------------------------------------------------------------------
# Analyze (streaming — NDJSON, one frame per line, no client-side buffering)
# ---------------------------------------------------------------------------


@router.post("/analyze/stream", tags=["analyze"])
async def analyze_stream(
    request: Request,
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_registry),
) -> StreamingResponse:
    """Run the pipeline streaming, returning NDJSON over chunked transfer."""
    _ensure_ready(registry)
    parsed, upload = await _parse_analyze_input(request)

    decode_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playercount-decode-api")
    source, tmp_path = await _materialise_source(parsed, upload, decode_pool)

    components = PipelineComponents(
        source=source,
        detector=registry.detector(),
        tracker=ByteTrackTracker(frame_rate=source.fps or 25.0),
        classifier=registry.team_classifier(),
        sink=_NullSink(),
    )
    runner = PipelineRunner(settings, components, decode_executor=decode_pool)

    async def stream_with_cleanup() -> AsyncIterator[bytes]:
        try:
            async for chunk in ndjson_stream(runner.stream()):
                yield chunk
        finally:
            decode_pool.shutdown(wait=False)
            if tmp_path is not None:
                with suppress(Exception):
                    tmp_path.unlink(missing_ok=True)

    return StreamingResponse(stream_with_cleanup(), media_type="application/x-ndjson")


__all__ = ["router"]
