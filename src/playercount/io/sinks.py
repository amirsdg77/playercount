"""Result sinks: JSON, NDJSON, and the annotated-MP4 eyeball-test sink.

All sinks share the :class:`ResultSink` protocol so the pipeline does not
have to know whether output is going to a file, an HTTP stream, or stdout.
Per-frame payloads are small (~1 KB) so :class:`JsonSink` and
:class:`NdjsonSink` perform their writes inline; :class:`AnnotatedVideoSink`
dispatches each frame write to a :class:`ThreadPoolExecutor` because
OpenCV's :class:`cv2.VideoWriter` blocks on disk.

Each sink is owned by exactly one :func:`sink_stage` task, so no internal
locking is required.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import IO, Protocol, runtime_checkable

import cv2  # type: ignore[import-not-found]
import numpy as np

from playercount.constants import (
    CLS_BALL,
    CLS_GOALKEEPER,
    CLS_REFEREE,
    COLOR_GK_OUTLINE,
    COLOR_HUD_BG,
    COLOR_HUD_TEXT,
    COLOR_REFEREE,
    COLOR_TEAM_A,
    COLOR_TEAM_B,
    COLOR_UNASSIGNED,
    TEAM_A,
    TEAM_B,
)
from playercount.schemas import FrameResult, Track

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ResultSink(Protocol):
    """Where :class:`FrameResult` objects go."""

    async def write(self, frame: FrameResult) -> None:
        """Persist one frame result. Must be safe to call repeatedly."""
        ...

    async def aclose(self) -> None:
        """Flush and release any resources. Idempotent."""
        ...


# ---------------------------------------------------------------------------
# JSON (one big document)
# ---------------------------------------------------------------------------


class JsonSink:
    """Buffer all frame results in memory, write a single JSON document on close.

    Suitable for short clips and unit tests; do **not** use for long videos —
    memory grows linearly with frame count. Long videos should use
    :class:`NdjsonSink` or stream over HTTP.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._buf: list[FrameResult] = []
        self._closed = False

    async def write(self, frame: FrameResult) -> None:
        if self._closed:
            raise RuntimeError("write() after aclose()")
        self._buf.append(frame)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Build the array on bytes-side to avoid an intermediate Python list →
        # dict → str conversion that ``json.dumps([model_dump(), ...])`` does.
        # Use model_dump_json per-frame; concatenate with brackets + commas.
        parts = [b"[\n"]
        for i, f in enumerate(self._buf):
            if i:
                parts.append(b",\n")
            parts.append(b"  ")
            parts.append(f.model_dump_json().encode("utf-8"))
        parts.append(b"\n]\n")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(b"".join(parts))


# ---------------------------------------------------------------------------
# NDJSON (one frame per line)
# ---------------------------------------------------------------------------


class NdjsonSink:
    """Newline-delimited JSON sink — the canonical streaming output format.

    Each frame becomes one line. Files written this way can be:

    * ``tail -f``-ed live during a long run.
    * Loaded into pandas via ``pd.read_json(path, lines=True)``.
    * Streamed over HTTP via ``application/x-ndjson``.

    Pass a :class:`pathlib.Path`, an open binary file-like object, or
    ``None`` to keep an in-memory buffer (handy for HTTP streaming).

    Owned by exactly one task — no lock needed.
    """

    def __init__(
        self,
        path_or_buffer: str | os.PathLike[str] | IO[bytes] | None = None,
    ) -> None:
        self._owns_handle = False
        self._buf: IO[bytes]
        if path_or_buffer is None:
            self._buf = BytesIO()
        elif hasattr(path_or_buffer, "write"):
            self._buf = path_or_buffer  # type: ignore[assignment]
        else:
            target = Path(path_or_buffer)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Binary mode is the canonical NDJSON encoding (UTF-8 bytes).
            # buffering=0 would defeat OS-level write batching; the default
            # is fine because we explicitly flush on aclose.
            self._buf = target.open("wb")
            self._owns_handle = True
        self._closed = False

    async def write(self, frame: FrameResult) -> None:
        if self._closed:
            raise RuntimeError("write() after aclose()")
        # model_dump_json skips the intermediate dict allocation that
        # json.dumps(model.model_dump()) would do — measurably cheaper in the
        # hot path. Write a single flat bytes object so the OS buffers it
        # without re-encoding.
        self._buf.write(frame.model_dump_json().encode("utf-8"))
        self._buf.write(b"\n")
        # Flush every line so external observers (`tail -f`) and progress
        # monitors see results in real time. The per-frame payload is small
        # (~150 bytes) so flush cost is dominated by the OS write barrier;
        # at our throughput (≤ a few hundred fps) this is invisible.
        self._buf.flush()

    def getvalue(self) -> str:
        """If backed by a BytesIO, return the accumulated NDJSON text."""
        if not isinstance(self._buf, BytesIO):
            raise RuntimeError("getvalue() is only valid for in-memory NdjsonSink")
        return self._buf.getvalue().decode("utf-8")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if hasattr(self._buf, "flush"):
            self._buf.flush()
        if self._owns_handle:
            self._buf.close()


# ---------------------------------------------------------------------------
# Annotated video (eyeball test)
# ---------------------------------------------------------------------------


class AnnotatedVideoSink:
    """Optional sink that draws boxes + HUD on each frame and writes an MP4.

    The runner forwards ``(FrameResult, frame_bgr, tracks)`` triples to
    :meth:`write_with_frame`; the sink draws team-coloured boxes, a GK
    outline, and a top-left HUD with OpenCV, then writes the frame off-thread.
    Used by the CLI; the HTTP API does not produce annotated video.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fps: float,
        size: tuple[int, int],
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._path = Path(path)
        self._fps = fps
        self._size = size  # (width, height)
        self._executor = executor
        self._writer: object | None = None  # cv2.VideoWriter
        self._closed = False

    async def write(self, frame: FrameResult) -> None:
        raise RuntimeError(
            "AnnotatedVideoSink requires the original frame: "
            "use write_with_frame(...) from the pipeline."
        )

    async def write_with_frame(
        self,
        frame_result: FrameResult,
        frame_bgr: np.ndarray,
        tracks: list[Track],
    ) -> None:
        """Annotate ``frame_bgr``, write to MP4 off-thread."""
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            self._writer = cv2.VideoWriter(
                str(self._path), fourcc, self._fps, self._size
            )
            if not self._writer.isOpened():  # type: ignore[attr-defined]
                raise RuntimeError(
                    f"cv2.VideoWriter failed to open {self._path} "
                    f"(fps={self._fps}, size={self._size})"
                )

        annotated = self._annotate(frame_bgr, frame_result, tracks)

        if self._executor is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor, self._writer.write, annotated  # type: ignore[attr-defined]
            )
        else:
            self._writer.write(annotated)  # type: ignore[attr-defined]

    @staticmethod
    def _annotate(
        frame_bgr: np.ndarray,
        result: FrameResult,
        tracks: list[Track],
    ) -> np.ndarray:
        """Draw team-coloured boxes + HUD onto a copy of the frame.

        Never mutates the caller's ndarray (the same buffer may still be
        referenced upstream during cancellation cleanup).
        """
        out = frame_bgr.copy()

        for tr in tracks:
            cls_id = tr.detection.class_id
            if cls_id == CLS_BALL:
                continue
            x1 = int(round(tr.detection.bbox.x1))
            y1 = int(round(tr.detection.bbox.y1))
            x2 = int(round(tr.detection.bbox.x2))
            y2 = int(round(tr.detection.bbox.y2))
            if cls_id == CLS_REFEREE:
                colour = COLOR_REFEREE
            elif tr.team_id == TEAM_A:
                colour = COLOR_TEAM_A
            elif tr.team_id == TEAM_B:
                colour = COLOR_TEAM_B
            else:
                colour = COLOR_UNASSIGNED
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            if cls_id == CLS_GOALKEEPER:
                cv2.rectangle(
                    out, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), COLOR_GK_OUTLINE, 1
                )
            cv2.putText(
                out,
                f"#{tr.track_id}",
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
                cv2.LINE_AA,
            )

        hud = (
            f"Team A: {result.team_a_count}   "
            f"Team B: {result.team_b_count}   "
            f"Refs: {result.referee_count}   "
            f"Frame: {result.frame_index}   "
            f"t={result.timestamp_s:.2f}s"
        )
        # Background strip behind the HUD for legibility on light pitches.
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), COLOR_HUD_BG, -1)
        cv2.putText(
            out, hud, (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD_TEXT, 1, cv2.LINE_AA,
        )
        return out

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer is None:
            return
        writer = self._writer
        self._writer = None
        if self._executor is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, writer.release)  # type: ignore[attr-defined]
        else:
            writer.release()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helper — lift any sync iterable to an NDJSON streaming iterator for FastAPI
# ---------------------------------------------------------------------------


async def ndjson_stream(results: AsyncIterator[FrameResult]) -> AsyncIterator[bytes]:
    """Adapter for :class:`fastapi.responses.StreamingResponse`.

    Encodes each :class:`FrameResult` as a UTF-8 NDJSON line and yields it as
    ``bytes``. Useful directly from the route handler::

        return StreamingResponse(
            ndjson_stream(runner.stream()),
            media_type="application/x-ndjson",
        )
    """
    async for frame in results:
        yield frame.model_dump_json().encode("utf-8") + b"\n"


__all__ = [
    "AnnotatedVideoSink",
    "JsonSink",
    "NdjsonSink",
    "ResultSink",
    "ndjson_stream",
]
