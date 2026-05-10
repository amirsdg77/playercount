"""Pydantic v2 DTOs and HTTP API models for playercount.

Every value that crosses a module boundary, an async queue, or an HTTP boundary
goes through one of the models defined here. Models are ``frozen=True`` wherever
mutation would be a bug — the only mutable model is :class:`Track`, whose
``team_id`` is filled in by the team-classification stage.

The module has zero runtime dependencies on torch/cv2 — keeping it cheap to
import means tests can exercise the data-flow layer without any ML stack.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

# ---------------------------------------------------------------------------
# Class taxonomy (matches the Roboflow football-players-detection schema)
# ---------------------------------------------------------------------------

ClassId = Literal[0, 1, 2, 3]
"""Detector class id: 0=player, 1=goalkeeper, 2=referee, 3=ball."""

CLASS_NAMES: dict[int, str] = {
    0: "player",
    1: "goalkeeper",
    2: "referee",
    3: "ball",
}

TeamId = Literal[0, 1]
"""Two teams; 0 == "team A", 1 == "team B" (assignment is arbitrary per video)."""


# ---------------------------------------------------------------------------
# Geometry / detections
# ---------------------------------------------------------------------------


class BBox(BaseModel):
    """Pixel-space axis-aligned bounding box (x1,y1) top-left, (x2,y2) bottom-right."""

    model_config = ConfigDict(frozen=True)

    x1: float = Field(ge=0.0)
    y1: float = Field(ge=0.0)
    x2: float
    y2: float

    @field_validator("x2")
    @classmethod
    def _x2_gt_x1(cls, v: float, info: ValidationInfo) -> float:
        x1 = info.data.get("x1")
        if x1 is not None and v <= x1:
            raise ValueError("x2 must be > x1")
        return v

    @field_validator("y2")
    @classmethod
    def _y2_gt_y1(cls, v: float, info: ValidationInfo) -> float:
        y1 = info.data.get("y1")
        if y1 is not None and v <= y1:
            raise ValueError("y2 must be > y1")
        return v

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class Detection(BaseModel):
    """One detection from the object detector (pre-tracking)."""

    model_config = ConfigDict(frozen=True)

    bbox: BBox
    score: float = Field(ge=0.0, le=1.0)
    class_id: ClassId
    class_name: str

    @field_validator("class_name")
    @classmethod
    def _name_matches_id(cls, v: str, info: ValidationInfo) -> str:
        cid = info.data.get("class_id")
        if cid is not None and CLASS_NAMES.get(cid) != v:
            raise ValueError(f"class_name {v!r} does not match class_id={cid}")
        return v


class Track(BaseModel):
    """A detection annotated with a stable per-stream tracking id and (optional) team.

    ``team_id`` is filled in by the team-classification stage; it stays ``None``
    for referees, balls, and players whose track has not yet accumulated enough
    votes for the majority-vote stabilizer to commit.
    """

    model_config = ConfigDict(frozen=False)  # team_id is filled in after construction

    track_id: int = Field(ge=0)
    detection: Detection
    team_id: int | None = Field(default=None, ge=0, le=1)


class TeamAssignment(BaseModel):
    """A per-frame team prediction for a single track id."""

    model_config = ConfigDict(frozen=True)

    track_id: int = Field(ge=0)
    team_id: TeamId
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Per-frame and per-video aggregates
# ---------------------------------------------------------------------------


class FrameResult(BaseModel):
    """The aggregated count for one processed frame; the unit of streaming output."""

    model_config = ConfigDict(frozen=True)

    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0.0)
    team_a_count: int = Field(ge=0)
    team_b_count: int = Field(ge=0)
    referee_count: int = Field(ge=0)
    goalkeeper_a_count: int = Field(ge=0)
    goalkeeper_b_count: int = Field(ge=0)


# ---------------------------------------------------------------------------
# HTTP API surface
# ---------------------------------------------------------------------------


# gs:// was advertised in the schema but never implemented; dropped to keep
# the schema's promises honest. Add back when GCS fetcher lands in v3.
_SUPPORTED_SCHEMES = frozenset({"http", "https", "file"})


class AnalyzeRequest(BaseModel):
    """Request body for the synchronous ``POST /analyze`` endpoint.

    Either ``video_uri`` is provided (server fetches), or the video is supplied
    as a ``multipart/form-data`` file part — the route handler picks one.
    """

    model_config = ConfigDict(extra="forbid")

    video_uri: AnyUrl | None = None
    sampling_stride: Annotated[int, Field(ge=1)] = 1
    return_mode: Literal["summary", "per_second", "per_frame", "stream"] = "summary"

    @field_validator("video_uri")
    @classmethod
    def _supported_scheme(cls, v: AnyUrl | None) -> AnyUrl | None:
        if v is None:
            return v
        if v.scheme not in _SUPPORTED_SCHEMES:
            raise ValueError(
                f"video_uri must use one of {sorted(_SUPPORTED_SCHEMES)}, got {v.scheme!r}"
            )
        return v


class FrameCount(BaseModel):
    """Compact per-frame count returned in the API response."""

    model_config = ConfigDict(frozen=True)

    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0.0)
    team_a: int = Field(ge=0)
    team_b: int = Field(ge=0)
    referees: int = Field(ge=0)


class VideoCounts(BaseModel):
    """Response model for ``POST /analyze`` (non-streaming modes)."""

    model_config = ConfigDict(frozen=True)

    fps: float = Field(gt=0.0)
    duration_s: float = Field(gt=0.0)
    frames_processed: int = Field(ge=0)
    counts: list[FrameCount]
    summary: dict[str, float] = Field(
        description="Aggregate stats keyed by name, e.g. team_a_mean, team_b_max, ratio.",
    )


class JobStatus(BaseModel):
    """For an async-jobs API extension (not wired in the PoC, but typed for v2)."""

    model_config = ConfigDict(frozen=True)

    job_id: UUID
    state: Literal["queued", "running", "done", "failed"]
    progress: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """``GET /healthz`` payload.

    ``status`` and ``models_loaded`` describe liveness (the process is up and
    has the model attributes set). ``ready`` describes readiness — true only
    after a successful ``ModelRegistry.warm()``. The ``/readyz`` endpoint
    returns 503 if ``ready=False``; ``/healthz`` does not, so a
    container-orchestrator can keep the pod alive while a slow model load
    finishes.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    version: str
    device: str
    models_loaded: bool
    ready: bool = False
    warm_error: str | None = None


class VersionResponse(BaseModel):
    """``GET /version`` payload."""

    model_config = ConfigDict(frozen=True)

    name: Literal["playercount"] = "playercount"
    version: str


__all__ = [
    "CLASS_NAMES",
    "AnalyzeRequest",
    "BBox",
    "ClassId",
    "Detection",
    "FrameCount",
    "FrameResult",
    "HealthResponse",
    "JobStatus",
    "TeamAssignment",
    "TeamId",
    "Track",
    "VersionResponse",
    "VideoCounts",
]
