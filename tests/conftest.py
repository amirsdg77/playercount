"""Shared pytest fixtures.

These fixtures avoid touching the GPU / network so the unit suite stays
deterministic and fast (sub-second). Integration tests under
``tests/integration/`` use the real sample.mp4 and downloaded weights — they
are marked ``slow`` and skipped by default.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING

import pytest

from playercount.config import Settings
from playercount.schemas import BBox, Detection, Track

if TYPE_CHECKING:
    from fastapi import FastAPI


# ---------------------------------------------------------------------------
# Generic builders
# ---------------------------------------------------------------------------


@pytest.fixture
def settings_factory(tmp_path) -> object:
    """Return a builder that produces fresh :class:`Settings` instances per test."""

    def _make(**overrides: object) -> Settings:
        defaults = dict(
            yolo_weights=tmp_path / "weights.pt",
            weights_dir=tmp_path,
            teams_clusterer_path=tmp_path / "teams.joblib",
            log_json=False,
            log_level="DEBUG",
        )
        defaults.update(overrides)
        return Settings(**defaults)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def settings(settings_factory) -> Settings:
    return settings_factory()


# ---------------------------------------------------------------------------
# Detection / track helpers
# ---------------------------------------------------------------------------


def make_detection(
    *,
    cls: int = 0,
    score: float = 0.9,
    x1: float = 100,
    y1: float = 100,
    x2: float = 150,
    y2: float = 200,
) -> Detection:
    names = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
    return Detection(
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        score=score,
        class_id=cls,  # type: ignore[arg-type]
        class_name=names[cls],
    )


def make_track(track_id: int, *, cls: int = 0, team_id: int | None = None) -> Track:
    return Track(track_id=track_id, detection=make_detection(cls=cls), team_id=team_id)


@pytest.fixture
def make_detection_factory():
    return make_detection


@pytest.fixture
def make_track_factory():
    return make_track


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------


@pytest.fixture
def app(settings) -> Iterator[FastAPI]:
    """Build a FastAPI app with overridden settings and a fake registry."""
    from playercount.api.dependencies import get_registry, get_settings
    from playercount.api.main import create_app

    fastapi_app = create_app(settings)

    class _FakeRegistry:
        """In-memory fake mirroring the surface ModelRegistry exposes to routes."""

        # Tests can override these by monkeypatching the instance.
        is_ready: bool = True
        warm_error: str | None = None

        def device(self) -> str:
            return "cpu"

        @property
        def models_loaded(self) -> bool:
            return False

        def warm(self) -> None:
            return None

    fake = _FakeRegistry()
    fastapi_app.state.registry = fake
    fastapi_app.dependency_overrides[get_registry] = lambda: fake
    fastapi_app.dependency_overrides[get_settings] = lambda: settings

    yield fastapi_app

    fastapi_app.dependency_overrides.clear()


@pytest.fixture
async def client(app) -> AsyncIterator[object]:
    """An ``httpx.AsyncClient`` bound to the FastAPI app via ASGITransport."""
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
