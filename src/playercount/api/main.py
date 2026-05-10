"""FastAPI application factory + lifespan handler.

Construction is via factory function so we can plug in test settings without
patching globals. Run in production with::

    uvicorn playercount.api.main:create_app --factory --host 0.0.0.0 --port 8000

The lifespan is responsible for **all** expensive initialization: configuring
logging, building the :class:`ModelRegistry`, and warming the models. By the
time FastAPI accepts the first request, the GPU is hot and weight loads are
done — no first-hit cliff.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from playercount import __version__
from playercount.api.routes import router
from playercount.config import Settings, get_settings
from playercount.models import ModelRegistry
from playercount.utils import Counters, configure_logging, get_logger

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured :class:`fastapi.FastAPI` instance."""
    cfg = settings or get_settings()
    configure_logging(level=cfg.log_level, json=cfg.log_json)

    app = FastAPI(
        title="playercount",
        version=__version__,
        summary="Per-team on-screen player counts from broadcast soccer video.",
        lifespan=_lifespan,
    )

    # Pre-bind shared state — the lifespan will populate the heavy bits.
    app.state.settings = cfg
    app.state.counters = Counters()

    app.include_router(router)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Construct + warm the model registry; tear it down on shutdown."""
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    registry = ModelRegistry(settings)
    app.state.registry = registry

    logger.info(
        "playercount.startup",
        version=__version__,
        device=registry.device(),
    )
    # Warming is best-effort *for this iteration only*: if the weights aren't
    # on disk yet, we let the container come up in degraded mode and the
    # /readyz probe will return 503. Routes refuse with 503 until ready;
    # /healthz stays 200 so the orchestrator doesn't restart the pod while a
    # slow weight download finishes.
    #
    # We deliberately don't catch NotImplementedError specifically anymore —
    # the bodies are implemented now; if a NotImplementedError reaches us
    # it's a real bug to surface, not to mask.
    try:
        registry.warm()
        logger.info("playercount.startup.ready", device=registry.device())
    except Exception as exc:
        logger.error(
            "playercount.startup.warm_failed",
            error=repr(exc),
            hint="check models/yolov8m-soccer.pt and SigLIP cache",
        )
        # registry.warm() already recorded warm_error; readyz will report it.

    yield

    logger.info("playercount.shutdown")


__all__ = ["create_app"]
