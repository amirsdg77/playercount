"""Structured logging via ``structlog``.

Two modes:

* ``log_json=True`` (default in containers) — JSON lines on stdout, suitable
  for ingestion by Loki / Cloud Logging / ELK with no parsing.
* ``log_json=False`` — coloured key/value console output for dev.

Library code calls ``get_logger(__name__)`` and never imports the stdlib
``logging`` module directly. Configuration is applied once at process startup
via :func:`configure_logging` (the API does this in its lifespan handler;
the CLI does it in :func:`playercount.cli.app`).
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog
from structlog.types import Processor


def configure_logging(level: str = "INFO", *, json: bool = True) -> None:
    """Configure structlog + the stdlib root logger.

    Idempotent: safe to call multiple times (e.g. once from the CLI, again
    from the FastAPI lifespan when running ``playercount serve``).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: Processor
    if json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a bound structlog logger. ``name`` defaults to the caller's module."""
    return cast("Any", structlog.get_logger(name) if name else structlog.get_logger())


__all__ = ["configure_logging", "get_logger"]
