"""FastAPI dependency-injection providers.

The DI graph is intentionally shallow: routes depend on
:class:`Settings`, :class:`ModelRegistry`, and the :class:`Counters` instance.
Tests override these via :attr:`fastapi.FastAPI.dependency_overrides` to
inject fakes with no globals to reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import Depends, Request

from playercount.config import Settings
from playercount.config import get_settings as _get_settings_cached
from playercount.utils import Counters

if TYPE_CHECKING:
    from playercount.models import ModelRegistry


def get_settings() -> Settings:
    """Process-wide :class:`Settings` (cached). Override in tests if needed."""
    return _get_settings_cached()


def get_registry(request: Request) -> ModelRegistry:
    """Pull the :class:`ModelRegistry` off ``app.state`` (created in lifespan)."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None:  # pragma: no cover - lifespan should always populate this
        raise RuntimeError("ModelRegistry was not initialized in the FastAPI lifespan")
    return cast("ModelRegistry", registry)


def get_counters(request: Request) -> Counters:
    """Pull the shared :class:`Counters` instance off ``app.state``."""
    counters = getattr(request.app.state, "counters", None)
    if counters is None:  # pragma: no cover
        counters = Counters()
        request.app.state.counters = counters
    return counters


# Annotated-style aliases so route signatures stay short. These are not used
# directly by Python — they exist as type hints to make the FastAPI signatures
# crisp when read in routes.py.
SettingsDep = Depends(get_settings)
RegistryDep = Depends(get_registry)
CountersDep = Depends(get_counters)


__all__ = [
    "CountersDep",
    "RegistryDep",
    "SettingsDep",
    "get_counters",
    "get_registry",
    "get_settings",
]
