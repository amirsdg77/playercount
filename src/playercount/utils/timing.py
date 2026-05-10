"""Stage-level timing helpers and a tiny Prometheus-text exporter.

The pipeline is fast in aggregate but the ratio of decode/detect/embed/sink
time is the most useful production metric — if decode dominates, raise the
executor pool; if embed dominates, batch larger; etc. This module provides:

* :class:`StageTimer` — context manager that records elapsed seconds.
* :class:`Counters` — thread-safe dict of numeric counters/observations.
* :func:`render_prometheus` — minimal text-format exporter for ``/metrics``.

Kept dependency-free intentionally: pulling in ``prometheus_client`` would
double the wheel size and we only need one endpoint.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

__all__ = ["Counters", "StageTimer"]


@dataclass
class _Histogram:
    """Light-weight running stats: count, sum, min, max."""

    count: int = 0
    total_s: float = 0.0
    min_s: float = float("inf")
    max_s: float = 0.0

    def observe(self, value_s: float) -> None:
        self.count += 1
        self.total_s += value_s
        if value_s < self.min_s:
            self.min_s = value_s
        if value_s > self.max_s:
            self.max_s = value_s

    @property
    def mean_s(self) -> float:
        return self.total_s / self.count if self.count else 0.0


@dataclass
class Counters:
    """Process-wide counters and stage timings.

    The instance is shared between the event loop and worker threads, so all
    public methods take a lock. Contention is negligible for the volumes we
    deal with (a handful of ``observe`` calls per frame).
    """

    _counters: dict[str, int] = field(default_factory=dict)
    _histograms: dict[str, _Histogram] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + n

    def observe(self, name: str, seconds: float) -> None:
        with self._lock:
            hist = self._histograms.get(name)
            if hist is None:
                hist = _Histogram()
                self._histograms[name] = hist
            hist.observe(seconds)

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        """Context manager flavour: ``with counters.time("decode"): ...``."""
        with StageTimer(name, self):
            yield

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {
                    k: {
                        "count": h.count,
                        "total_s": h.total_s,
                        "mean_s": h.mean_s,
                        "min_s": (h.min_s if h.count else 0.0),
                        "max_s": h.max_s,
                    }
                    for k, h in self._histograms.items()
                },
            }

    def render_prometheus(self) -> str:
        """Minimal Prometheus text-format export for ``GET /metrics``."""
        lines: list[str] = []
        with self._lock:
            for name, val in sorted(self._counters.items()):
                lines.append(f"# TYPE playercount_{name} counter")
                lines.append(f"playercount_{name} {val}")
            for name, hist in sorted(self._histograms.items()):
                lines.append(f"# TYPE playercount_{name}_seconds summary")
                lines.append(f"playercount_{name}_seconds_count {hist.count}")
                lines.append(f"playercount_{name}_seconds_sum {hist.total_s}")
                lines.append(f"playercount_{name}_seconds_max {hist.max_s}")
        lines.append("")
        return "\n".join(lines)


class StageTimer:
    """Context manager that records elapsed wall-clock seconds into a :class:`Counters`."""

    __slots__ = ("_name", "_sink", "_t0")

    def __init__(self, name: str, sink: Counters) -> None:
        self._name = name
        self._sink = sink
        self._t0 = 0.0

    def __enter__(self) -> StageTimer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed = time.perf_counter() - self._t0
        self._sink.observe(self._name, elapsed)
