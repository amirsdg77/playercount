"""Analyse a ``playercount run`` output (NDJSON) and print a summary report.

Usage::

    python scripts/analyse_run.py out.ndjson
    python scripts/analyse_run.py out.ndjson --plot count_timeseries.png

What it prints:

* Per-team mean & median count.
* Min / max counts (and where in the video they occurred).
* Count stability: stddev over each 1-second sliding window of frames.
* Histogram of (team_a, team_b) pairs.
* Optional matplotlib time-series PNG.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def _read_ndjson(path: Path) -> list[dict]:
    """Load NDJSON into a list of dicts. Skips blank lines."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _moving_window_std(values: list[int], window: int) -> list[float]:
    """Population stddev over rolling windows of size ``window``."""
    out: list[float] = []
    for i in range(len(values) - window + 1):
        chunk = values[i : i + window]
        # statistics.pstdev handles short chunks; we already enforce the size.
        out.append(statistics.pstdev(chunk))
    return out


def analyse(path: Path, plot: Path | None) -> int:
    if not path.is_file():
        print(f"[analyse] not found: {path}")
        return 1
    rows = _read_ndjson(path)
    if not rows:
        print(f"[analyse] {path} is empty")
        return 1

    team_a = [int(r["team_a_count"]) for r in rows]
    team_b = [int(r["team_b_count"]) for r in rows]
    refs = [int(r["referee_count"]) for r in rows]
    gk_a = [int(r["goalkeeper_a_count"]) for r in rows]
    gk_b = [int(r["goalkeeper_b_count"]) for r in rows]
    timestamps = [float(r["timestamp_s"]) for r in rows]
    indices = [int(r["frame_index"]) for r in rows]

    n = len(rows)
    duration_s = max(timestamps) - min(timestamps) if timestamps else 0.0
    fps_est = n / duration_s if duration_s > 0 else 0.0

    print("=" * 70)
    print(f" playercount summary — {path.name}")
    print("=" * 70)
    print(f"  frames processed   : {n}")
    print(f"  duration (s)       : {duration_s:.2f}")
    print(f"  estimated fps      : {fps_est:.1f}")
    print()

    def _stats(name: str, vals: list[int]) -> None:
        if not vals:
            print(f"  {name:14s}  (empty)")
            return
        mn, mx = min(vals), max(vals)
        mean = statistics.fmean(vals)
        median = statistics.median(vals)
        # Index of first min/max occurrence (in source frame indices)
        i_min = vals.index(mn)
        i_max = vals.index(mx)
        print(
            f"  {name:14s}  mean={mean:5.2f}  median={median:5.1f}  "
            f"min={mn:2d}@frame{indices[i_min]:>4d}  max={mx:2d}@frame{indices[i_max]:>4d}"
        )

    _stats("team_a_count", team_a)
    _stats("team_b_count", team_b)
    _stats("referees", refs)
    _stats("gk_a", gk_a)
    _stats("gk_b", gk_b)
    print()

    # Count stability: stddev over 1-second sliding windows.
    if fps_est > 0:
        win = max(2, int(round(fps_est)))
        sa = _moving_window_std(team_a, win)
        sb = _moving_window_std(team_b, win)
        print(f"  count stability over {win}-frame windows (lower is better):")
        if sa:
            print(
                f"    team_a stddev  mean={statistics.fmean(sa):.3f}  max={max(sa):.3f}"
            )
        if sb:
            print(
                f"    team_b stddev  mean={statistics.fmean(sb):.3f}  max={max(sb):.3f}"
            )
    print()

    # Distribution histogram of (team_a, team_b) pairs — top 10
    pairs = list(zip(team_a, team_b, strict=True))
    counter = Counter(pairs)
    print("  most common (team_a, team_b) pairs:")
    for (a, b), c in counter.most_common(10):
        bar = "#" * min(40, c)
        print(f"    ({a:2d}, {b:2d}) × {c:4d}  {bar}")
    print()

    # Quality flags
    print("  quality flags:")
    high = sum(1 for x in team_a if x > 11) + sum(1 for x in team_b if x > 11)
    if high:
        print(f"    !  {high} frames had a team count > 11 (likely double-detection)")
    zero = sum(1 for a, b in zip(team_a, team_b, strict=True) if a == 0 and b == 0)
    if zero:
        print(f"    !  {zero} frames had zero players in *both* teams")
    if not high and not zero:
        print("    OK — no obvious anomalies")
    print()

    # Optional plot
    if plot is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print(f"[analyse] matplotlib not installed; skipping {plot}")
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(timestamps, team_a, label="Team A", color="tab:red", linewidth=1)
            ax.plot(timestamps, team_b, label="Team B", color="tab:blue", linewidth=1)
            ax.plot(timestamps, refs, label="Referees", color="black", linewidth=0.7, linestyle=":")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("on-screen count")
            ax.set_ylim(bottom=0)
            ax.legend(loc="upper right")
            ax.set_title(f"playercount — {path.name}")
            ax.grid(True, linestyle=":", alpha=0.5)
            fig.tight_layout()
            fig.savefig(plot, dpi=120)
            print(f"  plot   : {plot}")

    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ndjson", type=Path, help="Path to a playercount NDJSON output.")
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional path to write a PNG time-series plot.",
    )
    args = parser.parse_args()
    return analyse(args.ndjson, args.plot)


if __name__ == "__main__":
    raise SystemExit(main())
