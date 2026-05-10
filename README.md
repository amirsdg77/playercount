# playercount

> **Per-team on-screen player counts from broadcast soccer video.**

`playercount` ingests a soccer match recording and reports, for every frame,
how many players from each team are visible — plus referees and goalkeepers,
broken out per team. The output is JSON, streaming NDJSON, or an annotated
MP4 for eyeballing the result.

It does the same job whether the team kits look anything alike or not,
*without any labelled training data*: a SigLIP image-embedding model maps
player crops into a feature space where the two team kits cluster naturally,
and KMeans separates them.

```text
PyAV decode
  → YOLOv8m  (player / goalkeeper / referee / ball)
  → ByteTrack                                       (stable per-stream IDs)
  → SigLIP embedding → UMAP(3) → KMeans(k=2)        (unsupervised teams)
  → per-track sliding-window mode vote              (kills flicker)
  → per-frame distinct-ID counts per team
  → JSON / NDJSON / annotated MP4
```

---

## Contents

1. [What it does](#what-it-does)
2. [Why it's interesting](#why-its-interesting)
3. [Quickstart](#quickstart)
4. [How it works](#how-it-works)
5. [Configuration](#configuration)
6. [HTTP API](#http-api)
7. [Project layout](#project-layout)
8. [Development](#development)
9. [Evaluation & limitations](#evaluation--limitations)
10. [Roadmap](#roadmap)
11. [References](#references)

---

## What it does

Given a video file (broadcast or fixed camera), `playercount` produces, per
processed frame, a structured count of who's on screen:

```json
{
  "frame_index": 137,
  "timestamp_s": 13.7,
  "team_a_count": 9,
  "team_b_count": 10,
  "referee_count": 2,
  "goalkeeper_a_count": 1,
  "goalkeeper_b_count": 1
}
```

Three output flavours:

| Mode             | Surface                               | Best for                                                      |
| ---------------- | ------------------------------------- | ------------------------------------------------------------- |
| One JSON doc     | CLI `--out results/out.json`          | Short clips and batch postprocessing                          |
| NDJSON stream    | CLI default / streaming HTTP endpoint | Long matches; tail-able live; `pd.read_json(lines=True)`-able |
| Annotated MP4    | CLI default / `--annotated <path>`    | Eyeball test of the model at work                             |

---

## Why it's interesting

* **No labels needed.** Team identity is unsupervised: SigLIP image embeddings
  → UMAP → KMeans(k=2). Calibrate once per match (~150 player crops); after
  that, every player is assigned in a single batched forward pass per frame.
* **Stable counts.** A per-track sliding-window majority vote stabilises the
  team label, so a single-frame mis-classification doesn't show up as a count
  flicker.
* **Real async pipeline.** Decode, detect+track, embed+assign, aggregate, and
  sink each run as their own coroutine, connected by bounded `asyncio.Queue`s.
  The bound *is* the backpressure: when the GPU is the bottleneck, decoding
  stalls within milliseconds and RAM stays bounded.
* **Strict typing end-to-end.** Every cross-module value is a `pydantic` v2
  DTO with field validators; `mypy --strict` is part of CI.

---

## Quickstart

### Install

```bash
# CPU-only host (smallest install footprint):
pip install -e ".[cpu,dev]" --extra-index-url https://download.pytorch.org/whl/cpu

# GPU host (CUDA 12.1):
pip install -e ".[gpu,dev]" --extra-index-url https://download.pytorch.org/whl/cu121
```

### Pull weights

```bash
make download-weights
```

The script tries a public HuggingFace mirror first, falls back to Roboflow
(if `ROBOFLOW_API_KEY` is set), then to a direct URL
(`PLAYERCOUNT_YOLO_WEIGHTS_URL`), and finally — with `--allow-coco-fallback`
— to vanilla COCO `yolov8m.pt` so the pipeline runs end-to-end even without
soccer-tuned weights.

### CLI

```bash
# Run end-to-end. Auto-calibrates the team clusterer on the first ~150
# player crops if no models/teams.joblib exists yet. Writes
# results/out.ndjson and results/annotated.mp4 by default.
playercount run data/sample.mp4

# NDJSON only (skip the MP4):
playercount run data/sample.mp4 --no-annotated

# Subsample frames to trade accuracy for throughput:
playercount run data/sample.mp4 --stride 2

# Pre-fit and persist the clusterer separately if you prefer:
playercount calibrate data/sample.mp4 --frames 150 --out models/teams.joblib
```

### HTTP API

```bash
# Dev: hot-reload on file changes.
make serve

# Production: uvicorn factory + the lifespan warms the model registry.
python -m uvicorn playercount.api.main:create_app --factory --host 0.0.0.0 --port 8000
```

```bash
# Liveness / readiness
curl -s localhost:8000/healthz | jq .
curl -s localhost:8000/readyz  | jq .

# One-shot analysis (returns the full VideoCounts)
curl -s -X POST -F file=@data/sample.mp4 localhost:8000/analyze | jq '.summary'

# Streaming (NDJSON, one frame per line; great with jq -c)
curl -N -X POST -F file=@data/sample.mp4 localhost:8000/analyze/stream | head
```

### Docker

```bash
make docker-build       # CPU image
docker compose up api   # serve on :8000

# GPU (requires the NVIDIA Container Toolkit on the host)
make docker-build-gpu
docker compose --profile gpu up api-gpu
```

---

## How it works

### Pipeline

```text
                       Q1: maxsize=8           Q2: maxsize=8
PyAV decode ─────────▶ detect+track ─────────▶ embed+assign ─────────▶ aggregate ─────────▶ sink
  (decode worker       (YOLO forward in        (SigLIP forward in       pure-Python,         async writes,
   in its own           detect pool)            embed pool)              microseconds)        ndjson/json/mp4)
   thread)
                                          Q3: maxsize=8           Q4: maxsize=16
```

* **Detection.** Ultralytics YOLOv8m + the Roboflow
  [`football-players-detection-3zvbc`](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
  weights — four classes (player / goalkeeper / referee / ball). The detector
  is wrapped behind a `Detector` Protocol so it can be swapped for YOLO11 or
  RT-DETR with no pipeline changes.
* **Tracking.** [`supervision.ByteTrack`](https://supervision.roboflow.com/)
  for per-stream stable IDs. ByteTrack over BoT-SORT because counting only
  cares about *currently active* IDs — short-occlusion ID switches are cheap
  for us, since the team classifier re-snaps the team label on the new
  track.
* **Team identification.** SigLIP image encoder → UMAP(3) → KMeans(k=2),
  fitted once per match on ~150 player-only crops. Goalkeepers are excluded
  from the fit (different kit) and snapped at predict time to the nearer
  team centroid in UMAP space. Referees are filtered by detector class and
  never assigned a team. A swappable `HsvTeamClassifier` (CPU-only,
  torso-region HSV histogram) is provided as a fallback.
* **Stabilisation.** Per-track team votes are stored in a sliding window; the
  per-frame team label is the *mode* over that window. This single change
  removes virtually all visible count flicker.
* **Counting.** Distinct active track IDs per stabilised team, per frame.

### Concurrency model

* Stages are `async def` coroutines connected by **bounded** `asyncio.Queue`
  instances. A full queue blocks the upstream stage — that's the
  backpressure mechanism that keeps memory bounded if the GPU stalls.
* Heavy work is dispatched to **dedicated** `ThreadPoolExecutor`s — one for
  detection, one for embedding, one (single-worker) for PyAV decode — so a
  backlog in one stage cannot starve the others. CUDA forwards and PyAV
  decode both release the GIL, so threading overlaps host preprocessing
  with the previous batch's GPU work without spawning extra processes.
* The pipeline lives in an `asyncio.TaskGroup` (Python 3.11+). Any stage
  failure cancels its siblings, and `aclose()` is idempotent so it's safe
  to call from `__aexit__` and from a streaming-response client disconnect.
* Every stage's `finally` block uses a sentinel-once invariant + a
  cancellation-aware `_emit_eof` that uses `put_nowait` when cancelled — so
  the TaskGroup teardown can never deadlock on a full downstream queue.
* `ModelRegistry` uses double-checked locking under a `threading.Lock` so
  the FastAPI lifespan warmer and the first incoming request can race
  without loading weights twice.

---

## Configuration

`playercount` reads configuration in three layers, with each layer winning
over the one above it:

1. **Defaults** baked into [`Settings`](src/playercount/config.py).
2. **YAML overlay** — pass `--config path/to.yaml` on the CLI, or set
   `PLAYERCOUNT_CONFIG_FILE`. See [`configs/default.yaml`](configs/default.yaml)
   for the canonical layout.
3. **Environment variables** prefixed `PLAYERCOUNT_`. See
   [`.env.example`](.env.example).

Knob ownership:

* **Env** — anything that varies per deployment: weight paths, device, log
  level/format.
* **YAML** — algorithmic tunables: thresholds, batch sizes, window sizes
  (easy to A/B and version-control).

---

## HTTP API

| Method | Path              | Returns                                | Notes                                          |
| ------ | ----------------- | -------------------------------------- | ---------------------------------------------- |
| GET    | `/healthz`        | `HealthResponse`                       | Liveness — process is up                       |
| GET    | `/readyz`         | `HealthResponse` (or 503)              | Readiness — models warmed successfully         |
| GET    | `/version`        | `VersionResponse`                      | Package version                                |
| GET    | `/metrics`        | `text/plain` (Prometheus text format)  | Per-stage timings + counters                   |
| POST   | `/analyze`        | `VideoCounts`                          | Full result; multipart upload or JSON          |
| POST   | `/analyze/stream` | `application/x-ndjson` (chunked)       | One `FrameResult` per line                     |

Schemas live in [`src/playercount/schemas.py`](src/playercount/schemas.py);
the auto-generated OpenAPI doc is at `/docs` when the service is up.

---

## Project layout

```text
playercount/
├── configs/default.yaml                  # tunable thresholds, batch sizes
├── data/                                 # input videos at runtime (gitignored)
├── models/                               # downloaded weights (gitignored)
├── results/                              # per-run outputs (gitignored)
├── scripts/
│   ├── analyse_run.py                    # NDJSON → summary + matplotlib plot
│   ├── download_weights.py               # fetch YOLO weights into models/
│   └── smoke_run.py                      # end-to-end sanity check on fakes
├── src/playercount/
│   ├── schemas.py                        # all pydantic v2 DTOs
│   ├── config.py                         # Settings (env + YAML overlay)
│   ├── cli.py                            # `playercount run|serve|calibrate`
│   ├── io/{video_source,sinks}.py        # PyAV source, JSON/NDJSON/MP4 sinks
│   ├── detection/detector.py             # Detector protocol + YOLO impl
│   ├── tracking/tracker.py               # Tracker protocol + ByteTrack impl
│   ├── team_id/                          # SigLIP embedder, UMAP+KMeans, classifier
│   ├── aggregation/                      # TrackState (mode-vote) + frame counter
│   ├── pipeline/{stages,runner}.py       # async stages + TaskGroup runner
│   ├── api/{main,routes,dependencies}.py # FastAPI app + DI
│   ├── models/registry.py                # thread-safe lazy YOLO + SigLIP holder
│   └── utils/{logging,timing}.py         # structlog config + StageTimer
├── tests/
│   ├── unit/                             # deterministic, no GPU/network
│   └── integration/                      # @pytest.mark.slow, needs sample.mp4
├── Dockerfile                            # multi-stage; ARG BASE for cpu vs gpu
├── docker-compose.yml                    # api service + optional gpu profile
├── Makefile                              # install / lint / test / serve / run
└── pyproject.toml                        # PEP 621, hatchling, ruff, mypy, pytest
```

---

## Development

```bash
make install       # editable install with the [cpu,dev] extras
make lint          # ruff
make format        # ruff --fix
make typecheck     # mypy --strict (config in pyproject.toml)
make test          # pytest -m "not slow"  (fast, no GPU/network)
make test-all      # everything, including @pytest.mark.slow integration tests
make cov           # coverage report
```

The unit suite covers the load-bearing pure-Python logic in full:

* Pydantic schema validators (BBox, Detection, AnalyzeRequest, …)
* Settings with env / YAML / default precedence
* `TrackState` sliding-window mode vote (including tie-breaks and `None`
  handling)
* `build_frame_result` distinct-ID counting per team
* JSON / NDJSON sinks
* Async pipeline runner — drains to sink, streams, cancellation, no hangs
* FastAPI routes (`/healthz`, `/readyz`, `/version`, `/metrics`, validation
  errors, OpenAPI shape)
* `StageTimer` / `Counters` and the Prometheus exporter

Tests under `tests/integration/` are marked `slow`; they exercise the full
ML stack against `data/sample.mp4` and downloaded weights.

---

## Evaluation & limitations

**Eyeball test.** Run `playercount run data/sample.mp4` and open
`results/annotated.mp4`. Player boxes are colour-coded by team; goalkeepers
are outlined in their own team's colour; referees are white. A HUD in the
top-left shows
`Team A: <n>   Team B: <n>   Refs: <n>   Frame: <i>   t=<s>`.

Counts should:

* Stay within ±1 across consecutive frames during steady play.
* Show no team-label flicker on any single track within a 5-second window.

**Quantitative analysis.** [`scripts/analyse_run.py`](scripts/analyse_run.py)
reads `results/out.ndjson` and prints per-team mean / median / max counts,
count-stability stddev over sliding windows, a histogram of frequent
`(team_a, team_b)` pairs, and an optional matplotlib time-series PNG.

**Known edge cases.**

* **Similar kit colours.** Red-vs-orange or two pale kits can collapse the
  KMeans clusters; SigLIP helps but isn't immune. The HSV fallback is worse
  here.
* **Heavy occlusion / pile-ups.** The tracker can split one player into two
  IDs; the count momentarily ticks up by one until the new ID's team-label
  vote settles.
* **Broadcast cuts.** Hard cuts to a different camera angle confuse the
  tracker; we never assume identity across camera changes. The team-label
  re-snap means the count recovers within `track_window` frames.
* **First N frames.** Counts are conservative until each track has
  accumulated enough votes for the majority to be unambiguous.

---

## Roadmap

* **Jersey-number OCR** for full player identification (SoccerNet GSR
  territory). Lets us go from "team A has 10 on screen" to "team A is
  missing #7" — much more useful for scouting and broadcast graphics.
* **Pitch homography** (PnLCalib / No-Bells-Just-Whistles) for a top-down
  minimap and "on-pitch only" filtering of bench players.
* **ONNX / TensorRT export** of YOLO and SigLIP for 2–5× inference speedups
  on the same hardware.
* **Multi-stream serving** by running multiple replicas behind a load
  balancer — one CUDA context per process, no shared state to invalidate.

---

## References

* Roboflow `football-players-detection-3zvbc` dataset & weights:
  [universe.roboflow.com](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
* Roboflow `sports` example (the recipe this project wraps in a service):
  [github.com/roboflow/sports](https://github.com/roboflow/sports)
* SigLIP — Zhai et al., *Sigmoid Loss for Language Image Pre-Training*
  (2023): [arxiv.org/abs/2303.15343](https://arxiv.org/abs/2303.15343)
* ByteTrack — Zhang et al., *ByteTrack: Multi-Object Tracking by Associating
  Every Detection Box* (ECCV 2022):
  [arxiv.org/abs/2110.06864](https://arxiv.org/abs/2110.06864)
* Ultralytics YOLOv8: [docs.ultralytics.com](https://docs.ultralytics.com/)
* `supervision`: [supervision.roboflow.com](https://supervision.roboflow.com/)
