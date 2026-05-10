"""Typer CLI: ``playercount run|serve|calibrate|version``.

The CLI exists for two use cases:

1. **Offline batch processing.** ``playercount run video.mp4 --out out.ndjson``
   produces NDJSON without spinning up a server, useful for one-off analyses
   and the eyeball test (``--annotated out.mp4``).
2. **Service launch.** ``playercount serve`` is a thin wrapper over uvicorn so
   the same entry point handles both modes.

The ``calibrate`` subcommand fits the team clusterer on a representative
sample of player crops from the supplied video and persists it to
``models/teams.joblib``. ``run`` will *auto-calibrate* if no clusterer file
exists, so the calibrate step is optional for one-off analysis.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import typer

from playercount import __version__
from playercount.config import Settings, load_settings
from playercount.io.sinks import AnnotatedVideoSink, NdjsonSink
from playercount.io.video_source import PyAvVideoSource
from playercount.models import ModelRegistry
from playercount.pipeline.runner import PipelineComponents, PipelineRunner
from playercount.pipeline.stages import _crop_bgr
from playercount.team_id import EmbeddingTeamClassifier
from playercount.tracking import ByteTrackTracker
from playercount.utils import configure_logging, get_logger

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
    help="Per-team on-screen player counts from broadcast soccer video.",
)

logger = get_logger("playercount.cli")


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@app.command("version")
def version_cmd() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


# All run artifacts (NDJSON, annotated MP4, plots, logs) land here by default.
# Override with --out / --annotated explicitly when you need a different path.
RESULTS_DIR = Path("results")


@app.command("run")
def run_cmd(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output NDJSON path. Default: results/out.ndjson"),
    ] = RESULTS_DIR / "out.ndjson",
    annotated: Annotated[
        Path | None,
        typer.Option(
            "--annotated",
            help="Optional annotated MP4 path (eyeball test). Default: results/annotated.mp4",
        ),
    ] = RESULTS_DIR / "annotated.mp4",
    no_annotated: Annotated[
        bool,
        typer.Option(
            "--no-annotated",
            help="Skip writing the annotated MP4 (NDJSON only).",
        ),
    ] = False,
    stride: Annotated[int, typer.Option("--stride", min=1)] = 1,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="YAML configuration overlay."),
    ] = None,
    calibrate_frames: Annotated[
        int,
        typer.Option(
            "--calibrate-frames",
            min=10,
            help="If the clusterer isn't on disk, auto-calibrate on this many crops.",
        ),
    ] = 150,
) -> None:
    """Process a video offline and write NDJSON (and optionally annotated MP4)."""
    if no_annotated:
        annotated = None
    settings = load_settings(config)
    configure_logging(level=settings.log_level, json=settings.log_json)
    # CPU-only inference is slow on YOLOv8m at 1080p; smaller batches keep
    # memory bounded and let progress show up in the NDJSON sooner. We
    # detect GPU availability here rather than after warmup so the batch
    # size is right from the first forward pass.
    try:
        import torch  # type: ignore[import-not-found]

        on_cpu = settings.device == "cpu" or (
            settings.device == "auto" and not torch.cuda.is_available()
        )
    except ImportError:
        on_cpu = True
    yolo_batch = 1 if on_cpu else settings.yolo_batch_size
    settings = settings.model_copy(
        update={"sampling_stride": stride, "yolo_batch_size": yolo_batch}
    )

    logger.info(
        "cli.run.start",
        video=str(video),
        out=str(out),
        annotated=str(annotated) if annotated else None,
        stride=stride,
    )
    asyncio.run(
        _run_impl(
            video=video,
            out=out,
            annotated=annotated,
            settings=settings,
            calibrate_frames=calibrate_frames,
        )
    )
    logger.info("cli.run.done", out=str(out))


async def _run_impl(
    *,
    video: Path,
    out: Path,
    annotated: Path | None,
    settings: Settings,
    calibrate_frames: int,
) -> None:
    """Build the pipeline components and drive :class:`PipelineRunner` to completion."""
    registry = ModelRegistry(settings)
    t0 = time.perf_counter()
    typer.echo("[run] warming model registry…")
    registry.warm()
    typer.echo(f"[run] models loaded on {registry.device()} in {time.perf_counter() - t0:.1f}s")

    classifier = registry.team_classifier()
    if isinstance(classifier, EmbeddingTeamClassifier) and classifier.needs_calibration():
        typer.echo(f"[run] calibrating team clusterer on first ~{calibrate_frames} player crops…")
        t1 = time.perf_counter()
        await _auto_calibrate(
            video=video,
            classifier=classifier,
            registry=registry,
            settings=settings,
            target_crops=calibrate_frames,
        )
        typer.echo(f"[run] calibration done in {time.perf_counter() - t1:.1f}s")
        # Persist for future runs on the same match.
        try:
            classifier._clusterer.save(settings.teams_clusterer_path)  # type: ignore[attr-defined]
            typer.echo(f"[run] saved clusterer to {settings.teams_clusterer_path}")
        except Exception as exc:
            logger.warning("cli.run.save_clusterer_failed", error=repr(exc))

    # Open the source for the real run.
    decode_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playercount-decode-cli")
    source = PyAvVideoSource(video, executor=decode_pool)

    sink = NdjsonSink(out)
    annotator = None
    annotator_pool: ThreadPoolExecutor | None = None
    if annotated is not None:
        # cv2.VideoWriter needs (fps, width, height) at construction. Use the
        # cheap synchronous probe rather than opening a second decode worker.
        fps_for_writer, width_for_writer, height_for_writer = PyAvVideoSource.probe(video)
        # The annotator MUST have its own pool — sharing the decode pool
        # causes deadlock: when the decoder is mid-decode, an annotator
        # write() queues behind it, and the decoder is in turn blocked on
        # downstream queue capacity that can't free without the annotator
        # making progress.
        annotator_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="playercount-annot"
        )
        annotator = AnnotatedVideoSink(
            annotated,
            fps=fps_for_writer,
            size=(width_for_writer, height_for_writer),
            executor=annotator_pool,
        )

    components = PipelineComponents(
        source=source,
        detector=registry.detector(),
        tracker=ByteTrackTracker(frame_rate=source.fps if source.fps else 25.0),
        classifier=classifier,
        sink=sink,
    )

    typer.echo(f"[run] processing {video} …")
    t2 = time.perf_counter()
    async with PipelineRunner(
        settings,
        components,
        annotator=annotator,
        decode_executor=decode_pool,
    ) as runner:
        await runner.run()
    elapsed = time.perf_counter() - t2
    typer.echo(f"[run] processed in {elapsed:.1f}s")
    typer.echo(f"[run] OK: {out}{' + ' + str(annotated) if annotated else ''}")


async def _auto_calibrate(
    *,
    video: Path,
    classifier: EmbeddingTeamClassifier,
    registry: ModelRegistry,
    settings: Settings,
    target_crops: int,
) -> None:
    """Decode the video until we've collected ``target_crops`` player crops, fit, save."""
    detector = registry.detector()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playercount-cal")
    source = PyAvVideoSource(video, executor=pool)
    import numpy as np

    crops: list[np.ndarray] = []
    # Process in small batches to amortise YOLO call overhead.
    batch: list[np.ndarray] = []
    batch_size = 4
    hard_limit_frames = max(target_crops * 5, 600)

    seen = 0
    try:
        async for _idx, _ts, frame in source.frames(stride=1):
            seen += 1
            batch.append(frame)
            if len(batch) >= batch_size:
                det_lists = detector.infer(batch)
                for fr, dets in zip(batch, det_lists, strict=True):
                    for d in dets:
                        if d.class_id != 0:  # players only
                            continue
                        # Build a fake Track to reuse _crop_bgr.
                        from playercount.schemas import Track

                        tk = Track(track_id=0, detection=d)
                        c = _crop_bgr(fr, tk)
                        if c is not None and c.size > 0:
                            crops.append(c)
                            if len(crops) >= target_crops:
                                break
                    if len(crops) >= target_crops:
                        break
                batch = []
            if len(crops) >= target_crops or seen >= hard_limit_frames:
                break
    finally:
        await source.close()
        pool.shutdown(wait=False)

    if len(crops) < 2:
        raise RuntimeError(
            f"calibration failed: only collected {len(crops)} player crops from "
            f"{seen} frames (need ≥2). Lower --calibrate-frames or check the detector."
        )

    typer.echo(f"[run] collected {len(crops)} crops from {seen} frames; fitting…")
    classifier.calibrate(crops)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command("serve")
def serve_cmd(
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port")] = 8000,
    reload: Annotated[bool, typer.Option("--reload/--no-reload")] = False,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
) -> None:
    """Launch the HTTP service (thin uvicorn wrapper)."""
    import uvicorn

    uvicorn.run(
        "playercount.api.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_config=None,  # we configure logging ourselves in the lifespan
    )


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


@app.command("calibrate")
def calibrate_cmd(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    frames: Annotated[
        int,
        typer.Option("--frames", min=10, help="Number of player-only crops to sample."),
    ] = 150,
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Joblib path to write the fitted clusterer."),
    ] = Path("models/teams.joblib"),
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="YAML configuration overlay."),
    ] = None,
) -> None:
    """Fit the team clusterer on a sample of player crops from ``video``."""
    settings = load_settings(config)
    configure_logging(level=settings.log_level, json=settings.log_json)

    logger.info("cli.calibrate.start", video=str(video), frames=frames, out=str(out))
    asyncio.run(_calibrate_impl(video=video, frames=frames, out=out, settings=settings))
    logger.info("cli.calibrate.done", out=str(out))


async def _calibrate_impl(
    *, video: Path, frames: int, out: Path, settings: Settings
) -> None:
    registry = ModelRegistry(settings)
    typer.echo("[calibrate] warming registry…")
    registry.warm()
    classifier = registry.team_classifier()
    if not isinstance(classifier, EmbeddingTeamClassifier):
        raise RuntimeError(
            f"`calibrate` requires EmbeddingTeamClassifier, got {type(classifier).__name__}"
        )
    typer.echo(f"[calibrate] collecting up to {frames} player crops…")
    await _auto_calibrate(
        video=video,
        classifier=classifier,
        registry=registry,
        settings=settings,
        target_crops=frames,
    )
    classifier._clusterer.save(out)  # type: ignore[attr-defined]
    typer.echo(f"[calibrate] saved {out}")


__all__ = ["app"]
