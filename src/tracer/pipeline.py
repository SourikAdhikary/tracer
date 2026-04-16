"""Pipeline orchestrator — coordinates Scout -> Auditor -> QoE -> Report.

Manages the sequential loading of models and the flow of data
through the two-stage detection pipeline.
"""

import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from tracer.auditor import Auditor
from tracer.config import Config
from tracer.crop import crop_all_detections
from tracer.models.mlx_backend import unload_all
from tracer.qoe import QoEScorer
from tracer.report import build_audit_report, save_json_report, save_markdown_report
from tracer.scout import Scout
from tracer.video import extract_frames, frame_timestamp, resolve_video

console = Console()


def run_pipeline(
    video_path: str | Path,
    brands: list[str],
    config: Config | None = None,
) -> dict:
    """Run the full Tracer audit pipeline.

    Args:
        video_path: Path to the video file.
        brands: List of brand names to detect.
        config: Optional Config override.

    Returns:
        Dict with paths to generated reports and stats.
    """
    if config is None:
        config = Config()

    video_path = str(video_path)

    # Resolve YouTube URLs to local files
    video_path = str(resolve_video(video_path, config.paths.output_dir))

    config.brands = brands
    config.ensure_dirs()

    console.rule("[bold green]Tracer v4 — Sponsorship Audit Pipeline")
    console.print(f"Video: {video_path}")
    console.print(f"Brands: {', '.join(brands)}")
    console.print()

    # ============================================================
    # PHASE 1: Frame Extraction
    # ============================================================
    console.rule("[bold cyan]Phase 1: Frame Extraction")
    t0 = time.time()

    frames, duration = extract_frames(
        video_path,
        fps=config.pipeline.extraction_fps,
        frame_size=config.pipeline.frame_size,
    )

    # Build timestamp map
    timestamps = {
        i: frame_timestamp(i, config.pipeline.extraction_fps)
        for i in range(len(frames))
    }

    console.print(f"Extracted {len(frames)} frames from {duration:.0f}s video ({time.time()-t0:.1f}s)")
    console.print()

    # ============================================================
    # PHASE 2: Scout (E4B frame triage)
    # ============================================================
    console.rule("[bold cyan]Phase 2: Scout (E4B Frame Triage)")
    t0 = time.time()

    scout = Scout(config)
    scout.load()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning frames...", total=len(frames))

        def scout_progress(current, total):
            progress.update(task, completed=current)

        flagged_indices = scout.scan(frames, progress_callback=scout_progress)

    console.print(f"Scout complete: {len(flagged_indices)}/{len(frames)} frames flagged ({time.time()-t0:.1f}s)")

    # Free Scout memory before loading Auditor
    scout.unload()
    unload_all()
    console.print()

    if not flagged_indices:
        console.print("[yellow]No frames flagged by Scout. No branding detected.")
        return {"status": "no_detections", "frames_scanned": len(frames)}

    # ============================================================
    # PHASE 3: Auditor (26B-A4B logo detection)
    # ============================================================
    console.rule("[bold cyan]Phase 3: Auditor (26B-A4B Logo Detection)")
    t0 = time.time()

    auditor = Auditor(config)
    auditor.load()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Auditing flagged frames...", total=len(flagged_indices))

        def auditor_progress(current, total):
            progress.update(task, completed=current)

        detections_by_frame = auditor.audit_frames(
            frames, flagged_indices, progress_callback=auditor_progress
        )

    console.print(f"Auditor complete: {len(detections_by_frame)} frames with detections ({time.time()-t0:.1f}s)")

    # ============================================================
    # PHASE 4: QoE Scoring
    # ============================================================
    console.rule("[bold cyan]Phase 4: QoE Scoring")
    t0 = time.time()

    scorer = QoEScorer(config)

    total_detections = 0
    for frame_idx, detections in detections_by_frame.items():
        for det in detections:
            scorer.score_detection(frames[frame_idx], det)
            total_detections += 1

    console.print(f"QoE scored: {total_detections} detections ({time.time()-t0:.1f}s)")

    # ============================================================
    # PHASE 5: Crop (PoE Gallery)
    # ============================================================
    console.rule("[bold cyan]Phase 5: Proof of Exposure Crops")
    t0 = time.time()

    detections_by_frame = crop_all_detections(
        frames,
        detections_by_frame,
        config.paths.crops_dir,
        frame_timestamps=timestamps,
    )

    console.print(f"Crops saved to {config.paths.crops_dir} ({time.time()-t0:.1f}s)")

    # ============================================================
    # PHASE 6: Reporting
    # ============================================================
    console.rule("[bold cyan]Phase 6: Report Generation")
    t0 = time.time()

    # Build report
    report = build_audit_report(
        config=config,
        video_path=video_path,
        duration=duration,
        frames_extracted=len(frames),
        flagged_indices=flagged_indices,
        detections_by_frame=detections_by_frame,
        frame_timestamps=timestamps,
    )

    # Save JSON
    json_path = config.paths.output_dir / "audit_report.json"
    save_json_report(report, json_path)

    # Save Markdown
    md_path = config.paths.output_dir / "audit_report.md"
    save_markdown_report(report, md_path)

    # Free Auditor memory
    auditor.unload()
    unload_all()

    console.print(f"Reports saved ({time.time()-t0:.1f}s)")
    console.print(f"  JSON: {json_path}")
    console.print(f"  Markdown: {md_path}")
    console.print()

    # ============================================================
    # Summary
    # ============================================================
    console.rule("[bold green]Audit Complete")
    console.print(f"Total detections: {total_detections}")
    console.print(f"Frames scanned: {len(frames)}")
    console.print(f"Frames flagged: {len(flagged_indices)}")
    console.print(f"Frames with detections: {len(detections_by_frame)}")
    console.print()

    return {
        "status": "complete",
        "json_report": str(json_path),
        "markdown_report": str(md_path),
        "crops_dir": str(config.paths.crops_dir),
        "total_detections": total_detections,
        "frames_scanned": len(frames),
        "frames_flagged": len(flagged_indices),
    }
