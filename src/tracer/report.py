"""Reporting module — JSON audit log + Markdown executive summary.

Generates structured audit reports from pipeline results.
"""

import json
from datetime import datetime
from pathlib import Path

from tracer.config import Config
from tracer.schemas import AuditReport, FrameResult, Detection


def build_audit_report(
    config: Config,
    video_path: str,
    duration: float,
    frames_extracted: int,
    flagged_indices: list[int],
    detections_by_frame: dict[int, list[dict]],
    frame_timestamps: dict[int, str],
) -> AuditReport:
    """Build a structured AuditReport from pipeline results."""

    results = []
    for frame_idx in sorted(detections_by_frame.keys()):
        dets = [
            Detection(
                brand=d.get("label", "Unknown").split("_")[0],
                box_2d=d["box_2d"],
                confidence=d.get("confidence", 0.0),
                label=d.get("label", ""),
                qoe=d.get("qoe", 0.0),
                qoe_clarity=d.get("qoe_clarity", 0.0),
                qoe_size=d.get("qoe_size", 0.0),
                qoe_occlusion=d.get("qoe_occlusion", 0.0),
                qoe_context=d.get("qoe_context", 0.0),
                context=d.get("context", ""),
                valuation_logic=d.get("valuation_logic", ""),
                crop_path=d.get("crop_path", ""),
            )
            for d in detections_by_frame[frame_idx]
        ]

        results.append(FrameResult(
            frame_index=frame_idx,
            timestamp=frame_timestamps.get(frame_idx, "00:00:00"),
            scout_has_branding=True,
            detections=dets,
        ))

    total_detections = sum(len(r.detections) for r in results)

    return AuditReport(
        match_id=datetime.now().strftime("%Y-%m-%d_%H%M"),
        video_source=video_path,
        duration_seconds=duration,
        brands_tracked=config.brands,
        scout_model=config.models.scout_model_id,
        auditor_model=config.models.auditor_model_id,
        scout_token_budget=config.models.scout_token_budget,
        auditor_token_budget=config.models.auditor_token_budget,
        frames_extracted=frames_extracted,
        frames_flagged=len(flagged_indices),
        frames_audited=len(flagged_indices),
        total_detections=total_detections,
        results=results,
    )


def save_json_report(report: AuditReport, output_path: str | Path) -> str:
    """Save audit report as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)

    return str(output_path)


def generate_markdown_report(report: AuditReport) -> str:
    """Generate a Markdown executive summary from the audit report."""

    # Aggregate per-brand stats
    brand_stats: dict[str, dict] = {}
    for result in report.results:
        for det in result.detections:
            brand = det.brand
            if brand not in brand_stats:
                brand_stats[brand] = {
                    "count": 0,
                    "total_qoe": 0.0,
                    "seconds": [],
                    "top_qoe": 0.0,
                    "top_moment": "",
                }
            stats = brand_stats[brand]
            stats["count"] += 1
            stats["total_qoe"] += det.qoe
            stats["seconds"].append(result.timestamp)
            if det.qoe > stats["top_qoe"]:
                stats["top_qoe"] = det.qoe
                stats["top_moment"] = f"{result.timestamp} — {det.context or 'N/A'}"

    # Build markdown
    lines = [
        f"# Tracer Audit Report",
        f"",
        f"**Match:** {report.match_id}",
        f"**Video:** {report.video_source}",
        f"**Duration:** {report.duration_seconds:.0f}s ({report.duration_seconds/60:.1f} min)",
        f"**Brands tracked:** {', '.join(report.brands_tracked)}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"---",
        f"",
        f"## Pipeline Stats",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Frames extracted | {report.frames_extracted} |",
        f"| Frames flagged (Scout) | {report.frames_flagged} |",
        f"| Frames audited | {report.frames_audited} |",
        f"| Total detections | {report.total_detections} |",
        f"| Scout model | `{report.scout_model}` |",
        f"| Auditor model | `{report.auditor_model}` |",
        f"",
        f"---",
        f"",
        f"## Share of Voice",
        f"",
        f"| Brand | Detections | Avg QoE | Top QoE | Top Moment |",
        f"|-------|-----------|---------|---------|------------|",
    ]

    for brand, stats in sorted(brand_stats.items(), key=lambda x: -x[1]["count"]):
        avg_qoe = stats["total_qoe"] / stats["count"] if stats["count"] > 0 else 0
        lines.append(
            f"| {brand} | {stats['count']} | {avg_qoe:.2f} | {stats['top_qoe']:.2f} | {stats['top_moment']} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## Top Detections (Highest QoE)",
        f"",
    ])

    # Collect all detections and sort by QoE
    all_dets = []
    for result in report.results:
        for det in result.detections:
            all_dets.append((result.timestamp, det))
    all_dets.sort(key=lambda x: -x[1].qoe)

    for ts, det in all_dets[:10]:
        lines.append(f"### [{ts}] {det.brand} — QoE: {det.qoe:.2f}")
        lines.append(f"")
        lines.append(f"- **Clarity:** {det.qoe_clarity:.2f} | **Size:** {det.qoe_size:.2f} | **Occlusion:** {det.qoe_occlusion:.2f} | **Context:** {det.qoe_context:.2f}")
        if det.context:
            lines.append(f"- **Context:** {det.context}")
        if det.valuation_logic:
            lines.append(f"- **Valuation:** {det.valuation_logic}")
        if det.crop_path:
            lines.append(f"- **Proof:** ![]({det.crop_path})")
        lines.append(f"")

    lines.extend([
        f"---",
        f"",
        f"## Executive Summary",
        f"",
        report.executive_summary or "_Summary not yet generated. Run Analyst phase to generate._",
        f"",
    ])

    return "\n".join(lines)


def save_markdown_report(report: AuditReport, output_path: str | Path) -> str:
    """Save Markdown report to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md = generate_markdown_report(report)
    with open(output_path, "w") as f:
        f.write(md)

    return str(output_path)
