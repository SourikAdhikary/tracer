"""Full pipeline: E4B Scout -> 26B-A4B Auditor -> QoE -> Report."""
import sys
from pathlib import Path
for p in list(sys.modules.keys()):
    if 'tracer' in p: del sys.modules[p]
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracer.config import Config
from tracer.video import extract_frames, resolve_video, frame_timestamp
from tracer.scout import Scout
from tracer.auditor import Auditor
from tracer.qoe import QoEScorer
from tracer.crop import crop_all_detections
from tracer.report import build_audit_report, save_json_report, save_markdown_report
from tracer.models.mlx_backend import unload_all

config = Config()
config.brands = ["Oracle"]
config.ensure_dirs()

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E", config.paths.output_dir))
frames, duration = extract_frames(video_path, fps=1.0, frame_size=1000)
timestamps = {i: frame_timestamp(i, 1.0) for i in range(len(frames))}
print(f"Video: {video_path}")
print(f"Frames: {len(frames)}, Duration: {duration:.0f}s")

# Phase 1: Scout (E4B)
print("\n=== Scout Phase (E4B) ===")
scout = Scout(config)
scout.load()
flagged = []
for i in range(len(frames)):
    r = scout.classify(frames[i])
    if r.get("has_branding", False):
        flagged.append(i)
scout.unload()
unload_all()
print(f"Scout flagged {len(flagged)} frames")

# Phase 2: Auditor (26B-A4B) on flagged frames
print("\n=== Auditor Phase (26B-A4B MoE) ===")
auditor = Auditor(config)
auditor.load()
detections_by_frame = auditor.audit_frames(frames, flagged)
auditor.unload()
unload_all()

total = sum(len(v) for v in detections_by_frame.values())
for frame_idx in sorted(detections_by_frame.keys()):
    ts = timestamps[frame_idx]
    for d in detections_by_frame[frame_idx]:
        print(f"  [{ts}] Frame {frame_idx}: {d.get('label','?')} conf={d.get('confidence',0):.2f} box={d.get('box_2d','?')}")
print(f"\nTotal: {total} detections in {len(detections_by_frame)} frames")

if not detections_by_frame:
    print("No detections. Trying Auditor directly on all frames...")
    # Fallback: run Auditor on ALL frames
    auditor = Auditor(config)
    auditor.load()
    detections_by_frame = auditor.audit_frames(frames, list(range(len(frames))))
    auditor.unload()
    unload_all()
    total = sum(len(v) for v in detections_by_frame.values())
    print(f"Direct audit: {total} detections")

if not detections_by_frame:
    print("No detections found.")
    sys.exit(0)

# QoE
scorer = QoEScorer(config)
for frame_idx, dets in detections_by_frame.items():
    for det in dets:
        scorer.score_detection(frames[frame_idx], det)

# Crops
import shutil
shutil.rmtree(config.paths.crops_dir, ignore_errors=True)
detections_by_frame = crop_all_detections(frames, detections_by_frame, config.paths.crops_dir, timestamps)

# Report
report = build_audit_report(config, video_path, duration, len(frames),
                            list(detections_by_frame.keys()), detections_by_frame, timestamps)
json_path = save_json_report(report, config.paths.output_dir / "oracle_audit.json")
md_path = save_markdown_report(report, config.paths.output_dir / "oracle_audit.md")
print(f"\nJSON: {json_path}")
print(f"Markdown: {md_path}")
print(f"Crops: {config.paths.crops_dir}")
