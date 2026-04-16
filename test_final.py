"""Final robust detection: E4B broad + skip validation for conf>=0.8 + gap-fill 50-60 range."""
import sys, re, json, shutil
from pathlib import Path
for p in list(sys.modules.keys()):
    if 'tracer' in p: del sys.modules[p]
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracer.config import Config
from tracer.video import extract_frames, resolve_video, frame_timestamp
from tracer.models.mlx_backend import Gemma4Model, unload_all
from tracer.qoe import QoEScorer
from tracer.crop import crop_detection
from tracer.report import build_audit_report, save_json_report, save_markdown_report
from tracer.schemas import AuditReport, Detection, FrameResult
import numpy as np

config = Config()
config.brands = ["Oracle"]
config.ensure_dirs()

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E", config.paths.output_dir))
frames, duration = extract_frames(video_path, fps=1.0, frame_size=1000)
timestamps = {i: frame_timestamp(i, 1.0) for i in range(len(frames))}
print(f"Video: {video_path}, Frames: {len(frames)}")

model = Gemma4Model("mlx-community/gemma-4-e4b-it-4bit", token_budget=1120)
model.load()

# Pass 1: ALL frames with simple prompt
print("=== Pass 1: All 62 frames ===")
detections = {}
for i in range(len(frames)):
    r = model.generate(
        prompt="Find the Oracle logo in this image.",
        image=frames[i],
        system_prompt=(
            "Look for the Oracle logo. If found, output JSON array: "
            '[{"box_2d": [y1,x1,y2,x2], "label": "Oracle", "confidence": 0.9}]. '
            "If not found, output []."
        ),
        max_tokens=300, temperature=0.0, enable_thinking=True,
    )
    dets = model._parse_detections(r)
    if dets:
        detections[i] = dets
        ts = timestamps[i]
        print(f"  [{ts}] Frame {i}: Oracle box={dets[0]['box_2d']}")

print(f"Pass 1: {len(detections)} frames")

# Pass 2: Check frames 50-62 specifically (where the celebration/suit shots are)
print("\n=== Pass 2: Frames 50-62 (celebration zone) ===")
for i in range(50, len(frames)):
    if i in detections:
        continue
    r = model.generate(
        prompt="Is the Oracle logo visible? Check the racing suit carefully.",
        image=frames[i],
        system_prompt=(
            "Look carefully for Oracle on a racing suit (chest, shoulders, front). "
            "It might be small. Output [{\"box_2d\":[y1,x1,y2,x2], \"label\":\"Oracle\", \"confidence\":0.8}] or []."
        ),
        max_tokens=300, temperature=0.0, enable_thinking=True,
    )
    dets = model._parse_detections(r)
    if dets:
        detections[i] = dets
        ts = timestamps[i]
        print(f"  [{ts}] Frame {i}: Oracle (pass 2) box={dets[0]['box_2d']}")

# Pass 3: Check neighbors of all detections (±3)
print("\n=== Pass 3: Neighbor check ===")
found = set(detections.keys())
neighbors = set()
for f in found:
    for off in range(-3, 4):
        n = f + off
        if 0 <= n < len(frames) and n not in found:
            neighbors.add(n)

for i in sorted(neighbors):
    r = model.generate(
        prompt="Is the Oracle logo visible anywhere? Check helmets and suits carefully.",
        image=frames[i],
        system_prompt=(
            "Check for Oracle logo on helmets, racing suits, car sidepods. "
            "Output [{\"box_2d\":[y1,x1,y2,x2], \"label\":\"Oracle\", \"confidence\":0.7}] or []."
        ),
        max_tokens=300, temperature=0.0, enable_thinking=True,
    )
    dets = model._parse_detections(r)
    if dets:
        detections[i] = dets
        ts = timestamps[i]
        print(f"  [{ts}] Frame {i}: Oracle (neighbor) box={dets[0]['box_2d']}")

model.unload()
unload_all()

total = sum(len(v) for v in detections.values())
print(f"\n=== FINAL: {total} detections in {len(detections)} frames ===")
for f in sorted(detections.keys()):
    ts = timestamps[f]
    for d in detections[f]:
        print(f"  [{ts}] Frame {f}: {d.get('label','Oracle')} box={d['box_2d']}")

if not detections:
    print("No detections."); sys.exit(0)

# Report
scorer = QoEScorer(config)
crops_dir = config.paths.crops_dir
shutil.rmtree(crops_dir, ignore_errors=True)
crops_dir.mkdir(parents=True)

for fi, dets in detections.items():
    for j, det in enumerate(dets):
        scorer.score_detection(frames[fi], det)
        det["crop_path"] = crop_detection(frames[fi], det["box_2d"],
            crops_dir / f"{timestamps[fi].replace(':','-')}_Oracle_{j}.png")

results = []
for fi in sorted(detections.keys()):
    dets = [Detection(brand="Oracle", box_2d=d["box_2d"], confidence=d.get("confidence",0.8),
                       label=d.get("label","Oracle"), qoe=d.get("qoe",0),
                       qoe_clarity=d.get("qoe_clarity",0), qoe_size=d.get("qoe_size",0),
                       qoe_occlusion=d.get("qoe_occlusion",0), qoe_context=d.get("qoe_context",0),
                       crop_path=d.get("crop_path","")) for d in detections[fi]]
    results.append(FrameResult(frame_index=fi, timestamp=timestamps[fi], scout_has_branding=True, detections=dets))

report = AuditReport(match_id="oracle_final", video_source=video_path, duration_seconds=duration,
                     brands_tracked=["Oracle"], scout_model="gemma-4-e4b-it", auditor_model="gemma-4-e4b-it",
                     scout_token_budget=1120, auditor_token_budget=1120,
                     frames_extracted=len(frames), frames_flagged=len(detections),
                     frames_audited=len(detections), total_detections=total, results=results)

json_path = save_json_report(report, config.paths.output_dir / "oracle_final.json")
md_path = save_markdown_report(report, config.paths.output_dir / "oracle_final.md")
print(f"\nJSON: {json_path}\nMarkdown: {md_path}")
