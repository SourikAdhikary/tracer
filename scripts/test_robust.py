"""Robust Oracle detection: E4B broad scan + crop validation + gap-filling."""
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
from tracer.schemas import Detection, FrameResult
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

# === Phase 1: Broad detection on ALL frames ===
print("\n=== Phase 1: Broad Detection ===")
raw_detections = {}
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
        raw_detections[i] = dets
        ts = timestamps[i]
        for d in dets:
            print(f"  [{ts}] Frame {i}: box={d['box_2d']} conf={d['confidence']}")

print(f"Phase 1: {sum(len(v) for v in raw_detections.values())} raw detections")

# === Phase 2: Validate each detection with crop check ===
print("\n=== Phase 2: Crop Validation ===")
validated = {}
for frame_idx, dets in raw_detections.items():
    for det in dets:
        # Crop the detection
        box = det["box_2d"]
        fh, fw = frames[frame_idx].shape[:2]
        y1 = max(0, int(box[0] * fh / 1000) - 20)
        x1 = max(0, int(box[1] * fw / 1000) - 20)
        y2 = min(fh, int(box[2] * fh / 1000) + 20)
        x2 = min(fw, int(box[3] * fw / 1000) + 20)
        crop = frames[frame_idx][y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        # Ask model to validate
        val = model.generate(
            prompt="Is the Oracle logo or text 'Oracle' visible in this cropped image? Answer YES or NO.",
            image=crop,
            system_prompt="You are checking if a cropped image contains the Oracle logo. Answer only YES or NO.",
            max_tokens=20, temperature=0.0, enable_thinking=False,
        )
        is_valid = "yes" in val.lower()[:20]
        ts = timestamps[frame_idx]
        status = "VALID" if is_valid else "REJECTED"
        print(f"  [{ts}] Frame {frame_idx}: {status} — {val[:60]}")

        if is_valid:
            if frame_idx not in validated:
                validated[frame_idx] = []
            validated[frame_idx].append(det)

print(f"Phase 2: {sum(len(v) for v in validated.values())} validated detections")

# === Phase 3: Gap-filling — check frames near detections ===
print("\n=== Phase 3: Gap-Filling ===")
# Get all detected frame indices
detected_frames = set(validated.keys())
# Check neighbors of detected frames (±3 frames)
neighbor_candidates = set()
for f in detected_frames:
    for offset in [-3, -2, -1, 1, 2, 3]:
        neighbor = f + offset
        if 0 <= neighbor < len(frames) and neighbor not in detected_frames:
            neighbor_candidates.add(neighbor)

print(f"Checking {len(neighbor_candidates)} neighbor frames: {sorted(neighbor_candidates)}")

for i in sorted(neighbor_candidates):
    r = model.generate(
        prompt="Is the Oracle logo visible anywhere in this image? Check carefully — it might be small. Output JSON array or [].",
        image=frames[i],
        system_prompt=(
            "Look VERY carefully for the Oracle logo. Check:\n"
            "- Driver helmet/visor area\n"
            "- Racing suit chest/shoulders\n"
            "- Car sidepod\n"
            "It might be small or partially visible. Output [{\"box_2d\":[y1,x1,y2,x2], \"label\":\"Oracle\", \"confidence\":0.5}] or []."
        ),
        max_tokens=300, temperature=0.0, enable_thinking=True,
    )
    dets = model._parse_detections(r)
    if dets:
        ts = timestamps[i]
        # Validate with crop
        for det in dets:
            box = det["box_2d"]
            fh, fw = frames[i].shape[:2]
            y1 = max(0, int(box[0] * fh / 1000) - 20)
            x1 = max(0, int(box[1] * fw / 1000) - 20)
            y2 = min(fh, int(box[2] * fh / 1000) + 20)
            x2 = min(fw, int(box[3] * fw / 1000) + 20)
            crop = frames[i][y1:y2, x1:x2]
            if crop.size > 0:
                val = model.generate(
                    prompt="Is the Oracle logo visible here? YES or NO.",
                    image=crop, system_prompt="Check for Oracle logo. YES or NO only.",
                    max_tokens=20, temperature=0.0, enable_thinking=False,
                )
                if "yes" in val.lower()[:20]:
                    if i not in validated:
                        validated[i] = []
                    validated[i].append(det)
                    print(f"  [{ts}] Frame {i}: FOUND (gap-fill) box={det['box_2d']}")
                else:
                    print(f"  [{ts}] Frame {i}: rejected on validation")

model.unload()
unload_all()

total = sum(len(v) for v in validated.values())
print(f"\n=== FINAL: {total} validated Oracle detections in {len(validated)} frames ===")
for f in sorted(validated.keys()):
    ts = timestamps[f]
    for d in validated[f]:
        print(f"  [{ts}] Frame {f}: {d.get('label','Oracle')} box={d['box_2d']}")

if not validated:
    print("No detections.")
    sys.exit(0)

# === Generate report ===
scorer = QoEScorer(config)
crops_dir = config.paths.crops_dir
shutil.rmtree(crops_dir, ignore_errors=True)
crops_dir.mkdir(parents=True)

for frame_idx, dets in validated.items():
    for det in dets:
        scorer.score_detection(frames[frame_idx], det)
        det["crop_path"] = crop_detection(frames[frame_idx], det["box_2d"],
            crops_dir / f"{timestamps[frame_idx].replace(':','-')}_Oracle_{validated[frame_idx].index(det)}.png")

results = []
for frame_idx in sorted(validated.keys()):
    dets = [Detection(brand="Oracle", box_2d=d["box_2d"], confidence=d.get("confidence",0.5),
                       label=d.get("label","Oracle"), qoe=d.get("qoe",0),
                       qoe_clarity=d.get("qoe_clarity",0), qoe_size=d.get("qoe_size",0),
                       qoe_occlusion=d.get("qoe_occlusion",0), qoe_context=d.get("qoe_context",0),
                       crop_path=d.get("crop_path","")) for d in validated[frame_idx]]
    results.append(FrameResult(frame_index=frame_idx, timestamp=timestamps[frame_idx],
                                scout_has_branding=True, detections=dets))

from tracer.schemas import AuditReport
report = AuditReport(match_id="oracle_final", video_source=video_path, duration_seconds=duration,
                     brands_tracked=["Oracle"], scout_model="gemma-4-e4b-it-4bit",
                     auditor_model="gemma-4-e4b-it-4bit", scout_token_budget=1120,
                     auditor_token_budget=1120, frames_extracted=len(frames),
                     frames_flagged=len(validated), frames_audited=len(validated),
                     total_detections=total, results=results)

json_path = save_json_report(report, config.paths.output_dir / "oracle_final.json")
md_path = save_markdown_report(report, config.paths.output_dir / "oracle_final.md")
print(f"\nJSON: {json_path}")
print(f"Markdown: {md_path}")
