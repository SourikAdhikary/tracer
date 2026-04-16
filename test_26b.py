"""26B-A4B direct audit: F1-specific prompt, all 62 frames, thinking mode."""
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

config = Config()
config.brands = ["Oracle"]
config.ensure_dirs()

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E", config.paths.output_dir))
frames, duration = extract_frames(video_path, fps=1.0, frame_size=1000)
timestamps = {i: frame_timestamp(i, 1.0) for i in range(len(frames))}
print(f"Video: {video_path}, Frames: {len(frames)}")

model = Gemma4Model("mlx-community/gemma-4-26b-a4b-it-4bit", token_budget=1120)
model.load()

# F1-specific system prompt with thinking mode
SYSTEM_PROMPT = (
    "<|think|>\n"
    "You are an expert Formula 1 sponsorship auditor. Your job is to find the Oracle logo "
    "in every frame of this F1 broadcast.\n\n"
    "The Oracle logo (white text 'ORACLE' on black background) appears in these locations:\n"
    "- Red Bull Racing car: sidepod (large, near cockpit), nose cone, rear wing\n"
    "- Red Bull driver helmets: visor strip, side of helmet\n"
    "- Red Bull racing suits: upper chest/shoulders (large), front chest, back\n"
    "- Trackside: Oracle-branded billboards and pit wall signage\n\n"
    "IMPORTANT: Check EVERY part of the frame. Even if the logo is small, partially visible, "
    "or at an angle, report it. Do NOT skip frames just because the logo is tiny.\n\n"
    "For each detection output: "
    '{"box_2d": [y1,x1,y2,x2], "label": "Oracle_Location", "confidence": 0.8}\n'
    "Coordinates on 1000x1000 grid. If Oracle is NOT visible, output [].\n"
    "Always output a JSON array at the end."
)

detections = {}
for i in range(len(frames)):
    r = model.generate(
        prompt="Analyze this F1 broadcast frame. Find the Oracle logo anywhere — car, helmet, suit, or trackside.",
        image=frames[i],
        system_prompt=SYSTEM_PROMPT,
        max_tokens=500,
        temperature=0.0,
        enable_thinking=True,
    )
    dets = model._parse_detections(r)
    if dets:
        detections[i] = dets
        ts = timestamps[i]
        for d in dets:
            print(f"  [{ts}] Frame {i}: {d.get('label','Oracle')} box={d['box_2d']} conf={d.get('confidence',0)}")
    if (i + 1) % 10 == 0:
        print(f"  ... {i+1}/{len(frames)}")

model.unload()
unload_all()

total = sum(len(v) for v in detections.values())
print(f"\n=== {total} Oracle detections in {len(detections)} frames ===")
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

report = AuditReport(match_id="oracle_26b_final", video_source=video_path, duration_seconds=duration,
                     brands_tracked=["Oracle"], scout_model="none (direct audit)",
                     auditor_model="gemma-4-26b-a4b-it-4bit",
                     scout_token_budget=0, auditor_token_budget=1120,
                     frames_extracted=len(frames), frames_flagged=len(detections),
                     frames_audited=len(detections), total_detections=total, results=results)

json_path = save_json_report(report, config.paths.output_dir / "oracle_26b.json")
md_path = save_markdown_report(report, config.paths.output_dir / "oracle_26b.md")
print(f"\nJSON: {json_path}\nMarkdown: {md_path}")
