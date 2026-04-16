"""Full run: detect Oracle across all 62 frames, generate report + crops."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import json
import re
from tracer.config import Config
from tracer.video import extract_frames, resolve_video, frame_timestamp
from tracer.models.mlx_backend import Gemma4Model, unload_all
from tracer.qoe import QoEScorer
from tracer.crop import crop_all_detections
from tracer.report import build_audit_report, save_json_report, save_markdown_report

config = Config()
config.brands = ["Oracle"]
config.ensure_dirs()

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E", config.paths.output_dir))
frames, duration = extract_frames(video_path, fps=1.0, frame_size=1000)
timestamps = {i: frame_timestamp(i, 1.0) for i in range(len(frames))}

print(f"Video: {video_path}")
print(f"Frames: {len(frames)}, Duration: {duration:.0f}s")

model = Gemma4Model("mlx-community/gemma-4-e4b-it-4bit", token_budget=1120)
model.load()

# Detection prompt that works
DETECTION_PROMPT = (
    "Look carefully at this image for the 'Oracle' logo (the Oracle Corporation wordmark). "
    "It may appear on F1 cars (sidepod, rear wing, nose), signage, or merchandise. "
    "If Oracle is visible, estimate its position as [y1,x1,y2,x2] on a 1000x1000 grid "
    "(top-left to bottom-right). Output ONLY a JSON array: "
    '[{"box_2d": [y1,x1,y2,x2], "label": "Oracle_Location", "confidence": 0.0-1.0}]. '
    "If Oracle is not visible, output []."
)

detections_by_frame = {}
for i in range(len(frames)):
    r = model.generate(
        prompt="Find the Oracle logo in this image.",
        image=frames[i],
        system_prompt=DETECTION_PROMPT,
        max_tokens=300,
        temperature=0.0,
        enable_thinking=True,
    )

    # Use model's fixed parser
    dets = model._parse_detections(r)

    if dets:
        detections_by_frame[i] = dets
        ts = timestamps[i]
        for d in dets:
            print(f"  [{ts}] Frame {i}: Oracle conf={d['confidence']:.2f} box={d['box_2d']}")

    if (i + 1) % 10 == 0:
        print(f"  ... {i+1}/{len(frames)}")

model.unload()
unload_all()

total = sum(len(v) for v in detections_by_frame.values())
print(f"\nFound {total} Oracle detections in {len(detections_by_frame)} frames.")

if not detections_by_frame:
    print("No detections found.")
    sys.exit(0)

# QoE
scorer = QoEScorer(config)
for frame_idx, dets in detections_by_frame.items():
    for det in dets:
        scorer.score_detection(frames[frame_idx], det)

# Crops
detections_by_frame = crop_all_detections(frames, detections_by_frame, config.paths.crops_dir, timestamps)

# Report
report = build_audit_report(config, video_path, duration, len(frames),
                            list(detections_by_frame.keys()), detections_by_frame, timestamps)
json_path = save_json_report(report, config.paths.output_dir / "oracle_audit.json")
md_path = save_markdown_report(report, config.paths.output_dir / "oracle_audit.md")

print(f"\nJSON: {json_path}")
print(f"Markdown: {md_path}")
print(f"Crops: {config.paths.crops_dir}")
