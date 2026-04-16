"""Debug: see raw model output for each frame."""
import sys, re, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracer.video import extract_frames, resolve_video, frame_timestamp
from tracer.models.mlx_backend import Gemma4Model

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E"))
frames, _ = extract_frames(video_path, fps=1.0, frame_size=1000)

model = Gemma4Model("mlx-community/gemma-4-e4b-it-4bit", token_budget=1120)
model.load()

# Test frames 22, 15, 0 — known interesting frames
for idx in [0, 15, 22, 31]:
    ts = frame_timestamp(idx)
    r = model.generate(
        prompt="Find the Oracle logo in this image.",
        image=frames[idx],
        system_prompt=(
            "Look for the Oracle logo. If found, output JSON array: "
            '[{"box_2d": [y1,x1,y2,x2], "label": "Oracle", "confidence": 0.9}]. '
            "If not found, output []."
        ),
        max_tokens=300, temperature=0.0, enable_thinking=True,
    )
    dets = model._parse_detections(r)
    print(f"\nFrame {idx} [{ts}]: {len(dets)} detections")
    print(f"  Raw: {repr(r[:200])}")
    print(f"  Stripped: {repr(model._strip_thinking(r)[:200])}")
    if dets:
        for d in dets:
            print(f"  FOUND: {d}")

model.unload()
