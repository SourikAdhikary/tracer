"""Test the 4 known Oracle frames and nearby frames."""
import sys
from pathlib import Path
for p in list(sys.modules.keys()):
    if 'tracer' in p: del sys.modules[p]
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracer.video import extract_frames, resolve_video, frame_timestamp
from tracer.models.mlx_backend import Gemma4Model

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E"))
frames, _ = extract_frames(video_path, fps=1.0, frame_size=1000)

model = Gemma4Model("mlx-community/gemma-4-e4b-it-4bit", token_budget=1120)
model.load()

# Test the 4 known Oracle frames + neighbors
for idx in [17, 18, 19, 21, 22, 23, 54, 55, 56, 57]:
    ts = frame_timestamp(idx)
    r = model.generate(
        prompt="Find the Oracle logo in this image.",
        image=frames[idx],
        system_prompt=(
            "Look for the Oracle logo (text 'Oracle' or Oracle wordmark). "
            "It may appear on F1 cars (sidepod, rear wing, nose), helmets, or racing suits. "
            "If found, output JSON array: "
            '[{"box_2d": [y1,x1,y2,x2], "label": "Oracle_Location", "confidence": 0.5}]. '
            "If not found, output []. "
            "Be thorough — check ALL parts of the image."
        ),
        max_tokens=300, temperature=0.0, enable_thinking=True,
    )
    dets = model._parse_detections(r)
    marker = "FOUND" if dets else "miss "
    print(f"Frame {idx:2d} [{ts}]: {marker} — {len(dets)} det(s)")
    if dets:
        for d in dets:
            print(f"    box={d['box_2d']} conf={d['confidence']}")
    # Also show raw stripped for non-detections to debug
    if not dets:
        stripped = model._strip_thinking(r)
        print(f"    raw: {stripped[:150]}")

model.unload()
