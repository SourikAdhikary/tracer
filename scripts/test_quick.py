"""Minimal test: detect Oracle in 5 sampled frames using E4B."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from tracer.config import Config
from tracer.video import extract_frames, resolve_video, frame_timestamp
from tracer.models.mlx_backend import Gemma4Model, unload_all

config = Config()
config.brands = ["Oracle"]

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E", config.paths.output_dir))
print(f"Video: {video_path}")

frames, duration = extract_frames(video_path, fps=1.0, frame_size=1000)
print(f"Extracted {len(frames)} frames, duration={duration:.0f}s")

# Sample 5 frames spread across the video
indices = [0, 15, 30, 45, 60]
indices = [i for i in indices if i < len(frames)]
print(f"Testing frames: {indices}")

model = Gemma4Model(model_id="mlx-community/gemma-4-e4b-it-4bit", token_budget=560)
model.load()

for idx in indices:
    ts = frame_timestamp(idx)
    print(f"\n--- Frame {idx} [{ts}] ---")
    dets = model.detect_logos(frames[idx], brands=["Oracle"])
    if dets:
        for d in dets:
            print(f"  FOUND: {d.get('label','?')} conf={d.get('confidence',0):.2f} box={d.get('box_2d','?')}")
    else:
        print("  No detections.")

model.unload()
unload_all()
print("\nDone.")
