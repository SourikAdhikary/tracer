"""Debug: test image passing to mlx_vlm.generate directly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import tempfile
import numpy as np
from PIL import Image
from tracer.video import extract_frames, resolve_video

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E"))
frames, _ = extract_frames(video_path, fps=1.0, frame_size=1000)

# Save frame 0 to a known path
frame = frames[0]
img = Image.fromarray(frame)
test_path = "/tmp/tracer_test_frame.png"
img.save(test_path)
print(f"Saved frame to {test_path}, size={img.size}")

# Try direct mlx_vlm.generate with explicit path
from mlx_vlm import load, generate

print("Loading model...")
model, processor = load("mlx-community/gemma-4-e4b-it-4bit")
print("Model loaded.")

print("\nCalling generate with image path...")
result = generate(
    model,
    processor,
    prompt="Describe what you see in this image. List all brand logos and text.",
    image=test_path,
    max_tokens=300,
    temp=0.0,
)
print(f"\nResult type: {type(result)}")
print(f"Result: {result}")
