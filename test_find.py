"""Find Red Bull frames, then look for Oracle on those."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracer.video import extract_frames, resolve_video, frame_timestamp
from tracer.models.mlx_backend import Gemma4Model

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E"))
frames, _ = extract_frames(video_path, fps=1.0, frame_size=1000)

model = Gemma4Model("mlx-community/gemma-4-e4b-it-4bit", token_budget=560)
model.load()

# Pass 1: Find frames with Red Bull or F1 cars
print("=== Pass 1: Finding Red Bull / F1 frames ===")
redbull_frames = []
for i in range(len(frames)):
    r = model.generate(
        prompt="Is there a Red Bull branded car or F1 car visible? Answer YES or NO and describe briefly.",
        image=frames[i],
        max_tokens=100, temperature=0.0, enable_thinking=False,
    )
    if "yes" in r.lower()[:20]:
        ts = frame_timestamp(i)
        print(f"  Frame {i} [{ts}]: {r[:120]}")
        redbull_frames.append(i)

print(f"\nFound {len(redbull_frames)} Red Bull/F1 frames")

# Pass 2: On Red Bull frames, look for ALL text
print("\n=== Pass 2: Reading all text on Red Bull frames ===")
for idx in redbull_frames[:10]:
    ts = frame_timestamp(idx)
    r = model.generate(
        prompt="Read EVERY piece of text visible on the car. List them all, even if small. Include sponsor names, numbers, team names.",
        image=frames[idx],
        max_tokens=300, temperature=0.0, enable_thinking=False,
    )
    print(f"\n  Frame {idx} [{ts}]:")
    print(f"  {r[:300]}")

model.unload()
