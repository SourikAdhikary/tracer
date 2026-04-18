"""Debug: strip thinking on frame 22 raw output."""
import sys
from pathlib import Path
for p in list(sys.modules.keys()):
    if 'tracer' in p: del sys.modules[p]
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracer.video import extract_frames, resolve_video
from tracer.models.mlx_backend import Gemma4Model

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E"))
frames, _ = extract_frames(video_path, fps=1.0, frame_size=1000)

model = Gemma4Model("mlx-community/gemma-4-e4b-it-4bit", token_budget=1120)
model.load()

r = model.generate(
    prompt="Find the Oracle logo in this image.",
    image=frames[22],
    system_prompt='Look for Oracle logo. Output JSON: [{"box_2d":[y1,x1,y2,x2],"label":"Oracle","confidence":0.9}] or [].',
    max_tokens=500, temperature=0.0, enable_thinking=True,
)

print(f"FULL raw length: {len(r)}")
print(f"FULL raw: {repr(r)}")
print()

stripped = model._strip_thinking(r)
print(f"Stripped length: {len(stripped)}")
print(f"Stripped: {repr(stripped)}")
print()

dets = model._parse_detections(r)
print(f"Detections: {dets}")

model.unload()
