"""Debug: raw detect_logos output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracer.video import extract_frames, resolve_video
from tracer.models.mlx_backend import Gemma4Model

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E"))
frames, _ = extract_frames(video_path, fps=1.0, frame_size=1000)

model = Gemma4Model("mlx-community/gemma-4-e4b-it-4bit", token_budget=560)
model.load()

# Raw generation with detection prompt
response = model.generate(
    prompt="Find all instances of: Oracle. First describe what you see, then output a JSON array of detections.",
    image=frames[15],
    system_prompt=(
        "You are a logo detector. Output a JSON array:\n"
        '[{"box_2d": [y1,x1,y2,x2], "label": "Brand_Location", "confidence": 0.0-1.0}]\n'
        "1000x1000 grid. If nothing found, output []."
    ),
    max_tokens=1000,
    temperature=0.0,
    enable_thinking=True,
)

print("=== RAW RESPONSE ===")
print(repr(response[:2000]))
print("\n=== STRIPPED ===")
print(model._strip_thinking(response)[:1000])

model.unload()
