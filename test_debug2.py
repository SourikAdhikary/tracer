"""Debug: raw response for frame 22 with detection prompt."""
import sys, re, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracer.video import extract_frames, resolve_video
from tracer.models.mlx_backend import Gemma4Model

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E"))
frames, _ = extract_frames(video_path, fps=1.0, frame_size=1000)

model = Gemma4Model("mlx-community/gemma-4-e4b-it-4bit", token_budget=1120)
model.load()

DETECTION_PROMPT = (
    "Look carefully at this image for the 'Oracle' logo (the Oracle Corporation wordmark). "
    "It may appear on F1 cars (sidepod, rear wing, nose), signage, or merchandise. "
    "If Oracle is visible, estimate its position as [y1,x1,y2,x2] on a 1000x1000 grid "
    "(top-left to bottom-right). Output ONLY a JSON array: "
    '[{"box_2d": [y1,x1,y2,x2], "label": "Oracle_Location", "confidence": 0.0-1.0}]. '
    "If Oracle is not visible, output []."
)

r = model.generate(
    prompt="Find the Oracle logo in this image.",
    image=frames[22],
    system_prompt=DETECTION_PROMPT,
    max_tokens=300,
    temperature=0.0,
    enable_thinking=True,
)

print("=== RAW ===")
print(repr(r))
print("\n=== STRIPPED ===")
stripped = re.sub(r'<\|channel\|>thought[\s\S]*?<channel\|>', '', r).strip()
stripped = re.sub(r'<turn\|>', '', stripped).strip()
print(repr(stripped))
print("\n=== JSON MATCH ===")
match = re.search(r'\[[\s\S]*?\]', stripped)
if match:
    print(f"Match: {match.group()}")
    try:
        data = json.loads(match.group())
        print(f"Parsed: {data}")
    except:
        print("Parse failed")
else:
    print("No JSON array found")

model.unload()
