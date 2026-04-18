"""Focused test: detect Oracle on frames 22-35 (where Red Bull/Oracle appears)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracer.video import extract_frames, resolve_video, frame_timestamp
from tracer.models.mlx_backend import Gemma4Model

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E"))
frames, _ = extract_frames(video_path, fps=1.0, frame_size=1000)

model = Gemma4Model("mlx-community/gemma-4-e4b-it-4bit", token_budget=1120)
model.load()

# On each frame, ask: where specifically is the Oracle logo?
for idx in range(22, 36):
    ts = frame_timestamp(idx)
    r = model.generate(
        prompt=(
            "Look carefully at this F1 car. I need to find the 'Oracle' logo specifically. "
            "Where on the car is it located? (e.g., sidepod, rear wing, nose, helmet) "
            "Estimate its position on a 1000x1000 grid as [y1,x1,y2,x2] where the logo text appears. "
            "If Oracle is not visible, say 'NOT VISIBLE'."
        ),
        image=frames[idx],
        max_tokens=300, temperature=0.0, enable_thinking=False,
    )
    has_oracle = "not visible" not in r.lower()
    marker = ">>> ORACLE <<<" if has_oracle else ""
    print(f"Frame {idx} [{ts}]: {marker}")
    print(f"  {r[:250]}")
    print()

model.unload()
