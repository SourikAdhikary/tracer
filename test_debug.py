"""Debug: what does Gemma 4 E4B actually see in a frame?"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracer.video import extract_frames, resolve_video, frame_timestamp
from tracer.models.mlx_backend import Gemma4Model

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E"))
frames, _ = extract_frames(video_path, fps=1.0, frame_size=1000)

model = Gemma4Model(model_id="mlx-community/gemma-4-e4b-it-4bit", token_budget=560)
model.load()

# Just describe what you see
for idx in [0, 15, 30]:
    ts = frame_timestamp(idx)
    print(f"\n=== Frame {idx} [{ts}] ===")
    response = model.generate(
        prompt="Describe this image in detail. List every brand logo, text, and sign you can see.",
        image=frames[idx],
        system_prompt="You are an expert image analyst.",
        max_tokens=500,
        temperature=0.0,
        enable_thinking=False,
    )
    print(response)

model.unload()
