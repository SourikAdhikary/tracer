"""Debug: test different prompt formats with mlx_vlm."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracer.video import extract_frames, resolve_video
from mlx_vlm import load, generate

video_path = str(resolve_video("https://www.youtube.com/watch?v=viQC-6xoJ3E"))
frames, _ = extract_frames(video_path, fps=1.0, frame_size=1000)

from PIL import Image
test_path = "/tmp/tracer_test_frame.png"
Image.fromarray(frames[0]).save(test_path)

model, processor = load("mlx-community/gemma-4-e4b-it-4bit")

# Test 1: Simple prompt, no system
print("=== Test 1: Simple prompt ===")
r = generate(model, processor,
    prompt="What brands and logos are visible in this image?",
    image=test_path, max_tokens=200, temp=0.0)
print(f"  {r.text[:300]}")

# Test 2: Use chat template via processor
print("\n=== Test 2: Chat template ===")
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "What brands and logos are visible in this image?"},
    ]},
]
chat_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"  Template: {chat_prompt[:200]}...")
r = generate(model, processor,
    prompt=chat_prompt,
    image=test_path, max_tokens=200, temp=0.0)
print(f"  {r.text[:300]}")

# Test 3: With system prompt
print("\n=== Test 3: With system ===")
messages = [
    {"role": "system", "content": "You are an expert at identifying brand logos in images."},
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "List every brand logo you can see."},
    ]},
]
chat_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"  Template: {chat_prompt[:200]}...")
r = generate(model, processor,
    prompt=chat_prompt,
    image=test_path, max_tokens=200, temp=0.0)
print(f"  {r.text[:300]}")
