"""Test: strip thinking with fresh import (no cache)."""
import sys
from pathlib import Path
# Force fresh import
for p in list(sys.modules.keys()):
    if 'tracer' in p:
        del sys.modules[p]
sys.path.insert(0, str(Path(__file__).parent / "src"))

import re

# Test the regex directly
test_input = '<|channel|>thought\nHere are the detections:\n- `[369, 269, 411, 379]`\n<channel|>[{"box_2d": [369, 269, 411, 379], "label": "Oracle", "confidence": 0.9}]'
pattern = r'<\|channel\|>thought[\s\S]*?(?:<\|channel\|>|<channel\|>)'
result = re.sub(pattern, '', test_input)
print(f"Input: {repr(test_input[:100])}")
print(f"Result: {repr(result)}")

# Now test via model
from tracer.models.mlx_backend import Gemma4Model
model = Gemma4Model("test")
model_result = model._strip_thinking(test_input)
print(f"Model._strip_thinking: {repr(model_result)}")

# Parse
dets = model._parse_detections(test_input)
print(f"Detections: {dets}")
