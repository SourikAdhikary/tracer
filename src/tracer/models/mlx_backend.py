"""MLX-VLM backend for Gemma 4 inference.

Wraps mlx-vlm to provide a clean interface for loading models,
running inference with images, and parsing structured outputs.
"""

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


class Gemma4Model:
    """Gemma 4 model wrapper using mlx-vlm.

    Handles loading, inference, and output parsing for both
    Scout (E4B) and Auditor (26B-A4B) roles.
    """

    def __init__(self, model_id: str, token_budget: int = 70):
        self.model_id = model_id
        self.token_budget = token_budget
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load model and processor from HuggingFace (cached locally)."""
        # Lazy import — mlx_vlm is heavy
        from mlx_vlm import load

        print(f"[Tracer] Loading {self.model_id} (token_budget={self.token_budget})...")
        self._model, self._processor = load(self.model_id)
        print(f"[Tracer] Model loaded.")

    def unload(self) -> None:
        """Free model memory."""
        self._model = None
        self._processor = None
        import gc
        import mlx.core as mx
        gc.collect()
        mx.metal.clear_cache()
        print(f"[Tracer] Model unloaded: {self.model_id}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(
        self,
        prompt: str,
        image: np.ndarray | Image.Image | None = None,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        enable_thinking: bool = True,
    ) -> str:
        """Run inference with optional image input.

        Args:
            prompt: User prompt text.
            image: Optional image as numpy array (H,W,3) RGB or PIL Image.
            system_prompt: System prompt (prepended with <|think|> if thinking enabled).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            enable_thinking: Whether to enable thinking mode.

        Returns:
            Generated text response.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        from mlx_vlm import generate
        from mlx_vlm.utils import load_image

        # Build messages in chat format
        messages = []

        # System prompt
        if system_prompt:
            if enable_thinking:
                system_content = "<|think|>\n" + system_prompt
            else:
                system_content = system_prompt
            messages.append({"role": "system", "content": system_content})

        # User message with optional image
        if image is not None:
            # Convert numpy to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Save to temp for mlx-vlm (it expects file paths or URLs)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f, format="PNG")
                temp_path = f.name

            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            })
        else:
            messages.append({"role": "user", "content": prompt})

        # Run generation
        result = generate(
            self._model,
            self._processor,
            prompt=prompt if image is None else None,
            image=temp_path if image is not None else None,
            messages=messages if image is not None else None,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # Clean up temp file
        if image is not None:
            import os
            os.unlink(temp_path)

        # Extract text from result
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return result.get("text", str(result))
        else:
            return str(result)

    def classify_frame(
        self,
        frame: np.ndarray,
        system_prompt: str = "",
    ) -> dict:
        """Scout mode: classify whether a frame contains branding.

        Args:
            frame: numpy array (H, W, 3) RGB.
            system_prompt: Scout system prompt.

        Returns:
            Dict with "has_branding" (bool) and "confidence" (float).
        """
        if not system_prompt:
            system_prompt = (
                "You are a sports broadcast frame classifier. Analyze this frame "
                "and determine if it contains ANY branded content: logos on jerseys, "
                "helmets, equipment; billboard advertisements; sponsor banners; "
                "branded products; team crest/sponsor patches on kits.\n\n"
                'Respond with ONLY a JSON object: {"has_branding": true/false, "confidence": 0.0-1.0}'
            )

        response = self.generate(
            prompt="Analyze this frame for branded content.",
            image=frame,
            system_prompt=system_prompt,
            max_tokens=100,
            temperature=0.0,  # Deterministic for classification
            enable_thinking=True,
        )

        return self._parse_classification(response)

    def detect_logos(
        self,
        frame: np.ndarray,
        brands: list[str],
        system_prompt: str = "",
    ) -> list[dict]:
        """Auditor mode: detect and localize brand logos.

        Args:
            frame: numpy array (H, W, 3) RGB.
            brands: List of brand names to detect.
            system_prompt: Auditor system prompt (auto-generated if empty).

        Returns:
            List of detection dicts with box_2d, label, confidence.
        """
        brand_list = ", ".join(brands)

        if not system_prompt:
            system_prompt = (
                "You are a professional sports broadcast logo detector. "
                "Your task is to locate and identify brand logos in this frame "
                "with high precision.\n\n"
                f"Target brands: {brand_list}\n\n"
                "For each detection, output a JSON object:\n"
                '{"box_2d": [y1, x1, y2, x2], "label": "BrandName_Location", "confidence": 0.0-1.0}\n\n'
                "Coordinate system:\n"
                "- Normalized to a 1000x1000 grid\n"
                "- [y1, x1] = top-left corner\n"
                "- [y2, x2] = bottom-right corner\n"
                "- y increases downward, x increases rightward\n\n"
                "Label format: BrandName_Location (e.g., Emirates_Chest, Etihad_Board)\n\n"
                "Rules:\n"
                "- Only output detections with confidence > 0.7\n"
                "- If a logo appears multiple times, create separate detections\n"
                "- Consider partial occlusion\n"
                "- Output a JSON array of detections: [{...}, {...}]"
            )

        response = self.generate(
            prompt=f"Detect all instances of these brands: {brand_list}",
            image=frame,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.0,
            enable_thinking=True,
        )

        return self._parse_detections(response)

    def _parse_classification(self, response: str) -> dict:
        """Parse Scout classification output."""
        # Strip thinking tags if present
        response = self._strip_thinking(response)

        try:
            # Try to find JSON in the response
            match = re.search(r'\{[^}]+\}', response)
            if match:
                data = json.loads(match.group())
                return {
                    "has_branding": bool(data.get("has_branding", False)),
                    "confidence": float(data.get("confidence", 0.0)),
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: check for keywords
        lower = response.lower()
        if "true" in lower and "false" not in lower:
            return {"has_branding": True, "confidence": 0.5}
        return {"has_branding": False, "confidence": 0.5}

    def _parse_detections(self, response: str) -> list[dict]:
        """Parse Auditor detection output."""
        response = self._strip_thinking(response)

        detections = []

        # Try parsing as JSON array first
        try:
            # Look for array pattern
            match = re.search(r'\[[\s\S]*\]', response)
            if match:
                data = json.loads(match.group())
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "box_2d" in item:
                            detections.append({
                                "box_2d": item["box_2d"],
                                "label": item.get("label", "Unknown"),
                                "confidence": float(item.get("confidence", 0.5)),
                            })
                    return detections
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: look for individual JSON objects
        for match in re.finditer(r'\{[^}]+\}', response):
            try:
                item = json.loads(match.group())
                if "box_2d" in item:
                    detections.append({
                        "box_2d": item["box_2d"],
                        "label": item.get("label", "Unknown"),
                        "confidence": float(item.get("confidence", 0.5)),
                    })
            except (json.JSONDecodeError, ValueError):
                continue

        return detections

    def _strip_thinking(self, text: str) -> str:
        """Remove <|channel>thought...<channel|> blocks from output."""
        # Remove thinking blocks
        text = re.sub(r'<\|channel\|>thought[\s\S]*?<channel\|>', '', text)
        # Remove any remaining special tokens
        text = re.sub(r'<\|[^|]+\|>', '', text)
        return text.strip()


def unload_all():
    """Force cleanup of all MLX memory."""
    import gc
    try:
        import mlx.core as mx
        gc.collect()
        mx.metal.clear_cache()
    except ImportError:
        pass
