"""MLX-VLM backend for Gemma 4 inference.

Key: MUST use processor.apply_chat_template() with messages format.
The image placeholder is <|image|> in the user message content.
"""

import json
import os
import re
import tempfile

import numpy as np
from PIL import Image


class Gemma4Model:
    """Gemma 4 model wrapper using mlx-vlm."""

    def __init__(self, model_id: str, token_budget: int = 70):
        self.model_id = model_id
        self.token_budget = token_budget
        self._model = None
        self._processor = None

    def load(self) -> None:
        from mlx_vlm import load
        print(f"[Tracer] Loading {self.model_id}...")
        self._model, self._processor = load(self.model_id)
        print(f"[Tracer] Model loaded.")

    def unload(self) -> None:
        self._model = None
        self._processor = None
        import gc, mlx.core as mx
        gc.collect()
        mx.clear_cache()
        print(f"[Tracer] Unloaded: {self.model_id}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _save_image(self, image: np.ndarray | Image.Image | str) -> str:
        """Resolve image to a file path."""
        if isinstance(image, str):
            return image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(f, format="PNG")
        f.close()
        return f.name

    def _build_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        has_image: bool,
        enable_thinking: bool = True,
    ) -> str:
        """Build prompt using processor.apply_chat_template()."""
        messages = []

        if system_prompt:
            sys_content = system_prompt
            if enable_thinking:
                sys_content = "<|think|>\n" + sys_content
            messages.append({"role": "system", "content": sys_content})

        if has_image:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            })
        else:
            messages.append({"role": "user", "content": user_prompt})

        return self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        prompt: str,
        image: np.ndarray | Image.Image | str | None = None,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        enable_thinking: bool = True,
    ) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        from mlx_vlm import generate as vlm_generate

        has_image = image is not None
        image_path = self._save_image(image) if has_image else None
        full_prompt = self._build_prompt(system_prompt, prompt, has_image, enable_thinking)

        try:
            result = vlm_generate(
                self._model,
                self._processor,
                prompt=full_prompt,
                image=image_path,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        finally:
            if has_image and image_path and os.path.exists(image_path):
                os.unlink(image_path)

        # Extract text
        if isinstance(result, str):
            return result
        if hasattr(result, "text"):
            return result.text
        return str(result)

    def classify_frame(self, frame: np.ndarray, system_prompt: str = "") -> dict:
        """Scout: classify whether frame contains branding."""
        if not system_prompt:
            system_prompt = (
                "You are a sports broadcast frame classifier. Determine if this frame "
                "contains ANY branded content: logos, jerseys, billboards, sponsor banners.\n\n"
                'Respond with ONLY: {"has_branding": true/false, "confidence": 0.0-1.0}'
            )
        response = self.generate(
            prompt="Analyze this frame for branded content.",
            image=frame,
            system_prompt=system_prompt,
            max_tokens=100,
            temperature=0.0,
            enable_thinking=True,
        )
        return self._parse_classification(response)

    def detect_logos(self, frame: np.ndarray, brands: list[str], system_prompt: str = "") -> list[dict]:
        """Auditor: detect and localize brand logos."""
        brand_list = ", ".join(brands)
        if not system_prompt:
            system_prompt = (
                "You are a professional sports broadcast logo detector. "
                "Locate and identify brand logos with high precision.\n\n"
                f"Target brands: {brand_list}\n\n"
                "Output a JSON array of detections:\n"
                '[{"box_2d": [y1, x1, y2, x2], "label": "BrandName_Location", "confidence": 0.0-1.0}]\n\n'
                "Coordinates: 1000x1000 grid, [y1,x1]=top-left, [y2,x2]=bottom-right.\n"
                "Label format: Brand_Location (e.g., Oracle_Sidepod, Oracle_Helmet, Oracle_Suit)\n\n"
                "Be thorough — check ALL parts of the image:\n"
                "- Cars: sidepods, rear wing, nose, engine cover, front wing\n"
                "- Drivers: helmet, visor, racing suit (chest, shoulders, back)\n"
                "- Trackside: billboards, barriers, pit wall signage\n\n"
                "Only include detections with confidence > 0.5.\n"
                "If no logos are found, output []."
            )
        response = self.generate(
            prompt=f"Find all instances of: {brand_list}. Describe what you see first, then output the JSON array.",
            image=frame,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.0,
            enable_thinking=True,
        )
        return self._parse_detections(response)

    def describe_frame(self, frame: np.ndarray) -> str:
        """Describe what the model sees in a frame (debug)."""
        return self.generate(
            prompt="Describe this image in detail. List every brand logo, text, and sign you can see.",
            image=frame,
            system_prompt="You are an expert image analyst.",
            max_tokens=500,
            temperature=0.0,
            enable_thinking=False,
        )

    def _parse_classification(self, response: str) -> dict:
        response = self._strip_thinking(response)
        try:
            match = re.search(r'\{[^}]+\}', response)
            if match:
                data = json.loads(match.group())
                return {"has_branding": bool(data.get("has_branding", False)),
                        "confidence": float(data.get("confidence", 0.0))}
        except (json.JSONDecodeError, ValueError):
            pass
        lower = response.lower()
        if "true" in lower and "false" not in lower:
            return {"has_branding": True, "confidence": 0.5}
        return {"has_branding": False, "confidence": 0.5}
    def _clean_json(self, text: str) -> str:
        """Fix common JSON errors from model output."""
        # Remove double closing braces: }} -> }
        text = re.sub(r'\}\s*\}', '}', text)
        # Remove trailing commas before ] or }
        text = re.sub(r',\s*([\]}])', r'\1', text)
        return text

    def _parse_detections(self, response: str) -> list[dict]:
        """Parse Auditor detection output."""
        response = self._strip_thinking(response)
        detections = []

        # Try JSON array first
        try:
            match = re.search(r'\[[\s\S]*?\]', response)
            if match:
                cleaned = self._clean_json(match.group())
                data = json.loads(cleaned)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "box_2d" in item:
                            label = str(item.get("label", "Unknown"))[:50]  # Cap label length
                            detections.append({
                                "box_2d": item["box_2d"],
                                "label": label,
                                "confidence": float(item.get("confidence", 0.5)),
                            })
                    return detections
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: individual JSON objects
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
        """Remove thinking blocks and format artifacts from output."""
        # Remove thinking blocks: <|channel>thought...<channel|> or <|channel|>
        # Opening tag: <|channel> (one pipe left), closing varies: <channel|> or <|channel|>
        text = re.sub(r'<\|channel>thought[\s\S]*?(?:<\|channel\|>|<channel\|>)', '', text)
        # Remove code fences
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        # Remove turn markers
        text = re.sub(r'<turn\|>', '', text)
        # Remove any remaining special tokens
        text = re.sub(r'<\|?[^>]+\|?>', '', text)
        return text.strip()


def unload_all():
    import gc
    try:
        import mlx.core as mx
        gc.collect()
        mx.clear_cache()
    except ImportError:
        pass
