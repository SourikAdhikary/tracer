"""Quality of Exposure (QoE) scoring module.

Computes a 0-1.0 quality score for each detection based on:
- Clarity: Laplacian variance (sharpness)
- Size: Area ratio on 1000x1000 grid
- Occlusion: Model-assessed visibility
- Context: Emotional value of the moment
"""

import math

import cv2
import numpy as np

from tracer.config import Config
from tracer.models.mlx_backend import Gemma4Model


class QoEScorer:
    """Quality of Exposure scorer — deterministic + model-assisted."""

    def __init__(self, config: Config, model: Gemma4Model | None = None):
        self.config = config
        self.model = model  # Reuse Auditor's model if available

    def score_detection(
        self,
        frame: np.ndarray,
        detection: dict,
        context: str = "",
    ) -> dict:
        """Compute QoE score for a single detection.

        Args:
            frame: Full frame numpy array (H, W, 3) RGB.
            detection: Detection dict with "box_2d" key.
            context: Optional context string (e.g., "Goal celebration").

        Returns:
            Updated detection dict with qoe scores added.
        """
        box = detection["box_2d"]
        frame_h, frame_w = frame.shape[:2]

        # Scale box_2d from 1000x1000 grid to actual frame coordinates
        y1 = int(box[0] * frame_h / 1000)
        x1 = int(box[1] * frame_w / 1000)
        y2 = int(box[2] * frame_h / 1000)
        x2 = int(box[3] * frame_w / 1000)

        # Clamp to frame bounds
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(y1 + 1, min(y2, frame_h))
        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(x1 + 1, min(x2, frame_w))

        # 1. Clarity — Laplacian variance
        crop = frame[y1:y2, x1:x2]
        clarity = self._compute_clarity(crop)

        # 2. Size — area ratio
        size = self._compute_size(box)

        # 3. Occlusion — model-assessed (default to 0.8 if no model)
        occlusion = self._assess_occlusion(crop, detection)

        # 4. Context bonus
        context_bonus = self._assess_context(context)

        # Weighted combination
        p = self.config.pipeline
        qoe = (
            p.qoe_clarity_weight * clarity
            + p.qoe_size_weight * size
            + p.qoe_occlusion_weight * occlusion
            + p.qoe_context_weight * context_bonus
        )

        # Update detection
        detection["qoe"] = round(min(qoe, 1.0), 3)
        detection["qoe_clarity"] = round(clarity, 3)
        detection["qoe_size"] = round(size, 3)
        detection["qoe_occlusion"] = round(occlusion, 3)
        detection["qoe_context"] = round(context_bonus, 3)

        return detection

    def _compute_clarity(self, crop: np.ndarray) -> float:
        """Compute sharpness via Laplacian variance."""
        if crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize — higher variance = sharper
        return min(variance / self.config.pipeline.laplacian_normalization, 1.0)

    def _compute_size(self, box_2d: list[int]) -> float:
        """Compute size score via log-scaled area ratio."""
        area = (box_2d[2] - box_2d[0]) * (box_2d[3] - box_2d[1])
        ratio = area / 1_000_000  # Relative to 1000x1000 grid

        # Log scaling — diminishing returns for large logos
        return min(math.log10(ratio * 1000 + 1) / 2.0, 1.0)

    def _assess_occlusion(self, crop: np.ndarray, detection: dict) -> float:
        """Assess occlusion level.

        Uses a simple heuristic: compare crop sharpness to expected.
        If the crop is blurry in patches, assume partial occlusion.
        For now, returns a default value. The model can override this
        in the Auditor prompt if occlusion assessment is requested.
        """
        # Default: assume no occlusion
        # A more sophisticated approach would use the model to assess
        # occlusion by asking "is this logo partially blocked?"
        return 0.8

    def _assess_context(self, context: str) -> float:
        """Assess emotional context value.

        High-value moments: goals, celebrations, replays, penalties.
        Medium-value: active play, player close-ups.
        Low-value: static shots, crowd shots.
        """
        if not context:
            return 0.5  # Neutral

        lower = context.lower()

        # High-value keywords
        high_value = ["goal", "celebration", "replay", "slow-motion", "penalty", "winner", "trophy"]
        if any(kw in lower for kw in high_value):
            return 1.0

        # Medium-value keywords
        medium_value = ["attack", "shot", "save", "tackle", "close-up", "substitution"]
        if any(kw in lower for kw in medium_value):
            return 0.5

        return 0.2  # Low-value moment
