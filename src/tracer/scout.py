"""Scout module — fast frame triage using Gemma 4 E4B.

Processes frames at low resolution (70 token budget) to determine
which frames contain branding. Only flagged frames are sent to the
slower Auditor model.
"""

import numpy as np

from tracer.config import Config
from tracer.models.mlx_backend import Gemma4Model


class Scout:
    """Frame classifier using Gemma 4 E4B."""

    def __init__(self, config: Config):
        self.config = config
        self.model = Gemma4Model(
            model_id=config.models.scout_model_id,
            token_budget=config.models.scout_token_budget,
        )

    def load(self) -> None:
        """Load the Scout model."""
        self.model.load()

    def unload(self) -> None:
        """Free Scout model memory."""
        self.model.unload()

    def classify(self, frame: np.ndarray) -> dict:
        """Classify a single frame.

        Args:
            frame: numpy array (H, W, 3) RGB.

        Returns:
            Dict with "has_branding" (bool) and "confidence" (float).
        """
        return self.model.classify_frame(frame)

    def scan(
        self,
        frames: np.ndarray,
        progress_callback=None,
    ) -> list[int]:
        """Scan all frames and return indices of flagged frames.

        Args:
            frames: numpy array (N, H, W, 3) — all extracted frames.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            List of frame indices that were flagged as containing branding.
        """
        flagged_indices = []
        total = len(frames)

        for i in range(total):
            result = self.classify(frames[i])
            has_branding = result.get("has_branding", False)
            confidence = result.get("confidence", 0.0)

            if has_branding and confidence >= self.config.pipeline.scout_confidence_threshold:
                flagged_indices.append(i)

            if progress_callback:
                progress_callback(i + 1, total)

        return flagged_indices
