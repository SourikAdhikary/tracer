"""Auditor module — open-vocabulary logo detection using Gemma 4 26B-A4B.

Processes flagged frames at high resolution (560-1120 token budget) to
detect and localize brand logos with bounding boxes.
"""

import numpy as np

from tracer.config import Config
from tracer.models.mlx_backend import Gemma4Model


class Auditor:
    """Logo detector using Gemma 4 26B-A4B MoE."""

    def __init__(self, config: Config):
        self.config = config
        self.model = Gemma4Model(
            model_id=config.models.auditor_model_id,
            token_budget=config.models.auditor_token_budget,
        )

    def load(self) -> None:
        """Load the Auditor model."""
        self.model.load()

    def unload(self) -> None:
        """Free Auditor model memory."""
        self.model.unload()

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect logos in a single frame.

        Args:
            frame: numpy array (H, W, 3) RGB.

        Returns:
            List of detection dicts: {"box_2d", "label", "confidence"}.
        """
        detections = self.model.detect_logos(frame, self.config.brands)

        # Filter by confidence threshold
        return [
            d for d in detections
            if d.get("confidence", 0.0) >= self.config.pipeline.auditor_confidence_threshold
        ]

    def audit_frames(
        self,
        frames: np.ndarray,
        frame_indices: list[int],
        progress_callback=None,
    ) -> dict[int, list[dict]]:
        """Audit flagged frames for logo detections.

        Args:
            frames: numpy array (N, H, W, 3) — all extracted frames.
            frame_indices: List of frame indices flagged by Scout.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            Dict mapping frame_index -> list of detections.
        """
        results = {}
        total = len(frame_indices)

        for i, frame_idx in enumerate(frame_indices):
            detections = self.detect(frames[frame_idx])
            if detections:
                results[frame_idx] = detections

            if progress_callback:
                progress_callback(i + 1, total)

        return results
