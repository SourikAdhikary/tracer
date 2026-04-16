"""Crop module — auto-crop detection regions for Proof of Exposure gallery.

Uses box_2d coordinates to extract high-resolution stills of each detection.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def crop_detection(
    frame: np.ndarray,
    box_2d: list[int],
    output_path: str | Path,
    padding: int = 20,
) -> str:
    """Crop a detection region from a frame and save as PNG.

    Args:
        frame: numpy array (H, W, 3) RGB.
        box_2d: [y1, x1, y2, x2] on 1000x1000 grid.
        output_path: Where to save the crop.
        padding: Extra pixels around the crop (in 1000x1000 units).

    Returns:
        Path to saved crop file.
    """
    frame_h, frame_w = frame.shape[:2]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Scale from 1000x1000 to actual frame coordinates
    scale_y = frame_h / 1000
    scale_x = frame_w / 1000

    y1 = int((box_2d[0] - padding) * scale_y)
    x1 = int((box_2d[1] - padding) * scale_x)
    y2 = int((box_2d[2] + padding) * scale_y)
    x2 = int((box_2d[3] + padding) * scale_x)

    # Clamp to frame bounds
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(frame_h, y2)
    x2 = min(frame_w, x2)

    # Crop and save
    crop = frame[y1:y2, x1:x2]
    img = Image.fromarray(crop)
    img.save(output_path, format="PNG")

    return str(output_path)


def crop_all_detections(
    frames: np.ndarray,
    detections_by_frame: dict[int, list[dict]],
    crops_dir: str | Path,
    frame_timestamps: dict[int, str] | None = None,
) -> dict[int, list[dict]]:
    """Crop all detections and add crop_path to each detection.

    Args:
        frames: numpy array (N, H, W, 3).
        detections_by_frame: Dict mapping frame_index -> list of detections.
        crops_dir: Directory to save crops.
        frame_timestamps: Optional dict mapping frame_index -> timestamp string.

    Returns:
        Updated detections_by_frame with crop_path added to each detection.
    """
    crops_dir = Path(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, detections in detections_by_frame.items():
        ts = frame_timestamps.get(frame_idx, f"frame_{frame_idx}") if frame_timestamps else f"frame_{frame_idx}"
        ts_safe = ts.replace(":", "-")

        for i, det in enumerate(detections):
            brand_safe = det.get("label", det.get("brand", "unknown")).replace(" ", "_")[:30]
            filename = f"{ts_safe}_{brand_safe}_{i}.png"
            output_path = crops_dir / filename

            det["crop_path"] = crop_detection(
                frames[frame_idx],
                det["box_2d"],
                output_path,
            )

    return detections_by_frame
