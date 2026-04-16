"""Video frame extraction via ffmpeg subprocess.

Extracts frames as numpy arrays — no intermediate files written to disk.
"""

import subprocess
from pathlib import Path

import numpy as np


def extract_frames(
    video_path: str | Path,
    fps: float = 1.0,
    frame_size: int = 1000,
) -> tuple[np.ndarray, float]:
    """Extract frames from a video file.

    Args:
        video_path: Path to the video file.
        fps: Frames per second to extract.
        frame_size: Target frame size (square, frame_size x frame_size).

    Returns:
        Tuple of:
            frames: numpy array of shape (N, frame_size, frame_size, 3), dtype uint8, RGB
            duration: video duration in seconds
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get video duration
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())

    # Extract frames as raw RGB bytes via pipe
    extract_cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", (
            f"fps={fps},"
            f"scale={frame_size}:{frame_size}:force_original_aspect_ratio=decrease,"
            f"pad={frame_size}:{frame_size}:(ow-iw)/2:(oh-ih)/2:color=black"
        ),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-loglevel", "error",
        "pipe:1",
    ]

    proc = subprocess.Popen(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_bytes = frame_size * frame_size * 3  # RGB
    frames_list = []

    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(frame_size, frame_size, 3)
        frames_list.append(frame)

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {stderr}")

    if not frames_list:
        raise RuntimeError(f"No frames extracted from {video_path}")

    frames = np.stack(frames_list, axis=0)
    return frames, duration


def extract_frames_original_res(
    video_path: str | Path,
    fps: float = 1.0,
) -> tuple[list[np.ndarray], float]:
    """Extract frames at original resolution (for high-res PoE crops).

    Returns:
        Tuple of:
            frames: list of numpy arrays (H, W, 3), varying sizes
            duration: video duration in seconds
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get duration
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())

    # Get video dimensions
    dim_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(video_path),
    ]
    dim_result = subprocess.run(dim_cmd, capture_output=True, text=True, check=True)
    width, height = map(int, dim_result.stdout.strip().split("x"))

    # Extract at original resolution
    extract_cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-loglevel", "error",
        "pipe:1",
    ]

    proc = subprocess.Popen(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_bytes = width * height * 3
    frames_list = []

    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        frames_list.append(frame)

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {stderr}")

    return frames_list, duration


def frame_timestamp(frame_index: int, fps: float = 1.0) -> str:
    """Convert frame index to HH:MM:SS timestamp."""
    seconds = frame_index / fps
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
