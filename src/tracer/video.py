"""Video frame extraction via ffmpeg subprocess.

Supports local files and YouTube URLs (via yt-dlp).
Extracts frames as numpy arrays — no intermediate files written to disk.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np


def is_youtube_url(path: str) -> bool:
    """Check if the input is a YouTube URL."""
    return any(domain in path for domain in [
        "youtube.com/watch",
        "youtu.be/",
        "youtube.com/shorts/",
        "youtube.com/live/",
    ])


def download_youtube(url: str, output_dir: str | Path | None = None) -> Path:
    """Download a YouTube video to a temp file using yt-dlp.

    Args:
        url: YouTube URL.
        output_dir: Optional directory for download. Defaults to system temp.

    Returns:
        Path to the downloaded video file.
    """
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Download best quality mp4, cap at 1080p to keep file size sane
    output_template = str(output_dir / "tracer_yt_%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        url,
    ]

    print(f"[Tracer] Downloading: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    # Find the downloaded file (yt-dlp prints the destination)
    for line in result.stderr.split("\n"):
        if "[download] Destination:" in line:
            path = line.split("Destination:")[-1].strip()
            if Path(path).exists():
                print(f"[Tracer] Downloaded: {path}")
                return Path(path)
        if "has already been downloaded" in line:
            path = line.split("has already been downloaded")[0].replace("[download]", "").strip()
            if Path(path).exists():
                print(f"[Tracer] Already cached: {path}")
                return Path(path)

    # Fallback: search for the file
    for f in output_dir.glob("tracer_yt_*.mp4"):
        print(f"[Tracer] Found downloaded file: {f}")
        return f

    raise RuntimeError("Could not locate downloaded video file")


def resolve_video(input_path: str, output_dir: str | Path | None = None) -> Path:
    """Resolve a video input — handles both local paths and YouTube URLs.

    Args:
        input_path: Local file path or YouTube URL.
        output_dir: Directory for downloaded files (if YouTube).

    Returns:
        Path to the local video file.
    """
    if is_youtube_url(input_path):
        return download_youtube(input_path, output_dir)

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


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
