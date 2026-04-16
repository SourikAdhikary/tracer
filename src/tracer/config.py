"""Tracer configuration — paths, model IDs, thresholds."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Model identifiers and quantization settings."""

    # Scout model — frame triage (E4B, fast, multimodal)
    scout_model_id: str = "mlx-community/gemma-4-e4b-it-4bit"
    scout_token_budget: int = 70  # Ultra-low res for classification

    # Auditor model — open-vocab detection (26B-A4B MoE, high quality)
    auditor_model_id: str = "mlx-community/gemma-4-26b-a4b-it-4bit"
    auditor_token_budget: int = 560  # High res for detection
    auditor_token_budget_high: int = 1120  # Max res fallback

    # Analyst — reuse Auditor weights, thinking mode for QoE/reports
    # No separate model needed


@dataclass
class PipelineConfig:
    """Pipeline behavior settings."""

    # Frame extraction
    extraction_fps: float = 1.0  # Frames per second
    frame_size: int = 1000  # 1000x1000 normalized grid

    # Scout thresholds
    scout_confidence_threshold: float = 0.5  # Flag frame if >= this

    # Auditor thresholds
    auditor_confidence_threshold: float = 0.7  # Only keep detections >= this

    # QoE weights
    qoe_clarity_weight: float = 0.4
    qoe_size_weight: float = 0.3
    qoe_occlusion_weight: float = 0.2
    qoe_context_weight: float = 0.1

    # Laplacian sharpness normalization (higher = sharper = better clarity)
    laplacian_normalization: float = 500.0

    # Sampling parameters (from Gemma 4 best practices)
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_tokens: int = 1024


@dataclass
class PathConfig:
    """File system paths."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "output")
    crops_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "output" / "crops")
    templates_dir: Path = field(default_factory=lambda: Path(__file__).parent / "templates")


@dataclass
class Config:
    """Top-level configuration."""

    models: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # Brand list — set at runtime via CLI
    brands: list[str] = field(default_factory=list)

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.crops_dir.mkdir(parents=True, exist_ok=True)
