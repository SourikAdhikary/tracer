"""Data models for Tracer audit pipeline."""

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """A single logo detection."""

    brand: str
    box_2d: list[int] = Field(..., min_length=4, max_length=4, description="[y1, x1, y2, x2] on 1000x1000 grid")
    confidence: float = Field(..., ge=0.0, le=1.0)
    label: str = ""  # e.g. "Emirates_Chest"
    qoe: float = Field(default=0.0, ge=0.0, le=1.0)
    qoe_clarity: float = 0.0
    qoe_size: float = 0.0
    qoe_occlusion: float = 0.0
    qoe_context: float = 0.0
    context: str = ""
    valuation_logic: str = ""
    crop_path: str = ""


class FrameResult(BaseModel):
    """Result of processing a single frame."""

    frame_index: int
    timestamp: str  # HH:MM:SS
    scout_has_branding: bool = False
    scout_confidence: float = 0.0
    detections: list[Detection] = Field(default_factory=list)


class AuditReport(BaseModel):
    """Complete audit report for a video."""

    match_id: str = ""
    video_source: str = ""
    duration_seconds: float = 0.0
    brands_tracked: list[str] = Field(default_factory=list)

    # Config snapshots
    scout_model: str = ""
    auditor_model: str = ""
    scout_token_budget: int = 0
    auditor_token_budget: int = 0

    # Stats
    frames_extracted: int = 0
    frames_flagged: int = 0
    frames_audited: int = 0
    total_detections: int = 0

    # Results
    results: list[FrameResult] = Field(default_factory=list)

    # Generated summary
    executive_summary: str = ""
