# Tracer v4 — Autonomous Sponsorship Auditor

Local-first audit engine that identifies, localizes, and quantifies brand exposure in sports broadcasts. Zero training — just change a text string to audit new logos instantly.

Built on **Gemma 4** (Google DeepMind) via **MLX** (Apple Silicon). Runs entirely on your Mac.

---

## Quick Start

```bash
cd ~/Projects/tracer

# Audit a YouTube video for Oracle branding
uv run tracer audit --video "https://youtube.com/watch?v=..." --brands "Oracle"

# Audit multiple brands
uv run tracer audit --video match.mp4 --brands "Oracle,Red Bull,Mobil 1,Pirelli"

# View results in browser
uv run tracer serve
```

---

## What It Does

1. **Downloads** video from YouTube (via yt-dlp) or uses a local file
2. **Extracts** frames at 1fps using ffmpeg
3. **Detects** brand logos using Gemma 4 26B-A4B (open-vocabulary, no training needed)
4. **Scores** each detection with Quality of Exposure (QoE) index
5. **Crops** high-res proof images for every detection
6. **Generates** JSON + Markdown reports with Share of Voice analysis
7. **Serves** a visual dashboard showing all detections with heatmaps

---

## Architecture

```
YouTube URL / Local File
        │
        ▼
   yt-dlp download
        │
        ▼
   ffmpeg extract ──► 62 frames (1fps, 1000x1000)
        │
        ▼
   Gemma 4 26B-A4B (MoE, 3.8B active params)
   • Thinking mode enabled
   • 1120 token vision budget
   • Checks: jerseys, helmets, suits, billboards
        │
        ▼
   QoE Scoring (clarity + size + occlusion + context)
        │
        ▼
   Auto-crop + JSON/Markdown Report + Dashboard
```

### Models

| Role | Model | Size (Q4) | Why |
|------|-------|-----------|-----|
| **Auditor** | `gemma-4-26b-a4b-it` | ~15 GB | MoE — 26B knowledge at 3.8B active speed. 550M vision encoder. |
| **Scout** (optional) | `gemma-4-e4b-it` | ~5 GB | Fast frame triage. Not used in current pipeline. |

Both models run via **mlx-vlm** on Apple Silicon Metal GPU.

### Why Gemma 4

- **Zero training:** Open-vocabulary detection — describe logos in text, no dataset needed
- **MoE speed:** 26B-A4B runs at 4B-class inference speed despite 26B knowledge
- **Thinking mode:** Model reasons about detections, justifies its own quality scores
- **Apache 2.0:** No commercial restrictions on audit outputs
- **256K context:** Can process multiple frames for temporal reasoning

---

## Installation

```bash
cd ~/Projects/tracer
uv sync
```

**Requirements:**
- macOS (Apple Silicon M1/M2/M3/M4/M5)
- Python 3.11+
- ffmpeg (`brew install ffmpeg`)
- ~20 GB free disk space for models (downloaded on first run)
- Optional: HuggingFace token for faster downloads

**First run** downloads the Gemma 4 models (~5 GB E4B + ~15 GB 26B-A4B). Subsequent runs use the cache.

---

## Usage

### Audit a video

```bash
# YouTube link
uv run tracer audit --video "https://youtube.com/watch?v=XXXXX" --brands "Oracle"

# Local file
uv run tracer audit --video /path/to/match.mp4 --brands "Oracle,Adidas,Puma"

# Multiple brands from a file (one per line)
uv run tracer audit --video match.mp4 --brands-file brands.txt

# Custom output directory
uv run tracer audit --video match.mp4 --brands "Oracle" --output ./my_audit

# Adjust frame extraction rate
uv run tracer audit --video match.mp4 --brands "Oracle" --fps 2.0
```

### View results

```bash
# Launch dashboard at http://localhost:8080
uv run tracer serve

# Custom port
uv run tracer serve --port 3000

# View a specific audit run
uv run tracer serve --output ./my_audit
```

### Output files

```
output/
├── audit_report.json        # Structured detections with QoE scores
├── audit_report.md          # Human-readable Markdown report
├── dashboard.html           # Visual dashboard (self-contained)
├── crops/                   # Auto-cropped proof images
│   ├── 00-00-18_Oracle_0.png
│   ├── 00-00-22_Oracle_0.png
│   └── ...
└── tracer_yt_*.mp4          # Downloaded video (cached)
```

---

## Reports

### JSON Schema

```json
{
  "match_id": "oracle_run",
  "video_source": "match.mp4",
  "duration_seconds": 62,
  "brands_tracked": ["Oracle"],
  "total_detections": 4,
  "results": [
    {
      "frame_index": 18,
      "timestamp": "00:00:18",
      "detections": [
        {
          "brand": "Oracle",
          "box_2d": [528, 288, 555, 406],
          "confidence": 0.95,
          "qoe": 0.71,
          "qoe_clarity": 1.0,
          "qoe_size": 0.33,
          "qoe_occlusion": 0.8,
          "qoe_context": 0.5,
          "label": "Oracle_Location",
          "crop_path": "output/crops/00-00-18_Oracle_0.png"
        }
      ]
    }
  ]
}
```

### Markdown Report

Includes:
- Pipeline stats (frames scanned, detected)
- Share of Voice (per-brand detection count, avg/peak QoE)
- Top detections with crop proof images
- QoE breakdown (clarity, size, occlusion, context)

---

## QoE Scoring

Each detection gets a 0–1.0 Quality of Exposure score:

| Factor | Weight | Method |
|--------|--------|--------|
| **Clarity** | 0.4 | Laplacian variance (sharpness) |
| **Size** | 0.3 | Log-scaled area ratio on 1000x1000 grid |
| **Occlusion** | 0.2 | Model-assessed visibility |
| **Context** | 0.1 | Emotional value (goal replay = high, static shot = low) |

---

## Dashboard

The visual dashboard (`uv run tracer serve`) shows:

- **Summary stats** — total detections, frames scanned, avg QoE
- **Brand cards** — per-brand detection count, avg/peak QoE
- **Timeline bar** — when detections occurred across the video
- **Heatmap** — where logos appeared on the 1000x1000 grid
- **Detection cards** — each with crop image, timestamp, QoE bar, confidence, bounding box

---

## Project Structure

```
tracer/
├── PRD.md                          # Product Requirements Document
├── PLAN.md                         # Development plan
├── TECHNICAL_DESIGN.md             # Deep-dive technical reference
├── pyproject.toml                  # uv project config
├── src/tracer/
│   ├── cli.py                      # CLI entry point (audit, serve)
│   ├── pipeline.py                 # Orchestrator (5 phases)
│   ├── config.py                   # Model/pipeline/path configuration
│   ├── schemas.py                  # Pydantic data models
│   ├── video.py                    # Frame extraction + YouTube download
│   ├── models/
│   │   └── mlx_backend.py          # Gemma 4 MLX-VLM wrapper
│   ├── auditor.py                  # Logo detection module
│   ├── scout.py                    # Frame triage (optional)
│   ├── qoe.py                      # Quality of Exposure scoring
│   ├── crop.py                     # Auto-crop for PoE gallery
│   ├── report.py                   # JSON + Markdown generation
│   └── templates/                  # Jinja2 report templates
├── tests/                          # Unit tests
├── scripts/                        # Debug/test scripts from development
└── output/                         # Generated reports, crops, dashboard
```

---

## How It Was Built

Developed in a single session with an AI agent (Hermes). Key decisions:

- **Gemma 4 26B-A4B** chosen for its MoE architecture — 26B knowledge at 3.8B active speed
- **Thinking mode** (`<|think|>`) critical for accurate detections — model reasons before outputting JSON
- **mlx-vlm** for Apple Silicon inference — direct Metal GPU access, no CUDA
- **Sequential model loading** required — 26B-A4B takes 15 GB, can't run alongside E4B on 24 GB
- **1120 token vision budget** — max resolution for logo detection
- **3 regex bugs** fixed in thinking block parser (opening tag is `<|channel>` not `<|channel|>`)

---

## Limitations

- **Speed:** ~3 min for a 62-second video (26B-A4B inference on all frames)
- **Small logos:** Logos < 1% of frame may be missed (resolution limit at 1000x1000)
- **False positives:** Occasional hallucinations on non-branded frames — crop validation helps
- **Apple Silicon only:** MLX requires macOS with M-series chip
- **Model size:** ~20 GB download on first run

---

## License

Apache 2.0 (Gemma 4 model license)
