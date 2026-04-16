# Tracer v4 — Development Plan

**Author:** Hermes (AI Agent)
**Date:** 2026-04-16
**Status:** DRAFT — Awaiting user review
**Target Hardware:** Mac M5, 24GB Unified Memory

---

## 0. Model Stack: Gemma 4 (Confirmed Available)

Gemma 4 shipped ~April 10, 2026. All models are on HuggingFace with MLX 4-bit quantizations from `mlx-community`.

### Confirmed Model Specs (from HuggingFace model card)

| Model | Architecture | Total Params | Active Params | MLX 4-bit (on disk) | Modalities | Context |
|-------|-------------|-------------|---------------|---------------------|------------|---------|
| **26B-A4B-it** | MoE (8 active / 128 total + 1 shared) | 25.2B | **3.8B** | ~5 GB | Text, Image | 256K |
| **E4B-it** | Dense + PLE | 8B (4.5B effective) | 4.5B | ~2 GB | Text, Image, Audio | 128K |
| **E2B-it** | Dense + PLE | 5.1B (2.3B effective) | 2.3B | ~1 GB | Text, Image, Audio | 128K |
| **31B-it** | Dense | 30.7B | 30.7B | ~5 GB | Text, Image | 256K |

### Tracer Model Assignments

| Role | Model | HuggingFace ID | Why |
|------|-------|----------------|-----|
| **Scout** (frame triage) | Gemma 4 E4B-it (Q4) | `mlx-community/gemma-4-e4b-it-4bit` | 2GB on disk, multimodal (vision+audio), fast ~4.5B active, Any-to-Any capable |
| **Auditor** (detection) | Gemma 4 26B-A4B-it (Q4) | `mlx-community/gemma-4-26b-a4b-it-4bit` | 5GB on disk, MoE runs at 3.8B speed despite 26B knowledge, Image-Text-to-Text, 256K context |
| **Analyst** (QoE, report) | Gemma 4 26B-A4B-it (Q4) | same as Auditor | Reuse loaded weights — thinking mode for valuation reasoning, report generation |

### Memory Budget (24GB M5) — Sequential Loading Required

**Critical finding:** MLX memory footprint is higher than raw model size.
- 26B-A4B Q4: **15.6 GB** in MLX (not ~5 GB on-disk)
- E4B Q4: **5.22 GB** in MLX
- Combined: 20.8 GB — leaves only 3.2 GB headroom

**Solution: Sequential loading — never both models in memory simultaneously.**

| Phase | Model Loaded | Memory Used | Free |
|-------|-------------|-------------|------|
| Scout | E4B Q4 | ~6 GB | ~18 GB |
| Cool-down | None | ~2 GB | ~22 GB |
| Auditor | 26B-A4B Q4 | ~17 GB | ~7 GB |
| Analyst | 26B-A4B Q4 | ~17 GB | ~7 GB |

Model loading adds ~10-20s per swap from SSD. Acceptable since it happens at phase boundaries.

**Source:** [mlx-community/gemma-4-26b-a4b-it-4bit](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit), [Unsloth Gemma 4 Guide](https://docs.unsloth.ai/models/gemma-4)

**Key insight from model card:** The 26B-A4B MoE "runs almost as fast as a 4B-parameter model" because only 3.8B params are active per token. This is why the PRD's architecture works on M5 — you get 26B knowledge at 4B inference speed.

---

## 1. Project Structure

```
tracer/
├── PRD.md                  # Original PRD (you provided)
├── PLAN.md                 # This file
├── pyproject.toml          # uv-managed dependencies
├── src/
│   ├── __init__.py
│   ├── config.py           # Paths, model names, thresholds
│   ├── pipeline.py         # Main orchestrator (Scout -> Auditor -> Analyst)
│   ├── scout.py            # Frame triage — fast model, skip/trigger logic
│   ├── auditor.py          # Open-vocab detection — bounding boxes + labels
│   ├── analyst.py          # QoE scoring, valuation, report generation
│   ├── video.py            # Frame extraction, ffmpeg wrapper
│   ├── crop.py             # Auto-crop from box_2d coords
│   ├── report.py           # JSON audit + Markdown/PDF report builder
│   ├── hud.py              # Real-time transparent overlay (optional)
│   └── models/
│       ├── __init__.py
│       ├── base.py         # Abstract model interface
│       ├── mlx_backend.py  # MLX inference (quantized loading)
│       └── prompts.py      # System/user prompts for each role
├── templates/
│   ├── audit_report.md     # Markdown report template
│   └── executive_summary.md
├── output/                 # Generated audits, crops, reports
├── tests/
│   ├── test_scout.py
│   ├── test_auditor.py
│   └── test_pipeline.py
└── README.md
```

---

## 2. Development Phases

### Phase 0: Foundation (Day 1)
- [ ] `uv init tracer` + pyproject.toml with deps
- [ ] `config.py` — model paths, resolution settings, VRAM budget
- [ ] `models/base.py` — abstract interface (load, infer, unload)
- [ ] `models/mlx_backend.py` — MLX quantized model loader
- [ ] `video.py` — ffmpeg frame extraction at configurable FPS

**Key decisions:**
- Use `ffmpeg` (via subprocess) for frame extraction — fastest, most flexible for 4K streams
- MLX quantization: Q4_K_M for 7B models, FP16 for 256M scout
- Frame extraction as numpy arrays, not saved to disk (I/O bottleneck)

### Phase 1: Scout Loop (Day 2)
- [ ] `scout.py` — Gemma 4 E4B-it frame classifier
- [ ] Input: raw frame (numpy array) + system prompt asking "does this frame contain any branded content (logos, jerseys, billboards)?"
- [ ] Output: `{"has_branding": bool, "confidence": float, "frame_idx": int}`
- [ ] Skip rate: process 1 in N frames (tunable, start at 1 in 5)
- [ ] Temperature management: throttle if thermal pressure detected
- [ ] E4B supports audio natively — future: detect commentary mentioning brand names as trigger

**Architecture:**
```
Video Stream
    |
    v
[Frame Extractor] --1fps--> [Scout (E4B, 4.5B)] --flagged frames--> [Auditor (26B-A4B, 3.8B active)]
                                       |
                                  skip (no branding)
```

### Phase 2: Auditor Loop (Day 3-4)
- [ ] `auditor.py` — Gemma 4 26B-A4B-it open-vocabulary detection
- [ ] Input: flagged frame + brand prompt ("Detect Fly Emirates on jerseys")
- [ ] Output: list of `{"box_2d": [y1,x1,y2,x2], "label": str, "confidence": float}`
- [ ] Coordinate system: normalize to 1000x1000 grid (as PRD specifies)
- [ ] Batch mode: accumulate flagged frames, process in batches to maximize GPU utilization
- [ ] Leverage Gemma 4's native `<|think|>` mode for complex scenes (multiple overlapping logos)

**Prompt template:**
```
You are a sports broadcast logo detector. Analyze this frame and detect the following brands:
{brand_list}

For each detection, output JSON:
{"box_2d": [y1, x1, y2, x2], "label": "BrandName_Location", "confidence": 0.0-1.0}

Coordinates are on a 1000x1000 grid (y1=top, x1=left, y2=bottom, x2=right).
Only output detections you are confident about (>0.7 confidence).
```

### Phase 3: QoE & Analyst (Day 5)
- [ ] `analyst.py` — QoE scoring + valuation reasoning
- [ ] Input: frame + detection box + context (game time, previous events)
- [ ] QoE factors:
  - Clarity: compute Laplacian variance inside crop region (sharpness metric)
  - Size: (box_area / frame_area) ratio
  - Occlusion: heuristic (compare box area to expected brand size)
- [ ] Output: `{"qoe": 0.0-1.0, "context": str, "valuation_logic": str}`

**QoE scoring formula (deterministic + model-assisted):**
```
qoe = 0.4 * clarity_score + 0.3 * size_score + 0.2 * occlusion_score + 0.1 * context_bonus
```
- clarity_score: Laplacian variance (normalized)
- size_score: area ratio (log-scaled)
- occlusion_score: model-assessed visibility
- context_bonus: +0.1 if during goal/replay/high-value moment (model-detected)

### Phase 4: Reporting (Day 6-7)
- [ ] `crop.py` — PIL-based auto-crop from box_2d coords
- [ ] `report.py` — JSON audit log + Markdown report
- [ ] JSON schema per PRD spec
- [ ] Markdown report sections:
  - Executive Summary (model-generated narrative)
  - Per-brand breakdown (total seconds, avg QoE, top moments)
  - Proof gallery (embedded crop images)
  - Share of Voice comparison
  - Integrity alerts (negative-event flagging)
- [ ] Optional: PDF export via `weasyprint` or `mdpdf`

### Phase 5: HUD & Polish (Day 8+)
- [ ] `hud.py` — transparent window overlay using `pygame` or `tkinter`
- [ ] Real-time display: current QoE score, active detections, brand names
- [ ] CLI interface: `tracer audit --video match.mp4 --brands "Fly Emirates,Etihad,Adidas"`
- [ ] Thermal monitoring: use `sudo powermetrics` or `psutil` to detect throttling
- [ ] Adaptive resolution: drop frame resolution if thermal pressure detected

---

## 3. Key Technical Decisions

### 3.1 Why MLX (not llama.cpp or PyTorch)
- Native Apple Silicon optimization — 2-3x faster than PyTorch on M-series
- Built-in 4-bit quantization (Q4_K_M) with minimal quality loss
- **Gemma 4 26B-A4B already has mlx-community 4-bit quantization** (~5GB on disk, 3.8B active per token)
- Direct Metal GPU access — no CUDA dependency

### 3.2 Why Two-Stage Pipeline (Scout -> Auditor)
- Processing every frame at 26B resolution = unsustainable memory bandwidth
- Scout (E4B, 4.5B active) is ~3-5x faster than Auditor (26B-A4B, 3.8B active but MoE routing overhead)
- Expected skip rate: 60-80% of frames have no branding = massive speedup
- Thermal benefit: Scout keeps GPU cool between Auditor bursts
- Both models are Gemma 4 — shared tokenizer, consistent prompt format, thinking mode on both

### 3.3 Why Not YOLO/Traditional Detection
- PRD explicitly says "Zero Training" — no per-logo model training
- Open-vocabulary detection via VLMs handles new brands instantly
- Gemma 4's native vision reasoning + `<|think|>` mode = self-justifying detections

### 3.4 Frame Extraction Strategy
- **Live stream:** Extract at 1fps (Scout processes 1 frame/sec)
- **Post-processing:** Extract at 2fps for higher coverage
- **4K input, 1000x1000 processing:** Downscale for detection, keep original for crop
- Use `ffmpeg` pipe: `ffmpeg -i input.mp4 -vf fps=1 -f rawvideo -pix_fmt rgb24 pipe:1`

---

## 4. Dependency Stack

```toml
[project]
name = "tracer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mlx>=0.25.0",
    "mlx-lm>=0.25.0",
    "numpy>=1.26",
    "pillow>=10.0",
    "opencv-python-headless>=4.9",
    "rich>=13.0",           # CLI output
    "pydantic>=2.0",        # Data validation
    "jinja2>=3.1",          # Report templates
]
# Note: Frame extraction uses ffmpeg subprocess (no python binding needed)
# Gemma 4 MLX models downloaded via `huggingface-cli` at first run

[project.optional-dependencies]
hud = ["pygame>=2.5"]
pdf = ["weasyprint>=62.0"]
```

---

## 5. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| MLX MoE inference slower than expected | High | Benchmark early (Phase 0); fallback to E4B for both Scout+Auditor if needed |
| 26B-A4B Q4 quality loss for detection | Medium | Test against ground truth in Phase 2; unsloth UD-MLX or 8-bit variant available |
| 24GB tight with both models loaded | High | Sequential loading: never both in memory. E2B Scout as fallback if E4B too large. |
| Open-vocab detection inaccurate | Medium | Ensemble: combine VLM boxes with OCR for brand text confirmation |
| Thermal throttling on M5 | Low | Adaptive resolution + frame rate throttling |

---

## 6. Testing Strategy

1. **Unit tests:** Each module independently (mock model outputs)
2. **Integration test:** Process 60-second clip with known brands, verify detections
3. **Benchmark:** Measure frames/sec at Scout and Auditor levels
4. **Accuracy test:** Manual ground-truth labeling on 100 frames, compute mAP
5. **Thermal test:** Run 90-minute match, monitor temp + throttle events

---

## 7. Milestones

| Milestone | Deliverable | ETA |
|-----------|-------------|-----|
| M0: Foundation | Repo structure, model loading, frame extraction | Day 1 |
| M1: Scout Working | Frame triage with Gemma 4 E4B, skip/trigger logic | Day 2 |
| M2: Auditor Working | Open-vocab detection with Gemma 4 26B-A4B, bounding boxes | Day 4 |
| M3: Full Pipeline | Scout->Auditor->QoE->JSON output end-to-end | Day 5 |
| M4: Reporting | Markdown/PDF reports, PoE gallery, executive summary | Day 7 |
| M5: Polish | HUD, CLI, thermal management, README | Day 8+ |

---

## Next Steps (Pending Your Review)

1. Approve this plan (or request changes)
2. I'll create the repo structure and start Phase 0
3. First working demo = M2 (Auditor detecting logos in a test clip)

---

*Plan written by Hermes — ready for review.*
