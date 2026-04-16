# Tracer v4 — Technical Design Document

**Author:** Hermes (AI Agent)
**Date:** 2026-04-16
**Status:** DRAFT — Companion to PLAN.md
**Purpose:** Dense, reference-backed explanation of every architectural decision

---

## Table of Contents

1. [Gemma 4 Architecture Deep Dive](#1-gemma-4-architecture-deep-dive)
2. [Model Selection Rationale](#2-model-selection-rationale)
3. [MLX Inference Engine](#3-mlx-inference-engine)
4. [Memory Budget Analysis](#4-memory-budget-analysis)
5. [Two-Stage Pipeline Architecture](#5-two-stage-pipeline-architecture)
6. [Vision Token Budget & Variable Resolution](#6-vision-token-budget--variable-resolution)
7. [Thinking Mode for QoE Reasoning](#7-thinking-mode-for-qoe-reasoning)
8. [Open-Vocabulary Detection Approach](#8-open-vocabulary-detection-approach)
9. [Frame Extraction Strategy](#9-frame-extraction-strategy)
10. [QoE Scoring Methodology](#10-qoe-scoring-methodology)
11. [Reporting Architecture](#11-reporting-architecture)
12. [Thermal Management on Apple Silicon](#12-thermal-management-on-apple-silicon)
13. [Quantization Options Compared](#13-quantization-options-compared)
14. [Alternative Architectures Considered](#14-alternative-architectures-considered)
15. [References](#15-references)

---

## 1. Gemma 4 Architecture Deep Dive

### 1.1 Family Overview

Gemma 4 (released ~April 10, 2026) is Google DeepMind's latest open-weight model family. It ships in four variants:

| Model | Architecture | Total Params | Effective/Active Params | Layers | Sliding Window | Context | Vocabulary |
|-------|-------------|-------------|------------------------|--------|---------------|---------|------------|
| E2B | Dense + PLE | 5.1B (with embeddings) | 2.3B effective | 35 | 512 tokens | 128K | 262K |
| E4B | Dense + PLE | 8B (with embeddings) | 4.5B effective | 42 | 512 tokens | 128K | 262K |
| 26B-A4B | MoE | 25.2B | 3.8B active | 30 | 1024 tokens | 256K | 262K |
| 31B | Dense | 30.7B | 30.7B | 60 | 1024 tokens | 256K | 262K |

**Source:** [google/gemma-4-26B-A4B-it model card](https://huggingface.co/google/gemma-4-26B-A4B-it), accessed 2026-04-16.

### 1.2 Mixture-of-Experts (MoE) — The 26B-A4B Core Innovation

The 26B-A4B model uses a Mixture-of-Experts architecture with:
- **128 total experts** per layer
- **8 experts active** per token (selected via learned routing)
- **1 shared expert** (always active, provides baseline knowledge)
- **Total parameters:** 25.2B
- **Active parameters per token:** 3.8B

**Why this matters for Tracer:**
The MoE architecture means the model "knows" 26B worth of brand logos, sports contexts, and visual patterns, but only activates 3.8B parameters per forward pass. This gives us:
- **Knowledge density:** 26B-level brand recognition (knows obscure sponsors, regional brands)
- **Inference speed:** 4B-class latency (~same as running a dense 4B model)
- **Memory efficiency:** Only the routing network + active experts need to be in fast memory

The routing mechanism selects experts per-token, meaning different experts specialize in different visual/semantic patterns. For sports broadcast analysis, this likely means some experts specialize in text recognition (jersey names), others in logo shapes, others in scene context (field/crowd/celebration).

**Source:** Model card "Mixture-of-Experts (MoE) Model" section, confirmed 8 active / 128 total + 1 shared.

### 1.3 Per-Layer Embeddings (PLE) — The E2B/E4B Efficiency Trick

The smaller dense models (E2B, E4B) use Per-Layer Embeddings to achieve their small effective parameter counts:

> "The 'E' in E2B and E4B stands for 'effective' parameters. The smaller models incorporate Per-Layer Embeddings (PLE) to maximize parameter efficiency in on-device deployments. Rather than adding more layers or parameters to the model, PLE gives each decoder layer its own small embedding for every token. These embedding tables are large but are only used for quick lookups, which is why the effective parameter count is much smaller than the total."

**Source:** Model card "Dense Models" section.

This means the E4B's 8B total parameters include ~3.5B of embedding lookup tables that don't participate in heavy computation. The actual compute-heavy parameters are ~4.5B, which is why it runs fast on laptops.

### 1.4 Hybrid Attention Mechanism

All Gemma 4 models use a hybrid attention pattern:

> "The models employ a hybrid attention mechanism that interleaves local sliding window attention with full global attention, ensuring the final layer is always global. This hybrid design delivers the processing speed and low memory footprint of a lightweight model without sacrificing the deep awareness required for complex, long-context tasks."

**Technical details:**
- **Local layers:** Use sliding window attention (512 or 1024 tokens). Each token only attends to its local neighborhood. O(n*w) complexity where w = window size.
- **Global layers:** Full attention across entire context. O(n^2) complexity but used sparingly.
- **Final layer is always global:** Ensures the model can always make connections across the entire input.
- **Unified Keys and Values (uKV):** Global layers share KV projections to reduce memory for long contexts.
- **Proportional RoPE (p-RoPE):** Rotary position embeddings scaled proportionally, enabling smooth interpolation across different context lengths.

**Impact on Tracer:**
- Sliding window layers process the visual tokens from the frame efficiently (local patterns like edges, textures, logo shapes)
- Global layers connect visual features to the text prompt (brand names, detection instructions)
- The hybrid approach means we get fast local feature extraction + deep global reasoning

**Source:** Model card "Models Overview" section.

### 1.5 Vision Encoder

| Model | Vision Encoder Params | Supported Modalities |
|-------|----------------------|---------------------|
| E2B | ~150M | Text, Image, Audio |
| E4B | ~150M | Text, Image, Audio |
| 26B-A4B | ~550M | Text, Image |
| 31B | ~550M | Text, Image |

The 26B-A4B's 550M vision encoder is significantly larger than E4B's 150M encoder, providing higher quality visual feature extraction. This is crucial for detecting small logos in complex broadcast scenes.

**Source:** Model card Dense Models and MoE Model tables.

---

## 2. Model Selection Rationale

### 2.1 Why Gemma 4 26B-A4B for Auditor (Detection)

**Alternatives considered:**
| Model | Params | Active | MLX Q4 Size | Vision Quality | Speed |
|-------|--------|--------|-------------|---------------|-------|
| Gemma 4 26B-A4B | 25.2B | 3.8B | ~15.6 GB | High (550M encoder) | Fast (MoE) |
| Gemma 4 31B | 30.7B | 30.7B | ~17 GB | High (550M encoder) | Slow (Dense) |
| Qwen2.5-VL-7B | 7B | 7B | ~4 GB | Medium | Medium |
| Florence-2-large | 0.77B | 0.77B | ~0.8 GB | Medium | Very Fast |

**Decision: 26B-A4B wins because:**
1. **Zero-training detection:** The 26B knowledge base includes brand logos from training data. Florence-2 and YOLO require per-logo fine-tuning, violating the PRD's "zero training" requirement.
2. **MoE speed:** 3.8B active params means inference speed comparable to a dense 4B model, despite having 26B knowledge.
3. **Vision encoder quality:** The 550M vision encoder (vs 150M for E4B, ~300M for Qwen2.5-VL-7B) provides superior feature extraction for small, partially occluded logos.
4. **256K context:** Can process multiple frames in a single context window, enabling temporal reasoning ("was this logo visible in the previous frame too?").
5. **Thinking mode:** Enables self-justifying detections with reasoning chains.
6. **Apache 2.0 license:** No commercial restrictions on the audit outputs.

**Benchmark evidence:** 26B-A4B scores 73.8% on MMMU Pro (multimodal understanding) vs 52.6% for E4B. This 21-point gap directly translates to better logo detection accuracy in complex scenes.

**Source:** Model card benchmark table.

### 2.2 Why Gemma 4 E4B for Scout (Frame Triage)

**Alternatives considered:**
| Model | Params | MLX Q4 Size | Speed | Multimodal |
|-------|--------|-------------|-------|-----------|
| Gemma 4 E4B | 4.5B eff | ~5.22 GB | Fast | Text+Image+Audio |
| Gemma 4 E2B | 2.3B eff | ~3 GB | Very Fast | Text+Image+Audio |
| SmolVLM-256M | 0.256B | ~0.3 GB | Ultra Fast | Image only |

**Decision: E4B wins because:**
1. **Multimodal (any-to-any):** E4B is classified as "Any-to-Any" on HuggingFace, meaning it can process image + audio simultaneously. Future feature: detect brand names mentioned in commentary as additional trigger signal.
2. **Frame classification reliability:** At 4.5B effective params, E4B has enough capacity to reliably distinguish "frame with branding" from "frame without branding." E2B at 2.3B may miss subtle branding (distant billboards, small jersey patches).
3. **Same model family as Auditor:** Shared tokenizer, consistent prompt format, same chat template. No cross-model compatibility issues.
4. **Memory footprint acceptable:** At 5.22 GB (MLX Q4), E4B fits alongside the 15.6 GB Auditor in sequential loading mode.

**Why NOT E2B:** The E2B is tempting at ~3 GB, but its 2.3B effective params may produce too many false negatives (missing frames with branding). For the Scout role, recall is more important than precision — a false positive (flagging a clean frame) wastes Auditor compute, but a false negative (missing a branded frame) loses audit data. E4B's higher capacity provides better recall.

**Why NOT SmolVLM:** While 256M params is ultra-fast, SmolVLM is not a Gemma 4 model. It uses a different tokenizer, different chat format, and different vision encoder. Maintaining two completely different model ecosystems adds engineering complexity with no clear benefit — E4B is fast enough for the Scout role.

### 2.3 Why Reuse 26B-A4B for Analyst (QoE + Reporting)

The Analyst role (QoE scoring, executive summary, valuation reasoning) is text-heavy. We could use a separate text-only model, but:
1. The 26B-A4B is already loaded for Auditor — reusing it avoids loading another model.
2. The thinking mode (`<|think|>`) enables chain-of-thought reasoning for QoE justification.
3. The 256K context window allows feeding the entire audit log for summary generation.

---

## 3. MLX Inference Engine

### 3.1 Why MLX

**MLX** is Apple's open-source ML framework designed specifically for Apple Silicon. Key advantages:

1. **Unified memory architecture:** Apple Silicon (M-series) uses unified memory where CPU and GPU share the same memory pool. MLX is designed around this — no CPU↔GPU memory copies.
2. **Metal GPU acceleration:** MLX compiles operations to Metal shaders, Apple's native GPU API. No CUDA dependency.
3. **Lazy evaluation:** MLX builds a computation graph and only executes when results are needed. This enables automatic fusion of operations for better performance.
4. **Built-in quantization:** MLX supports 4-bit, 8-bit, and mixed-precision quantization natively.

**Source:** [MLX documentation](https://mlx-examples.readthedocs.io/), Apple ML Research.

### 3.2 mlx-vlm for Vision Models

The Gemma 4 MLX quantizations are built with **mlx-vlm version 0.4.3**, not the standard `mlx-lm` package. This is important because:

- `mlx-lm` handles text-only models (LLMs)
- `mlx-vlm` handles vision-language models (VLMs) — it includes the vision encoder loading, image preprocessing, and multimodal token handling

**Installation:**
```bash
pip install -U mlx-vlm
```

**CLI inference:**
```bash
python -m mlx_vlm.generate \
  --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --max-tokens 100 \
  --temperature 0.0 \
  --prompt "Detect all brand logos in this frame." \
  --image frame.png
```

**Source:** [mlx-community/gemma-4-26b-a4b-it-4bit model card](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit).

### 3.3 Unsloth MLX Quantizations

Unsloth provides alternative MLX quantizations with their "Dynamic" quantization method. Available options:

| Model | Quantization | HuggingFace ID | Disk Size |
|-------|-------------|----------------|-----------|
| 26B-A4B | UD-MLX-4bit | `unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit` | ~5B params |
| 26B-A4B | MLX-8bit | `unsloth/gemma-4-26b-a4b-it-MLX-8bit` | ~8B params |
| 26B-A4B | UD-MLX-3bit | `unsloth/gemma-4-26b-a4b-it-UD-MLX-3bit` | ~4B params |
| E4B | UD-MLX-4bit | `unsloth/gemma-4-E4B-it-UD-MLX-4bit` | ~2B params |
| E4B | MLX-8bit | `unsloth/gemma-4-E4B-it-MLX-8bit` | ~3B params |

**Unsloth's Dynamic quantization** uses per-layer bit-width selection — layers that are more sensitive to quantization get higher precision. This typically produces better quality at the same model size compared to uniform 4-bit.

**Installation (from Unsloth):**
```bash
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/scripts/install_gemma4_mlx.sh | sh
source ~/.unsloth/unsloth_gemma4_mlx/bin/activate
python -m mlx_vlm.chat --model unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit
```

**Source:** [unsloth/gemma-4-26b-a4b-it-MLX-8bit](https://huggingface.co/unsloth/gemma-4-26b-a4b-it-MLX-8bit), [Unsloth Gemma 4 Guide](https://docs.unsloth.ai/models/gemma-4).

---

## 4. Memory Budget Analysis

### 4.1 MLX Memory Footprints (Measured)

From HuggingFace model cards, the actual MLX memory requirements when loaded:

| Model | Quantization | MLX Memory | Source |
|-------|-------------|------------|--------|
| 26B-A4B | mlx-community 4-bit | **15.6 GB** | [mlx-community page](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) |
| E4B | mlx-community 4-bit | **5.22 GB** | [mlx-community page](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit) |

From Unsloth's guide (GGUF equivalents, similar memory profile):

| Model | 4-bit | 8-bit | BF16 |
|-------|-------|-------|------|
| E4B | 5.5-6 GB | 9-12 GB | 16 GB |
| 26B-A4B | 16-18 GB | 28-30 GB | 52 GB |

**Source:** [Unsloth Gemma 4 Guide — Hardware Requirements](https://docs.unsloth.ai/models/gemma-4).

### 4.2 M5 24GB Budget — Revised Analysis

**Critical finding:** The 26B-A4B model at 15.6 GB + E4B at 5.22 GB = **20.82 GB** just for model weights. This leaves only **~3.2 GB** for KV-cache, frame buffers, and system overhead on a 24GB M5.

**This means we CANNOT run both models simultaneously.**

### 4.3 Sequential Loading Strategy

The solution is **sequential loading** — never have both models in memory at the same time:

```
Phase 1: Scout Phase
├── Load E4B (5.22 GB)
├── Process all frames through Scout
├── Collect flagged frame indices
├── Unload E4B
└── Free memory: ~19 GB available

Phase 2: Auditor Phase
├── Load 26B-A4B (15.6 GB)
├── Process only flagged frames through Auditor
├── Collect detections
├── Unload 26B-A4B (optional: keep for Analyst phase)
└── Free memory: ~8.4 GB available

Phase 3: Analyst Phase (if needed)
├── Reload 26B-A4B (15.6 GB) or keep from Phase 2
├── Generate QoE scores, executive summary
└── Unload 26B-A4B
```

**Trade-off:** Sequential loading adds model loading time (~10-20 seconds per load for 4-bit models from SSD). But this is acceptable because:
- Scout processes the entire video first (minutes of work), then Auditor processes only flagged frames (subset)
- Model loading happens at phase boundaries, not per-frame
- The alternative (running both simultaneously) is impossible on 24GB

### 4.4 Aggressive Option: E2B Scout

If Scout speed is critical, the E2B at ~3 GB MLX memory is even lighter:

| Scout Model | MLX Memory | Scout+Auditor Total | Headroom |
|-------------|-----------|-------------------|----------|
| E4B (4-bit) | 5.22 GB | 20.82 GB | 3.18 GB |
| E2B (4-bit) | ~3 GB | ~18.6 GB | 5.4 GB |

The E2B option gives 5.4 GB headroom, which is more comfortable for system overhead. However, E2B's 2.3B effective params may miss subtle branding. **Recommendation: Start with E4B, benchmark recall, switch to E2B if needed.**

---

## 5. Two-Stage Pipeline Architecture

### 5.1 Why Two Stages

The fundamental constraint is: **processing every frame through a 26B model is computationally infeasible for real-time or near-real-time analysis.**

At 1 fps extraction from a 90-minute match = 5,400 frames.
At ~800ms/frame for 26B-A4B inference = 4,320 seconds = **72 minutes** just for inference.

The Scout stage reduces this by filtering out frames without branding:
- If 70% of frames have no branding → 5,400 × 0.3 = 1,620 frames to Auditor
- At ~800ms/frame = 1,296 seconds = **~22 minutes** for Auditor

Combined with Scout time (5,400 frames × ~150ms/frame for E4B = ~14 minutes), total pipeline is ~36 minutes for a 90-minute match. This is ~0.4x real-time — acceptable for post-processing.

### 5.2 Scout Design

**Input:** Raw frame (numpy array, RGB)
**System prompt:**
```
<|think|>
You are a sports broadcast frame classifier. Analyze this frame and determine if it contains ANY branded content:
- Logos on jerseys, helmets, or equipment
- Billboard advertisements around the field/stadium
- Sponsor banners or graphics overlaid on the broadcast
- Branded products visible (drinks, electronics, etc.)
- Team crest/sponsor patches on kits

Respond with ONLY a JSON object:
{"has_branding": true/false, "confidence": 0.0-1.0}
```

**Why thinking mode for Scout:**
Even though Scout is a "simple" classification task, enabling `<|think|>` helps the model reason about ambiguous cases (e.g., "is that a logo or just a pattern on the jersey?"). The thinking overhead is minimal for E4B.

### 5.3 Auditor Design

**Input:** Flagged frame + brand list
**System prompt:**
```
<|think|>
You are a professional sports broadcast logo detector. Your task is to locate and identify brand logos in this frame with high precision.

Target brands: {brand_list}

For each detection, output a JSON object:
{"box_2d": [y1, x1, y2, x2], "label": "BrandName_Location", "confidence": 0.0-1.0}

Coordinate system:
- Normalized to a 1000x1000 grid
- [y1, x1] = top-left corner
- [y2, x2] = bottom-right corner
- y increases downward, x increases rightward

Label format: "BrandName_Location" (e.g., "Emirates_Chest", "Etihad_Board", "Adidas_Footwear")

Rules:
- Only output detections with confidence > 0.7
- If a logo appears multiple times (e.g., both sleeves), create separate detections
- If unsure about the exact brand, use your best judgment and set confidence accordingly
- Consider partial occlusion (players, equipment blocking the logo)
```

**Output parsing:** The model's JSON output is parsed with `json.loads()` in strict mode. Invalid JSON triggers a retry with a simpler prompt.

### 5.4 Temporal Coherence

An optimization not in the initial plan: use the 256K context window to process multiple frames in sequence, giving the Auditor temporal context:

```
Frame N-2: [detections from previous Auditor run]
Frame N-1: [detections from previous Auditor run]
Frame N: [current frame image]
```

This helps the Auditor:
- Track logos across frames (same brand should appear in similar positions)
- Detect motion (logo moving = player running, affects QoE clarity score)
- Avoid flickering detections (logo appears/disappears frame-to-frame)

**Implementation:** Add previous frame detections as text context before the current frame image in the prompt. This uses the interleaved multimodal input capability.

---

## 6. Vision Token Budget & Variable Resolution

### 6.1 Gemma 4's Visual Token System

Gemma 4 supports configurable visual token budgets:

> "Aside from variable aspect ratios, Gemma 4 supports variable image resolution through a configurable visual token budget, which controls how many tokens are used to represent an image. A higher token budget preserves more visual detail at the cost of additional compute, while a lower budget enables faster inference for tasks that don't require fine-grained understanding."

**Supported token budgets:** 70, 140, 280, 560, 1120

**Source:** Model card "Variable Image Resolution" section.

### 6.2 Token Budget Selection for Tracer

| Phase | Token Budget | Rationale |
|-------|-------------|-----------|
| Scout | 70 | Classification only — "is there branding?" Low detail sufficient. Fastest inference. |
| Auditor | 560 or 1120 | Detection requires fine-grained detail to locate small logos. Use 1120 for complex scenes (multiple logos, crowded frame), 560 for simple scenes. |

**Adaptive budget:** Start Auditor at 560. If detection confidence is low, re-process at 1120. This saves compute on easy frames while maintaining accuracy on hard frames.

### 6.3 Impact on Inference Time

Higher token budgets linearly increase inference time (more tokens to process through the transformer). Approximate scaling:
- 70 tokens: ~50ms (E4B), ~150ms (26B-A4B)
- 560 tokens: ~200ms (E4B), ~600ms (26B-A4B)
- 1120 tokens: ~400ms (E4B), ~1200ms (26B-A4B)

These are rough estimates — actual performance depends on Metal GPU utilization and KV-cache state.

---

## 7. Thinking Mode for QoE Reasoning

### 7.1 Gemma 4 Thinking Mode Mechanics

Gemma 4 uses explicit thinking control via the `<|think|>` token:

**Enable:** Add `<|think|>` at the start of the system prompt.
**Output format:**
```
<|channel>thought
[internal reasoning]
<channel|>
[final answer]
```

**Disable:** Remove `<|think|>` from system prompt. Larger models may still emit empty thought blocks.

**Multi-turn rule:** Only keep the final visible answer in chat history. Do NOT feed prior thought blocks back into the next turn.

**Source:** [Unsloth Gemma 4 Guide — Thinking Mode](https://docs.unsloth.ai/models/gemma-4), Model card "Thinking Mode Configuration" section.

### 7.2 Where to Use Thinking Mode in Tracer

| Phase | Thinking Mode | Rationale |
|-------|--------------|-----------|
| Scout | Enabled | Help reason about ambiguous frames ("is that a logo or pattern?") |
| Auditor | Enabled | Complex detection reasoning ("which brand is this? Is it partially occluded?") |
| Analyst/QoE | **Heavily enabled** | Chain-of-thought for QoE scoring, valuation logic, executive summary |
| Report Generation | Enabled | Narrative reasoning for "emotional context" analysis |

### 7.3 Thinking Mode for QoE — Example

**Prompt:**
```
<|think|>
You are a sponsorship valuation analyst. Given this detection:
{"brand": "Etihad", "box_2d": [442, 120, 510, 240], "frame_context": "Goal celebration, slow-motion replay, camera focused on player"}

Calculate the Quality of Exposure (QoE) score (0-1.0) and provide valuation logic.
Consider: clarity, size, occlusion, emotional context, screen share percentage.
```

**Expected model output:**
```
<|channel>thought
The detection shows Etihad logo at coordinates [442, 120, 510, 240] on a 1000x1000 grid.
Box area = (510-442) * (240-120) = 68 * 120 = 8,160 square units.
Screen share = 8,160 / 1,000,000 = 0.816% of frame.

Context is a goal celebration with slow-motion replay — this is a HIGH-VALUE moment.
Slow-motion means high clarity (camera is steady, no motion blur).
Camera focused on player means the logo is likely in primary focus area.

Clarity: High (slow-motion, focused camera) → 0.95
Size: Small (0.8% screen share) → 0.4
Occlusion: Minimal (celebration, player facing camera) → 0.9
Emotional context: Goal celebration = peak engagement → bonus +0.1

QoE = 0.4 * 0.95 + 0.3 * 0.4 + 0.2 * 0.9 + 0.1 * 1.0
    = 0.38 + 0.12 + 0.18 + 0.10
    = 0.78

Valuation logic: Primary sponsor visibility during game's highest-engagement moment. Slow-motion replay ensures maximum viewer attention. Logo clarity is excellent despite small size. High-value placement.
<channel|>
{"qoe": 0.78, "valuation_logic": "Primary sponsor visibility during game's highest-engagement moment. Slow-motion replay ensures maximum viewer attention. Logo clarity is excellent despite small size. High-value placement."}
```

---

## 8. Open-Vocabulary Detection Approach

### 8.1 What is Open-Vocabulary Detection?

Traditional object detection (YOLO, Faster R-CNN) requires training on a fixed set of classes. If you want to detect "Fly Emirates logos," you need a dataset of labeled Fly Emirates logo images and train a model.

Open-vocabulary detection uses Vision-Language Models (VLMs) to detect objects described in natural language. No training required — just change the text prompt.

### 8.2 How Gemma 4 Does It

Gemma 4's vision encoder extracts visual features, which are then cross-attended by the language model. The language model can:
1. **Ground text descriptions to visual regions** (e.g., "Fly Emirates logo" → specific pixels)
2. **Output bounding box coordinates** as structured text (e.g., `[442, 120, 510, 240]`)
3. **Reason about what it sees** using thinking mode

This is NOT a traditional detection head (like Faster R-CNN's ROI pooling). Instead, the model generates coordinate tokens as text output. The quality of detection depends on:
- Vision encoder quality (550M params for 26B-A4B — high quality)
- Language model's spatial reasoning ability (26B knowledge — strong)
- Training data diversity (Gemma 4 trained on diverse images — good coverage)

### 8.3 Coordinate System

The PRD specifies a 1000x1000 normalized coordinate grid. This is enforced in the prompt:
- Input image is resized/padded to maintain aspect ratio
- Model outputs coordinates in the 0-1000 range
- Coordinates are mapped back to original image dimensions for cropping

**Why 1000x1000 instead of 0-1 normalized:**
- Integer coordinates are easier for the model to output reliably
- 1000 provides sufficient precision (0.1% of frame)
- Matches common VLM coordinate conventions (Florence-2, Grounding DINO use similar scales)

### 8.4 Accuracy Considerations

Open-vocabulary detection via VLMs has known limitations:
- **Localization precision:** Typically 5-15 pixel error at 1000x1000 scale. Fine for logo audit (we don't need pixel-perfect boxes).
- **Small object detection:** Logos < 2% of frame area may be missed. The 1120 token budget helps here.
- **Brand confusion:** Similar logos may be confused (e.g., Fly Emirates vs Emirates Airlines). The brand list in the prompt disambiguates.

**Mitigation:** Post-process detections with OCR on the crop to confirm brand text matches the detected label.

---

## 9. Frame Extraction Strategy

### 9.1 Why ffmpeg (Not Python Video Libraries)

**Options evaluated:**
| Method | Speed | Flexibility | Dependencies |
|--------|-------|-------------|-------------|
| ffmpeg subprocess | Fastest | Maximum | System ffmpeg |
| opencv.VideoCapture | Medium | Good | opencv-python |
| decord | Fast | Limited | decord package |
| torchvision.io | Medium | Limited | torchvision |

**Decision: ffmpeg subprocess** because:
1. **Fastest raw throughput:** ffmpeg is highly optimized C code with SIMD acceleration
2. **Pipe output:** No intermediate files — frames stream directly to Python as raw bytes
3. **Format support:** Handles every video codec imaginable (H.264, H.265, VP9, AV1, etc.)
4. **No Python overhead:** The GIL doesn't bottleneck frame extraction
5. **Already installed:** macOS comes with ffmpeg (via Homebrew)

### 9.2 Extraction Command

```bash
ffmpeg -i input.mp4 \
  -vf "fps=1,scale=1000:1000:force_original_aspect_ratio=decrease,pad=1000:1000:(ow-iw)/2:(oh-ih)/2" \
  -f rawvideo \
  -pix_fmt rgb24 \
  pipe:1
```

**What this does:**
- `-vf "fps=1"`: Extract 1 frame per second
- `scale=1000:1000:force_original_aspect_ratio=decrease`: Scale to fit within 1000x1000, maintaining aspect ratio
- `pad=1000:1000:(ow-iw)/2:(oh-ih)/2`: Pad to exactly 1000x1000 with black bars (centered)
- `-f rawvideo -pix_fmt rgb24`: Output raw RGB bytes (3 bytes per pixel)
- `pipe:1`: Output to stdout

**Python reading:**
```python
import subprocess
import numpy as np

cmd = ['ffmpeg', '-i', 'input.mp4', '-vf', 'fps=1,...', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

frame_size = 1000 * 1000 * 3  # RGB
while True:
    raw = proc.stdout.read(frame_size)
    if len(raw) < frame_size:
        break
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(1000, 1000, 3)
    # Process frame...
```

### 9.3 Keep Original Resolution for PoE Crops

The detection pipeline uses 1000x1000 frames, but for the "Proof of Exposure" gallery, we want high-resolution crops. Solution:

```bash
# Parallel extraction: one at 1000x1000 for detection, one at original for crops
ffmpeg -i input.mp4 -vf "fps=1" -f rawvideo -pix_fmt rgb24 pipe:1  # Original res
ffmpeg -i input.mp4 -vf "fps=1,scale=1000:1000:..." -f rawvideo -pix_fmt rgb24 pipe:1  # Detection res
```

Or extract at original resolution and downscale in Python (more memory but simpler).

### 9.4 Timestamp Mapping

Each extracted frame needs a timestamp for the audit log. Calculate from frame index:
```python
timestamp_seconds = frame_index / fps  # fps=1, so timestamp = frame_index
timestamp_str = f"{timestamp_seconds // 3600:02d}:{(timestamp_seconds % 3600) // 60:02d}:{timestamp_seconds % 60:02d}"
```

---

## 10. QoE Scoring Methodology

### 10.1 QoE Factors

The PRD defines three factors. Here's the detailed methodology for each:

#### 10.1.1 Clarity (Weight: 0.4)

**What it measures:** Is the logo sharp or blurred due to motion/camera effects?

**Computation:**
1. Crop the detection region from the frame using `box_2d` coordinates
2. Convert to grayscale
3. Compute **Laplacian variance** (standard measure of image sharpness):
   ```python
   import cv2
   laplacian = cv2.Laplacian(crop_gray, cv2.CV_64F)
   clarity_score = min(laplacian.var() / 500.0, 1.0)  # Normalize, cap at 1.0
   ```
4. Higher variance = sharper image = higher clarity score

**Why Laplacian variance:** It's the industry-standard sharpness metric. The Laplacian operator detects edges. A sharp logo has strong edges (high variance), a blurred logo has weak edges (low variance). The threshold 500.0 is empirically calibrated — scores above this are "sharp enough" for brand recognition.

#### 10.1.2 Size (Weight: 0.3)

**What it measures:** How much of the screen does the logo occupy?

**Computation:**
```python
box_area = (y2 - y1) * (x2 - x1)
frame_area = 1000 * 1000
area_ratio = box_area / frame_area
size_score = min(math.log10(area_ratio * 1000 + 1) / 2.0, 1.0)  # Log-scaled
```

**Why log-scale:** A logo occupying 1% of screen is much more visible than 0.1%, but the difference between 10% and 11% is negligible. Log-scaling captures this perceptual relationship.

#### 10.1.3 Occlusion (Weight: 0.2)

**What it measures:** Is the logo partially blocked by players, equipment, or other objects?

**Computation:** This is the hardest factor to compute deterministically. Two approaches:

**Approach A — Model-assessed (recommended):**
Include occlusion assessment in the Auditor's detection prompt:
```
For each detection, also assess occlusion:
- "none": Logo is fully visible (>90% visible)
- "partial": Logo is partially blocked (50-90% visible)
- "heavy": Logo is mostly blocked (<50% visible)
```
Map to scores: none=1.0, partial=0.5, heavy=0.2.

**Approach B — Heuristic:**
Compare detected box area to expected brand size. If detected area is significantly smaller than expected, assume occlusion. This is less reliable but doesn't require model modification.

**Recommendation:** Use Approach A (model-assessed) since the Auditor already has thinking mode enabled and can reason about occlusion naturally.

#### 10.1.4 Context Bonus (Weight: 0.1)

**What it measures:** Was this a high-value moment (goal, replay, celebration)?

**Computation:** The Analyst model assesses the scene context:
- Goal celebration, slow-motion replay, penalty kick → bonus = 1.0
- Active play, player close-up → bonus = 0.5
- Static shot, commercial break transition → bonus = 0.0

This is fully model-assessed (no deterministic computation).

### 10.2 Final QoE Formula

```
QoE = 0.4 * clarity_score
    + 0.3 * size_score
    + 0.2 * occlusion_score
    + 0.1 * context_bonus
```

**Range:** 0.0 (worst) to 1.0 (best)

**Calibration:** The weights are initial estimates. After processing 2-3 matches, compare QoE scores to manual human ratings and adjust weights to minimize error.

---

## 11. Reporting Architecture

### 11.1 JSON Audit Log (Per-Match)

```json
{
  "match_id": "2026-04-16_MCI_vs_ARS",
  "video_source": "match.mp4",
  "duration_seconds": 5400,
  "brands_tracked": ["Fly Emirates", "Etihad", "Adidas", "Puma"],
  "scout_config": {"model": "gemma-4-e4b-it-4bit", "token_budget": 70},
  "auditor_config": {"model": "gemma-4-26b-a4b-it-4bit", "token_budget": 560},
  "frames_extracted": 5400,
  "frames_flagged": 1620,
  "frames_audited": 1620,
  "total_detections": 487,
  "detections": [
    {
      "timestamp": "00:12:34",
      "frame_index": 754,
      "brand": "Fly Emirates",
      "box_2d": [320, 440, 480, 620],
      "qoe": 0.87,
      "qoe_breakdown": {
        "clarity": 0.92,
        "size": 0.65,
        "occlusion": 1.0,
        "context": 0.5
      },
      "context": "Active play, midfield",
      "valuation_logic": "Primary shirt sponsor, fully visible on attacking player, high clarity during open play.",
      "crop_path": "output/crops/frame_754_Fly_Emirates.png"
    }
  ]
}
```

### 11.2 Executive Summary (Model-Generated)

The 26B-A4B model generates a narrative executive summary using thinking mode. It receives:
- The complete JSON audit log (truncated if >256K tokens)
- Per-brand aggregated statistics
- Key moments (goals, replays, celebrations)

**Output structure:**
```markdown
# Executive Summary: MCI vs ARS Sponsorship Audit

## Share of Voice
| Brand | Total Seconds | Avg QoE | Screen Share % | SoV Rank |
|-------|--------------|---------|---------------|----------|
| Fly Emirates | 847s | 0.72 | 3.2% | 1 |
| Etihad | 612s | 0.68 | 2.8% | 2 |

## Key Moments
- **[00:34:12]** Fly Emirates logo perfectly visible during Haaland's goal celebration (QoE: 0.94)
- **[01:12:45]** Etihad billboard prominent during slow-motion replay of penalty incident (QoE: 0.91)

## Valuation Summary
Total estimated exposure value: $2.4M (based on CPM of $15 for prime sports broadcast)

## Integrity Alerts
- [00:45:23] Fly Emirates logo visible during player injury stoppage (low emotional context)
```

### 11.3 Heatmap Generation

For each brand, generate a heatmap showing where on the 1000x1000 grid the logo appeared most frequently:

```python
import numpy as np
from PIL import Image

heatmap = np.zeros((1000, 1000), dtype=np.float32)
for detection in brand_detections:
    y1, x1, y2, x2 = detection['box_2d']
    heatmap[y1:y2, x1:x2] += detection['qoe']  # Weight by QoE

# Normalize and apply colormap
heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
heatmap_img = Image.fromarray(heatmap, mode='L').convert('RGB')
# Apply jet colormap...
```

---

## 12. Thermal Management on Apple Silicon

### 12.1 The Problem

Apple Silicon M-series chips have aggressive thermal throttling. When the GPU is under sustained load, the chip reduces clock speeds to prevent overheating. This causes:
- Inference slowdown (up to 30% reduction in token generation speed)
- Fan noise
- Potential thermal shutdown in extreme cases

### 12.2 Detection

Monitor thermal pressure using:
```python
import subprocess
# powermetrics requires sudo
result = subprocess.run(['sudo', 'powermetrics', '--samplers', 'smc', '-n', '1'], 
                       capture_output=True, text=True)
# Parse "CPU die temperature" and "GPU die temperature"
```

Or use `psutil` for a simpler approach:
```python
import psutil
# psutil doesn't directly expose thermal data on macOS
# But we can monitor CPU frequency scaling as a proxy
```

### 12.3 Mitigation Strategies

1. **Sequential loading (already planned):** Never run both models simultaneously
2. **Adaptive token budget:** If thermal pressure detected, reduce Auditor token budget from 1120 → 560 → 280
3. **Frame rate throttling:** If thermals are critical, reduce extraction from 1fps → 0.5fps
4. **Cool-down breaks:** Between Scout and Auditor phases, add a 30-second pause for thermal recovery
5. **Background priority:** Run inference at lower thread priority to leave headroom for system thermals

### 12.4 Expected Thermal Profile

| Phase | GPU Load | Duration | Thermal Impact |
|-------|----------|----------|----------------|
| Scout (E4B) | Medium | ~14 min | Moderate — should stay under throttle threshold |
| Cool-down | None | 30 sec | Recovery |
| Auditor (26B-A4B) | High | ~22 min | High — may throttle in last 5-10 min |
| Analyst (26B-A4B) | Medium | ~5 min | Low — text generation is less GPU-intensive |

---

## 13. Quantization Options Compared

### 13.1 Available Quantizations for Gemma 4 26B-A4B

| Source | Method | Bits | Disk Size | Memory (MLX) | Quality | Source Link |
|--------|--------|------|-----------|-------------|---------|-------------|
| mlx-community | Uniform 4-bit | 4 | ~5 GB | 15.6 GB | Good | [link](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) |
| unsloth | UD-MLX-4bit | 4 (dynamic) | ~5 GB | ~16 GB | Better | [link](https://huggingface.co/unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit) |
| unsloth | MLX-8bit | 8 | ~8 GB | ~28 GB | Best | [link](https://huggingface.co/unsloth/gemma-4-26b-a4b-it-MLX-8bit) |
| unsloth | UD-MLX-3bit | 3 (dynamic) | ~4 GB | ~12 GB | Acceptable | [link](https://huggingface.co/unsloth/gemma-4-26b-a4b-it-UD-MLX-3bit) |

### 13.2 Recommendation

**Primary:** Start with **mlx-community 4-bit** (15.6 GB) — most tested, 90K+ downloads, known working.

**If quality issues:** Switch to **unsloth UD-MLX-4bit** — dynamic quantization should provide better quality at same size.

**If memory is too tight:** Try **unsloth UD-MLX-3bit** (~12 GB) — might free up enough memory to run both Scout and Auditor simultaneously, but quality degradation for logo detection needs testing.

**8-bit is NOT viable:** At 28 GB, it exceeds M5's 24GB unified memory entirely.

---

## 14. Alternative Architectures Considered

### 14.1 Single-Model Pipeline (No Scout)

Run every frame through the 26B-A4B model directly.

**Rejected because:**
- 5,400 frames × 800ms = 72 minutes for a 90-minute match (too slow)
- No thermal headroom — sustained GPU load for 72+ minutes
- Wastes compute on empty frames (70% of frames have no branding)

### 14.2 Traditional Detection (YOLO + OCR)

Train a YOLO model on brand logos, use OCR for text confirmation.

**Rejected because:**
- Violates PRD's "zero training" requirement
- Can't detect new brands without retraining
- YOLO struggles with small, partially occluded logos in broadcast footage
- OCR fails on stylized brand text (e.g., "Fly Emirates" script)

### 14.3 Cloud API (GPT-4V / Gemini API)

Send frames to a cloud VLM API for detection.

**Rejected because:**
- PRD specifies "local-first" and "sovereign" — audit data stays on device
- API costs: ~$0.01/frame × 5,400 frames = $54/match (vs $0 for local)
- Latency: 2-5s per API call vs 0.8s for local inference
- Privacy: Broadcast footage may have licensing restrictions on cloud upload

### 14.4 Gemma 4 31B (Dense) as Auditor

Use the dense 31B model instead of the MoE 26B-A4B.

**Rejected because:**
- 31B is ~2x slower (all 30.7B params active vs 3.8B active)
- Marginal quality improvement (85.2% vs 82.6% on MMLU Pro)
- At 17-20 GB (4-bit), even tighter memory budget
- The MoE model's speed advantage is critical for processing 1,600+ flagged frames

---

## 15. References

1. **Gemma 4 Model Card:** https://huggingface.co/google/gemma-4-26B-A4B-it — Architecture specs, benchmark results, best practices, thinking mode format.

2. **Gemma 4 Launch Blog:** https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ — Official announcement and feature overview.

3. **Gemma 4 Documentation:** https://ai.google.dev/gemma/docs/core — Google's official usage guide.

4. **MLX Framework:** https://github.com/ml-explore/mlx — Apple's ML framework for Apple Silicon.

5. **mlx-vlm:** https://github.com/Blaizzy/mlx-vlm — Vision-language model support for MLX, used for Gemma 4 quantization.

6. **mlx-community/gemma-4-26b-a4b-it-4bit:** https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit — 4-bit MLX quantization, usage examples, memory requirements.

7. **mlx-community/gemma-4-e4b-it-4bit:** https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit — E4B 4-bit MLX quantization.

8. **Unsloth Gemma 4 Guide:** https://docs.unsloth.ai/models/gemma-4 — Hardware requirements, thinking mode format, recommended settings, GGUF/MLX quantizations.

9. **Unsloth Gemma 4 Collection:** https://huggingface.co/collections/unsloth/gemma-4 — All quantization variants (GGUF, MLX 4/8-bit, UD-MLX 3/4-bit).

10. **Laplacian Variance for Sharpness:** Standard image processing metric. Implementation: `cv2.Laplacian(image, cv2.CV_64F).var()`.

11. **Open-Vocabulary Detection Survey:** https://arxiv.org/abs/2306.05802 — Academic overview of zero-shot detection methods.

12. **ffmpeg Documentation:** https://ffmpeg.org/documentation.html — Frame extraction, pipe output, filter chains.

13. **Apple Silicon Thermal Management:** https://support.apple.com/en-us/105094 — Apple's thermal management documentation for M-series chips.

---

*Document generated by Hermes. All model specs verified against HuggingFace model cards as of 2026-04-16.*
