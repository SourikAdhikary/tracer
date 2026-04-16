# Tracer v4 — Autonomous Sponsorship Auditor

## Product Vision
Tracer v4 is a local-first, autonomous audit engine that identifies, localizes, and quantifies brand exposure in sports broadcasts. By using Gemma 4's native vision reasoning, it eliminates the need for per-logo training, allowing brands to audit new logos instantly via text description.

## Technical Infrastructure (M5 Optimized)
- **Primary Model:** Gemma-4-26B-A4B (Mixture of Experts)
  - Total Params: 26B (deep knowledge of brands/context)
  - Active Params: ~3.8B per token (4B-class inference speed on M5)
- **Secondary Model (The Sieve):** Gemma-4-E2B (low-latency frame-skipping)
- **Inference Engine:** MLX with 4-bit quantization (Q4_K_M)
- **VRAM Allocation:** 16GB weights + 4GB context KV-cache + 4GB system/video buffers

## Core Features

### 3.1 Elastic Visual Auditing
- **Scout Loop (E2B):** Monitors stream at 70 tokens/frame (Ultra-Low Res). Triggers Auditor on player/billboard/kit detection.
- **Auditor Loop (26B MoE):** Re-processes flagged events at 1120 tokens/frame (High Res). Uses Open-Vocabulary Detection.
- Prompt-based detection: "Detect the 'Fly Emirates' logo on player chests and 'Adidas' on footwear."
- Output: `{"box_2d": [y1, x1, y2, x2], "label": "Emirates_Primary"}`

### 3.2 Quality of Exposure (QoE) Index
- Score 0-1.0 based on:
  - **Clarity:** Logo blur from motion
  - **Size:** Area relative to 1000x1000 coordinate grid
  - **Occlusion:** Partial blocking by players/equipment

### 4.1 Proof of Exposure (PoE) Gallery
- Auto-crop detections using box_2d coordinates
- Save high-res stills as immutable proof for sponsors

### 4.2 Executive Valuation Summary
- Share of Voice (SoV) analysis
- Emotional context reporting (e.g., logo visible during game-winning goal)

### 5.1 Structured Data (JSON)
- Per-match audit files with timestamp, detections, QoE, context, valuation logic

### 5.2 Sovereign Audit Report (PDF/Dashboard)
- Real-time HUD overlay
- Final Markdown/PDF report with Total Value (USD), Heatmaps, Integrity Alerts

## Non-Vanilla Differentiators
- Zero Training: Swap sponsors via text prompt
- Hardware Sync: Variable resolution budget for thermal management
- Agentic Logic: Self-justifying valuation
