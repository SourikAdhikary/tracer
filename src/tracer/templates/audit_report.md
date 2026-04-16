# {{ match_id }} — Sponsorship Audit Report

**Video:** {{ video_source }}
**Duration:** {{ duration_seconds }}s
**Brands:** {{ brands_tracked | join(", ") }}
**Date:** {{ timestamp }}

---

## Pipeline Summary

| Metric | Value |
|--------|-------|
| Frames extracted | {{ frames_extracted }} |
| Frames flagged | {{ frames_flagged }} |
| Frames audited | {{ frames_audited }} |
| Total detections | {{ total_detections }} |

---

## Brand Breakdown

{% for brand, stats in brand_stats.items() %}
### {{ brand }}

- **Detections:** {{ stats.count }}
- **Average QoE:** {{ "%.2f"|format(stats.avg_qoe) }}
- **Peak QoE:** {{ "%.2f"|format(stats.top_qoe) }} at {{ stats.top_moment }}

{% endfor %}

---

## Executive Summary

{{ executive_summary or "_Run Analyst phase to generate summary._" }}
