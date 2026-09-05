# P4 Option 2 Measurement Results

**Date:** 2026-04-16 22:04
**Decision:** defer

## Summary

- **Seeds:** 10
- **Classes:** 10, 5 vision nodes each
- **Bridge concepts:** plant, garden, blossom
- **Class weight:** 0.7 (5 episodes)
- **Bridge weight:** 0.3 (1 episode)
- **EC threshold base:** 0.6 (±0.02)

## Aggregate Results

| Metric | Value |
|---|---|
| Mean single-hop recall (Option 1) | 1.0000 |
| Mean multi-hop recall (Option 2) | 1.0000 |
| **Mean Option 2 lift** | **0.0000 ± 0.0000** |
| Random baseline recall | 0.0900 |
| Seeds with EC merge | 10/10 |

## Decision

**defer:** Option 2 lift is negligible (0.0000 ± 0.0000). Multi-hop paths exist but same-class activation (0.490) dominates cross-class activation at top-5 ranking. Defer Option 2 as cleanup. Ship Stage 3 on single-hop. EC merge observation: 10/10 seeds had at least one bridge concept merge (check per-seed EC decisions to distinguish bridge-bridge vs bridge-class merges).

## Per-Seed Results

| Seed | EC thresh | Hebb init | Single-hop | Multi-hop | Lift | Random | EC decisions |
|---|---|---|---|---|---|---|---|
| 0 | 0.6055 | 0.2908 | 1.0000 | 1.0000 | 0.0000 | 0.1400 | plant=separated, garden=merged:95a04da4-8d12-4e26-827e-3a52a07f9713, blossom=separated |
| 1 | 0.6005 | 0.3180 | 1.0000 | 1.0000 | 0.0000 | 0.0600 | plant=separated, garden=merged:3ba14de3-0371-4876-add3-9e10eef84708, blossom=separated |
| 2 | 0.5905 | 0.2919 | 1.0000 | 1.0000 | 0.0000 | 0.0600 | plant=separated, garden=merged:4ef99157-1fe3-4043-a9d7-fafeaa6c639b, blossom=separated |
| 3 | 0.5834 | 0.2895 | 1.0000 | 1.0000 | 0.0000 | 0.1000 | plant=separated, garden=merged:96799db1-283f-4410-be94-c3efa13203d0, blossom=separated |
| 4 | 0.6177 | 0.3005 | 1.0000 | 1.0000 | 0.0000 | 0.1400 | plant=separated, garden=merged:e68d1b0e-7dd8-47b1-b4f1-772acee93145, blossom=separated |
| 5 | 0.6122 | 0.3123 | 1.0000 | 1.0000 | 0.0000 | 0.0800 | plant=separated, garden=merged:0577aa64-f400-458f-9ade-ffbabdb0a929, blossom=separated |
| 6 | 0.6015 | 0.2937 | 1.0000 | 1.0000 | 0.0000 | 0.0400 | plant=separated, garden=merged:2e307f22-f310-48c1-b2ae-7c23bad330b0, blossom=separated |
| 7 | 0.6050 | 0.3159 | 1.0000 | 1.0000 | 0.0000 | 0.0400 | plant=separated, garden=merged:24ce76c8-8ef3-4f5b-ad6d-5a14296227d3, blossom=separated |
| 8 | 0.5931 | 0.3195 | 1.0000 | 1.0000 | 0.0000 | 0.1600 | plant=separated, garden=merged:6b1da408-37a8-4845-8859-b502c654b261, blossom=separated |
| 9 | 0.6148 | 0.2915 | 1.0000 | 1.0000 | 0.0000 | 0.0800 | plant=separated, garden=merged:755e0338-78c6-453e-826d-aa6b10bf3113, blossom=separated |

## Per-Class Detail (seed 0)

| Class | Single-hop | Multi-hop | Lift | Cross-class in top-5 |
|---|---|---|---|---|
| mexican petunia | 1.00 | 1.00 | 0.00 | 0 |
| water lily | 1.00 | 1.00 | 0.00 | 0 |
| orange dahlia | 1.00 | 1.00 | 0.00 | 0 |
| pincushion flower | 1.00 | 1.00 | 0.00 | 0 |
| azalea | 1.00 | 1.00 | 0.00 | 0 |
| oxeye daisy | 1.00 | 1.00 | 0.00 | 0 |
| balloon flower | 1.00 | 1.00 | 0.00 | 0 |
| lotus | 1.00 | 1.00 | 0.00 | 0 |
| fritillary | 1.00 | 1.00 | 0.00 | 0 |
| morning glory | 1.00 | 1.00 | 0.00 | 0 |

## Analysis

### Why lift is zero

The activation math predicted this outcome. With `RetrievalConfig` defaults (`decay=0.7, threshold=0.001, max_depth=5`):

- **Same-class path** `text_X → vision_X_i`: activation = 1.0 × 0.7 × 0.7 = **0.490**
- **Cross-class path** `text_X → text_bridge → text_Y → vision_Y_i`: activation = 1.0 × 0.7 × 0.3 × 0.7 × 0.3 × 0.7 × 0.7 = **0.0216**

Same-class activation dominates cross-class by 22:1. In top-5 retrieval, all 5 same-class vision nodes (at 0.490 each) rank above all 45 cross-class vision nodes (at ~0.022 each). Option 2 adds breadth (cross-class nodes ARE reachable) but cannot improve top-5 precision.

### EC merge of 'garden' with 'plant'

The bridge concept 'garden' was pattern-completed to the 'plant' bridge node (cosine ~0.716 in paraphrase-mpnet space) on all seeds. This is a **bridge-bridge merge**, not a bridge-class merge — the pre-screening verified 'garden' stays below 0.60 for all 10 class names. The merge reduces the bridge layer from 3 concepts to 2 ('plant' + 'blossom') but does not invalidate the measurement since 2/3 concepts still separate and create honest bridge topology.

## Reproduction

```bash
PYTHONPATH=src python scripts/p4_option2_measurement.py  # D27: add --write-experiment-results to update the committed record
```

Requires `sentence-transformers` (`pip install sentence-transformers`).
No GPU required.
