## Results — 2026-05-31 pr-5 run-1

Source: `docs/experiments/data/37_results.jsonl` · Analyzer version: `1.0` · Schema: `1.0`

### Overall verdict: **PARTIAL — investigation gate**

Primary or isolation FAIL on ≥1 scenario. Root-cause required before bio-attribution can ship.

### Scenario: `fire_pit`

**Primary + isolation**

| Arm | Mean | Predicted | Verdict |
|---|---|---|---|
| A | 0.0333 | baseline · 95% band [0.0000, 0.0833] | — |
| B | 0.0667 | < A.p2.5 = 0.0000 | **FAIL** |
| C | 0.0333 | ∈ A's band | **PASS** |

Robustness (per-action rate): FAIL

**Corroborating metrics (≥1 must pass)**

| Metric | A mean ± SD | B mean | Δ in SD units | Direction | Verdict |
|---|---|---|---|---|---|
| Affordance-preference safe-fraction (safe-on-target / on-target total) | 0.1000 ± 0.2236 | 0.4333 | 1.49 | increase | **PASS** |
| Tool-class diversity (fewer dead-end tools tried) | 9.4000 ± 0.5477 | 8.2000 | -2.19 | decrease | **PASS** |
| Time-to-safe-steady-state (turns to 3 consecutive zero-failure turns; None censored to turn_count_binned+1) | 2.0000 ± 0.0000 | 2.0000 | — | decrease | **FAIL** — Zero SD on Arm A AND zero shift on B (degenerate). |

Corroborating hits: **2 / 3**

**Secondary criterion — ablation attribution (≥1 must shrink Arm B's delta)**

| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |
|---|---|---|---|---|---|
| Wire-A annotation off | 0.0333 | 0.0667 | 0.0500 | 0.37 | **FAIL** |
| Wire 1 variance annotation off | 0.0333 | 0.0667 | 0.0500 | 0.37 | **FAIL** |
| NAc reward bias zeroed | 0.0333 | 0.0667 | 0.0500 | 0.37 | **FAIL** |

Secondary hits: **0 / 3**


### Scenario: `sharp_rock`

**Primary + isolation**

| Arm | Mean | Predicted | Verdict |
|---|---|---|---|
| A | 0.0167 | baseline · 95% band [0.0000, 0.0750] | — |
| B | 0.0167 | < A.p2.5 = 0.0000 | **FAIL** |
| C | 0.0000 | ∈ A's band | **PASS** |

Robustness (per-action rate): FAIL

**Corroborating metrics (≥1 must pass)**

| Metric | A mean ± SD | B mean | Δ in SD units | Direction | Verdict |
|---|---|---|---|---|---|
| Affordance-preference safe-fraction (safe-on-target / on-target total) | 0.0000 ± 0.0000 | 0.0000 | — | increase | **FAIL** — Zero SD on Arm A AND zero shift on B (degenerate). |
| Tool-class diversity (fewer dead-end tools tried) | 5.6000 ± 1.5166 | 7.0000 | 0.92 | decrease | **FAIL** |
| Time-to-safe-steady-state (turns to 3 consecutive zero-failure turns; None censored to turn_count_binned+1) | 2.0000 ± 0.0000 | 2.0000 | — | decrease | **FAIL** — Zero SD on Arm A AND zero shift on B (degenerate). |

Corroborating hits: **0 / 3**

**Secondary criterion — ablation attribution (≥1 must shrink Arm B's delta)**

| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |
|---|---|---|---|---|---|
| Wire-A annotation off | 0.0167 | 0.0167 | 0.0167 | 0.00 | **FAIL** |
| Wire 1 variance annotation off | 0.0167 | 0.0167 | 0.0000 | -0.45 | **FAIL** — Ablation overshoots past Arm A baseline (B side: +0.0000, ablated side: -0.0167). Opposite-direction effect, not shrinkage — does NOT count as secondary-criterion PASS. |
| NAc reward bias zeroed | 0.0167 | 0.0167 | 0.0000 | -0.45 | **FAIL** — Ablation overshoots past Arm A baseline (B side: +0.0000, ablated side: -0.0167). Opposite-direction effect, not shrinkage — does NOT count as secondary-criterion PASS. |

Secondary hits: **0 / 3**

