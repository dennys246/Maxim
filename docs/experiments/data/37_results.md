## Results — 2026-06-06 Qwen14B fire

Source: `docs/experiments/data/37_results.jsonl` · Analyzer version: `1.0` · Schema: `1.0`

### Overall verdict: **PARTIAL — investigation gate**

Primary or isolation FAIL on ≥1 scenario. Root-cause required before bio-attribution can ship.

### Scenario: `fire_pit`

**Primary + isolation**

| Arm | Mean | Predicted | Verdict |
|---|---|---|---|
| A | 0.5333 | baseline · 95% band [0.3333, 0.9600] | — |
| B | 0.5167 | Δ = -0.06 SD (need ≥+1.0 SD) | **FAIL** |
| C | 0.6167 | ∈ A's band | **PASS** |

Robustness (legacy per-action failure rate, decrease): FAIL

**Corroborating metrics (≥1 must pass)**

| Metric | A mean ± SD | B mean | Δ in SD units | Direction | Verdict |
|---|---|---|---|---|---|
| Affordance-preference safe-fraction (safe-on-target / on-target total) | 0.8800 ± 0.1789 | 0.9333 | 0.30 | increase | **FAIL** |
| Tool-class diversity (fewer dead-end tools tried) | 7.6000 ± 2.3022 | 7.8000 | 0.09 | decrease | **FAIL** |
| Time-to-safe-steady-state (turns to 3 consecutive zero-failure turns; None censored to turn_count_binned+1) | 2.0000 ± 0.0000 | 2.0000 | — | decrease | **FAIL** — Zero SD on Arm A AND zero shift on B (degenerate). |
| Time-to-first-warm-self (action index of first warm_self; None censored to turn_count_binned+1) | 1.2000 ± 1.3038 | 1.2000 | 0.00 | decrease | **FAIL** |

Corroborating hits: **0 / 4**

**Descriptive corroborating — `fire_approach_action_count` (NOT pre-reg gated)**

| Arm | Mean count |
|---|---|
| A | 1.60 |
| B | 1.40 |

Δ (B − A) = -0.20; predicted direction: same_or_higher.

_Note: Δ < 0: Arm B shows FEWER positive-approach actions than Arm A. Substrate transfer predicts the positive edge ('fire = warm') is preserved; investigate whether the agent avoided fire entirely on B (general caution) or specifically lost the approach association._

**Secondary criterion — ablation attribution (≥1 must shrink Arm B's delta)**

| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |
|---|---|---|---|---|---|
| Wire-A annotation off | 0.5333 | 0.5167 | 0.5667 | -0.06 | **FAIL** — Ablation overshoots past Arm A baseline (B side: -0.0167, ablated side: +0.0333). Opposite-direction effect, not shrinkage — does NOT count as secondary-criterion PASS. |
| Wire 1 variance annotation off | 0.5333 | 0.5167 | 0.4567 | -0.21 | **FAIL** |
| NAc reward bias zeroed | 0.5333 | 0.5167 | 0.6000 | -0.18 | **FAIL** — Ablation overshoots past Arm A baseline (B side: -0.0167, ablated side: +0.0667). Opposite-direction effect, not shrinkage — does NOT count as secondary-criterion PASS. |

Secondary hits: **0 / 3**


### Scenario: `sharp_rock`

**Primary + isolation**

| Arm | Mean | Predicted | Verdict |
|---|---|---|---|
| A | 0.0000 | baseline · 95% band [0.0000, 0.0000] | — |
| B | 0.0000 | Zero-SD fallback (need ≥+1.0 SD) | **FAIL** |
| C | 0.0000 | ∈ A's band | **PASS** |

Robustness (legacy per-action failure rate, decrease): FAIL

**Corroborating metrics (≥1 must pass)**

| Metric | A mean ± SD | B mean | Δ in SD units | Direction | Verdict |
|---|---|---|---|---|---|
| Affordance-preference safe-fraction (safe-on-target / on-target total) | 0.0000 ± 0.0000 | 0.0000 | — | increase | **FAIL** — Zero SD on Arm A AND zero shift on B (degenerate). |
| Tool-class diversity (fewer dead-end tools tried) | 8.2000 ± 1.9235 | 7.8000 | -0.21 | decrease | **FAIL** |
| Time-to-safe-steady-state (turns to 3 consecutive zero-failure turns; None censored to turn_count_binned+1) | 2.0000 ± 0.0000 | 2.0000 | — | decrease | **FAIL** — Zero SD on Arm A AND zero shift on B (degenerate). |
| Time-to-first-warm-self (action index of first warm_self; None censored to turn_count_binned+1) | 2.0000 ± 0.0000 | 2.0000 | — | decrease | **FAIL** — Zero SD on Arm A AND zero shift on B (degenerate). |

Corroborating hits: **0 / 4**

**Secondary criterion — ablation attribution (≥1 must shrink Arm B's delta)**

| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |
|---|---|---|---|---|---|
| Wire-A annotation off | 0.0000 | 0.0000 | 0.0000 | — | **FAIL** — Insufficient data for ablation comparison. |
| Wire 1 variance annotation off | 0.0000 | 0.0000 | 0.0000 | — | **FAIL** — Insufficient data for ablation comparison. |
| NAc reward bias zeroed | 0.0000 | 0.0000 | 0.0000 | — | **FAIL** — Insufficient data for ablation comparison. |

Secondary hits: **0 / 3**

