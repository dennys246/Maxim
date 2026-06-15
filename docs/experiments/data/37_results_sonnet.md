## Results — claude-sonnet 2026-06-10

Source: `docs/experiments/data/37_results_sonnet.jsonl` · Analyzer version: `1.0` · Schema: `1.0`

### Overall verdict: **PARTIAL — investigation gate**

Primary or isolation FAIL on ≥1 scenario. Root-cause required before bio-attribution can ship.

### Scenario: `fire_pit`

**Primary + isolation**

| Arm | Mean | Predicted | Verdict |
|---|---|---|---|
| A | 0.5933 | baseline · 95% band [0.3500, 0.7867] | — |
| B | 0.5000 | Δ = -0.52 SD (need ≥+1.0 SD) | **FAIL** |
| C | 0.7167 | ∈ A's band | **PASS** |

Robustness (legacy per-action failure rate, decrease): FAIL

**Corroborating metrics (≥1 must pass)**

| Metric | A mean ± SD | B mean | Δ in SD units | Direction | Verdict |
|---|---|---|---|---|---|
| Affordance-preference safe-fraction (safe-on-target / on-target total) | 0.6600 ± 0.1065 | 0.6500 | -0.09 | increase | **FAIL** |
| Tool-class diversity (fewer dead-end tools tried) | 10.0000 ± 1.2247 | 10.0000 | 0.00 | decrease | **FAIL** |
| Time-to-safe-steady-state (turns to 3 consecutive zero-failure turns; None censored to turn_count_binned+1) | 1.8000 ± 0.4472 | 2.0000 | 0.45 | decrease | **FAIL** |
| Time-to-first-warm-self (action index of first warm_self; None censored to turn_count_binned+1) | 1.2000 ± 1.0954 | 1.0000 | -0.18 | decrease | **FAIL** |

Corroborating hits: **0 / 4**

**Descriptive corroborating — `fire_approach_action_count` (NOT pre-reg gated)**

| Arm | Mean count |
|---|---|
| A | 2.00 |
| B | 1.60 |

Δ (B − A) = -0.40; predicted direction: same_or_higher.

_Note: Δ < 0: Arm B shows FEWER positive-approach actions than Arm A. Substrate transfer predicts the positive edge ('fire = warm') is preserved; investigate whether the agent avoided fire entirely on B (general caution) or specifically lost the approach association._

**Secondary criterion — ablation attribution (≥1 must shrink Arm B's delta)**

| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |
|---|---|---|---|---|---|
| Wire-A annotation off | 0.5933 | 0.5000 | 0.7233 | -0.20 | **FAIL** — Ablation overshoots past Arm A baseline (B side: -0.0933, ablated side: +0.1300). Opposite-direction effect, not shrinkage — does NOT count as secondary-criterion PASS. |
| Wire 1 variance annotation off | 0.5933 | 0.5000 | 0.5800 | 0.44 | **FAIL** |
| NAc reward bias zeroed | 0.5933 | 0.5000 | 0.6333 | 0.30 | **FAIL** — Ablation overshoots past Arm A baseline (B side: -0.0933, ablated side: +0.0400). Opposite-direction effect, not shrinkage — does NOT count as secondary-criterion PASS. |

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
| Affordance-preference safe-fraction (safe-on-target / on-target total) | 0.1000 ± 0.2236 | 0.0000 | -0.45 | increase | **FAIL** |
| Tool-class diversity (fewer dead-end tools tried) | 8.6000 ± 1.6733 | 8.2000 | -0.24 | decrease | **FAIL** |
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

