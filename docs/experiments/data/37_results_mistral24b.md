## Results — 2026-06-11 Mistral24B fire

Source: `docs/experiments/data/37_results_mistral24b.jsonl` · Analyzer version: `1.0` · Schema: `1.0`

### Overall verdict: **PARTIAL — investigation gate**

Primary or isolation FAIL on ≥1 scenario. Root-cause required before bio-attribution can ship.

### Scenario: `fire_pit`

**Primary + isolation**

| Arm | Mean | Predicted | Verdict |
|---|---|---|---|
| A | 1.0000 | baseline · 95% band [1.0000, 1.0000] | — |
| B | 0.6000 | Zero-SD fallback (need ≥+1.0 SD) | **FAIL** |
| C | 0.4000 | ∈ A's band | **FAIL** |

Robustness (legacy per-action failure rate, decrease): FAIL

**Corroborating metrics (≥1 must pass)**

| Metric | A mean ± SD | B mean | Δ in SD units | Direction | Verdict |
|---|---|---|---|---|---|
| Affordance-preference safe-fraction (safe-on-target / on-target total) | 1.0000 ± 0.0000 | 0.6000 | — | increase | **FAIL** — Zero-SD fallback: shift moves AGAINST the predicted direction. |
| Tool-class diversity (fewer dead-end tools tried) | 4.4000 ± 1.6733 | 7.6000 | 1.91 | decrease | **FAIL** |
| Time-to-safe-steady-state (turns to 3 consecutive zero-failure turns; None censored to turn_count_binned+1) | 2.0000 ± 0.0000 | 2.0000 | — | decrease | **FAIL** — Zero SD on Arm A AND zero shift on B (degenerate). |
| Time-to-first-warm-self (action index of first warm_self; None censored to turn_count_binned+1) | 0.4000 ± 0.5477 | 2.6000 | 4.02 | decrease | **FAIL** |

Corroborating hits: **0 / 4**

**Descriptive corroborating — `fire_approach_action_count` (NOT pre-reg gated)**

| Arm | Mean count |
|---|---|
| A | 1.00 |
| B | 0.80 |

Δ (B − A) = -0.20; predicted direction: same_or_higher.

_Note: Δ < 0: Arm B shows FEWER positive-approach actions than Arm A. Substrate transfer predicts the positive edge ('fire = warm') is preserved; investigate whether the agent avoided fire entirely on B (general caution) or specifically lost the approach association._

**Secondary criterion — ablation attribution (≥1 must shrink Arm B's delta)**

| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |
|---|---|---|---|---|---|
| Wire-A annotation off | 1.0000 | 0.6000 | 0.6000 | — | **FAIL** — Insufficient data for ablation comparison. |
| Wire 1 variance annotation off | 1.0000 | 0.6000 | 0.6000 | — | **FAIL** — Insufficient data for ablation comparison. |
| NAc reward bias zeroed | 1.0000 | 0.6000 | 0.7000 | — | **FAIL** — Insufficient data for ablation comparison. |

Secondary hits: **0 / 3**

**Notes / warnings**

- Arm C mean 0.4000 for fire_pit falls outside Arm A's band [1.0000, 1.0000] — 'general caution' confound.
- Arm C mean 0.0182 for fire_pit falls outside Arm A's band [0.0000, 0.0000] — 'general caution' confound.


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
| Tool-class diversity (fewer dead-end tools tried) | 5.6000 ± 0.5477 | 4.6000 | -1.83 | decrease | **PASS** |
| Time-to-safe-steady-state (turns to 3 consecutive zero-failure turns; None censored to turn_count_binned+1) | 2.0000 ± 0.0000 | 2.0000 | — | decrease | **FAIL** — Zero SD on Arm A AND zero shift on B (degenerate). |
| Time-to-first-warm-self (action index of first warm_self; None censored to turn_count_binned+1) | 2.0000 ± 0.0000 | 2.0000 | — | decrease | **FAIL** — Zero SD on Arm A AND zero shift on B (degenerate). |

Corroborating hits: **1 / 4**

**Secondary criterion — ablation attribution (≥1 must shrink Arm B's delta)**

| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |
|---|---|---|---|---|---|
| Wire-A annotation off | 0.0000 | 0.0000 | 0.0000 | — | **FAIL** — Insufficient data for ablation comparison. |
| Wire 1 variance annotation off | 0.0000 | 0.0000 | 0.0000 | — | **FAIL** — Insufficient data for ablation comparison. |
| NAc reward bias zeroed | 0.0000 | 0.0000 | 0.0000 | — | **FAIL** — Insufficient data for ablation comparison. |

Secondary hits: **0 / 3**

