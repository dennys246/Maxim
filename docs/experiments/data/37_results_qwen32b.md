## Results — 2026-06-08 Qwen32B fire

Source: `docs/experiments/data/37_results_qwen32b.jsonl` · Analyzer version: `1.0` · Schema: `1.0`

### Overall verdict: **PARTIAL — investigation gate**

Primary or isolation FAIL on ≥1 scenario. Root-cause required before bio-attribution can ship.

### Scenario: `fire_pit`

**Primary + isolation**

| Arm | Mean | Predicted | Verdict |
|---|---|---|---|
| A | 0.4200 | baseline · 95% band [0.0333, 0.6600] | — |
| B | 0.8000 | Δ = +1.43 SD (need ≥+1.0 SD) | **PASS** |
| C | 0.6667 | ∈ A's band | **FAIL** |

Robustness (legacy per-action failure rate, decrease): PASS

**Corroborating metrics (≥1 must pass)**

| Metric | A mean ± SD | B mean | Δ in SD units | Direction | Verdict |
|---|---|---|---|---|---|
| Affordance-preference safe-fraction (safe-on-target / on-target total) | 0.7433 ± 0.1832 | 0.9667 | 1.22 | increase | **PASS** |
| Tool-class diversity (fewer dead-end tools tried) | 8.0000 ± 1.2247 | 6.6000 | -1.14 | decrease | **PASS** |
| Time-to-safe-steady-state (turns to 3 consecutive zero-failure turns; None censored to turn_count_binned+1) | 2.2000 ± 0.4472 | 2.0000 | -0.45 | decrease | **FAIL** |
| Time-to-first-warm-self (action index of first warm_self; None censored to turn_count_binned+1) | 1.2000 ± 0.4472 | 1.0000 | -0.45 | decrease | **FAIL** |

Corroborating hits: **2 / 4**

**Descriptive corroborating — `fire_approach_action_count` (NOT pre-reg gated)**

| Arm | Mean count |
|---|---|
| A | 1.60 |
| B | 3.00 |

Δ (B − A) = 1.40; predicted direction: same_or_higher.

**Secondary criterion — ablation attribution (≥1 must shrink Arm B's delta)**

| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |
|---|---|---|---|---|---|
| Wire-A annotation off | 0.4200 | 0.8000 | 0.6500 | 0.56 | **FAIL** |
| Wire 1 variance annotation off | 0.4200 | 0.8000 | 0.8167 | -0.06 | **FAIL** |
| NAc reward bias zeroed | 0.4200 | 0.8000 | 0.7933 | 0.03 | **FAIL** |

Secondary hits: **0 / 3**

**Notes / warnings**

- Arm C mean 0.6667 for fire_pit falls outside Arm A's band [0.0333, 0.6600] — 'general caution' confound.
- Arm C mean 0.0000 for fire_pit falls outside Arm A's band [0.0077, 0.0909] — 'general caution' confound.


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
| Tool-class diversity (fewer dead-end tools tried) | 7.6000 ± 0.8944 | 6.2000 | -1.57 | decrease | **PASS** |
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

