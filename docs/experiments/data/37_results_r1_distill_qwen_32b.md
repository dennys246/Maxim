## Results — 2026-06-13 R1-Distill-Qwen-32B reasoning-axis fire

Source: `docs/experiments/data/37_results_r1_distill_qwen_32b.jsonl` · Analyzer version: `1.0` · Schema: `1.0`

### Overall verdict: **PARTIAL — investigation gate**

Primary or isolation FAIL on ≥1 scenario. Root-cause required before bio-attribution can ship.

### Scenario: `fire_pit`

**Primary + isolation**

| Arm | Mean | Predicted | Verdict |
|---|---|---|---|
| A | 0.2590 | baseline · 95% band [0.1452, 0.4786] | — |
| B | 0.5657 | Δ = +2.11 SD (need ≥+1.0 SD) | **PASS** |
| C | 0.5270 | ∈ A's band | **FAIL** |

Robustness (legacy per-action failure rate, decrease): FAIL — DIVERGES from positive-approach-engagement primary; see protocol §1 (substrate may be biasing warm_self without reducing touch, or vice versa)

**Corroborating metrics (≥1 must pass)**

| Metric | A mean ± SD | B mean | Δ in SD units | Direction | Verdict |
|---|---|---|---|---|---|
| Affordance-preference safe-fraction (safe-on-target / on-target total) | 0.9714 ± 0.0639 | 1.0000 | 0.45 | increase | **FAIL** |
| Tool-class diversity (fewer dead-end tools tried) | 5.8000 ± 2.2804 | 5.6000 | -0.09 | decrease | **FAIL** |
| Time-to-safe-steady-state (turns to 3 consecutive zero-failure turns; None censored to turn_count_binned+1) | 1.6000 ± 0.8944 | 2.0000 | 0.45 | decrease | **FAIL** |
| Time-to-first-warm-self (action index of first warm_self; None censored to turn_count_binned+1) | 0.8000 ± 0.4472 | 0.2000 | -1.34 | decrease | **PASS** |

Corroborating hits: **1 / 4**

**Descriptive corroborating — `fire_approach_action_count` (NOT pre-reg gated)**

| Arm | Mean count |
|---|---|
| A | 1.40 |
| B | 3.20 |

Δ (B − A) = 1.80; predicted direction: same_or_higher.

**Secondary criterion — ablation attribution (≥1 must shrink Arm B's delta)**

| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |
|---|---|---|---|---|---|
| Wire-A annotation off | 0.2590 | 0.5657 | 0.4016 | 1.13 | **PASS** |
| Wire 1 variance annotation off | 0.2590 | 0.5657 | 0.4657 | 0.69 | **FAIL** |
| NAc reward bias zeroed | 0.2590 | 0.5657 | 0.1971 | 1.69 | **FAIL** — Ablation overshoots past Arm A baseline (B side: +0.3067, ablated side: -0.0619). Opposite-direction effect, not shrinkage — does NOT count as secondary-criterion PASS. |

Secondary hits: **1 / 3**

**Notes / warnings**

- Arm C mean 0.5270 for fire_pit falls outside Arm A's band [0.1452, 0.4786] — 'general caution' confound.
- Primary / robustness divergence on fire_pit: positive-approach-engagement primary=True vs per-action-failure-rate robustness=False. Substrate may be biasing toward warm_self without reducing touch (or vice versa). Investigate before claiming the verdict (per protocol §1).


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
| Affordance-preference safe-fraction (safe-on-target / on-target total) | 0.0000 ± 0.0000 | 0.2000 | — | increase | **PASS** — Zero-SD fallback: Arm A baseline collapsed but shift moves in the predicted direction; PASS. |
| Tool-class diversity (fewer dead-end tools tried) | 4.6000 ± 1.1402 | 5.4000 | 0.70 | decrease | **FAIL** |
| Time-to-safe-steady-state (turns to 3 consecutive zero-failure turns; None censored to turn_count_binned+1) | 1.2000 ± 1.0954 | 1.6000 | 0.37 | decrease | **FAIL** |
| Time-to-first-warm-self (action index of first warm_self; None censored to turn_count_binned+1) | 3.2000 ± 1.6432 | 2.4000 | -0.49 | decrease | **FAIL** |

Corroborating hits: **1 / 4**

**Secondary criterion — ablation attribution (≥1 must shrink Arm B's delta)**

| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |
|---|---|---|---|---|---|
| Wire-A annotation off | 0.0000 | 0.0000 | 0.0000 | — | **FAIL** — Insufficient data for ablation comparison. |
| Wire 1 variance annotation off | 0.0000 | 0.0000 | 0.0000 | — | **FAIL** — Insufficient data for ablation comparison. |
| NAc reward bias zeroed | 0.0000 | 0.0000 | 0.0000 | — | **FAIL** — Insufficient data for ablation comparison. |

Secondary hits: **0 / 3**

