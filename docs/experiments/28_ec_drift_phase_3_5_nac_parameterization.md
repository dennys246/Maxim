# 28 — EC centroid drift fix, Phase 3.5 (NAc threshold-override base parameterization)

**Date:** 2026-05-23
**Branch:** [`feat/0-9-1-ec-drift-phase-3-5-parameterize-nac`](https://github.com/dennys246/Maxim/tree/feat/0-9-1-ec-drift-phase-3-5-parameterize-nac)
**Plan:** [docs/plans/archive/ec_centroid_drift_fix.md § Phase 3.5](../plans/archive/ec_centroid_drift_fix.md) (post-merge fold from PR #261's reviewer Q2)
**Companion:** [25_ec_centroid_drift_fix_phase_1.md](25_ec_centroid_drift_fix_phase_1.md), [26_ec_drift_phase_2_regression.md](26_ec_drift_phase_2_regression.md), [results/p2_reward_modulation_sweep.json](results/p2_reward_modulation_sweep.json)

## Status

**VERDICT: SHIP (parameterized) — P2 IMPROVED, not regressed.**

Pre-registered tolerance was "mean target gain ≥ +40pp, 9-of-10 seeds pass." Post-fix sweep produced **mean target gain +58.4% pp ± 9.1%, 10-of-10 seeds pass, 100% target monotone**. Stricter than the pre-fix shipping result (+56.0% ± 29.0%, 9-of-10) on every dimension. No regression; the routing alternatives (halt, α-coefficient) are unnecessary.

## What this phase did

Removed the hardcoded `base = 0.44` from `NAc.get_threshold_overrides` and parameterized it via a keyword-only `base_threshold: float | None = None` argument. Production callers (`LinguisticEncoder._get_reward_overrides` and `LinguisticEncoder.encode_decomposed`) now pass `self.ec.config.pattern_complete_threshold` so the override always tracks the LIVE EC threshold. The hardcoded fallback (0.44) is preserved for legacy callers without an EC reference — three unit tests exercise it.

## Why this isn't just cleanup

At non-default EC thresholds the pre-fix hardcoded base produced a coupling artifact. With EC threshold 0.80 (P2 validation sweep) and a rewarded node's bias of 0.20:

| Setting | Pre-Phase-3.5 override | Post-Phase-3.5 override | Recognition radius widening |
|---|---:|---:|---:|
| EC threshold 0.44 (production default) | 0.24 | 0.24 | 0.20 (unchanged — bug invisible at default) |
| EC threshold 0.70 (P2 sweep) | 0.24 | 0.50 | pre: 0.46 cosine units; post: 0.20 |
| EC threshold 0.80 (P2 mechanism) | 0.24 | 0.60 | pre: 0.56 cosine units; post: 0.20 |

The bio-correct semantics is: **reward bias widens recognition by a constant cosine amount equal to the bias magnitude.** The pre-fix code accidentally widened by `(base - 0.44) + bias`, scaling with the gap between the live threshold and the hardcoded fallback. Invisible at production default; load-bearing at any non-default threshold including every P2 cell.

## Pre-registered gate

Per [kickoff_ec_drift_phase_3_5.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/kickoff_ec_drift_phase_3_5.md):

- **Mean target gain ≥ +40 pp** on the 10-seed P2 sweep
- **9-of-10 seeds pass individually**

Same tolerance band as Phase 2. Pinned BEFORE the run.

## Results

10-seed sweep with `paraphrase-mpnet-base-v2` on `scenarios/substrate/p2_reward_modulation.yaml`, EC threshold 0.70, reward 2.0:

| Metric | Pre-Phase-3.5 (shipped) | Post-Phase-3.5 | Δ |
|---|---:|---:|---:|
| Mean target gain | +56.0% pp | **+58.4% pp** | +2.4 pp |
| Std target gain | ±29.0% | **±9.1%** | -19.9 pp (3.2× tighter) |
| Mean distractor drift | 0.0% pp | 0.0% pp | 0 |
| Std distractor drift | ±0.0% | ±0.0% | 0 |
| Target monotone fraction | 0.94 | **1.00** | +0.06 |
| Seeds passing individually | 9/10 | **10/10** | +1 |

Per-seed target gains: 70.0, 62.0, 50.0, 52.0, 52.0, 44.0, 64.0, 72.0, 56.0, 62.0 — every seed clears the +40 pp threshold; minimum is +44 pp.

## Mechanism — why P2 tightened, not loosened

Counter-intuitively, the tighter parameterized override (0.50 instead of 0.24 at EC threshold 0.70) makes P2 MORE reliable. The mechanism:

- Pre-fix override 0.24 was wide enough to admit cross-cluster paraphrases on certain seeds (where shuffle order made a distractor land near a rewarded target node). Those seeds showed inflated target gain accompanied by distractor interference — the ±29 pp stdev was driven by these outlier seeds, not by signal variance.
- Post-fix override 0.50 still admits within-cluster paraphrases (which cluster at cosines 0.55-0.90 with rewarded centroids) but rejects cross-cluster admissions at cosines 0.30-0.50. Target gain is steadier across seeds; distractor drift stays at 0.

The shipped result was "+56% target gain, +0% distractor drift" — distractor drift being a HARD 0% pre-fix was the clue that contamination wasn't visible in the headline metric (because of plurality-ownership accounting, distractors stolen by target nodes show as "no drift" on the distractor cluster's collapse rate). The variance in target gain was the actual signal — and parameterization removed it.

## What changed

### `src/maxim/decisions/nac.py`

```python
def get_threshold_overrides(
    self,
    agent_id: str,
    *,
    base_threshold: float | None = None,
) -> dict[str, float]:
    ...
    base = 0.44 if base_threshold is None else base_threshold
    ...
```

Docstring expanded to explain the parameterization rationale + name the legacy-fallback test sites.

### `src/maxim/similarity/encoder.py`

Both call sites — `_get_reward_overrides` (line 346) and `encode_decomposed` (line 262) — now pass `base_threshold=self.ec.config.pattern_complete_threshold`.

### `tests/unit/test_ec_centroid_drift_fix.py`

- Existing `test_nac_threshold_override_base_tracks_ec_default` retained (verifies the fallback path still tracks the EC default — load-bearing for the three legacy-fallback callers).
- New `test_nac_threshold_override_accepts_base_threshold_parameter` verifies parameterization arithmetic at non-default thresholds.
- New `test_nac_threshold_override_clamps_to_floor_at_high_bias` pins the 0.10 clamp floor invariant independently of the base.

## What this phase did NOT do

- Did NOT change `ECConfig.pattern_complete_threshold` (already 0.44 from Phase 3).
- Did NOT touch the Roy-5 `H1C_LOWER_BOUND` coupling.
- Did NOT add an α-coefficient parameter to NAc (the routing alternative was reserved for the regression case — not needed).
- Did NOT modify P2 fixture, P2 metric, or P2 threshold. The only change is the live override math.

## Reproduction

```bash
# Fast unit tests (4 new tests in TestECCentroidDriftFix + 62 collateral):
PYTHONPATH=src python -m pytest tests/unit/test_ec_centroid_drift_fix.py tests/unit/test_substrate_recognition.py tests/substrate/test_p2_reward_modulation.py::TestP2Mechanism -xvs

# The P2 validation gate (~25s wall after model warm):
PYTHONPATH=src python -m pytest tests/substrate/test_p2_reward_modulation.py::TestP2ValidationSweep::test_sweep_10_seeds -xvs

# Results JSON updated in-place:
cat docs/experiments/results/p2_reward_modulation_sweep.json | jq .
```

## What's next

Phase 4 (Roy-2c behavioral validation) and Phase 5 (V1 thread-through) unchanged in scope and unblocked.
