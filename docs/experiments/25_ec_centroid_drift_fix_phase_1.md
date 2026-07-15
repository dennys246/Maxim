# 25 — EC centroid drift fix, Phase 1 (matrix sweep)

**Date:** 2026-05-23
**Branch:** [`feat/0-9-1-roy-paraphrase-diagnostic`](https://github.com/dennys246/Maxim/tree/feat/0-9-1-roy-paraphrase-diagnostic)
**Plan:** [docs/plans/archive/ec_centroid_drift_fix.md § Phase 1](../plans/archive/ec_centroid_drift_fix.md#phase-1--matrix-diagnostic-one-pr)
**Companion:** [24_roy_paraphrase_diagnostic.md](24_roy_paraphrase_diagnostic.md) (motivating diagnostic; `CENTROID_DRIFT_COLLAPSE` verdict)
**Script:** [scripts/diagnose_roy_paraphrase_collapse.py](../../scripts/diagnose_roy_paraphrase_collapse.py) `--matrix`
**Fixture:** [data/roy_paraphrase_pairs.json](../../data/roy_paraphrase_pairs.json)
**Raw output:** `/tmp/roy_matrix/` (16 per-cell JSON + `matrix.json`)

## Status

**WINNER: `d0_f0_t50`** — decomposition off, frozen-text-centroid off, **threshold 0.40 → 0.50**. Single-parameter fix from current defaults. 15 of 16 matrix cells pass the gate; the only failing cell is the current default `d0_f0_t40`. Phase 2 unblocked.

## Pre-registered gate

Same as Phase 0 (the [24 diagnostic](24_roy_paraphrase_diagnostic.md)):

- **Sequential pair collapse ≥ 70%** (paraphrase concepts pattern-complete onto the same EC node under streaming input)
- **AND sequential distractor collapse < 30%** (semantically-distant strings stay in separate nodes under streaming input)

Among passing cells, **smallest delta from current defaults** picks the winner. Sort key: `(decomposition_on, frozen_text_centroid_on, abs(threshold - 0.40))`.

## Matrix

```
  cell           pair_seq pair_iso dist_seq dist_iso  #nodes  gate    verdict
  d0_f0_t40          100%     100%      60%       0%       3   fail    CENTROID_DRIFT_COLLAPSE
  d0_f0_t50           90%     100%       0%       0%       7   PASS    CROSS_MODAL_ONLY
  d0_f0_t60           90%      80%       0%       0%      10   PASS    CROSS_MODAL_ONLY
  d0_f0_t70           70%      70%       0%       0%      14   PASS    CROSS_MODAL_ONLY
  d0_f1_t40           80%     100%       0%       0%       6   PASS    CROSS_MODAL_ONLY
  d0_f1_t50           90%     100%       0%       0%       9   PASS    CROSS_MODAL_ONLY
  d0_f1_t60           80%      80%       0%       0%      11   PASS    CROSS_MODAL_ONLY
  d0_f1_t70           70%      70%       0%       0%      14   PASS    CROSS_MODAL_ONLY
  d1_f0_t40           80%      80%       0%       0%      15   PASS    CROSS_MODAL_ONLY
  d1_f0_t50           80%      80%       0%       0%      21   PASS    CROSS_MODAL_ONLY
  d1_f0_t60           70%      70%       0%       0%      25   PASS    CROSS_MODAL_ONLY
  d1_f0_t70           70%      70%       0%       0%      27   PASS    CROSS_MODAL_ONLY
  d1_f1_t40           80%      80%       0%       0%      16   PASS    CROSS_MODAL_ONLY
  d1_f1_t50           80%      80%       0%       0%      22   PASS    CROSS_MODAL_ONLY
  d1_f1_t60           70%      70%       0%       0%      25   PASS    CROSS_MODAL_ONLY
  d1_f1_t70           70%      70%       0%       0%      27   PASS    CROSS_MODAL_ONLY
```

Legend: `d0/d1` = decomposition off/on; `f0/f1` = `"text"` not in / in `frozen_centroid_modalities`; `t40-70` = `pattern_complete_threshold ∈ {0.40, 0.50, 0.60, 0.70}`.

## Interpretation

### The fix is one number

Bumping `ECConfig.pattern_complete_threshold` from 0.40 to 0.50 alone:

- Drops sequential distractor collapse from **60% → 0%** (gains all 3 cross-class false-collapses).
- Drops sequential pair collapse from 100% to 90% (loses `pair_03_satiety_belly`, cosine 0.590, marginal under streaming centroid drift even though isolated mode still nails it at 100%).
- Bumps EC node count after the 22-string walk from **3 (mega-collapse) → 7 (proper concept separation)**.

No structural code change, no `frozen_centroid_modalities` change, no decomposition wiring. The fix is `pattern_complete_threshold: float = 0.50` in `ECConfig`.

### Why the threshold change works without freezing the centroid

The original failure was that successive low-but-above-threshold matches (cosines 0.42-0.50) pulled the running-mean centroid toward a generic prototype, then admitted EVERYTHING. Lifting the threshold to 0.50 rejects those marginal admissions at the door — the centroid never drifts because the drifty inputs never join the node in the first place. Frozen centroid is a separate fix for the same root cause; both work, but threshold is the minimal change.

### Decomposition over-fragments — confirmed

Comparing `d0_f0_t50` (winner) vs `d1_f0_t50` (decomposition on, same threshold):

- Pair collapse: **100% iso → 80% iso** (decomposition loses 2 pairs)
- EC nodes after walk: **7 → 21** (decomposition triples the cluster count)

Decomposition splits "warm food rises in your belly" into noun chunks like `["food", "belly"]` and "fullness settles in your stomach" into `["fullness", "stomach"]`. The whole-sentence cosine (0.590) is well above 0.50 and would pattern-complete. But the chunk-level overlap is empty — neither "food" matches "fullness" nor "belly" matches "stomach" at threshold 0.50. The synonym substitution that humans see as paraphrase doesn't survive noun-chunk extraction.

This confirms the [ec_centroid_drift_fix.md "Rejected approaches"](../plans/archive/ec_centroid_drift_fix.md#decomposition-as-the-fix) call to skip decomposition as the production fix. It also explains why MAXIM_CONCEPT_DECOMPOSITION ships off-by-default — the fragmentation cost is real.

### Frozen centroid + threshold 0.40 also works

`d0_f1_t40` (frozen text centroid, threshold unchanged) passes the gate at 80% sequential pair / 0% sequential distractor with 6 EC nodes. Slightly worse than `d0_f0_t50` on pair recall (80% vs 90%) but slightly more EC nodes. Mechanism: freezing the centroid prevents the drift that admitted distractors, but doesn't widen the rejection band, so low-cosine pairs (0.583, 0.590) at threshold 0.40 still struggle under sequential order effects.

Either approach is bio-defensible. Threshold 0.50 wins on the "smallest config delta" tiebreaker.

### Combined fixes don't add value

`d0_f1_t50` (frozen + threshold 0.50) gets 90% pair / 0% distractor with 9 nodes — identical pair rate to the winner, slightly more nodes. No additional safety margin worth the extra config change.

`d1_*` cells (decomposition on) consistently underperform on pair collapse and bloat node counts. Decomposition is not a corner of the matrix where stacking helps.

## Phase 2 target

The Phase 3 EC config change candidate is:

```python
# src/maxim/similarity/ec.py
@dataclass
class ECConfig:
    ...
    pattern_complete_threshold: float = 0.50  # was 0.40
    # frozen_centroid_modalities unchanged: {"interoception"}
```

Phase 2 ([plan § Phase 2](../plans/archive/ec_centroid_drift_fix.md#phase-2--regression-guard-against-p1p2-one-pr)) re-runs the P1 paraphrase-collapse sweep ([p1_recognition_sweep.md](p1_recognition_sweep.md), 91.7% pin at threshold 0.40) and the P2 reward-modulation sweep ([p2_reward_modulation_sweep.md](p2_reward_modulation_sweep.md), +56pp target gain at threshold 0.70) with the new default. Either could regress.

**Specific Phase 2 concerns surfaced by this matrix:**

- P1 was tuned at threshold 0.40. Moving to 0.50 might drop the 91.7% collapse rate. The matrix shows 0.50 still keeps isolated pair collapse at 100% on the Roy fixture, but P1's fixture is different (broader paraphrase coverage). Pre-registered Phase 2 tolerance: P1 must stay ≥ 85%.
- P2 shipped at threshold 0.70 with a frozen-pair-collapse arm. The matrix shows t70 cells get 70% pair / 0% distractor on the Roy fixture — narrower margin. Phase 2 needs to confirm P2's +56pp target gain holds at 0.50 OR re-anchor P2 on the new threshold.

If P1 or P2 regresses at 0.50, Phase 1 loops back: candidate `d0_f1_t50` (frozen + 0.50) is the next-smallest delta and provides a middle ground.

## Reproduction

```bash
MAXIM_SUBSTRATE_PATH=1 python scripts/diagnose_roy_paraphrase_collapse.py \
    --matrix \
    --input data/roy_paraphrase_pairs.json \
    --output-dir /tmp/roy_matrix
```

Cost: zero $ (substrate-only). Wall time: ~80 s for the 16-cell sweep (sentence-transformers model loaded once, shared across cells). Per-cell JSON + matrix summary in `/tmp/roy_matrix/`.

## What this phase does NOT do

- Does NOT change `ECConfig` defaults. That's Phase 3, gated on Phase 2.
- Does NOT validate behavioral signal in the Roy harness. That's Phase 4.
- Does NOT decide 0.9.1 vs 1.0 ship target. That's Phase 5 / operator decision after Phase 4.
- Does NOT touch the cancelled `cross_modal_substrate_binding.md` status — the fix is intra-modal hygiene, not cross-modal alignment. Roy-5 / JEPA direction unchanged.
