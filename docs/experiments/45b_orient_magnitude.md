# Exp 45b — orient magnitude: does the substrate learn how FAR, not just which way?

**Status:** PRE-REGISTERED 2026-07-16 (design sim-validated; **not yet run on hardware**).
**Plan:** [orient_magnitude_learning.md](../plans/orient_magnitude_learning.md) S0.
**Parent:** [Exp 45](45_reachy_orient_live.md) (all arms EARNED) — this trips its own
"orient-affordance YAML change" re-run rule, so: new pre-registration, fresh NAc, new
bundle version (queen-mind v0.2).

## Claim under test

The same `potential_diff` relief credit that taught **direction** also teaches
**magnitude** — given a 2×2 action set (`{left,right}` × `{normal,big}`), the substrate
learns to take the *big* step when the target is far and the *normal* step when it is
near, with no LLM and no hand-coded gain schedule.

**Trigger:** watching the Exp 45 policy live, it turns the right way but takes the same
step from |az| 0.15 and 0.65. Root cause is the **action set** (2 fixed-magnitude
actions), not bin resolution — bins are what it senses, actions are what it can express.

## Design (and why the magnitudes are load-bearing)

`bodies/reachy_mini.yaml` `orient` now declares four affordances:
`turn_left` (+0.3 rad) / `turn_right` (−0.3) / `turn_left_big` (+0.9) /
`turn_right_big` (−0.9). Scripts read the YAML magnitudes directly (`--step-scale`
rescales; `--step` is gone).

**The values are not arbitrary — they are derived from the measured hardware.**
`potential_diff` has NO cost for large moves, so a big step wins *everywhere* unless it
**overshoots** when the target is already near-centered. At the Exp 45 sweep's measured
DoA gain (0.58 az/rad):

| Action | Δaz | from near (az≈0.18) | from near (az≈0.42) | from far (az≈0.60) |
|---|---|---|---|---|
| normal (0.3 rad) | ~0.17 | +0.17 relief | +0.17 | +0.17 |
| big (0.9 rad) | ~0.52 | **−0.16** (overshoot) | +0.32 | **+0.52** (centers) |

Bin-averaged: near → normal (0.17) beats big (~0.08); far → big (0.52) beats normal
(0.17). **That asymmetry is the entire learnability of magnitude.** The plan's first
sketch (0.12/0.4 rad) would have produced "always big" — nothing overshoots at that
size. Re-tuning these values re-opens this experiment.

**Lock-safety interaction:** a 0.9 rad single jump would lose DoA lock (Exp 45's
tracking-estimator finding), so `LiveRig.goto_body_yaw` now **walks every motion** in
≤0.3 rad increments. Steps ≤0.3 (the whole 2-action era) are byte-identical single
commands; `_big` actions become 3 sub-moves. Trials get slower (~15–20 s), not different.

## Metrics (pre-registered)

1. **Direction correctness** (primary, replication): fraction of off-center bins whose
   frozen-policy argmax turns toward center. **Expect 1.00** (Exp 45 replication under a
   larger action set).
2. **Magnitude appropriateness** (primary, new): fraction of off-center bins whose
   argmax has the appropriate size — far → `_big`, near → normal.

**Verdicts:** PASS = direction 1.00 **and** magnitude ≥ 0.75. PARTIAL = direction 1.00,
magnitude 0.50 (one side learned it). FAIL = direction < 1.00 (regression).

## Sim validation of the design (done 2026-07-16, before hardware)

Dry-harness runs on the same loop (`--dry-run --perturb`, seeds 0–2):

| epsilon / trials | direction | magnitude |
|---|---|---|
| 0.25 / 80 | 1.00, 1.00, 1.00 | 0.50, 1.00, 1.00 |
| 0.40 / 120 | 1.00, 0.75, 1.00 | 1.00, 0.75, 0.75 |
| 0.60 / 120 | 1.00, 1.00, 1.00 | 1.00, 1.00, 0.75 |
| 0.40 / 200 | 1.00, 1.00, 1.00 | 1.00, 1.00, 0.75 |

**Direction is robust; magnitude is real but seed-dependent (~0.75–1.00, mean ≈0.8).**
Honest expectation for hardware, set in advance: magnitude will likely land 0.75, not
1.00. **Protocol: `--epsilon 0.5`, 100 trials, `--fresh`** (~30 min).

## Pre-registered diagnostic (the finding that makes a PARTIAL informative)

If magnitude < 1.00, **read the bias table** — the failure modes are distinguishable:

- **Exploration-limited** (expected): the better action's bias is **exactly 0.0** — it
  was *never sampled*. Greedy locked onto the first action that earned positive relief
  and never tried the better one. Observed in sim: a `far_right` bin ended with
  `turn_right`=0.468 and `turn_right_big`=**0.0** after 80 trials. This is the known NAc
  fixation pattern ([Exp 41](41_substrate_primary_exploration.md) VOID; deterministic
  argmax, no novelty/visit-count) resurfacing where a 2-action set could not expose it —
  a **positive** finding for [substrate_exploration_policy.md](../plans/archive/substrate_exploration_policy.md).
- **Credit-limited** (would be the interesting failure): the better action *was* sampled
  (bias ≠ 0.0) and still lost. That would falsify the overshoot analysis above and
  re-open the credit design (an explicit effort cost — bigger turns cost more energy —
  is the considered-and-deferred alternative; it is bio-faithful and the codebase has an
  energy layer, but it deviates from Phase 0b's validated `potential_diff`).

## What this says about S1 (Weber bins) — answers plan open-question 3

**S0 does not make S1 moot; S0's residual noise IS S1's motivation.** The `near` bin
(|az| 0.1–0.5) *spans the flip point*: big is wrong at az 0.18 (−0.16) and right at 0.42
(+0.32). A bin whose correct action changes inside the bin is under-resolved by
construction — bin-averaging works but is noisy and exploration-fragile, exactly as the
sim sweep shows. Finer/Weber-scaled bins put the flip *between* bins. This is now
evidence-backed rather than speculative.

## Backward compatibility (verified)

`turn_left`/`turn_right` keep their names and values, so **queen-mind v0.1 (the
2-action policy) still loads** under the new YAML — verified: the real merged v0.1 NAc
probes **direction 1.00 / magnitude 0.50**, i.e. "knows which way, has never met the big
actions." v0.1 stays demoable, and a v0.1 policy is a valid warm start for a v0.2 run
(`orient_demo.py --learn`, or `live_3_learn.py` without `--fresh`) — an incremental-
learning demo in its own right, though the pre-registered arm above uses `--fresh`.

## Open questions

1. Should the merge-arm gauntlet **gate** on magnitude, or only report it? (Direction is
   safety-critical — a wrong-direction policy is broken; magnitude is quality.) Currently
   report-only; a Queen tier could raise the bar for v0.2 bundles.
2. Is 0.9 rad (~52° of base rotation) acceptable demo behavior, or startling in a room?
3. If exploration-limited: fix in the harness (decaying epsilon / optimistic init) or at
   the NAc layer (the archived exploration-policy plan)? The latter is the substrate-
   native answer and generalizes past orienting.
