# Exp 45b — orient magnitude: does the substrate learn how FAR, not just which way?

**Status:** **PASS** (hardware, 2026-07-16). Direction 1.00 + magnitude 0.75,
stable across 8 consecutive probes (trials 15→50). Pre-registered before the run;
the predicted value (0.75, not 1.00) was hit exactly.
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
**overshoots** when the target is already near-centered. Derived against a gain of
0.58 az/rad — which came from the *contaminated* pre-headfix sweep but happens to
land within noise of the true post-fix value (0.562-0.58), so the design survived
by luck rather than by rigour:

| Action | Δaz | from near (az≈0.18) | from near (az≈0.42) | from far (az≈0.60) |
|---|---|---|---|---|
| normal (0.3 rad) | ~0.17 | +0.17 relief | +0.17 | +0.17 |
| big (0.9 rad) | ~0.52 | **−0.16** (overshoot) | +0.32 | **+0.52** (centers) |

Bin-averaged: near → normal (0.17) beats big (~0.08); far → big (0.52) beats normal
(0.17). **That asymmetry is the entire learnability of magnitude.** The plan's first
sketch (0.12/0.4 rad) would have produced "always big" — nothing overshoots at that
size. Re-tuning these values re-opens this experiment.

**Walked motion (rationale RETRACTED, behaviour kept):** `LiveRig.goto_body_yaw`
walks every motion in ≤0.3 rad increments. This was introduced for "DoA lock safety"
per Exp 45's tracking-estimator finding — **which was an artifact of the head-frame
bug and is retracted**. The walk is harmless (and `_big` actions become 3 sub-moves,
~15-20 s trials), so it stayed; but it is solving a problem that does not exist and
could be dropped after a clean post-headfix re-sweep.

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
construction — bin-averaging works but is noisy and exploration-fragile. Finer/
Weber-scaled bins put the flip *between* bins. **Predicted from sim, then confirmed on
hardware** (see Results: `near_right` drew az 0.44/0.49 placements and correctly learned
big there, while `near_left` drew lower ones and learned normal — same bin, opposite
lessons). S1's motivation is now hardware-confirmed, not speculative.

## Backward compatibility (verified)

`turn_left`/`turn_right` keep their names and values, so **queen-mind v0.1 (the
2-action policy) still loads** under the new YAML — verified: the real merged v0.1 NAc
probes **direction 1.00 / magnitude 0.50**, i.e. "knows which way, has never met the big
actions." v0.1 stays demoable, and a v0.1 policy is a valid warm start for a v0.2 run
(`orient_demo.py --learn`, or `live_3_learn.py` without `--fresh`) — an incremental-
learning demo in its own right, though the pre-registered arm above uses `--fresh`.

## Results (mag2, 2026-07-16): **PASS**

Ran only after the **head-frame bug** was found and fixed (see below) — the first
attempt (mag1) was incoherent because the microphones were counter-rotating away
from every commanded turn.

| metric | pre-registered bar | measured |
|---|---|---|
| direction correctness | 1.00 | **1.00** (from trial 15, 8 consecutive probes) |
| magnitude appropriateness | ≥ 0.75 (predicted 0.75, not 1.00) | **0.75** (stable, trials 15→50) |
| sign-check gain | — | 0.58/rad (matches the settle test's 0.562 — consistent physics at last) |

Final bias table (checkpoint @ trial 50; the session ended at 53 when the robot's
daemon backend went `ready: false` — an infra fault, not a learning event):

| bin | argmax | competing | verdict |
|---|---|---|---|
| far_left | `turn_left_big` **+0.593** | turn_right −0.020 | ✓ big for far |
| far_right | `turn_right_big` **+0.748** | turn_right +0.053 | ✓ big for far |
| near_left | `turn_left` **+0.185** | turn_left_big +0.072 | ✓ **normal for near** |
| near_right | `turn_right_big` +0.268 | turn_right +0.067 | ✗ (flip-point bin) |

**`near_left` is the claim.** Normal (+0.185) beat big (+0.072) — the substrate
learned *not to overshoot*, from relief alone, with no LLM, no effort cost, and no
new mechanism. Representative trial: `far_right turn_right_big  +0.57 → 0.00
relief=+0.567` — a big step from far landing dead centre.

### The 0.75, not 1.00 — the flip point, confirmed on hardware

`near_right` learned `_big` and scores mag=False. It is not a defect: its
placements happened to draw az **0.44 and 0.49** — the *top* of the near bin,
where big genuinely is optimal (it centres them: +0.367, +0.444 relief).
`near_left` drew lower placements and learned normal. **Same bin type, opposite
lessons, because the near bin spans the flip point** — the S1/Weber-bins argument,
predicted from sim in [orient_magnitude_learning.md](../plans/orient_magnitude_learning.md)
and now reproduced on hardware. S1's motivation is upgraded from *sim-suggested*
to *hardware-confirmed*.

Secondary mechanism worth recording: `update_cluster_reward` **accumulates**
(`bias += alpha * reward`) rather than averaging, so sample count competes with
mean reward — `near_left`'s normal won partly on ~6 samples vs big's 1. That is
the rich-get-richer dynamic behind the [Exp 41](41_substrate_primary_exploration.md)
fixation finding, surfacing again.

### Prerequisite: the head-frame bug (why mag1 failed and mag2 passed)

`goto_target(body_yaw=X, head=None)` does **not** leave the head alone — the
daemon re-solves IK against the retained *world-frame* head pose, counter-rotating
the Stewart platform. **The mic array is in the head.** Measured 0.32 rad of mic
rotation for a 0.9 rad body command. Fixed by shipping an explicit head matrix
(`head world-yaw == body_yaw`). Full account: [Exp 45](45_reachy_orient_live.md)
method section + [CLAUDE.md](../../CLAUDE.md) invariant.

This is also why the pre-registered **diagnostic** mattered: mag1's failure sat in
the *credit-limited* branch (near `_big` biases positive = sampled and won), which
correctly said "the expectation is wrong, not the learning" — the expectation was
wrong because the gain it assumed was measured through the bug.

## Open questions

1. Should the merge-arm gauntlet **gate** on magnitude, or only report it? (Direction is
   safety-critical — a wrong-direction policy is broken; magnitude is quality.) Currently
   report-only; a Queen tier could raise the bar for v0.2 bundles.
2. Is 0.9 rad (~52° of base rotation) acceptable demo behavior, or startling in a room?
3. If exploration-limited: fix in the harness (decaying epsilon / optimistic init) or at
   the NAc layer (the archived exploration-policy plan)? The latter is the substrate-
   native answer and generalizes past orienting.
