# Exp 37 — SD-Shift Statistical Test Swap

**Status:** AMENDMENT IN FLIGHT 2026-06-05
**Author:** Denny + Claude
**Blocks:** Exp 37 re-fire (the 65-run gate that decides
`behavioral_graduation_candidates.md` Tier 1 row 1)
**Supersedes:** the percentile-band variance-survival test in the 2026-05-30
pre-registration (kept for the isolation rule only, where it still works)

## What this amendment changes

**Old primary criterion:** Arm B mean of `positive_approach_engagement_fraction`
must lie ABOVE Arm A's empirical 97.5th-percentile band, computed across N=5
paired trials.

**New primary criterion:** `(B mean - A mean) / A.sd ≥ +1.0` SD in the
predicted direction.

The robustness check (legacy `per_action_failure_rate`, decrease direction)
gets the symmetric SD-shift treatment: `(B mean - A mean) / A.sd ≤ -1.0` SD.

The isolation rule (C within A's `[p2.5, p97.5]` empirical band) is
**unchanged** — it's direction-agnostic and tolerant of A piling up at one
bound.

## Empirical evidence that anchored this amendment

The 2026-06-05 validation smoke (5 Arm A trials, post-pivot harness on
`positive_approach_engagement_fraction`, 8 turns each, against the leader's
Qwen2.5-14B from a peer machine) produced this distribution:

| Trial | seed-base | positive_approach | engagement_count | warm_self | touch | per_action_failure_rate |
|---|---|---|---|---|---|---|
| 1 | 42 | 1.0 | 1 | 1 | 0 | 0.000 |
| 2 | 42 | 0.0 | 0 | 0 | 0 | 0.000 |
| 3 | 300 | 1.0 | 1 | 1 | 0 | 0.000 |
| 4 | 300+1 | 0.5 | 2 | 1 | 1 | 0.143 |
| 5 | 300+2 | 1.0 | 1 | 1 | 0 | 0.000 |

**Summary:** Arm A primary distribution = `[0.0, 0.5, 1.0, 1.0, 1.0]`, mean 0.7,
SD 0.45. Real fire engagement happened in 4 of 5 trials; a real
`fire_pit_touch` event (the pre-pivot failure-class action) happened in
trial 4.

## Why the percentile-band rule fails on this data

`statistics.quantiles(values, n=40, method="inclusive")` returns the
boundaries between 40 equal-count groups. For the sorted A values
`[0.0, 0.5, 1.0, 1.0, 1.0]`:

- p2.5 ≈ 0.0 (interpolated near the bottom of the data)
- p97.5 ≈ 1.0 (the upper data points cluster at the ceiling)

Empirical band: `[0.0, 1.0]`.

For the primary rule "B mean > A.p97.5 = 1.0", Arm B's mean must exceed 1.0.
Mathematically impossible: `positive_approach_engagement_fraction` is bounded
in [0, 1] by construction (it's a ratio with non-negative integer numerator
and positive denominator). **No substrate-transfer effect, no matter how
strong, can produce a B mean > 1.0.**

The same problem afflicts the legacy `per_action_failure_rate` in the
opposite direction: A's values `[0.000, 0.000, 0.000, 0.143, 0.000]` pile up
at the floor (0.0), so p2.5 = 0.0 and "B < A.p2.5" requires B < 0 — also
impossible (the rate is bounded ≥ 0).

The percentile-band rule fundamentally cannot detect substrate-transfer
effects on **any bounded metric where Arm A frequently hits the bound**.

## Why SD-shift works

`(B - A) / A.sd ≥ +1` doesn't have a bound-ceiling problem:

- Bounded metric in [0, 1]
- A mean 0.7, A SD 0.45 → threshold = 0.7 + 0.45 = **1.15**
- Wait, that's still > 1.0 — the SD-shift threshold lands OUTSIDE the
  metric range. **The test still can't be passed.**

This is the standing risk that motivated my pre-implementation note: even
SD-shift may not save the test if A's natural variance is high enough that
1 SD exceeds the metric's bound.

### Mitigation: accept the result honestly

If the 65-trial Exp 37 fire produces Arm A data where `1 SD > (max - mean)`,
the SD-shift test will FAIL primary regardless of substrate-transfer reality.
The honest interpretation in that case is "the metric measurement
infrastructure (LLM-AUT + cradle scenario + N=5 trials) doesn't have the
resolution to detect the substrate-transfer effect" — NOT "substrate
transfer doesn't happen."

We do NOT tune the SD-shift threshold post-hoc to fit the data. That would
be the calibration treadmill we explicitly rejected in
`exp37_metric_pivot.md`.

### Why a strict ≥1 SD threshold even given the standing risk

Three reasons:

1. **Pre-reg consistency.** The corroborating-metric threshold is already
   ≥1 SD per the original 2026-05-30 pre-registration. Using a different
   threshold for the primary would create an inconsistency reviewers would
   fairly flag.
2. **Honest failure is better than overfit pass.** A FAIL result on the SD
   shift test is real evidence about the measurement instrument's
   resolution. A PASS engineered by lowering the threshold is evidence of
   nothing.
3. **There are honest follow-up paths.** If FAIL, the explicit
   post-1.0 paths are: (a) richer cradle environment (electric-heater idea,
   see "Out of scope" below), (b) higher N per arm (N=10 or N=20 makes the
   SD-shift threshold easier to clear by reducing A's SD), (c) substrate-
   primary measurement (Exp 38) which removes the LLM-AUT noise floor
   entirely.

## Zero-SD fallback

When A's SD is exactly 0 (all 5 trials produced identical values — a true
zero-variance case), the SD-shift formula divides by zero. The
`_compute_primary_isolation` zero-SD branch falls back to "pass on
directional sign + non-zero shift" — matches the existing I2 corroborating
fallback. An explanatory note is appended to the verdict so the human
reader knows the fallback fired.

This is an edge case for primary; the corroborating metrics have hit it
more often historically. For the primary, A all-zero would mean either:

- The agent always engaged with fire and always warm_self'd (positive_approach
  = 1.0 every trial — common per the validation data).
- The agent never engaged with fire (positive_approach = 0.0 every trial —
  pure non-engagement case).

In both, the substrate-transfer claim becomes binary: did Arm B's substrate
shift A's plateau in the predicted direction? Yes (non-zero shift in
predicted direction) → PASS with note. No (shift is zero or wrong direction)
→ FAIL.

## What stays unchanged

- **Isolation rule** (C within A's empirical band): direction-agnostic,
  tolerant of A piling up at one bound. No change needed.
- **Secondary criterion** (ablation attribution via shrinkage in SD units):
  was already using SD-shift logic. No change.
- **Corroborating metrics** (≥1 SD shift in predicted direction): same.
- **Verdict matrix** in the analyzer's `overall_verdict`: unchanged. The
  EARNED / EARNED-footnoted / PARTIAL-reframed / PARTIAL-investigation
  labels still hinge on primary_pass + isolation_pass + secondary_hits +
  corroborating_hits; only HOW we compute primary_pass changed.

## Test plan

- Update `_compute_primary_isolation` to use SD-shift instead of percentile
  band (parameterized by direction, with the zero-SD fallback).
- Update analyzer test fixtures — existing engineered EARNED test passes
  trivially (A=0.35 SD ~0.04, B=0.80 → delta_sd ~11, PASS) so existing test
  semantics are preserved.
- Add a new test specifically validating the SD-shift rule on a
  bounded-distribution case where percentile-band would have failed: A's
  values pile up near the ceiling, B's mean shifts ≥1 SD higher (still
  bounded but mathematically possible if A's SD is low enough).
- Update the markdown renderer to display `Δ = X SD (need ≥+1.0 SD)`
  instead of `> A.p97.5 = X` — reviewers should know what they're checking.

## Open questions

1. **What if Arm A's SD is so high that 1 SD > (max - mean)?** Acknowledged
   above. Honest FAIL, document, move to follow-up paths.
2. **Should we run the Exp 37 fire anyway given the standing risk?** YES.
   Even a FAIL result tells us something real: it pins which follow-up
   paths are needed. Skipping the fire because we suspect FAIL is the
   calibration-treadmill move (don't run the test because we don't like
   the projected answer).
3. **Should we increase N to 10 per arm preemptively?** Defer. N=5 is the
   pre-registered count; ANY deviation here without separate authorization
   moves more goalposts than the SD-shift swap itself does. If the N=5
   fire produces FAIL with the explanation "SD too wide for the
   threshold," then THAT result authorizes an N=10 follow-up.

## Out of scope (explicitly)

- **Cradle environment enrichment** (electric-heater idea, broken-heater
  SEM repair affordance, additional thermal-domain components for variety):
  the discussion 2026-06-05 surfaced this as a legitimate
  measurement-instrument improvement. Genuinely good idea for the cradle
  scenario regardless of Exp 37. **Tracked for 1.1 cradle polish, not as
  an Exp 37 blocker.** The user's instinct that environmental sparsity
  contributes to A's variance shape is correct; the fix is real but it's
  not in the critical path for 1.0 ship.
- **Substrate-primary action selection (Exp 38 / Oasis):** the LLM-AUT
  noise-floor problem is fundamental. Removing the LLM (substrate-primary)
  is the principled long-term answer. Out of scope for 1.0.
- **Higher N per arm (N=10, N=20):** see Open Question 3. Available as a
  follow-up if the N=5 fire fails with "SD too wide" as the explanation.
- **Different statistical tests (Mann-Whitney U, Fisher exact, etc.):**
  considered but rejected — they have even lower statistical power at
  N=5, and changing the test family is a bigger pre-reg move than the
  SD-shift swap.

## Sequencing

1. Land this plan + analyzer patch + amendment as a single PR off main.
2. Validation: 68 + 7038 existing tests stay green; new SD-shift test
   exercises the bounded-distribution case.
3. Fire Exp 37 (65 runs × ~30 min/run ≈ 33 hours wall on the Mac Mini
   leader). Local Qwen, $0 cost. `caffeinate`/`pmset` already configured
   per the 2026-06-05 leader setup.
4. Analyzer run on the resulting JSONL → results doc → row 1 status flip
   in `behavioral_graduation_candidates.md`.
5. If the verdict is FAIL with "SD too wide" as the explanation, surface
   the explicit follow-up paths (cradle enrichment / N=10 / Exp 38) in
   the same results commit.

## Cross-references

- [37_cross_session_graduation.md](../../experiments/37_cross_session_graduation.md)
  — pre-registration (now carries the 2026-06-05 amendment).
- [protocols/37_cross_session_graduation_reproduction.md](../../experiments/protocols/37_cross_session_graduation_reproduction.md)
  — protocol (now carries the SD-shift rule in §1 and §D).
- [exp37_metric_pivot.md](exp37_metric_pivot.md) — the prior amendment
  (metric pivot to positive_approach_engagement_fraction). This amendment
  stacks on top.
- [cradle_activation_fixes.md](cradle_activation_fixes.md) — the source
  of PRs C/D/E that landed the drive calibration; everything from this
  amendment runs on top of that infrastructure.
- [behavioral_graduation_candidates.md](../behavioral_graduation_candidates.md)
  — Tier 1 row 1 is the recipient of the verdict; status flip happens
  after the analyzer produces its report.
