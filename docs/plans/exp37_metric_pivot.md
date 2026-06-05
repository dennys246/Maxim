# Exp 37 — Primary Metric Pivot

**Status:** DRAFT 2026-06-04
**Author:** Denny + Claude (post-cradle-smoke session)
**Blocks:** Exp 37 re-fire, behavioral_graduation_candidates.md Tier 1 row 1 → Earned
**Supersedes:** the pre-reg-amendment 2026-05-31 swap to `per_action_failure_rate` (which fixed binning but did NOT fix the underlying degeneracy)

## Problem

The 2026-06-04 cradle smoke (5 turns of `maxim --sim cradle --embodiment bodies/infant_humanoid --language-model qwen2.5-14b-instruct` from a peer machine, routed to the leader's Qwen2.5-14B) confirmed that PR D + PR E's calibration chain works end-to-end: cold body → drive deviation → warmth-seeking → `fire_pit_warm_self` → self_effect write → homeostasis restored. The positive substrate edge forms exactly as designed (NAc: `tool:fire_pit_warm_self → positive, RPE=+0.50`).

But the smoke also revealed a **subtler failure mode than `cradle_activation_fixes.md` Finding C anticipated.** The plan doc's framing was "LLM avoids fire because of adult priors → no aversive substrate." Actual observation: **the LLM seeks warmth correctly, but finds the SAFE warming path (`warm_self`, `shelter`) and never NEEDS to touch the fire.** Adult priors + methodical exploration + available safe affordances = perfect avoidance of the failure-class affordances.

### Observed action sequence (5 turns, Arm-A-equivalent run on PR E branch)

| Turn | Action | Plan text | NAc valence |
|---|---|---|---|
| 1 | `sense_cool_air` | "I wonder about its intensity" | +0.50 |
| 2 | `fire_pit_warm_self` | "if standing closer to the fire will help me feel warmer" | +0.50 |
| 3 | `cool_air_feel` | "how the cool draft feels on my skin" | +0.50 |
| 4 | `cool_air_shelter` | "shiver despite the fire's warmth … moving to a sheltered spot" | +0.50 |
| 5 | `examine(cute_stuffed_teddy_bear)` | imagined entity, no real effect | +0.50 |

**`fire_pit_touch` count: 0.** `infant_humanoid_pick_up(fire_pit)` count: 0. The smoke ran 5 of 12 budgeted turns before manual stop; the pattern was stable and unlikely to flip in the remaining 7.

### Why this breaks the existing primary metric

`failure_class_action_count = count(fire_pit_touch | pick_up(fire_pit))`. If Arm A's mean across 5 trials is consistently 0:

- `a_mean = 0`, `a_sd = 0`, `a_band = [0, 0]`
- Variance-survival requires `Arm B mean < Arm A's 2.5th-percentile = 0`. Mathematically impossible — `failure_class_action_count` is a non-negative integer, so `B < 0` cannot hold.
- Analyzer verdict: **PARTIAL — investigation gate (exit 4)**.

The zero-SD fallback in `_compute_corroborating` (which handles zero-SD on integer metrics by accepting any directional shift) is for *corroborating* metrics, not the primary. The primary gate explicitly uses percentile-band comparison and will fail when A's band collapses to a point.

## Decision space

Three paths considered; reasoning for each:

### Path 1 — Calibration treadmill: force higher touch propensity

Tighten the world to make `warm_self` insufficient (e.g., make cool_air write to body continuously each tick rather than per-action, set magnitudes so `warm_self` alone can't reach the comfort band, set `touch` warming bonus larger than `warm_self`). Forces the agent into the touch/burn trade-off.

**Why not:** This is the path the original `cradle_activation_fixes.md` P2 took. PR E shipped that calibration. Doing it again at a tighter setting risks an infinite-iteration loop ("tune until experiment produces the result we want"). At some point this crosses from experimental design into confirmation-bias engineering. Also: tuning the world specifically to force touch behaviors makes the substrate-transfer claim ("B inherits A's touch→pain memory") trivially true rather than informative — the transfer would have nothing to actually generalize about, because the touch was forced by world calibration, not chosen by the agent.

Saves it for a follow-up `Exp 37-b` (substrate-primary, decoupled from Exp 37's LLM-AUT scope) per the post-smoke discussion 2026-06-04. The cool_air-ambient-drift design is bio-realistic and worth doing — just not as Exp 37's unblock.

### Path 2 — Reframe primary to a metric that has variance under LLM-AUT

Replace `failure_class_action_count` with a metric whose Arm A baseline reliably has non-zero variance because it measures something the LLM-AUT *does* engage with, rather than something it avoids.

The substrate-transfer claim under LLM-AUT is "the substrate carries learned associations across sessions, biasing the LLM's selection toward known-good options earlier than naive exploration would." That's measurable as **how strongly Arm B's agent prefers `fire_pit_warm_self` over other on-target fire engagement options**, given that BOTH arms eventually engage with the fire.

### Path 3 — Use the narrative-reflex aversion as the negative edge

The `fire_burn` reflex (Layer 3 of the three-layer sensation model) fires on every percept mentioning fire keywords, producing `damage_component` events that NAc records as aversive. In the smoke, the reflex fired in 5/5 turns at intensities 0.30 / 0.15 / 0.124 / 0.107 / 0.084 (refractory-dampened). That IS substrate-accumulating aversion — just via narrative interpretation, not direct touch.

**Why not (for now):** Two structural problems.

1. The narrative reflex's firing rate depends on the *narrator's percept text*, not on the agent's behavior or the substrate state. Arm A and Arm B would see roughly the same reflex firings because the narrator is scenario-deterministic. The substrate doesn't change WHEN reflexes fire, only what they MEAN downstream. So measuring reflex count gives no signal about substrate transfer.
2. Reflex-triggered `damage_component` events aren't currently in `actions.jsonl`'s recorded action shape (they're logged at the `[REFLEX]` system layer, not via the executor's action recorder). Wiring them through would require schema changes, harness updates, and a `_format_version` bump — substantial work for a metric that doesn't actually differentiate A from B.

Path 3 stays viable as a *secondary* corroborating measurement once it's wired (e.g., "fear-mediated tool gating activates earlier on B"), but isn't the primary fix.

## Chosen path: Path 2

### Proposed primary metric

```
positive_approach_engagement_fraction =
    fire_pit_warm_self_action_count
    /
    (fire_pit_warm_self_action_count + fire_pit_observe_action_count + fire_pit_touch_action_count + pick_up_fire_pit_action_count)
```

Numerator: positive-approach actions only. Denominator: ALL on-target fire engagement (positive + neutral + aversive). Sessions with zero on-target engagement get the denominator clamped to 1 → metric value = 0 (consistent with "agent never engaged with target").

**Predicted direction:** Arm B HIGHER than Arm A (substrate-transferred positive edge biases B toward warm_self when it does engage with the fire). Direction: `increase`.

**Why this works under LLM-AUT:**

- Arm A baseline: the LLM EXPLORES on-target affordances during the session. In the smoke, turn 2's choice was `warm_self`; in 5 trials some sessions might also hit `observe` or even rarely `touch`. Non-zero variance expected.
- Arm B with substrate: NAc's `warm_self → positive` bias makes warm_self more attractive when the agent considers any fire engagement. Fewer exploratory `observe`s, fewer rare `touch`es. Higher ratio.
- Has variance: ratio is a continuous value in [0, 1] across trials.
- Robust under zero-engagement edge case: clamped denominator → 0, doesn't NaN.

**Why not just `warm_self_count / total_actions`:** confounded by total session length. Long sessions where the agent explores many non-fire affordances dilute the signal. The ratio over on-target-engagement specifically controls for "how much the agent engages with fire at all," isolating the substrate-driven preference shift.

### Secondary corroborating metrics (descriptive, retained from PR E)

- `fire_approach_action_count` (raw warm_self count) — already shipping in PR E. Stays.
- `affordance_preference_safe_fraction` — already shipping. Stays as broader safe-vs-failure ratio.
- New: `time_to_first_warm_self` (turn index of first warm_self, None if never) — measures "does B converge faster?" descriptively. Censored to `total_turns + 1` for never-reached sessions (mirrors `time_to_safe_steady_state_turns` handling).

The existing pre-reg variance-survival rule transfers to the new primary: Arm B mean must lie ABOVE Arm A's 97.5th-percentile band (direction flipped because higher is better). Isolation check: Arm C within A's band. Same three ablation arms.

## Implementation breakdown

### A. Harness (`scripts/benchmark_cross_session.py`)

Add to `FAILURE_CLASS["fire_pit"]`:

```python
"direct_engagement_tools": frozenset({"fire_pit_warm_self", "fire_pit_observe", "fire_pit_touch"}),
"body_engagement_rules": (("infant_humanoid_pick_up", "object", "fire_pit"),),
```

For `sharp_rock` keep parallel structure (note: sharp_rock has no positive analog so its `positive_approach_engagement_fraction` would be structurally 0; scenario asymmetric by design, matching the existing `fire_approach_action_count` asymmetry).

In `compute_metrics`:

```python
positive_count = sum(1 for a in actions if a.get("tool") == "fire_pit_warm_self")
engagement_count = sum(1 for a in actions if _is_engagement(a, rules))
positive_fraction = positive_count / engagement_count if engagement_count > 0 else 0.0
```

Add to record output:

```python
"positive_approach_engagement_fraction": positive_fraction,
"fire_pit_engagement_count": engagement_count,
"time_to_first_warm_self_turn": <int or None>,
```

Bump `SCHEMA_VERSION` (probably 1.0 → 1.1 since this is a schema addition) and update the `_format_version` invariant per CLAUDE.md.

Extend `_assert_failure_class_matches_yaml` to validate `direct_engagement_tools` against the YAML.

### B. Analyzer (`scripts/analyze_exp37.py`)

Swap `PRIMARY_METRIC`:

```python
PRIMARY_METRIC = "positive_approach_engagement_fraction"
ROBUSTNESS_METRIC = "per_action_failure_rate"  # old primary becomes the robustness cross-check
```

Flip the direction of `_compute_primary_isolation`'s percentile check:

- Old: `B.mean < A.p2.5` (lower is better)
- New: `B.mean > A.p97.5` (higher is better)

Add `time_to_first_warm_self_turn` to `CORROBORATING_METRICS` with `none_handling="censor_to_max"` (mirroring the existing time-to-steady-state pattern).

Update the verdict-matrix table in `render_markdown` to use the new metric's label and direction.

### C. Pre-reg amendment

Amend `docs/experiments/37_cross_session_graduation.md` with a dated entry:

> **2026-06-XX amendment (per `docs/plans/exp37_metric_pivot.md`):** primary metric pivoted from `per_action_failure_rate` (count of touch + pick_up actions) to `positive_approach_engagement_fraction` (warm_self share of fire engagement). Direction: `increase`. Rationale: the cradle smoke 2026-06-04 confirmed that the LLM-AUT achieves thermal homeostasis via the safe `warm_self` affordance and never touches the fire under normal calibration, producing structurally-zero failure_class counts that defeat the variance-survival math. The substrate-transfer claim under LLM-AUT is more accurately framed as "B's substrate-biased preference for the positive-approach affordance is stronger than A's exploration-driven preference," which `positive_approach_engagement_fraction` directly measures. The `failure_class_action_count` field is retained as a descriptive corroborating signal (when non-zero, it carries the direct-touch-aversion claim the pre-reg originally targeted).

Update `docs/experiments/protocols/37_cross_session_graduation_reproduction.md`:
- §1 (primary metric): document the new pivot
- §D (analysis): update verdict-matrix table to reflect new direction
- §2 (failure-class detection rules): add the engagement-tools table

### D. Tests

- `tests/behavioral/test_exp37_harness_smoke.py`:
  - Add `positive_approach_engagement_fraction` and `fire_pit_engagement_count` and `time_to_first_warm_self_turn` to `_REQUIRED_FIELDS`
  - New unit tests for the new metric (empty engagement → 0.0; warm_self-only → 1.0; mixed → expected ratio; sharp_rock structurally 0)
- `tests/behavioral/test_exp37_analyzer_smoke.py`:
  - Update fixture builders to emit the new fields
  - Update existing engineered-variance fixtures to flip direction (B's primary now goes UP not DOWN)
  - Existing EARNED/PARTIAL test cases stay structurally the same; only the metric direction flips

### E. Smoke validation before re-fire

Run the cradle smoke at least once with the new metric live and confirm:
- `positive_approach_engagement_fraction` is non-zero on Arm A (sanity)
- Variance across 2-3 short manual runs at different seeds is non-trivial
- Time-to-first-warm-self records integer turn indices

Smoke timing: ~5-8 min per Arm A run at Qwen14B latency from the peer. Three seeds = ~30 min total.

## Cost estimate

| Item | Effort |
|---|---|
| Harness changes (A) | 1-2 hours |
| Analyzer changes (B) | 2-3 hours |
| Pre-reg + protocol doc amendment (C) | 1 hour |
| Tests (D) | 2-3 hours |
| Validation smoke (E) | 1-2 hours wall (mostly waiting on Qwen) |
| Review round | 1 hour |
| **Total** | **8-12 hours** |

Plus the actual Exp 37 re-fire (65 runs × ~12 turns × ~235s/turn ≈ 50 hours wall on the Mac Mini; runs in background, just needs monitoring).

## Risks

1. **The new metric might also have low variance.** If Arm A consistently hits warm_self on turn 2 and never touches anything else fire-related, the ratio stays close to 1.0 across all 5 trials → degenerate again. Mitigation: the smoke validation step E catches this BEFORE the $14 re-fire. If degenerate, fall back to `time_to_first_warm_self` as primary (which has natural variance across exploration paths).

2. **The "exploration" arm A pattern may not be naturally variable.** The smoke showed 4/5 actions on cool_air affordances + 1 on warm_self. Across 5 trials at different seeds the LLM might consistently produce similar sequences. Mitigation: increase trial count if early runs show low cross-trial variance. Pre-reg's 5-trial budget is per the original metric; if we need 10 trials for the new metric to have stable variance, that doubles the cost but stays under $30 still cheap.

3. **The substrate-transfer claim under the new metric is WEAKER than under the original.** "B reaches warm_self faster" is more easily explained by "B's prompt context includes a hippocampal recall of warm_self success" rather than NAc reward_bias. The ablation arms (Wire-A off, Wire-1 off, NAc-bias off) still help disentangle these, but the claim becomes "the substrate biases LLM selection," not "the substrate learns autonomously." This is consistent with the 1.0 framing ("bio-inspired LLM harness") per the 2026-06-04 framing discussion.

4. **Imagined-entity noise (P3 in `cradle_activation_fixes.md`).** Turn 5's `examine(teddy_bear)` was an imagined entity contaminating the action stream. The new metric is denominator-protected against this (teddy_bear actions don't count in either numerator or denominator), but it does dilute signal density per session. Worth noting in the results discussion.

## Open questions

1. **Should the schema bump be 1.0 → 1.1, or do we treat new fields as additive-default-zero and stay at 1.0?** The CLAUDE.md `_format_version` invariant treats additive fields with defaults as non-breaking. Recommendation: stay at 1.0, accept that older JSONL records (from pre-pivot runs that don't exist anyway since Exp 37 hasn't fired) would have the field missing.

2. **Should we drop the per-turn binning entirely?** It was retained at the 2026-05-31 amendment as a robustness check, but it's been structurally degenerate on cradle since day one (no say/respond). Drop it from the analyzer output entirely to reduce noise. Recommendation: yes drop.

3. **Should sharp_rock get a positive-approach analog?** Currently the scenario has only `examine` (safe-observe), `touch` (aversive), and `pick_up` (aversive). No positive-approach affordance. If we add one (e.g., `sharp_rock_polish` or `sharp_rock_balance` — touch lightly without grasping), the metric becomes symmetric across scenarios. Recommendation: defer — sharp_rock's asymmetry is informative on its own (validates that the metric is scenario-shape-sensitive rather than universally inflated).

## Sequencing

1. Ship this plan as a doc PR (`feat/v1-exp37-metric-pivot` branch). User review.
2. Implement A-D in a single PR (`feat/v1-exp37-metric-pivot` branch — same branch as the plan).
3. Validation smoke (E). If passes, mark plan COMPLETE.
4. Open the pre-reg amendment commit.
5. Fire Exp 37 (65 runs) — monitored background, ~2 days wall.
6. Analyzer run → results doc → `behavioral_graduation_candidates.md` row 1 status flip.

## Out of scope (explicitly)

- The cool_air-ambient-drift / "force touch via drive pressure" design → Exp 37-b, post-1.0 work
- Substrate-primary action selection → Exp 38 / Oasis
- Layer-3 narrative reflex redesign → separate concern, not part of this metric pivot
- The imagined-entity noise issue → 1.1 scenario polish
