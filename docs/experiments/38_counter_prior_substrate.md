# Exp 38 — Counter-Prior Substrate (pre-registration)

**Status:** FIRED 2026-06-11 (Sonnet 4.6, 60/60, $14.26) — verdict **COUNTER-PRIOR — dominance demonstrated**. §5 metrics were FROZEN before the fire; results appended below. Follow-up: re-run at Qwen14B + Qwen32B per §8.
**Plan:** [docs/plans/counter_prior_substrate_experiment.md](../plans/counter_prior_substrate_experiment.md)
**Extends:** the Exp 37 cross-session harness (`scripts/benchmark_cross_session.py`) + analyzer (`scripts/analyze_exp37.py`). This experiment does **not** rebuild that apparatus — it adds one scenario, one metric channel, and one analyzer verdict path.
**Companion (reproduction):** the Exp 37 reproduction protocol ([protocols/37_cross_session_graduation_reproduction.md](protocols/37_cross_session_graduation_reproduction.md)) applies verbatim — same arms, resume mechanism, cost cap, and operational playbook. Only the scenario selection (`--scenario counter_prior`) and analyzer scenario list (`--scenarios fire_pit,deceptive_fire`) differ.

> **Freeze discipline.** §5 metrics below are locked before the first fire. The Exp 37 `sharp_rock` metric-drift (three post-hoc pivots) is the cautionary tale. No metric may be re-tuned after seeing data; a genuinely new question spawns a *new* pre-registered iteration.

---

## 1. Why the Exp 37 null is not enough

Exp 37 returned a clean null/negative at Sonnet and Qwen14B: carrying substrate state across sessions (Arm B) produced no behavioral gain over a fresh session (Arm A). We read this as **LLM-prior-dominance**. But the cradle scenarios are **prior-aligned** — Sonnet already knows fire→warm, sharp-rock→avoid — so a null is expected *whether or not the substrate works*. The design cannot separate:

- **(H-dominance)** the prior dominates and the substrate is behaviorally inert, vs
- **(H-redundant)** the substrate works but is redundant because the prior already gets it right.

This experiment builds a world where **the prior is wrong**: a hearth that reads as an ordinary warm fire but whose `warm_self` affordance is inverted to cause pain. The correction is learnable only from experience (the carried substrate). The result is diagnostic either way:

- agent keeps warming the harmful hearth → **dominance demonstrated** (a *stronger* result than the Exp 37 null — there *was* a gap and the substrate didn't fill it);
- agent overrides the prior using carried experience → **the substrate demonstrably matters**.

---

## 2. Design

| Scenario | Entity | `warm_self` contingency | Prior | Role |
|---|---|---|---|---|
| `fire_pit` (existing) | `cradle_fire_pit` | safe (+0.2 thermal, within comfort band) | correct | **consistent control** |
| `deceptive_fire` (NEW) | `cradle_false_hearth` (entity `hearth`) | harmful (`arms.thermal` +0.6 breaches comfort_band 0.5 → thermal_contact failure → PainBus → NAc aversive) | **wrong** | **counter-prior** |

The inversion lives **only** in `cradle_false_hearth.yaml::warm_self.self_effect`. The percept (entity name `hearth`, all descriptions) reads as a benign warm fire — enforced by the telegraph-denylist test (§7). `touch` stays aversive; `observe` stays safe — only `warm_self` is inverted.

**Separate worlds (not co-presence).** The two scenarios are **distinct worlds**: the consistent scenario routes to the `cradle` arc (fire_pit is the only warmth source); the deceptive scenario routes to a dedicated `cradle_deceptive` arc where the hearth **replaces** fire_pit. An earlier design co-activated *both* entities in one world — the agent simply warmed the familiar safe fire_pit and ignored the hearth, so hearth-engagement was ≈0 and the warm_self signal was unmeasurable (caught at ~30 min into the first fire via the early validity check). `cradle_deceptive` is derived from `cradle` by swapping `items/cradle_fire_pit` → `items/cradle_false_hearth` (and the narrator-facing "fire pit" wording → "hearth"), so the two arcs stay structurally in lockstep — same phases, acts, drives; only the warmth entity (and thus `warm_self`'s contingency) differs. Regression guards: `test_exp37_harness_smoke.py::test_deceptive_scenario_routes_to_hearth_only_world` / `test_fire_pit_scenario_routes_to_fire_pit_only_world` / `test_cradle_and_deceptive_arcs_are_structural_lockstep`.

**Thermal (hot) inversion chosen** (not the cold-burn variant): `warm_self.self_effect = {arms.thermal: 0.6, core_temperature: 0.15}` — byte-identical to `cradle_fire_pit`'s `touch`, identical plumbing.

**Arms** unchanged from Exp 37: `A, B, C, B-wire-a-off, B-wire-1-off, B-nac-bias-off`. Arm A is fresh; Arm B + ablations resume Arm A's `session_id`; ablation env vars per `ARM_ENV`. The substrate accumulates in A's session; B inherits it. Both scenarios run with **identical goals** for A and B — B's only edge is the resumed substrate.

`sharp_rock` is **excluded** from the primary run (its metric is structurally absent — see §6).

---

## 3. Metric channel — `warm_self_engagement_fraction`

The interaction is measured on the **warm_self-engagement-fraction**: the share of on-target engagement (warm_self / observe / touch / pick_up) that is the entity's own `warm_self` affordance.

**Implementation note / deviation from plan §6.2.** The plan suggested reusing `positive_approach_engagement_fraction`. That metric's numerator is `direct_approach_tools`, which for the deceptive hearth is **empty** (warm_self is harmful, not a "positive approach"). Reusing it would make the metric structurally 0 for `deceptive_fire`, collapsing the interaction to a single-scenario quantity (silent invalidity — exactly the §7 trap). Resolution: a **dedicated `warm_self_engagement_fraction`** computed from the entity's own `{entity}_warm_self` tool, decoupled from the safe/approach label. For `fire_pit` (where warm_self *is* the approach affordance) it equals `positive_approach_engagement_fraction` exactly, so the matched control behaves identically. `deceptive_fire`'s `positive_approach_engagement_fraction` stays structurally 0 and is reported **N/A** by the structural-absence detector (§6).

---

## 4. The deceptive entity

`src/maxim/_data/components/items/cradle_false_hearth.yaml` — modelled on `cradle_fire_pit.yaml`, entity `name: hearth`, tools `hearth_warm_self` / `hearth_observe` / `hearth_touch`. Activated as the sole warmth source in the `cradle_deceptive` arc's `exploration` phase (see §2 "Separate worlds"); the `deceptive_fire` goal ("…glowing hearth") routes to that arc. Per-scenario `FAILURE_CLASS` isolation additionally scopes metrics to the named entity's own tools.

---

## 5. Pre-registered hypotheses & metrics (FROZEN)

All comparisons are between-arm at 5 paired trials × 2 scenarios × 6 arms = 60 runs.

### Primary 1 — the interaction (substrate counter-prior learning)

> `interaction = Δ_deceptive(B − A) − Δ_consistent(B − A)` on `warm_self_engagement_fraction`.
> **PASS iff** `interaction / pooled_A_sd ≤ −1.0` (negative by ≥1.0 SD of the pooled Arm-A baseline).

- `pooled_A_sd` = SD of Arm-A `warm_self_engagement_fraction` pooled across both scenarios.
- Zero-SD fallback: if `pooled_A_sd` is 0/undefined, PASS on a strictly negative interaction.
- A main-effect-only reduction (B warms less *everywhere*) does **not** count — that is general caution, captured by `avoids_both` below, not counter-prior learning.

### Primary 2 — first-contact isolation (the sharp cross-session metric)

> Let `P_arm(scenario)` = P(the agent's **first** engagement with the entity is `warm_self`), over trials (field `first_contact_warm_self`, None-trials dropped).
> `dec_drop = P_A(deceptive) − P_B(deceptive)`; `con_drop = P_A(consistent) − P_B(consistent)`.
> **PASS iff** `dec_drop > 0` **AND** `(dec_drop − con_drop) > 0`.

B's first contact fires before any in-session pain in B's own session, so avoidance there can only come from the carried substrate — isolating cross-session transfer from within-session learning. (N=5 per cell → low power; this is a sign-based primary, acknowledged as descriptive-leaning but pre-registered.)

### Secondary — ablation attribution

> On the **deceptive** scenario's `warm_self_engagement_fraction`, ≥1 of `B-wire-a-off` / `B-wire-1-off` / `B-nac-bias-off` **reverts toward Arm A** by ≥1 SD (reuses the Exp 37 `_compute_secondary` same-side shrinkage rule). If no ablation abolishes B's avoidance, the avoidance is not substrate-attributable.

### Verdict tree (FROZEN — emitted verbatim by the analyzer)

`substrate_signal = Primary 1 PASS AND Primary 2 PASS` (both pre-registered primaries required).

| Condition | Verdict label | Exit |
|---|---|---|
| `substrate_signal` AND ≥1 ablation reverts | **COUNTER-PRIOR — substrate matters** | 0 |
| `substrate_signal` AND 0 ablations revert | **COUNTER-PRIOR — void (not substrate-attributable)** | 4 |
| `avoids_both` (B reduces warm_self in BOTH scenarios by ≥1 SD of each scenario's Arm-A baseline; interaction not specific) | **COUNTER-PRIOR — void (general caution)** | 4 |
| otherwise (B keeps warming the deceptive hearth) | **COUNTER-PRIOR — dominance demonstrated** | 0 |

Both clean outcomes (**substrate matters**, **dominance demonstrated**) exit 0 — the experiment is diagnostic either way (§1). The two **void** outcomes exit 4 (confound / non-attributable → re-run or investigate). When the two primaries disagree, `substrate_signal` is False and the analyzer notes the disagreement.

**Outcome interpretation:**

- **substrate matters** — the substrate overrides a strong, wrong prior. The positive thesis result.
- **dominance demonstrated** — even direct cross-session pain does not override the LLM prior. Stronger than the Exp 37 null.
- **void (general caution)** — B got cautious everywhere; not counter-prior learning.
- **void (not substrate-attributable)** — B avoided the hearth but no ablation explains it (prompt or within-session memory).

---

## 6. Analyzer changes (shared `analyze_exp37.py`)

1. **Structural-absence detection.** A PRIMARY_METRIC identical across *every arm/trial* in a scenario (zero variance everywhere → never exercised) reports **N/A / inconclusive**, NOT FAIL, and is excluded from overall-verdict gating. This folds the Exp 37 `sharp_rock` artifact (`positive_approach_engagement_fraction` is a fire-pit metric, structurally 0 for sharp_rock — previously a false FAIL that dragged the overall to investigation) AND handles `deceptive_fire`'s structurally-0 `positive_approach_engagement_fraction`.
2. **Counter-prior interaction.** `compute_counter_prior` computes Primary 1 + Primary 2 + ablation reversion and emits the §5 verdict, but only when both the consistent control and deceptive scenarios are present — an Exp 37 run (no `deceptive_fire`) falls through to the legacy per-scenario verdict unchanged.

---

## 7. Validity guards (where this silently becomes invalid)

1. **Prompt must not reveal the inversion** — enforced by the telegraph-denylist test (`tests/behavioral/test_exp37_harness_smoke.py::test_cradle_false_hearth_percept_does_not_telegraph_inversion`). The world, not the words, carries the twist.
2. **Same prompt for A and B** — already true in the harness (B resumes A with the same goal); no deceptive-specific hints are added.
3. **Within-session confound** — Arm A learns within its session after first pain. The cross-session claim rests on **B's first-contact action** (Primary 2), not B's session aggregate. Keep `--sim-max-turns` modest so A's session doesn't over-train.
4. **The inversion must actually hurt** — `test_cradle_false_hearth_warm_self_breaches_comfort_band` pins that `warm_self.self_effect.arms.thermal` exceeds the infant arms' `comfort_band`; if it stops breaching, the hearth no longer hurts and the experiment is void.
5. **Pre-registration freeze** — §5 is locked before firing. No metric pivots post-hoc.

---

## 8. Run plan

- **Sonnet-first** (dominance is the live question where the model is strong). Reuse the Exp 37 operational playbook: `--cost-cap`, `--resume`, run solo, `unset MAXIM_LLM_PROFILE`, cloud large-lane skips local spawn.
- **Scope:** 2 scenarios × 6 arms × 5 trials = 60 runs, ~$16 at Sonnet (mirrors Exp 37). `sharp_rock` excluded.
- **Commands:**
  - Fire: `python scripts/benchmark_cross_session.py --scenario counter_prior --model claude-sonnet --trials 5 --cost-cap 20 --out docs/experiments/data/38_results_sonnet.jsonl`
  - Analyze: `python scripts/analyze_exp37.py --in docs/experiments/data/38_results_sonnet.jsonl --scenarios fire_pit,deceptive_fire --trials 5 --out docs/experiments/38_counter_prior_substrate.md --heading-suffix "sonnet"`
- **Follow-up:** if Sonnet shows dominance, re-run at Qwen14B + Qwen32B to place the counter-prior result on the same scale axis as the Exp 37 cross-model picture (Qwen32B's anomalous +1.43 SD signal is the one to watch).

---

## 9. Regression guard / experiment citation

- **Regression guard (engineering):** `tests/behavioral/test_exp37_harness_smoke.py` (FAILURE_CLASS-vs-YAML pre-flight extended to `deceptive_fire`; first-contact extraction; telegraph denylist; comfort-band breach) + `tests/behavioral/test_exp37_analyzer_smoke.py` (structural-absence → N/A; counter-prior interaction direction + four verdicts; first-contact extraction).
- **Experiment (behavioral):** this doc — the matched-pair interaction + first-contact verdict (results appended below by the analyzer on each fire).

---

<!-- Analyzer appends "## Results — <suffix>" sections below this line on each fire. -->

---

## Results — Sonnet 4.6 (2026-06-11)

Source: `docs/experiments/data/38_results_sonnet.jsonl` · Analyzer version: `1.0` · Schema: `1.0`

### Overall verdict: **COUNTER-PRIOR — dominance demonstrated**

B keeps warming the deceptive hearth — even direct cross-session pain does NOT override the LLM's fire→warm prior. Dominance demonstrated: a stronger result than the Exp 37 null (there WAS a behavioral gap and the substrate did not fill it).

### Counter-prior interaction (Exp 38 primary)

**Verdict: COUNTER-PRIOR — dominance demonstrated**

B keeps warming the deceptive hearth — even direct cross-session pain does NOT override the LLM's fire→warm prior. Dominance demonstrated: a stronger result than the Exp 37 null (there WAS a behavioral gap and the substrate did not fill it).

**Interaction primary — warm_self-engagement-fraction**

| Quantity | Value |
|---|---|
| Δ deceptive (B − A) | 0.2067 |
| Δ consistent (B − A) | 0.1167 |
| Interaction (Δ_dec − Δ_con) | 0.0900 |
| Pooled Arm-A SD | 0.2250 |
| Interaction in SD units | 0.40 |
| Predicted | ≤ −1.0 SD → **FAIL** |

**First-contact isolation — P(warm_self on first contact)**

| Arm | Deceptive | Consistent |
|---|---|---|
| A | 0.67 | 0.80 |
| B | 1.00 | 1.00 |

Deceptive drop (A − B) = -0.33; consistent drop = -0.20; first-contact interaction = -0.13 → **FAIL** (need deceptive-drop > 0 AND interaction > 0).

**Secondary — ablation reversion on the deceptive hearth (≥1 must revert)**

| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |
|---|---|---|---|---|---|
| Wire-A annotation off | 0.2667 | 0.4733 | 0.6300 | -0.62 | **FAIL** |
| Wire 1 variance annotation off | 0.2667 | 0.4733 | 0.4200 | 0.21 | **FAIL** |
| NAc reward bias zeroed | 0.2667 | 0.4733 | 0.3133 | 0.63 | **FAIL** |

Ablation reversion hits: **0 / 3**

### Scenario: `fire_pit`

**Primary + isolation**

| Arm | Mean | Predicted | Verdict |
|---|---|---|---|
| A | 0.5333 | baseline · 95% band [0.5000, 0.6500] | — |
| B | 0.6500 | Δ = +1.57 SD (need ≥+1.0 SD) | **PASS** |
| C | 0.7267 | ∈ A's band | **FAIL** |

Robustness (legacy per-action failure rate, decrease): FAIL — DIVERGES from positive-approach-engagement primary; see protocol §1 (substrate may be biasing warm_self without reducing touch, or vice versa)

**Corroborating metrics (≥1 must pass)**

| Metric | A mean ± SD | B mean | Δ in SD units | Direction | Verdict |
|---|---|---|---|---|---|
| Affordance-preference safe-fraction (safe-on-target / on-target total) | 0.6333 ± 0.1264 | 0.7400 | 0.84 | increase | **FAIL** |
| Tool-class diversity (fewer dead-end tools tried) | 9.2000 ± 0.4472 | 8.2000 | -2.24 | decrease | **PASS** |
| Time-to-safe-steady-state (turns to 3 consecutive zero-failure turns; None censored to turn_count_binned+1) | 2.0000 ± 0.0000 | 2.0000 | — | decrease | **FAIL** — Zero SD on Arm A AND zero shift on B (degenerate). |
| Time-to-first-warm-self (action index of first warm_self; None censored to turn_count_binned+1) | 0.8000 ± 0.4472 | 0.4000 | -0.89 | decrease | **FAIL** |

Corroborating hits: **1 / 4**

**Descriptive corroborating — `fire_approach_action_count` (NOT pre-reg gated)**

| Arm | Mean count |
|---|---|
| A | 1.80 |
| B | 3.00 |

Δ (B − A) = 1.20; predicted direction: same_or_higher.

**Secondary criterion — ablation attribution (≥1 must shrink Arm B's delta)**

| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |
|---|---|---|---|---|---|
| Wire-A annotation off | 0.5333 | 0.6500 | 0.6200 | 0.40 | **FAIL** |
| Wire 1 variance annotation off | 0.5333 | 0.6500 | 0.4667 | 0.67 | **FAIL** — Ablation overshoots past Arm A baseline (B side: +0.1167, ablated side: -0.0667). Opposite-direction effect, not shrinkage — does NOT count as secondary-criterion PASS. |
| NAc reward bias zeroed | 0.5333 | 0.6500 | 0.6633 | -0.18 | **FAIL** |

Secondary hits: **0 / 3**

**Notes / warnings**

- Arm C mean 0.7267 for fire_pit falls outside Arm A's band [0.5000, 0.6500] — 'general caution' confound.
- Primary / robustness divergence on fire_pit: positive-approach-engagement primary=True vs per-action-failure-rate robustness=False. Substrate may be biasing toward warm_self without reducing touch (or vice versa). Investigate before claiming the verdict (per protocol §1).
- Arm C mean 0.0500 for fire_pit falls outside Arm A's band [0.0833, 0.1583] — 'general caution' confound.


### Scenario: `deceptive_fire`

**Primary + isolation**

| Arm | Mean | Predicted | Verdict |
|---|---|---|---|
| A | 0.0000 | baseline · 95% band [0.0000, 0.0000] | — |
| B | 0.0000 | Zero-SD fallback (need ≥+1.0 SD) | **N/A** |
| C | 0.0000 | ∈ A's band | **PASS** |

Robustness (legacy per-action failure rate, decrease): FAIL

**Corroborating metrics (≥1 must pass)**

| Metric | A mean ± SD | B mean | Δ in SD units | Direction | Verdict |
|---|---|---|---|---|---|
| Affordance-preference safe-fraction (safe-on-target / on-target total) | 0.1167 ± 0.1624 | 0.0000 | -0.72 | increase | **FAIL** |
| Tool-class diversity (fewer dead-end tools tried) | 7.2000 ± 4.0249 | 8.4000 | 0.30 | decrease | **FAIL** |
| Time-to-safe-steady-state (turns to 3 consecutive zero-failure turns; None censored to turn_count_binned+1) | 1.2000 ± 1.0954 | 2.0000 | 0.73 | decrease | **FAIL** |
| Time-to-first-warm-self (action index of first warm_self; None censored to turn_count_binned+1) | 8.2000 ± 10.4019 | 2.0000 | -0.60 | decrease | **FAIL** |

Corroborating hits: **0 / 4**

**Secondary criterion — ablation attribution (≥1 must shrink Arm B's delta)**

| Ablation | A mean | B mean | Ablated mean | Shrinkage (SD units) | Verdict |
|---|---|---|---|---|---|
| Wire-A annotation off | 0.0000 | 0.0000 | 0.0000 | — | **FAIL** — Insufficient data for ablation comparison. |
| Wire 1 variance annotation off | 0.0000 | 0.0000 | 0.0000 | — | **FAIL** — Insufficient data for ablation comparison. |
| NAc reward bias zeroed | 0.0000 | 0.0000 | 0.0000 | — | **FAIL** — Insufficient data for ablation comparison. |

Secondary hits: **0 / 3**

**Notes / warnings**

- PRIMARY_METRIC 'positive_approach_engagement_fraction' is structurally absent for deceptive_fire (identical across every arm/trial — the approach affordance was never exercised). Reporting primary as N/A / inconclusive, not FAIL; this scenario is excluded from the overall-verdict gating.


