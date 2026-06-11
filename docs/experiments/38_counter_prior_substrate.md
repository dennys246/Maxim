# Exp 38 — Counter-Prior Substrate (pre-registration)

**Status:** PRE-REGISTERED — metrics FROZEN, not yet fired (2026-06-10)
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

**Thermal (hot) inversion chosen** (not the cold-burn variant): `warm_self.self_effect = {arms.thermal: 0.6, core_temperature: 0.15}` — byte-identical to `cradle_fire_pit`'s `touch`, identical plumbing.

**Arms** unchanged from Exp 37: `A, B, C, B-wire-a-off, B-wire-1-off, B-nac-bias-off`. Arm A is fresh; Arm B + ablations resume Arm A's `session_id`; ablation env vars per `ARM_ENV`. The substrate accumulates in A's session; B inherits it. Both scenarios run with **identical goals** for A and B — B's only edge is the resumed substrate.

`sharp_rock` is **excluded** from the primary run (its metric is structurally absent — see §6).

---

## 3. Metric channel — `warm_self_engagement_fraction`

The interaction is measured on the **warm_self-engagement-fraction**: the share of on-target engagement (warm_self / observe / touch / pick_up) that is the entity's own `warm_self` affordance.

**Implementation note / deviation from plan §6.2.** The plan suggested reusing `positive_approach_engagement_fraction`. That metric's numerator is `direct_approach_tools`, which for the deceptive hearth is **empty** (warm_self is harmful, not a "positive approach"). Reusing it would make the metric structurally 0 for `deceptive_fire`, collapsing the interaction to a single-scenario quantity (silent invalidity — exactly the §7 trap). Resolution: a **dedicated `warm_self_engagement_fraction`** computed from the entity's own `{entity}_warm_self` tool, decoupled from the safe/approach label. For `fire_pit` (where warm_self *is* the approach affordance) it equals `positive_approach_engagement_fraction` exactly, so the matched control behaves identically. `deceptive_fire`'s `positive_approach_engagement_fraction` stays structurally 0 and is reported **N/A** by the structural-absence detector (§6).

---

## 4. The deceptive entity

`src/maxim/_data/components/items/cradle_false_hearth.yaml` — modelled on `cradle_fire_pit.yaml`, entity `name: hearth`, tools `hearth_warm_self` / `hearth_observe` / `hearth_touch`. Activated in the `cradle` arc's `exploration` phase alongside `cradle_fire_pit`; the `deceptive_fire` goal routes the narrator to surface the hearth. Per-scenario `FAILURE_CLASS` isolation scopes metrics to the named entity's own tools, so co-activation does not cross-contaminate the matched pair.

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