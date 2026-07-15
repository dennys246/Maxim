# Counter-Prior Substrate Experiment (Exp 38)

**Status:** DRAFT — pre-registration pending
**Author:** drafted 2026-06-10 (follow-up to Exp 37 Sonnet result)
**Depends on:** the Exp 37 cross-session harness (`scripts/benchmark_cross_session.py`) + analyzer (`scripts/analyze_exp37.py`) + the cradle SEM stack. This plan **extends** that harness; it does not rebuild it.

---

## 1. Motivation — why the Exp 37 null is not enough

Exp 37 (cross-session graduation) returned a clean **null/negative** at Sonnet and Qwen14B: carrying substrate state across sessions (Arm B) produced no detectable behavioral improvement over a fresh session (Arm A). We interpreted this as **LLM-prior-dominance** — the strong model already does the sensible thing from pretraining, leaving nothing for the substrate to add.

**The flaw in that inference:** the cradle scenarios are **prior-aligned**. Sonnet already "knows" fire → warm, sharp rock → avoid. So a null substrate result is *expected whether or not the substrate works* — there is no behavioral gap for it to fill. The current design **cannot distinguish**:

- **(H-dominance)** the LLM prior dominates and the substrate is behaviorally inert, vs
- **(H-redundant)** the substrate works fine but is redundant because the prior already gets it right.

Both produce the same null. To separate them we need a scenario where **the LLM's prior is *wrong*** and the correct behavior is learnable **only from experience** (the substrate's cross-session carry-forward). Then:

- agent follows the wrong prior → **H-dominance confirmed** (a *stronger* result than the current null — there *was* a gap and the substrate didn't fill it);
- agent overrides the prior using carried experience → **the substrate demonstrably matters** (the positive result the thesis wants).

This is the single highest-value next experiment. It is diagnostic either way.

Secondary cleanup folded in: the Exp 37 `sharp_rock` "degeneracy" (analyzer reported FAIL on a metric that was structurally 0 across *all* arms — `positive_approach_engagement_fraction` is a fire-pit metric inapplicable to sharp_rock). This plan adds **structural-absence detection** to the analyzer so a metric that never varies in a scenario reports **N/A / inconclusive**, not FAIL. See §6.3.

---

## 2. Design principle

A valid counter-prior test needs all three:

1. **Percept triggers the strong, real-world-correct prior.** The scene reads as an ordinary warm fire so Sonnet's `fire → warm_self` prior fires hard.
2. **This world inverts the contingency.** Warming at *this* entity causes pain, not comfort. The inversion lives **only in the world's affordance `self_effect`** — never in the prompt text.
3. **The correction is learnable only from experience.** Arm A and Arm B receive **identical prompts**; B's only edge is the resumed substrate. If the prompt ever hints "this fire hurts," the LLM can reason it out and the test is void.

Held constant against the prior by a **matched control**: a normal fire (existing `cradle_fire_pit`) where `warm_self` is genuinely safe. The substrate signal is the **interaction** — B reduces `warm_self` *specifically* when it's harmful (deceptive) and *not* when it's safe (consistent).

---

## 3. Scenarios & arms

| Scenario | Entity | `warm_self` contingency | Prior | Role |
|---|---|---|---|---|
| `fire_pit` (existing) | `cradle_fire_pit` | safe (+0.2 thermal, within comfort band) | correct | **consistent control** |
| `deceptive_fire` (NEW) | `cradle_false_hearth` | harmful (breaches comfort band → thermal_contact failure → PainBus → NAc aversive) | **wrong** | **counter-prior** |

`sharp_rock` is **excluded** from the primary criterion (kept optionally as an exploration/robustness scenario only — its metric is structurally absent per the §6.3 fix).

**Arms unchanged** from Exp 37 — reuse `ARMS = ("A", "B", "C", "B-wire-a-off", "B-wire-1-off", "B-nac-bias-off")` and the resume mechanism (Arm A fresh; Arm B + ablations resume Arm A's `session_id`; ablation env vars in `ARM_ENV`). The substrate accumulates in A's session; B inherits it.

---

## 4. The deceptive entity (`cradle_false_hearth.yaml`)

New file: `src/maxim/_data/components/items/cradle_false_hearth.yaml`. Modelled byte-for-byte on `cradle_fire_pit.yaml` **except the `warm_self.self_effect`**, which is inverted to breach the thermal comfort band (the same mechanism `cradle_fire_pit`'s `touch` uses today: `arms.thermal: 0.6` breaches `comfort_band 0.5` → `thermal_contact` failure → PainBus → NAc aversive memory).

**Non-negotiable validity rules for the YAML:**
- **Entity `name:`** must read as a benign warm fire and must NOT telegraph harm. Recommended `name: hearth` (tool names become `hearth_warm_self` / `hearth_observe` / `hearth_touch`). Do **not** name it `false_hearth` / `trap` / `deceptive_*` — the agent sees the entity name.
- **`description:` and every affordance `description:`** must be copy-identical in spirit to `cradle_fire_pit`'s ("A small fire/hearth that radiates heat", "Stand close to feel the radiated heat without touching it"). No "eerie", "unstable", "blue flame", "something feels wrong". The harm is in the `self_effect`, discoverable only by acting.
- **`warm_self.self_effect`** breaches the comfort band (e.g. `arms.thermal: 0.6`, `core_temperature: 0.15` — or a cold-burn variant `core_temperature: -0.6` if a freezing inversion is preferred; pick one and pre-register it). The result must be a real `thermal_contact` (or cold) failure that fires PainBus → NAc, identical plumbing to `fire_pit_touch`.
- **`touch`** stays aversive (consistency); **`observe`** stays safe.

Add `cradle_false_hearth` to a cradle arc phase's `world_entities` in `src/maxim/simulation/arcs.py` (`BUILTIN_ARCS["cradle"]`), parallel to how `items/cradle_fire_pit` is activated. The `deceptive_fire` scenario goal routes the narrator to surface the hearth.

---

## 5. Pre-registered hypotheses & metrics

Freeze these **before** the first fire (the Exp 37 `sharp_rock` metric drift is the cautionary tale).

**Primary — the interaction (substrate counter-prior learning):**
> `Δ_deceptive(B − A) − Δ_consistent(B − A)` on `warm_self`-engagement-fraction is **negative** by ≥ 1.0 SD of the Arm-A baseline.
>
> i.e. Arm B reduces `warm_self` *specifically* in the deceptive scenario (where it's harmful) and *not* in the consistent scenario (where it's safe). A main-effect-only reduction (B warms less everywhere) does **not** count — that's general caution, not counter-prior learning.

**Primary cross-session isolation — first-contact (the sharp metric):**
> On Arm B's **first encounter with the hearth** (before any in-session pain in B's own session), `P(warm_self)` is **lower** in `deceptive` than Arm A's first-contact `P(warm_self)`, and **not** lower in `consistent`.
>
> Avoidance on first contact — having never been hurt *in B's session* — can only come from the carried substrate. This isolates cross-session transfer from within-session learning.

**Secondary — ablation attribution:**
> `B-wire-a-off` / `B-wire-1-off` / `B-nac-bias-off` should **revert toward Arm A** (re-approach the harmful hearth). If an ablation does NOT abolish B's avoidance, B's avoidance came from the prompt or within-session memory, not that substrate wire — and the positive result is void for that wire.

**Outcome interpretation matrix:**

| First-contact result | Ablations | Verdict |
|---|---|---|
| B avoids deceptive (not consistent) | abolish avoidance | **Substrate matters** — overrides a strong wrong prior |
| B still warms the deceptive hearth | n/a | **Dominance demonstrated** — even direct cross-session pain doesn't override the prior |
| B avoids both deceptive AND consistent | — | Void — general caution confound, not counter-prior |
| Ablations don't abolish | — | Void — avoidance not substrate-attributable |

---

## 6. Implementation

### 6.1 Entity + arc (§4)
- `src/maxim/_data/components/items/cradle_false_hearth.yaml` (new).
- `src/maxim/simulation/arcs.py` — add `items/cradle_false_hearth` to a cradle phase `world_entities`.

### 6.2 Harness (`scripts/benchmark_cross_session.py`)
- `SCENARIOS` — add `"deceptive_fire"`.
- `SCENARIO_GOAL["deceptive_fire"]` = e.g. `"cradle infant explores the warm room with the glowing hearth"`.
- `FAILURE_CLASS["deceptive_fire"]` — `direct_failure_tools = {"hearth_warm_self", "hearth_touch"}`; `direct_approach_tools = frozenset()` (warm_self is no longer a positive approach); `direct_engagement_tools = {"hearth_warm_self", "hearth_observe", "hearth_touch"}`; `direct_safe_tools = {"hearth_observe"}`; body rules parallel to `fire_pit`.
- `compute_metrics` (line ~618) — add a `first_contact_warm_self: bool | None` field (did the agent `warm_self` on its first engagement with the scenario entity, and at what action index) so the analyzer's first-contact metric has a source. The existing `positive_approach_engagement_fraction` is reused as the warm_self-fraction channel.
- `validate_failure_class_against_yaml` (line ~942) — extend so it validates the new `deceptive_fire` tools against `cradle_false_hearth.yaml` (the pre-flight that pins FAILURE_CLASS to the YAML affordance names).

### 6.3 Analyzer (`scripts/analyze_exp37.py`) — two changes
1. **Structural-absence detection (folds the sharp_rock cleanup):** when a metric is identical across *every arm including A* in a scenario (zero variance everywhere → nothing was exercised), report **N/A / inconclusive** for that scenario, NOT FAIL. A FAIL falsely implies the hypothesis was tested and rejected. This is the direct fix for the Exp 37 `sharp_rock` artifact.
2. **Interaction + first-contact metrics:** add the §5 interaction primary (per-scenario direction; `deceptive` predicts DECREASE in warm_self-fraction, `consistent` predicts ≈0) and the first-contact isolation metric. The existing single-scenario `_compute_primary_isolation` becomes a building block; the new top-level criterion is the cross-scenario interaction.

### 6.4 Tests (regression guards)
- `tests/behavioral/test_exp37_harness_smoke.py` — extend the FAILURE_CLASS-vs-YAML pre-flight test to cover `deceptive_fire` / `cradle_false_hearth`; add a mock-action test asserting `first_contact_warm_self` is computed correctly.
- Analyzer unit test — structural-absence → N/A (pin the sharp_rock-class artifact); interaction-metric direction; first-contact extraction.
- A **percept-no-telegraph guard**: a test asserting `cradle_false_hearth.yaml`'s `description` + affordance descriptions contain none of a denylist of telegraph words (`trap`, `false`, `deceptive`, `eerie`, `unstable`, `wrong`, `danger`, `cold`-if-thermal-variant) — so a future edit can't silently leak the inversion into the prompt and void the experiment.

---

## 7. Validity guards (where this silently becomes invalid)

1. **Prompt must not reveal the inversion** — enforced by the §6.4 telegraph-denylist test. A and B get identical scene text; the world, not the words, carries the twist.
2. **Same prompt for A and B** — already true in the harness (B resumes A with the same `goal`); do not add deceptive-specific prompt hints.
3. **Within-session confound** — Arm A *will* learn within its session after the first pain. The cross-session claim rests on **B's first-contact action**, not B's session-aggregate. Keep `--sim-max-turns` modest so A's session doesn't over-train within-session in a way that swamps the cross-session signal.
4. **Pre-registration freeze** — lock §5 metrics (interaction + first-contact + ablation) in the experiment doc before firing. No metric pivots post-hoc (the Exp 37 lesson).

---

## 8. Run plan

- **Sonnet-first.** Dominance is the live question precisely because Sonnet is strong; run it where the answer matters most. Reuse the Exp 37 operational playbook (`project_exp37_sonnet_cloud_fire` memory): `--cost-cap`, `--resume`, run solo, `unset MAXIM_LLM_PROFILE`, cloud large-lane skips local spawn.
- **Scope:** 2 scenarios (`fire_pit` consistent + `deceptive_fire`) × 6 arms × 5 trials = 60 runs, ~$16 at Sonnet (mirrors Exp 37). `sharp_rock` excluded from the primary run.
- **Follow-up:** if Sonnet shows dominance, re-run at Qwen14B + Qwen32B to place the counter-prior result on the same scale axis as the Exp 37 cross-model picture (Qwen32B's anomalous +1.43 SD signal is the one to watch).

---

## 9. Out of scope

- Rebuilding the harness/analyzer — this **extends** Exp 37 (front-gate scope pressure: ride on existing infra).
- New bio-mechanisms — the experiment tests existing substrate wiring (Wire-A / Wire-1 / NAc reward bias), it does not add any.
- The arbitrary-contingency variant (a "no-prior" purer substrate test where the correct action is randomized per world) — a complementary future experiment, not this one. Counter-prior is the sharper discriminator for *dominance* specifically.

---

## 10. Regression guard / experiment citation

- **Regression guard (engineering):** `tests/behavioral/test_exp37_harness_smoke.py` (FAILURE_CLASS-vs-YAML pre-flight extended to `deceptive_fire`) + analyzer structural-absence unit test + the percept-no-telegraph denylist test.
- **Experiment (behavioral):** `docs/experiments/38_counter_prior_substrate.md` (to be created at pre-registration) — the matched-pair interaction + first-contact verdict.
