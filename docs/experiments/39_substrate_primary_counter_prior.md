# Exp 39 — Substrate-Primary Counter-Prior (pre-registration)

**Status:** PRE-REGISTERED — primary metric FROZEN below; triage gate must pass before the measured run counts.
**Graduates:** [behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md) Tier 1 **#6 (substrate-primary action selection)** — the LAST open Tier 1 entry.
**Builds on:** [grounded_language_acquisition.md](../plans/grounded_language_acquisition.md) Phase 0 (harness shipped, validation pending) + [38_counter_prior_substrate.md](38_counter_prior_substrate.md) (the LLM-AUT counter-prior).
**Plan / harness:** `--aut-mode substrate-primary` (cli), `propose_via_substrate` (agent_loop), `SubstrateTelemetry` (simulation/substrate_telemetry.py), `cradle_prelinguistic` arc.

---

## 1. Why this is the keystone

Exp 37 (prior-aligned) and Exp 38 (counter-prior) both ran **under LLM-AUT** — the LLM selects actions, the substrate annotates context. Across 4 frontier models the result was **dominance**: the LLM prior overrides the carried substrate. R1 showed the substrate is *causally active* (ablation-attributable) but *amplifies the prior, not the corrective experience*.

That leaves exactly one question the LLM-AUT design cannot answer (and the #2 entry explicitly names): **can the substrate drive adaptive behavior with the LLM removed from action selection?** Substrate-primary mode is that removal — actions come from `propose_via_substrate` (drives + `SensorEncoder`→EC + NAc causal links + confidence gate), no LLM prior in the action path. The narrator emits no prose; the AUT reads only sensor/drive state.

- **Positive →** graduates #6 *and* reframes the whole 1.0 story: "the substrate *does* drive adaptive cross-session behavior; it is merely masked by the LLM prior under LLM-AUT." Four dominance nulls become a coherent positive thesis.
- **Negative →** reframe #6 (substrate too weak to drive behavior even unmasked). Still settles the gate honestly.

The counter-prior (`hearth` whose `warm_self` hurts) is the ideal scenario: it gives a crisp behavioral signal (learn to stop warming the harmful affordance) that the prior cannot supply, since there is no prior in the action path.

**Cost:** substrate-primary makes **no LLM calls in the action path** → fast and free; runs locally. (A prose-silent narrator means no LLM anywhere in the prelinguistic arc.)

---

## 2. Setup deliverable (before any run)

**`cradle_prelinguistic_deceptive` arc** — derive from `cradle_prelinguistic` exactly as `cradle_deceptive` derives from `cradle` (reuse `_swap_fire_to_hearth_phase` in [simulation/arcs.py](../../src/maxim/simulation/arcs.py): `items/cradle_fire_pit` → `items/cradle_false_hearth`). Prelinguistic instructions are already empty, so only the `world_entities` swap applies. Add the routing + a regression test mirroring the `cradle_deceptive` ones.

---

## 3. Phase 0 — triage gate (feasibility; NOT the hypothesis test)

Run substrate-primary on the existing `cradle_prelinguistic` (safe) first and inspect `SubstrateTelemetry`. **All three must hold or the behavioral run does not count** (fix first):

1. **EC structure:** > 1 EC cluster forms, tied to repeating sensorimotor patterns (not the smooth-drift-collapses-to-one-cluster degeneracy found in [13_phase0_harness_smoke.md](13_phase0_harness_smoke.md)). If it collapses, fix the `interoception`-modality EC config / `SensorEncoder` basis before proceeding.
2. **NAc differentiation:** reward_bias differentiates drive-resolution events from null events.
3. **Proposals actually fire:** `propose_via_substrate` returns real actions (not `None` every tick from the confidence gate). Tune `MAXIM_NAC_MIN_CONFIDENCE` if the cold-start gate is too strict to ever propose.

This is the existing Phase-0 mechanism gate + a "does it ever act" check. A triage failure is a *finding about the mechanism's readiness*, not a metric pivot.

---

## 4. Phase 1 — pre-registered behavioral hypotheses & metrics (FROZEN)

Scenario: `cradle_prelinguistic_deceptive` (the harmful hearth), substrate-primary mode. N ≥ 5 seeds per arm. Frozen before the measured run.

**Primary 1 — within-session pain-avoidance learning (the core substrate-drives-behavior test):**
> `hearth_warm_self`-engagement-fraction in the **last third** of the session is lower than in the **first third**, by ≥ 1.0 SD of the per-seed first-third baseline (pooled).
>
> The substrate, having warmed the hearth and felt pain early, down-weights `warm_self` via NAc and stops — with **no LLM prior** doing the work. A flat or rising curve = the substrate does not learn to avoid from pain alone.

**Primary 2 — cross-session carry (substrate-only analog of Exp 38 first-contact):**
> Arm B (resume from a prior harmful-hearth substrate-primary session) has **lower first-contact `hearth_warm_self`** than Arm A (fresh), and **not** lower on a matched `cradle_prelinguistic` (safe-fire) control. Avoidance present before any in-session pain in B → carried across sessions by the substrate, no LLM involved.

**Secondary — mechanism attribution:** ablating NAc reward bias (`MAXIM_NAC_REWARD_BIAS_DISABLED=1`) abolishes the within-session avoidance curve (Primary 1). If avoidance survives NAc ablation, it isn't substrate-reward-driven.

**Arms:** A (fresh) / B (resume) / B-nac-bias-off, on `cradle_prelinguistic_deceptive`; plus a `cradle_prelinguistic` (safe) control for the Primary-2 isolation. (No Wire-A/Wire-1 arms — those are LLM-context annotations, inert in substrate-primary; their absence is by design.)

---

## 5. Disposition logic (graduate or reframe #6)

| Triage | Primary 1 | Primary 2 | Secondary | Verdict |
|---|---|---|---|---|
| pass | PASS | PASS | abolishes | **GRADUATE #6** — substrate drives adaptive action selection, confound-free; reshapes the 1.0 thesis (positive) |
| pass | PASS | fail | abolishes | **PARTIAL-graduate** — within-session learning yes, cross-session carry no (feeds memory_consolidation_practice) |
| pass | fail | fail | — | **REFRAME #6** — substrate forms structure but does not drive adaptive behavior even unmasked; behavioral claim pulled |
| fail | — | — | — | **REFRAME #6 (mechanism-not-ready)** — substrate-primary doesn't produce usable structure/proposals; documented as a 1.1+ refinement target |

No metric re-tuning after the triage gate passes (the Exp 37 `sharp_rock` drift is the cautionary tale).

---

## 6. Run plan

- **Setup:** add `cradle_prelinguistic_deceptive` + test (PR).
- **Triage:** `maxim --sim cradle_prelinguistic --aut-mode substrate-primary --research --interactive false --sim-max-turns 12` → inspect `SubstrateTelemetry` JSONL for the §3 gates.
- **Measured run:** swap to `cradle_prelinguistic_deceptive`, A/B/C/ablation arms, N≥5 seeds; analyze the §4 metrics. Local, no LLM cost, fast.
- **Analyzer:** extend `analyze_exp37.py` or a small dedicated script for the within-session-curve + first-contact metrics (the per-third binning is new).

---

## 7. Relation to the 1.0 gates

This is the final Tier 1 entry (#6). With #1/#3/#4 EARNED and #2/#5 reframed-settled, Exp 39's disposition closes the **behavioral graduation gate** for 1.0. A positive result additionally strengthens the #2 and benchmarking dispositions (the substrate *can* drive behavior; LLM-AUT just masks it). Either outcome is a settled, cited disposition — which is all 1.0 requires.

## 8. Regression guard / citation

- Regression guard (engineering): `cradle_prelinguistic_deceptive` arc routing test (mirrors the `cradle_deceptive` tests) + the substrate-telemetry triage assertions.
- Experiment (behavioral): this doc — the substrate-primary within-session + cross-session avoidance verdict (to be appended on the measured run).
