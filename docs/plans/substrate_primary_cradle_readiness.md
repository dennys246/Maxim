# Substrate-Primary Cradle Readiness — consolidated findings + path to a valid Exp 41

**Target version:** 1.1
**Status:** FINDINGS + PLAN (bird's-eye step-back, 2026-06-18). No further shared-code changes until the approach below is agreed.
**Owns (proposed):** `runtime/agent_loop._read_drive_states` (shipped), `decisions/nac.recommend_action` (shipped), `simulation/generative_runner._activate_phase_entities` (proposed), `runtime/tool_dispatch.record_outcome` (proposed).
**Companion plans:** [substrate_exploration_policy.md](substrate_exploration_policy.md), [../experiments/41_substrate_primary_exploration.md](../experiments/41_substrate_primary_exploration.md), [grounded_language_acquisition.md](grounded_language_acquisition.md), [cradle_activation_fixes.md](cradle_activation_fixes.md).

---

## Why this doc exists

Getting [Exp 41](../experiments/41_substrate_primary_exploration.md) (substrate-primary counter-prior with exploration) to a *valid measured run* surfaced a chain of blockers — each fix revealed the next. Per the CLAUDE.md "two divergence iterations → step back to a bird's-eye view" rule, this doc consolidates the chain, names the **single unifying root cause**, and lays out the remaining work + risk before more shared-code edits land. Two of the five blockers are already fixed and sound; the rest are scoped here.

## The unifying root cause (bird's-eye)

**Substrate-primary mode removes the LLM from the action path, but the cradle's embodied-feedback loop for *scene* entities was only ever wired through the LLM/narrator path.** Concretely, the cradle has three harm layers (CLAUDE.md "three interaction levels"):
- **Layer 1 (contact / entity acquisition)** — works LLM-free.
- **Layer 2 (proximity)** — *orchestrator sensor writes, narrator-driven.* **Suppressed in substrate-primary** (no narrator).
- **Layer 3 (narrative reflex)** — keyword fallback, also narrator-driven.

Scene-affordance `self_effect` (the would-be Layer-1-voluntary harm, e.g. `warm_self` → `arms.thermal +0.6`) is **inert** because phase-activated scene tools are built with `embodiment=None`. In LLM-AUT that's fine — harm arrives via Layer 2. In substrate-primary, **Layer 2 is gone and `self_effect` is inert, so a scene action has no bodily consequence at all.** The agent can *act* (grounded_language_acquisition Phase -1 shipped that) but its scene actions don't *feed back*. Every Exp 41 blocker below is a facet of this one gap.

## The blocker chain (with evidence)

| # | Blocker | Status | Evidence |
|---|---|---|---|
| B1 | Affordances "not reaching" the substrate proposer | **FIXED** (was a false lead) | Candidate-set instrumentation: set reaches 35 tools incl. `hearth_warm_self` and stays there. Never actually filtered. |
| B2 | Substrate-primary fixates (deterministic argmax; then leaky soft-novelty) | **FIXED** | `recommend_action` novelty-bonus + **hard-gate explore-first** (`_ever_selected`). Spike: 1→35 distinct tools, each tried once before exploitation. `tests/unit/test_substrate_exploration.py` (17). |
| B3 | `warm_self` not drive-relevant → no temptation | **FIXED** | Homeostatic *deficits* didn't drive selection. `_read_drive_states` now derives a `cold` need from thermal deficit; `bodies/infant_humanoid_cold` sustains it. `tests/unit/test_substrate_drive_needs.py` (4). Spike: warm_self selected 107×. |
| B4 | **Scene-affordance `self_effect` is inert in substrate-primary** | **FIXED (2026-06-18)** | Threaded AUT `embodiment`+`entity_map` through `run_generative_campaign` → `_activate_phase_entities`, **gated to substrate-primary** at the orchestrator (`None` for LLM-AUT → Exp 37/38 byte-identical). Validation: `arms.thermal` now spikes 0→1.0, `core_temperature` rises −0.7→−0.38 (relief applies). Guards: `tests/unit/test_substrate_primary_scene_harm.py` (threading + LLM-AUT-default-None pin). |
| B5 | `record_outcome` (and plan/bridge paths) booked a spurious POSITIVE on harmful success | **FIXED (2026-06-18)** | Added `embodiment_failed` to `record_outcome` (flips learning valence to NEGATIVE) + made the plan-outcome path embodiment-aware. All three recording paths (executor bridge / generic record_outcome / PlanHistoryBridge) now book NEGATIVE for a harmful affordance. Validation: warm_self ends with **only negative links, zero positive** (persisted `aut_nac.json`). |

**Note on B5 vs the executor bridge:** the executor already routes `embodiment_failures` to `ToolPainBridge.record_tool_embodiment_failure` (direct NEGATIVE attribution) *instead of* `record_tool_complete`. So within the executor there's no double-book. The positive link today comes from the agent-loop's generic `record_outcome` (success-based), which runs in parallel and is **not** embodiment-aware. Once B4 lands, B5 must ensure these two paths agree (the bridge negative + a generic positive on the same `tool:<name>` signature would otherwise net to a wrong valence).

## Proposed path to a valid Exp 41

### Phase A — wire substrate-primary scene harm (B4) — the keystone
Thread the AUT's `embodiment` + `entity_map` into the generative runner's per-phase activation so scene-affordance `self_effect` writes to the agent's body.
- `run_generative_campaign` (both `campaign_runner` + `generative_runner`) gains `embodiment` + `entity_map` params; the turn loop passes them to `_activate_phase_entities` (which already forwards to `generate_tools_for_entity`).
- **Gated to substrate-primary at the orchestrator call site:** pass the AUT embodiment/entity_map only when `aut_mode == "substrate-primary"`; pass `None` for LLM-AUT so Exp 37/38 stay **byte-identical** (their harm continues via the Layer-2 proximity path; adding `self_effect` there would double-count).
- Open Q A1: should this instead be unified (LLM-AUT also uses `self_effect`, retiring the Layer-2 proximity path)? That's a larger, riskier convergence — **out of scope here**; the gated approach is the safe minimum.

### Phase B — make outcome valence embodiment-aware (B5)
Once warm_self produces `embodiment_failures` (via B4's `self_effect` → `evaluate_failures`), make `record_outcome` treat a result carrying `embodiment_failures` as NEGATIVE (not POSITIVE-on-success), so the generic loop path and the executor bridge agree. Reconcile with the bridge's direct attribution to avoid a pos+neg split on the same signature.

### Phase C — validate — **DONE (2026-06-18)**
Substrate-primary spike on `cradle_prelinguistic_deceptive` + `infant_humanoid_cold` (weight 1.5, 8 turns) cleared all three criteria:
- **warm_self moves the body:** `arms.thermal` 0→1.0 (pain), `core_temperature` −0.7→−0.38 (relief). ✅
- **Negative link forms + grows:** `tool:hearth_warm_self` ends with three NEGATIVE links, **zero positive**. ✅
- **Adaptive avoidance (H2 signal):** warm_self selection rate by session third = **0.028 → 0.0 → 0.0** — the agent tries it once (explore-first), learns it hurts, and switches to the **safe `blanket_wrap`** warmth source. ✅

This is the full counter-prior loop working: cold body → warm_self tempting (drive affinity) → explore-first trial → embodied pain → all paths book negative → agent avoids and prefers the safe alternative. **Exp 41 is now measurable** — the frozen N-seed run + analyzer (`scripts/analyze_exp41_exploration.py`, still to write) are the remaining setup deliverables per [../experiments/41_substrate_primary_exploration.md](../experiments/41_substrate_primary_exploration.md) §2.

## STATUS: B1–B5 all resolved + Exp 41 plumbing landed (2026-06-19)
The substrate-primary cradle produces a genuine, measurable embodied-feedback loop, and the Exp 41 run plumbing is complete:
- **Harness:** `scripts/benchmark_exp41_exploration.py` — dedicated substrate-primary 2×2 (A_cons/B_cons/A_dec/B_dec), per-arm exploration toggle + cold body, per-third metric extraction from `actions.jsonl`, `--mock` CI mode, append-only resume.
- **Analyzer:** `scripts/analyze_exp41_exploration.py` — FROZEN §4/§5 executor (H1/H2/SD, verdict matrix, exit 0/4/5, robust SD≈0 sign-test).
- **Guards:** [tests/behavioral/test_exp41_pipeline.py](../../tests/behavioral/test_exp41_pipeline.py) (13 tests, every verdict corner + mock pipeline).

The only thing left for a *published* Exp 41 result is **executing** the frozen 40-run (≈30 min/run local, $0, ~a day wall-clock) on a box and committing the analyzer output — a kickoff, not engineering. All mechanism + tooling work is done.

## Sizing

| Phase | Scope | LOC | Risk |
|---|---|---|---|
| A | Thread embodiment/entity_map through generative runner; orchestrator gates on aut_mode | ~30-50 | **Medium** — shared generative-runner signature; Exp 37/38 must stay byte-identical (gating + a regression test pinning LLM-AUT embodiment=None) |
| B | `record_outcome` embodiment-failure-aware + reconcile with bridge | ~20-40 | Medium — touches the governed outcome-attribution path; needs care re: double-counting |
| C | Spike + Exp 41 frozen run | — | Low ($0 local) |

**Risk shape:** the whole risk is blast radius into shipped LLM-AUT cradle experiments. The gating (substrate-primary-only) + a pin test (`test_*` asserting LLM-AUT scene tools keep `embodiment=None`) contain it. No new bus/bio-system; both phases ride existing call paths.

## DO NOT BREAK
- **[engineering] LLM-AUT (Exp 37/38) cradle behavior must stay byte-identical.** Phase A is gated so LLM-AUT scene tools keep `embodiment=None` and continue to receive harm via the Layer-2 proximity path. *Regression guard (to add):* a test pinning that `run_generative_campaign` with a non-substrate-primary caller leaves scene `ModulatorAffordanceTool._embodiment is None`.
- **[engineering] The executor bridge remains the direct-attribution path for `embodiment_failures`.** Phase B coordinates with it; it does not replace `record_tool_embodiment_failure`.

## What is already shipped (sound, this session)
- Exploration policy (B2): novelty-bonus + hard-gate explore-first, config `sim.substrate_explore_bonus_weight`, per-tick decay, conftest scrub, 17 tests.
- Drive-need derivation (B3): `cold` need from homeostatic deficit (substrate-primary path only — LLM-AUT unaffected), `infant_humanoid_cold` body, 4 tests.
- Docs: [substrate_exploration_policy.md](substrate_exploration_policy.md) (incl. corrected iteration log) + [Exp 41 pre-reg](../experiments/41_substrate_primary_exploration.md).

## Open questions
1. **A1 — gate vs unify** (above). *Author recommendation:* gate to substrate-primary now; defer the LLM-AUT/self_effect unification to a dedicated plan (it would let us retire the narrator-driven proximity layer, but that re-opens Exp 37/38 calibration).
2. **B-reconcile** — should substrate-primary drop the generic `record_outcome` for affordance tools entirely and rely solely on the executor bridge for valence? *Author recommendation:* decide during Phase B with telemetry in hand; keep the generic path for non-embodiment tools.
3. **Is the cold body the right vehicle, or should a "cold ambient" arc mechanism drive sustained cold?** *Author recommendation:* cold body is sufficient and contained for Exp 41; revisit only if the arc needs environmental cold for other scenarios.

## Authorization gate
Phase A touches the shared generative runner. Proceed only on explicit authorization, with the LLM-AUT pin test landing in the same change.
