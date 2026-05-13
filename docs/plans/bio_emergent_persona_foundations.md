# Bio-system foundations for emergent persona

**Status:** ESCALATED to 0.9.1 — field reservations shipped 2026-04-30 (PR #216) under the V1 Phase A clean-pass branch. Stages 0-3 implementation now lands in 0.9.1 (see [release_0_9_1.md](release_0_9_1.md)). Stages 4-5 (Wires 4 + 5) remain 1.1+.
**Ships in:** ~~1.0 (Stages 0-3)~~ → ~~1.1 (Stages 0-3)~~ → **0.9.1 (Stages 0-3 + new Wire-A cluster-bias annotation)**. Roy-2pc (PR #243) reproduced the structural-vs-behavioral gap on a positive-control fixture — five Roy iterations established that the cluster_reward_bias path is behaviorally inert across both AUT modes regardless of percept overlap. The annotation pattern routes around the block.
**1.0 disposition rationale (now superseded):** [docs/experiments/12_v1_phased_attribution.md](../experiments/12_v1_phased_attribution.md) Phase A reproduced cross-session recall without scaffolds — that result still holds, but the Roy harness produced a *different* falsification target (substrate writes correctly, doesn't translate to action selection) that the annotation wires address.
**Owns:** decision-time wiring across [decisions/nac.py](../../src/maxim/decisions/nac.py), [runtime/agent_loop.py](../../src/maxim/runtime/agent_loop.py), [runtime/gating.py](../../src/maxim/runtime/gating.py), [embodiment/](../../src/maxim/embodiment/), [proprioception/pain_bus.py](../../src/maxim/proprioception/pain_bus.py)
**Companion plans:** [persona_cleanup_and_mode_transition.md](persona_cleanup_and_mode_transition.md) (clears the cognitive dissonance), [persona_convergence_crucible.md](persona_convergence_crucible.md) (uses these foundations for Roy experiments)

## Context

A pre-implementation architectural review identified a pattern across the bio-systems: **state that already persists per-agent isn't being read at decision time**.

- NAc reward biases persist across sessions but only modulate EC recognition thresholds — they don't influence action ranking.
- Episodic memory carries valence per entity, but only shapes behavior when something explicitly queries it.
- Embodiment tracks component integrity, but no decision-time path filters affordances when a body part is damaged.
- Pain → causal-link valence wiring exists, but there's no path for a *percept* (independent of action context) to acquire learned aversion — no Pavlovian fear conditioning.
- The SCN oscillator learns event-time phase patterns; the agent never queries `current_phase()` for decisions.

These are **bio-system gaps**, not persona-emergence enablers. They earn their place in 1.0 because each is an independent architectural omission — the kind a bio-inspired repo shouldn't ship without. That they're also *necessary* for emergent persona to be visible later (per [persona_convergence_crucible.md](persona_convergence_crucible.md)) is a downstream consequence, not the justification.

This plan ships foundation. Whether the foundation is sufficient for persona convergence is a separate, longer-running question handled by the Crucible doc.

## Framing rule

**Each wire is justified by the bio-system gap it closes, not by the persona behavior it might enable.** If a wire ships and emergent persona never materializes, the wire was still right to ship. This framing is load-bearing — it keeps the 1.0 scope grounded and avoids tying releasability to a downstream research question we don't yet know the answer to.

## The five wires

Identified in the decision-boundary review:

| # | Wire | Status | Sizing | Persistence |
|---|---|---|---|---|
| 1 | Risk-sensitive action selection | Ships in 1.0 | M (~200 LOC) | Mutable field on existing dataclass |
| 2 | Stimulus-class aversion (Pavlovian fear) | Ships in 1.0 | M (~250 LOC) | New persisted dict on NAc |
| 3 | Embodiment-state → action filter | Ships in 1.0 | S (~80 LOC) | None |
| 4 | Streak detection / exploration policy | Deferred 1.1+ | M (~200 LOC) | None |
| 5 | Oscillator decision coupling | Deferred 1.1+ | S (~80 LOC) | None |

Wires 4 and 5 deferred not because they're harder but because the Crucible doc's first iterations (`Roy-1`, `Roy-2`) need to teach us which decision-boundary mechanisms are actually load-bearing for visible persona. Shipping all five upfront is over-investing in untested architecture.

## Stage 0 — Telemetry instrumentation prerequisite (~3hr, blocks Stage 4 of Crucible)

The single biggest measurement gap: [actions.jsonl](../../src/maxim/simulation/) writers don't emit `agent_id` per record. Without it, no divergence analysis is possible across two agents. This is also useful for general bio-system debugging beyond persona work.

- Thread `agent_id` + `session_id` into every action record. The `RequestContext` ContextVar at [utils/http.py](../../src/maxim/utils/http.py) already exists; extend the writer to read it.
- Add `entity_class` field to MOTOR/PERCEPT events (sim_log subsystems). Without normalized exposure counts, pain-aversion divergence is unnormalizable.
- Save NAc snapshots at session boundary (not just final) so reward_bias evolution is plottable.
- Add `_format_version` bump to schema check tests per [CLAUDE.md](../../CLAUDE.md) CC1 contract.

Lands first; independent of all wires.

## Wire 3 — Embodiment-state → action filter (Stage 1, ships in 1.0)

**Why first:** smallest LOC, highest behavioral signal per unit work, no new persistence, no frozen-contract risk.

### What's there today
Component integrity already tracked via [embodiment/reflex.py:40](../../src/maxim/embodiment/reflex.py#L40) `_get_integrity(component_name)`. [simulation/tools.py](../../src/maxim/simulation/tools.py) writes `context["component_integrity"]` on tool outcomes. **Nothing reads integrity at decision time.**

### Implementation
- New method on `Embodiment`: `get_disabled_affordances() -> set[str]` returning affordances routed through components below integrity threshold.
- New method: `get_degraded_affordances() -> dict[str, float]` returning `affordance_name → integrity` for partially-damaged paths.
- Hook in [agent_loop.py](../../src/maxim/runtime/agent_loop.py) before tool description assembly: filter disabled tools from the tool list, append `[DAMAGED: integrity 0.X]` to degraded ones.
- Default thresholds: `integrity < 0.3` disables; `integrity < 0.6` annotates.

### Frozen contract impact
None. No new persisted state, no dataclass changes. Pure read-side wiring.

### Behavioral signal
An agent with a damaged arm visibly stops calling arm-routed affordances. Cleanest emergent "trait" — physical history shapes behavior without prompt injection.

### Test surface
- Unit: `Embodiment.get_disabled_affordances()` returns expected set when components below threshold.
- Integration: in a sim, damage a component, verify the affordance disappears from the next prompt's tool list.

## Wire 2 — Stimulus-class aversion (Stage 2, ships in 1.0)

**The only wire with new persistence. Treat carefully.**

### What's there today
Pain → causal-link wiring via `record_outcome` (action context). No path where a *percept* (entity_class, failure_mode) acquires learned valence independent of action attribution. The PainSignal context dict at [proprioception/pain_bus.py](../../src/maxim/proprioception/pain_bus.py) already carries `entity` and `failure_mode` keys — the data is there, the consumer isn't.

### Implementation
- Add `NAc._percept_valences: dict[tuple[str, str], float]` keyed by `(entity_class, failure_mode)` with values in `[-1.0, +1.0]`.
- Persist via `dump()` / `load_state()` with `_format_version` bump per CC1 contract. Backward-compatible reader: missing field → empty dict.
- New method `NAc.record_percept_valence(entity_class, failure_mode, valence, *, agent_id)` with explicit `agent_id` keyword-only per the multi-agent attribution rules in [CLAUDE.md](../../CLAUDE.md) (`Per-agent stash dicts` rule).
- New method `NAc.get_percept_valence(entity_class, failure_mode, *, agent_id) -> float`.
- Decay: same per-tick decay shape as `_reward_bias` (the existing `decay_reward_biases` per-tick call site at [agent_loop.py](../../src/maxim/runtime/agent_loop.py) section 8.5 extends to call `decay_percept_valences`).
- New PainBus subscriber `create_percept_valence_subscriber(nac)` registered via [proprioception/pain_bus.py](../../src/maxim/proprioception/pain_bus.py) `build_pain_bus()` — auto-wired in `build_bio_stack`. Per the build_pain_bus invariant, forgetting it is a `TypeError`, not a silent no-op.
- Read site: extend `GatingContext` at [runtime/gating.py:67](../../src/maxim/runtime/gating.py#L67) with `learned_aversions: dict | None = None` (frozen-safe additive field with default per CC3 contract). `TextSalienceScorer._compute_salience` queries it on percept arrival.

### Frozen contract impact
- `GatingContext` is shape-frozen. Adding `learned_aversions: dict | None = None` at the end of the field list is non-breaking per the CC3 audit rules. Audit gate: docstring update declaring the addition.
- New persistence in NAc: `_format_version` bump on the NAc dump. Backward-compat reader required.

### Behavioral signal
An agent burned by `dragon_fire` once carries elevated salience for subsequent dragon percepts even before any action — bio-grounded fear, not prompt-driven caution. Substrate-attributable: divergence shows up in the persisted `_percept_valences` dict.

### Test surface
- Unit: `record_percept_valence` writes correct keys; `get_percept_valence` round-trips.
- Unit: persistence round-trip preserves dict; old `nac.json` without the field loads cleanly.
- Integration: PainBus subscriber wires automatically through `build_pain_bus`; verify with a regression test similar to the latent bridge×subscriber trap test referenced in CLAUDE.md.
- Multi-agent: two agents sharing one NAc instance attribute valence to distinct `agent_id` keys per the CC4 rules.

## Wire 1 — Risk-sensitive action selection (Stage 3, ships in 1.0)

**Depends on Wires 2+3 having generated data.** Without persistent percept valences and embodiment-state read paths, this wire has thin signal to weigh.

### What's there today
Action ranking is implicit — the LLM picks tools given the assembled tool list. NAc tracks `confidence` per `CausalLink` but no variance estimator. `OutcomePrediction.confidence` is a scalar; no uncertainty interval.

### Implementation
- Add `CausalLink.variance_estimate: float = 0.0` (mutable dataclass, no frozen impact). Update via Welford's online variance in `record_outcome`.
- Extend `OutcomePrediction` with `uncertainty_interval: tuple[float, float] = (0.0, 0.0)` — additive default-having field is frozen-safe per CC3.
- New method `NAc.get_action_risk_profile(event_sig, *, agent_id) -> dict[str, float]` returning `{action_signature → risk_score}` from variance + observation count.
- Hook in [agent_loop.py](../../src/maxim/runtime/agent_loop.py) tool description assembly: append `[high variance]` / `[reliable]` annotations based on risk profile.

### Honest scope caveat
This wire's behavioral effect goes through the LLM (it reads the annotations and adjusts). It is **hybrid bio-system + LLM**, not pure substrate-driven. A cleaner post-1.0 design would add a real risk-weighted action ranker that pre-filters tools before the LLM sees them. The hybrid version ships in 1.0 to keep scope tight and avoid building a new ranker subsystem prematurely.

The Crucible doc's three-arm comparison (substrate-primed vs prompt-injected vs blank) will reveal whether this hybrid wiring carries enough substrate signal or whether the post-1.0 pre-filter ranker is needed.

### Frozen contract impact
- `OutcomePrediction` adds optional default-having field — frozen-safe.
- `CausalLink` is mutable; new field is invisible to frozen-contract surface.

### Test surface
- Unit: `record_outcome` updates Welford variance correctly across N observations.
- Unit: `get_action_risk_profile` returns expected ordering for synthetic link distributions.
- Integration: tool descriptions in agent_loop output carry the annotation when variance crosses thresholds.

## Wires 4 + 5 — Deferred to 1.1+

### Wire 4: Streak detection / exploration meta-policy
Per-action success/failure streaks influencing exploration vs exploitation. Session-scoped (not persisted) at first; could be lifted to NAc-persisted later. Deferred because the Crucible doc's first iterations need to teach us whether streak-based exploration is what's actually missing or whether the substrate-grounded wires (1+2+3) plus stable LLM policy is sufficient.

### Wire 5: Oscillator decision coupling
Read-side wiring from `OscillatorNetwork.get_event_phase()` into action-time decisions. Smallest wire by LOC. Deferred because circadian-phase decision-making has zero existing callers and adding the first one without a real motivating use case risks shipping speculative architecture.

Both wires get explicit "deferred to 1.1+" entries in the [docs/plans/README.md](../README.md) roadmap with the trigger: "revive when a Roy iteration's findings indicate this wire is the load-bearing missing piece."

## Cross-cutting: persistence schema

Only Wire 2 adds persisted state. Schema impact:

```json
{
  "_format_version": "1.1",  // bumped from 1.0 for the new field
  "_links": { ... },         // existing
  "_reward_bias": { ... },   // existing
  "_percept_valences": {     // NEW
    "<agent_id>": {
      "<entity_class>:<failure_mode>": 0.X
    }
  }
}
```

Reader policy (mirrors existing patterns):
- Missing `_percept_valences` → empty dict; no warning beyond the standard `_format_version` drift log.
- Missing `_format_version` → "0.x" sentinel per [CLAUDE.md](../../CLAUDE.md) CC1; one warning per file_type per process.

## Cross-cutting: frozen contract impact

Per [CLAUDE.md](../../CLAUDE.md) CC3 audit rules:
- `GatingContext` adds `learned_aversions: dict | None = None` at field-list end. Docstring update declares the addition. Frozen-safe.
- `OutcomePrediction` adds `uncertainty_interval: tuple[float, float] = (0.0, 0.0)` at field-list end. Docstring update. Frozen-safe.
- No new frozen dataclasses introduced.
- No existing frozen dataclasses modified beyond optional-field-with-default appends.

This is the audit gate the docstrings must declare per CC3 before merge.

## Sizing summary

| Stage | Wire | LOC | Files | Persistence | Frozen impact |
|---|---|---|---|---|---|
| 0 | Telemetry | ~150 | sim_logger, action writer, http context | _format_version bump on action JSONL | none |
| 1 | Wire 3 | ~80 | embodiment/, agent_loop.py | none | none |
| 2 | Wire 2 | ~250 | nac.py, pain_bus.py, gating.py | new dict on NAc | GatingContext field add |
| 3 | Wire 1 | ~200 | nac.py, causal_link.py, agent_loop.py | none | OutcomePrediction field add |
| **Total 1.0** | | **~680** | | | |

**Estimated calendar:** 3-5 days for Stages 0-3, including review rounds.

## Order of implementation

1. **Stage 0 first** (telemetry) — independent; no dependencies; useful immediately.
2. **Wire 3 second** (embodiment filter) — smallest, no persistence, demonstrates the framing without risk.
3. **Wire 2 third** (percept valences) — only persistence change; lands with full schema discipline.
4. **Wire 1 last** (risk-sensitive ranker) — depends on Wire 2 having generated data to weigh.

Each stage gets its own pre-merge two-lens review (Executor + Architecture lenses, per [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md)). The latent-bridge×subscriber trap referenced in [CLAUDE.md](../../CLAUDE.md) for Wave 1 of biosystem_unification is a known shape — when adding the percept-valence subscriber in Wire 2, pre-merge review must specifically check that subscriber and bridge don't double-count.

## Definition of done

- All four stages shipped to main, each behind a pre-merge two-lens review.
- `_percept_valences` round-trips through dump/load.
- Sim with `MAXIM_LOG_FILE` produces JSONL records with `agent_id` on every action.
- Damaged-component test: an agent's tool list visibly drops affordances on integrity drop.
- Pavlovian test: an agent's percept salience score on `(entity_class, failure_mode)` shifts measurably after a single pain event with that signature, persists across session restart.
- Risk annotation test: high-variance action gets `[high variance]` annotation in tool list after >5 observations with high RPE std.
- No regressions on the [tests/integration/test_persistence_compat.py](../../tests/integration/test_persistence_compat.py) baseline.

## What this plan deliberately does NOT do

- Does not promise visible persona divergence. That's the Crucible doc's question.
- Does not ship Wires 4 or 5. They wait for Crucible findings to motivate them.
- Does not redesign action ranking as a substrate-driven pre-filter. That's a post-1.0 architectural decision the Crucible's three-arm results will inform.
- Does not touch the prompt-injection persona surface. That's [persona_cleanup_and_mode_transition.md](persona_cleanup_and_mode_transition.md).
