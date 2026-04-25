# SCN Oscillator Feedback — anticipatory temporal credit

**Status:** Shell plan (2026-04-24), **deferred until temporal credit Phases 1-7 are stable in production**
**Scope:** ~100-150 LOC across SCN, TemporalCreditDistributor
**Depends on:** [temporal_credit_integration.md](temporal_credit_integration.md) (Phases 1-7 shipped), [tool_pain_bridge_temporal_migration.md](tool_pain_bridge_temporal_migration.md)
**Branch:** TBD

---

## Problem

The temporal credit system (shipped) distributes reward based on **historical** temporal phase similarity — "this event type happened at this time of day before."  The SCN oscillator already has coupling weights that learn co-occurrence patterns (Hebbian on Kuramoto phases), but these weights are never fed back into the credit system.

The gap: the system can credit events that already happened but cannot **anticipate** events that are likely to happen based on learned temporal patterns.

## Design

1. **SCN oscillator `observe(signature)` called on each `TemporalEvent`.**  The oscillator learns which event types co-occur in temporal phase space.

2. **Coupling weights learn co-occurrence patterns** via Hebbian rule on Kuramoto phases.  When two event types consistently fire at the same circadian phase, their coupling strengthens.

3. **`predict_next_occurrence(event_signature)` becomes actionable.**  Given an event type, the oscillator predicts when it's likely to fire next based on learned phase patterns.

4. **Anticipatory credit: pre-activate eligibility traces** for events predicted by the oscillator.  When the oscillator predicts "tool X tends to fire at 2pm" and it's approaching 2pm, the distributor pre-activates eligibility for tool X's substrate nodes.

## Key constraint

This is a **future extension**, not a correction.  The shipped temporal credit system works correctly without oscillator feedback — it just can't anticipate.  Only implement after Phases 1-7 have been validated in production sims.

## Trigger condition

Implement when:
- Temporal credit integration has been stable for 2+ weeks in production sims
- A sim scenario demonstrates the value of anticipatory credit (e.g., recurring daily encounters where the agent should prepare)
- The ToolPainBridge temporal migration (Phase 6 remainder) is complete, providing enough diverse TemporalEvents for the oscillator to learn patterns from
