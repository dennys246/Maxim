# SCN Oscillator Feedback — anticipatory temporal credit

**Status:** SHIPPED (2026-04-26), Branch: `feat/v1-scn-oscillator`
**Scope:** ~120 LOC across OscillatorNetwork, SCN, TemporalCreditDistributor, build_bio_stack
**Depends on:** [temporal_credit_integration.md](archive/temporal_credit_integration.md) (Phases 1-7 shipped), [tool_pain_bridge_temporal_migration.md](archive/tool_pain_bridge_temporal_migration.md) (P1 shipped)

---

## Problem (resolved)

The temporal credit system distributed reward based on **historical** temporal phase similarity — "this event type happened at this time of day before."  The SCN oscillator had coupling weights that learned co-occurrence patterns (Hebbian on Kuramoto phases), but these weights were never fed back into the credit system.

The gap: the system could credit events that already happened but could not **anticipate** events likely to happen based on learned temporal patterns.

## Implementation

### 1. Per-event-type phase tracking (OscillatorNetwork)

`observe_event(event_signature, signature)` records the circadian phase at which each event type fires.  Ring buffer capped at `max_event_phases` (default 50).

### 2. Imminence prediction (OscillatorNetwork)

`predict_event_imminence(event_signature)` computes how close the current circadian phase is to an event type's learned phase pattern.  Uses circular mean + concentration (R/N) — events with scattered timing produce low imminence scores.  Cold-start guard: < 3 observations returns 0.0.

### 3. Anticipatory pre-activation (TemporalCreditDistributor)

`anticipatory_pre_activate(agent_id)` primes NAc eligibility traces for events the oscillator predicts are imminent.  Called once per tick BEFORE `distribute()`.  When the predicted event actually fires and a reward arrives, the pre-activated trace is credited through the normal fast-decay path in `distribute()`.

This is the biologically correct mechanism: anticipation primes the system for future rewards — it does NOT distribute reward itself.  Events with already-active traces are skipped (no double-priming).

Pre-activation strength = `OscillatorConfig.anticipatory_weight` (default 0.2) × imminence score.

### 4. Production wiring (build_bio_stack)

`build_bio_stack` now calls `scn.enable_oscillator()` after SCN construction.  The oscillator is enabled by default for all production paths.

`record_event()` calls `scn.observe_event(event_signature, temporal_sig)` on every TemporalEvent, feeding the oscillator's per-event-type phase tracker.

### Key files changed

| File | Change |
|------|--------|
| `time/oscillator.py` | `observe_event`, `predict_event_imminence`, `get_anticipatory_signatures`, serialization |
| `time/scn.py` | `observe_event`, `get_anticipatory_signatures` delegation |
| `decisions/temporal_credit.py` | Third credit path in `distribute()`, `observe_event` call in `record_event()` |
| `runtime/bio_stack.py` | `scn.enable_oscillator()` after construction |

### Tests

21 new tests in `tests/unit/test_scn_oscillator_feedback.py`:
- Phase recording + ring buffer
- Cold-start guard (< 3 observations)
- High imminence when phase-aligned, low when scattered/distant
- Anticipatory threshold filtering
- SCN delegation (with + without oscillator)
- Serialization round-trip + backward compat
- Distributor integration (record feeds oscillator, anticipatory credit fires, no double-count)
- build_bio_stack enables oscillator
