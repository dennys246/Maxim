# SCN oscillator is enabled by default in `build_bio_stack`

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

- **[engineering] SCN oscillator is enabled by default in `build_bio_stack`** (B2, 2026-04-26). `scn.enable_oscillator()` runs after SCN construction. The oscillator learns per-event-type circadian phase patterns via `OscillatorNetwork.observe_event()` (called from `TemporalCreditDistributor.record_event()`). Anticipatory pre-activation: `TemporalCreditDistributor.anticipatory_pre_activate(agent_id)` primes NAc eligibility traces for oscillator-predicted imminent events (call once per tick before `distribute()`). `distribute()` has two credit paths: (1) fast-decay eligibility (includes pre-activated traces), (2) phase-similarity fallback. Anticipatory strength = `OscillatorConfig.anticipatory_weight` (default 0.2x) × imminence. Cold-start guard: < 3 observations per event type returns 0.0 imminence. `_event_phases` on OscillatorNetwork is the ring-buffer store (capped at `max_event_phases=50`); only written under the distributor's RLock. Persisted via `oscillator.to_dict()` → `scn.dump()` → SCN JSON. Regression guard: [src/maxim/runtime/bio_stack.py::build_bio_stack](src/maxim/runtime/bio_stack.py) (oscillator enable at construction) + [src/maxim/decisions/temporal_credit.py](src/maxim/decisions/temporal_credit.py) (TemporalCreditDistributor composition).
