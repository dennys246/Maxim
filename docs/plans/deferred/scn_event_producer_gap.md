# SCN temporal-event producer gap — five of six declared categories are unwired

**Status:** Deferred, drafted 2026-07-29. Discovered while measuring an unrelated
oscillator-density question during the transition-based drive-pain fold
([transition_based_drive_pain.md](transition_based_drive_pain.md)); the concern that
prompted it turned out void, and this is what was underneath.

**Not caused by that fold.** Every gap below predates it. Nothing here blocks
PR #435.

## The finding

`TemporalEvent.event_type` declares six signal-source categories
([time/temporal_event.py:30-32](../../../src/maxim/time/temporal_event.py)):
`"tool"`, `"pain"`, `"reaction"`, `"percept"`, `"affordance"`, `"deliberation"`.

Live producers in `src/maxim/`, audited 2026-07-29:

| event_type | producers | status |
|---|---|---|
| `tool` | 12, all in `bridges/tool_pain_bridge.py` | **LIVE** |
| `pain` | 0 | unwired |
| `reaction` | 0 | unwired |
| `percept` | 0 | unwired |
| `affordance` | 0 | unwired |
| `deliberation` | 0 | unwired — **and `record_event` special-cases it** |
| `drive:<sensor>:*` (dynamic) | 1, `embodiment/body.py` | **DEAD — two independent bugs** |

So `OscillatorNetwork`'s circadian phase learning — and therefore
`TemporalCreditDistributor.anticipatory_pre_activate`, the B2 oscillator feedback into
NAc eligibility traces — runs on **one input channel**. Empirically confirmed: a real
substrate-primary cradle run persists `aut_scn.json` with 10 event signatures, all
`tool`, zero `drive:*`.

## The drive path is broken twice — a one-line wire will NOT fix it

This matters because the obvious fix is wrong, and it fails *silently* in both stages.

1. **Unwired distributor.** `Body._emit_drive_temporal_event`
   ([embodiment/body.py](../../../src/maxim/embodiment/body.py)) early-returns on
   `self._distributor is None`. `runtime/bootstrap.py::build_executor` threads its
   `distributor=` into `ToolPainBridge` (`:439`) but constructs
   `Embodiment(entity, pain_bus=..., agent_id=...)` **without** it (`:456`) — in the
   same function, ~17 lines apart.
2. **Malformed constructor.** Even wired, the call raises. It passes
   `temporal_signature=` and `metadata=`, neither of which exists (`temporal_sig` and
   `context` are the fields), and omits the required `event_id` and `event_signature`.
   Verified:
   ```
   TypeError: TemporalEvent.__init__() got an unexpected keyword argument 'temporal_signature'
   ```
   The call is wrapped in `except Exception as exc: log.debug(...)`, so wiring the
   distributor alone converts a silent early-return into a silent `TypeError` — the
   path would look connected and still emit nothing.

`deliberation` has a matching asymmetry: `TemporalCreditDistributor.record_event`
assigns it SCN significance `0.1` "so thoughts lose eviction battles to real memories"
([decisions/temporal_credit.py](../../../src/maxim/decisions/temporal_credit.py)) — a
branch no producer can reach.

## Why this is deferred, not fixed

Fixing any of these **activates a learning path that has never run**, changing what the
oscillator learns and what `anticipatory_pre_activate` returns. That is a behavioural
change requiring its own validation — not a drive-by edit inside an attribution PR
(and per dormancy-over-deletion, unearned mechanisms don't accrete features).

There is also a prior question worth answering *before* wiring anything: **does
per-event-type circadian phase learning earn its keep at all?** The B2 oscillator is
`[engineering]`-tier, never behaviourally graduated. Wiring five more producers into a
mechanism that hasn't demonstrated behavioural weight would be building on sand. The
honest sequence is: pick ONE additional category, wire it correctly, and measure
whether anticipatory pre-activation changes behaviour — before wiring the rest.

## If revived

1. Fix `body.py`'s `TemporalEvent(...)` call (correct field names + supply
   `event_id`/`event_signature`) **and** pass `distributor=distributor` to the
   `Embodiment` construction in `build_executor`. Both, or neither — either alone is a
   silent no-op.
2. Narrow the bare `except Exception` around the emit so a malformed event is loud.
   This class of bug survived precisely because it was swallowed at DEBUG.
3. Add a test asserting each *declared* `event_type` category either has a producer or
   is explicitly marked reserved — the missing-producer condition should be visible,
   per the missing-is-the-signal principle.
4. Re-run `scripts/check_oscillator_coldstart.py --session <id>` and confirm the
   category appears with a real observation count.

**Revive when:** the oscillator earns behavioural weight and a second category is
genuinely needed; or a drive-conditioned anticipatory behaviour is on the critical path
(the cradle / orient work is the likely trigger).

## Regression guard

None yet — this documents an absence. `scripts/check_oscillator_coldstart.py` reports
`drive=0` with an explanation pointing here, which is the closest thing to a tripwire
until item 3 above ships.