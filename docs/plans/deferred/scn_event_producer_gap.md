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

## Front-gate: should this be an "SCN bus"? — **No.**

The natural reaction to "five producers are unwired" is to centralize: an SCN/temporal
event bus everything publishes to. Running the mandated front-gate question — *does this
need to be its own mechanism, or can it ride on existing infrastructure?* — the answer is
ride on existing, for three reasons:

1. **The centralized intake already exists.**
   `TemporalCreditDistributor.record_event(TemporalEvent)` is already the single funnel:
   one call fans out to `nac.update_eligibility()` (fast-decay trace), `scn.register()`
   (bin indexing), and `scn.observe_event()` (oscillator phase). A bus would sit in front
   of a funnel that already works. A fourth consumer belongs *inside* `record_event`, not
   behind a new pub/sub layer.
2. **It would not fix the actual failure.** Every gap here is a producer that never called
   the intake, or called it malformed. A producer that forgets `record_event` would forget
   `bus.publish` identically. Swapping the callee changes nothing about the forgetting.
3. **Bus proliferation is already a known cost.** `PainBus`, `ReactionBus`, `AgentBus`,
   `LocalMessageBus` exist, and a *Unified* Event Bus is already deferred (CLAUDE.md
   "Active initiatives"). Adding a fifth, single-purpose bus moves in the opposite
   direction. If a bus-shaped answer is ever right, temporal events should ride the
   unified bus — not get a private one first.

**What the failure actually is: a silent no-op with multiple call sites, missed more than
once.** That is precisely the shape CLAUDE.md already prescribes a cure for — *push
silent-no-op invariants into types, not helpers*. The canonical precedents are
`build_executor(pain_bus=...)` and `build_pain_bus(hippocampus=..., nac=...)`: a required
keyword-only parameter, so forgetting is a `TypeError` at construction rather than silence
at runtime. Applied here, that means **less** machinery than a bus, not more:

- **(a) Make the emitter's dependency structural.** Any object that emits temporal events
  takes `distributor` as required keyword-only, with `None` as the *explicit* opt-out
  (sandboxes, foundry, tests). Forgetting it at a new call site becomes a `TypeError`;
  today it is an early `return`. This alone would have caught the `Embodiment` gap the day
  `build_executor` was written.
- **(b) Make the event impossible to build wrong.** Add a
  `TemporalCreditDistributor.record(*, event_type, event_signature, agent_id, activation=,
  context=)` convenience that constructs the `TemporalEvent` internally. Producers then
  cannot misname `temporal_sig`/`context` or omit `event_id` — the exact four-part
  mistake in `body.py`. Keep the raw `record_event(event)` for callers that already hold
  an event.
- **(c) Stop swallowing malformed events.** The bare `except Exception: log.debug(...)`
  around emission is why a `TypeError` survived indefinitely. Narrow it to the transport
  failures worth tolerating, and let programming errors raise.
- **(d) Make absence visible.** A test asserting every declared `event_type` category
  either has a producer or is explicitly marked reserved — per missing-is-the-signal, the
  gap should fail a test, not wait for someone to grep.

Items (a)-(d) are cheap, additive, and behaviour-preserving *until* a producer is
actually wired — which keeps the "does the oscillator earn its keep?" question properly
downstream of them.

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
## Persistence asymmetry (noted 2026-07-30, nac_cross_session_persistence fold)

`build_bio_stack` now restores hippocampus + NAc + EC from the agent home, but
still constructs `SCN()` with no `persistence_path` — oscillator phases and
event bins restart cold every session (the save side in
`MemoryHub.on_session_end` is guarded on a path that is never set on this
path). Deliberately left unwired: with only one of six event categories
producing (`tool`), persisting the oscillator would preserve state a mostly
dead intake barely feeds. Decide SCN persistence together with the producer
gap above — wiring persistence before producers exist would be backwards.
