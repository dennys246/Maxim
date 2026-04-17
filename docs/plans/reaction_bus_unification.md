# ReactionBus Unification — `build_reaction_bus` with required producers/subscribers

**Status:** SHELL ONLY. Audit + design pending.
**Parent index:** [biosystem_unification.md](biosystem_unification.md)
**Wave:** 1 of 4 (parallel-safe with `pain_bus_unification.md`)
**Depends on:** none
**Parallel-safe with:** [pain_bus_unification.md](pain_bus_unification.md) — different files, no overlap.
**Blocks:** [memory_hub_unification.md](memory_hub_unification.md), [bio_stack_unification.md](bio_stack_unification.md).

## Goal

Same shape as PainBus unification, applied to `ReactionBus`. Push the producer/subscriber wiring invariant DOWN into `build_reaction_bus(...)` so constructing one without explicit decisions is a `TypeError`.

## The repeating shape

`ReactionBus` is the typed isolation surface for evaluative signals (per `reactions/types.py`). It coexists with PainBus by design — PainBus is the rich-context carrier, ReactionBus is the strict typed surface. Both have producers (sensors → PerceptProducer, modulators → ReactionProducer) and subscribers. The current construction pattern is scattered: each entry point that wants reactions builds its own bus and threads producers/subscribers manually.

Forgetting a producer or subscriber is **silent** — the bus accepts publishes, dispatches to whatever subscribers exist (which may be zero), and the missing learning signal corrupts everything downstream. Same shape as PainBus, smaller surface area today but same risk.

## Audit (2026-04-16)

### Key finding: surface is fundamentally different from PainBus

The shell plan expected a PainBus-shaped surface (multiple scattered construction sites, each forgetting different subscribers). **That is not what exists.** ReactionBus is an **internal component of PainBus** — there is exactly ONE production construction site, and every consumer accesses it via `pain_bus.reaction_bus`. The structural enforcement question is therefore different from PainBus: it's not "N call sites forget to wire subscriber X" but rather (a) downstream plans (Wave 3 `bio_stack_unification.md`) require `build_reaction_bus(...)` to exist as an independent construction door, and (b) one factory (CerebellumModulator) silently drops all modulator failure reactions because `reaction_bus=` is never wired.

### Construction sites in `src/maxim/` (production code)

| # | Site | File:line | What happens | Subscribers wired? | Notes |
|---|---|---|---|---|---|
| 1 | PainBus internal | [pain_bus.py:105](src/maxim/proprioception/pain_bus.py#L105) | `self.reaction_bus = ReactionBus(history_size=..., refractory_overrides={"pain": 0.5})` | ✅ `_sim_log_reaction` (subscribe_all) + ✅ `_bridge_reaction_to_pain_subs` (subscribe "pain") | **The ONLY production construction site.** PainBus owns ReactionBus as an internal component. Both internal subscribers auto-wired at construction. |
| 2 | reactions/bus.py docstring | [bus.py:30](src/maxim/reactions/bus.py#L30) | `bus = ReactionBus()` | ➖ | Docstring example only, not real code. |

The shell plan predicted sites in `runtime/agent_pool.py`, `agents/maxim_agent.py`, `simulation/orchestrator.py`. **None of these construct a ReactionBus.** All ReactionBus consumers access it via `pain_bus.reaction_bus`.

### Consumers (access via PainBus, never construct independently)

| Module | How it accesses ReactionBus |
|---|---|
| [simulation/fixture_orchestrator.py:336](src/maxim/simulation/fixture_orchestrator.py#L336) | `pain_bus.reaction_bus.history` for reaction snapshots |
| [simulation/sandbox.py:533](src/maxim/simulation/sandbox.py#L533) | `self._pain_bus.reaction_bus` to publish Reactions directly |
| [runtime/pain_interceptor.py:156](src/maxim/runtime/pain_interceptor.py#L156) | `self._pain_bus.reaction_bus.publish(reaction)` |
| [runtime/sim_adapter.py:61](src/maxim/runtime/sim_adapter.py#L61) | `getattr(_pb, "reaction_bus", _pb)` for pain percept routing |
| [proprioception/perceived_pain.py:357,491](src/maxim/proprioception/perceived_pain.py#L357) | `self._pain_bus.reaction_bus.publish(reaction)` |
| [simulation/conversational_source.py:99](src/maxim/simulation/conversational_source.py#L99) | `getattr(pain_bus, "reaction_bus", pain_bus)` for injection |

All of these access ReactionBus through the PainBus wrapper. No independent construction.

### Gap A (Silent, in-scope) — CerebellumModulator factory never wires `reaction_bus=`

[embodiment/backends/cerebellum_modulator.py:68](src/maxim/embodiment/backends/cerebellum_modulator.py#L68) accepts an optional `reaction_bus: Any = None` constructor parameter. But `cerebellum_modulator_factory` at [line 280](src/maxim/embodiment/backends/cerebellum_modulator.py#L280) constructs `CerebellumModulator(...)` WITHOUT passing `reaction_bus=`. Result: `self._reaction_bus` is ALWAYS `None` in production. Every call to `_emit_failure_reaction` at [line 196](src/maxim/embodiment/backends/cerebellum_modulator.py#L196) silently returns at `if self._reaction_bus is None: return`.

**Blast radius:** every SEM modulator failure signal (Cerebellum-predicted execution failure, affordance violation, sensor-range breach) is silently dropped. No evaluative Reaction reaches the bus when Cerebellum's predicted execution fails. The agent never learns from modulator-level failures that DON'T propagate as body-level pain. This is a real silent-no-op gap, same L1 shape as PainBus Gap A — except it's a factory-wiring bug, not a construction-site-duplication bug.

**In-scope to fix in this PR.** The fix is: (a) `cerebellum_modulator_factory` accepts `reaction_bus=` and passes it through, (b) the call site that invokes `cerebellum_modulator_factory` passes `pain_bus.reaction_bus` (or the standalone ReactionBus once `build_reaction_bus` exists).

### Gap B (Dead protocol, informational) — ReactionProducer has no implementations

[reactions/protocols.py:36](src/maxim/reactions/protocols.py#L36) defines `ReactionProducer` Protocol with `next_reaction() -> Reaction | None`. **No class in `src/maxim/` implements `next_reaction`.** The protocol exists for future SEM work (modulators → ReactionProducer via CerebellumModulator mediation per the protocols.py docstring). Not a bug — it's an unexercised protocol. Not in-scope for this PR; the protocol shape is correct, it just has no implementors yet.

### Why `build_reaction_bus(...)` is still needed despite N=1 today

The justification is NOT the structural-enforcement N-sites-drift pattern (N=1 today). It's **downstream sequencing**:

1. `bio_stack_unification.md` (Wave 3) **explicitly prescribes** `reaction_bus = build_reaction_bus(...)` constructed BEFORE `pain_bus = build_pain_bus(..., reaction_bus=reaction_bus)`. Wave 3 requires the builder to exist.
2. `reaction_bus_unification.md:57` (this plan) says "PainBus depends on ReactionBus existing at construction time. In `bio_stack_unification.md` (Wave 3), ReactionBus must be constructed BEFORE PainBus."
3. `memory_hub_unification.md` (Wave 2) lists both buses as dependencies.
4. Establishing the door NOW when the surface is clean (one caller, simple) avoids a refactor during Wave 3 when the surface is complex (bio_stack composing multiple systems).

The builder has ZERO production callers today — `PainBus.__init__` constructs `ReactionBus()` directly. Wave 3's `build_bio_stack` will be the first production caller. Same principle as writing a Protocol before it has multiple implementors — the interface is the deliverable, not the call count.

### Wave 3 migration notes (surfaced by pre-merge architecture review)

When Wave 3 extracts ReactionBus from PainBus and passes it as a parameter:

1. **`build_pain_bus` needs a `reaction_bus=` parameter.** Today it constructs PainBus which internally builds its own ReactionBus. Wave 3 will change this to: `build_reaction_bus(...)` → `build_pain_bus(..., reaction_bus=rb)`. This requires adding `reaction_bus=` to `build_pain_bus`'s signature and modifying `PainBus.__init__` to accept an external ReactionBus instead of constructing one.

2. **PainBus's auto-registered subscribers need to relocate.** Today `PainBus.__init__` auto-registers `_sim_log_reaction` (subscribe_all) and `_bridge_reaction_to_pain_subs` (subscribe "pain") on its internal ReactionBus. When Wave 3 passes an externally-constructed ReactionBus, PainBus must either (a) register those on the externally-provided bus, or (b) Wave 3's `build_bio_stack` passes them via `build_reaction_bus(per_kind_subscribers={"pain": (bridge_cb,)}, all_subscribers=(sim_log,))` and PainBus stops auto-registering. The `per_kind_subscribers`/`all_subscribers` signature handles both shapes cleanly — this is a migration coordination question, not a design gap.

Neither of these is in-scope for this PR. They're documented here so Wave 3 doesn't discover them mid-implementation.

## Design sketch

```python
def build_reaction_bus(
    *,
    producers: tuple[ReactionProducer, ...] = (),
    subscribers: dict[ReactionKind, tuple[Callable, ...]] = ...,
    isolation_check: bool = True,
) -> ReactionBus:
    """Construct a ReactionBus with explicit producer/subscriber wiring.

    Producers and subscribers are required keyword args (no defaults
    that would silently accept an empty bus). Empty tuples are legal
    explicit opt-outs for tests.
    """
```

**Open design questions:**
- Should producers be a required kwarg or accepted as an empty tuple by default? (My lean: required, tests pass empty explicitly.)
- Should the isolation-rule check (no cross-agent intent, no learned-policy hints) be enforced at construction time on each producer, or stay at publish time?
- Does this plan need to coordinate with PainBus unification on the shared `pain_signal_to_reaction` adapter?

## Coordination with PainBus unification

Per `proprioception/pain_bus.py` rewritten module docstring: PainBus publishes pain signals AND forwards converted Reactions to ReactionBus (lossy by design). This means **PainBus depends on ReactionBus existing** at construction time. Implication: in `bio_stack_unification.md` (Wave 3), ReactionBus must be constructed BEFORE PainBus so the latter can hold a reference. In Wave 1, the two plans don't need to coordinate because each builder is independent — but the integration question lands in Wave 3.

## Pre-merge review round (mandatory)

**Executor lens:**
- Does the new `build_reaction_bus` correctly enforce the per-kind dispatch and the `(kind, source)` refractory gate?
- Are there construction sites the audit missed?
- Does the migration introduce any double-dispatch between PainBus and ReactionBus on the same signal?

**Architecture lens:**
- Should producers be passed at construction or registered after via `.add_producer()`? Trade-off: construction-time is structural; post-construction allows dynamic wiring at runtime.
- Does this plan need to be coordinated with `pain_bus_unification.md`'s author so the `pain_signal_to_reaction` integration stays consistent?
- Should the isolation-rule docstring in `reactions/types.py` be promoted to a runtime check inside the constructor?

## Estimated scope

~120 LOC + ~200 LOC of new tests. Single PR. ~1-2 days.

## Out of scope

- PainBus subscribers — see `pain_bus_unification.md`.
- The `pain_signal_to_reaction` adapter inside `pain_bus.py` — touch only if the audit reveals a drift.
- Bio-stack composition — see `bio_stack_unification.md`.
