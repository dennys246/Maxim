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

## Audit (PENDING — do this first when the plan opens)

Every `ReactionBus` construction site in `src/maxim/`:

| # | Site | producers wired? | subscribers wired? | Migration |
|---|---|---|---|---|
| TBD | runtime/agent_pool.py | ? | ? | TBD |
| TBD | agents/maxim_agent.py | ? | ? | TBD |
| TBD | simulation/orchestrator.py | ? | ? | TBD |
| TBD | tests/* | ? | ? | TBD |

**Notes from prior context:** `reactions/bus.py::ReactionBus` was generalized from PainBus during the reaction abstraction Phase 4 work. Per-kind dispatch + isolation-rule enforcement live in the bus itself. The construction sites are fewer than PainBus today but growing.

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
