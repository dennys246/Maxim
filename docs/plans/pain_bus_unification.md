# PainBus Unification — `build_pain_bus` with required learning subscribers

**Status:** SHELL ONLY. Audit + design pending. Open this when the executor bootstrap PR merges and a session has bandwidth.
**Parent index:** [biosystem_unification.md](biosystem_unification.md)
**Wave:** 1 of 4 (parallel-safe with `reaction_bus_unification.md`)
**Depends on:** none — `executor_bootstrap_unification.md` should land first to keep the migration audit clean, but technically independent.
**Parallel-safe with:** [reaction_bus_unification.md](reaction_bus_unification.md) — different files, no overlap.
**Blocks:** [memory_hub_unification.md](memory_hub_unification.md), [default_network_unification.md](default_network_unification.md), [bio_stack_unification.md](bio_stack_unification.md).

## Goal

Push the PainBus subscriber-wiring invariant DOWN into `build_pain_bus(...)` so that constructing a PainBus without explicit learning-subscriber decisions is a `TypeError`. The current scattered pattern across 5+ entry points is the next instance of the same bug class `executor_bootstrap_unification.md` closed.

## The repeating shape (silent failure)

Every entry point that builds a PainBus does this:

```python
pain_bus = PainBus()
pain_bus.subscribe(create_pain_memory_subscriber(hippocampus))   # easy to forget
pain_bus.subscribe(create_pain_nac_subscriber(nac))               # easy to forget
```

Forgetting either subscriber is a **silent no-op** — the bus accepts publishes, dispatches to zero learners, pain signals fire into the void, NAc never learns from out-of-band pain, hippocampus never captures pain memories. Fits the L1 silent-failure-mode rule perfectly.

## Audit (PENDING — do this first when the plan opens)

Every PainBus construction site in `src/maxim/`:

| # | Site | hippocampus subscribed? | nac subscribed? | other subscribers? | Migration |
|---|---|---|---|---|---|
| TBD | cli.py non-sim | ? | ? | ? | TBD |
| TBD | cli.py --sim agent | ? | ? | ? | TBD |
| TBD | cli.py --sim interactive | ? | ? | ? | TBD |
| TBD | simulation/orchestrator.py (aut_pain_bus) | ? | ? | ? | TBD |
| TBD | embodied_runtime/agentic_runtime.py | ? | ? | ? | TBD |
| TBD | api.py headless | ? | ? | ? | TBD |
| TBD | tests/* (count + classify) | ? | ? | ? | TBD |

**Expected pre-existing bugs to surface** (based on the `executor_bootstrap_unification.md` audit pattern): at least one entry point will be missing one or both subscribers. The api.py headless mode is the prime suspect (already known to be missing the bridge from the executor unification audit).

## Design sketch

```python
def build_pain_bus(
    *,
    hippocampus: Hippocampus | None,
    nac: NAc | None,
    additional_subscribers: tuple[Callable[[PainSignal], None], ...] = (),
) -> PainBus:
    """Construct a PainBus with explicit learning-subscriber decisions.

    `hippocampus` and `nac` are REQUIRED keyword args. Pass real
    instances to auto-subscribe the standard learners
    (create_pain_memory_subscriber, create_pain_nac_subscriber); pass
    None to explicitly opt out (sandbox / test).
    """
```

**Open design questions:**
- Should `additional_subscribers` be a list of callables, or a list of `Subscriber` protocol instances?
- Should the function support per-pain-type filtering at the constructor level, or delegate that to subscribers?
- Should there be a separate `build_test_pain_bus()` factory for test code so test authors don't have to type `hippocampus=None, nac=None` everywhere?

## Migration call sites (rough scope)

5-6 production sites + ~? test sites. Each migrates from `PainBus(); pain_bus.subscribe(...)` to `build_pain_bus(hippocampus=..., nac=...)` with an explicit decision.

## Pre-existing bugs to fix as part of migration

Likely `api.py::maxim.create.agent` headless mode — the executor bootstrap audit already found it constructs no PainBus at all. This plan should construct one (with the user-facing API decision: bio-learning on by default or off by default). This is the **right plan to fix that gap**, and `executor_bootstrap_unification.md` already left a TODO pointing here.

## Doc + memory refinement (load-bearing scope)

- `CLAUDE.md` invariant addition: "`build_pain_bus` is the canonical PainBus construction site. `hippocampus` and `nac` are required keyword args. Forgetting subscribers is a TypeError."
- Update `executor_bootstrap_unification.md`'s "TODO: api.py headless" note to point at this plan as the resolver.
- Update `biosystem_unification.md` status row.

## Pre-merge review round (mandatory)

Two parallel reviewers (Executor + Architecture lenses). Specific questions:

**Executor lens:**
- Does the new `build_pain_bus` correctly subscribe the standard learners? Verify both subscribers fire on a published `PainSignal`.
- Are there any PainBus construction sites I missed in the audit?
- Does the migration introduce any double-subscription (caller subscribes + `build_pain_bus` also subscribes)?
- Does the test surface include a regression guard for the original silent-no-op bug shape?

**Architecture lens:**
- Is "required keyword arg with no default" the right strictness for `hippocampus`/`nac`, or is `None` acceptable as the default?
- Should `additional_subscribers` be a kwarg or a separate `.subscribe()` call after construction?
- Does this plan supersede or complement `bio_stack_unification.md`?
- Is the pre-existing `api.py` PainBus gap correctly in-scope for this plan, or should it stay deferred?

## Estimated scope

~150 LOC + ~250 LOC of new tests. Single PR. ~2-3 days of focused work.

## Out of scope

- ReactionBus subscribers — see `reaction_bus_unification.md`.
- MemoryHub bridge wiring — see `memory_hub_unification.md`.
- The AgentFactory canonicalization that uses this builder — see `agent_factory_canonicalization.md`.
