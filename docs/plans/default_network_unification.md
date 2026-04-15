# DefaultNetwork Unification — required deps at construction

**Status:** SHELL ONLY. Audit + design pending.
**Parent index:** [biosystem_unification.md](biosystem_unification.md)
**Wave:** 2 of 4 (parallel-safe with `memory_hub_unification.md`)
**Depends on:** [pain_bus_unification.md](pain_bus_unification.md) — DefaultNetwork consumes the bus.
**Parallel-safe with:** [memory_hub_unification.md](memory_hub_unification.md) — different files.
**Blocks:** [bio_stack_unification.md](bio_stack_unification.md).

## Goal

Make `DefaultNetwork` construction structurally enforce the (maxim, bus, nac) dependency triple. Today it's constructed inconsistently across entry points — some include it, some don't, some pass a stub `_sim_maxim_stub = object()`. Forgetting to construct it skips the reactive fear-gate silently.

## The repeating shape

```python
default_network = None
try:
    from maxim.default_network.network import DefaultNetwork, DefaultNetworkConfig
    default_network = DefaultNetwork(
        maxim=...,
        bus=...,
        config=DefaultNetworkConfig(...),
        nac=...,
    )
    logger.info("DefaultNetwork active")
except Exception as e:
    logger.debug("DefaultNetwork creation failed: %s", e)  # ← silent skip
```

The `except Exception: logger.debug(...)` pattern means construction failures fall through silently. The default-network-driven fear-gate doesn't fire on this path; nothing tells the user. **Silent failure** with the additional smell of broad-except swallowing.

## Audit (PENDING)

| # | Site | constructed? | swallows exception? | bus passed? | nac passed? |
|---|---|---|---|---|---|
| TBD | cli.py non-sim | ? | ? | ? | ? |
| TBD | simulation/orchestrator.py | ✓ (with `_sim_maxim_stub`) | ✓ swallows | ? | ? |
| TBD | embodied_runtime/agentic_runtime.py | ? | ? | ? | ? |
| TBD | api.py | ? | ? | ? | ? |

**Expected pre-existing bugs to surface:** at least one entry point will be missing DefaultNetwork construction entirely, OR will be swallowing a non-trivial exception during construction. The `_sim_maxim_stub = object()` pattern is a code smell — DefaultNetwork accepts a stub maxim because it only `getattr`s sync methods, but this is undocumented and brittle.

## Design sketch

```python
def build_default_network(
    *,
    nac: NAc,
    pain_bus: PainBus | None = None,
    bus: AgentBus | None = None,
    maxim: Any | None = None,
    config: DefaultNetworkConfig | None = None,
) -> DefaultNetwork:
    """Construct a DefaultNetwork with explicit bio-system deps.

    `nac` is required (the network drives NAc-prediction-based fear);
    `bus` and `maxim` are optional with sensible defaults for headless
    use. `config` defaults to a reasonable DefaultNetworkConfig.
    """
```

**Open design questions:**
- Should the `_sim_maxim_stub = object()` pattern be replaced with an explicit `HeadlessMaximStub` class with documented null-method behavior?
- Should the broad-except wrapping disappear (DefaultNetwork construction failures become user-visible) or become typed (`DefaultNetworkConstructionError`)?
- Is there a `DefaultNetwork` opt-out for sandbox/test use, or should every test pass a real one?

## Pre-merge review round

**Executor lens:**
- Does the new builder correctly handle the sim-mode stub-maxim pattern without re-introducing the `object()` smell?
- Are all DefaultNetwork construction sites migrated?
- Does any test depend on the silent-skip behavior (assumes construction can fail) and need updating?

**Architecture lens:**
- Should DefaultNetwork construction failure be a runtime warning, a runtime error, or a TypeError at signature level?
- Is `Any | None` the right type for `maxim`, or should we declare a `MaximStub` Protocol?
- Cross-check: does Plan 5 (bio_stack) need a specific construction order between PainBus, NAc, and DefaultNetwork?

## Estimated scope

~100 LOC + ~150 LOC tests. Single PR. ~1-2 days.

## Out of scope

- The DefaultNetworkConfig schema itself.
- The fear-gate routing inside DefaultNetwork (separate concern).
- Sleep-replay timing (lives in the sleep-replay plan).
