# MemoryHub Unification — required bridge wiring at construction

**Status:** SHELL ONLY. Audit + design pending.
**Parent index:** [biosystem_unification.md](biosystem_unification.md)
**Wave:** 2 of 4 (parallel-safe with `default_network_unification.md`)
**Depends on:** [pain_bus_unification.md](pain_bus_unification.md), [reaction_bus_unification.md](reaction_bus_unification.md) — MemoryHub consumes both buses, so the builders must exist first.
**Parallel-safe with:** [default_network_unification.md](default_network_unification.md) — different files, no overlap.
**Blocks:** [bio_stack_unification.md](bio_stack_unification.md).

## Goal

Push the `MemoryHub.connect(...)` bridge wiring into the `MemoryHub` constructor itself. Today the hub is constructed with bio-system handles, then a separate `.connect(fear_agent=..., default_network=..., ...)` call wires cross-layer bridges. Forgetting `.connect()` or omitting a bridge arg silently skips that bridge.

## The repeating shape

```python
hub = MemoryHub(hippocampus=..., scn=..., nac=..., ec=...)
# easy to forget:
hub.connect(fear_agent=fear_agent, default_network=default_network, ...)
```

If `hub.connect()` is never called: the hub exists, bio-systems are present, but cross-layer signal routing (memory promotion on fear, sleep-replay triggering, etc.) is dead. **Silent failure** — no exception, no log, just missing coordination.

If `hub.connect()` is called but a kwarg is omitted: that specific bridge is dead, the rest work. Even more silent because partial coordination looks like working coordination.

## Audit (PENDING)

Every `MemoryHub(...)` call site in `src/maxim/`:

| # | Site | `.connect()` called? | bridges wired | missing |
|---|---|---|---|---|
| TBD | cli.py | ? | ? | ? |
| TBD | simulation/orchestrator.py | ? | ? | ? |
| TBD | embodied_runtime/agentic_runtime.py | ? | ? | ? |
| TBD | api.py | ? | ? | ? |
| TBD | tests/* | ? | ? | ? |

## Design sketch

**Option A: Fold `.connect()` into `__init__`**

```python
class MemoryHub:
    def __init__(
        self,
        *,
        hippocampus: Hippocampus,
        scn: SCN,
        nac: NAc,
        ec: EntorhinalCortex,
        fear_agent: FearAgent | None = None,
        default_network: DefaultNetwork | None = None,
        atl: ATL | None = None,
        # ... all bridge deps
    ) -> None:
        ...
        # connect() logic moved here
```

Pro: single construction step. Con: large parameter list; some bridge deps may not be available at construction time (default network is built later in some entry points).

**Option B: Required `.connect()` enforced by a sentinel**

```python
class MemoryHub:
    def __init__(self, ...):
        self._connected = False
        ...
    def connect(self, *, fear_agent, default_network, ...):
        self._connected = True
        ...
    def __getattr__(self, name):
        if not self._connected and name in (load-bearing methods):
            raise RuntimeError("MemoryHub.connect() must be called before use")
```

Pro: handles staged construction. Con: still discipline-based at the `connect()` call.

**Option C: `build_memory_hub(...)` helper that does both**

```python
def build_memory_hub(*, hippocampus, scn, nac, ec, fear_agent=None, default_network=None, ...) -> MemoryHub:
    hub = MemoryHub(hippocampus=..., ...)
    hub.connect(fear_agent=..., default_network=..., ...)
    return hub
```

Pro: matches the `build_executor` / `build_pain_bus` pattern. Con: still allows callers to bypass via direct `MemoryHub(...)` construction.

**My current lean: Option A** — collapse `connect()` into `__init__` and require all bridge deps as kwargs (with explicit `None` opt-out). This matches the L1 silent-failure rule most strictly. The "deps not available at construction" objection probably resolves once we audit — most entry points do construct everything before MemoryHub anyway.

**Decision deferred to plan-open audit.**

## Pre-merge review round (mandatory)

**Executor lens:**
- Does the chosen option correctly preserve the lazy-construction pattern in entry points where some bridge deps are constructed AFTER MemoryHub today?
- Are all `.connect()` call sites accounted for?
- Does the migration introduce any double-wiring (deps passed at construction AND via `.connect()`)?

**Architecture lens:**
- Is Option A the right strictness, or does Option C provide a better escape hatch for tests?
- Should `MemoryHub` itself become immutable post-construction, or stay mutable for runtime bridge updates?
- Cross-check: does `agent_factory_canonicalization.md` Stage F1 design need this plan to commit to a specific option before opening?

## Estimated scope

~150 LOC + ~200 LOC tests. Single PR. ~2 days.

## Out of scope

- Memory tier progression (already enforced by `TierTransitionError`).
- The three separate `EpisodicMemory` instances on Hippocampus/NAc/ATL (load-bearing invariant, do not merge).
- Hippocampus persistence path resolution — separate concern.
