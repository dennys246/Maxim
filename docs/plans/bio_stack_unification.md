# Bio-Stack Unification — `build_bio_stack` umbrella

**Status:** SHELL ONLY. Awaits Wave 1 + Wave 2 plans.
**Parent index:** [biosystem_unification.md](biosystem_unification.md)
**Wave:** 3 of 4 (single PR, not parallel-safe with anything in the catalog).
**Depends on:** ALL of [pain_bus_unification.md](pain_bus_unification.md), [reaction_bus_unification.md](reaction_bus_unification.md), [memory_hub_unification.md](memory_hub_unification.md), [default_network_unification.md](default_network_unification.md). Each must ship first; this plan composes them.
**Blocks:** [agent_factory_canonicalization.md](agent_factory_canonicalization.md) Stage F1+ — that plan's design pass becomes a downhill rewrite once `build_bio_stack` exists.

## Goal

Collapse the 30-line bio-pipeline-construction block that gets reproduced in every agent entry point into a single `build_bio_stack(...)` call returning a frozen dataclass containing all wired bio-systems. Every entry point (cli, sim orchestrator, Reachy, public API, future AgentFactory.create_agent) becomes a single function call instead of a hand-rolled 30 lines.

## The repeating shape

Every entry point reproduces this:

```python
hippocampus = Hippocampus(config=...)
nac = NAc()
scn = SCN()
ec = EntorhinalCortex()
atl = ATL()
memory_hub = MemoryHub(hippocampus=..., nac=..., scn=..., ec=..., atl=...)
memory_hub.connect(fear_agent=..., default_network=...)

reaction_bus = build_reaction_bus(...)        # post Wave 1
pain_bus = build_pain_bus(hippocampus=..., nac=..., reaction_bus=...)  # post Wave 1
default_network = build_default_network(nac=..., pain_bus=..., ...)    # post Wave 2

agent.wire_memory_hub(memory_hub)
```

After Waves 1 + 2 each individual builder is structurally enforced, but the **composition** is still hand-rolled at every entry point. Wave 3 collapses it.

## Design sketch

```python
@dataclass(frozen=True)
class BioStack:
    hippocampus: Hippocampus
    nac: NAc
    scn: SCN
    ec: EntorhinalCortex
    atl: ATL
    memory_hub: MemoryHub
    pain_bus: PainBus
    reaction_bus: ReactionBus
    default_network: DefaultNetwork | None
    # ...

def build_bio_stack(
    *,
    persistence_path: str,
    enable_default_network: bool = True,
    fear_agent: FearAgent | None = None,
    additional_pain_subscribers: tuple[Callable, ...] = (),
    additional_reaction_producers: tuple[ReactionProducer, ...] = (),
    config_overrides: BioStackConfig | None = None,
) -> BioStack:
    """Construct the full bio-pipeline as a coherent unit.

    Order matters: ReactionBus first (PainBus depends on it), then
    Hippocampus/NAc/SCN/EC/ATL, then MemoryHub (depends on
    hippocampus/nac/scn/ec), then PainBus (depends on hippocampus,
    nac, reaction_bus), then DefaultNetwork (depends on nac, pain_bus).
    """
```

**Open design questions (resolve at plan-open time):**
- Should `BioStack` be frozen (immutable post-construction) or mutable for runtime extension?
- Should `enable_default_network=False` produce `default_network=None` or a no-op stub?
- Is `persistence_path: str` the right shape, or should it be `paths: BioStackPaths` to accommodate per-bio-system path overrides?
- Should `fear_agent` be a constructor dep or a separate `.attach_fear_agent()` method (lifecycle question)?
- How does `BioStack` interact with `MaximAgent.wire_memory_hub()` — does `build_bio_stack` call it, or does the caller?

## Migration call sites

Same six logical sites as `executor_bootstrap_unification.md`. After this plan ships, each entry point's bootstrap shrinks from ~30 lines to ~5:

```python
bio = build_bio_stack(persistence_path=memory_path, fear_agent=fear_agent)
agent.wire_memory_hub(bio.memory_hub)
executor = build_executor(registry, pain_bus=bio.pain_bus, nac=bio.nac, ...)
```

Total deletion across all entry points: estimated 150-250 LOC of duplicated bootstrap code.

## Pre-merge review round (mandatory + extra-careful)

This is the highest-risk plan in the catalog because it touches every entry point in one PR. Specific review questions:

**Executor lens:**
- Does the construction order in `build_bio_stack` match every entry point's pre-existing implicit order? Any hidden ordering assumption?
- Are there entry-point-specific bio-systems that don't fit the umbrella? (e.g., the sim orchestrator's `aut_introspector._pain_detector` tie-in for telemetry.)
- Does the migration preserve every entry point's current behavior, or are there quiet behavior changes?
- Does `BioStack` correctly compose with the `executor_bootstrap_unification.md` `build_executor` signature? Verify with a test that exercises both builders end-to-end.

**Architecture lens:**
- Is `BioStack` the right abstraction, or is this a god-object?
- Does the frozen dataclass shape preclude any future extension (e.g., adding new bio-systems)?
- Should `build_bio_stack` enforce the construction order via internal sequencing, or leave it documented in the docstring?
- Cross-check: does this plan supersede `agent_factory_canonicalization.md` Stage F1, or do they coexist?
- Does this plan need to coordinate with any active substrate-track work (P3, P4, etc.) that might add new bio-systems?

## Estimated scope

~300 LOC for the builder + ~250 LOC of new tests + ~250 LOC of deletions across migrated entry points = net ~300 LOC. Single PR. ~3-5 days of focused work given the breadth of touched files.

## Doc + memory refinement

- `CLAUDE.md` invariant: "`build_bio_stack` is the canonical bio-pipeline construction site. Every agent entry point calls it; no hand-rolled bio-system construction outside this builder."
- New memory file: `feedback_one_function_one_bio_pipeline.md` — capturing the umbrella-builder lesson.
- Update `agent_factory_canonicalization.md` Stage F1 to reflect that the design pass is now downhill.
- Update `biosystem_unification.md` status row to "SHIPPED" + add a "the catalog is closed" note.
- Sweep CLAUDE.md "Architectural invariants" to remove any per-bio-system construction guidance that's now subsumed.

## Out of scope

- Adding new bio-systems (substrate-track territory).
- Changing the bio-systems' internal interfaces.
- The `wire_memory_hub` lifecycle — separate concern.
- Per-agent bio-isolation in AgentPool — that's `agent_factory_canonicalization.md` Stage F1+.

## Why this is Wave 3, not Wave 1

The temptation is to ship `build_bio_stack` first as a single mega-PR. Resist. Here's why the wave ordering matters:

1. **Each individual builder needs its own audit + review** — combining them hides bugs in the noise.
2. **`build_bio_stack` cannot be reviewed independently** of `build_pain_bus`, `build_reaction_bus`, `build_memory_hub`, `build_default_network` — a reviewer would need to understand all four signatures simultaneously.
3. **Risk compounds** — one PR touching every entry point is a bisect nightmare.
4. **Wave 1 + Wave 2 each ship in 1-3 days**; Wave 3 ships in 3-5. The total wall-clock is similar, but the per-PR risk is much smaller.
5. **Pre-existing bugs surface in Wave 1/2 audits** and get fixed in scope. By Wave 3 the surface is clean.

The structural ceiling is `agent_factory_canonicalization.md`. The structural floor under that is `build_bio_stack`. The structural floor under THAT is the four individual builders. Don't skip rungs.
