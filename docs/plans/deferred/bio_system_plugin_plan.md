# Bio-System Plugin Discovery — Deferred Platform Work

**Status:** Deferred. Not on the critical path to 1.0.
**Revive when:** (a) the first external contributor wants to add a bio-system, OR (b) the `BioSystem` Protocol exists (via substrate_plan's incremental contracts layer) AND there's appetite to extend the existing `maxim.robots` entry-point pattern to bio-systems, OR (c) a research collaborator needs to A/B-test substrate variants without forking the codebase.
**Prereq:** the `BioSystem` Protocol must exist. That Protocol is defined incrementally during substrate phase work, so by the time this plan revives, the Protocol is already in place and this plan is a "move + register" refactor, not a ground-up design exercise.

## Why this is deferred, not killed

The Maxim codebase already has plugin discovery for robots via a `maxim.robots` entry-point group (per CLAUDE.md). Extending the same pattern to bio-systems would mean a new bio-system is "a class that implements `BioSystem` + an entry-point registration" — no touches to `MemoryHub`, `AgentFactory`, lifecycle code, persistence code, or testing harness.

This is genuinely valuable if Maxim becomes a platform that other researchers extend. Grad students adding "amygdala for emotional modulation" or "dorsal stream for spatial vision" should be able to do it without the cross-cutting editing pain that adding a bio-system requires today.

But **it's only valuable if those contributors exist.** For a single-developer research project, the cost of maintaining plugin discovery machinery outweighs the cost of touching seven files once a quarter when you personally add a new bio-system. The plugin system pays for itself exactly when the second developer joins — and not before.

## What this plan would look like when revived

### Phase 1 — `BioSystem` Protocol (already done by the time you read this)

Defined as part of substrate_plan's incremental contracts layer. See [../substrate_plan.md](../substrate_plan.md) "Contracts layer" section. The Protocol captures: `name`, `depends_on: list[str]`, `save`, `load`, `snapshot`, `on_percept`, `on_tick`, plus whatever else the real bio-systems actually expose by the time this plan revives.

If the Protocol didn't land incrementally (because substrate phase work didn't need all of it), reviving this plan starts with finishing the Protocol definition. That's ~50 LOC of reverse-engineering from the existing bio-systems (ATL, Hippocampus, NAc, SCN, PerceptTraceBuffer).

### Phase 2 — `maxim.biosystems` entry-point group

Mirror the `maxim.robots` pattern. Add a new entry-point group in `pyproject.toml`:

```toml
[project.entry-points."maxim.biosystems"]
atl = "maxim.memory.atl:ATL"
hippocampus = "maxim.memory.hippocampus:Hippocampus"
nac = "maxim.decisions.nac:NAc"
scn = "maxim.time.scn:SCN"
trace_buffer = "maxim.memory.trace_buffer:PerceptTraceBuffer"
```

At runtime, `AgentFactory` discovers bio-systems via `importlib.metadata.entry_points(group="maxim.biosystems")`, instantiates each one, wires dependencies, and runs them in topological order based on `depends_on`.

**Scope:** ~200 LOC (entry-point discovery + AgentFactory refactor + topological sort + tests).

### Phase 3 — Dependency graph + lifecycle

Bio-systems declare what they read from via `depends_on`. AgentFactory validates the graph (no cycles, all dependencies exist) and runs `on_percept` / `on_tick` hooks in order. This replaces the current hand-wired sequence in `AgentFactory.create_agent()`.

**Scope:** ~150 LOC (topological sort + lifecycle runner + dependency validator + tests).

### Phase 4 — Per-agent config for bio-system enablement

Campaign YAML or runtime config can enable/disable bio-systems per agent. Useful for A/B tests: "this agent has NAc, this one doesn't." Without the plugin system, this is a code change; with it, it's a config toggle.

```yaml
agent_profile:
  biosystems:
    enabled: [atl, hippocampus, scn, trace_buffer]
    disabled: [nac]  # A/B test: reward-modulated vs not
```

**Scope:** ~100 LOC (config loader + AgentFactory respecting enablement flags + tests).

### Phase 5 — Plugin integration tests

A CI check that verifies:
1. All registered bio-systems implement the `BioSystem` Protocol (mypy-level and runtime)
2. Dependencies form a valid DAG
3. Removing any optional bio-system doesn't crash the runtime
4. Adding a stub bio-system via entry-point is discovered without code changes

**Scope:** ~100 LOC (test suite).

**Total scope when revived:** ~550 LOC, 1–2 weeks of focused work. Most of the cost is the AgentFactory refactor (Phase 2 + 3); the rest is incremental.

## Why this isn't bigger

The substrate plan's incremental contracts layer does the hard intellectual work — defining what a bio-system is and what Protocol it satisfies. By the time this plan revives, that's already done. What's left is mechanical: replace hand-wiring with discovery, add a toposort, document the registration pattern.

If the contracts layer is *not* in place when this plan revives (because the substrate work didn't need the full `BioSystem` Protocol), add a Phase 0: finish defining the Protocol. That's still small — ~50–100 LOC.

## When to revive

**Strong signals:**
- A second developer joins the project and wants to add their own bio-system
- A research collaborator proposes an A/B test between substrate variants and asks how to disable one bio-system
- Maxim is being positioned as a platform with external contributors in mind
- You (the primary developer) find yourself adding bio-systems frequently enough that the cross-cutting editing pain is a bottleneck

**Weak signals (don't revive yet):**
- "It would be nice to have"
- "A plugin system feels more professional"
- "What if someone eventually..."

Speculative abstraction is the enemy of simple plans. This one stays in `deferred/` until there's a concrete trigger.

## Relationship to other plans

- **[../substrate_plan.md](../substrate_plan.md)** — provides the `BioSystem` Protocol via the contracts layer. Without it, this plan starts with a larger Phase 0.
- **[../foundations_plan.md](../foundations_plan.md)** — F0.5 (agent_id threading), F0.2 (PerceptTraceBuffer as a shared resource), and F0.6 (factory consolidation) are all load-bearing for the plugin system because they establish per-agent isolation patterns that plugins need to respect. Without them, a plugin system would inherit silent multi-agent bugs.
- **robot plugin discovery (existing)** — this plan copies the pattern from `maxim.robots`. If that pattern changes substantially before this plan revives, re-read the robot plugin code first and adjust.

## Non-goals

- **No plugin discovery for sensors, tools, or LLM backends.** Those are different abstractions with different needs. A general "plugin everything" refactor is out of scope — each plugin system stands on its own.
- **No hot-reload.** Bio-systems are wired at agent construction time; reloading them at runtime is out of scope.
- **No plugin marketplace or registry service.** Entry-point discovery via Python's standard mechanism is the whole system. No HTTP, no third-party registry.

## If you're reading this cold

You found this plan because you're considering adding a bio-system plugin system. Before you start:

1. **Confirm the trigger.** Is there a concrete need, or is this speculative? If speculative, close this file and come back when the need is real.
2. **Check the contracts layer.** Does `src/maxim/contracts/biosystem.py` exist with a real `BioSystem` Protocol? If yes, start at Phase 2. If no, start at Phase 0 (define the Protocol).
3. **Re-read the current `AgentFactory`.** The refactor in Phases 2–3 replaces hand-wiring with discovery. Understand what the hand-wiring currently does before replacing it.
4. **Run the existing robot plugin discovery.** Find a robot entry-point, trace how it's discovered and instantiated. This plan mirrors that pattern.
