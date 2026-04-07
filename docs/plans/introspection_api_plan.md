# Introspection API Plan — Programmatic AUT Access

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1 | **DONE** | `AUTIntrospector` class in `simulation/introspection.py` (~180 LOC) |
| 2 | **DONE** | Wired into orchestrator — `aut_introspector` created once, passed to `InspectAUTTool` |
| 3 | **DONE** | `InspectAUTTool` delegates all queries to `AUTIntrospector.dispatch()` |
| 4 | Not started | Standalone experiment runner (`run_campaign()` function) |
| 5 | Not started | AUT self-introspection tool (needs design discussion) |

## Context

The `InspectAUTTool` provides read-only access to the AUT's cognitive state (memory recall, causal links, system stats, etc.) via the simulation tool interface. However, it was designed for LLM orchestrators to call via JSON tool params — which caused persistent issues:

1. **JSON escaping failures**: Qwen2.5-14B produces `{'query': 'memory_recall'}` (Python-style single quotes) instead of valid JSON, causing parse failures and 60s stalls.
2. **Awkward programmatic access**: The workaround accessed the tool through the registry (`orch_registry._tools`), which was brittle and implementation-dependent.
3. **No standalone experiment scripting**: Researchers can't write simple Python scripts that introspect the AUT without bootstrapping the full simulation pipeline.

The hippocampal recall experiment proved that programmatic campaign delivery + programmatic analysis is far more reliable than LLM-driven analysis. This plan extends that principle to a clean API.

**Phases 1-3 shipped:** The `AUTIntrospector` class now provides typed Python methods for all introspection queries. The orchestrator's post-campaign analysis uses `aut_introspector.full_analysis()` instead of the old registry hack. `InspectAUTTool` is a thin wrapper that delegates to `AUTIntrospector.dispatch()`.

## Current State

### What exists
- `InspectAUTTool` in `simulation/tools.py` — 10 query types (memory_recall, causal_links, predict_outcome, pain_history, energy_status, system_stats, concept_query, temporal_patterns, similarity_search, scene_summary)
- Programmatic analysis in `orchestrator.py` — accesses `InspectAUTTool` through registry after campaign delivery
- `ExplainTool` for provenance traces
- Direct hippocampus/NAc access via memory_hub

### What's needed
- A clean `AUTIntrospector` class that wraps all introspection without going through tool dispatch
- Standalone usage outside the sim loop (for experiment scripts)
- Integration with the experiment recording system

## Proposed API

### Core class: `AUTIntrospector`

```python
from maxim.simulation.introspection import AUTIntrospector

# Created from sim components (inside start_simulation_mode)
introspector = AUTIntrospector(
    hippocampus=aut_hippocampus,
    nac=aut_nac,
    memory_hub=aut_memory_hub,
    energy_registry=aut_energy_registry,
)

# Clean method calls — no JSON, no tool dispatch
recall = introspector.memory_recall(keyword="Verath")
stats = introspector.system_stats()
links = introspector.causal_links(tool="respond")
pain = introspector.pain_history(limit=10)
energy = introspector.energy_status()
concepts = introspector.concept_query(name="Elara")
temporal = introspector.temporal_patterns(hours=1)
prediction = introspector.predict_outcome(tool="bash", context="rm -rf /tmp")

# Batch query for experiment recording
analysis = introspector.full_analysis(seed_keywords=["Verath", "Elara", "silver elm"])
# Returns: {
#   "memory_recall": {keyword: results for each},
#   "system_stats": {...},
#   "causal_links": [...],
#   "graph_topology": {"nodes": N, "edges": M, "clusters": [...]},
# }
```

### Integration with experiments

```python
# In research_orchestrator.py (replaces current registry hack)
introspector = AUTIntrospector(
    hippocampus=aut_hippocampus,
    nac=aut_nac,
    memory_hub=aut_memory_hub,
)

analysis = introspector.full_analysis(seed_keywords=["Verath"])
experiment_log.record(
    hypothesis="Verath survives interference",
    method="direct_injection_short",
    result=analysis,
    conclusion=introspector.summarize(analysis),
)
```

### Standalone experiment scripts

```python
#!/usr/bin/env python
"""Quick experiment: does Verath survive 10 interference turns?"""
from maxim.simulation.experiment import run_campaign

result = run_campaign(
    campaign="scenarios/experiments/hippocampal_recall_long.yaml",
    aut_model="mistral-7b",
    seed_keywords=["Verath"],
)

print(f"Memory survived: {result.memory_survived}")
print(f"Recall score: {result.recall_activation_score}")
print(f"Graph path: {result.shortest_path_to_seed}")
```

## Implementation

### Phase 1: AUTIntrospector class (~150 LOC)

**File:** `src/maxim/simulation/introspection.py` (new)

Extract the dispatch logic from `InspectAUTTool._dispatch()` into standalone methods on `AUTIntrospector`. Each method returns typed data (not `ToolOutput`). `InspectAUTTool` becomes a thin wrapper around `AUTIntrospector`.

```python
class AUTIntrospector:
    def __init__(self, *, hippocampus=None, nac=None, memory_hub=None, energy_registry=None):
        ...

    def memory_recall(self, keyword: str = "", limit: int = 10) -> list[dict]: ...
    def system_stats(self) -> dict: ...
    def causal_links(self, tool: str = "") -> list[dict]: ...
    def predict_outcome(self, tool: str, context: str = "") -> dict: ...
    def pain_history(self, limit: int = 10) -> list[dict]: ...
    def energy_status(self) -> dict: ...
    def concept_query(self, name: str) -> dict: ...
    def temporal_patterns(self, hours: float = 1.0) -> dict: ...
    def full_analysis(self, seed_keywords: list[str] | None = None) -> dict: ...
    def summarize(self, analysis: dict) -> str: ...
```

### Phase 2: Wire into orchestrator (~50 LOC)

- Create `AUTIntrospector` in `start_simulation_mode()` after AUT components are built
- Replace the registry hack in post-campaign analysis with `introspector.full_analysis()`
- Pass `introspector` back via `SimulationResult` or a callback

### Phase 3: Refactor InspectAUTTool (~30 LOC)

- `InspectAUTTool` delegates all queries to `AUTIntrospector`
- Keeps the same tool interface for LLM orchestrators
- Single source of truth for introspection logic

### Phase 4: Standalone experiment runner (~100 LOC)

**File:** `src/maxim/simulation/experiment.py` (new)

```python
def run_campaign(
    campaign: str,
    aut_model: str = "mistral-7b",
    seed_keywords: list[str] | None = None,
    debug: bool = False,
) -> ExperimentResult:
    """Run a campaign and return structured results."""
    ...
```

### Phase 5: Agent-facing introspection tool (~50 LOC)

Give the AUT itself access to introspect its own memory (self-awareness):

```python
# Registered on AUT (not orchestrator)
class SelfIntrospectTool(Tool):
    """Let the AUT query its own memory system."""
    name = "remember"
    def execute(self, keyword="", **kw):
        return self._introspector.memory_recall(keyword=keyword)
```

This would let the AUT actively search its memory when encountering the door in the recall phase — currently it can't because it has no tool to query its own hippocampus.

## Open Questions

1. **Should `AUTIntrospector` be available to the AUT itself?**
   - Pro: enables self-directed memory search (crucial for recall experiments)
   - Con: changes the AUT's capability set, affecting experiment validity
   - Recommendation: opt-in via `--aut-introspection` flag

2. **Should `full_analysis()` include spreading activation scores?**
   - Pro: directly measures associative recall strength
   - Con: requires seed memory IDs, which aren't always known
   - Recommendation: yes, when seed_keywords are provided

3. **How to handle introspection during live sim vs post-mortem?**
   - During sim: state is changing, need read-lock consistency
   - Post-mortem: state is frozen, simpler
   - Recommendation: both, with appropriate locking

4. **Should experiment scripts manage their own LLM workers?**
   - The standalone runner needs to set up AUT LLM, bridge, etc.
   - Could reuse `start_simulation_mode` infrastructure
   - Recommendation: wrap `start_simulation_mode` with campaign-only defaults

## Dependencies

- Phase 1-3: no external deps, refactors existing code
- Phase 4: depends on phases 1-2
- Phase 5: needs design discussion (changes AUT capabilities)

## Estimated Scope

| Phase | LOC | Priority | Status |
|-------|-----|----------|--------|
| 1: AUTIntrospector class | ~180 | High | **DONE** |
| 2: Wire into orchestrator | ~20 | High | **DONE** |
| 3: Refactor InspectAUTTool | ~20 | Medium | **DONE** |
| 4: Standalone experiment runner | ~100 | Medium | Not started |
| 5: AUT self-introspection tool | ~50 | Low (needs discussion) | Not started |
| **Total** | **~370** | | **~220 shipped** |

## Related Plans

- [Generative campaign plan](generative_campaign_plan.md) — LLM narrator would use introspector to adapt difficulty
- [Realtime refinement plan](realtime_refinement_plan.md) — refinement persona currently uses InspectAUTTool via LLM
- [Research protocol plan](research_protocol_plan.md) — experiment recording system
