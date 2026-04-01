# Causal Memory Integration Plan

> **Status:** Not started. Depends on cognitive pain system (implemented) and RPE flow (implemented).

Wire NAc's causal learning into EC's similarity space and hippocampus's associative graph. Currently, causal links are siloed in NAc's flat dict — EC can't find similar causal patterns, the hippocampal graph has no causal edges (EdgeType.CAUSES exists but is never created), and spreading activation can't traverse causal chains.

---

## Current State (Verified Against Code)

### What Works

| System | What It Does | Storage |
|--------|-------------|---------|
| **NAc** | Rescorla-Wagner causal learning (event → outcome) | `_links: dict[str, list[CausalLink]]` — flat, by event_signature |
| **EC** | Multi-modal similarity (LSH, structural, temporal, semantic) | `_signatures: dict[str, SituationSignature]` — by memory_id |
| **Hippocampus** | Episodic memory with associative graph | `_memories` + `_graph` with ASSOCIATES edges |
| **ToolPainBridge** | Routes tool errors → NAc + SCN, exposes RPE | `_pending_tools`, `_last_rpe` |

### What's Missing

1. **EC doesn't index causal links.** NAc can find links by exact event_signature, but can't answer "what causal patterns are similar to this one?" EC has the similarity engine but doesn't know about causal links.

2. **CAUSES edges never created.** `EdgeType.CAUSES` exists in the enum (bus.py:77). `spreading_activation()` already traverses it (hippocampus.py). But NO code path creates CAUSES edges. The graph only has ASSOCIATES edges (from `_form_associations()` during capture).

3. **NAc context similarity is naive.** `_context_similarity()` (nac.py) does key-by-key exact matching. EC's LSH + semantic embeddings would provide much richer similarity for causal pattern matching.

4. **CausalLink.memory_ids is a weak backlink.** CausalLinks store memory IDs that contributed to the link, but the hippocampal graph has no edge FROM those memories TO the causal outcome. You can go CausalLink → memory_ids → episodes, but you can't go episode → spreading_activation → related causal outcomes.

---

## Biological Mapping

| Brain Structure | Maxim Component | Role in Causal Memory |
|----------------|----------------|----------------------|
| **Entorhinal Cortex** | `EC` | Gateway — indexes causal patterns in similarity space, enables "find similar causes" |
| **Hippocampus** | `Hippocampus._graph` | Episodic binding — CAUSES edges link event episodes to outcome episodes |
| **Nucleus Accumbens** | `NAc` | Reward learning — computes RPE, updates predicted_value via Rescorla-Wagner |
| **SCN** | `SCN` | Temporal context — when did this causal pattern occur? (already wired) |
| **Amygdala** | `FearAgent + PainDetector` | Aversive gating — should we avoid this action? (already wired) |

The flow mirrors dopaminergic signaling: NAc detects surprise (RPE), hippocampus forms the episodic memory, EC indexes it for future retrieval, and spreading activation through CAUSES edges enables causal reasoning ("last time I did X in context Y, Z happened").

---

## Implementation

### Phase 1: Create CAUSES Edges in Hippocampus Graph

When NAc records an outcome and the CausalLink references hippocampal episodes, create CAUSES edges between the event episode and the outcome episode.

**Where:** In `ToolPainBridge` (or a new `CausalGraphBridge`), after `nac.record_outcome()` returns updated links:

```python
def _create_causal_edges(self, links: list[CausalLink], outcome_memory_id: str | None):
    """Create CAUSES edges from event episodes to outcome episode."""
    if not outcome_memory_id or not self._hippocampus:
        return
    for link in links:
        for event_mem_id in link.memory_ids[-5:]:  # Last 5 contributing episodes
            if event_mem_id != outcome_memory_id:
                self._hippocampus.graph.add_edge(
                    source=event_mem_id,
                    target=outcome_memory_id,
                    edge_type=EdgeType.CAUSES,
                    weight=link.confidence,  # Strong causal links get stronger edges
                )
```

**Effect:** `spreading_activation()` already traverses CAUSES edges — now it finds them. Recalling an event episode activates related outcome episodes through causal chains.

**Integration point:** ToolPainBridge needs an optional `hippocampus` parameter (like it already has `nac` and `scn`). Wire in MemoryHub or agentic_runtime.

### Phase 2: Register Causal Links in EC Similarity Space

Index CausalLinks in EC so "find similar causal patterns" works. A causal link IS a situation signature — it has tool_name, context, outcome, temporal context.

**Where:** In NAc after creating/updating a CausalLink, register it with EC:

```python
# In NAc.record_outcome_full(), after link is created/updated:
if self._ec is not None:
    sig = SituationSignature(
        structural_hash=hash(f"{link.event_signature}:{link.outcome_signature}"),
        temporal_hash=self._scn.current_bins() if self._scn else (0, 0, 0, 0),
        tool_name=link.event_signature.split(":")[-1] if ":" in link.event_signature else "",
        outcome_type=link.outcome_valence.value,
        mode="",
        goal_keywords=tuple(link.event_context.get("goal", "").split()[:3]),
        context_hash=hash(frozenset(link.event_context.items())),
        semantic_hash=(),
    )
    self._ec.register(f"causal:{link.id}", sig)
```

**Effect:** EC can now answer "find causal patterns similar to this tool execution." This enables:
- "What happened last time I ran tests in a directory like this?"
- "What tools tend to fail in contexts similar to the current one?"
- "What causal chains led to success when the goal was similar?"

**Integration point:** NAc needs an optional `ec` parameter. Currently NAc has no EC reference. Add it in MemoryHub wiring.

### Phase 3: EC-Enhanced Causal Prediction

Replace NAc's naive `_context_similarity()` with EC's LSH + semantic similarity for causal link retrieval. When predicting an outcome, first query EC for similar causal situations, then score with NAc's Rescorla-Wagner values.

**Where:** In `NAc.predict()`:

```python
def predict(self, event_type, event_signature, context=None):
    # Current: only looks up _links[event_signature] (exact match)
    local_links = self._links.get(event_signature, [])
    
    # NEW: also query EC for similar causal patterns
    if self._ec is not None:
        sig = self._build_signature(event_signature, context)
        similar = self._ec.find_similar(sig, k=10, min_similarity=0.5)
        for causal_id, score in similar:
            if causal_id.startswith("causal:"):
                link_id = causal_id[7:]  # Strip "causal:" prefix
                link = self._get_link_by_id(link_id)
                if link and link not in local_links:
                    local_links.append(link)  # Include similar causal patterns
    
    # Score and return best prediction (existing logic)
    ...
```

**Effect:** NAc prediction becomes context-aware through EC's similarity engine. "internet_search on restricted network" finds causal links from past restricted network situations, even if the exact event_signature differs slightly.

**Caution:** This changes prediction behavior. Roll out incrementally — start with EC results as a secondary signal (weight 0.3) alongside exact matches (weight 1.0), then tune.

### Phase 4: RPE-Driven Capture with Causal Context

Close the loop: when the agent loop calls `hippocampus.capture_from_loop_async()`, boost the perception's salience by the RPE magnitude from `executor.get_last_rpe()`.

**Where:** In `agent_loop.py`, where `capture_from_loop_async` is called (lines 1293-1305):

```python
# After tool execution, before capture:
rpe = executor.get_last_rpe()
if rpe > 0.0:
    # Boost salience for surprising outcomes
    observation["_rpe_salience_boost"] = min(1.0, rpe)
    
# In capture_from_loop / capture_from_loop_async:
salience = observation.get("salience", 0.0)
rpe_boost = observation.get("_rpe_salience_boost", 0.0)
effective_salience = min(1.0, salience + rpe_boost * 0.5)  # RPE contributes up to +0.5
```

**Effect:** Surprising tool outcomes (high RPE) produce high-salience memories that get promoted to long-term storage. Routine outcomes stay low-salience and may be evicted during consolidation.

---

## Implementation Sequencing

| Phase | What | Effort | Impact | Dependencies |
|-------|------|--------|--------|-------------|
| **1** | CAUSES edges in hippocampus graph | Small | High | ToolPainBridge + hippocampus param |
| **2** | Register causal links in EC | Medium | High | NAc + EC param, SituationSignature building |
| **3** | EC-enhanced causal prediction | Medium | High | Phase 2, careful tuning needed |
| **4** | RPE-driven capture salience | Small | Medium | get_last_rpe() (already implemented) |

Phase 1 and Phase 4 are independent and can be done in parallel. Phase 2 before Phase 3 (indexing before querying). Phase 3 requires the most care — it changes prediction behavior.

---

## Wiring Changes

### ToolPainBridge

```python
class ToolPainBridge:
    def __init__(self, nac, pain_detector, scn=None, hippocampus=None):
        # ... existing ...
        self._hippocampus = hippocampus  # NEW: for CAUSES edges
```

### NAc

```python
class NAc:
    def __init__(self, config=None, ec=None):
        # ... existing ...
        self._ec = ec  # NEW: for causal link indexing + similarity queries
```

### MemoryHub wiring (agentic_runtime.py)

```python
# Currently:
nac = NAc(config=nac_config)
tool_pain_bridge = ToolPainBridge(nac=nac, pain_detector=detector, scn=scn)

# After:
nac = NAc(config=nac_config, ec=ec)  # NAc can index causal links in EC
tool_pain_bridge = ToolPainBridge(
    nac=nac, pain_detector=detector, scn=scn,
    hippocampus=hippocampus,  # ToolPainBridge can create CAUSES edges
)
```

---

## What NOT to Change

- **NAc's internal `_links` storage** — it's the primary causal link store. EC provides similarity indexing ON TOP of it, not replacing it.
- **EC's SituationSignature structure** — it already has the fields needed (tool_name, outcome_type, temporal_hash, context_hash). Just populate them from CausalLink data.
- **Hippocampus capture logic** — the interestingness criteria (salience > 0.7 → consolidation candidate) are correct. We just boost salience with RPE, not change the thresholds.
- **Spreading activation parameters** — decay=0.5, max_depth=3, threshold=0.05 are tuned for ASSOCIATES edges. CAUSES edges should use the same parameters initially.

---

## Risks

1. **CAUSES edge proliferation.** If every tool execution creates a CAUSES edge, the graph could grow large. Mitigate: only create CAUSES edges when RPE > 0.3 (surprising outcomes). Routine outcomes don't need causal edges.

2. **EC index bloat.** Registering every CausalLink in EC adds entries. Mitigate: only register links with observation_count >= 3 (established patterns, not one-off events).

3. **Prediction behavior change (Phase 3).** EC-enhanced prediction could surface unexpected causal links. Mitigate: start with low weight (0.3) for EC-sourced links vs 1.0 for exact matches. Tune based on prediction accuracy.

4. **Circular activation.** CAUSES edges + ASSOCIATES edges could create activation loops (A associates B, B causes C, C associates A). Mitigate: spreading_activation already has max_depth=3 and decay=0.5, which bounds activation naturally.

---

## Tests Needed

- CAUSES edge created when RPE > threshold (Phase 1)
- CAUSES edge NOT created for routine outcomes (Phase 1)
- Spreading activation traverses CAUSES edges (Phase 1)
- CausalLink registered in EC with correct signature (Phase 2)
- EC.find_similar returns causal links (Phase 2)
- NAc.predict finds similar causal patterns via EC (Phase 3)
- Capture salience boosted by RPE magnitude (Phase 4)
- RPE=0 produces no salience boost (Phase 4)
