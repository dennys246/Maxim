# Memory Layer Lifecycle

How memories form, connect, and are managed long-term in Maxim's
bio-inspired memory architecture.

---

## Memory Layers

Three layers, each storing a different memory type:

| Layer | Record Type | Brain Mapping | Purpose |
|---|---|---|---|
| **Hippocampus** | `EpisodicMemory` | Hippocampus | Observe-decide-act-evaluate episodes |
| **ATL** | `SemanticMemory` / `Concept` | Anterior Temporal Lobe | Facts, entities, concept relationships |
| **Angular Gyrus** | `MathMemory` | Angular Gyrus | Numerical knowledge, statistics, patterns |

All layers implement the `MemoryLayer` ABC (`src/maxim/memory/layer.py`):
`store()`, `get()`, `remove()`, `recall()`, `recall_by_ids()`,
`recall_associated()`, `save()`, `load()`, `consolidate()`.

---

## Memory Tiers

Memories flow through three lifecycle tiers (defined in `src/maxim/agents/bus.py`):

```
FORMING ──> SHORT_TERM ──> LONG_TERM
```

| Tier | Duration | Eviction | Description |
|---|---|---|---|
| **FORMING** | During agent pipeline | Protected | Fields populated incrementally (perception, decision, action, outcome) |
| **SHORT_TERM** | Minutes | Buffer-based (fixed window) | Recent context, fast salience decay |
| **LONG_TERM** | Persistent | Age-based (consolidation) | Consolidated knowledge, 2x retention boost |

Active-reference context (recent percepts, outcomes, speech, etc.) is owned by
`WorkingMemorySet` in `agents/working_memory.py` — an Exec-owned layer, not a
memory tier. The old `MemoryTier.WORKING` was removed in 0.8.

`WorkingMemoryEntry[T]` wraps any `MemoryRecord` subclass with agent-level
metadata: `tier`, `salience`, `decay_rate`, `predicted_outcomes`, `source`.

---

## Lifecycle Diagram

```
Perception Event
    |
[Capture] --> Hippocampus stores EpisodicMemory
    |
[Formation] --> FORMING tier
    |-- ConceptExtractor registers percept concepts --> ATL
    |-- PatternCompleter attaches PredictedOutcome
    |-- Cross-layer edges created (INSTANCE_OF)
    |
[Outcome] --> SHORT_TERM tier (outcome-triggered promotion)
    |-- Also written to WorkingMemorySet for Exec prompt context
    |
[Recall] --> MemoryAgent retrieves for LLM context
    |-- ConceptGrounder enriches with AG math (async via WorkerPool)
    |-- Cross-layer spreading activation
    |-- access_count incremented
    |
[Sleep Consolidation]
    |-- Consolidate: important memories --> LONG_TERM (2x removal boost)
    |-- Compress: old records --> CompressedMemory (~200 bytes vs ~2.5KB)
    |-- Remove: stale, low-score memories --> deleted
    |-- Promote: NAc/StatisticianAgent patterns --> ATL concepts
    |
[Long-Term Storage] --> Resistant to removal, slow decay
```

---

## Cross-Layer Connections

### CrossLayerGraph

Bidirectional edges between memories in different layers
(`src/maxim/memory/cross_layer.py`):

| Edge Type | Direction | Meaning |
|---|---|---|
| `INSTANCE_OF` | Episode --> Concept | Episode is an instance of this concept |
| `DERIVED_FROM` | Concept --> Episode(s) | Concept derived from these episodes |
| `QUANTIFIES` | AG Record --> Concept | AG provides numerical characterization |
| `STATISTICALLY_CONFIRMS` | AG --> Concept | AG pattern validates concept |
| `COMPUTED_FROM` | AG --> Layer data | AG result derived from another layer |
| `TEMPORALLY_CORRELATES` | SCN --> Memory | Temporal rhythm linkage |

### Concept.memory_refs

Concepts track which memories reference them across layers:

```python
memory_refs: dict[str, dict[str, None]]
# layer_name -> ordered dict of memory_ids (FIFO pruning at MAX_REFS_PER_LAYER=200)
```

Updated by ConceptExtractor on capture and ConceptGrounder on recall.

---

## Consolidation & Pruning

### Sleep Consolidation (Hippocampus.sleep())

Three operations during sleep cycle:

1. **Promotion to long-term**: High salience (>0.9), high novelty (>0.9),
   successful user interaction, or high access count (>=5). Marked with
   `long_term=True` and `consolidated_at` timestamp.

2. **Compression**: Full `EpisodicMemory` --> `CompressedMemory` (goal, tool,
   success, salience, novelty). Triggered when age > threshold and score < threshold.

3. **Removal**: Score below retention threshold. Graph edges and callbacks
   cleaned up automatically.

### Eviction Strategies (`src/maxim/memory/strategies.py`)

| Strategy | Scoring | Used By |
|---|---|---|
| `AccessBasedStrategy` | Recency + access frequency + centrality | Default for both layers |
| `ImportanceBasedStrategy` | Novelty + salience + success + user interaction | Composite component |
| `TemporalAwareStrategy` | SCN rhythm-aware (sole representatives boosted) | When SCN connected |
| `CompositeStrategy` | Weighted combination of multiple strategies | Configurable |

Both Hippocampus and ATL use `_get_memory_strategy()` for consistent
scoring between store-time eviction and sleep-time consolidation.

---

## Promotion Pipeline

`SemanticPromoter` (`src/maxim/memory/semantic_promoter.py`) orchestrates
multi-source promotion from episodic patterns to ATL concepts:

1. Collect candidates from all `PromotionSource` instances (NAc, StatisticianAgent)
2. Apply IPS randomness quality gate (filters noise)
3. `ATL.find_or_create()` -- create or reinforce concept
4. Create cross-layer edges linking concept to source records

Provenance tracking: `EPISODIC_CONSOLIDATION` (NAc rewards),
`AGENT_INFERENCE` (StatisticianAgent patterns), `DIRECT_INGESTION` (RAG).

---

## Concept Pipeline

### ConceptExtractor (capture-time, async worker thread)

Registered as capture callback on Hippocampus. Extracts concepts from
episodic memories:

- Objects (detected_objects), People (detected_people)
- Location (observations), Goal tokens (normalized), Actions (tool_name)
- Forms categorical relationships between co-occurring concepts (confidence 0.3)
- Updates `Concept.memory_refs["hippocampus"]` to track source episodes

### ConceptGrounder (recall-time, async via WorkerPool)

Piggybacks on concept recall for numerical enrichment:

- **Sync path**: Extract numerics from episodes, compute IPS/AG stats, cache results
- **Async path** (small tier): Jaccard co-occurrence scoring, relationship modulation
- **Async path** (small tier): AG MathMemory stores, QUANTIFIES edges

Falls back to sync when no WorkerPool available.

### PatternCompleter (FORMING stage, sync)

Provides predictions during memory formation:

1. Match concepts from current percepts (objects, people, goal tokens)
2. Load linked episodes via `concept.memory_refs["hippocampus"]`
3. Extract `PredictedOutcome` from past decisions/actions/outcomes
4. Enrich with per-concept `MathContextEntry` from AG

---

## Key Files

| File | Purpose |
|---|---|
| `src/maxim/memory/layer.py` | MemoryLayer ABC |
| `src/maxim/memory/types.py` | MemoryRecord, EpisodicMemory, CompressedMemory |
| `src/maxim/memory/semantic_types.py` | Concept, SemanticMemory |
| `src/maxim/memory/hippocampus.py` | Episodic layer + sleep consolidation |
| `src/maxim/memory/atl.py` | Semantic layer + capacity enforcement |
| `src/maxim/memory/cross_layer.py` | CrossLayerGraph edge types |
| `src/maxim/memory/strategies.py` | Eviction/consolidation strategies |
| `src/maxim/memory/concept_extractor.py` | Percept-to-concept extraction |
| `src/maxim/memory/concept_grounder.py` | Numerical enrichment + async pipeline |
| `src/maxim/memory/concept_context.py` | ConceptContextBuilder |
| `src/maxim/memory/pattern_completer.py` | Cross-layer prediction |
| `src/maxim/memory/semantic_promoter.py` | Pattern promotion pipeline |
| `src/maxim/agents/bus.py` | MemoryTier, WorkingMemoryEntry |
| `src/maxim/agents/memory_agent.py` | Formation orchestration |
| `src/maxim/integration/memory_hub.py` | Central coordinator |
