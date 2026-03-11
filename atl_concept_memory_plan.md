# ATL Concept Memory Plan

Defines the `Concept` class as a universal abstraction bridging memory types
(episodic, semantic, mathematical, future types) and outlines how the ATL
evolves from a pattern-promotion store into a true semantic concept memory.
AG (Angular Gyrus) serves as the numerical backbone — computing statistical
properties during recall and grounding concept relationships with evidence.

**Depends on:** the unified memory system (WorkingMemoryEntry, PredictedOutcome,
MathContextEntry, and the `_pattern_completion_fn` hook — all implemented)
for true piggybacking in concept grounding (Phase A3). Phases A1-A2
can proceed independently — Hippocampus capture callbacks and ATL
`find_or_create()` already exist.

**Prerequisites in existing code:** AG, IPS, CrossLayerGraph, and the
Semantics relationship system are all implemented and operational.
`Semantics.update_edge()` (Phase A0) and `recall_by_ids()` (Phase A5a)
are implemented.

---

## Problem

### What ATL is today

The ATL is a functional `MemoryLayer` implementation that stores
`SemanticMemory` records — named concepts with categories, definitions,
typed relationships, and provenance tracking. It receives concepts through
two paths:

1. **SemanticPromoter** — during sleep, scans NAc reward patterns and
   StatisticianAgent confirmed patterns, promotes qualifying entries to
   ATL concepts with cross-layer edges back to source episodes.
2. **Direct ingestion** — future RAG/document ingestion path (not yet
   implemented).

ATL already has:
- `SemanticMemory` type (extends `MemoryRecord`): name, definition, category,
  properties, provenance, source_episode_ids, confidence, reinforcement_count
- `Semantics` manager: typed relationships (IS_A, HAS_PART, CAUSES, etc.)
  with extensible `RelationshipRegistry`
- `CrossLayerGraph`: edges linking ATL concepts to Hippocampus episodes and
  Angular Gyrus math records
- Consolidation: compress old concepts, remove stale ones
- Context index: `name:X` and `category:Y` hash lookups

### What's missing

1. **Concepts are only promoted patterns.** The current ATL stores causal
   patterns ("grasp -> success") and operational patterns from the
   StatisticianAgent. It doesn't store object concepts ("chair"), person
   concepts ("Dennis"), or action concepts ("navigate"). These live as
   raw strings in Hippocampus's `_context_index` (`"objects:chair"` -> IDs)
   but never become first-class ATL concepts.

2. **No percept-to-concept bridge.** When Maxim sees a chair, Hippocampus
   indexes `"objects:chair"` as a hash key. But there's no ATL concept
   for "chair" that accumulates properties, relationships, and cross-layer
   references over time. The percept "chair" is a string, not a concept.

3. **No cross-memory-type connectivity.** If Hippocampus knows "I saw Dennis
   move a chair" and Angular Gyrus knows "chair weighs ~5kg", there's no
   "chair" concept connecting them. `CrossLayerGraph` has the edge types
   (`INSTANCE_OF`, `QUANTIFIES`) but nothing populates them for percept-
   derived concepts.

4. **AG is isolated from episodic data.** AG stores learned patterns from
   StatisticianAgent escalations but never touches the numerical data in
   episodic memories (execution times, salience, distances, success rates).
   It computes in isolation rather than grounding its analysis in the
   memories that produced the numbers.

### What ATL should become

Biologically, the anterior temporal lobe is the convergence zone where
multimodal information integrates into unified semantic representations.
The angular gyrus provides numerical/spatial cognition that grounds those
representations in quantitative evidence. Together, they form a system
where percepts become concepts, concepts form relationships, and those
relationships are backed by statistical evidence.

In Maxim terms: **ATL is the concept graph. AG is its numerical backbone.**
When Maxim sees a "chair", ATL should know what a chair is and what it's
related to. AG should know the numerical properties: typical distance,
grasp success rate, average interaction time. The `Concept` class is the
node that holds all of this together, and the typed relationship graph —
grounded by AG's math — is the tissue connecting concepts.

---

## Architecture

### The Concept class

A `Concept` is a lightweight, universal node that lives in ATL and
aggregates references to memories across all `MemoryLayer` implementations.
It extends the existing `SemanticMemory` with cross-layer reference tracking.

```python
@dataclass
class Concept(SemanticMemory):
    """A semantic concept with cross-layer memory references.

    Extends SemanticMemory with explicit tracking of which memories across
    all layers reference this concept. This is the ATL's core unit —
    the bridge between percepts and memories.

    Inherits from SemanticMemory: name, definition, category, properties,
    provenance, source_episode_ids, confidence, reinforcement_count,
    embedding_text, and all MemoryRecord fields.
    """

    # Cross-layer references: layer_name -> set of memory_ids
    # These are the memories that mention/involve this concept.
    # Bounded per layer to prevent unbounded growth on frequently
    # seen concepts (e.g., "kitchen" across thousands of episodes).
    memory_refs: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    # Maximum refs tracked per layer. When exceeded, oldest refs are
    # pruned (FIFO via insertion order — Python 3.7+ sets maintain
    # insertion order for iteration, so we pop from the front).
    MAX_REFS_PER_LAYER: int = 200

    def add_ref(self, layer_name: str, memory_id: str) -> None:
        """Register a memory that references this concept.

        If the ref set for this layer exceeds MAX_REFS_PER_LAYER, the
        oldest ref is pruned. This bounds memory_refs to prevent
        unbounded growth on frequently-seen concepts.
        """
        refs = self.memory_refs[layer_name]
        refs.add(memory_id)
        if len(refs) > self.MAX_REFS_PER_LAYER:
            # Pop oldest (first inserted) ref
            oldest = next(iter(refs))
            refs.discard(oldest)
        self.touch()

    def remove_ref(self, layer_name: str, memory_id: str) -> None:
        """Unregister a memory reference (e.g., when memory is deleted or compressed)."""
        self.memory_refs[layer_name].discard(memory_id)

    def ref_count(self, layer_name: str | None = None) -> int:
        """Count references, optionally filtered by layer."""
        if layer_name:
            return len(self.memory_refs.get(layer_name, set()))
        return sum(len(ids) for ids in self.memory_refs.values())
```

**Why extend SemanticMemory, not replace it?** `SemanticMemory` already has
name, category, definition, properties, confidence, reinforcement, provenance,
and source_episode_ids. `Concept` adds `memory_refs` for cross-layer tracking.
This is backward compatible — existing promoted SemanticMemory records
continue to work. The migration is additive: concepts promoted from the
existing pipeline get empty `memory_refs` that populate over time.

**Why not a separate class from SemanticMemory?** Because biologically,
concepts ARE semantic memories. The ATL doesn't have two types of things —
it has concepts with varying richness. A freshly perceived "chair" starts
sparse (just a name and category). A well-reinforced "chair" accumulates
relationships and cross-layer refs. Same type, different maturity.

**Why no `observed_properties` or `activation` fields?** A concept's
properties ARE its relationships in the ATL graph — grounded by AG's
numerical analysis. "Chair has typical_distance 1.2m" is a `QUANTIFIES`
relationship from an AG MathMemory, not a separate data structure.
Activation/priming is unnecessary — concept recall is a direct lookup
based on current percept, not a decay simulation.

### Concept serialization

`memory_refs` is `dict[str, set[str]]` but JSON has no set type. Concept
overrides `to_dict()` and `from_dict()` to handle the conversion, building
on `SemanticMemory`'s existing serialization. A `"_concept": True` flag
distinguishes Concept from plain SemanticMemory during deserialization
(same pattern as `CompressedSemantic`'s `"_compressed": True` flag).

```python
@dataclass
class Concept(SemanticMemory):
    # ... fields from above ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize Concept, converting sets to sorted lists for JSON."""
        data = super().to_dict()
        data["_concept"] = True
        # Convert set values to sorted lists for deterministic JSON output
        data["memory_refs"] = {
            layer: sorted(ids)
            for layer, ids in self.memory_refs.items()
            if ids  # Skip empty sets
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Concept:
        """Deserialize Concept, converting lists back to sets."""
        # Build base SemanticMemory fields
        prov_str = data.get("provenance", "EPISODIC_CONSOLIDATION")
        try:
            provenance = ConceptProvenance[prov_str]
        except KeyError:
            provenance = ConceptProvenance.EPISODIC_CONSOLIDATION

        # Convert list values back to sets
        raw_refs = data.get("memory_refs", {})
        memory_refs = defaultdict(set)
        for layer, ids in raw_refs.items():
            memory_refs[layer] = set(ids)

        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            created_at=data.get("created_at", data["timestamp"]),
            accessed_at=data.get("accessed_at", data["timestamp"]),
            access_count=data.get("access_count", 1),
            long_term=data.get("long_term", False),
            consolidated_at=data.get("consolidated_at"),
            name=data.get("name", ""),
            definition=data.get("definition", ""),
            category=data.get("category", ""),
            properties=data.get("properties", {}),
            provenance=provenance,
            source_episode_ids=data.get("source_episode_ids", []),
            source_document=data.get("source_document"),
            confidence=data.get("confidence", 0.5),
            reinforcement_count=data.get("reinforcement_count", 1),
            embedding_text=data.get("embedding_text", ""),
            memory_refs=memory_refs,
        )
```

**ATL `load()` dispatch:** ATL's `load()` must check for both `_compressed`
and `_concept` flags to deserialize the correct type:

```python
# In ATL.load(), replace the current concept deserialization block:
for cid, c_data in data.get("concepts", {}).items():
    if c_data.get("_compressed"):
        concept = CompressedSemantic.from_dict(c_data)
        self._compressed_count += 1
    elif c_data.get("_concept"):
        concept = Concept.from_dict(c_data)
    else:
        concept = SemanticMemory.from_dict(c_data)
    self._concepts[cid] = concept
    self._index_concept(cid, concept)
```

**Backward compatibility:** Existing persisted `SemanticMemory` records
(no `_concept` flag) deserialize as plain `SemanticMemory` — they continue
to work. When a `SemanticMemory` is first reinforced by ConceptExtractor,
it gets promoted to a `Concept` with empty `memory_refs` that populate
over time. The migration is organic — no bulk conversion needed.

**Sorted output:** `sorted(ids)` produces deterministic JSON, so
`git diff` on persistence files shows meaningful changes, not set-ordering
noise. This matters for debugging persistence issues.

### Concept categories

Expand the existing category system to cover percept-derived concepts:

| Category | Source | Example |
|----------|--------|---------|
| `object` | Percept (detected_objects) | "chair", "mug", "table" |
| `person` | Percept (detected_people) | "Dennis", "unknown_person_1" |
| `location` | Percept + spatial context | "kitchen", "charging_station" |
| `action` | Action (tool_name) | "navigate", "grasp", "speak" |
| `goal` | Decision (active_goal) | "navigate_to_kitchen", "find_charger" |
| `causal_pattern` | NAc promotion (existing) | "grasp -> success" |
| `operational_pattern` | StatisticianAgent (existing) | "morning_activity_peak" |
| `fact` | RAG / direct ingestion (future) | "earth orbits sun" |

### Capacity management

ATL's config already has `max_concepts: int = 10_000` but it's never
enforced. ConceptExtractor creates concepts on every capture — without
a cap, long-running sessions accumulate concepts unboundedly. This
mirrors Hippocampus's `max_nodes` (also unenforced), but ATL is more
susceptible because every detected object/person in every frame produces
a `find_or_create()` call.

**Enforcement strategy:** Check capacity in `ATL.store()`. When at cap,
evict the lowest-scored concept before inserting the new one. This is the
same pattern as a bounded cache with LRU-like eviction.

#### SCN-aware eviction scoring

Raw `time.time() - accessed_at` treats all quiet periods equally. But a
concept like "kitchen" might be heavily accessed during morning routines
and quiet at night — that's not decay, that's rhythm. SCN already knows
these patterns via its temporal bins (`BoundedBin` with significance-based
eviction). Leveraging SCN for concept eviction adds temporal intelligence:
concepts that are rhythmically relevant survive, even if they haven't been
accessed recently.

**How it works:** ConceptExtractor registers each concept in SCN with the
episode's temporal signature (same call that registers the episode itself).
SCN accumulates temporal bin entries for the concept. At eviction time,
ATL queries SCN for the concept's temporal pattern strength — how many
time bins the concept appears in, and whether it has rhythmic patterns.

```python
# In ConceptExtractor._register_concept(), after ATL registration:
if self._scn:
    sig = TemporalSignature.from_timestamp(record.timestamp)
    self._scn.register(concept_id, sig, significance=concept.confidence)
```

The store-time eviction uses the same `MemoryStrategy` pipeline as
sleep-time consolidation (see "Unified pruning model" below). This
ensures consistent scoring — a concept that survives eviction also
survives consolidation, and vice versa.

```python
# In ATL.store():
def store(self, record: MemoryRecord, **kwargs: Any) -> str:
    with self._rwlock.write():
        # Capacity check — evict before insert
        if len(self._concepts) >= self.config.max_concepts:
            self._evict_one()
        # ... existing store logic ...

def _evict_one(self) -> None:
    """Evict the lowest-scored concept to make room.

    Reuses _get_memory_strategy() for consistent scoring with
    consolidate(). TemporalAwareStrategy wraps SCN when connected,
    using public SCN APIs (get_bins, is_sole_representative,
    is_rhythmic_bin) — no private internals access.
    """
    import heapq

    strategy = self._get_memory_strategy()
    if hasattr(strategy, 'prepare'):
        strategy.prepare()

    now = time.time()
    scored: list[tuple[float, str]] = []

    for cid, concept in self._concepts.items():
        edge_count = len(self._graph._outgoing.get(cid, []))
        degree = edge_count
        if isinstance(concept, Concept):
            degree += concept.ref_count()
        score = strategy.score_for_retention(concept, now, degree)
        heapq.heappush(scored, (score, cid))

    if scored:
        _, evict_id = heapq.heappop(scored)
        self._remove_concept(evict_id)
        if self._scn:
            self._scn.unregister(evict_id)
        logger.debug(
            "ATL capacity eviction: removed %s (at %d/%d)",
            evict_id, len(self._concepts), self.config.max_concepts,
        )
```

**SCN registration is additive:** ConceptExtractor already runs during
capture (when the episode's timestamp is available). Adding a
`self._scn.register(concept_id, sig)` call is one line. SCN's
`unregister()` is called on eviction and concept deletion for cleanup.

**Interaction with consolidation:** Both `_evict_one()` and
`consolidate()` use `_get_memory_strategy()` — same scorer, same SCN
wrapping, same degree calculation (edge_count + ref_count). Eviction
is the store-time overflow valve; consolidation is the sleep-time deep
clean. See "Unified pruning model" for the full picture.

---

### Unified pruning model: store-time eviction + sleep-time retention

Both ATL and Hippocampus need two complementary pruning mechanisms:

1. **Store-time eviction** — when at capacity (`max_concepts` / `max_nodes`),
   evict the lowest-scored record before inserting the new one. This is the
   overflow valve that prevents unbounded growth between sleep cycles.

2. **Sleep-time retention** — during `consolidate()` / `sleep()`, score all
   records and compress/remove low-value ones in bulk. This is the scheduled
   cleanup that reshapes the memory landscape.

Both mechanisms should use the same scoring signals for consistency. A record
that survives store-time eviction should also survive sleep-time retention
(and vice versa). Using different scoring criteria would create confusing
behavior where a memory survives capture but gets immediately pruned at
sleep, or vice versa.

#### Current state

| Layer | Store-time eviction | Sleep-time retention |
|-------|-------------------|---------------------|
| **Hippocampus** | `max_nodes` unenforced (warning only) | Fully implemented via `MemoryStrategy` + `TemporalAwareStrategy` |
| **ATL** | Planned (`_evict_one` above) | Implemented but simplistic: age + access_count + confidence thresholds, no `MemoryStrategy` |

#### Target state

Both layers get both mechanisms, using consistent scoring:

**Hippocampus store-time eviction** — enforce `max_nodes` in `capture()`:

```python
# In Hippocampus.capture(), inside the write lock, before storing:
if len(self._memories) >= self.config.max_nodes:
    self._evict_one()

def _evict_one(self) -> None:
    """Evict the lowest-scored memory to make room.

    Reuses the existing MemoryStrategy (including TemporalAwareStrategy
    when SCN is connected) for consistent scoring between store-time
    eviction and sleep consolidation. Never evicts long-term memories
    at store time — those are only pruned during sleep with the
    long_term_retention_boost applied.
    """
    import heapq

    strategy = self._get_memory_strategy()
    if hasattr(strategy, 'prepare'):
        strategy.prepare()  # Pre-compute SCN bin populations

    now = time.time()
    scored: list[tuple[float, str]] = []

    for memory_id, record in self._memories.items():
        if record.long_term:
            continue  # Never evict long-term memories at store time
        edge_count = (
            record.edge_count if isinstance(record, CompressedMemory)
            else len(self._graph.get_associated(memory_id))
        )
        score = strategy.score_for_retention(record, now, edge_count)
        heapq.heappush(scored, (score, memory_id))

    if scored:
        _, evict_id = heapq.heappop(scored)
        self._remove_memory(evict_id)
        logger.debug("Evicted memory %s at capacity", evict_id[:8])
```

Key design: reuses `_get_memory_strategy()` which already auto-wraps with
`TemporalAwareStrategy` when SCN is connected. No new scoring logic needed.
Long-term memories are protected at store time — only sleep can remove them
(with the `long_term_retention_boost` multiplier applied).

**ATL sleep-time retention** — upgrade `consolidate()` to use `MemoryStrategy`:

The current ATL `consolidate()` uses hardcoded threshold checks
(`time_since_access > max_age`, `access_count < 3`, `confidence < 0.7`).
This should be upgraded to use `MemoryStrategy`, matching Hippocampus's
pattern. ATL concepts have different decay characteristics (slower —
30-day `max_age_without_access` vs Hippocampus's 7-day), so ATL gets
its own strategy configuration, but the same `MemoryStrategy` ABC.

```python
# In ATL.consolidate():
def consolidate(self, **kwargs: Any) -> dict[str, int]:
    """Consolidation cycle: score concepts via MemoryStrategy, compress/remove."""
    strategy = self._get_memory_strategy()
    if hasattr(strategy, 'prepare'):
        strategy.prepare()

    now = time.time()
    removed = 0
    compressed = 0

    with self._rwlock.write():
        to_remove: list[str] = []
        to_compress: list[str] = []

        for cid, concept in self._concepts.items():
            if isinstance(concept, CompressedSemantic):
                score = strategy.score_for_retention(concept, now, concept.edge_count)
                if score < self.config.retention_threshold:
                    to_remove.append(cid)
                continue

            edge_count = len(self._graph._outgoing.get(cid, []))

            # For Concept instances, include ref_count as bonus degree
            degree = edge_count
            if isinstance(concept, Concept):
                degree += concept.ref_count()

            score = strategy.score_for_retention(concept, now, degree)

            if score < self.config.retention_threshold:
                to_remove.append(cid)
            elif score < self.config.compression_threshold:
                if strategy.should_compress(concept, now, degree):
                    to_compress.append(cid)

        # Remove
        for cid in to_remove:
            mem = self._concepts.get(cid)
            if isinstance(mem, CompressedSemantic):
                self._compressed_count -= 1
            self._remove_concept(cid)
            if self._scn:
                self._scn.unregister(cid)
            removed += 1

        # Compress
        for cid in to_compress:
            concept = self._concepts.get(cid)
            if isinstance(concept, SemanticMemory):
                edge_count = len(self._graph._outgoing.get(cid, []))
                comp = CompressedSemantic.from_semantic(concept, edge_count)
                self._concepts[cid] = comp
                self._compressed_count += 1
                compressed += 1

    return {"removed": removed, "compressed": compressed}

def _get_memory_strategy(self) -> MemoryStrategy:
    """Get the configured memory strategy, wrapped with SCN if available."""
    from maxim.memory.strategies import (
        AccessBasedStrategy,
        TemporalAwareStrategy,
    )

    base = AccessBasedStrategy(
        max_age_without_access=self.config.max_age_without_access,
        compression_age=7 * 86400,  # Compress after 1 week (concepts are slower)
    )

    if self._scn:
        return TemporalAwareStrategy(self._scn, base_strategy=base)
    return base
```

This gives ATL the same `MemoryStrategy` + `TemporalAwareStrategy` pipeline
as Hippocampus. The key difference: ATL uses a 30-day access window (vs 7-day)
and feeds `ref_count()` as bonus degree — concepts with many cross-layer refs
score higher. Both `_evict_one()` and `consolidate()` use `_get_memory_strategy()`
— same scorer, same SCN wrapping, consistent behavior.

#### Why both mechanisms, not just one?

- **Sleep-only** would let memory grow unboundedly between cycles. A burst
  of captures (rapid object detection) could push well past `max_nodes`
  before the next sleep.
- **Eviction-only** would never do bulk cleanup. Old low-value memories
  would persist as long as new captures don't push past the cap.

Sleep is the deep clean. Eviction is the pressure relief valve. Both use
the same scoring philosophy (recency + importance + connectivity + temporal
relevance) for predictable behavior.

---

### Documentation: Memory Layer Formation, Connection, and Lifecycle

As the memory system grows more sophisticated, a documentation guide should
be created (separate from this implementation plan) covering:

**How memory layers form:**
- Each layer implements the `MemoryLayer` ABC (`layer.py`)
- Layer-specific record types extend `MemoryRecord` (episodic, semantic, math)
- Each layer owns its internal `DependencyGraph` for intra-layer associations
- Layers are instantiated and configured in `MemoryHub._wire_multi_layer()`

**How layers connect:**
- `CrossLayerGraph` provides typed edges between layers (INSTANCE_OF,
  DERIVED_FROM, QUANTIFIES, STATISTICALLY_CONFIRMS)
- `Semantics` wraps intra-layer graphs with typed relationships
- Capture callbacks bridge layers without coupling (ConceptExtractor,
  NAc, SCN, EC)
- Constructor-injected `layers: dict[str, MemoryLayer]` enables iteration
  without hardcoded layer knowledge

**How layers are managed long-term:**
- Store-time eviction: capacity enforcement via scored eviction in `store()`
- Sleep-time retention: bulk scoring, compression, and removal via
  `MemoryStrategy` pipeline
- `TemporalAwareStrategy` wraps SCN for temporal-aware scoring on both paths
- Consolidation pipeline: promotion (short→long-term), compression
  (full→compressed), pruning (removal of low-value records)
- Persistence: each layer saves/loads independently, versioned JSON with
  backward compatibility
- Cross-layer cleanup: deletion callbacks cascade through
  `CrossLayerGraph.remove_record()`, SCN `unregister()`, and
  ConceptExtractor's reverse index

**Target file:** `docs/memory-layer-lifecycle.md` or a dedicated guide page.
This should be written after the ATL concept memory implementation is complete
(Phases A1-A5) so it documents the actual implemented system, not the plan.
The guide pages in `htmls-guides/` could include a section on this as part
of `maxim-memory-systems.html`.

---

## Phase A0: Semantics.update_edge()

The existing `Semantics` class has `define()`, `find()`, and `remove()`,
but no way to modify an existing relationship's weight or confidence
in-place. Both inline reinforcement (Path 1) and AG modulation (Path 2)
require this. `DependencyGraph.add_edge()` always appends — calling
`define()` twice creates duplicate edges.

**Status: IMPLEMENTED.** `DependencyGraph.update_edge()` and
`DependencyGraph.find_edge()` added to `bus.py` as public APIs.
`Semantics.update_edge()` delegates to these — no layer violations.

```python
# DependencyGraph (bus.py) — new public methods:
def update_edge(self, source, target, edge_type, weight=None, metadata_updates=None) -> bool
def find_edge(self, source, target, edge_type=None, metadata_match=None) -> Edge | None

# Semantics (semantics.py) — delegates to DependencyGraph:
def update_edge(self, source_id, target_id, rel_type, weight=None,
                confidence=None, confidence_delta=None) -> bool
```

This is a prerequisite for both ConceptExtractor (inline reinforcement)
and ConceptGrounder (modulation).

---

## Two-Path Relationship Model

Concept relationships form through two complementary paths. Path 1 creates
the graph structure immediately. Path 2 grounds it with numerical evidence.

### Path 1: Categorical relationships (during extraction, inline)

When ConceptExtractor processes a captured episode, it creates tentative
relationships between concepts that co-occur in that episode. These are
immediate, lightweight, and category-aware.

```python
class ConceptExtractor:
    """Extracts concepts from episodic memories and registers them in ATL.

    Registered as a capture callback on Hippocampus. Fires after the write
    lock releases, so concept registration is non-blocking.

    Also forms inline categorical relationships between concepts found in
    the same episode — no batch sleep job needed.
    """

    def __init__(self, atl: ATL, cross_layer: CrossLayerGraph, scn: SCN | None = None):
        self._atl = atl
        self._cross_layer = cross_layer
        self._scn = scn  # Optional: temporal registration for SCN-aware eviction
        # Reverse index for O(1) cleanup on memory deletion
        self._reverse_index: dict[str, set[str]] = defaultdict(set)

    def on_memory_captured(self, memory_id: str, record: MemoryRecord) -> None:
        """Extract concepts from a newly captured episodic memory."""
        if not isinstance(record, EpisodicMemory):
            return

        concepts_found: list[tuple[str, str]] = []  # (name, category)

        # Objects
        for obj in record.perception.detected_objects:
            concepts_found.append((obj.lower(), "object"))

        # People
        for person in record.perception.detected_people:
            concepts_found.append((person, "person"))

        # Location (extracted from observations if available)
        # Location names come from spatial context, mode context, or
        # explicit location fields in observations. This is sensor-
        # dependent — not all episodes have location data.
        location = record.perception.observations.get("location")
        if not location:
            location = record.perception.observations.get("room")
        if isinstance(location, str) and location:
            concepts_found.append((location.lower(), "location"))

        # Goal: tokenize into individual words so "navigate_to_kitchen"
        # becomes concepts "navigate" (action) and "kitchen" (goal_token),
        # not a single concept named "navigate_to_kitchen". This aligns
        # with how PatternCompleter and ConceptContextBuilder search goals.
        if record.context.active_goal:
            for token in normalize_tokens(record.context.active_goal):
                concepts_found.append((token, "goal"))

        # Action/tool
        if record.action.tool_name:
            concepts_found.append((record.action.tool_name, "action"))

        # Register each concept and collect IDs
        concept_ids: list[tuple[str, str, str]] = []  # (id, name, category)
        for name, category in concepts_found:
            cid = self._register_concept(name, category, memory_id, record)
            if cid:
                concept_ids.append((cid, name, category))
                self._reverse_index[memory_id].add(cid)

        # Form inline relationships between co-occurring concepts
        self._form_inline_relationships(concept_ids)

    def _register_concept(
        self, name: str, category: str, memory_id: str, record: EpisodicMemory
    ) -> str | None:
        """Find or create a concept and link it to the source memory."""
        concept_id, was_created = self._atl.find_or_create(
            name=name,
            category=category,
            definition=f"{category}: {name}",
            provenance=ConceptProvenance.EPISODIC_CONSOLIDATION,
            source_episode_id=memory_id,
        )

        concept = self._atl.get(concept_id)
        if concept and isinstance(concept, Concept):
            concept.add_ref("hippocampus", memory_id)
        elif concept and not was_created:
            concept.reinforce(memory_id)

        # Cross-layer edge: episode INSTANCE_OF concept
        self._cross_layer.add_edge(
            source_layer="hippocampus",
            source_id=memory_id,
            target_layer="atl",
            target_id=concept_id,
            edge_type=CrossLayerEdgeType.INSTANCE_OF,
            weight=1.0,
        )

        # Register concept in SCN for temporal rhythm tracking
        if self._scn:
            sig = TemporalSignature.from_timestamp(record.timestamp)
            self._scn.register(concept_id, sig, significance=concept.confidence if concept else 0.5)

        return concept_id

    # Max relationships formed per episode to prevent noise
    MAX_RELATIONSHIPS_PER_EPISODE = 6

    def _form_inline_relationships(
        self, concept_ids: list[tuple[str, str, str]]
    ) -> None:
        """Form tentative relationships between concepts in the same episode.

        Only relates concepts where at least one is an "active" concept —
        the goal being pursued, the action being performed, or the object
        being interacted with. Background objects that happen to share a
        frame don't auto-relate to each other (that's noise, not signal).

        Uses category pairing to infer relationship type. New relationships
        start at low confidence (0.3) — AG will strengthen or weaken them
        during recall based on numerical evidence.
        """
        # Active categories: concepts the agent is engaging with
        active_categories = {"goal", "action"}
        active_ids = {cid for cid, _, cat in concept_ids if cat in active_categories}

        formed = 0
        for i, (cid_a, name_a, cat_a) in enumerate(concept_ids):
            if formed >= self.MAX_RELATIONSHIPS_PER_EPISODE:
                break
            for cid_b, name_b, cat_b in concept_ids[i + 1:]:
                if cid_a == cid_b:
                    continue
                if formed >= self.MAX_RELATIONSHIPS_PER_EPISODE:
                    break

                # At least one concept must be active (goal/action),
                # OR both must be non-object categories.
                # This prevents "mug RELATED_TO chair" just because
                # both were in the camera frame.
                if not (
                    cid_a in active_ids
                    or cid_b in active_ids
                    or (cat_a != "object" and cat_b != "object")
                ):
                    continue

                rel_type = self._infer_relationship_type(cat_a, cat_b)

                # Check if relationship already exists
                existing = self._atl.find_by_relationship(
                    cid_a, rel_type=rel_type, direction="outgoing", limit=20
                )
                already_linked = any(oid == cid_b for oid, _ in existing)

                if already_linked:
                    # Reinforce existing relationship — bump confidence
                    self._atl.semantics.update_edge(
                        cid_a, cid_b, rel_type, confidence_delta=0.05,
                    )
                else:
                    # New tentative relationship — low confidence, AG will validate
                    self._atl.define_relationship(
                        cid_a, cid_b, rel_type,
                        weight=0.3,
                        confidence=0.3,
                    )
                formed += 1

    def _infer_relationship_type(self, cat_a: str, cat_b: str) -> str:
        """Infer relationship type from concept category pairing."""
        cats = frozenset({cat_a, cat_b})

        if cats == {"person", "location"}:
            return "RELATED_TO"
        elif cats == {"object", "location"}:
            return "RELATED_TO"
        elif cats == {"person", "object"}:
            return "RELATED_TO"
        elif cats == {"action", "object"}:
            return "RELATED_TO"
        elif cats == {"goal", "action"}:
            return "HAS_PART"
        elif cat_a == cat_b == "object":
            return "RELATED_TO"
        else:
            return "ASSOCIATES"

    def on_memory_deleted(self, memory_id: str) -> None:
        """Clean up concept references when an episode is deleted.

        Uses reverse index for O(1) lookup instead of scanning all concepts.
        """
        self._remove_refs_for(memory_id)

    def on_memory_compressed(self, memory_id: str) -> None:
        """Clean up concept references when an episode is compressed.

        CompressedMemory drops the fields ConceptGrounder and
        PatternCompleter need (action timing, decision confidence,
        detected_objects list, etc.). Keeping stale refs would cause
        those systems to load CompressedMemory records and silently
        skip all numerical extraction. Removing the ref is cleaner —
        the concept retains its confidence and reinforcement count
        (those were already accumulated), it just loses the ref to
        an episode that can no longer contribute stats.

        Registered as a compression callback on Hippocampus alongside
        the deletion callback.
        """
        self._remove_refs_for(memory_id)

    def _remove_refs_for(self, memory_id: str) -> None:
        """Remove all concept refs for a memory ID."""
        concept_ids = self._reverse_index.pop(memory_id, set())
        for cid in concept_ids:
            concept = self._atl.get(cid)
            if concept and isinstance(concept, Concept):
                concept.remove_ref("hippocampus", memory_id)

    def rebuild_reverse_index(self) -> None:
        """Rebuild reverse index from ATL concept memory_refs.

        Called on startup to restore the reverse index (which is not
        persisted). O(concepts × avg_refs) but only runs once on boot.
        At max 10K concepts × ~50 refs, this is ~500K iterations —
        milliseconds in Python. Timing is logged so degradation is
        visible if concept count grows significantly.
        """
        import time as _time
        start = _time.monotonic()
        self._reverse_index.clear()
        ref_count = 0
        for concept in self._atl:
            if isinstance(concept, Concept):
                for mem_id in concept.memory_refs.get("hippocampus", set()):
                    self._reverse_index[mem_id].add(concept.id)
                    ref_count += 1
        elapsed_ms = (_time.monotonic() - start) * 1000
        logger.info(
            "ConceptExtractor reverse index rebuilt: %d refs across %d concepts in %.1fms",
            ref_count, len(self._reverse_index), elapsed_ms,
        )
```

**Note on reverse index persistence:** The reverse index is NOT persisted.
It's rebuilt on startup from ATL's persisted `memory_refs` via
`rebuild_reverse_index()`, called during `MemoryHub.on_session_start()`.
This is O(concepts) but only runs once. Persisting the reverse index
separately would create a sync risk — if ATL's `memory_refs` and the
reverse index disagree after a crash, you get ghost references.

**Compression callback:** `on_memory_compressed()` is registered as a
Hippocampus compression callback alongside the existing deletion callback.
Hippocampus calls compression callbacks from `_compress_memory()` after
replacing the full EpisodicMemory with CompressedMemory. This requires
adding a `_on_memory_compressed` callback list to Hippocampus (same
pattern as `_on_memory_deleted`). The wiring happens in
`MemoryHub._wire_multi_layer()`:
```python
hippocampus.register_compression_callback(concept_extractor.on_memory_compressed)
```

**Note on relationship type inference:** The initial implementation uses
simple category-based heuristics. `RELATED_TO` and `ASSOCIATES` are safe
defaults. Future work can add LLM-assisted relationship typing where the
model suggests richer types (IS_A, HAS_PART, CAUSES) based on episodic
context. The `RelationshipRegistry` is already extensible for agent-
proposed types at runtime.

**Async extraction via queue:** ConceptExtractor runs in a Hippocampus
capture callback. If ATL is locked (e.g., during consolidation),
`find_or_create()` blocks, stalling all subsequent capture callbacks
(NAc, SCN, EC). In high-frequency capture scenarios (rapid object
detection), this is a bottleneck.

Solution: queue-based async extraction, same pattern as Hippocampus's
existing `_capture_queue` / `_capture_worker_thread` infrastructure.
ConceptExtractor's `on_memory_captured()` becomes a lightweight
snapshot-and-enqueue operation (no lock contention), and a background
worker thread drains the queue to process concept registration:

```python
class ConceptExtractor:
    def __init__(self, atl, cross_layer, scn=None):
        # ... existing fields ...
        self._queue: queue.Queue[tuple[str, EpisodicMemory]] = queue.Queue(maxsize=200)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._stop = threading.Event()
        self._worker.start()

    def on_memory_captured(self, memory_id: str, record: MemoryRecord) -> None:
        """Enqueue for background processing — non-blocking callback."""
        if not isinstance(record, EpisodicMemory):
            return
        try:
            self._queue.put_nowait((memory_id, record))
        except queue.Full:
            logger.warning("ConceptExtractor queue full, dropping %s", memory_id[:8])

    def _worker_loop(self) -> None:
        """Background worker: drain queue, register concepts."""
        while not self._stop.is_set():
            try:
                memory_id, record = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_capture(memory_id, record)
            except Exception as e:
                logger.warning("ConceptExtractor failed for %s: %s", memory_id[:8], e)

    def _process_capture(self, memory_id: str, record: EpisodicMemory) -> None:
        """Actual concept extraction — runs in worker thread."""
        # ... existing on_memory_captured logic (concept registration,
        #     inline relationships, cross-layer edges) ...

    def shutdown(self) -> None:
        """Stop the worker thread. Called during MemoryHub shutdown."""
        self._stop.set()
        self._worker.join(timeout=5.0)
```

This decouples concept extraction from the capture callback chain.
ATL lock contention only affects the worker thread, not Hippocampus's
capture path. The queue size (200) matches Hippocampus's capture queue.
If the queue fills (sustained burst faster than extraction), oldest
entries are dropped with a warning — the concept will be created on
the next capture of that percept anyway.

### Path 2: AG numerical grounding (during recall)

When memories linked to a concept are recalled, AG piggybacks on the
loaded episodes to compute statistical properties. These computations
strengthen, weaken, or create relationships with numerical evidence.

**Cost model:** All data is in-memory — Hippocampus stores memories in a
dict, ATL stores concepts in a dict. Episode lookups are O(1) dict access,
not disk I/O. The real cost is IPS/AG computation, which is arithmetic on
small arrays. AG escalation (regression) only triggers for 8+ data points
with uncertain IPS, so it's rare. Cache prevents recomputation on repeated
recalls within the TTL window.

**After memory unification** (see the unified memory system):
MemoryAgent holds `WorkingMemoryEntry[EpisodicMemory]` objects. Concept
grounding can filter the already-loaded working set by concept refs and
only supplement from Hippocampus when coverage is insufficient — true
piggybacking. Before unification, all episodes are loaded from Hippocampus
directly (still trivially fast, just architecturally redundant).

```python
class ConceptGrounder:
    """Grounds ATL concept relationships with AG numerical analysis.

    Piggybacks on concept recall: when episodes linked to a concept are
    loaded, extracts numerical fields and runs IPS/AG analysis. Results
    become QUANTIFIES edges and modulate existing relationship confidence.

    Brain mapping: Angular gyrus provides numerical/spatial cognition
    that grounds the ATL's semantic representations in quantitative
    evidence. IPS handles fast approximate assessment; AG handles
    precise analysis when needed.
    """

    # Jaccard weight scaling factor. Maps Jaccard similarity [0, 1] to
    # edge weight [0, 1]. The initial value 2.0 means Jaccard >= 0.5
    # saturates at weight 1.0. This is a tunable parameter — the optimal
    # value depends on the co-occurrence distribution in actual usage.
    #
    # Calibration approach:
    # 1. Start with 2.0 (a priori reasonable: 50% co-occurrence = max weight)
    # 2. After accumulating data (100+ concepts with 10+ episodes each),
    #    log the Jaccard distribution across all concept pairs
    # 3. Compute the median and 90th percentile Jaccard values
    # 4. Set JACCARD_WEIGHT_SCALE so that the 75th percentile maps to
    #    weight ~0.7 (strong but not saturated). Formula:
    #    JACCARD_WEIGHT_SCALE = 0.7 / percentile_75
    # 5. Alternatively, use a sigmoid or log transform if the distribution
    #    is heavily skewed (most pairs near 0 or 1)
    #
    # The strengthen threshold (0.3) and weaken threshold (0.05) should
    # be calibrated alongside this constant using the same distribution.
    JACCARD_WEIGHT_SCALE: float = 2.0

    def __init__(
        self,
        atl: ATL,
        angular_gyrus: AngularGyrus,
        ips: IPS,
        cross_layer: CrossLayerGraph,
        cache_ttl: float = 300.0,  # 5 min cache
        jaccard_weight_scale: float | None = None,
    ):
        self._atl = atl
        self._ag = angular_gyrus
        self._ips = ips
        self._cross_layer = cross_layer
        self._cache_ttl = cache_ttl
        if jaccard_weight_scale is not None:
            self.JACCARD_WEIGHT_SCALE = jaccard_weight_scale
        # concept_id -> (timestamp, stats_dict)
        self._stats_cache: dict[str, tuple[float, dict]] = {}

    def ground_concept(
        self, concept: Concept, episodes: list[EpisodicMemory]
    ) -> dict[str, Any]:
        """Compute numerical properties for a concept from its linked episodes.

        Called during concept recall when episodes are already loaded.
        Returns stats dict for inclusion in LLM context.

        Uses IPS (fast path) for basic stats. Escalates to AG (slow path)
        for regression/trend analysis when enough data points exist.
        """
        if not episodes:
            return {}

        # Check cache
        cached = self._stats_cache.get(concept.id)
        if cached:
            cache_time, cache_stats = cached
            if time.time() - cache_time < self._cache_ttl:
                # Cache valid unless new refs added since cache time
                if concept.ref_count("hippocampus") <= cache_stats.get("_ref_count", 0):
                    return cache_stats

        # Extract numerical fields from episodes
        numerics = self._extract_numerics(episodes)
        if not numerics:
            return {}

        stats = self._compute_stats(concept, numerics)

        # Cache results
        stats["_ref_count"] = concept.ref_count("hippocampus")
        self._stats_cache[concept.id] = (time.time(), stats)

        # Update relationship confidence based on numerical evidence
        self._modulate_relationships(concept)

        # Store AG MathMemory with QUANTIFIES edge if significant
        self._store_quantifications(concept, stats)

        return stats

    def _extract_numerics(
        self, episodes: list[EpisodicMemory | CompressedMemory]
    ) -> dict[str, list[float]]:
        """Extract numerical fields from loaded episodes.

        Returns field_name -> list of values across episodes.
        Only includes fields present in at least 2 episodes.

        Handles CompressedMemory gracefully: compressed records have
        novelty, salience, and success_rate but lack timing, confidence,
        and fear_level. These are still useful for stats on those fields.
        ConceptExtractor's on_memory_compressed() removes the ref from
        the concept, so compressed records shouldn't appear here in
        practice — but this handles edge cases (race between compression
        and recall, or records compressed before the callback was wired).
        """
        collectors: dict[str, list[float]] = defaultdict(list)

        for ep in episodes:
            # CompressedMemory: extract the limited fields it retains
            if isinstance(ep, CompressedMemory):
                if ep.salience is not None:
                    collectors["salience"].append(ep.salience)
                if ep.novelty is not None:
                    collectors["novelty"].append(ep.novelty)
                if ep.success is not None:
                    collectors["success_rate"].append(1.0 if ep.success else 0.0)
                continue
            # Action timing
            if ep.action.execution_time_ms is not None:
                collectors["execution_time_ms"].append(ep.action.execution_time_ms)

            # Perception scores
            if ep.perception.salience is not None:
                collectors["salience"].append(ep.perception.salience)
            if ep.perception.novelty is not None:
                collectors["novelty"].append(ep.perception.novelty)

            # Decision confidence
            if ep.decision.confidence is not None:
                collectors["decision_confidence"].append(ep.decision.confidence)

            # Context
            if ep.context.fear_level is not None:
                collectors["fear_level"].append(ep.context.fear_level)

            # Outcome
            if ep.outcome.success is not None:
                collectors["success_rate"].append(1.0 if ep.outcome.success else 0.0)

        # Filter to fields with enough data points
        return {k: v for k, v in collectors.items() if len(v) >= 2}

    def _compute_stats(
        self, concept: Concept, numerics: dict[str, list[float]]
    ) -> dict[str, Any]:
        """Compute statistics using IPS (fast) with AG escalation (precise).

        IPS handles: mean (ApproximateResult), trend (TrendResult)
        AG escalation: regression analysis when 8+ data points and IPS uncertain

        Note: IPS.estimate_mean() returns ApproximateResult (access .value).
        IPS.detect_trend() returns TrendResult (access .direction, .confidence).
        """
        stats: dict[str, Any] = {}

        for field, values in numerics.items():
            # IPS fast path: basic stats
            approx = self._ips.estimate_mean(values)
            trend = self._ips.detect_trend(values) if len(values) >= 3 else None

            field_stats = {
                "mean": approx.value,
                "n": len(values),
                "min": min(values),
                "max": max(values),
            }
            if trend and trend.direction.value != "flat":
                field_stats["trend"] = trend.direction.value

            # AG escalation: precise analysis when IPS trend is uncertain
            if (
                len(values) >= 8
                and trend
                and 0.3 < trend.confidence < 0.65
            ):
                analysis = self._ag.analyze(values, method="linear")
                if analysis and analysis.confidence > 0.5:
                    field_stats["r_squared"] = analysis.confidence
                    field_stats["slope"] = analysis.parameters.get("slope")
                    field_stats["ag_analysis"] = True

            stats[field] = field_stats

        return stats

    def _modulate_relationships(self, concept: Concept) -> None:
        """Strengthen or weaken concept relationships based on Jaccard
        co-occurrence similarity.

        Uses Jaccard index (|A ∩ B| / |A ∪ B|) instead of asymmetric
        overlap (|A ∩ B| / |A|). Jaccard is symmetric — it doesn't
        matter which concept you compute from — and naturally handles
        size imbalance: if concept A has 100 episodes and B has 3, and
        all 3 overlap, Jaccard = 3/100 (low, correctly reflecting that
        A's experience is mostly without B). Asymmetric overlap would
        give 3/100 from A's side but 3/3 from B's side — inconsistent.
        """
        relationships = self._atl.find_by_relationship(
            concept.id, direction="both", limit=50
        )
        if not relationships:
            return

        concept_refs = concept.memory_refs.get("hippocampus", set())

        for other_id, rel in relationships:
            other = self._atl.get(other_id)
            if not isinstance(other, Concept):
                continue

            other_refs = other.memory_refs.get("hippocampus", set())
            shared = len(concept_refs & other_refs)
            union = len(concept_refs | other_refs)

            if union < 3:
                continue  # Not enough evidence to modulate

            jaccard = shared / union

            # Strengthen: high co-occurrence with enough shared evidence
            if jaccard > 0.3 and shared >= 3:
                self._atl.semantics.update_edge(
                    concept.id, other_id, rel.relationship_type,
                    weight=min(1.0, jaccard * self.JACCARD_WEIGHT_SCALE),
                    confidence_delta=0.05,
                )
            # Weaken: low co-occurrence despite many total observations
            elif jaccard < 0.05 and union >= 10:
                self._atl.semantics.update_edge(
                    concept.id, other_id, rel.relationship_type,
                    confidence_delta=-0.1,
                )

    def _store_quantifications(
        self, concept: Concept, stats: dict[str, Any]
    ) -> None:
        """Store significant numerical properties as AG MathMemory records
        linked to the concept via QUANTIFIES edges.

        Only creates/updates records for fields with enough data (n >= 5)
        and meaningful variation.
        """
        for field, field_stats in stats.items():
            if field.startswith("_"):
                continue
            n = field_stats.get("n", 0)
            if n < 5:
                continue

            # Check if AG already has a record for this concept+field
            existing_name = f"{concept.name}:{field}"
            existing = self._ag.recall(limit=1, name=existing_name)

            if existing:
                # Update existing record via AG's public API — ensures
                # indexes, stats, and persistence state stay consistent.
                # Direct field mutation would bypass AG's internal tracking.
                record = existing[0]
                self._ag.update_record(record.id, {
                    "properties": {**record.properties, **field_stats},
                    "observation_count": n,
                    "confidence": min(0.9, 0.3 + 0.05 * n),
                })
            else:
                # Create new MathMemory
                record = MathMemory(
                    name=existing_name,
                    category=MathCategory.PATTERN,
                    domain="concept_property",
                    verbal=f"{concept.name} {field}: mean={field_stats['mean']:.2f} (n={n})",
                    code="",
                    inputs=[concept.name],
                    outputs=[field],
                    source="derived",
                    confidence=min(0.9, 0.3 + 0.05 * n),
                    observation_count=n,
                    properties=field_stats,
                )
                record_id = self._ag.store(record)

                # QUANTIFIES edge: AG record -> ATL concept
                # Only created for NEW records — existing records already
                # have their edge from the first time they were stored.
                self._cross_layer.add_edge(
                    source_layer="angular_gyrus",
                    source_id=record_id,
                    target_layer="atl",
                    target_id=concept.id,
                    edge_type=CrossLayerEdgeType.QUANTIFIES,
                    weight=1.0,
                    metadata={"field": field},
                )

                # Track ref in concept
                concept.add_ref("angular_gyrus", record_id)
```

### How the two paths interact

```
Episode captured: "saw mug in kitchen, grasped it, 340ms, success"
  |
  +-- Path 1 (ConceptExtractor, immediate):
  |   +-- find_or_create("mug", "object")
  |   +-- find_or_create("kitchen", "location")
  |   +-- find_or_create("grasp", "action")
  |   +-- tentative edges (confidence 0.3):
  |   |   mug RELATED_TO kitchen
  |   |   mug RELATED_TO grasp (or reinforce if exists)
  |   |   grasp RELATED_TO kitchen
  |   +-- INSTANCE_OF edges to episode
  |
  +-- Later, on recall of "mug":
      +-- Load linked episodes (already happening for recall)
      +-- Path 2 (ConceptGrounder, piggybacking):
          +-- IPS: mean grasp time 310ms, success rate 83%
          +-- AG (if 8+ points, IPS uncertain): r^2=0.78 distance->time
          +-- Modulate: mug<->kitchen Jaccard 8/12 = 0.67 -> confidence 0.85
          +-- Modulate: mug<->lamp Jaccard 2/50 = 0.04 -> confidence decays
          +-- Store: MathMemory "mug:execution_time_ms" QUANTIFIES mug
          +-- Context to LLM includes stats + relationships
```

---

## Concept-Aware Recall

When MemoryAgent builds context for the LLM, it looks up concepts matching
the current percept and iterates over registered memory layers to collect
context for each concept. The `ConceptContextBuilder` is a standalone
collaborator — same pattern as ConceptExtractor, ConceptGrounder, and
PatternCompleter. It lives in the integration layer
(`src/maxim/memory/concept_context.py`).

The builder receives registered layers via constructor injection. For each
matching concept, it iterates all layers (except hippocampus, which provides
source episodes) to collect enrichment context — currently AG stats, but
extensible to future layer types (social, emotional, spatial). This is the
"iterator over the memory layer registry" pattern: concepts come from ATL,
context comes from iterating layers.

```python
class ConceptContextBuilder:
    """Builds LLM context entries from ATL concepts and registered layers.

    Standalone collaborator in the integration layer. Iterates registered
    memory layers to collect enrichment context for each concept matching
    the current percept. Same pattern as ConceptExtractor and ConceptGrounder.

    Lives in: src/maxim/memory/concept_context.py
    Wired in: MemoryHub._wire_multi_layer()
    Called by: MemoryAgent._build_context() via self._concept_context_builder
    """

    # Default time budget for concept grounding on the recall hot path.
    GROUNDING_BUDGET_MS: float = 50.0

    def __init__(
        self,
        atl: ATL,
        layers: dict[str, MemoryLayer],
        concept_grounder: ConceptGrounder | None = None,
    ):
        self._atl = atl
        self._layers = layers  # "hippocampus" -> Hippocampus, "angular_gyrus" -> AG, etc.
        self._concept_grounder = concept_grounder

    def build(
        self,
        percept: Percept,
        limit: int = 5,
        budget_ms: float | None = None,
    ) -> list[dict]:
        """Build concept context entries for the current percept.

        Synchronous — all callers of build_context() are sync (including
        ExecAgent._worker_loop which runs in a background thread). AG
        grounding is arithmetic on in-memory arrays, not blocking I/O.

        For each matching concept:
        1. Look up concept in ATL by percept names and goal tokens
        2. Load linked episodes from hippocampus layer
        3. Run ConceptGrounder (synchronous, budget-bounded) for AG stats
        4. Iterate all non-hippocampus layers for additional enrichment
        5. Collect relationships (always — cheap graph lookup)

        If the budget is exhausted, remaining concepts get relationship
        context only (no numerical stats). Graceful degradation.

        Returns list of context dicts for StructuredContext.concept_context.
        """
        if budget_ms is None:
            budget_ms = self.GROUNDING_BUDGET_MS

        concepts = self._find_matching_concepts(percept)
        if not concepts:
            return []

        # Rank by confidence, take top `limit`
        concepts.sort(key=lambda c: c.confidence, reverse=True)
        concepts = concepts[:limit]

        context_entries = []
        budget_start = _time.monotonic()
        budget_exhausted = False

        hippocampus = self._layers.get("hippocampus")

        for concept in concepts:
            # Load linked episodes (for AG grounding)
            episodes = []
            if hippocampus:
                episode_ids = list(concept.memory_refs.get("hippocampus", set()))
                episodes = hippocampus.recall_by_ids(episode_ids[:20])

            # AG grounding — synchronous with time budget
            stats = {}
            if self._concept_grounder and episodes and not budget_exhausted:
                elapsed_ms = (_time.monotonic() - budget_start) * 1000
                if elapsed_ms < budget_ms:
                    stats = self._concept_grounder.ground_concept(
                        concept, episodes
                    )
                else:
                    budget_exhausted = True
                    logger.debug(
                        "Concept grounding budget exhausted (%.1fms), "
                        "skipping AG stats for remaining %d concepts",
                        elapsed_ms,
                        len(concepts) - len(context_entries),
                    )

            # Iterate layers for additional enrichment context
            layer_enrichment = self._collect_layer_enrichment(concept)

            # Get relationships (always — cheap O(edges) lookup)
            rel_summaries = self._collect_relationships(concept)

            # Format AG quantifications
            ag_props = {}
            for field, field_stats in stats.items():
                if field.startswith("_"):
                    continue
                ag_props[field] = {
                    "mean": field_stats.get("mean"),
                    "n": field_stats.get("n"),
                    "trend": field_stats.get("trend"),
                }

            entry = {
                "name": concept.name,
                "category": concept.category,
                "confidence": concept.confidence,
                "episode_count": concept.ref_count("hippocampus"),
                "relationships": rel_summaries,
                "properties": ag_props,
            }
            if layer_enrichment:
                entry["layer_context"] = layer_enrichment

            context_entries.append(entry)

        return context_entries

    def _find_matching_concepts(self, percept: Percept) -> list[Concept]:
        """Find ATL concepts matching the percept's objects, people, and goal.

        Objects and people are single-word concept names, so direct lookup
        works. Goals are free-form phrases, so we tokenize via
        normalize_tokens() — same approach as PatternCompleter.
        """
        matches = []
        seen: set[str] = set()
        search_terms: list[str] = list(
            percept.detected_objects + percept.detected_people
        )

        # Goal: tokenize, filter stop words, lemmatize
        if hasattr(percept, 'active_goal') and percept.active_goal:
            search_terms.extend(normalize_tokens(percept.active_goal))

        for term in search_terms:
            results = self._atl.recall(limit=1, name=term.lower())
            for concept in results:
                if isinstance(concept, Concept) and concept.id not in seen:
                    matches.append(concept)
                    seen.add(concept.id)

        return matches

    def _collect_layer_enrichment(self, concept: Concept) -> list[dict]:
        """Iterate registered layers for enrichment context.

        Skips hippocampus (provides episodes, not enrichment). For each
        layer, loads records referenced by the concept's memory_refs and
        formats them. Currently only AG produces MathContextEntry, but
        future layers (social, emotional) can contribute their own entries.
        """
        entries = []
        for layer_name, layer in self._layers.items():
            if layer_name == "hippocampus":
                continue
            ref_ids = concept.memory_refs.get(layer_name, set())
            if not ref_ids:
                continue
            records = layer.recall_by_ids(list(ref_ids)[:5])
            for record in records:
                if isinstance(record, MathMemory):
                    entries.append({
                        "layer": layer_name,
                        "name": record.name,
                        "verbal": record.verbal,
                        "confidence": record.confidence,
                    })
        return entries

    def _collect_relationships(self, concept: Concept) -> list[dict]:
        """Collect relationship summaries for a concept."""
        relationships = self._atl.find_by_relationship(
            concept.id, direction="both", limit=10
        )
        summaries = []
        for other_id, rel in relationships:
            other = self._atl.get(other_id)
            if other:
                summaries.append({
                    "type": rel.relationship_type,
                    "target": other.name,
                    "confidence": rel.confidence,
                })
        return summaries
```

**Why a standalone class, not a method on MemoryAgent?** Same reasoning as
PatternCompleter and ConceptGrounder — it touches ATL, Hippocampus, AG, and
potentially future layers. Putting it on MemoryAgent would make MemoryAgent
responsible for layer iteration, concept lookup, AG grounding delegation,
and relationship formatting. That's integration logic, not agent logic.

**Why iterate layers via constructor-injected dict?** This is the same
pattern as PatternCompleter's `layers` dict. Adding a new memory type
(social, emotional) means adding it to the dict at wiring time — no code
changes in ConceptContextBuilder. The builder doesn't know what layers
exist; it just iterates what it's given.

**Why a time budget?** Without a budget, grounding all matched concepts
could take too long in aggregate. The budget ensures a hard ceiling:
concepts are ranked by confidence (most important first), so if the
budget runs out, the least-confident concepts lose AG stats but still
get relationship context.

**Async boundary resolution:** `MemoryAgent.build_context()` is currently
synchronous. `ConceptContextBuilder.build()` is async because AG grounding
runs via `asyncio.to_thread()`. Caller audit reveals **all 6 call sites
are synchronous**:

| File | Caller | Async? | Context |
|------|--------|--------|---------|
| `maxim_agent.py:432` | `get_mode()` | sync | Debug/mode query |
| `maxim_agent.py:437` | `get_context_summary()` | sync | Debug/summary |
| `exec_agent.py:957` | `_complete_memory_recording()` | sync | Staging eval |
| `exec_agent.py:1466` | `_worker_loop()` | sync | **Background thread** |
| `exec_agent.py:1492` | `propose_intent()` | sync | Goal proposal |
| `agent_loop.py:2024` | `run_agent_loop()` | sync | Main loop |

**Critical finding:** `_worker_loop()` runs in a background thread
(`threading.Thread`) — it cannot be made async without restructuring
ExecAgent's worker architecture. Making `build_context()` async would
break this caller.

**Recommendation: keep `build_context()` synchronous.** Instead, make
`ConceptContextBuilder` provide a synchronous `build()` that runs AG
grounding in-thread (it's arithmetic on small arrays — fast enough
for synchronous execution). The budget mechanism still applies:

```python
def build(self, percept: Percept, limit: int = 5,
          budget_ms: float | None = None) -> list[dict]:
    """Build concept context — synchronous, budget-bounded.

    AG grounding is arithmetic on in-memory arrays (IPS mean/trend,
    optional regression). At typical episode counts (5-50 per concept),
    this completes in <5ms per concept. The budget bounds total work
    across all concepts, not individual computations.
    """
    # ... same logic as before, but synchronous (no await/to_thread)
```

If AG grounding becomes expensive in the future (e.g., large datasets,
complex regressions), revisit with an async variant — but the current
IPS/AG computation is microseconds-to-milliseconds, not blocking I/O.

**StructuredContext addition:**
```python
@dataclass
class StructuredContext:
    # ... existing fields ...
    concept_context: list[dict] = field(default_factory=list)
    # e.g., [{"name": "mug", "category": "object", "confidence": 0.85,
    #         "episode_count": 12,
    #         "relationships": [
    #             {"type": "RELATED_TO", "target": "kitchen", "confidence": 0.85},
    #             {"type": "RELATED_TO", "target": "Dennis", "confidence": 0.72}
    #         ],
    #         "properties": {
    #             "execution_time_ms": {"mean": 310, "n": 12, "trend": null},
    #             "success_rate": {"mean": 0.83, "n": 12, "trend": null}
    #         }}]
```

---

## Integration with Existing Systems

### File locations

New collaborator classes follow the existing convention: cross-layer
collaborators live in `src/maxim/memory/`, not `src/maxim/integration/`.
(`MemoryHub` is the exception — it's the wiring entrypoint in
`src/maxim/integration/memory_hub.py`.)

| Class | File | Pattern follows |
|-------|------|-----------------|
| `Concept` | `src/maxim/memory/semantic_types.py` | Extends `SemanticMemory` in same file |
| `ConceptExtractor` | `src/maxim/memory/concept_extractor.py` | Same as `semantic_promoter.py` |
| `ConceptGrounder` | `src/maxim/memory/concept_grounder.py` | Same as `consolidation.py` |
| `ConceptContextBuilder` | `src/maxim/memory/concept_context.py` | Same pattern |
| `PatternCompleter` | `src/maxim/memory/pattern_completer.py` | Same pattern |
| `normalize_tokens` | `src/maxim/memory/text.py` | Shared utility for concept matching |
| `PredictedOutcome`, `MathContextEntry` | `src/maxim/memory/types.py` | **Already implemented** |

All wiring happens in `MemoryHub._wire_multi_layer()` — the same method
that already creates `CrossLayerGraph` and `SemanticPromoter`.

### Where concept extraction hooks in

```
Hippocampus.capture()
    -> stores EpisodicMemory
    -> fires capture callbacks (after write lock release)
        -> ConceptExtractor.on_memory_captured()
            -> ATL.find_or_create() for each percept concept
            -> Inline relationships between co-occurring concepts
            -> CrossLayerGraph.add_edge(INSTANCE_OF)
            -> Concept.add_ref("hippocampus", memory_id)
```

No changes to Hippocampus capture path. The ConceptExtractor is registered
as a callback in `MemoryHub._wire_multi_layer()` alongside existing
callbacks (NAc, SCN, EC).

### Where AG grounding happens

```
MemoryAgent._build_context()
    -> concept_context_builder.build(current_percept, budget_ms=50)
        -> ATL.recall() for matching concepts + goal tokens
        -> Hippocampus.recall_by_ids() for linked episodes (bulk load)
        -> Iterate registered layers for enrichment context
        -> For each concept (while budget remains):
            -> ConceptGrounder.ground_concept() (sync, in-memory arithmetic):
                -> IPS fast stats on loaded episodes
                -> AG escalation if IPS uncertain + enough data
                -> Modulate relationship confidence (Jaccard co-occurrence)
                -> Store QUANTIFIES edges for significant properties
        -> If budget exhausted: remaining concepts get relationships only
        -> Format relationships + stats for LLM context
```

### Sleep = maintenance only

Sleep does NOT discover relationships or compute properties. It maintains
the concept graph:

```
MemoryHub.on_session_end()
    -> hippocampus.sleep()                      # existing
    -> consolidation_orchestrator.run_wave()     # existing
    -> semantic_promoter.scan_for_promotions()   # existing (NAc patterns -> ATL)
    -> atl.consolidate()                         # existing: compress/remove
    -> save all layers                           # existing
```

ATL consolidation handles (upgraded to use `MemoryStrategy` pipeline —
see "Unified pruning model" section):
- **Score**: each concept via `MemoryStrategy.score_for_retention()`,
  with `TemporalAwareStrategy` wrapping SCN when connected
- **Remove**: concepts scoring below `retention_threshold` (0.2)
- **Compress**: concepts scoring below `compression_threshold` (0.4)
  -> CompressedSemantic
- **Prune relationships**: AG-weakened relationships below confidence
  threshold are removed during consolidation

### Interaction with existing ATL promotion

The existing `SemanticPromoter` promotes NAc patterns and StatisticianAgent
patterns. The `ConceptExtractor` promotes percept-derived concepts. These
are complementary:

| Source | What it promotes | When | Provenance |
|--------|-----------------|------|------------|
| `SemanticPromoter` | Reward patterns, statistical patterns | Sleep (on_session_end) | EPISODIC_CONSOLIDATION, AGENT_INFERENCE |
| `ConceptExtractor` | Objects, people, goals, actions | Every capture (callback) | EPISODIC_CONSOLIDATION |
| Future RAG | Document facts | Direct ingestion | DIRECT_INGESTION |

Concepts from all sources live in the same ATL, connected by the same
typed relationship graph. A NAc-promoted pattern like "grasp -> success"
can have a `RELATED_TO` edge to the "grasp" action concept and the object
concept it was performed on — linking causal knowledge to perceptual
knowledge.

### Interaction with CrossLayerGraph

The `CrossLayerGraph` already has the right edge types:

- `INSTANCE_OF`: episode -> concept (created by ConceptExtractor)
- `DERIVED_FROM`: concept -> source episodes (created by SemanticPromoter)
- `QUANTIFIES`: AG math record -> concept (created by ConceptGrounder)
- `STATISTICALLY_CONFIRMS`: AG pattern -> concept (existing)

---

## Lifecycle

### Concept birth

1. Maxim perceives "chair" in detected_objects.
2. Hippocampus captures the episodic memory (existing path).
3. ConceptExtractor callback fires -> `ATL.find_or_create("chair", "object")`.
4. First time: creates `Concept(name="chair", category="object", confidence=0.5)`.
5. CrossLayerGraph: hippocampus episode `INSTANCE_OF` ATL concept.
6. If episode also contains "kitchen", tentative `chair RELATED_TO kitchen`
   edge created (confidence 0.3).

### Concept growth

1. "Chair" seen again in a different episode, also in kitchen.
2. ConceptExtractor fires -> `find_or_create` finds existing concept.
3. `concept.reinforce(episode_id)` -> `reinforcement_count` increments,
   `confidence` grows: `min(0.99, 0.5 + 0.1 * sqrt(count))`.
4. New `INSTANCE_OF` edge added to CrossLayerGraph.
5. New `memory_refs["hippocampus"]` entry added.
6. Existing `chair RELATED_TO kitchen` edge reinforced (confidence bumps).

### Concept grounding

1. MemoryAgent recalls context, loads episodes linked to "chair".
2. ConceptGrounder piggybacks: extracts execution_time_ms, success_rate, etc.
3. IPS computes: mean grasp time 310ms, success rate 83%.
4. AG computes (if enough data): distance->time correlation r^2=0.78.
5. Relationship modulation: chair<->kitchen Jaccard 8/12 = 0.67 -> confidence strengthened.
6. MathMemory stored: "chair:execution_time_ms" with QUANTIFIES edge.
7. LLM receives: "Known concept: chair (object, confidence 0.92, usually in
   kitchen [0.85], avg grasp time 310ms, success rate 83%)."

### Concept decay

1. During ATL consolidation, concepts with no recent access and low
   confidence may be compressed (`CompressedSemantic`) or removed.
2. Concept deletion fires ATL deletion callback -> `CrossLayerGraph.remove_record()`.
3. `memory_refs` cleanup happens automatically via ConceptExtractor's
   reverse index and deletion callback.
4. AG MathMemory records linked via QUANTIFIES become orphaned and are
   cleaned up during AG's own consolidation cycle.

---

## Phase A5: Graph Chaining + Pattern Completion

### Overview

Graph chaining traverses the concept graph to find prior experiences
matching a partially-formed episodic memory. ATL provides the pattern
completion function that MemoryAgent hooks into during FORMING stage
(see the unified memory system for the
hook infrastructure).

The chain: **ATL concepts → linked episodes → prior decisions/actions/outcomes**,
enriched with per-concept math context from AG.

### PredictedOutcome and MathContextEntry

Pattern completion returns structured `PredictedOutcome` predictions
enriched with `MathContextEntry` AG stats. **Both are already implemented**
in `src/maxim/memory/types.py` as part of the unified memory system.
PatternCompleter imports from there — single source of truth.

Key fields (see `types.py` for full implementation):
- `PredictedOutcome`: tool, success, goal, confidence, math_context,
  source_episode_id — with `to_dict()` / `from_dict()` round-trip.
- `MathContextEntry`: name, verbal, confidence, domain — typed contract
  for AG enrichment, replacing raw dicts with implicit keys.

### Text normalization for concept matching

Goal strings, CLI input, and transcript text are free-form. To match
concepts reliably, we normalize tokens before lookup:

```python
# In a shared utility (e.g. maxim/utils/text.py or maxim/memory/text.py)

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "to", "in", "on", "of", "for", "is", "it",
    "and", "or", "but", "not", "with", "at", "by", "from", "as",
    "be", "was", "were", "been", "are", "am", "do", "does", "did",
    "has", "had", "have", "will", "would", "could", "should", "can",
    "this", "that", "these", "those", "i", "me", "my", "we", "our",
})

def normalize_tokens(text: str) -> list[str]:
    """Tokenize, filter stop words, and lemmatize for concept matching.

    Used by:
    - _find_matching_concepts() for goal string tokenization
    - EpisodicMemory.keywords() for CLI input tokenization
    - Future: transcript text processing

    Splits on whitespace AND underscores so compound identifiers like
    "navigate_to_kitchen" become ["navigate", "kitchen"] (after stop-word
    removal). Lemmatization uses basic suffix stripping (no NLTK dependency).
    Covers common English inflections: -ing, -ed, -s, -ly, -tion.
    Not linguistically perfect, but sufficient for concept name matching.
    """
    import re

    words = re.split(r"[\s_]+", text.lower())
    result = []
    for w in words:
        if w in _STOP_WORDS or len(w) < 2:
            continue
        result.append(_lemmatize(w))
    return result

def _lemmatize(word: str) -> str:
    """Basic suffix stripping. No external dependencies.

    Handles common English inflections without NLTK. Not linguistically
    perfect, but sufficient for concept name matching where the goal is
    "grasping" → "grasp", "mugs" → "mug", etc.

    Edge-case aware: checks stem validity (min length, vowel presence)
    to avoid garbage stems like "used" → "us" or "placed" → "plac".
    """
    # -ing: grasping → grasp, running → run
    if word.endswith("ing") and len(word) > 4:
        stem = word[:-3]
        if stem.endswith(stem[-1]) and len(stem) > 2:
            stem = stem[:-1]  # running → run
        if _has_vowel(stem):
            return stem
        return word
    # -ed: grasped → grasp, used → use, placed → place
    if word.endswith("ed") and len(word) > 3:
        if word.endswith("eed"):
            return word  # "freed" stays "freed"
        if word.endswith("ied") and len(word) > 4:
            return word[:-3] + "y"  # "carried" → "carry"
        # Try removing -ed, check if stem is valid
        stem = word[:-2]
        if len(stem) >= 2 and _has_vowel(stem):
            return stem
        # Try removing -d (for words like "placed" → "place")
        stem_d = word[:-1]
        if stem_d.endswith("e") and len(stem_d) >= 3:
            return stem_d
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"  # batteries → battery
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _has_vowel(word: str) -> bool:
    """Check if word contains at least one vowel."""
    return any(c in "aeiou" for c in word)
```

This is intentionally simple — no NLTK, no spaCy, no external deps.
The suffix stripping handles ~80% of inflections relevant to concept
names. If Maxim later adds a proper NLP pipeline, `normalize_tokens`
can delegate to it without changing callers.

### PatternCompleter — separated from ATL

Pattern completion is a cross-cutting concern that touches ATL, Hippocampus,
and AG. Putting it on ATL would make ATL responsible for: storing concepts,
managing relationships, serving as a layer registry, running pattern
completion, and finding matching concepts. That's a god-object trajectory.

Instead, `PatternCompleter` is a standalone collaborator — same pattern as
`ConceptGrounder` and `ConceptExtractor`. It lives in the integration layer,
receives memory layers via constructor injection, and has a single
responsibility: predict outcomes from concept-linked episodes.

```python
class PatternCompleter:
    """Predicts outcomes for partially-formed episodes via concept graph chaining.

    Traverses ATL concepts → linked episodes → past decisions/actions/outcomes,
    enriched with per-concept math context from registered layers.

    Separated from ATL to avoid god-object accumulation. ATL stores concepts;
    PatternCompleter queries them. Same separation as ConceptGrounder.

    Wired into MemoryAgent via set_pattern_completion_fn(completer.complete).
    """

    def __init__(
        self,
        atl: ATL,
        layers: dict[str, MemoryLayer],
    ):
        self._atl = atl
        self._layers = layers  # "hippocampus" -> Hippocampus, "angular_gyrus" -> AG, etc.

    def complete(self, episodic: EpisodicMemory) -> list[PredictedOutcome]:
        """Pattern completion function wired into MemoryAgent.

        Called during FORMING stage with partial EpisodicMemory
        (has Perception+Context, lacks Decision/Action/Outcome).
        Returns predicted outcomes from similar past experiences.
        """
        # 1. Find matching concepts from percept
        concepts = self._find_matching_concepts(episodic)
        if not concepts:
            return []

        # 2. Collect episode IDs from concept refs (deduplicated)
        hippocampus = self._layers.get("hippocampus")
        if not hippocampus:
            return []

        episode_ids: set[str] = set()
        for concept in concepts:
            episode_ids.update(concept.memory_refs.get("hippocampus", set()))

        if not episode_ids:
            return []

        # 3. Load ALL matched episodes, sort by recency, then cap.
        # Loading first ensures we get the most recent 20, not an
        # arbitrary 20 from set iteration order. recall_by_ids is
        # an in-memory dict lookup so loading all is cheap — the
        # cap bounds downstream processing, not I/O.
        MAX_EPISODES = 20
        all_episodes = hippocampus.recall_by_ids(list(episode_ids))
        episodes = sorted(
            all_episodes, key=lambda ep: ep.timestamp, reverse=True
        )[:MAX_EPISODES]

        # 4. Extract predictions from past outcomes
        # Handles both EpisodicMemory and CompressedMemory. Compressed
        # records have tool_name, success, and goal but lack
        # decision.confidence — use a default. ConceptExtractor's
        # on_memory_compressed removes refs, so compressed records are
        # rare here but handled for robustness.
        predictions = []
        for ep in episodes:
            if isinstance(ep, CompressedMemory):
                predictions.append(PredictedOutcome(
                    tool=ep.tool_name,
                    success=ep.success,
                    goal=ep.goal,
                    confidence=0.3,  # No decision.confidence on compressed
                    source_episode_id=ep.id,
                ))
            else:
                predictions.append(PredictedOutcome(
                    tool=ep.action.tool_name,
                    success=ep.outcome.success,
                    goal=ep.decision.intent.get("goal"),
                    confidence=ep.decision.confidence,
                    source_episode_id=ep.id,
                ))

        # 5. Enrich with per-concept math context using memory_refs
        # intersection — NOT string matching. A prediction matches a
        # concept if the prediction's source episode is in the concept's
        # memory_refs. This is exact (O(1) set lookup), handles compound
        # names, and doesn't depend on goal text formatting.
        for concept in concepts:
            layer_context = self._get_concept_layer_context(concept)
            if not layer_context:
                continue
            concept_episode_ids = concept.memory_refs.get("hippocampus", set())
            for pred in predictions:
                if pred.source_episode_id in concept_episode_ids:
                    pred.math_context = layer_context

        return predictions

    def _find_matching_concepts(self, episodic: EpisodicMemory) -> list[Concept]:
        """Find concepts matching the percept's objects, people, and goal.

        Objects and people are single-word concept names, so direct lookup works.
        Goals are free-form phrases ("grasp the mug") or underscore-separated
        identifiers ("navigate_to_kitchen"), so we tokenize via
        normalize_tokens() (splits on whitespace + underscores, filters stop
        words, lemmatizes) and look up each word individually.
        """
        matches = []
        seen: set[str] = set()  # Deduplicate concept IDs

        # Objects and people: direct single-term lookup (already normalized)
        search_terms: list[str] = list(
            episodic.perception.detected_objects
            + episodic.perception.detected_people
        )

        # Goal: tokenize, filter stop words, lemmatize
        if episodic.context.active_goal:
            search_terms.extend(
                normalize_tokens(episodic.context.active_goal)
            )

        for term in search_terms:
            concept = self._atl.recall(limit=1, name=term.lower())
            if concept and concept[0].id not in seen:
                matches.append(concept[0])
                seen.add(concept[0].id)

        return matches

    def _get_concept_layer_context(self, concept: Concept) -> list[MathContextEntry] | None:
        """Get enrichment context from all registered layers for a concept.

        Iterates over registered layers (except hippocampus, which provides
        episodes not enrichment). Uses ID-based lookup from concept.memory_refs.
        Currently only AG produces MathContextEntry, but future layers
        (social, emotional) can contribute their own context entries.
        """
        entries: list[MathContextEntry] = []

        for layer_name, layer in self._layers.items():
            if layer_name == "hippocampus":
                continue  # Hippocampus provides episodes, not enrichment

            ref_ids = concept.memory_refs.get(layer_name, set())
            if not ref_ids:
                continue

            records = layer.recall_by_ids(list(ref_ids)[:5])
            for record in records:
                from maxim.math.math_types import MathMemory
                if isinstance(record, MathMemory):
                    entries.append(MathContextEntry(
                        name=record.name,
                        verbal=record.verbal,
                        confidence=record.confidence,
                        domain=record.domain,
                    ))

        return entries if entries else None
```

### Key design decisions

1. **memory_refs intersection, not string matching.** Math context is
   attached per-concept by checking if the prediction's `source_episode_id`
   is in the concept's `memory_refs["hippocampus"]` set. This is O(1) set
   lookup, handles compound concept names, and doesn't depend on goal text
   formatting. Previous approach (checking `concept.name in goal.split()`)
   missed underscore-separated identifiers and could false-match substrings.

2. **ID-based AG lookup.** Uses `concept.memory_refs["angular_gyrus"]`
   IDs to look up AG records via `recall_by_ids()`, not name-based
   `recall(name=mid)`. This matches how memory_refs actually stores
   data (as ID sets).

3. **Separated from ATL.** PatternCompleter, ConceptGrounder, and
   ConceptExtractor are all standalone collaborators in the integration
   layer. ATL's responsibility stays focused: store concepts, manage
   relationships, support recall. Cross-cutting concerns live outside it.

4. **Episode ordering.** All linked episodes are loaded (in-memory dict
   lookups — cheap), then sorted by timestamp descending and capped at
   20. This ensures the most recent experiences inform predictions, not
   an arbitrary subset from set iteration. The cap bounds downstream
   processing (prediction extraction, math context enrichment), not I/O.

5. **recall_by_ids() prerequisite.** Both Hippocampus and AG need a
   bulk-load method (`recall_by_ids(ids: list[str]) -> list[MemoryRecord]`)
   that loads multiple records by ID in one call. Iterating
   `recall_by_id()` in a loop is wasteful. **IMPLEMENTED** — added to
   MemoryLayer ABC, Hippocampus, AngularGyrus, and ATL.

### Wiring

All wiring happens in `MemoryHub._wire_multi_layer()`:

```python
# In MemoryHub._wire_multi_layer():
layers = {
    "hippocampus": hippocampus,
    "angular_gyrus": angular_gyrus,
    # Future: "social": social_memory
}

# A4: Concept-aware recall
concept_context_builder = ConceptContextBuilder(
    atl, layers, concept_grounder=concept_grounder
)
# MemoryAgent calls builder.build() during _build_context()

# A5b: Pattern completion
pattern_completer = PatternCompleter(atl, layers)
memory_agent.set_pattern_completion_fn(pattern_completer.complete)
```

---

## Implementation Order

| Phase | Effort | What | Dependencies |
|-------|--------|------|--------------|
| A0. Semantics.update_edge() | Small (~30 lines) | **IMPLEMENTED.** DependencyGraph.update_edge() + find_edge() public APIs, Semantics.update_edge() delegates. | None |
| A1. Concept class + ATL capacity | Small (~80 lines) | Extend SemanticMemory with `memory_refs` (bounded, MAX_REFS_PER_LAYER=200), `to_dict`/`from_dict` with set↔list, ATL `_concept` flag dispatch, capacity eviction via `_get_memory_strategy()` | None |
| A1b. Unified pruning | Small (~60 lines) | ATL `_get_memory_strategy()` + `consolidate()` upgrade to use `MemoryStrategy`; Hippocampus `_evict_one()` store-time capacity enforcement; Hippocampus `register_compression_callback` | A1 |
| A2. ConceptExtractor | Medium (~180 lines) | Queue-based async extraction (worker thread), goal tokenization via `normalize_tokens()`, inline relationships, reverse index with rebuild, deletion + compression cleanup | A0, A1 |
| A3. ConceptGrounder + AG wiring | Medium (~200 lines) | IPS/AG stats (sync), CompressedMemory handling, relationship modulation via update_edge(), AG `update_record()` for existing records, QUANTIFIES edges, stats caching | A0, A2 |
| A4. Concept-aware recall | Medium (~120 lines) | `ConceptContextBuilder` (sync, budget-bounded), goal tokenization in `_find_matching_concepts()`, layer iterator pattern, StructuredContext.concept_context field | A2, A3 |
| A5a. recall_by_ids() prerequisite | Small (~20 lines) | **IMPLEMENTED.** Added to MemoryLayer ABC, Hippocampus, AngularGyrus, and ATL. | None |
| A5b. PatternCompleter | Medium (~150 lines) | Standalone class, CompressedMemory handling, normalize_tokens(), memory_refs intersection, recency-sorted episode selection | A2, A3, A5a, unified memory system |
| A6. Memory lifecycle docs | Small | Document how layers form, connect, and are managed long-term (`docs/memory-layer-lifecycle.md` or guide page update) | A1b-A5b |

**A0** is a prerequisite — without `update_edge()`, neither inline
reinforcement nor AG modulation can work.

**A1-A1b** are the foundation — concepts exist with bounded refs and
capacity enforcement, both layers get unified store-time + sleep-time
pruning using the same `MemoryStrategy` pipeline.

**A2** adds extraction — queue-based async worker decouples concept
registration from the capture callback chain. Goals are tokenized into
individual word concepts (matching how recall searches work).

**A3** is the AG integration — concepts get numerical depth. AG updates
go through `update_record()` public API. CompressedMemory records are
handled gracefully (limited field extraction).

**A4** is the payoff — synchronous `ConceptContextBuilder` (all callers
are sync, including ExecAgent's background thread). Budget-bounded
grounding with graceful degradation.

**A5a** is a small prerequisite — bulk ID lookup for Hippocampus and AG.
**A5b** is cross-plan integration — requires memory unification's
`_pattern_completion_fn` hook (U3) and concept memory refs (A2/A3).

**A6** is documentation — written after implementation.

Total new code: ~810 lines production + ~700 lines tests.

---

## Future Directions

### LLM-assisted relationship typing

The initial `_infer_relationship_type()` uses simple category heuristics.
A future enhancement: during sleep consolidation, ask the LLM to suggest
richer relationship types for high-confidence co-occurring concepts:

```
Given concepts "chair" (object) and "Dennis" (person) that co-occur in
12 episodes, what relationship best describes their connection?
Options: USES, OWNS, SITS_ON, RELATED_TO, ...
```

The `RelationshipRegistry.register()` method already supports runtime
type registration, so LLM-proposed types slot in naturally.

### Hierarchical concepts

IS_A relationships enable concept hierarchies: `chair IS_A furniture`,
`furniture IS_A household_item`. These emerge naturally from
LLM-assisted relationship typing or explicit RAG ingestion. The ATL
graph + `recall_associated()` spreading activation already traverses
hierarchies — no special hierarchy code needed.

### Concept merging

When two concepts refer to the same thing (e.g., "kitchen_chair" and
"dining_chair" are the same object), a merge operation combines their
`memory_refs` and relationships. This is a future deduplication
enhancement building on ATL's existing `name_similarity_threshold` (0.8).

### Additional memory types as spokes

The Concept/ATL hub is designed to accommodate future memory types:

| Future Memory Type | ATL Integration |
|--------------------|-----------------|
| Procedural memory | Action concepts link to learned procedures via HAS_PART |
| Spatial memory | Location concepts link to coordinate records via QUANTIFIES |
| Social memory | Person concepts accumulate interaction patterns |
| Emotional memory | Concepts linked to valence/arousal profiles |

Each new memory type adds a new `memory_refs` layer and new relationship
types, but the Concept class and ATL graph need no structural changes.

---

## Testing

- **Unit: `test_semantics_update_edge.py`** (Phase A0)
  - update_edge modifies weight on existing edge
  - update_edge modifies confidence on existing edge
  - confidence_delta adds to current confidence, clamped to [0.05, 0.95]
  - Symmetric edges updated in both directions
  - Returns False if edge not found
  - Does not create duplicate edges

- **Unit: `test_concept_class.py`**
  - Concept extends SemanticMemory (all inherited fields work)
  - add_ref / remove_ref track memory references per layer
  - ref_count returns total and per-layer counts
  - to_dict / from_dict round-trip preserves memory_refs
  - Backward compat: existing SemanticMemory records load without error

- **Unit: `test_concept_extractor.py`**
  - Extracts object concepts from detected_objects
  - Extracts person concepts from detected_people
  - Extracts goal concepts from active_goal
  - Extracts action concepts from tool_name
  - find_or_create: first call creates, second call reinforces
  - INSTANCE_OF cross-layer edge created for each extraction
  - memory_refs updated on concept
  - Inline relationships formed between co-occurring concepts
  - Active-concept gating: background objects don't auto-relate
  - MAX_RELATIONSHIPS_PER_EPISODE cap respected
  - Existing relationships reinforced via update_edge on repeat co-occurrence
  - Location extracted from observations["location"] or observations["room"]
  - Reverse index enables O(1) cleanup on deletion
  - rebuild_reverse_index() reconstructs from ATL memory_refs
  - Non-EpisodicMemory records ignored (CompressedMemory, etc.)

- **Unit: `test_concept_grounder.py`**
  - Extracts numerical fields from episodes (execution_time_ms, salience, etc.)
  - IPS fast path computes mean, min, max, trend
  - AG escalation triggers on 8+ data points with uncertain IPS
  - Relationship confidence modulated by Jaccard co-occurrence similarity
  - Jaccard is symmetric: same result regardless of which concept you compute from
  - High Jaccard (>0.3) with shared >= 3 strengthens relationship
  - Low Jaccard (<0.05) with union >= 10 weakens relationship
  - MathMemory stored with QUANTIFIES edge for significant properties
  - Stats cached and invalidated when new refs added
  - Empty episodes return empty stats

- **Unit: `test_concept_recall.py`**
  - get_concept_context returns concepts matching percept objects/people
  - Relationships included in context output
  - AG-computed stats included in context output
  - StructuredContext.concept_context populated correctly
  - Async with time budget: first concepts get full AG stats
  - Budget exhaustion: remaining concepts get relationships only (graceful degradation)
  - Uses recall_by_ids for bulk episode loading (not get() in a loop)

- **Unit: `test_pattern_completer.py`** (Phase A5)
  - PatternCompleter.complete() finds concepts from detected_objects and active_goal
  - Follows concept memory_refs to load linked episodes
  - Extracts decisions/actions/outcomes as predictions
  - Math context enrichment uses memory_refs intersection (source_episode_id in concept refs)
  - Does NOT use string matching — compound names like "navigate_to_kitchen" work correctly
  - ID-based AG lookup via concept.memory_refs["angular_gyrus"] (not name)
  - Returns empty list when no concepts match
  - Caps episode loading at 20 to bound work
  - Layer-based: new layer passed in constructor dict is picked up
  - recall_by_ids loads multiple episodes in one call
  - _find_matching_concepts searches objects, people, and goal with normalize_tokens()
  - PatternCompleter is a standalone class, not a method on ATL

- **Unit: `test_unified_pruning.py`** (Phase A1b)
  - ATL `_evict_one()` removes lowest-scored concept at capacity
  - ATL `_evict_one()` uses same `_get_memory_strategy()` as `consolidate()`
  - ATL `consolidate()` scores via MemoryStrategy, not hardcoded thresholds
  - ATL concepts with high ref_count survive eviction over low-ref concepts
  - ATL eviction with SCN connected uses TemporalAwareStrategy
  - ATL eviction without SCN falls back to AccessBasedStrategy
  - ATL SCN unregister called on eviction
  - Hippocampus `_evict_one()` removes lowest-scored memory at capacity
  - Hippocampus long-term memories never evicted at store time
  - Hippocampus eviction reuses `_get_memory_strategy()` (no custom scorer)
  - Hippocampus eviction with SCN uses TemporalAwareStrategy boosts
  - Both layers: eviction score and consolidation score are consistent
    (same concept scores low on both scales)
  - Capacity boundary: store at max_nodes triggers exactly one eviction
  - Capacity boundary: store below max_nodes does not trigger eviction

- **Unit: `test_memory_refs_bounded.py`**
  - memory_refs capped at MAX_REFS_PER_LAYER per layer
  - Oldest refs pruned when cap exceeded (FIFO via insertion-ordered set)
  - Compression callback removes ref from concept
  - Deletion callback removes ref from concept
  - Concept with zero refs after cleanup scored lower by strategy

- **Integration: `test_concept_lifecycle.py`**
  - Full lifecycle: perceive -> capture -> extract -> recall -> ground
  - Multiple episodes reinforce same concept (confidence grows)
  - Inline relationships form and reinforce over multiple captures
  - AG grounding enriches concept with numerical properties
  - Hippocampus deletion cleans up concept refs via reverse index
  - Hippocampus compression cleans up concept refs (compressed records not tracked)
  - ATL consolidation compresses old, low-confidence concepts
  - Cross-layer QUANTIFIES edges link AG records to concepts
  - Cross-layer INSTANCE_OF edges link episodes to concepts

---

## Architecture Notes

### recall_by_ids and access tracking

`recall_by_ids()` skips `touch()` — it's a bulk-load utility, not a
"user accessed this memory" signal. ConceptGrounder's `ground_concept()`
calls `concept.touch()` explicitly as a safeguard — ensures grounded
concepts are marked as recently accessed for consolidation scoring.
`touch()` is thread-safe (inherited from MemoryRecord).

### Relationship confidence vs. relationship weight

- **Weight** = how strongly this relationship influences spreading
  activation. Set by ConceptGrounder from Jaccard co-occurrence.
- **Confidence** = how certain we are this relationship is real.
  Starts at 0.3, grows with inline reinforcement, modulated by AG.

High weight + high confidence = strong, well-evidenced relationship.
Low weight + low confidence = pruned during ATL consolidation.

### TOCTOU between ConceptExtractor and ConceptGrounder

ConceptExtractor runs in its worker thread (writes to `memory_refs`).
ConceptGrounder runs during recall in the main thread (reads
`memory_refs`). These can overlap if a new episode is being processed
while a recall is in progress.

**Why it's low-risk in practice:**
1. ConceptExtractor's `add_ref()` calls `set.add()` which is atomic under
   CPython's GIL. ConceptGrounder reads `memory_refs` as a snapshot — it
   doesn't iterate while the set is being modified.
2. The worst case is a missed ref: ConceptGrounder computes stats without
   the latest episode. The 5-minute cache TTL means the next recall picks
   it up. This is eventual consistency, not data corruption.

**Formal resolution:** ConceptExtractor should acquire ATL's write lock
when modifying a concept's `memory_refs`, not just Hippocampus's callback
lock. This is already implicit if `add_ref()` is called through ATL's
public API (e.g., `atl.get()` + modify + the concept is stored in-place).
To make it explicit:

```python
# In ConceptExtractor._register_concept():
concept = self._atl.get(concept_id)
if concept and isinstance(concept, Concept):
    with self._atl._rwlock.write():
        concept.add_ref("hippocampus", memory_id)
```

However, since `memory_refs` is a mutable field on the concept object
stored in ATL's `_concepts` dict, and `set.add()` is GIL-atomic, this
lock is defensive rather than strictly necessary. Add it for correctness
but don't expect it to fix observable bugs.

**Alternative:** Make `Concept.memory_refs` use a thread-safe set wrapper
(e.g., wrapping mutations with a per-concept lock). This is over-engineering
for the current use case — revisit only if Maxim moves to multi-threaded
concurrent recalls.

### How NAc knowledge flows in (without direct wiring)

NAc's causal knowledge enters ATL through the existing SemanticPromoter
path — not through direct relationship modulation:

```
NAc learns: "grasp -> success" (Rescorla-Wagner)
    |
SemanticPromoter promotes -> ATL concept "grasp -> success" (causal_pattern)
    |
ConceptExtractor links "grasp" (action) to "mug" (object) inline
    |
ATL graph: mug <-RELATED_TO-> grasp <-DERIVED_FROM-> "grasp -> success"
    |
AG grounds: grasp success rate 83% on mug (QUANTIFIES edge)
```

NAc predicts outcomes. AG quantifies properties. ATL connects concepts.
Each system does what it's built for — no cross-wiring needed.
