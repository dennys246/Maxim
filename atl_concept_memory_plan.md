# ATL Concept Memory Plan

Defines the `Concept` class as a universal abstraction bridging memory types
(episodic, semantic, mathematical, future types) and outlines how the ATL
evolves from a pattern-promotion store into a true semantic concept memory.
AG (Angular Gyrus) serves as the numerical backbone — computing statistical
properties during recall and grounding concept relationships with evidence.

**Depends on:** [memory-unification-plan.md](memory-unification-plan.md)
for true piggybacking in concept grounding (Phase A3). Phases A1-A2
can proceed independently — Hippocampus capture callbacks and ATL
`find_or_create()` already exist.

**Prerequisites in existing code:** AG, IPS, CrossLayerGraph, and the
Semantics relationship system are all implemented and operational.
`Semantics.update_edge()` must be added (see Phase A0).

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
    # These are the memories that mention/involve this concept
    memory_refs: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_ref(self, layer_name: str, memory_id: str) -> None:
        """Register a memory that references this concept."""
        self.memory_refs[layer_name].add(memory_id)
        self.touch()

    def remove_ref(self, layer_name: str, memory_id: str) -> None:
        """Unregister a memory reference (e.g., when memory is deleted)."""
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

---

## Phase A0: Semantics.update_edge()

The existing `Semantics` class has `define()`, `find()`, and `remove()`,
but no way to modify an existing relationship's weight or confidence
in-place. Both inline reinforcement (Path 1) and AG modulation (Path 2)
require this. `DependencyGraph.add_edge()` always appends — calling
`define()` twice creates duplicate edges.

```python
# Addition to Semantics class:
def update_edge(
    self,
    source_id: str,
    target_id: str,
    rel_type: str,
    weight: float | None = None,
    confidence: float | None = None,
    confidence_delta: float | None = None,
) -> bool:
    """Update an existing relationship's weight and/or confidence.

    Args:
        weight: Set absolute weight (None = unchanged).
        confidence: Set absolute confidence (None = unchanged).
        confidence_delta: Add to current confidence, clamped to [0.05, 0.95].
            Mutually exclusive with confidence.

    Returns True if edge was found and updated, False if not found.
    """
    with self._graph._lock:
        for edge in self._graph._outgoing.get(source_id, []):
            if (
                edge.target == target_id
                and edge.metadata.get("relationship_type") == rel_type
            ):
                if weight is not None:
                    edge.weight = weight
                if confidence is not None:
                    edge.metadata["confidence"] = confidence
                elif confidence_delta is not None:
                    current = edge.metadata.get("confidence", 0.5)
                    edge.metadata["confidence"] = max(0.05, min(0.95, current + confidence_delta))

                # Update symmetric reverse edge
                if self._registry.is_symmetric(rel_type):
                    for rev in self._graph._outgoing.get(target_id, []):
                        if (
                            rev.target == source_id
                            and rev.metadata.get("relationship_type") == rel_type
                        ):
                            if weight is not None:
                                rev.weight = weight
                            rev.metadata["confidence"] = edge.metadata["confidence"]
                            break
                return True
    return False
```

This is ~30 lines added to `semantics.py`. It's a prerequisite for both
ConceptExtractor (inline reinforcement) and ConceptGrounder (modulation).

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

    def __init__(self, atl: ATL, cross_layer: CrossLayerGraph):
        self._atl = atl
        self._cross_layer = cross_layer
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

        # Goal
        if record.context.active_goal:
            concepts_found.append((record.context.active_goal, "goal"))

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
        concept_ids = self._reverse_index.pop(memory_id, set())
        for cid in concept_ids:
            concept = self._atl.get(cid)
            if concept and isinstance(concept, Concept):
                concept.remove_ref("hippocampus", memory_id)

    def rebuild_reverse_index(self) -> None:
        """Rebuild reverse index from ATL concept memory_refs.

        Called on startup to restore the reverse index (which is not
        persisted). O(concepts) but only runs once on boot.
        """
        self._reverse_index.clear()
        for concept in self._atl:
            if isinstance(concept, Concept):
                for mem_id in concept.memory_refs.get("hippocampus", set()):
                    self._reverse_index[mem_id].add(concept.id)
```

**Note on reverse index persistence:** The reverse index is NOT persisted.
It's rebuilt on startup from ATL's persisted `memory_refs` via
`rebuild_reverse_index()`, called during `MemoryHub.on_session_start()`.
This is O(concepts) but only runs once. Persisting the reverse index
separately would create a sync risk — if ATL's `memory_refs` and the
reverse index disagree after a crash, you get ghost references.

**Note on relationship type inference:** The initial implementation uses
simple category-based heuristics. `RELATED_TO` and `ASSOCIATES` are safe
defaults. Future work can add LLM-assisted relationship typing where the
model suggests richer types (IS_A, HAS_PART, CAUSES) based on episodic
context. The `RelationshipRegistry` is already extensible for agent-
proposed types at runtime.

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

**After memory unification** (see [memory-unification-plan.md](memory-unification-plan.md)):
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

    def __init__(
        self,
        atl: ATL,
        angular_gyrus: AngularGyrus,
        ips: IPS,
        cross_layer: CrossLayerGraph,
        cache_ttl: float = 300.0,  # 5 min cache
    ):
        self._atl = atl
        self._ag = angular_gyrus
        self._ips = ips
        self._cross_layer = cross_layer
        self._cache_ttl = cache_ttl
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
        self._modulate_relationships(concept, episodes)

        # Store AG MathMemory with QUANTIFIES edge if significant
        self._store_quantifications(concept, stats)

        return stats

    def _extract_numerics(
        self, episodes: list[EpisodicMemory]
    ) -> dict[str, list[float]]:
        """Extract numerical fields from loaded episodes.

        Returns field_name -> list of values across episodes.
        Only includes fields present in at least 2 episodes.
        """
        collectors: dict[str, list[float]] = defaultdict(list)

        for ep in episodes:
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

    def _modulate_relationships(
        self, concept: Concept, episodes: list[EpisodicMemory]
    ) -> None:
        """Strengthen or weaken concept relationships based on numerical
        co-occurrence evidence.

        For each related concept, compute what fraction of this concept's
        episodes also reference the related concept. High co-occurrence
        rate + enough observations = strengthen. Low rate = weaken.
        """
        relationships = self._atl.find_by_relationship(
            concept.id, direction="both", limit=50
        )
        if not relationships:
            return

        episode_ids = {ep.id for ep in episodes}

        for other_id, rel in relationships:
            other = self._atl.get(other_id)
            if not isinstance(other, Concept):
                continue

            other_refs = other.memory_refs.get("hippocampus", set())
            shared = len(episode_ids & other_refs)
            total = len(episode_ids)

            if total < 3:
                continue  # Not enough evidence to modulate

            co_occurrence_rate = shared / total

            # Strengthen: high co-occurrence with enough evidence
            if co_occurrence_rate > 0.5 and shared >= 3:
                self._atl.semantics.update_edge(
                    concept.id, other_id, rel.relationship_type,
                    weight=min(1.0, co_occurrence_rate),
                    confidence_delta=0.05,
                )
            # Weaken: low co-occurrence despite many observations
            elif co_occurrence_rate < 0.1 and total >= 10:
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
                # Update existing record in place — no new edges needed
                record = existing[0]
                record.properties.update(field_stats)
                record.observation_count = n
                record.confidence = min(0.9, 0.3 + 0.05 * n)
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
          +-- Modulate: mug<->kitchen 8/10 co-occurrence -> confidence 0.85
          +-- Modulate: mug<->lamp 2/10 co-occurrence -> confidence decays
          +-- Store: MathMemory "mug:execution_time_ms" QUANTIFIES mug
          +-- Context to LLM includes stats + relationships
```

---

## Concept-Aware Recall

When MemoryAgent builds context for the LLM, it looks up concepts matching
the current percept and traverses their relationships. The concept's
"properties" are its relationships and AG quantifications — no separate
data structure needed.

```python
def get_concept_context(
    self, percept: Percept, limit: int = 5
) -> list[dict]:
    """Get ATL concepts relevant to current perception.

    Returns concept summaries with relationships and AG-computed stats
    for inclusion in StructuredContext.
    """
    if not self._atl:
        return []

    concepts: list[tuple[Concept, float]] = []

    # Look up concepts for detected objects and people
    for name in self._extract_percept_names(percept):
        results = self._atl.recall(limit=1, name=name.lower())
        for concept in results:
            if isinstance(concept, Concept):
                concepts.append((concept, concept.confidence))

    # Deduplicate and rank by confidence
    seen = set()
    unique: list[Concept] = []
    for concept, score in sorted(concepts, key=lambda x: -x[1]):
        if concept.id not in seen:
            seen.add(concept.id)
            unique.append(concept)
    unique = unique[:limit]

    # Build context for each concept
    context_entries = []
    for concept in unique:
        # Load linked episodes (for AG grounding)
        episode_ids = list(concept.memory_refs.get("hippocampus", set()))
        episodes = [self._hippocampus.get(eid) for eid in episode_ids[:20]]
        episodes = [e for e in episodes if e is not None]

        # AG grounding (piggybacks on loaded episodes)
        stats = {}
        if self._concept_grounder and episodes:
            stats = self._concept_grounder.ground_concept(concept, episodes)

        # Get relationships
        relationships = self._atl.find_by_relationship(
            concept.id, direction="both", limit=10
        )
        rel_summaries = []
        for other_id, rel in relationships:
            other = self._atl.get(other_id)
            if other:
                rel_summaries.append({
                    "type": rel.rel_type,
                    "target": other.name,
                    "confidence": rel.confidence,
                })

        # Get AG quantifications
        ag_props = {}
        for field, field_stats in stats.items():
            if field.startswith("_"):
                continue
            ag_props[field] = {
                "mean": field_stats.get("mean"),
                "n": field_stats.get("n"),
                "trend": field_stats.get("trend"),
            }

        context_entries.append({
            "name": concept.name,
            "category": concept.category,
            "confidence": concept.confidence,
            "episode_count": concept.ref_count("hippocampus"),
            "relationships": rel_summaries,
            "properties": ag_props,
        })

    return context_entries
```

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
    -> get_concept_context(current_percept)
        -> ATL.recall() for matching concepts
        -> Hippocampus.get() for linked episodes (already loaded for recall)
        -> ConceptGrounder.ground_concept() piggybacks:
            -> IPS fast stats on loaded episodes
            -> AG escalation if IPS uncertain + enough data
            -> Modulate relationship confidence
            -> Store QUANTIFIES edges for significant properties
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

ATL consolidation (already implemented) handles:
- **Remove**: concepts with no recent access, low confidence, no refs
- **Compress**: old concepts with few accesses -> CompressedSemantic
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
5. Relationship modulation: chair<->kitchen 8/10 co-occurrence -> confidence 0.85.
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
(see [memory-unification-plan.md](memory-unification-plan.md) for the
hook infrastructure).

The chain: **ATL concepts → linked episodes → prior decisions/actions/outcomes**,
enriched with per-concept math context from AG.

### PredictedOutcome dataclass

Pattern completion returns structured predictions, not raw dicts.
This enforces a contract between ATL (producer) and MemoryAgent (consumer):

```python
@dataclass
class PredictedOutcome:
    """A predicted outcome from pattern completion.

    Produced by ATL graph chaining, consumed by MemoryAgent during
    FORMING stage. Typed fields enforce the contract between the
    two systems — no implicit dict key expectations.
    """
    tool: str                                    # Action tool used in the past
    success: bool                                # Whether the past action succeeded
    goal: str | None = None                      # Goal from the past decision
    confidence: float = 1.0                      # Decision confidence from the past episode
    math_context: list[MathContextEntry] | None = None  # Per-concept layer stats (attached during enrichment)
    source_episode_id: str = ""                  # Which episode this prediction came from

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool, "success": self.success,
            "goal": self.goal, "confidence": self.confidence,
            "math_context": [m.to_dict() for m in self.math_context] if self.math_context else None,
            "source_episode_id": self.source_episode_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictedOutcome":
        math_ctx = data.get("math_context")
        return cls(
            tool=data["tool"], success=data["success"],
            goal=data.get("goal"), confidence=data.get("confidence", 1.0),
            math_context=[MathContextEntry.from_dict(m) for m in math_ctx] if math_ctx else None,
            source_episode_id=data.get("source_episode_id", ""),
        )


@dataclass
class MathContextEntry:
    """AG math context for a single property of a concept.

    Typed contract for the math enrichment step in pattern completion.
    Eliminates raw dict with implicit keys.
    """
    name: str              # MathMemory name (e.g. "mug:execution_time_ms")
    verbal: str = ""       # Human-readable label (e.g. "typically ~310ms")
    confidence: float = 0.0
    domain: str = ""       # e.g. "timing", "success_rate"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "verbal": self.verbal,
                "confidence": self.confidence, "domain": self.domain}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MathContextEntry":
        return cls(name=data["name"], verbal=data.get("verbal", ""),
                   confidence=data.get("confidence", 0.0),
                   domain=data.get("domain", ""))
```

**Defined in:** `types.py` (alongside other memory types). Both ATL
and MemoryAgent import from there — single source of truth.

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

    Lemmatization uses basic suffix stripping (no NLTK dependency).
    Covers common English inflections: -ing, -ed, -s, -ly, -tion.
    Not linguistically perfect, but sufficient for concept name matching.
    """
    words = text.lower().split()
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

### Memory layer registry

Instead of hardcoded `elif` chains for each memory system, graph chaining
uses a registry of memory layers to iterate over:

```python
class ATL:
    def __init__(self, ...):
        # ... existing init ...
        # Registry: layer_name → MemoryLayer instance (initialized per-instance,
        # NOT as class variable — mutable default on class attr is shared across
        # all instances, breaking test isolation)
        self._chaining_layers: dict[str, MemoryLayer] = {}

    def register_layer(self, name: str, layer: MemoryLayer) -> None:
        """Register a memory layer for graph chaining.

        New memory systems (e.g. social memory, emotional memory)
        register here and get picked up automatically.
        """
        self._chaining_layers[name] = layer

    def pattern_complete(self, episodic: EpisodicMemory) -> list[PredictedOutcome]:
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
        hippocampus = self._chaining_layers.get("hippocampus")
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
        predictions = []
        for ep in episodes:
            predictions.append(PredictedOutcome(
                tool=ep.action.tool_name,
                success=ep.outcome.success,
                goal=ep.decision.intent.get("goal"),
                confidence=ep.decision.confidence,
                source_episode_id=ep.id,
            ))

        # 5. Enrich with per-concept context from registered layers
        # Match per-concept: a prediction matches a concept if the concept
        # name appears as a word in the prediction's goal string.
        for concept in concepts:
            layer_context = self._get_concept_layer_context(concept)
            if layer_context:
                for pred in predictions:
                    if pred.goal and concept.name.lower() in pred.goal.lower().split():
                        pred.math_context = layer_context

        return predictions

    def _find_matching_concepts(self, episodic: EpisodicMemory) -> list[Concept]:
        """Find concepts matching the percept's objects, people, and goal.

        Objects and people are single-word concept names, so direct lookup works.
        Goals are free-form phrases ("grasp the mug"), so we tokenize, filter
        stop words, lemmatize, and look up each word individually.

        Text normalization (stop-word removal + lemmatization) is shared with
        CLI input and transcript processing via normalize_tokens(). This
        ensures concept names match regardless of inflection ("grasping" →
        "grasp", "mugs" → "mug").

        Performance note: This does N serial recall() calls (one per search
        term). ATL recall is an in-memory scan, so this is fine for now. If
        ATL recall ever becomes async or I/O-bound, batch with recall_by_names().
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
            concept = self.recall(limit=1, name=term.lower())
            if concept and concept[0].id not in seen:
                matches.append(concept[0])
                seen.add(concept[0].id)

        return matches

    def _get_concept_layer_context(self, concept: Concept) -> list[MathContextEntry] | None:
        """Get enrichment context from all registered layers for a concept.

        Iterates over _chaining_layers (except hippocampus, which provides
        episodes not enrichment). Uses ID-based lookup from concept.memory_refs.
        Currently only AG produces MathContextEntry, but future layers
        (social, emotional) can contribute their own context entries.
        """
        entries: list[MathContextEntry] = []

        for layer_name, layer in self._chaining_layers.items():
            if layer_name == "hippocampus":
                continue  # Hippocampus provides episodes, not enrichment

            ref_ids = concept.memory_refs.get(layer_name, set())
            if not ref_ids:
                continue

            records = layer.recall_by_ids(list(ref_ids)[:5])
            for record in records:
                # Use to_context_dict() (MemoryRecord ABC method) to extract
                # structured context. isinstance check ensures we only create
                # MathContextEntry from actual MathMemory records — no duck
                # typing via hasattr which could false-match future record types.
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

1. **Per-concept math context.** Math context is attached per-concept
   (only to predictions matching that concept's goal), not broadcast
   to all predictions. This avoids polluting unrelated predictions
   with irrelevant numerical data.

2. **ID-based AG lookup.** Uses `concept.memory_refs["angular_gyrus"]`
   IDs to look up AG records via `recall_by_ids()`, not name-based
   `recall(name=mid)`. This matches how memory_refs actually stores
   data (as ID sets).

3. **Uniform registry access.** All memory layers (including Hippocampus)
   are accessed through `_chaining_layers`. No separate `self._hippocampus`
   reference — Hippocampus is just another registered layer. New layers
   register via `register_layer()` and get picked up by
   `_get_concept_layer_context()` automatically.

4. **Episode ordering.** All linked episodes are loaded (in-memory dict
   lookups — cheap), then sorted by timestamp descending and capped at
   20. This ensures the most recent experiences inform predictions, not
   an arbitrary subset from set iteration. The cap bounds downstream
   processing (prediction extraction, math context enrichment), not I/O.

5. **recall_by_ids() prerequisite.** Both Hippocampus and AG need a
   bulk-load method (`recall_by_ids(ids: list[str]) -> list[MemoryRecord]`)
   that loads multiple records by ID in one call. Iterating
   `recall_by_id()` in a loop is wasteful. This is a prerequisite for
   A5 — see implementation order. The implementation is trivial: filter
   the in-memory dict by the provided IDs.

### Wiring

All memory layers are registered through the same `register_layer()` API.
There is no separate `self._hippocampus` reference — Hippocampus is a
memory layer and is accessed through the registry like everything else.

```python
# In MemoryHub or MaximAgent.wire_memory_hub():
atl.register_layer("hippocampus", hippocampus)
atl.register_layer("angular_gyrus", angular_gyrus)
# Future: atl.register_layer("social", social_memory)

# Wire pattern completion into MemoryAgent
memory_agent.set_pattern_completion_fn(atl.pattern_complete)
```

---

## Implementation Order

| Phase | Effort | What | Dependencies |
|-------|--------|------|--------------|
| A0. Semantics.update_edge() | Small (~30 lines) | Add update_edge() to Semantics for in-place weight/confidence modification | None |
| A1. Concept class | Small (~40 lines) | Extend SemanticMemory with `memory_refs`, update ATL serialization | None |
| A2. ConceptExtractor | Medium (~150 lines) | Capture callback, inline relationships with active-concept gating, reverse index with rebuild, deletion cleanup | A0, A1 |
| A3. ConceptGrounder + AG wiring | Medium (~200 lines) | IPS/AG stats during recall, relationship modulation via update_edge(), QUANTIFIES edges, stats caching | A0, A2 |
| A4. Concept-aware recall | Small (~80 lines) | `get_concept_context()` in MemoryAgent/MemoryHub, StructuredContext.concept_context field | A2, A3 |
| A5a. recall_by_ids() prerequisite | Small (~20 lines) | Add `recall_by_ids(ids) -> list[MemoryRecord]` to Hippocampus and AngularGyrus. Bulk in-memory dict lookup by ID list. | None |
| A5b. Graph chaining + pattern completion | Medium (~150 lines) | `PredictedOutcome` dataclass, `pattern_complete()`, `_find_matching_concepts()` with word-tokenized goal matching, `_get_concept_layer_context()`, layer registry, recency-sorted episode selection | A2, A3, A5a, memory-unification U3 |

**A0** is a prerequisite — without `update_edge()`, neither inline
reinforcement nor AG modulation can work.

**A1-A2** are the foundation — concepts exist and form relationships
inline. Can be implemented and tested independently.

**A3** is the AG integration — concepts get numerical depth. Benefits
from memory unification ([memory-unification-plan.md](memory-unification-plan.md))
for true piggybacking, but works without it (in-memory dict lookups).

**A4** is the payoff — the LLM gets rich concept context.

**A5a** is a small prerequisite — bulk ID lookup for Hippocampus and AG.
**A5b** is cross-plan integration — requires memory unification's
`_pattern_completion_fn` hook (U3) and concept memory refs (A2/A3).
This is what makes pattern completion actually work end-to-end.

Total new code: ~650 lines production + ~550 lines tests.

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
  - Relationship confidence modulated by co-occurrence rate
  - High co-occurrence strengthens relationship
  - Low co-occurrence weakens relationship
  - MathMemory stored with QUANTIFIES edge for significant properties
  - Stats cached and invalidated when new refs added
  - Empty episodes return empty stats

- **Unit: `test_concept_recall.py`**
  - get_concept_context returns concepts matching percept objects/people
  - Relationships included in context output
  - AG-computed stats included in context output
  - StructuredContext.concept_context populated correctly

- **Unit: `test_graph_chaining.py`** (Phase A5)
  - pattern_complete finds concepts from detected_objects and active_goal
  - Follows concept memory_refs to load linked episodes
  - Extracts decisions/actions/outcomes as predictions
  - Per-concept math context: only predictions matching concept goal get math_context
  - ID-based AG lookup via concept.memory_refs["angular_gyrus"] (not name)
  - Returns empty list when no concepts match
  - Caps episode loading at 20 to bound work
  - Registry-based: new layer registered via register_layer() is picked up
  - recall_by_ids loads multiple episodes in one call
  - _find_matching_concepts searches objects, people, and goal

- **Integration: `test_concept_lifecycle.py`**
  - Full lifecycle: perceive -> capture -> extract -> recall -> ground
  - Multiple episodes reinforce same concept (confidence grows)
  - Inline relationships form and reinforce over multiple captures
  - AG grounding enriches concept with numerical properties
  - Hippocampus deletion cleans up concept refs via reverse index
  - ATL consolidation compresses old, low-confidence concepts
  - Cross-layer QUANTIFIES edges link AG records to concepts
  - Cross-layer INSTANCE_OF edges link episodes to concepts

---

## Architecture Notes

### Why Concept extends SemanticMemory (not replaces)

SemanticMemory is a `MemoryRecord` subclass that ATL already stores and
persists. Adding `memory_refs` is additive. Existing `SemanticMemory`
records in persistence files deserialize as `Concept` instances with empty
`memory_refs` — they work but don't have cross-layer tracking until
they're observed again.

The alternative — a separate `Concept` class alongside `SemanticMemory` —
would require ATL to store two types, with different query paths, different
consolidation logic, and different serialization.

### Why extraction happens in capture callbacks (not in ATL)

The `ConceptExtractor` is registered as a Hippocampus capture callback.
This means:
1. No changes to Hippocampus code.
2. Extraction runs after the write lock releases (non-blocking).
3. ATL concept operations are independent (their own RWLock).
4. Easy to extend: add more extraction logic without touching the capture path.

The extractor lives in the integration layer (alongside SemanticPromoter),
not in ATL or Hippocampus. It bridges the two.

### Why AG grounds during recall (not during capture or sleep)

- **Not during capture**: AG analysis requires multiple observations. A
  single episode can't establish trends or correlations. ConceptExtractor
  handles the lightweight extraction; AG waits for data to accumulate.
- **Not during sleep**: A separate sleep scan would load episodes just
  to analyze them. Recall-time grounding does the same work at a point
  where the results are immediately useful (LLM context).
- **During recall**: The concept is being accessed (so the work is
  useful) and the results go directly into LLM context. Episode loading
  is trivially fast (in-memory dict lookups on Hippocampus). After memory
  unification, MemoryAgent's working set can be filtered directly —
  true piggybacking with fallback to Hippocampus for supplemental data.
  Cache (5-min TTL, invalidated on new refs) prevents recomputation.

### Why concepts don't have observed_properties

A concept's properties ARE its relationship graph:
- "Chair is usually in kitchen" = `chair RELATED_TO kitchen (confidence 0.85)`
- "Chair grasp takes ~310ms" = AG MathMemory QUANTIFIES chair
- "Chair co-occurs with table" = `chair RELATED_TO table (confidence 0.72)`

A separate `observed_properties` dict would duplicate information already
represented in the graph, without the typed structure, confidence tracking,
or persistence that the graph provides. The graph IS the property system.

### Relationship confidence vs. relationship weight

- **Weight** (graph edge weight) = how strongly this relationship influences
  spreading activation. Set by ConceptGrounder from co-occurrence rate.
- **Confidence** (edge metadata) = how certain we are this relationship
  is real. Starts at 0.3 (tentative), grows with inline reinforcement,
  modulated by AG's numerical analysis during recall.

High weight + high confidence = strong, well-evidenced relationship.
Low weight + low confidence = weak, tentative association (pruned during
ATL consolidation).

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
