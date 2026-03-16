# Worker Pool × ATL Concept Memory Integration Plan

> **Goal:** Wire the dormant `review` and `record` lanes from WorkerPool into the
> ATL concept memory pipeline, moving graph-heavy relationship analysis off the
> synchronous recall path and into background processing.

---

## Problem

ConceptGrounder runs **synchronously** during recall within a 50ms budget
([concept_context.py:56](src/maxim/memory/concept_context.py)). Two of its
operations are expensive and block the LLM context-building path:

1. **`_modulate_relationships()`** — Jaccard co-occurrence scoring across all
   related concepts. O(related_concepts × ref_unions). Reads from multiple
   concepts' `memory_refs`, computes set intersections, then writes edge updates.

2. **`_store_quantifications()`** — For each numerical field with n >= 5, creates
   or updates an AG MathMemory record, adds a QUANTIFIES cross-layer edge, and
   updates concept refs. Multiple ATL + AG write operations per concept per field.

When budget is exhausted, remaining concepts get **relationship context only**
(graceful degradation). But the relationship data itself may be stale because
modulation hasn't run recently.

---

## Current Architecture

```
ConceptContextBuilder.build()  [SYNC, 50ms budget]
  └─> FOR EACH concept:
       ├─> hippocampus.recall_by_ids()     [fast: dict lookups]
       ├─> ConceptGrounder.ground_concept() [BOTTLENECK]
       │    ├─> _extract_numerics()         [fast: field extraction]
       │    ├─> _compute_stats()            [medium: IPS + conditional AG]
       │    ├─> _modulate_relationships()   [SLOW: Jaccard × related concepts]
       │    └─> _store_quantifications()    [SLOW: AG writes × fields]
       ├─> _collect_layer_enrichment()      [fast: ref lookups]
       └─> _collect_relationships()         [fast: graph query]
```

ConceptExtractor already has its own queue + worker thread for capture-time
concept registration — this works well and should not be replaced.

---

## Proposed Architecture

Split ConceptGrounder into a **fast sync path** (stats + cache) and a
**background async path** (relationship modulation + quantification storage):

```
ConceptContextBuilder.build()  [SYNC, 50ms budget]
  └─> FOR EACH concept:
       ├─> hippocampus.recall_by_ids()          [fast]
       ├─> ConceptGrounder.ground_concept()      [FAST NOW]
       │    ├─> _extract_numerics()              [fast]
       │    ├─> _compute_stats()                 [medium, cached]
       │    ├─> _enqueue_background_work()       [non-blocking]
       │    │    └─> WorkerPool.submit("review") [fire-and-forget]
       │    └─> return cached stats immediately
       ├─> _collect_layer_enrichment()           [fast]
       └─> _collect_relationships()              [fast]

                        ↓ (async, background)

REVIEW LANE: _review_concept_relationships(concept_id, stats)
  ├─> ATL.find_by_relationship()                 [graph read]
  ├─> Jaccard co-occurrence scoring              [compute]
  ├─> Propose edge changes: [(src, tgt, delta)]  [compute]
  ├─> Identify AG quantification candidates      [compute]
  └─> WorkerPool.submit("record", changes)       [hand off]

                        ↓ (async, background)

RECORD LANE: _apply_concept_updates(changes)
  ├─> ATL.semantics.update_edge() × N            [graph write]
  ├─> AG.store() or update × N                   [memory write]
  ├─> CrossLayerGraph.add_edge(QUANTIFIES) × N   [graph write]
  └─> concept.add_ref("angular_gyrus") × N       [ref write]
```

**Key insight:** The LLM gets stats from cache (fast), and relationships are
kept fresh by background workers that run *between* recall calls. The 50ms
budget is almost entirely freed up for loading episodes and collecting context.

---

## Design Decisions

### Why not replace ConceptExtractor's queue?

ConceptExtractor's queue serves a different purpose — it decouples hippocampus
capture callbacks from ATL write locks. It handles **capture-time extraction**
(episode → concepts). The review/record lanes handle **recall-time analysis**
(concepts → relationship updates). Different triggers, different data flows.

### Why two lanes instead of one?

The review lane does **read-heavy computation** (Jaccard scoring reads from
multiple concepts' `memory_refs`, loads related concepts, computes set
intersections). The record lane does **write-heavy I/O** (ATL edge updates,
AG stores, cross-layer edges). Separating them means:

- Review can run at `max_workers=1` (CPU-bound, no lock contention)
- Record can run at `max_workers=2` (I/O-bound, batches writes)
- A slow review job doesn't block fast record writes from other sources

### What about stale relationships?

On first recall after startup, relationships may be stale (no background
modulation has run yet). This is acceptable because:

1. `_compute_stats()` still runs synchronously — the LLM gets fresh numerical
   properties immediately
2. Relationships from the previous session are persisted and still valid
3. After the first recall triggers background modulation, subsequent recalls
   see fresh relationship data (typically within seconds)

### What about ConceptExtractor integration?

ConceptExtractor's `_form_inline_relationships()` creates initial low-confidence
edges (0.3) between co-occurring concepts. These are currently synchronous
within its worker thread. Phase 2 of this plan optionally moves relationship
formation to the review lane too, so all relationship work flows through the
same pipeline.

---

## Implementation Phases

### Phase 1: Background Relationship Modulation

**Goal:** Move `_modulate_relationships()` and `_store_quantifications()` off
the sync path.

**Files to modify:**
- `src/maxim/memory/concept_grounder.py` — Split ground_concept() into sync + async
- `src/maxim/integration/memory_hub.py` — Pass WorkerPool to ConceptGrounder
- `src/maxim/agents/llm_worker.py` — Share WorkerPool reference

**Changes to ConceptGrounder:**

```python
class ConceptGrounder:
    def __init__(self, atl, angular_gyrus, ips, cross_layer_graph=None,
                 worker_pool=None):  # NEW
        ...
        self._pool = worker_pool

    def ground_concept(self, concept, episodes):
        """Fast sync path: stats + cache. Heavy work enqueued."""
        if self._check_cache(concept.id):
            return self._stats_cache[concept.id][1]

        numerics = self._extract_numerics(episodes)
        if not numerics:
            return {}

        stats = self._compute_stats(concept, numerics)
        self._cache_stats(concept.id, stats)

        # Enqueue heavy work if pool available, otherwise run inline
        if self._pool is not None:
            self._enqueue_background_work(concept.id, stats)
        else:
            # Fallback: sync (preserves existing behavior without pool)
            self._modulate_relationships(concept)
            self._store_quantifications(concept, stats)

        return stats

    def _enqueue_background_work(self, concept_id, stats):
        """Submit relationship modulation + quantification to review lane."""
        self._pool.submit(
            lane="review",
            job_id=f"concept-review-{concept_id}-{time.monotonic_ns()}",
            fn=partial(self._review_concept, concept_id, dict(stats)),
            priority=5,
        )

    def _review_concept(self, concept_id, stats, prefetched=None):
        """REVIEW LANE: Compute relationship changes + quantification proposals."""
        concept = self._atl.get(concept_id)
        if concept is None:
            return  # Concept was evicted between enqueue and execution

        # Compute proposed changes (read-heavy, no writes)
        edge_updates = self._compute_relationship_updates(concept)
        quant_proposals = self._compute_quantification_proposals(concept, stats)

        if not edge_updates and not quant_proposals:
            return

        # Hand off writes to record lane
        self._pool.submit(
            lane="record",
            job_id=f"concept-record-{concept_id}-{time.monotonic_ns()}",
            fn=partial(self._apply_updates, concept_id, edge_updates,
                       quant_proposals),
            priority=7,  # Lower priority than review
        )

    def _apply_updates(self, concept_id, edge_updates, quant_proposals,
                       prefetched=None):
        """RECORD LANE: Apply computed changes under write locks."""
        concept = self._atl.get(concept_id)
        if concept is None:
            return

        for source_id, target_id, confidence_delta in edge_updates:
            self._atl.semantics.update_edge(source_id, target_id,
                                            confidence_delta=confidence_delta)

        for proposal in quant_proposals:
            self._store_single_quantification(concept, proposal)
```

**Changes to MemoryHub._wire_multi_layer():**

```python
# In _wire_multi_layer():
pool = getattr(self, '_worker_pool', None)  # Set by AgenticRuntime

self._concept_grounder = ConceptGrounder(
    atl=self.atl,
    angular_gyrus=self.angular_gyrus,
    ips=ips_instance,
    cross_layer_graph=self._cross_layer,
    worker_pool=pool,  # NEW
)
```

**Tests needed:**
- ground_concept() returns cached stats without blocking on review/record
- Review job computes correct Jaccard proposals
- Record job applies edge updates under ATL write lock
- Fallback: sync path works when worker_pool is None
- Evicted concept: review/record jobs handle missing concept gracefully
- Concurrent: multiple concepts grounded simultaneously don't deadlock

---

### Phase 2: ConceptExtractor Relationship Offload (optional)

**Goal:** Move `_form_inline_relationships()` from ConceptExtractor's worker
thread to the review lane, unifying all relationship work.

**Why optional:** ConceptExtractor's worker thread runs in the background
already. The benefit is unification (one pipeline for all relationship work)
rather than performance.

**Changes to ConceptExtractor:**

```python
def _process_capture(self, memory_id, record):
    concept_ids = []
    for name, category in self._extract_concept_candidates(record):
        cid = self._register_concept(name, category, memory_id, record)
        if cid:
            concept_ids.append(cid)

    # Instead of inline relationship formation:
    if self._pool is not None and len(concept_ids) >= 2:
        self._pool.submit(
            lane="review",
            job_id=f"rel-formation-{memory_id}",
            fn=partial(self._form_inline_relationships, list(concept_ids)),
            priority=8,  # Lower priority than concept grounding reviews
        )
    elif len(concept_ids) >= 2:
        self._form_inline_relationships(concept_ids)  # Fallback
```

---

### Phase 3: Deduplication & Batching (optimization)

**Goal:** Avoid redundant review jobs when the same concept is grounded
multiple times in quick succession.

**Mechanism:** A lightweight dedup filter in ConceptGrounder:

```python
class ConceptGrounder:
    def __init__(self, ...):
        ...
        self._pending_reviews: dict[str, float] = {}  # concept_id → enqueue_time
        self._pending_lock = threading.Lock()
        self._review_cooldown_s = 2.0  # Don't re-review within 2s

    def _enqueue_background_work(self, concept_id, stats):
        with self._pending_lock:
            last = self._pending_reviews.get(concept_id, 0)
            if time.monotonic() - last < self._review_cooldown_s:
                return  # Already enqueued recently
            self._pending_reviews[concept_id] = time.monotonic()

        self._pool.submit(
            lane="review",
            job_id=f"concept-review-{concept_id}-{time.monotonic_ns()}",
            fn=partial(self._review_concept, concept_id, dict(stats)),
            priority=5,
        )

    def _review_concept(self, concept_id, stats, prefetched=None):
        # Clear pending flag after execution
        with self._pending_lock:
            self._pending_reviews.pop(concept_id, None)
        ...
```

---

## Concurrency Safety

| Operation | Lock Required | Lane | Notes |
|---|---|---|---|
| `_extract_numerics()` | ATL read lock (via `recall_by_ids`) | sync | Episodes already loaded by caller |
| `_compute_stats()` | None (pure computation) | sync | Operates on local data |
| `_compute_relationship_updates()` | ATL read lock | review | Reads concept refs + related concepts |
| `_compute_quantification_proposals()` | None (pure computation) | review | Operates on stats dict |
| `ATL.semantics.update_edge()` | ATL write lock | record | Short-held write |
| `AG.store()` | AG write lock | record | Independent lock from ATL |
| `CrossLayerGraph.add_edge()` | CrossLayer lock | record | Independent lock |
| `concept.add_ref()` | ATL write lock | record | Part of concept update |

**No circular lock dependencies:** Review lane only reads (ATL read lock).
Record lane writes to ATL, AG, and CrossLayerGraph — all independent locks,
never held simultaneously.

**ConceptExtractor interaction:** ConceptExtractor's worker thread writes to
ATL (find_or_create, add_ref). Record lane also writes to ATL. Both acquire
ATL write lock independently — no contention beyond normal RWLock serialization.

---

## Wiring Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │              AgenticRuntime                      │
                    │  owns WorkerPool (infer, review, record lanes)   │
                    └──────┬──────────────────────────────────────────┘
                           │ passes pool reference
                    ┌──────▼──────┐
                    │  MemoryHub  │
                    │             │─────── ConceptExtractor (own queue)
                    │             │         └─ Phase 2: submits to review lane
                    │             │
                    │             │─────── ConceptGrounder (sync stats + async review)
                    │             │         ├─ ground_concept() → cached stats [sync]
                    │             │         ├─ submit to review lane [async]
                    │             │         └─ review → submit to record lane [async]
                    │             │
                    │             │─────── ConceptContextBuilder (sync, budget-bounded)
                    │             │         └─ calls ConceptGrounder.ground_concept()
                    │             │
                    │             │─────── PatternCompleter (sync, FORMING stage)
                    └─────────────┘

WorkerPool lanes:
  infer  ──── LLMWorker (existing)
  review ──── ConceptGrounder._review_concept()
              ConceptExtractor._form_inline_relationships() (Phase 2)
  record ──── ConceptGrounder._apply_updates()
```

---

## Expected Impact

| Metric | Before | After |
|---|---|---|
| ConceptGrounder.ground_concept() latency | 5-50ms (graph-dependent) | <2ms (cache hit) or ~5ms (stats only) |
| 50ms budget utilization | Often exhausted by 2-3 concepts | Can ground 10+ concepts per call |
| Relationship freshness | Updated only during recall (sync) | Continuously updated in background |
| AG quantification latency | Blocks recall path | Background, never blocks |

---

## Migration & Backwards Compatibility

- **No WorkerPool available:** ConceptGrounder falls back to sync
  `_modulate_relationships()` + `_store_quantifications()` inline (existing
  behavior preserved)
- **No code changes needed** in ConceptContextBuilder, PatternCompleter, or
  any downstream consumers — they see the same `ground_concept()` return value
- **Existing tests** continue to pass because they don't provide a WorkerPool
  (sync fallback)
- **New tests** verify async path with mock WorkerPool
