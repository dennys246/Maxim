# Memory System Interactions — Runtime Analysis

This document maps how Maxim's memory systems interact with each other and
the agent loop during active operation. It corrects earlier assumptions
about when each system is accessed and documents the threading/locking model.

---

## Memory Systems Overview

| System | Type | Lock | Location |
|--------|------|------|----------|
| **Hippocampus** | Episodic memory | `RWLock` (writer-priority) | `src/maxim/memory/hippocampus.py` |
| **ATL** | Semantic concepts | `RWLock` | `src/maxim/memory/atl.py` |
| **Angular Gyrus** | Math knowledge | `RWLock` | `src/maxim/math/angular_gyrus.py` |
| **CrossLayerGraph** | Inter-layer links | Thread-safe dict | `src/maxim/memory/cross_layer.py` |
| **SemanticPromoter** | Hippocampus → ATL | No own lock (uses ATL's) | `src/maxim/memory/semantic_promoter.py` |
| **EC / NeuralEmbedder** | Semantic embeddings | Own thread pool queue | `src/maxim/memory/ec.py` |
| **MemoryHub** | Coordinator | No own lock (delegates) | `src/maxim/integration/memory_hub.py` |
| **ContextPool** | Growing LLM context | `RLock` | `src/maxim/agents/context_pool.py` |
| **MemoryAgent** | Agent-level memory | Own `_association_index`, `_association_graph` | `src/maxim/agents/memory_agent.py` |

All three core memory layers (Hippocampus, ATL, Angular Gyrus) implement
the `MemoryLayer` ABC (`src/maxim/memory/layer.py`) which provides a shared
protocol: `store`, `get`, `remove`, `recall`, `recall_associated`, `graph`,
`save`, `load`, `consolidate`.

---

## Access Patterns by Phase

### Phase 1: Session Start (`MemoryHub.on_session_start()`)

This is the only time bridges do heavy hippocampus reads.

```
MemoryHub.on_session_start()
  ├─ hippocampus.load()                    [READ disk → memory]
  ├─ spatial_bridge.on_session_start()
  │   ├─ hippocampus.recall(success=True)  [READ — 500 memories]
  │   ├─ hippocampus.recall_associated()   [READ — spreading activation]
  │   └─ [builds internal location cache]
  ├─ salience_bridge.on_session_start()
  │   ├─ hippocampus.recall(success=True)  [READ — 500 memories]
  │   ├─ hippocampus.recall(success=False) [READ — 200 memories]
  │   ├─ hippocampus.recall_associated()   [READ — spreading activation]
  │   └─ [builds internal interaction history]
  ├─ planning_bridge.on_session_start()    [similar pattern]
  ├─ fear_bridge.on_session_start()        [similar pattern]
  └─ escalation_bridge.on_session_start()  [similar pattern]
```

After session start, **bridges serve from their internal caches** and do
NOT re-query hippocampus during the active loop.

### Phase 2: Active Loop (30Hz agent loop)

#### What the agent loop does directly

```
run_agentic_loop() — per tick:
  ├─ PERCEPTION: capture frames, run YOLO, build Percept
  ├─ CHECK LLM PROPOSALS: poll llm_worker for new LLMProposal
  ├─ AGENT FALLBACK: if no LLM, use DefaultNetwork
  ├─ EXECUTE ACTION: run tool via environment
  │   └─ POST-EXECUTION (after tool runs):
  │       ├─ context_pool.add_outcome()           [WRITE — sub-ms, RLock]
  │       ├─ llm_worker.record_outcome()          [WRITE — sub-ms]
  │       ├─ recent_outcomes.append()              [WRITE — sub-ms, no lock]
  │       ├─ memory_hub.record_plan_outcome()      [WRITE — sub-ms]
  │       └─ hippocampus.capture_from_loop()       [WRITE — 10-200ms, write lock]
  │           ├─ store memory
  │           ├─ build index
  │           ├─ check promotion thresholds
  │           ├─ _form_associations()              [SLOW: O(n) scan of all memories]
  │           └─ fire capture callbacks
  │               └─ _on_memory_captured()
  │                   └─ ec.schedule_embedding()   [async — own queue]
  │
  ├─ BUILD CONTEXT: memory.build_context()
  │   ├─ _get_relevant_memories()                 [uses own _association_index, NOT hippocampus]
  │   └─ _build_knowledge_context()
  │       ├─ hub.atl.recall(limit=5)              [READ ATL — own RWLock]
  │       ├─ hub.atl.find_by_relationship()       [READ ATL — own RWLock]
  │       └─ hub.angular_gyrus.recall()           [READ AG — own RWLock]
  │
  ├─ SUBMIT TO LLM: llm_worker.submit_context()
  └─ LOOP TIMING: sleep to maintain 30Hz
```

#### What bridges/math_bridge do (called from memory_agent or tools)

```
math_bridge.enrich_context(goal):         [called per-tick by memory_agent]
  └─ angular_gyrus.recall_method(goal)    [READ AG — own RWLock]

math_bridge.promote_patterns():           [called periodically]
  ├─ angular_gyrus.recall(name=key)       [READ AG]
  └─ angular_gyrus.store(record)          [WRITE AG — own RWLock]
```

#### What MemoryHub exposes for active-loop queries

```
MemoryHub.recall_concepts()               → atl.recall()           [READ ATL]
MemoryHub.recall_with_knowledge()         → cross_layer_activation() [READ CrossLayer]
MemoryHub.enrich_salience(detections)     → salience_bridge cache   [READ cache, no hippocampus]
MemoryHub.get_plan_templates(goal)        → planning_bridge cache   [READ cache, no hippocampus]
MemoryHub.get_spatial_boosts(goal)        → spatial_bridge cache    [READ cache, no hippocampus]
MemoryHub.get_escalation_threshold()      → escalation_bridge cache [READ cache, no hippocampus]
```

### Phase 3: Session End (`MemoryHub.on_session_end()`)

```
MemoryHub.on_session_end()
  ├─ hippocampus.flush()                  [drain async capture queue if passive]
  ├─ hippocampus.sleep()                  [consolidation: compress, prune, promote]
  │   ├─ _compress_old_memories()
  │   ├─ _remove_low_retention()
  │   └─ _promote_to_long_term()
  ├─ promoter.scan_for_promotions()       [hippocampus patterns → ATL concepts]
  │   └─ atl.store(concept)              [WRITE ATL]
  ├─ atl.consolidate()                    [compress/prune ATL]
  ├─ ag.consolidate()                     [compress/prune AG]
  ├─ hippocampus.save()                   [persist to disk]
  ├─ atl.save()                           [persist to disk]
  ├─ ag.save()                            [persist to disk]
  └─ bridge.on_session_end() for each     [cleanup]
```

---

## Threading Model

### Active threads during the loop

| Thread | Owner | Purpose | Hippocampus access |
|--------|-------|---------|-------------------|
| Main loop | `agent_loop.py` | 30Hz perception-decision-action | WRITE (capture) |
| LLM worker | `LLMWorker` | Process LLM requests sequentially | None |
| DefaultNetwork | `default_network.py` | 30Hz reactive fallback | None |
| Capture thread | `capture.py` | Camera frame capture | None |
| Segmentation | `segmentation.py` | YOLO inference | None |
| Audio | `audio.py` | Whisper transcription | None |
| EC embedder | `NeuralEmbedder` | Async semantic embeddings | None (triggered by callback) |
| Hippocampus worker | (proposed) | Async capture processing | WRITE (exclusive) |

### Lock contention analysis

```
Hippocampus RWLock:
  Writers: agent_loop capture (or proposed async worker)
  Readers: bridge.on_session_start() ONLY (before loop starts)
  → Zero read-write contention during active loop

ATL RWLock:
  Writers: SemanticPromoter at session end; never during active loop
  Readers: memory_agent._build_knowledge_context() per-tick
  → Zero contention during active loop (readers only)

Angular Gyrus RWLock:
  Writers: math_bridge.promote_patterns() (infrequent, pattern-triggered)
  Readers: math_bridge.enrich_context() per-tick
  → Minimal contention (writes are rare, RWLock allows concurrent reads)

ContextPool RLock:
  Writers: context_pool.add_outcome(), add_conversation_turn()
  Readers: context_pool.get_context_text() during LLM submission
  → RLock serializes; all operations are sub-millisecond
```

---

## Data Flow Diagram

```
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                        Agent Loop (Main Thread)                 │
                    │                                                                 │
                    │  Perception ──► Decision ──► Action ──► Post-Execution          │
                    │                                              │                  │
                    │                                   ┌──────────┼──────────┐       │
                    │                                   │ sync     │ async    │       │
                    │                          context_pool  hippocampus.capture│       │
                    │                          llm_worker     │ (proposed)     │       │
                    │                          recent_outcomes│               │       │
                    │                          plan_outcome   │               │       │
                    └─────────────────────────────────────────│───────────────────────┘
                                                              │
                                                              ▼
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                    Hippocampus (own thread)                     │
                    │                                                                 │
                    │  Capture Queue ──► store ──► index ──► associations ──► callback│
                    │                                                          │      │
                    │                                                          ▼      │
                    │                                              EC schedule_embed  │
                    │                                              bridge.on_capture  │
                    └─────────────────────────────────────────────────────────────────┘
                                                              │
                                         ┌────────────────────┼─────────────────┐
                                         ▼                    ▼                 ▼
                                   ┌──────────┐        ┌──────────┐     ┌──────────────┐
                                   │   ATL    │        │    AG    │     │ CrossLayer   │
                                   │ (read by │        │ (read/  │     │ (read by     │
                                   │ mem_agent│        │ write by│     │ MemoryHub    │
                                   │ per-tick)│        │ math_   │     │ queries)     │
                                   │          │        │ bridge) │     │              │
                                   └──────────┘        └──────────┘     └──────────────┘
                                        ▲                                      ▲
                                        │                                      │
                                        └──────── SemanticPromoter ────────────┘
                                                 (session-end only)
```

---

## Capture Callback Chain

When hippocampus processes a capture, the following callback chain fires:

```
hippocampus.capture() completes
  │
  ├─► _on_memory_captured callbacks (registered via register_capture_callback)
  │     │
  │     ├─► MemoryHub._on_memory_captured()
  │     │     └─► ec._neural_embedder.schedule_embedding(memory_id, text)
  │     │         └─► [async: embed and store in EC's vector index]
  │     │
  │     └─► (proposed) bridge.update_from_capture() — reactive cache updates
  │
  └─► _on_memory_deleted callbacks (if consolidation removes memories)
        ├─► SCN.remove_memory(memory_id) — remove temporal index entry
        ├─► NAc.remove_prediction(memory_id) — remove reward prediction
        └─► CrossLayerGraph deletion callback — remove cross-layer edges
```

---

## Association Formation (the slow part)

The main bottleneck in hippocampus capture is `_form_associations()`:

```python
# hippocampus.py L1170
def _form_associations(self, memory_id, memory):
    # 1. Find similar memories (O(n) scan)
    similar = self._recall_similar_unlocked(
        perception=memory.perception,
        limit=self.config.association_limit,     # default: 5
        threshold=self.config.association_threshold,  # default: 0.5
    )

    # 2. For each similar memory, create bidirectional ASSOCIATES edge
    for similar_id, score in similar:
        weight = (
            0.6 * perceptual_overlap +
            0.25 * goal_overlap +
            0.15 * temporal_proximity
        )
        self._graph.add_edge(Edge(
            source=memory_id, target=similar_id,
            edge_type=EdgeType.ASSOCIATES, weight=weight,
        ))
        self._graph.add_edge(Edge(
            source=similar_id, target=memory_id,
            edge_type=EdgeType.ASSOCIATES, weight=weight,
        ))
```

`_recall_similar_unlocked()` scores memories by overlapping detected
objects and people — it's an O(n) scan where n = total memory count.
With 1000+ memories this is the dominant cost (50-200ms).

This is the primary reason to make hippocampus capture async: the main
loop shouldn't wait for this O(n) scan.

---

## Key Invariants

1. **Bridges never query hippocampus during the active loop.** They load at
   session start and serve from cache. Async hippocampus capture cannot
   cause bridge staleness during the loop.

2. **ATL and AG have independent locks.** Their reads (per-tick by
   memory_agent and math_bridge) never contend with hippocampus writes.

3. **CrossLayerGraph edges are only created at session end** (by
   SemanticPromoter). Active-loop `cross_layer_activation()` calls are
   read-only against a stable graph.

4. **The only hippocampus write during the active loop is `capture_from_loop()`.**
   There are no concurrent hippocampus writes from other subsystems.

5. **EC embedding is already async.** The capture callback schedules it
   on EC's own thread pool. Making hippocampus capture async just moves
   *where* the callback fires (worker thread vs main thread), not *how*.

6. **memory_agent uses its own association structures** (`_association_index`,
   `_association_graph`) for per-tick relevant memory lookup. These are
   NOT the hippocampus DependencyGraph — they're separate, lighter-weight
   data structures local to the memory agent.

---

## Common Misconceptions (Corrected)

| Misconception | Reality |
|---------------|---------|
| "ATL/AG/CrossLayer are session-end-only" | ATL and AG are READ per-tick (by memory_agent and math_bridge). AG is also WRITTEN by math_bridge pattern promotion. CrossLayerGraph is read by MemoryHub queries. Only consolidation/promotion is session-end. |
| "Bridges read hippocampus every tick" | Bridges read hippocampus ONCE at session start, then serve from internal caches. No per-tick hippocampus reads. |
| "memory_agent queries hippocampus for relevant memories" | memory_agent queries its own `_association_index` and `_association_graph`, NOT the hippocampus. These are separate data structures. |
| "Async hippocampus would break bridge reads" | No contention — bridges already finished reading before the loop starts. |
| "Async hippocampus would break ATL/AG queries" | No relation — ATL/AG have independent locks and don't depend on hippocampus write timing. |
