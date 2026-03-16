# Worker Pool × ATL Concept Memory Integration

> **Status:** Phase 1 (core async pipeline) and Phase 3 (dedup cooldown)
> are **IMPLEMENTED**. Only Phase 2 (optional ConceptExtractor offload) remains.

---

## What's Done

**Phase 1 — Background Relationship Modulation** (implemented):
- `ConceptGrounder` accepts optional `worker_pool` parameter
- `ground_concept()` returns stats immediately; heavy work enqueued to review lane
- Review lane: `_review_concept()` computes Jaccard edge updates + quantification proposals
- Record lane: `_apply_updates()` applies edge changes + AG MathMemory stores
- Fallback: runs inline sync when no WorkerPool available
- `MemoryHub` passes `worker_pool` field through to ConceptGrounder
- Tests: 19 in `tests/unit/test_concept_grounder_async.py`

**Phase 3 — Dedup Cooldown** (implemented as part of Phase 1):
- `_pending_reviews` dict with `REVIEW_COOLDOWN_S = 2.0` prevents redundant enqueues
- Pending flag cleared on execution start and on submit failure

---

## Phase 2: ConceptExtractor Relationship Offload (optional)

**Goal:** Move `_form_inline_relationships()` from ConceptExtractor's worker
thread to the review lane, unifying all relationship work in one pipeline.

**Why optional:** ConceptExtractor already runs in a background thread. The
benefit is unification (one pipeline for all relationship work) rather than
performance. ConceptExtractor's own queue handles backpressure fine.

**Changes (~20 lines):**
- Add `worker_pool` parameter to `ConceptExtractor.__init__()`
- In `_process_capture()`, submit `_form_inline_relationships()` to review lane
  when pool available, else run inline (existing behavior)
- Wire pool through `MemoryHub._wire_multi_layer()` → ConceptExtractor

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

No circular lock dependencies. Review lane only reads. Record lane writes to
ATL, AG, and CrossLayerGraph — all independent locks.

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
                    │             │─────── ConceptGrounder (sync stats + async review) ✅
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
  review ──── ConceptGrounder._review_concept() ✅
              ConceptExtractor._form_inline_relationships() (Phase 2)
  record ──── ConceptGrounder._apply_updates() ✅
```
