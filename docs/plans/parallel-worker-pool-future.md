# Parallel Worker Pool — Remaining Work

> **Status:** Core infrastructure (Phases 1-3) fully implemented in
> `src/maxim/runtime/worker_pool.py`. Phase 4 (review/record lanes for ATL
> concept memory) implemented in ConceptGrounder. This document tracks the
> one remaining phase.

---

## What's Done

- `Job`, `JobStatus`, `JobRegistry` with GC and `_completed_ids` safety
- `DependencySpec`, `DependencyGate` with two-phase prefetch (early/late)
- `LaneConfig`, `Lane` with priority queue + monotonic tiebreaker + gate-watcher threads
- `WorkerPool` with start/stop/submit/wait_for/cancel_lane/cancel_all/status
- `LLMWorker` integration: infer lane + infer_net lane for cloud providers
- **Phase 4:** ConceptGrounder wired to review/record lanes with dedup cooldown,
  fallback to sync when no pool available. See `worker-pool-concept-memory-integration.md`.
- Comprehensive unit tests in `tests/unit/test_worker_pool.py`,
  `test_llm_worker_pool.py`, and `test_concept_grounder_async.py`

---

## Phase 5: Multi-LLM Scaling

**When useful:** When running on hardware with multiple GPUs, or when using
heterogeneous models (large model for planning, small model for evaluation).

**Needs:**
- Add `gpu_id: int | None` to `LaneConfig`
- Thread-local GPU assignment in lane executor init (`CUDA_VISIBLE_DEVICES`)
- Support heterogeneous model backends per lane
- Config-driven lane definitions (load from `config.yaml`)
- Metrics: per-lane throughput, queue depth, dep-wait time

```python
# Example: heterogeneous models
LaneConfig("infer-planning", max_workers=1, requires_gpu=True)   # large model
LaneConfig("infer-review", max_workers=1, requires_gpu=True)     # smaller model
```

---

## Concurrency Reference

| Component | Interaction with WorkerPool |
|---|---|
| `ConcurrentExecutor` | Orthogonal — parallel tool execution |
| `RWLock` (hippocampus) | Safe — single writer thread via capture queue |
| `AgentBus` | Already thread-safe |
| `PlanManager._lock` | Already safe — RLock allows reentrant reads |
| `ThreadRegistry` | Lane dispatchers registered in `WorkerPool.start()` |
