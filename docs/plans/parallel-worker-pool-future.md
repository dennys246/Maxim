# Parallel Worker Pool — Remaining Work

> **Status:** Core infrastructure (Phase 1-3) is fully implemented and tested in
> `src/maxim/runtime/worker_pool.py`. The `infer` lane is wired into `LLMWorker`.
> This document tracks the remaining phases.

---

## What's Done

- `Job`, `JobStatus`, `JobRegistry` with GC and `_completed_ids` safety
- `DependencySpec`, `DependencyGate` with two-phase prefetch (early/late)
- `LaneConfig`, `Lane` with priority queue + monotonic tiebreaker + gate-watcher threads
- `WorkerPool` with start/stop/submit/wait_for/cancel_lane/cancel_all/status
- `LLMWorker` integration: infer lane + infer_net lane for cloud providers
- Comprehensive unit tests in `tests/unit/test_worker_pool.py` and `test_llm_worker_pool.py`

All issues from the original plan (#1-#18) were addressed during implementation.

---

## Phase 4: Review + Record Lane Integration — ATL Concept Memory

> **Concrete workload identified.** See full plan:
> [worker-pool-concept-memory-integration.md](worker-pool-concept-memory-integration.md)

The review and record lanes will serve the ATL concept memory pipeline:

- **Review lane:** ConceptGrounder relationship modulation (Jaccard co-occurrence
  scoring) and quantification analysis — read-heavy computation currently blocking
  the 50ms recall budget.
- **Record lane:** Applying computed changes — ATL edge updates, AG MathMemory
  stores, CrossLayerGraph QUANTIFIES edges, concept ref updates.

This moves graph-heavy analysis off the synchronous recall path, freeing the
50ms budget for loading episodes and collecting context (10+ concepts per call
instead of 2-3).

**Other potential review/record consumers (future):**
- Post-action plan phase evaluation
- Safety/guardrail checking on LLM outputs
- Async telemetry/metrics persistence
- Deferred media I/O (screenshots, audio clips)

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
