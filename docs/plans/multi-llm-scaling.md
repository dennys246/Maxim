# Multi-LLM Scaling Plan (WorkerPool Phase 5)

> **Status:** Not started. Only useful when running on multi-GPU hardware
> or using heterogeneous models.

---

## Goal

Enable multiple LLM backends on different GPUs within the same WorkerPool,
supporting heterogeneous model assignments (large model for planning,
small model for evaluation).

---

## Prerequisites

- WorkerPool core infrastructure (Phases 1-3): **done**
- Review/record lane integration (Phase 4): **done**
- Hardware with multiple GPUs or need for heterogeneous models

---

## Changes Needed

### LaneConfig GPU assignment

Add `gpu_id: int | None` to `LaneConfig` for explicit GPU targeting:

```python
LaneConfig("infer-planning", max_workers=1, requires_gpu=True, gpu_id=0)  # large model
LaneConfig("infer-review", max_workers=1, requires_gpu=True, gpu_id=1)    # smaller model
```

### Thread-local GPU assignment

In lane executor init, set `CUDA_VISIBLE_DEVICES` per-thread so each
lane's workers target a specific GPU.

### Heterogeneous model backends

Support different model backends per lane. Each infer lane can use a
different LLM (e.g., large model for planning, small model for evaluation).

### Config-driven lane definitions

Load lane configurations from `config.yaml` instead of hardcoding in
LLMWorker. Enables runtime configuration without code changes.

### Metrics

Per-lane observability:
- Throughput (jobs/sec)
- Queue depth (pending jobs)
- Dependency wait time (gate latency)
- GPU utilization per lane

---

## Files to Modify

- `src/maxim/runtime/worker_pool.py` — `LaneConfig.gpu_id`, thread-local GPU setup
- `src/maxim/agents/llm_worker.py` — Config-driven lane creation, multi-model support
- `config.yaml` schema — Lane definitions section

---

## Concurrency Reference

| Component | Interaction with WorkerPool |
|---|---|
| `ConcurrentExecutor` | Orthogonal — parallel tool execution |
| `RWLock` (hippocampus) | Safe — single writer thread via capture queue |
| `AgentBus` | Already thread-safe |
| `PlanManager._lock` | Already safe — RLock allows reentrant reads |
| `ThreadRegistry` | Lane dispatchers registered in `WorkerPool.start()` |
