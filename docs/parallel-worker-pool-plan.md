# Parallel Worker Pool — Implementation Plan

## Motivation

The agentic loop currently serializes all cognitive operations through a single
`LLMWorker` thread: planning, review, record, and inference share one queue.
This means the system cannot plan the next phase while recording the outcome of
the last one, or review a past failure while the LLM is generating a new
proposal.

This plan introduces a **worker pool with typed lanes and dependency gates** —
allowing operations to run in parallel, declare prerequisites ("wait for X
before starting"), and prefetch data while blocked on those prerequisites.

---

## Architecture Overview

```
                        ┌──────────────────────────────────┐
                        │         WorkerPool               │
                        │  (owns threads, schedules jobs)  │
                        └──────┬───────────────────────────┘
                               │
            ┌──────────────────┼──────────────────────┐
            │                  │                      │
     ┌──────▼──────┐   ┌──────▼──────┐   ┌───────────▼──────┐
     │  InferLane   │   │ ReviewLane  │   │   RecordLane     │
     │  (LLM calls) │   │ (evaluate)  │   │  (memory write)  │
     │  max_workers  │   │ max_workers │   │  max_workers     │
     │  = gpu_count  │   │ = 1 (cpu)   │   │  = 2 (I/O)       │
     └──────┬──────┘   └──────┬──────┘   └──────┬───────────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                        ┌──────▼──────┐
                        │  DependencyGate  │
                        │  (wait + prefetch) │
                        └─────────────┘
```

### Key concepts

| Concept | What it is |
|---------|-----------|
| **Lane** | A named category of work (infer, review, record). Each lane has its own bounded thread pool and priority queue. |
| **Job** | A unit of work submitted to a lane. Carries a callable, priority, and an optional `DependencySpec`. |
| **DependencyGate** | A per-job object that blocks execution until prerequisite jobs finish — but allows **prefetch work** to run while blocked. |
| **PrefetchHook** | A callable attached to a job that runs *immediately* on submission, even before dependencies resolve. Gathers data the job will need (context pool snapshot, hippocampus recall, file reads). |

---

## Part 1 — DependencyGate & PrefetchHook

### 1.1 DependencySpec

> **Updated per Issue #1:** Split prefetch into two phases to avoid stale data.

```python
# src/maxim/runtime/worker_pool.py

@dataclass
class DependencySpec:
    """Declares what a job needs before it can execute."""

    # Job IDs that must complete before this job starts
    wait_for: list[str] = field(default_factory=list)

    # Runs immediately on submit — gather stable/expensive data
    # (agent states, tool descriptions, mode info)
    prefetch_early: Callable[[], Any] | None = None

    # Runs AFTER deps resolve — gather dep-dependent data (should be fast)
    # (context_pool snapshot with fresh outcome, reasoning carryover)
    prefetch_late: Callable[[], Any] | None = None

    # Timeout — if deps don't resolve in time, job runs anyway
    # with whatever prefetch data is available
    timeout_s: float = 30.0
```

**Why prefetch?** In the common case (per Revised Recommendations),
the infer job does NOT wait for hippocampus capture — `context_pool` is
updated synchronously on the main thread. But prefetch is still valuable
for gathering expensive *inputs* to prompt building (agent states, tool
descriptions, conversation history) while the infer lane queue drains.
For the special cases that DO need dependency gates (plan phase
transitions, multi-step chains), `prefetch_early` gathers stable data
during the wait, and `prefetch_late` grabs fresh post-dep data cheaply.

### 1.2 DependencyGate (the lock mechanism)

```python
class DependencyGate:
    """Blocks a job until its prerequisites complete. Runs early prefetch immediately."""

    def __init__(self, spec: DependencySpec, registry: JobRegistry):
        self._spec = spec
        self._registry = registry
        self._early_result: Any = None
        self._early_done = threading.Event()
        self._failed_deps: list[str] = []

    def start_early_prefetch(self) -> None:
        """Kick off early prefetch in background. Called on job submission."""
        if self._spec.prefetch_early is None:
            self._early_done.set()
            return
        # Early prefetch runs in a short-lived thread so it doesn't
        # consume a lane worker slot (Issue #8: use shared pool in prod)
        threading.Thread(
            target=self._run_early_prefetch,
            daemon=True,
            name="prefetch-early",
        ).start()

    def _run_early_prefetch(self) -> None:
        try:
            self._early_result = self._spec.prefetch_early()
        except Exception as e:
            logger.warning("Early prefetch failed: %s", e)
        finally:
            self._early_done.set()

    def wait(self) -> tuple[dict[str, Any], list[str]]:
        """Block until all deps resolve OR timeout.

        Returns ({"early": ..., "late": ...}, failed_deps).
        """
        deadline = time.monotonic() + self._spec.timeout_s

        # Wait for each dependency
        for job_id in self._spec.wait_for:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._failed_deps.append(job_id)
                continue
            if not self._registry.wait_for_completion(job_id, timeout=remaining):
                self._failed_deps.append(job_id)

        # Ensure early prefetch is done
        self._early_done.wait(timeout=max(0, deadline - time.monotonic()))

        # Run late prefetch NOW (deps have resolved, data is fresh)
        late_result = None
        if self._spec.prefetch_late:
            try:
                late_result = self._spec.prefetch_late()
            except Exception as e:
                logger.warning("Late prefetch failed: %s", e)

        return {"early": self._early_result, "late": late_result}, self._failed_deps
```

### 1.3 Usage example — passive hippocampus + infer

> **Updated per Issues #1, #4, #12, and Passive Hippocampus Design.**
> Hippocampus capture uses its own async queue (not the record lane).
> Fast record operations stay synchronous. Data is snapshotted at
> submission time to avoid closure bugs.

```python
# After tool execution in the agent loop:

# 1. Fast synchronous records (sub-ms, stay on main thread):
recent_outcomes.append(outcome)
llm_worker.record_outcome(result_str, action)
context_pool.add_outcome(result_str, action)

# 2. Async hippocampus capture (own queue, fire-and-forget):
hippocampus.capture_from_loop_async(
    observation=observation,
    state=state,
    intent=intent,
    decision=decision,
    action=action_data,
    result=result,
    run_id=run_id,
)

# 3. Submit LLM inference (no dep on hippocampus — context already updated):
_agent_states = agent.get_agent_states()  # snapshot now
pool.submit(
    lane="infer",
    job_id=f"infer-{ts}",
    fn=partial(llm_worker._process_request, request),
    priority=0,
    deps=DependencySpec(
        prefetch_early=lambda: {
            "agent_states": _agent_states,
            "tool_descriptions": get_tool_descriptions(),
        },
        prefetch_late=lambda: {
            "context_text": context_pool.get_context_text(),
        },
    ),
)
```

The fast records complete inline. Hippocampus capture runs on its own
background thread. The LLM submission doesn't wait for hippocampus — the
context pool is already up to date from the synchronous `add_outcome()`.

---

## Part 2 — Lane & WorkerPool

### 2.1 Lane

```python
@dataclass
class LaneConfig:
    name: str
    max_workers: int
    queue_size: int = 10
    # Whether this lane needs GPU access (for future multi-GPU routing)
    requires_gpu: bool = False

class Lane:
    """A typed worker pool for a category of work."""

    _counter = itertools.count()  # monotonic tiebreaker (Issue #3)

    def __init__(self, config: LaneConfig, registry: JobRegistry):
        self._config = config
        self._registry = registry
        self._queue: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=config.queue_size
        )
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix=f"Lane-{config.name}",
        )
        self._stop = threading.Event()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name=f"Lane-{config.name}-dispatch",
        )

    def submit(self, job: Job) -> None:
        """Enqueue a job. Starts early prefetch immediately."""
        if job.deps:
            job.gate = DependencyGate(job.deps, self._registry)
            job.gate.start_early_prefetch()
        self._registry.register(job)
        self._queue.put((job.priority, next(self._counter), job))  # Issue #3: tiebreaker

    def _dispatch_loop(self) -> None:
        """Dequeue jobs; wait for deps in lightweight threads to avoid
        blocking worker slots (Issue #2)."""
        while not self._stop.is_set():
            try:
                _, _, job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job.gate:
                # Dep wait runs in a short-lived thread so the dispatcher
                # can keep dequeuing. When deps resolve, the job is
                # submitted to the executor.
                threading.Thread(
                    target=self._wait_then_dispatch,
                    args=(job,),
                    daemon=True,
                    name=f"gate-{job.job_id}",
                ).start()
            else:
                self._executor.submit(self._execute_job, job)

    def _wait_then_dispatch(self, job: Job) -> None:
        """Wait for deps in a gate-watcher thread, then hand off to executor."""
        prefetched, failed = job.gate.wait()
        if failed:
            logger.warning(
                "Job %s: deps timed out: %s (running anyway)",
                job.job_id, failed,
            )
        job._prefetched = prefetched
        self._executor.submit(self._execute_job, job)

    def _execute_job(self, job: Job) -> None:
        """Run the job callable. Deps already resolved by dispatcher."""
        self._registry.mark_running(job.job_id)
        prefetched = getattr(job, "_prefetched", None)
        try:
            result = job.fn(prefetched=prefetched) if prefetched else job.fn()
            self._registry.mark_completed(job.job_id, result=result)
        except Exception as e:
            logger.error("Job %s failed: %s", job.job_id, e)
            self._registry.mark_failed(job.job_id, error=e)
```

### 2.2 WorkerPool

```python
# Default lane configuration
DEFAULT_LANES = {
    "infer": LaneConfig(name="infer", max_workers=1, requires_gpu=True),
    "review": LaneConfig(name="review", max_workers=1),
    "record": LaneConfig(name="record", max_workers=2),
}

class WorkerPool:
    """Central pool that owns all lanes and the job registry."""

    def __init__(
        self,
        lane_configs: dict[str, LaneConfig] | None = None,
        thread_registry: ThreadRegistry | None = None,
    ):
        self._registry = JobRegistry()
        configs = lane_configs or DEFAULT_LANES
        self._lanes: dict[str, Lane] = {
            name: Lane(cfg, self._registry)
            for name, cfg in configs.items()
        }
        self._thread_registry = thread_registry

    def submit(
        self,
        lane: str,
        job_id: str,
        fn: Callable,
        priority: int = 5,
        deps: DependencySpec | None = None,
    ) -> str:
        """Submit a job to a lane. Returns job_id."""
        job = Job(job_id=job_id, fn=fn, priority=priority, deps=deps)
        self._lanes[lane].submit(job)
        return job_id

    def start(self) -> None:
        for lane in self._lanes.values():
            lane.start()
            if self._thread_registry:
                self._thread_registry.register(
                    f"Lane-{lane._config.name}", lane._dispatcher
                )

    def stop(self, timeout: float = 5.0) -> None:
        for lane in self._lanes.values():
            lane.stop(timeout=timeout / len(self._lanes))

    def wait_for(self, job_id: str, timeout: float = 30.0) -> bool:
        """Block until a specific job completes."""
        return self._registry.wait_for_completion(job_id, timeout)

    def get_result(self, job_id: str) -> Any:
        """Get the result of a completed job."""
        return self._registry.get_result(job_id)

    def get_completed(self, lane: str) -> Job | None:
        """Non-blocking poll: return the most recent completed job for a lane, or None."""
        return self._registry.pop_completed(lane)
```

### 2.3 JobRegistry

```python
class JobRegistry:
    """Thread-safe registry tracking job lifecycle and enabling cross-job waits."""

    def __init__(self):
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._results: dict[str, Any] = {}
        self._errors: dict[str, Exception] = {}
        self._completed_ids: set[str] = set()  # survives prune() for dep safety

    def register(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.job_id] = JobStatus.PENDING
            self._events[job.job_id] = threading.Event()

    def wait_for_completion(self, job_id: str, timeout: float) -> bool:
        """Block until job completes. Returns False on timeout."""
        with self._lock:
            event = self._events.get(job_id)
        if event is None:
            return True  # unknown job = assume done
        return event.wait(timeout=timeout)

    def mark_completed(self, job_id: str, result: Any = None) -> None:
        with self._lock:
            self._jobs[job_id] = JobStatus.COMPLETED
            self._results[job_id] = result
            self._completed_ids.add(job_id)
            event = self._events.get(job_id)
        if event:
            event.set()  # wake up anyone waiting

    def mark_failed(self, job_id: str, error: Exception) -> None:
        with self._lock:
            self._jobs[job_id] = JobStatus.FAILED
            self._errors[job_id] = error
            event = self._events.get(job_id)
        if event:
            event.set()  # still wake waiters — they check status

    def pop_completed(self, lane: str) -> Job | None:
        """Return and remove the most recent completed job for a lane, or None."""
        with self._lock:
            for job_id, status in list(self._jobs.items()):
                if status == JobStatus.COMPLETED and job_id.startswith(lane):
                    result = self._results.pop(job_id, None)
                    del self._jobs[job_id]
                    return Job(job_id=job_id, fn=None, result=result)
        return None

    # GC: periodically prune completed jobs older than N seconds
    def prune(self, max_age_s: float = 300.0) -> int: ...
```

---

## Part 3 — Integration with Agent Loop

### 3.1 Replacing the single LLMWorker

The existing `LLMWorker` becomes the backend for the `infer` lane. We don't
rewrite it — we wrap it:

```python
# In run_agentic_loop() setup:
pool = WorkerPool(
    lane_configs={
        "infer": LaneConfig("infer", max_workers=gpu_count, requires_gpu=True),
        "review": LaneConfig("review", max_workers=1),
        "record": LaneConfig("record", max_workers=2),
    },
    thread_registry=thread_registry,
)
pool.start()
```

The main loop polling changes from:
```python
# OLD: poll single LLMWorker
new_proposal = llm_worker.get_latest_proposal()
```

To:
```python
# NEW: poll infer lane for completed jobs
completed_infer = pool.get_completed("infer")
if completed_infer:
    new_proposal = completed_infer.result
```

### 3.2 Hippocampus capture moves to its own thread

> **Updated per Passive Hippocampus Design.** Hippocampus uses its own
> capture queue rather than the worker pool's record lane. See the
> "Passive Hippocampus Design" section for full implementation.

Currently (agent_loop.py ~L1264):
```python
# Synchronous — blocks the 30Hz loop
hippocampus.capture_from_loop(episode_data)
```

Becomes:
```python
# Non-blocking — queued for background processing
hippocampus.capture_from_loop_async(
    observation=observation,
    state=state,
    intent=intent,
    decision=decision,
    action=action_data,
    result=result,
    run_id=run_id,
)
# Main loop continues immediately
```

### 3.3 Review as a background lane (deferred)

> **Deferred per Revised Recommendations (Issue #18).** The review lane's
> workload is underspecified. Implement after a concrete evaluation use
> case emerges. If review ends up LLM-powered, it should be a job type
> in the infer lane with lower priority, not a separate lane.

Example of what this *could* look like when defined:

```python
def review_outcome(prefetched=None):
    """Evaluate tool outcome against plan expectations."""
    phase = plan_manager.get_current_phase_snapshot()  # thread-safe (Issue #11)
    if not phase:
        return None
    evaluation = evaluate_phase_progress(
        phase=phase,
        outcome=tool_result,
        context=prefetched["late"]["context"] if prefetched else None,
    )
    bus.publish(ReviewCompleted(
        phase_id=phase.id,
        evaluation=evaluation,
    ))
    return evaluation
```

### 3.4 Revised execution flow

> **Updated to reflect passive hippocampus + deferred review lane.**

```
Time ──────────────────────────────────────────────────────►

Main loop:  [execute tool]──►[sync records]──►[queue hipp]──►[submit infer]──►[continue 30Hz]
                                   │               │                │
                                   │               │                │
Hippocampus thread:                │               ├─[store]────────│──────────►
                                   │               │  [index]       │
                                   │               │  [associate]   │
                                   │               │  [callbacks]   │
                                   │               │                │
Infer lane:                        │               │   [early prefetch]──►[LLM call]──►
                                   │               │                │
                                   │               │                │
                           context_pool updated    fire-and-forget  no dep wait needed
                           (sync, sub-ms)          (async)          (context already fresh)
```

The main loop does fast synchronous records, queues hippocampus capture
(returns immediately), and submits LLM inference — all without blocking.
No dependency gates needed in the common case because `context_pool` is
already updated before the infer job runs.

**When dependency gates ARE needed:**
- Plan phase transitions: review should evaluate against persisted hippocampus memory
- Sleep consolidation: `hippocampus.flush()` before `on_session_end()`
- Multi-step chains where step N+1's prompt explicitly needs step N's hippocampus data

---

## Part 4 — Configuration

```yaml
# config.yaml additions
worker_pool:
  lanes:
    infer:
      max_workers: 1        # bump to 2+ with bigger GPU
      queue_size: 5
      requires_gpu: true
    review:
      max_workers: 1
      queue_size: 10
    record:
      max_workers: 2
      queue_size: 20
  dependency_timeout_s: 30.0
  job_gc_interval_s: 60.0
  job_gc_max_age_s: 300.0
```

---

## Part 5 — Multi-LLM Future (scalability path)

The lane architecture makes multi-LLM straightforward:

```python
# With 2 GPUs:
LaneConfig("infer", max_workers=2, requires_gpu=True)

# Or heterogeneous models:
LaneConfig("infer-planning", max_workers=1, requires_gpu=True)   # large model for plans
LaneConfig("infer-review", max_workers=1, requires_gpu=True)     # smaller model for eval
```

Each lane worker can be backed by a different `LLMBackend` instance. The
`WorkerPool` doesn't care — it just dispatches callables. The lane config
determines which GPU/model is used.

For multi-GPU:
- Each `LaneConfig` gets a `gpu_id: int | None` field
- The lane's ThreadPoolExecutor sets `CUDA_VISIBLE_DEVICES` per-thread via
  thread-local init
- The `infer` lane with `max_workers=2` can saturate two GPUs simultaneously

---

## Part 6 — Implementation Phases

### Phase 1: Core primitives (no behavior change)
**Files:** `src/maxim/runtime/worker_pool.py` (new)

- [ ] `Job`, `JobStatus`, `JobRegistry` dataclasses
- [ ] `DependencySpec`, `DependencyGate` with prefetch support
- [ ] `LaneConfig`, `Lane` with dispatch loop
- [ ] `WorkerPool` with start/stop/submit/wait_for
- [ ] Unit tests for dependency resolution, prefetch timing, timeout behavior
- [ ] Register all lane threads with `ThreadRegistry` for coordinated shutdown

### Phase 2: Passive Hippocampus
**Files:** `src/maxim/memory/hippocampus.py`, `src/maxim/runtime/agent_loop.py`

- [ ] Add `_CaptureRequest` dataclass to hippocampus
- [ ] Add `_capture_queue`, `_worker_thread`, `_stop_event` to `__init__`
- [ ] Implement `_capture_worker()` loop (drain queue, call existing `capture()`)
- [ ] Implement `capture_from_loop_async()` with data snapshotting
- [ ] Implement timed `flush()` (with `all_tasks_done` wait)
- [ ] Implement `start()` / `stop()` with `ThreadRegistry` integration
- [ ] Replace `capture_from_loop()` calls in `agent_loop.py` (L1266, L1761) with `capture_from_loop_async()`
- [ ] Update shutdown path: `flush()` → `save()` → `on_session_end()`
- [ ] Add timing instrumentation: log queue latency + capture processing time
- [ ] Keep `context_pool.add_outcome()`, `llm_worker.record_outcome()`, `recent_outcomes.append()` synchronous on main loop (per Issue #12)
- [ ] Unit tests for: queue overflow + drop-oldest, flush timeout, concurrent captures, shutdown ordering

### Phase 3: Infer lane with dependency-aware submission
**Files:** `src/maxim/agents/llm_worker.py`, `src/maxim/runtime/agent_loop.py`

- [ ] Replace `LLMWorker._worker_loop` with infer lane dispatcher (keep `_process_request` intact, per Issue #16)
- [ ] `submit_context()` enqueues to infer lane instead of internal PriorityQueue
- [ ] Main loop polls `WorkerPool` for completed infer jobs instead of `llm_worker.get_latest_proposal()`
- [ ] Infer jobs use `prefetch_early` for expensive input gathering (agent states, tool descriptions)
- [ ] Infer jobs use `prefetch_late` for fresh `context_pool` text (per Issue #1)
- [ ] Only add `wait_for` deps for plan phase transitions, not routine followups (per Issue #15)
- [ ] Staleness guard moves to `WorkerPool` level (discard stale results)

### Phase 4: Review lane integration (deferred — see Issue #18)
**Files:** `src/maxim/runtime/agent_loop.py`, new `src/maxim/runtime/review_worker.py`

> **Deferred until a concrete evaluation workload is defined.** If review
> is LLM-powered, it should be a job type in the infer lane with lower
> priority, not a separate lane.

- [ ] Define concrete review callable (heuristic vs LLM-based vs statistical)
- [ ] If CPU-bound: create `ReviewWorker` with own lane
- [ ] If LLM-based: submit as lower-priority infer lane job
- [ ] Wire review results to `context_pool` and `ReasoningCarryover`
- [ ] Publish `ReviewCompleted` events on the `AgentBus`

### Phase 5: Multi-LLM scaling
**Files:** `src/maxim/runtime/worker_pool.py`, config

- [ ] Add `gpu_id` to `LaneConfig`
- [ ] Thread-local GPU assignment in lane executor init
- [ ] Support heterogeneous model backends per lane
- [ ] Config-driven lane definitions (load from `config.yaml`)
- [ ] Metrics: per-lane throughput, queue depth, dep-wait time

---

## Interaction with Existing Concurrency

| Existing component | Interaction | Action needed |
|---|---|---|
| `LLMWorker` thread | Absorbed into `infer` lane | Keep as callable, remove its own thread management |
| `ConcurrentExecutor` | Parallel *tool* execution (orthogonal) | No change — tools still run via `ConcurrentExecutor` |
| `RWLock` on hippocampus | Hippocampus worker thread writes exclusively | Single writer thread — no write contention. Bridges read only at session start (before worker starts). |
| `AgentBus` pub/sub | Review lane publishes events | Already thread-safe — handlers called in publisher's thread |
| `ThreadRegistry` | All lane dispatchers must register | Wired in `WorkerPool.start()` |
| `PlanManager._lock` | Review lane reads plan state | Already safe — RLock allows reentrant reads |
| `DefaultNetwork` thread | Independent reactive loop | No change — continues at 30Hz independently |
| Existing `prefetch.py` | File-level prefetch for user input | Complementary — `PrefetchHook` is for inter-job data gathering. The two systems serve different purposes. |
| ATL (own RWLock) | Read per-tick by `memory_agent._build_knowledge_context()`; written only at session-end promotion | No contention with hippocampus — independent lock. Safe for async hippocampus. |
| Angular Gyrus (own RWLock) | Read + written per-tick by `math_bridge` | Independent lock. Math operations don't touch hippocampus state. |
| CrossLayerGraph | Read per-tick by `MemoryHub.recall_with_knowledge()`; edges added at session-end | No contention with hippocampus capture. Edge creation happens after flush. |
| SemanticPromoter | Session-end only (`scan_for_promotions`) | Runs after `hippocampus.flush()` — fully safe. |
| EC NeuralEmbedder | Async via `schedule_embedding()` — triggered by hippocampus capture callback | Already decoupled. Callback fires from hippocampus worker thread; EC has its own queue. |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Deadlock from circular deps | `JobRegistry` detects cycles at submit time. Cycles raise `ValueError`. |
| Memory growth from completed jobs | `JobRegistry.prune()` runs every 60s, clears jobs older than 5 min. |
| Prefetch reads stale data | See Issue #1 below — this needs a two-phase prefetch design. |
| Hippocampus capture queue overflow | Bounded queue (100) with drop-oldest + `task_done()` balance. Lost captures are logged. Non-fatal — hippocampus is best-effort during heavy load. |
| GPU contention with multiple infer workers | Start with `max_workers=1` for infer lane. Only increase when GPU VRAM confirmed sufficient for concurrent inference batches. |
| Debugging complexity | Structured logging with `job_id`, lane name, dep chain. `WorkerPool.status()` method for runtime introspection. |

---

## Review: Identified Pitfalls

### Issue #1 (Critical): Prefetch semantic contradiction

The plan says the infer job prefetches `context_pool.snapshot()` while waiting
for the record job to finish. But the *entire reason* the infer job depends on
the record is that the record updates the context pool. If prefetch captures
the context *before* record writes, the LLM gets a stale snapshot that's
missing the most recent outcome — exactly the data we were waiting for.

**Fix:** Split prefetch into two phases:
- **Phase A (immediate):** Gather *stable* data that won't change — agent
  states, mode info, tool descriptions, available tools, conversation history.
  This is the expensive part (prompt building, strategy selection).
- **Phase B (post-deps):** After deps resolve, do a fast `context_pool`
  snapshot that includes the freshly-recorded outcome. This is cheap (just a
  list copy under RLock).

```python
@dataclass
class DependencySpec:
    wait_for: list[str] = field(default_factory=list)
    # Runs immediately — gather stable/expensive data
    prefetch_early: Callable[[], Any] | None = None
    # Runs after deps resolve — gather dep-dependent data (should be fast)
    prefetch_late: Callable[[], Any] | None = None
    timeout_s: float = 30.0
```

Then `_execute_job` calls `gate.wait()` which handles both phases
internally — see `DependencyGate.wait()` in §1.2 for the canonical
implementation. `wait()` returns `({"early": ..., "late": ...}, failed_deps)`.

This preserves the latency win (expensive work done during wait) without
sacrificing correctness (fresh context read after deps complete).

---

### Issue #2 (High): Worker slots consumed by blocked dependency waits

In `_execute_job`, the thread blocks on `job.gate.wait()` *inside* the
`ThreadPoolExecutor`. With `max_workers=1` for the infer lane, a single
waiting job blocks the entire lane. A higher-priority job with no deps that
arrives later can't execute — the one thread is stuck waiting.

**Fix:** Don't dispatch to the executor until deps are resolved. Move the dep
wait into the dispatcher thread (or a dedicated "gate watcher" thread):

```python
def _dispatch_loop(self) -> None:
    while not self._stop.is_set():
        _, job = self._queue.get(timeout=0.5)
        if job.gate:
            # Wait in dispatcher, not in worker slot
            prefetched, failed = job.gate.wait()
            job._prefetched = prefetched
        self._executor.submit(self._execute_job, job)
```

**Caveat:** This makes the dispatcher single-threaded for dep waits. If you
need multiple jobs waiting on different deps concurrently, use a small
secondary thread pool for gate-watching (separate from the work executor).
Alternatively, have the dispatcher spawn lightweight gate-watcher threads:

```python
def _dispatch_loop(self) -> None:
    while not self._stop.is_set():
        _, job = self._queue.get(timeout=0.5)
        if job.gate:
            threading.Thread(
                target=self._wait_then_dispatch, args=(job,),
                daemon=True, name=f"gate-{job.job_id}",
            ).start()
        else:
            self._executor.submit(self._execute_job, job)

def _wait_then_dispatch(self, job: Job) -> None:
    job._prefetched, _ = job.gate.wait()
    self._executor.submit(self._execute_job, job)
```

---

### Issue #3 (High): PriorityQueue comparison crash

```python
self._queue.put((job.priority, job))
```

When two jobs have equal priority, Python's `PriorityQueue` falls back to
comparing the second tuple element. `Job` dataclasses don't implement `__lt__`,
so this raises `TypeError`. This is a classic Python PriorityQueue gotcha.

**Fix:** Add a monotonic tiebreaker:

```python
_counter = itertools.count()

def submit(self, job: Job) -> None:
    self._queue.put((job.priority, next(_counter), job))
```

Or implement `__lt__` on `Job` using submission order.

---

### Issue #4 (High): Lambda closure captures mutable references

```python
pool.submit(
    lane="record",
    fn=lambda: hippocampus.capture_from_loop(episode_data),
)
```

`episode_data` is a reference. The main loop mutates or rebinds this variable
on the next iteration (30Hz = 33ms later). By the time the record lane
executes the lambda, `episode_data` may point to different data or have been
mutated in-place.

**Fix:** Capture by value at submission time:

```python
# Shallow copy for dicts:
_data = dict(episode_data)
pool.submit(lane="record", fn=lambda _d=_data: hippocampus.capture_from_loop(_d))

# Or use functools.partial:
from functools import partial
pool.submit(lane="record", fn=partial(hippocampus.capture_from_loop, dict(episode_data)))
```

This applies to *every* lambda/closure in the plan that references loop-local
variables — `tool_result`, `context`, `action`, etc.

---

### Issue #5 (High): No preemption/cancellation for queued jobs

The existing preemption circuit (`preemption.py`) and `ConcurrentExecutor.cancel_all()`
can cancel in-flight tool execution. But there's no mechanism to cancel jobs
queued in the `WorkerPool`. If a preemption event fires (e.g., user says
"stop"), the record/review/infer jobs already queued keep running.

Worse: a preempted plan's review jobs keep evaluating a plan that's no longer
active, and their results contaminate the context pool.

**Fix:** Add cancellation to `WorkerPool`:

```python
class WorkerPool:
    def cancel_lane(self, lane: str) -> int:
        """Drain and discard all pending jobs in a lane."""
        ...

    def cancel_all(self) -> int:
        """Cancel everything across all lanes."""
        ...
```

Wire into preemption circuit:
```python
# In preemption handler:
pool.cancel_lane("review")  # stop evaluating old plan
pool.cancel_lane("infer")   # stop generating actions for old plan
# let record lane finish — data should still be persisted
```

Jobs should check a `cancelled` flag in `_execute_job` before running.

---

### Issue #6 (Medium): Session shutdown doesn't drain the pool

At shutdown (agent_loop.py ~L2407), the code persists `context_pool` and
`hippocampus`. But if record lane workers are still writing to hippocampus,
the persisted state may be incomplete or corrupted (save during write).

**Fix:** Drain the pool before final persistence:

```python
# Before final save:
pool.stop(timeout=10.0)  # waits for in-flight jobs to finish
# Now safe to persist
context_pool.save()
hippocampus.save()
```

This needs to be explicit in the integration plan (Phase 2).

---

### Issue #7 (Medium): "Unknown job = assume done" silently bypasses deps

```python
def wait_for_completion(self, job_id: str, timeout: float) -> bool:
    event = self._events.get(job_id)
    if event is None:
        return True  # unknown job = assume done
```

If a job ID is mistyped, or `prune()` GC'd a completed job and a *new* job
later references it as a dependency, the wait returns immediately. The
dependent job runs without its prerequisite — silent correctness violation.

**Fix:** Either:
- Raise `KeyError` for unknown jobs (fail-fast)
- Or track GC'd jobs separately: keep a `_completed_ids: set[str]` that
  survives pruning, so "GC'd but completed" is distinguishable from "never
  existed"

```python
def wait_for_completion(self, job_id: str, timeout: float) -> bool:
    with self._lock:
        event = self._events.get(job_id)
        if event is None:
            if job_id in self._completed_ids:
                return True  # was completed, then GC'd
            raise ValueError(f"Unknown job dependency: {job_id}")
    return event.wait(timeout=timeout)
```

---

### Issue #8 (Medium): Unbounded prefetch thread creation

Each job with a `DependencySpec.prefetch` spawns a new `threading.Thread`.
Record jobs are submitted at action frequency (potentially several per second
during batched exploration). Each spawns a prefetch thread. Thread creation
is ~1ms on Linux but involves kernel allocation.

**Fix:** Use a small shared `ThreadPoolExecutor` for prefetch work:

```python
class WorkerPool:
    def __init__(self, ...):
        self._prefetch_pool = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="Prefetch"
        )
```

---

### Issue #9 (Medium): `context_pool.snapshot()` doesn't exist

The plan references `context_pool.snapshot()` in multiple places, but
`ContextPool` has no such method. It would need to be implemented — a deep
copy of `_entries` under the `_lock` RLock.

**Fix:** Add to Phase 1 tasks:

```python
# In ContextPool:
def snapshot(self) -> list[ContextEntry]:
    """Return an immutable copy of current entries for cross-thread use."""
    with self._lock:
        return list(self._entries)  # shallow copy; entries are immutable
```

Note: `ContextEntry` is a dataclass — verify it's not mutated after creation.
If it is, need `copy.deepcopy`.

---

### Issue #10 (Low): Priority ordering lost after dispatch

The `PriorityQueue` maintains ordering, but once the dispatcher submits jobs
to the `ThreadPoolExecutor`, they enter the executor's internal FIFO queue.
If the executor is busy (all workers occupied), lower-priority jobs submitted
earlier run before higher-priority jobs submitted later.

**Fix:** This is mostly mitigated by the Issue #2 fix (don't dispatch until
deps resolve). But for true priority fairness, the dispatcher should peek
at the queue head and only dispatch when a worker slot is free — effectively
making the dispatcher a priority-aware scheduler rather than a blind drain.

For v1, this is acceptable. The lanes have small worker counts (1-2), so
the priority inversion window is short.

---

### Issue #11 (Low): PlanManager.active_plan read is unlocked

The review lane example reads:
```python
phase = plan_manager.active_plan.get_current_phase()
```

`active_plan` is a `@property` returning `self._active_plan` with no lock.
While Python's GIL makes the reference read atomic, the *plan object's
internal state* (phase statuses, `current_phase_index`) can be mid-mutation
if `advance_phase()` is running concurrently on the main thread.

**Fix:** Add a snapshot method to `PlanManager`:

```python
def get_current_phase_snapshot(self) -> Phase | None:
    """Thread-safe read of current phase state."""
    with self._lock:
        if not self._active_plan:
            return None
        phase = self._active_plan.get_current_phase()
        return copy.copy(phase) if phase else None
```

Use this in the review worker instead of direct property access.

---

---

## Review: Integration Concerns

### Issue #12 (Critical): Post-execution block is tightly coupled — can't move pieces independently

The plan says to move `context_pool.add_outcome()`, `llm_worker.record_outcome()`,
and `hippocampus.capture_from_loop()` to the record lane. But the post-execution
block (agent_loop.py L1630-1830) is a ~200-line sequence where later operations
depend on values computed earlier:

```
1. get_tool_followup_type()         → determines followup behavior
2. Format result_str from output     → used by 3, 4, 5, 7, 8
3. recent_outcomes.append(outcome)   → read by LLM submission in step 6
4. llm_worker.record_outcome()       → writes to ReasoningCarryover
5. context_pool.add_outcome()        → affects next LLM prompt
6. memory_hub.record_plan_outcome()  → writes to memory hub
7. Check followup → set pending_action_followup  → controls next LLM trigger
8. context_pool.add_conversation_turn()
9. environment.step(result) → update state
10. memory.store_raw()
11. hippocampus.capture_from_loop()  → the SLOW one (~50-200ms)
12. Handle failure → state.mark_failure()
```

Of these, **only #11 (hippocampus capture) is slow enough to justify async**.
Items 3-6 and 8 are sub-millisecond operations (append to list, append under
lock). Moving them off the main loop adds thread-safety complexity for
negligible performance gain.

**Fix:** Narrow the record lane scope. Only hippocampus capture (and potentially
`memory.store_raw()`) should move to the record lane. Everything else stays
synchronous on the main loop:

```python
# Main loop (synchronous — all fast):
recent_outcomes.append(outcome)
llm_worker.record_outcome(...)
context_pool.add_outcome(...)
memory_hub.record_plan_outcome(...)
pending_action_followup = {...}

# Passive hippocampus (async — slow):
hippocampus.capture_from_loop_async(**snapshot)
```

This dramatically simplifies integration. No dependency gates needed for the
fast operations. The only async handoff is hippocampus capture, which has no
downstream readers in the same iteration.

---

### Issue #13 (High): `recent_outcomes` is a main-loop-local list — not thread-safe

`recent_outcomes` is a plain `list[dict]` local to `run_agentic_loop()`. It's
passed by reference to `llm_worker.submit_context()` (L2332). If any record
lane worker appends to it while the main loop reads it for LLM submission,
you get a race condition.

This is resolved by Issue #12's fix: keep `recent_outcomes.append()` on the
main thread. But if the plan proceeds as originally written (moving record_outcome
to the record lane), `recent_outcomes` would need to become a thread-safe
structure (e.g., `collections.deque` with maxlen, or wrapped with a lock).

---

### Issue #14 (High): Prompt building is embedded inside LLMWorker — can't prefetch it

The plan says "prefetch hook builds prompt + context snapshot." But prompt
building is deeply embedded in `LLMWorker._process_request()` (L960) →
`_build_prompt()` (L1269). This method:

- Checks for action followups in `context.cli_inputs`
- Tries simple answer shortcuts (arithmetic, date/time)
- Builds full tool-aware prompts via `PromptBudgeter`
- Has complex branching based on mode, strategy, sleep state
- Reads `ReasoningCarryover` (which has its own lock)

You can't call `_build_prompt()` from a prefetch hook because:
1. It's an instance method on `LLMWorker`, not a standalone function
2. It reads `self._reasoning_carryover` which is being mutated by `record_outcome()`
3. The `LLMRequest` object it needs is constructed in `submit_context()` —
   the prefetch would need to build the request too

**Fix:** For v1, don't try to prefetch prompt building. Instead, prefetch
the *inputs* to prompt building — the expensive data gathering:

```python
# Prefetch (runs during dep wait):
prefetch=lambda: {
    "context_pool_text": context_pool.get_context_text(max_tokens=...),
    "conversation_text": context_pool.get_conversation_text(max_turns=5),
    "agent_states": agent.get_agent_states(),
    "tool_descriptions": {name: TOOL_DESCRIPTIONS[name] for name in available_tools},
}
```

Then `submit_context()` accepts pre-gathered data instead of gathering it
inline. The actual prompt is still built inside `_process_request()` on the
worker thread, but the data it needs is already ready.

---

### Issue #15 (High): The followup system is inherently synchronous

After tool execution, `pending_action_followup` is set as a main-loop local
variable (L1713). On the *next* loop iteration, step 6 checks this variable
and triggers a new LLM submission with the followup context (L2021-2029).

If hippocampus capture moves to the record lane, the followup trigger still
fires on the next main loop iteration — it doesn't depend on hippocampus.
This is actually fine as-is. But the plan's dependency chain diagram suggests
infer should wait for record, which would add unnecessary latency to followups.

**Fix:** Make the dependency graph smarter. Not all infer jobs need to wait
for the preceding record job. Followup-triggered LLM calls don't depend on
hippocampus completion — only on `context_pool.add_outcome()` which stays
synchronous (per Issue #12 fix). The dep chain should be:

```
Tool followup:  [execute] → [fast records on main] → [submit infer immediately]
                                                   → [submit hipp capture async, no dependents]

Plan phase transition:  [execute] → [fast records] → [hipp capture] → [review] → [submit infer]
                                                      ↑ only THIS needs the dep chain
```

---

### Issue #16 (Medium): LLMWorker does far more than "call the LLM" — wrapping it is non-trivial

The plan says "Existing LLMWorker becomes a thin wrapper submitting to the
infer lane." But `LLMWorker` owns:

1. `ReasoningCarryover` management (lock-protected append/read)
2. Prompt building via `PromptBudgeter` (complex branching)
3. LLM JSON response parsing into `LLMProposal`
4. Timeout handling with executor replacement (L808-816)
5. Simple answer shortcuts (arithmetic, date/time — bypasses LLM entirely)
6. Staleness checking (drops old requests)
7. Fallback responses when LLM is unavailable

Wrapping `_call_llm_with_timeout` is insufficient — you'd need to wrap
`_process_request()` which includes all of the above. But `_process_request`
returns an `LLMProposal` which the main loop consumes — this is already
exactly what the infer lane job should return.

**Fix:** The cleanest integration is:

```python
# LLMWorker keeps _process_request() intact.
# Its _worker_loop is replaced by the infer lane dispatcher.
# submit_context() enqueues to the infer lane instead of its own PriorityQueue.
# get_latest_proposal() polls the WorkerPool instead of _latest_proposal.

class LLMWorker:
    def start(self):
        # Instead of starting own thread:
        # self._pool_lane is set by the WorkerPool
        pass

    def submit_context(self, ...):
        request = LLMRequest(...)
        self._pool.submit(
            lane="infer",
            job_id=request.request_id,
            fn=partial(self._process_request, request),
        )
```

This keeps all the prompt building, parsing, and fallback logic inside
`LLMWorker` where it belongs. Only thread management moves to the pool.

---

### Issue #17 (Medium): AgentBus handlers run in the publisher's thread

The plan says the review lane publishes `ReviewCompleted` via the bus.
`AgentBus.publish()` (bus.py L869-878) calls handlers *synchronously in the
publisher's thread*. If any handler does expensive work or modifies shared
state, the review lane worker is blocked.

More subtly: if a bus handler for `ReviewCompleted` modifies `context_pool`
(to record the evaluation), now the review lane worker is writing to
context_pool through a handler, while the main loop may also be writing to
context_pool through `add_outcome()`. Both use the RLock, so it's technically
safe — but it creates a hidden coupling between lanes through the bus.

**Fix:** Document this explicitly: all bus handlers called from non-main
threads MUST be fast and non-blocking. For heavier reactions to bus events,
handlers should submit a new job to the appropriate lane rather than doing
work inline.

---

### Issue #18 (Medium): Review lane purpose is underspecified

The plan says review jobs "evaluate tool outcomes against plan expectations"
but doesn't define what this means concretely. Is it:

- **Heuristic comparison** (did the tool return expected output)? → CPU-only, fast, barely needs a lane
- **LLM-based evaluation** (ask the LLM to judge the outcome)? → Should use the infer lane, not a separate review lane
- **Statistical analysis** (track success rates, update NAc predictions)? → `StatisticianAgent` already does this on the main thread

If review is heuristic, it's fast enough to stay on the main loop. If it
calls the LLM, it should share the infer lane (separate lanes for the same
GPU resource doesn't help). The review lane only makes sense if it does
meaningful CPU-bound work that's too slow for the main loop (~10ms+) but
doesn't need GPU.

**Fix:** Define the review callable concretely before building the lane. If
review turns out to be LLM-powered, make it a job type in the infer lane
with lower priority, not a separate lane.

---

## Review: Memory System Interaction Correction

### Issue #19 (Critical): ATL/AG/CrossLayer are NOT session-end-only

An earlier analysis claimed "ATL/AG/CrossLayer/Promoter are completely safe —
they're session-end-only." **This was incorrect.** Here's what actually happens:

#### Active-loop access (during normal operation)

| System | Access type | Call site | Trigger |
|--------|-----------|-----------|---------|
| **ATL** | READ | `MemoryHub.recall_concepts()` → `atl.recall()` (memory_hub.py L531) | `memory_agent._build_knowledge_context()` per-tick |
| **ATL** | READ | `hub.atl.find_by_relationship()` | `memory_agent._build_knowledge_context()` per-tick |
| **Angular Gyrus** | READ | `ag.recall_method(goal)` (math_bridge.py L96) | `math_bridge.enrich_context()` per-tick |
| **Angular Gyrus** | READ | `ag.recall(name=pattern_key)` (math_bridge.py L134) | `math_bridge.promote_patterns()` |
| **Angular Gyrus** | WRITE | `ag.store(record)` (math_bridge.py L164) | `math_bridge.promote_patterns()` — writes learned patterns |
| **CrossLayerGraph** | READ | `cross_layer_activation()` (memory_hub.py L545) | `MemoryHub.recall_with_knowledge()` |
| **Hippocampus** | WRITE | `capture_from_loop()` (agent_loop.py L1266, L1761) | After action execution |

#### Session-end-only access

| System | Operation | What it does |
|--------|-----------|-------------|
| **SemanticPromoter** | `scan_for_promotions()` | Batch promotes hippocampus patterns → ATL concepts |
| **ATL** | `consolidate()` | Compress/prune old semantic concepts |
| **Angular Gyrus** | `consolidate()` | Compress/prune old math memories |
| **Hippocampus** | `sleep()` | Sleep consolidation cycle |

#### Key insight: hippocampus is already decoupled from reads

Despite all of the above, **the agent loop only WRITES to hippocampus** (via
`capture_from_loop()`). It **never reads** from it during the active loop:

- Bridges (spatial, salience, planning, fear, escalation) call `hippocampus.recall()`
  and `hippocampus.recall_associated()` — but **only during `on_session_start()`**.
  After that, they serve from internal caches.
- `memory_agent._get_relevant_memories()` queries its own `_association_index`
  and `_association_graph` — NOT the hippocampus.
- The per-tick reads that DO occur go to ATL and AG, which have their own
  independent RWLocks.

This means **async hippocampus capture has no active-loop read contention**.
The only coupling is through the capture callback chain
(`_on_memory_captured` → EC `schedule_embedding()`), which already targets
an async queue.

#### Impact on async hippocampus design

Because ATL/AG/CrossLayerGraph have their own locks and are independent of
hippocampus write timing:
- ATL reads are safe — written only at session-end promotions
- AG reads AND writes are safe — uses its own RWLock
- CrossLayerGraph reads are safe — edges only added during session-end promotion
- The one-tick staleness from async hippocampus capture does NOT propagate to
  ATL/AG/CrossLayerGraph queries

---

## Passive Hippocampus Design

### The model

Make hippocampus operate like ATL/AG — it runs passively on its own thread,
processes captures asynchronously, and fires events when data is ready. The
main loop never blocks on hippocampus writes.

This is biologically accurate: sensory input reaches the hippocampus and
encoding happens in the background. The cortex doesn't wait for the
hippocampus to finish writing before continuing.

### Why this works

The active-loop analysis above shows hippocampus has a clean separation:
- **Writes**: Only from `agent_loop.capture_from_loop()` (fire-and-forget)
- **Reads**: Only from bridge `on_session_start()` (before the loop starts)
  and session-end operations (after the loop stops)
- **No per-tick reads**: memory_agent uses its own association structures
- **Callbacks already async**: EC embedding via `schedule_embedding()` targets
  its own thread pool

### Architecture

```
Main Loop (30Hz)                    Hippocampus Worker Thread
     │                                       │
     ├─ execute action                       │
     ├─ fast records (sync):                 │
     │   context_pool.add_outcome()          │
     │   llm_worker.record_outcome()         │
     │   recent_outcomes.append()            │
     │                                       │
     ├─ queue capture ──────────────────────►│
     │   (returns immediately)               ├─ acquire write lock
     │                                       ├─ store memory
     ├─ continue 30Hz loop                   ├─ build index
     │                                       ├─ form associations
     │                                       ├─ fire capture callbacks
     │                                       │   └─ EC schedule_embedding()
     │                                       ├─ release write lock
     │                                       │
     │                               [optionally notify bridges]
     │                                       ├─ salience_bridge.on_capture(mem)
     │                                       ├─ spatial_bridge.on_capture(mem)
     │                                       └─ planning_bridge.on_capture(mem)
```

### Implementation

```python
# Added to hippocampus.py:

class _SnapshotState:
    """Minimal state-like wrapper around a pre-resolved dict snapshot.

    capture_from_loop() accesses state attributes (e.g. state.data,
    state.processing_state). This wrapper presents a dict snapshot as
    if it were a real state object so capture_from_loop() works unchanged.
    """

    def __init__(self, data: dict[str, Any] | None):
        self.data = data or {}
        # Provide common state attributes that capture_from_loop reads:
        self.processing_state = self.data.get("processing_state")
        self.operational_mode = self.data.get("operational_mode")
        self.strategy = self.data.get("strategy")

    def snapshot(self) -> dict[str, Any]:
        return dict(self.data)

@dataclass
class _CaptureRequest:
    """Immutable snapshot of capture data, queued for background processing."""
    observation: dict[str, Any]
    state_snapshot: dict[str, Any] | None
    intent: dict[str, Any]
    decision: dict[str, Any]
    action: dict[str, Any]
    result: Any
    run_id: str
    queued_at: float

# Added to Hippocampus class:

def __init__(self, config=None):
    # ... existing init ...
    self._capture_queue: queue.Queue[_CaptureRequest] = queue.Queue(maxsize=100)
    self._worker_thread: threading.Thread | None = None
    self._stop_event = threading.Event()

def start(self, thread_registry: "ThreadRegistry | None" = None) -> None:
    """Start the background capture worker."""
    self._stop_event.clear()
    self._worker_thread = threading.Thread(
        target=self._capture_worker,
        daemon=True,
        name="hippocampus-capture",
    )
    self._worker_thread.start()
    if thread_registry:
        thread_registry.register("hippocampus-capture", self._worker_thread)

def _capture_worker(self) -> None:
    """Background thread: drain capture queue, process each."""
    while not self._stop_event.is_set():
        try:
            request = self._capture_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            # This acquires write lock internally
            self._process_capture(request)
        except Exception as e:
            logger.error("Async capture failed: %s", e)
        finally:
            self._capture_queue.task_done()

def _process_capture(self, request: _CaptureRequest) -> None:
    """Process a queued capture request. Runs on worker thread."""
    latency_ms = (time.time() - request.queued_at) * 1000
    logger.debug("Processing capture (queue latency: %.1fms)", latency_ms)
    # Delegate to existing capture_from_loop, passing pre-resolved state
    # The state_snapshot was already resolved at queue time (Issue #25),
    # so we build a minimal state-like object for capture_from_loop.
    self.capture_from_loop(
        observation=request.observation,
        state=_SnapshotState(request.state_snapshot),
        intent=request.intent,
        decision=request.decision,
        action=request.action,
        result=request.result,
        run_id=request.run_id,
    )

def capture_from_loop_async(self, **kwargs) -> None:
    """Non-blocking capture: queue for background processing."""
    # Snapshot mutable data NOW to avoid closure issues (Issue #4).
    # State requires special handling: resolve to a dict snapshot
    # rather than shallow-copying the object (Issue #25).
    state = kwargs.get("state")
    state_snapshot = None
    if hasattr(state, "snapshot") and callable(state.snapshot):
        try:
            state_snapshot = state.snapshot()
        except Exception:
            pass
    elif hasattr(state, "data") and hasattr(state.data, "items"):
        state_snapshot = dict(state.data)

    request = _CaptureRequest(
        observation=dict(kwargs.get("observation", {})),
        state_snapshot=state_snapshot,
        intent=dict(kwargs.get("intent", {})),
        decision=dict(kwargs.get("decision", {})),
        action=dict(kwargs.get("action", {})),
        result=kwargs.get("result"),
        run_id=kwargs.get("run_id", ""),
        queued_at=time.time(),
    )
    try:
        self._capture_queue.put_nowait(request)
    except queue.Full:
        logger.warning("Hippocampus capture queue full, dropping oldest")
        try:
            self._capture_queue.get_nowait()
            self._capture_queue.task_done()  # balance unfinished_tasks counter
        except queue.Empty:
            pass
        self._capture_queue.put_nowait(request)

def flush(self, timeout: float = 10.0) -> bool:
    """Block until all queued captures are processed.

    Call before session-end consolidation or final save.
    Returns True if queue drained within timeout.
    """
    deadline = time.monotonic() + timeout
    while not self._capture_queue.empty():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "Hippocampus flush timed out, %d items remain",
                self._capture_queue.qsize(),
            )
            return False
        time.sleep(0.05)
    # Final check: all task_done() calls completed
    with self._capture_queue.all_tasks_done:
        remaining = max(0, deadline - time.monotonic())
        if self._capture_queue.unfinished_tasks > 0:
            self._capture_queue.all_tasks_done.wait(timeout=remaining)
    return self._capture_queue.unfinished_tasks == 0

def stop(self) -> None:
    """Stop the background worker. Drains queue first."""
    self.flush(timeout=5.0)
    self._stop_event.set()
    if self._worker_thread and self._worker_thread.is_alive():
        self._worker_thread.join(timeout=2.0)
```

### Reactive bridge updates (optional enhancement)

Currently bridges load from hippocampus at `on_session_start()` and never
update. With passive hippocampus, bridges can subscribe to capture events
for continuous learning:

```python
# In MemoryHub._wire_bridges():
self._hippocampus.register_capture_callback(self._on_capture_for_bridges)

def _on_capture_for_bridges(self, memory_id: str, memory: EpisodicMemory) -> None:
    """Update bridge caches reactively when new memories form."""
    # These run on the hippocampus worker thread — must be fast
    if self._salience_bridge:
        self._salience_bridge.update_from_capture(memory)
    if self._spatial_bridge:
        self._spatial_bridge.update_from_capture(memory)
    if self._plan_bridge:
        self._plan_bridge.update_from_capture(memory)
```

Each bridge adds a lightweight `update_from_capture()` method that updates
its internal cache incrementally, rather than re-querying hippocampus.

### Tradeoffs

| Tradeoff | Assessment |
|----------|-----------|
| **Back-to-back captures miss associations** | Low risk. Captures only happen after action execution, not every tick. At 30Hz, actions are spaced seconds apart. |
| **Session-end race** | Solved by `flush()`. Call before `on_session_end()`. |
| **Callback thread safety** | EC already uses its own queue. Bridge `update_from_capture()` methods must be fast and lock-free or use their own locks. |
| **Queue overflow under load** | Bounded queue (100) with drop-oldest policy. Lost captures are logged but non-fatal — hippocampus is best-effort during heavy load. |
| **Memory ordering** | Queue is FIFO — captures processed in submission order. Associations form correctly in sequence. |
| **One-tick staleness** | Negligible. No per-tick hippocampus readers exist in the active loop. |

### Integration with WorkerPool

The passive hippocampus design is **complementary** to the worker pool, not
a replacement. The worker pool manages LLM inference and review lanes. The
hippocampus manages its own capture thread. This separation is cleaner than
routing hippocampus captures through the worker pool's record lane because:

1. Hippocampus captures don't need dependency gates (no downstream consumers in the same tick)
2. The hippocampus worker thread can be registered with `ThreadRegistry` for coordinated shutdown
3. The capture queue's FIFO ordering is simpler than the priority queue in a lane
4. Future: hippocampus could run its own consolidation tasks on the same thread during idle periods

The record lane in the worker pool can be repurposed for other async
record operations (e.g., `memory.store_raw()`, cross-layer edge creation)
that DO need dependency gates.

### Agent loop integration (Phase 2 revision)

```python
# agent_loop.py — after action execution:

# Fast synchronous operations (unchanged):
recent_outcomes.append(outcome)
llm_worker.record_outcome(...)
context_pool.add_outcome(...)
memory_hub.record_plan_outcome(...)

# Async hippocampus capture (new):
hippocampus.capture_from_loop_async(
    observation=observation,
    state=state,
    intent=intent,
    decision=decision,
    action=action_data,
    result=result,
    run_id=run_id,
)
# Main loop continues immediately — no write lock wait

# ... at shutdown:
hippocampus.flush()      # drain capture queue
hippocampus.save()       # persist to disk
memory_hub.on_session_end()  # consolidation
```

---

## Revised Recommendations

Based on the integration review, here's what I'd change about the plan:

### Simplify Phase 2 dramatically

Move `hippocampus.capture_from_loop()` to its own async capture queue
(not the worker pool's record lane — see Passive Hippocampus Design).
Keep all fast operations (`context_pool.add_outcome`,
`llm_worker.record_outcome`, `recent_outcomes.append`,
`memory_hub.record_plan_outcome`) synchronous on the main loop. This
eliminates the need for dependency gates in the common case (tool
execution → fast records → submit LLM).

### Rethink the dependency chain

Most LLM submissions don't need to wait for hippocampus capture. The context
pool and reasoning carryover are already updated synchronously. Reserve
dependency gates for specific scenarios:
- Plan phase transitions (review should evaluate against persisted memory)
- Sleep consolidation (wait for all captures to finish before consolidation)
- Multi-step chains where step N+1's prompt needs step N's hippocampus data

### Keep LLMWorker largely intact

Don't rewrite `LLMWorker`. Replace its `_worker_loop` with the infer lane
dispatcher, but keep `_process_request`, `_build_prompt`, and all the
fallback/parsing logic inside `LLMWorker`. The lane just manages *when* and
*where* `_process_request` runs — not *what* it does.

### Defer the review lane

The review lane's purpose is too vague to implement well right now. Start
with just record + infer lanes. Add review later when there's a concrete
evaluation workload that's too slow for the main loop and too different from
LLM inference to share the infer lane.

### Prefetch = pre-gather inputs, not pre-build prompts

Prefetch hooks should gather the expensive *inputs* to prompt building
(context pool text, conversation history, agent states) rather than trying
to build the prompt itself. `_build_prompt()` stays inside `LLMWorker`.

---

### Updated Summary: All action items

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | Critical | Prefetch captures pre-record stale data | Two-phase prefetch (early + late) |
| 12 | Critical | Post-exec block is tightly coupled | Only move hippocampus capture async; keep fast ops synchronous |
| 2 | High | Worker slots blocked by dep waits | Gate-watcher threads, not executor slots |
| 3 | High | PriorityQueue comparison crash | Monotonic tiebreaker counter |
| 4 | High | Lambda closures capture mutable refs | `dict()` copy or `functools.partial` |
| 5 | High | No preemption for queued jobs | `cancel_lane()` + cancelled flag |
| 13 | High | `recent_outcomes` is not thread-safe | Keep on main thread (resolved by #12) |
| 14 | High | Prompt building can't be prefetched | Prefetch inputs, not the prompt itself |
| 15 | High | Followup system is synchronous | Not all infer jobs need to wait for record |
| 16 | Medium | LLMWorker wrapping is non-trivial | Replace `_worker_loop`, keep `_process_request` |
| 6 | Medium | Shutdown doesn't drain pool | `pool.stop()` before final persist |
| 7 | Medium | Unknown job ID = silent dep skip | Track GC'd completions separately |
| 8 | Medium | Unbounded prefetch threads | Shared prefetch `ThreadPoolExecutor` |
| 9 | Medium | `snapshot()` method missing | Add to `ContextPool` in Phase 1 |
| 17 | Medium | Bus handlers block publisher thread | Handlers must be fast; heavy work → submit new job |
| 18 | Medium | Review lane is underspecified | Define concrete workload or defer |
| 10 | Low | Priority ordering lost post-dispatch | Acceptable for v1 |
| 11 | Low | Unlocked plan reads from review lane | `get_current_phase_snapshot()` |
| 19 | Critical | ATL/AG/CrossLayer are NOT session-end-only | Corrected: they have active-loop reads/writes, but use independent locks; hippocampus async is still safe |
| 20 | Critical | `flush()` timeout unused — `queue.join()` blocks forever | Timed loop with `all_tasks_done` condition wait |
| 21 | Critical | Drop-oldest skips `task_done()` — permanent `flush()` deadlock | Call `task_done()` after `get_nowait()` for dropped items |
| 22 | High | Phase 2 checklist contradicts revised recommendations | Rewritten as "Passive Hippocampus" phase |
| 23 | High | `JobRegistry._results`/`_errors` never initialized | Added to `__init__` |
| 24 | Medium | Parts 1.1, 1.3, 3.2, 3.3, 3.4 stale — contradict revised plan | Updated inline with superseded notes |
| 25 | Medium | `copy.copy(state)` shallow — nested dicts shared with main thread | Resolve `state.snapshot()` or `dict(state.data)` at submission time |
| 26 | Low | Hippocampus worker thread not registered with `ThreadRegistry` | `start()` accepts optional `ThreadRegistry` |
| 27 | High | `DependencyGate` references deleted `self._spec.prefetch` field | Updated to use `prefetch_early`/`prefetch_late` two-phase flow |
| 28 | Medium | "Why prefetch?" paragraph describes obsolete flow | Updated to reflect revised recommendations |
| 29 | Medium | Phase 3/4 checklists not updated for revised plan | Phase 3 = infer lane; Phase 4 = review (deferred) |
| 30 | Medium | Interaction/Risks tables have stale hippocampus references | Updated for passive hippocampus (single worker thread) |
| 31 | Low | `_CaptureRequest` and `_process_capture` not defined | Added dataclass and method definitions |
| 32 | High | `Lane.submit()` calls `start_prefetch()` — renamed to `start_early_prefetch()` | Updated call site |
| 33 | Medium | `_completed_ids` initialized but never populated | Added `self._completed_ids.add(job_id)` in `mark_completed()` |
| 34 | Medium | Issue #1 fix code uses undefined `gate.wait_early()` | Replaced with reference to canonical `DependencyGate.wait()` |
| 35 | Medium | `WorkerPool.get_completed()` undefined — main loop can't poll | Added `get_completed()` to `WorkerPool` and `pop_completed()` to `JobRegistry` |
| 36 | Low | `_SnapshotState` class never defined | Added class definition with common state attributes |
| 37 | Low | "Simplify Phase 2" paragraph says "record lane" | Updated to say "own async capture queue" |
| 38 | Low | Part 2 canonical code doesn't incorporate Issue #2/#3 fixes | Updated `_dispatch_loop` with gate-watcher threads; added `_counter` tiebreaker |