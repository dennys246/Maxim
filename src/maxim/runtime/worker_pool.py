"""Parallel worker pool with typed lanes and dependency gates.

Provides a worker pool architecture where operations are categorized into
tier-based lanes (large, medium, small), each with its own bounded thread
pool. Jobs can declare dependencies on other jobs and prefetch data while
waiting for those dependencies to resolve.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from maxim.runtime.function_router import FunctionRouter
    from maxim.utils.thread_manager import ThreadRegistry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────


class JobStatus(Enum):
    """Lifecycle states of a job."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class DependencySpec:
    """Declares what a job needs before it can execute.

    Two-phase prefetch design:
    - prefetch_early runs immediately on submission (gather stable/expensive data)
    - prefetch_late runs AFTER deps resolve (gather dep-dependent data, should be fast)
    """

    wait_for: list[str] = field(default_factory=list)
    prefetch_early: Callable[[], Any] | None = None
    prefetch_late: Callable[[], Any] | None = None
    timeout_s: float = 30.0


@dataclass
class Job:
    """A unit of work submitted to a lane."""

    job_id: str
    fn: Callable[..., Any] | None
    priority: int = 5
    deps: DependencySpec | None = None
    lane: str = ""
    gate: DependencyGate | None = field(default=None, repr=False)
    result: Any = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)


@dataclass
class LaneConfig:
    """Configuration for a worker lane."""

    name: str
    max_workers: int
    queue_size: int = 10
    requires_gpu: bool = False
    # Per-lane LLM backend assignment (None = share the global backend, current behavior).
    # These fields are read by LaneBackendManager (multi-LLM Phase 3); the WorkerPool
    # itself ignores them.
    model_profile: str | None = None
    device: str = "auto"  # "gpu" | "cpu" | "auto"
    n_gpu_layers: int = -1  # -1 = all on GPU, 0 = CPU only
    # Remote backend via OpenAI-compatible API (model server, Cloudflare tunnel, peer).
    remote_url: str | None = None
    remote_api_key: str | None = None
    remote_model: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# JobRegistry
# ─────────────────────────────────────────────────────────────────────────────


class JobRegistry:
    """Thread-safe registry tracking job lifecycle and enabling cross-job waits."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._results: dict[str, Any] = {}
        self._errors: dict[str, Exception] = {}
        self._lanes: dict[str, str] = {}  # job_id -> lane name
        self._timestamps: dict[str, float] = {}
        self._completed_ids: set[str] = set()  # survives prune() for dep safety

    def register(self, job: Job) -> None:
        """Register a job for lifecycle tracking."""
        with self._lock:
            self._jobs[job.job_id] = JobStatus.PENDING
            self._events[job.job_id] = threading.Event()
            self._lanes[job.job_id] = job.lane
            self._timestamps[job.job_id] = job.created_at

    def mark_running(self, job_id: str) -> None:
        """Mark a job as currently executing."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id] = JobStatus.RUNNING

    def mark_completed(self, job_id: str, result: Any = None) -> None:
        """Mark a job as completed and wake any waiters."""
        with self._lock:
            self._jobs[job_id] = JobStatus.COMPLETED
            self._results[job_id] = result
            self._completed_ids.add(job_id)
            lane = self._lanes.get(job_id, "?")
            event = self._events.get(job_id)
        if event:
            event.set()
        # Sim trace
        try:
            from maxim.simulation.sim_logger import sim_log

            sim_log(
                "PIPELINE", f"WorkerPool job completed: {job_id[:20]} (lane={lane}, has_result={result is not None})"
            )
        except Exception:
            pass

    def mark_failed(self, job_id: str, error: Exception) -> None:
        """Mark a job as failed and wake any waiters."""
        with self._lock:
            self._jobs[job_id] = JobStatus.FAILED
            self._errors[job_id] = error
            self._completed_ids.add(job_id)
            event = self._events.get(job_id)
        if event:
            event.set()

    def wait_for_completion(self, job_id: str, timeout: float) -> bool:
        """Block until job completes. Returns False on timeout.

        Raises ValueError for truly unknown job IDs (not GC'd completions).
        """
        with self._lock:
            event = self._events.get(job_id)
            if event is None:
                if job_id in self._completed_ids:
                    return True  # was completed, then GC'd
                raise ValueError(f"Unknown job dependency: {job_id}")
        return event.wait(timeout=timeout)

    def get_result(self, job_id: str) -> Any:
        """Get the result of a completed job."""
        with self._lock:
            return self._results.get(job_id)

    def get_error(self, job_id: str) -> Exception | None:
        """Get the error of a failed job."""
        with self._lock:
            return self._errors.get(job_id)

    def get_status(self, job_id: str) -> JobStatus | None:
        """Get the current status of a job."""
        with self._lock:
            return self._jobs.get(job_id)

    def pop_completed(self, lane: str) -> Job | None:
        """Return and remove the most recent completed job for a lane, or None."""
        with self._lock:
            for job_id, status in list(self._jobs.items()):
                if status == JobStatus.COMPLETED and self._lanes.get(job_id) == lane:
                    result = self._results.pop(job_id, None)
                    del self._jobs[job_id]
                    self._events.pop(job_id, None)
                    self._lanes.pop(job_id, None)
                    self._timestamps.pop(job_id, None)
                    return Job(job_id=job_id, fn=None, lane=lane, result=result)
        return None

    def prune(self, max_age_s: float = 300.0) -> int:
        """Remove completed/failed jobs older than max_age_s.

        Returns count of pruned jobs. _completed_ids is preserved so
        dependency waits on GC'd jobs still resolve correctly.
        """
        deadline = time.time() - max_age_s
        pruned = 0
        with self._lock:
            to_remove = []
            for job_id, status in self._jobs.items():
                if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    if self._timestamps.get(job_id, time.time()) < deadline:
                        event = self._events.get(job_id)
                        # Only prune if the event is set (all waiters notified)
                        if event and event.is_set():
                            to_remove.append(job_id)
            for job_id in to_remove:
                del self._jobs[job_id]
                self._events.pop(job_id, None)
                self._results.pop(job_id, None)
                self._errors.pop(job_id, None)
                self._lanes.pop(job_id, None)
                self._timestamps.pop(job_id, None)
                pruned += 1
        return pruned

    def cancel_by_lane(self, lane: str) -> int:
        """Cancel all pending jobs for a lane. Returns count cancelled."""
        cancelled = 0
        with self._lock:
            for job_id, status in list(self._jobs.items()):
                if status == JobStatus.PENDING and self._lanes.get(job_id) == lane:
                    self._jobs[job_id] = JobStatus.CANCELLED
                    event = self._events.get(job_id)
                    if event:
                        event.set()
                    cancelled += 1
        return cancelled


# ─────────────────────────────────────────────────────────────────────────────
# DependencyGate
# ─────────────────────────────────────────────────────────────────────────────


class DependencyGate:
    """Blocks a job until its prerequisites complete.

    Two-phase prefetch:
    - start_early_prefetch(): Kicks off immediately on submission
    - wait(): Blocks for deps, then runs late prefetch after resolution
    """

    def __init__(self, spec: DependencySpec, registry: JobRegistry) -> None:
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
        self._failed_deps = []

        # Wait for each dependency
        for job_id in self._spec.wait_for:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._failed_deps.append(job_id)
                continue
            try:
                if not self._registry.wait_for_completion(job_id, timeout=remaining):
                    self._failed_deps.append(job_id)
            except ValueError:
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


# ─────────────────────────────────────────────────────────────────────────────
# Lane
# ─────────────────────────────────────────────────────────────────────────────


class Lane:
    """A typed worker pool for a category of work."""

    _counter = itertools.count()

    def __init__(self, config: LaneConfig, registry: JobRegistry) -> None:
        self._config = config
        self._registry = registry
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=config.queue_size)
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix=f"Lane-{config.name}",
        )
        self._stop = threading.Event()
        self._dispatcher: threading.Thread | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def start(self) -> None:
        """Start the dispatcher thread."""
        self._stop.clear()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name=f"Lane-{self._config.name}-dispatch",
        )
        self._dispatcher.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the dispatcher and drain the executor."""
        self._stop.set()
        if self._dispatcher and self._dispatcher.is_alive():
            self._dispatcher.join(timeout=timeout)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def submit(self, job: Job) -> None:
        """Enqueue a job. Starts early prefetch immediately.

        Raises queue.Full if the lane queue is at capacity.
        """
        job.lane = self._config.name
        if job.deps:
            job.gate = DependencyGate(job.deps, self._registry)
            job.gate.start_early_prefetch()
        self._registry.register(job)
        self._queue.put_nowait((job.priority, next(self._counter), job))

    def cancel_pending(self) -> int:
        """Drain and cancel all pending jobs."""
        cancelled = 0
        while True:
            try:
                _, _, job = self._queue.get_nowait()
                self._registry.mark_failed(job.job_id, error=RuntimeError("Cancelled"))
                self._queue.task_done()
                cancelled += 1
            except queue.Empty:
                break
        return cancelled

    def _dispatch_loop(self) -> None:
        """Dequeue jobs; wait for deps in lightweight threads."""
        while not self._stop.is_set():
            try:
                _, _, job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Check if job was cancelled while queued
            status = self._registry.get_status(job.job_id)
            if status == JobStatus.CANCELLED:
                self._queue.task_done()
                continue

            if job.gate:
                threading.Thread(
                    target=self._wait_then_dispatch,
                    args=(job,),
                    daemon=True,
                    name=f"gate-{job.job_id}",
                ).start()
            else:
                self._executor.submit(self._execute_job, job)
            self._queue.task_done()

    def _wait_then_dispatch(self, job: Job) -> None:
        """Wait for deps in a gate-watcher thread, then hand off to executor."""
        prefetched, failed = job.gate.wait()
        if failed:
            logger.warning(
                "Job %s: deps timed out: %s (running anyway)",
                job.job_id,
                failed,
            )
        job._prefetched = prefetched
        self._executor.submit(self._execute_job, job)

    def _execute_job(self, job: Job) -> None:
        """Run the job callable. Deps already resolved by dispatcher."""
        # Check cancellation before executing
        status = self._registry.get_status(job.job_id)
        if status == JobStatus.CANCELLED:
            return

        self._registry.mark_running(job.job_id)
        prefetched = getattr(job, "_prefetched", None)
        try:
            result = job.fn(prefetched=prefetched) if prefetched else job.fn()
            self._registry.mark_completed(job.job_id, result=result)
        except Exception as e:
            logger.error("Job %s failed: %s", job.job_id, e)
            self._registry.mark_failed(job.job_id, error=e)


# ─────────────────────────────────────────────────────────────────────────────
# WorkerPool
# ─────────────────────────────────────────────────────────────────────────────


# ─── Tier-based lane defaults (Lane Tier Architecture) ───────────────────────

DEFAULT_TIERS: dict[str, LaneConfig] = {
    "large": LaneConfig(name="large", max_workers=1, requires_gpu=True),
    "medium": LaneConfig(name="medium", max_workers=1),
    "small": LaneConfig(name="small", max_workers=2),
}



class WorkerPool:
    """Central pool that owns all lanes and the job registry."""

    def __init__(
        self,
        lane_configs: dict[str, LaneConfig] | None = None,
        thread_registry: ThreadRegistry | None = None,
        function_router: FunctionRouter | None = None,
    ) -> None:
        self._registry = JobRegistry()
        configs = lane_configs or DEFAULT_TIERS
        self._lanes: dict[str, Lane] = {name: Lane(cfg, self._registry) for name, cfg in configs.items()}
        self._thread_registry = thread_registry
        self._function_router: FunctionRouter | None = function_router
        self._gc_thread: threading.Thread | None = None
        self._gc_stop = threading.Event()
        self._gc_interval_s = 60.0
        self._gc_max_age_s = 300.0

    @property
    def registry(self) -> JobRegistry:
        return self._registry

    def set_function_router(self, router: FunctionRouter) -> None:
        """Attach a FunctionRouter for ``submit_function()`` support."""
        self._function_router = router

    def start(self) -> None:
        """Start all lanes and the GC thread."""
        for lane in self._lanes.values():
            lane.start()
            if self._thread_registry and lane._dispatcher:
                self._thread_registry.register(f"Lane-{lane.name}", lane._dispatcher)

        # Start job GC thread
        self._gc_stop.clear()
        self._gc_thread = threading.Thread(
            target=self._gc_loop,
            daemon=True,
            name="WorkerPool-GC",
        )
        self._gc_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop all lanes and the GC thread."""
        self._gc_stop.set()
        if self._gc_thread and self._gc_thread.is_alive():
            self._gc_thread.join(timeout=1.0)

        per_lane = timeout / max(len(self._lanes), 1)
        for lane in self._lanes.values():
            lane.stop(timeout=per_lane)

    def _resolve_lane(self, lane: str) -> str:
        """Resolve a lane name."""
        if lane in self._lanes:
            return lane
        raise ValueError(
            f"Unknown lane: {lane!r} (available: {set(self._lanes)})"
        )

    def submit(
        self,
        lane: str,
        job_id: str,
        fn: Callable[..., Any],
        priority: int = 5,
        deps: DependencySpec | None = None,
    ) -> str:
        """Submit a job to a lane. Returns job_id."""
        resolved = self._resolve_lane(lane)
        job = Job(job_id=job_id, fn=fn, priority=priority, deps=deps)
        self._lanes[resolved].submit(job)
        return job_id

    def submit_function(
        self,
        function: str,
        job_id: str,
        fn: Callable[..., Any],
        priority: int = 5,
        deps: DependencySpec | None = None,
    ) -> str:
        """Submit a job by function name. FunctionRouter resolves the tier.

        Requires ``function_router`` to be set (via ``set_function_router``).
        Falls back to ``submit(lane=function, ...)`` if no router is set.
        """
        if self._function_router is not None:
            tier, boost = self._function_router.resolve(function)
            return self.submit(
                lane=tier,
                job_id=job_id,
                fn=fn,
                priority=priority + boost,
                deps=deps,
            )
        # No router — treat function name as a lane name (backward compat)
        return self.submit(lane=function, job_id=job_id, fn=fn, priority=priority, deps=deps)

    def wait_for(self, job_id: str, timeout: float = 30.0) -> bool:
        """Block until a specific job completes."""
        return self._registry.wait_for_completion(job_id, timeout)

    def get_result(self, job_id: str) -> Any:
        """Get the result of a completed job."""
        return self._registry.get_result(job_id)

    def get_completed(self, lane: str) -> Job | None:
        """Non-blocking poll: return the most recent completed job, or None."""
        return self._registry.pop_completed(lane)

    def cancel_lane(self, lane: str) -> int:
        """Drain and discard all pending jobs in a lane."""
        if lane not in self._lanes:
            return 0
        return self._lanes[lane].cancel_pending()

    def cancel_all(self) -> int:
        """Cancel everything across all lanes."""
        total = 0
        for lane in self._lanes.values():
            total += lane.cancel_pending()
        return total

    def status(self) -> dict[str, Any]:
        """Runtime introspection: queue depths, active jobs per lane."""
        info: dict[str, Any] = {}
        for name, lane in self._lanes.items():
            info[name] = {
                "queue_size": lane._queue.qsize(),
                "max_workers": lane._config.max_workers,
            }
        return info

    def _gc_loop(self) -> None:
        """Periodically prune completed jobs."""
        while not self._gc_stop.wait(timeout=self._gc_interval_s):
            pruned = self._registry.prune(max_age_s=self._gc_max_age_s)
            if pruned:
                logger.debug("WorkerPool GC: pruned %d completed jobs", pruned)


__all__ = [
    "DEFAULT_TIERS",
    "DependencyGate",
    "DependencySpec",
    "Job",
    "JobRegistry",
    "JobStatus",
    "Lane",
    "LaneConfig",
    "WorkerPool",
]
