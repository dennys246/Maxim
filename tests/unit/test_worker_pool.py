"""Unit tests for parallel worker pool."""

from __future__ import annotations

import threading
import time

import pytest

from maxim.runtime.worker_pool import (
    DependencyGate,
    DependencySpec,
    Job,
    JobRegistry,
    JobStatus,
    Lane,
    LaneConfig,
    WorkerPool,
)


# ─────────────────────────────────────────────────────────────────────────────
# JobRegistry
# ─────────────────────────────────────────────────────────────────────────────


class TestJobRegistry:
    def test_register_and_status(self):
        reg = JobRegistry()
        job = Job(job_id="j1", fn=lambda: None, lane="infer")
        reg.register(job)
        assert reg.get_status("j1") == JobStatus.PENDING

    def test_mark_running(self):
        reg = JobRegistry()
        job = Job(job_id="j1", fn=lambda: None, lane="infer")
        reg.register(job)
        reg.mark_running("j1")
        assert reg.get_status("j1") == JobStatus.RUNNING

    def test_mark_completed(self):
        reg = JobRegistry()
        job = Job(job_id="j1", fn=lambda: None, lane="infer")
        reg.register(job)
        reg.mark_completed("j1", result=42)
        assert reg.get_status("j1") == JobStatus.COMPLETED
        assert reg.get_result("j1") == 42

    def test_completed_ids_survives(self):
        """_completed_ids persists after prune so dep waits still resolve."""
        reg = JobRegistry()
        job = Job(job_id="j1", fn=lambda: None, lane="infer")
        reg.register(job)
        reg.mark_completed("j1", result="ok")
        reg.prune(max_age_s=0)
        # Job is pruned from _jobs but _completed_ids preserves it
        assert reg.get_status("j1") is None
        assert reg.wait_for_completion("j1", timeout=0.1) is True

    def test_mark_failed(self):
        reg = JobRegistry()
        job = Job(job_id="j1", fn=lambda: None, lane="infer")
        reg.register(job)
        err = RuntimeError("boom")
        reg.mark_failed("j1", error=err)
        assert reg.get_status("j1") == JobStatus.FAILED
        assert reg.get_error("j1") is err

    def test_wait_for_completion_blocks_then_resolves(self):
        reg = JobRegistry()
        job = Job(job_id="j1", fn=lambda: None, lane="infer")
        reg.register(job)

        threading.Event()

        def completer():
            time.sleep(0.1)
            reg.mark_completed("j1", result="done")

        threading.Thread(target=completer, daemon=True).start()
        result = reg.wait_for_completion("j1", timeout=2.0)
        assert result is True
        assert reg.get_result("j1") == "done"

    def test_wait_for_completion_timeout(self):
        reg = JobRegistry()
        job = Job(job_id="j1", fn=lambda: None, lane="infer")
        reg.register(job)
        result = reg.wait_for_completion("j1", timeout=0.05)
        assert result is False

    def test_wait_for_unknown_raises(self):
        reg = JobRegistry()
        with pytest.raises(ValueError, match="Unknown job dependency"):
            reg.wait_for_completion("nonexistent", timeout=0.1)

    def test_pop_completed_returns_and_removes(self):
        reg = JobRegistry()
        job = Job(job_id="j1", fn=lambda: None, lane="infer")
        reg.register(job)
        reg.mark_completed("j1", result="val")

        popped = reg.pop_completed("infer")
        assert popped is not None
        assert popped.job_id == "j1"
        assert popped.result == "val"

        # Second pop returns None
        assert reg.pop_completed("infer") is None

    def test_pop_completed_filters_by_lane(self):
        reg = JobRegistry()
        j1 = Job(job_id="j1", fn=lambda: None, lane="infer")
        j2 = Job(job_id="j2", fn=lambda: None, lane="record")
        reg.register(j1)
        reg.register(j2)
        reg.mark_completed("j1", result="a")
        reg.mark_completed("j2", result="b")

        assert reg.pop_completed("record").job_id == "j2"
        assert reg.pop_completed("record") is None
        assert reg.pop_completed("infer").job_id == "j1"

    def test_cancel_by_lane(self):
        reg = JobRegistry()
        j1 = Job(job_id="j1", fn=lambda: None, lane="infer")
        j2 = Job(job_id="j2", fn=lambda: None, lane="record")
        reg.register(j1)
        reg.register(j2)

        cancelled = reg.cancel_by_lane("infer")
        assert cancelled == 1
        assert reg.get_status("j1") == JobStatus.CANCELLED
        assert reg.get_status("j2") == JobStatus.PENDING

    def test_prune_removes_old_jobs(self):
        reg = JobRegistry()
        job = Job(job_id="j1", fn=lambda: None, lane="infer")
        reg.register(job)
        reg.mark_completed("j1", result="ok")

        pruned = reg.prune(max_age_s=0)
        assert pruned == 1
        assert reg.get_status("j1") is None
        # But completed_ids survives
        assert "j1" in reg._completed_ids


# ─────────────────────────────────────────────────────────────────────────────
# DependencyGate
# ─────────────────────────────────────────────────────────────────────────────


class TestDependencyGate:
    def test_no_deps_no_prefetch(self):
        """Gate with no deps and no prefetch resolves immediately."""
        reg = JobRegistry()
        spec = DependencySpec()
        gate = DependencyGate(spec, reg)
        gate.start_early_prefetch()
        result, failed = gate.wait()
        assert failed == []
        assert result == {"early": None, "late": None}

    def test_early_prefetch_runs_on_submit(self):
        """Early prefetch data is available after wait."""
        reg = JobRegistry()
        spec = DependencySpec(
            prefetch_early=lambda: {"agent_states": [1, 2, 3]}
        )
        gate = DependencyGate(spec, reg)
        gate.start_early_prefetch()
        result, failed = gate.wait()
        assert result["early"] == {"agent_states": [1, 2, 3]}
        assert failed == []

    def test_late_prefetch_runs_after_deps(self):
        """Late prefetch runs after deps resolve, gets fresh data."""
        reg = JobRegistry()
        shared = {"value": 0}

        dep_job = Job(job_id="dep1", fn=lambda: None, lane="record")
        reg.register(dep_job)

        spec = DependencySpec(
            wait_for=["dep1"],
            prefetch_late=lambda: {"fresh": shared["value"]},
        )
        gate = DependencyGate(spec, reg)
        gate.start_early_prefetch()

        def resolve_dep():
            time.sleep(0.05)
            shared["value"] = 42
            reg.mark_completed("dep1")

        threading.Thread(target=resolve_dep, daemon=True).start()

        result, failed = gate.wait()
        assert failed == []
        assert result["late"]["fresh"] == 42

    def test_dep_timeout_reports_failed(self):
        """Timed-out deps are reported in failed_deps."""
        reg = JobRegistry()
        dep_job = Job(job_id="slow", fn=lambda: None, lane="record")
        reg.register(dep_job)

        spec = DependencySpec(wait_for=["slow"], timeout_s=0.05)
        gate = DependencyGate(spec, reg)
        gate.start_early_prefetch()

        result, failed = gate.wait()
        assert "slow" in failed

    def test_unknown_dep_reports_failed(self):
        """Unknown dep IDs appear in failed_deps (not exception)."""
        reg = JobRegistry()
        spec = DependencySpec(wait_for=["ghost"], timeout_s=0.1)
        gate = DependencyGate(spec, reg)
        gate.start_early_prefetch()

        result, failed = gate.wait()
        assert "ghost" in failed

    def test_early_prefetch_failure_doesnt_crash(self):
        """Failed early prefetch logs warning but gate still works."""
        reg = JobRegistry()
        spec = DependencySpec(
            prefetch_early=lambda: 1 / 0  # ZeroDivisionError
        )
        gate = DependencyGate(spec, reg)
        gate.start_early_prefetch()
        result, failed = gate.wait()
        assert result["early"] is None  # failed, so None
        assert failed == []

    def test_two_phase_prefetch_timing(self):
        """Early prefetch overlaps with dep wait; late runs after."""
        reg = JobRegistry()
        timeline = []

        dep_job = Job(job_id="dep1", fn=lambda: None, lane="record")
        reg.register(dep_job)

        def early():
            timeline.append(("early_start", time.monotonic()))
            time.sleep(0.05)
            timeline.append(("early_end", time.monotonic()))
            return "early_data"

        def late():
            timeline.append(("late", time.monotonic()))
            return "late_data"

        spec = DependencySpec(
            wait_for=["dep1"],
            prefetch_early=early,
            prefetch_late=late,
        )
        gate = DependencyGate(spec, reg)
        gate.start_early_prefetch()

        # Resolve dep after early prefetch starts
        def resolve():
            time.sleep(0.08)
            reg.mark_completed("dep1")

        threading.Thread(target=resolve, daemon=True).start()

        result, failed = gate.wait()
        assert result["early"] == "early_data"
        assert result["late"] == "late_data"
        assert failed == []

        # Early started before late
        early_start = next(t for name, t in timeline if name == "early_start")
        late_time = next(t for name, t in timeline if name == "late")
        assert early_start < late_time


# ─────────────────────────────────────────────────────────────────────────────
# Lane
# ─────────────────────────────────────────────────────────────────────────────


class TestLane:
    def test_submit_and_execute(self):
        """Jobs submitted to a lane execute and complete."""
        reg = JobRegistry()
        lane = Lane(LaneConfig("test", max_workers=1, queue_size=5), reg)
        lane.start()

        result_holder = []
        job = Job(
            job_id="j1",
            fn=lambda: result_holder.append(1) or "ok",
        )
        lane.submit(job)

        # Wait for completion
        assert reg.wait_for_completion("j1", timeout=2.0) is True
        assert reg.get_result("j1") == "ok"
        assert result_holder == [1]

        lane.stop(timeout=2.0)

    def test_priority_ordering(self):
        """Higher priority (lower number) jobs execute first."""
        reg = JobRegistry()
        lane = Lane(LaneConfig("test", max_workers=1, queue_size=10), reg)

        execution_order = []
        gate = threading.Event()

        # Block the lane with a sentinel job
        sentinel = Job(
            job_id="sentinel",
            fn=lambda: gate.wait(timeout=2.0),
            priority=0,
        )
        lane.submit(sentinel)
        lane.start()
        time.sleep(0.05)  # let sentinel start executing

        # Submit jobs in reverse priority order
        for i in range(3):
            p = 3 - i  # priorities: 3, 2, 1
            job = Job(
                job_id=f"j{p}",
                fn=lambda _p=p: execution_order.append(_p),
                priority=p,
            )
            lane.submit(job)

        # Release sentinel
        gate.set()

        # Wait for all
        for i in range(1, 4):
            reg.wait_for_completion(f"j{i}", timeout=2.0)

        lane.stop(timeout=2.0)
        assert execution_order == [1, 2, 3]

    def test_cancel_pending(self):
        """cancel_pending drains queued jobs."""
        reg = JobRegistry()
        lane = Lane(LaneConfig("test", max_workers=1, queue_size=10), reg)
        # Don't start the lane — jobs stay in queue

        for i in range(3):
            lane.submit(Job(job_id=f"j{i}", fn=lambda: None))

        cancelled = lane.cancel_pending()
        assert cancelled == 3

    def test_job_failure_marks_failed(self):
        """A job that raises is marked FAILED, not COMPLETED."""
        reg = JobRegistry()
        lane = Lane(LaneConfig("test", max_workers=1, queue_size=5), reg)
        lane.start()

        def bad_fn():
            raise ValueError("test error")

        job = Job(job_id="j1", fn=bad_fn)
        lane.submit(job)

        reg.wait_for_completion("j1", timeout=2.0)
        assert reg.get_status("j1") == JobStatus.FAILED
        assert "test error" in str(reg.get_error("j1"))

        lane.stop(timeout=2.0)

    def test_equal_priority_tiebreaker(self):
        """Jobs with equal priority don't crash (Issue #3)."""
        reg = JobRegistry()
        lane = Lane(LaneConfig("test", max_workers=1, queue_size=10), reg)
        lane.start()

        for i in range(5):
            lane.submit(Job(job_id=f"j{i}", fn=lambda: None, priority=5))

        for i in range(5):
            assert reg.wait_for_completion(f"j{i}", timeout=2.0) is True

        lane.stop(timeout=2.0)


# ─────────────────────────────────────────────────────────────────────────────
# WorkerPool
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkerPool:
    def test_submit_and_poll(self):
        """Submit a job and poll for completion."""
        pool = WorkerPool(
            lane_configs={"infer": LaneConfig("infer", max_workers=1)}
        )
        pool.start()

        pool.submit("infer", "j1", fn=lambda: 42)
        assert pool.wait_for("j1", timeout=2.0) is True
        assert pool.get_result("j1") == 42

        pool.stop(timeout=2.0)

    def test_get_completed(self):
        """get_completed returns and removes a completed job."""
        pool = WorkerPool(
            lane_configs={"infer": LaneConfig("infer", max_workers=1)}
        )
        pool.start()

        pool.submit("infer", "j1", fn=lambda: "result")
        pool.wait_for("j1", timeout=2.0)

        completed = pool.get_completed("infer")
        assert completed is not None
        assert completed.result == "result"

        # Second poll returns None
        assert pool.get_completed("infer") is None

        pool.stop(timeout=2.0)

    def test_unknown_lane_raises(self):
        pool = WorkerPool(
            lane_configs={"infer": LaneConfig("infer", max_workers=1)}
        )
        with pytest.raises(ValueError, match="Unknown lane"):
            pool.submit("bogus", "j1", fn=lambda: None)

    def test_cancel_lane(self):
        """cancel_lane drains pending jobs from a specific lane."""
        pool = WorkerPool(
            lane_configs={
                "infer": LaneConfig("infer", max_workers=1, queue_size=10),
                "record": LaneConfig("record", max_workers=1, queue_size=10),
            }
        )
        # Don't start — jobs stay queued

        pool.submit("infer", "i1", fn=lambda: None)
        pool.submit("infer", "i2", fn=lambda: None)
        pool.submit("record", "r1", fn=lambda: None)

        cancelled = pool.cancel_lane("infer")
        assert cancelled == 2

    def test_cancel_all(self):
        pool = WorkerPool(
            lane_configs={
                "infer": LaneConfig("infer", max_workers=1, queue_size=10),
                "record": LaneConfig("record", max_workers=1, queue_size=10),
            }
        )
        pool.submit("infer", "i1", fn=lambda: None)
        pool.submit("record", "r1", fn=lambda: None)

        cancelled = pool.cancel_all()
        assert cancelled == 2

    def test_dependency_between_jobs(self):
        """Job with deps waits for prerequisite to complete."""
        pool = WorkerPool(
            lane_configs={
                "record": LaneConfig("record", max_workers=1),
                "infer": LaneConfig("infer", max_workers=1),
            }
        )
        pool.start()

        order = []

        pool.submit(
            "record",
            "rec-1",
            fn=lambda: (time.sleep(0.1), order.append("record"))[1],
        )
        pool.submit(
            "infer",
            "inf-1",
            fn=lambda prefetched=None: order.append("infer"),
            deps=DependencySpec(wait_for=["rec-1"]),
        )

        pool.wait_for("inf-1", timeout=3.0)
        assert order == ["record", "infer"]

        pool.stop(timeout=2.0)

    def test_prefetch_with_dependency(self):
        """Prefetch data is passed to the job function."""
        pool = WorkerPool(
            lane_configs={"infer": LaneConfig("infer", max_workers=1)}
        )
        pool.start()

        result_holder = []

        def job_fn(prefetched=None):
            result_holder.append(prefetched)
            return prefetched

        pool.submit(
            "infer",
            "j1",
            fn=job_fn,
            deps=DependencySpec(
                prefetch_early=lambda: {"stable": True},
                prefetch_late=lambda: {"fresh": True},
            ),
        )

        pool.wait_for("j1", timeout=2.0)
        assert len(result_holder) == 1
        assert result_holder[0]["early"] == {"stable": True}
        assert result_holder[0]["late"] == {"fresh": True}

        pool.stop(timeout=2.0)

    def test_status(self):
        """status() returns lane info."""
        pool = WorkerPool(
            lane_configs={
                "infer": LaneConfig("infer", max_workers=2),
                "record": LaneConfig("record", max_workers=1),
            }
        )
        info = pool.status()
        assert "infer" in info
        assert info["infer"]["max_workers"] == 2
        assert "record" in info

    def test_start_stop_lifecycle(self):
        """Pool can start and stop cleanly."""
        pool = WorkerPool()
        pool.start()
        pool.stop(timeout=2.0)

    def test_multiple_jobs_concurrent_execution(self):
        """Multiple jobs in the same lane execute."""
        pool = WorkerPool(
            lane_configs={"work": LaneConfig("work", max_workers=2)}
        )
        pool.start()

        results = []
        lock = threading.Lock()

        def work(val):
            def fn():
                with lock:
                    results.append(val)
                return val
            return fn

        for i in range(5):
            pool.submit("work", f"j{i}", fn=work(i))

        for i in range(5):
            pool.wait_for(f"j{i}", timeout=3.0)

        assert sorted(results) == [0, 1, 2, 3, 4]

        pool.stop(timeout=2.0)
