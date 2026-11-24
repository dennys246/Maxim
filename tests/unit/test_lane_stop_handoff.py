"""A stopped lane never strands a waiter: every job it did not run ends FAILED, and it never submits to its
shut-down executor.

CI logged ``RuntimeError: cannot schedule new futures after shutdown`` from ``Lane._dispatch_loop`` (main, run
45342248149, shard 2, during ``test_llm_worker_pool``): ``Lane.stop()`` joined the dispatcher with a timeout and then
shut the executor down, so a dispatcher that outlived the join, or a gate-watcher thread (never joined), handed a job
to the closed executor; the thread died and the job stayed PENDING. The review round found the same hang on three more
paths: futures the shutdown cancelled, jobs still queued at stop, and a submit after stop. Every case is here.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from maxim.runtime.worker_pool import Job, JobRegistry, JobStatus, Lane, LaneConfig


def _lane(registry: JobRegistry) -> Lane:
    return Lane(LaneConfig(name="large", max_workers=1), registry)


def _thread_errors(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    errors: list[Any] = []
    monkeypatch.setattr(threading, "excepthook", errors.append)
    return errors


def test_a_dispatcher_that_outlived_the_join_fails_its_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real dispatch loop dequeues a job after the lane closed and its executor shut down (what stop() does
    once a timed-out join gives up on the dispatcher)."""
    errors = _thread_errors(monkeypatch)
    registry = JobRegistry()
    lane = _lane(registry)
    lane.submit(Job(job_id="late", fn=lambda: None))
    with lane._handoff_lock:
        lane._closed = True
    lane._executor.shutdown(wait=True)

    dispatcher = threading.Thread(target=lane._dispatch_loop, daemon=True)
    dispatcher.start()
    try:
        assert registry.wait_for_completion("late", timeout=3.0), "the job was stranded PENDING"
    finally:
        lane._stop.set()
        dispatcher.join(timeout=2.0)
    assert registry.get_status("late") == JobStatus.FAILED
    assert not errors, f"the dispatcher thread raised: {errors}"


class _DoneGate:
    def wait(self) -> tuple[dict, list]:
        return {}, []


def test_a_gate_watcher_after_stop_fails_its_job() -> None:
    registry = JobRegistry()
    lane = _lane(registry)
    lane.start()
    lane.stop(timeout=1.0)
    job = Job(job_id="gated", fn=lambda: None, lane="large")
    job.gate = _DoneGate()  # type: ignore[assignment]
    registry.register(job)

    lane._wait_then_dispatch(job)  # its deps resolved after the lane stopped

    assert registry.get_status("gated") == JobStatus.FAILED


def test_jobs_queued_behind_a_running_one_fail_at_stop() -> None:
    """Futures the shutdown cancels, and jobs still in the lane queue, both end FAILED."""
    registry = JobRegistry()
    lane = _lane(registry)
    lane.start()
    release = threading.Event()
    lane.submit(Job(job_id="busy", fn=lambda: release.wait(5.0)))
    for i in range(3):
        lane.submit(Job(job_id=f"q{i}", fn=lambda: None))
    time.sleep(0.3)  # the dispatcher hands the queued jobs to the executor behind "busy"
    stopper = threading.Thread(target=lane.stop, kwargs={"timeout": 1.0})
    stopper.start()
    time.sleep(0.2)
    release.set()
    stopper.join(timeout=5.0)
    for i in range(3):
        assert registry.wait_for_completion(f"q{i}", timeout=1.0), f"q{i} was stranded PENDING"
        assert registry.get_status(f"q{i}") == JobStatus.FAILED


def test_a_job_still_in_the_queue_at_stop_fails() -> None:
    registry = JobRegistry()
    lane = _lane(registry)  # never started: nothing dequeues
    lane.submit(Job(job_id="never", fn=lambda: None))
    lane.stop(timeout=0.1)
    assert registry.get_status("never") == JobStatus.FAILED


def test_a_stopped_lane_refuses_new_jobs_and_a_restart() -> None:
    registry = JobRegistry()
    lane = _lane(registry)
    lane.start()
    lane.stop(timeout=1.0)
    with pytest.raises(RuntimeError, match="stopped"):
        lane.submit(Job(job_id="after", fn=lambda: None))
    assert registry.get_status("after") is None, "a refused job was left registered"
    with pytest.raises(RuntimeError, match="stopped"):
        lane.start()


def test_a_cancelled_job_stays_cancelled() -> None:
    registry = JobRegistry()
    lane = _lane(registry)
    lane.start()
    lane.stop(timeout=1.0)
    job = Job(job_id="c", fn=lambda: None, lane="large")
    registry.register(job)
    assert registry.cancel_by_lane("large") == 1
    lane._hand_off(job)
    assert registry.get_status("c") == JobStatus.CANCELLED


def test_a_running_lane_still_runs_its_jobs() -> None:
    registry = JobRegistry()
    lane = _lane(registry)
    lane.start()
    try:
        lane.submit(Job(job_id="now", fn=lambda: 7))
        assert registry.wait_for_completion("now", timeout=5.0)
    finally:
        lane.stop(timeout=1.0)
    assert registry.get_status("now") == JobStatus.COMPLETED
