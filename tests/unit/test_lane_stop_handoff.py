"""A lane never submits to its shut-down executor: a job that reaches the hand-off after ``stop()`` is failed.

CI logged ``RuntimeError: cannot schedule new futures after shutdown`` from ``Lane._dispatch_loop`` (main, run
45342248149, shard 2, during ``test_llm_worker_pool``): ``Lane.stop()`` joins the dispatcher with a timeout and
then shuts the executor down, so a dispatcher that outlived the join, or a gate-watcher thread (never joined), could
hand a job to the closed executor. That raised inside the thread, killed the dispatcher, and left the job PENDING, so
its waiters waited forever. Both hand-off paths go through ``Lane._hand_off`` now.
"""

from __future__ import annotations

from maxim.runtime.worker_pool import Job, JobRegistry, JobStatus, Lane, LaneConfig


def _stopped_lane() -> tuple[Lane, JobRegistry]:
    registry = JobRegistry()
    lane = Lane(LaneConfig(name="large", max_workers=1), registry)
    lane.start()
    lane.stop(timeout=1.0)
    return lane, registry


class _DoneGate:
    def wait(self) -> tuple[dict, list]:
        return {}, []


def test_the_dispatcher_hand_off_after_stop_fails_the_job() -> None:
    lane, registry = _stopped_lane()
    ran: list[str] = []
    job = Job(job_id="late", fn=lambda: ran.append("ran"), lane="large")
    registry.register(job)

    lane._hand_off(job)  # what a dispatcher that outlived stop()'s join does with the job it dequeued

    assert registry.get_status("late") == JobStatus.FAILED
    assert ran == []


def test_the_gate_watcher_hand_off_after_stop_fails_the_job() -> None:
    lane, registry = _stopped_lane()
    job = Job(job_id="gated", fn=lambda: None, lane="large")
    job.gate = _DoneGate()  # type: ignore[assignment]
    registry.register(job)

    lane._wait_then_dispatch(job)  # its deps resolved after the lane stopped

    assert registry.get_status("gated") == JobStatus.FAILED


def test_a_running_lane_still_runs_its_jobs() -> None:
    registry = JobRegistry()
    lane = Lane(LaneConfig(name="large", max_workers=1), registry)
    lane.start()
    try:
        job = Job(job_id="now", fn=lambda: 7)
        lane.submit(job)
        assert registry.wait_for_completion("now", timeout=5.0)
    finally:
        lane.stop(timeout=1.0)
    assert registry.get_status("now") == JobStatus.COMPLETED
