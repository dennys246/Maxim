"""Loop liveness: what the heartbeat's stall rule reads (#1268 dormant producer 1).

A stall is a STUCK PLANNING TURN (owner decision 2026-10-14): ``loop_state.loop_liveness_scope`` (on
``run_agentic_loop``) registers one entry per RUN with the process heartbeat monitor's loop-state hook and clears it in
a ``finally``; ``pre_tick_gate`` records the run's ``llm_worker``; the hook reports the longest-outstanding planning
attempt over every live run (``LLMWorker.outstanding_attempt_s``). Waiting for a human or running a tool is no attempt.
"""

from __future__ import annotations

import threading
import time

import pytest


class _Worker:
    def __init__(self, outstanding: float | None) -> None:
        self.outstanding = outstanding

    def outstanding_attempt_s(self) -> float | None:
        return self.outstanding


@pytest.fixture()
def monitor(monkeypatch):
    from maxim.runtime import heartbeat

    mon = heartbeat.HeartbeatMonitor(interval_s=3600.0)
    monkeypatch.setattr(heartbeat, "get_heartbeat_monitor", lambda: mon)
    return mon


def test_a_run_registers_and_clears_its_entry_even_when_it_raises(monitor):
    from maxim.runtime import loop_state

    @loop_state.loop_liveness_scope
    def run():
        assert monitor.has_loop_state_hook and loop_state.current_liveness() is not None
        raise RuntimeError("loop died")

    with pytest.raises(RuntimeError):
        run()
    assert not monitor.has_loop_state_hook and loop_state.current_liveness() is None


def test_entries_are_per_run_and_the_hook_reports_the_longest_outstanding_turn(monitor):
    from maxim.runtime import loop_state

    inside, release, done = threading.Event(), threading.Event(), threading.Event()

    @loop_state.notes_loop_worker
    def tick(*, llm_worker):
        return None

    @loop_state.loop_liveness_scope
    def long_run():
        tick(llm_worker=_Worker(500.0))
        inside.set()
        release.wait(5)

    t = threading.Thread(target=lambda: (long_run(), done.set()))
    t.start()
    assert inside.wait(5)

    @loop_state.loop_liveness_scope
    def short_run():
        tick(llm_worker=_Worker(None))  # waiting for a human: no attempt outstanding
        assert monitor._loop_state_fn() == {"state": "planning", "outstanding_s": 500.0, "loops": 2}

    short_run()
    assert monitor.has_loop_state_hook  # the other run is still alive: its entry stays
    release.set()
    t.join(5)
    assert done.is_set() and not monitor.has_loop_state_hook


def test_no_outstanding_turn_is_idle_and_an_unreadable_worker_is_no_evidence(monitor):
    from maxim.runtime import loop_state

    class _Broken:
        def outstanding_attempt_s(self):
            raise RuntimeError("worker torn down")

    @loop_state.notes_loop_worker
    def tick(*, llm_worker):
        return None

    @loop_state.loop_liveness_scope
    def run():
        tick(llm_worker=None)
        assert monitor._loop_state_fn() == {"state": "idle", "outstanding_s": 0.0, "loops": 1}
        tick(llm_worker=_Broken())
        assert monitor._loop_state_fn()["state"] == "idle"

    run()


def test_run_agentic_loop_is_wrapped_and_keeps_its_signature():
    import inspect

    from maxim.runtime import agent_loop, loop_state

    fn = agent_loop.run_agentic_loop
    assert getattr(fn, "__wrapped__", None) is not None and fn in loop_state._LIVENESS_WRAPPED
    assert inspect.signature(fn) == inspect.signature(fn.__wrapped__)


def test_the_gate_records_the_runs_worker():
    """pre_tick_gate records the run's llm_worker on every tick, through the one decorator."""
    import inspect

    from maxim.runtime import loop_gates, loop_state

    assert loop_gates.pre_tick_gate in loop_state._LIVENESS_WRAPPED
    assert "llm_worker" in inspect.signature(loop_gates.pre_tick_gate).parameters


def test_a_real_worker_reports_its_planning_turn_until_the_loop_consumes_it():
    """LLMWorker.outstanding_attempt_s: an age while the attempt is queued, running or unconsumed; None once the loop
    pops the result."""
    from maxim.agents.llm_worker import LLMWorker
    from tests.unit.test_llm_worker_pool import FakeLLM, _submit_test_context

    worker = LLMWorker(llm=FakeLLM(delay=0.3), stale_threshold_s=30.0)
    worker.start()
    try:
        assert worker.outstanding_attempt_s() is None
        assert _submit_test_context(worker)
        assert worker.outstanding_attempt_s() is not None
        proposal = None
        for _ in range(60):
            proposal = worker.get_latest_proposal()
            if proposal is not None:
                break
            time.sleep(0.1)
        assert proposal is not None and worker.outstanding_attempt_s() is None
    finally:
        worker.stop()
