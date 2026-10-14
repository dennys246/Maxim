"""Loop liveness: the per-run state the heartbeat's stall rule reads (#1268 dormant producer 1).

``loop_state.loop_liveness_scope`` (on ``run_agentic_loop``) registers one entry per RUN with the process heartbeat
monitor's loop-state hook and clears it in a ``finally``; ``pre_tick_gate`` notes each tick working or idle, and
``tool_dispatch.execute_and_learn`` notes progress. Several loops share one process (phase-1 tests, multi-agent sims),
so the hook reports the worst WORKING loop and is cleared only when the last run ends.
"""

from __future__ import annotations

import threading

import pytest


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


def test_entries_are_per_run_and_the_hook_reports_the_worst_working_loop(monitor):
    from maxim.runtime import loop_state

    inside, release, done = threading.Event(), threading.Event(), threading.Event()

    @loop_state.loop_liveness_scope
    def long_run():
        live = loop_state.current_liveness()
        live.note_tick(running=True)
        live.last_progress_at -= 500.0
        inside.set()
        release.wait(5)

    t = threading.Thread(target=lambda: (long_run(), done.set()))
    t.start()
    assert inside.wait(5)

    @loop_state.loop_liveness_scope
    def short_run():
        loop_state.current_liveness().note_tick(running=False)
        state = monitor._loop_state_fn()
        assert state["state"] == "working" and state["idle_s"] >= 500.0 and state["loops"] == 2

    short_run()
    assert monitor.has_loop_state_hook  # the other run is still alive: its entry stays
    release.set()
    t.join(5)
    assert done.is_set() and not monitor.has_loop_state_hook


def test_run_agentic_loop_is_wrapped_and_keeps_its_signature():
    import inspect

    from maxim.runtime import agent_loop, loop_state

    fn = agent_loop.run_agentic_loop
    assert getattr(fn, "__wrapped__", None) is not None and fn in loop_state._LIVENESS_WRAPPED
    assert inspect.signature(fn) == inspect.signature(fn.__wrapped__)


def test_the_gate_and_dispatch_note_liveness(monitor):
    """pre_tick_gate notes a tick (working or idle) and execute_and_learn notes progress: structurally, through the one
    decorator each, so no return path of the gate escapes it."""
    from maxim.runtime import loop_gates, loop_state, tool_dispatch

    assert loop_gates.pre_tick_gate in loop_state._LIVENESS_WRAPPED
    assert tool_dispatch.execute_and_learn in loop_state._LIVENESS_WRAPPED


def test_a_running_tool_marks_the_run_acting_then_progressed(monitor):
    """A long tool is not a stall: the run is "acting" while execute_and_learn runs, then working with fresh
    progress, however the call exits."""
    from maxim.runtime import loop_state

    seen: list[str] = []

    @loop_state.notes_loop_action
    def tool(fail: bool) -> None:
        seen.append(loop_state.current_liveness().state)
        if fail:
            raise RuntimeError("tool broke")

    @loop_state.loop_liveness_scope
    def run():
        live = loop_state.current_liveness()
        live.note_tick(running=True)
        for fail in (False, True):
            live.last_progress_at -= 500.0
            stale = live.last_progress_at
            if fail:
                with pytest.raises(RuntimeError):
                    tool(fail)
            else:
                tool(fail)
            assert live.state == "working" and live.last_progress_at > stale

    run()
    assert seen == ["acting", "acting"]
