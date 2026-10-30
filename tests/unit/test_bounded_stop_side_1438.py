"""Red gates for bounded stop PR 1b, the stop side (#1438; docs/plans/bounded_stop.md §5).

A process stop today tells nothing to stop: an in-flight LLM call, a running loop and a console drain each run to
their caps, and a loop that exits cleanly ends its session unbounded. Each gate below is
``xfail(strict=True, raises=AssertionError)``: it fails on the pre-fix code by assertion, for the reason named, and
flips when PR 1b lands. A gate that needs a surface PR 1b adds asserts that surface first, so its absence is the
assertion that fails.
"""

from __future__ import annotations

import inspect
import signal
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.utils import cancellation
from maxim.utils.cancellation import current_stop, request_stop
from maxim.utils.deadline import Deadline

red = pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1438 PR 1b: the stop side")


# ── G1: the LLM worker ──────────────────────────────────────────────────────────────────────────────────────────


class _BlockingLLM:
    """``generate_json`` blocks until released or until the call's cancel event is set."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def generate_json(self, *_a: Any, **_k: Any) -> dict[str, Any]:
        from maxim.utils.cancellation import current_cancel_event

        self.started.set()
        while not self.release.is_set():
            ev = current_cancel_event()
            if ev is not None and ev.is_set():
                raise RuntimeError("cancelled")
            time.sleep(0.01)
        return {"ok": True}


@red
def test_g1_a_stop_request_abandons_an_in_flight_llm_call() -> None:
    from maxim.agents.llm_worker import LLMWorker

    llm = _BlockingLLM()
    worker = LLMWorker(llm=llm, llm_timeout_s=6.0)
    worker.start()
    try:
        result: list[Any] = []
        caller = threading.Thread(target=lambda: result.append(worker._call_llm_with_timeout("p", 0.0, 8)))
        caller.start()
        assert llm.started.wait(5.0)
        t0 = time.monotonic()
        request_stop("test", Deadline.after(5.0))
        caller.join(timeout=8.0)
        elapsed = time.monotonic() - t0
    finally:
        llm.release.set()
        worker.stop()
    assert elapsed < 2.0, f"the call ran {elapsed:.1f}s after the stop request (its timeout, not the stop)"


@red
def test_g1_a_worker_started_after_a_stop_request_is_born_stopped() -> None:
    from maxim.agents.llm_worker import LLMWorker

    request_stop("test", Deadline.after(5.0))
    worker = LLMWorker(llm=_BlockingLLM(), llm_timeout_s=1.0)
    worker.start()
    try:
        assert worker._stop_event.is_set(), "start() cleared the stop event and nothing re-set it"
    finally:
        worker.stop()


# ── G2: the loop's turn boundary ────────────────────────────────────────────────────────────────────────────────


@red
def test_g2_a_loop_with_no_stop_event_breaks_at_its_next_tick_after_a_stop_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maxim.runtime import loop_gates
    from maxim.runtime.loop_gates import GateOutcome, pre_tick_gate

    monkeypatch.setattr(loop_gates, "operational_mode", lambda *_a: None)  # past section 0, the pass idles

    request_stop("test", Deadline.after(5.0))
    state = MagicMock()
    state.data = {}
    outcome = pre_tick_gate(
        step_num=1,
        stop_event=None,
        state=state,
        executor=MagicMock(),
        ctrl=MagicMock(),
        autonomy_controller=MagicMock(is_paused=True),
        sim=MagicMock(),
        percept_source=None,
        llm_worker=None,
        planning_liveness_on=False,
        aut_mode="free",
        experience_driver=MagicMock(),
        idle_sleep_s=0.0,
    )
    assert outcome is GateOutcome.BREAK


# ── G3: a loop's clean exit under a stop is bounded by the stop's deadline (OD3) ───────────────────────────────


def _agent(tmp_path: Path, agent_id: str = "stop_agent") -> Any:
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory
    from maxim.tools.registry import ToolRegistry

    config = AgentConfig(
        agent_id=agent_id,
        role="pc",
        persistence_dir=str(tmp_path / agent_id),
        with_bio_stack=True,
        with_executor=False,
        with_pain_bridge=False,
        with_fear_gate=False,
    )
    return AgentFactory(base_data_dir=tmp_path).create_full_agent(config, tool_registry=ToolRegistry(), auto_load=True)


@red
def test_g3_a_clean_loop_exit_under_a_stop_saves_and_skips_consolidation_past_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maxim.runtime.bio_integration import bio_session

    inst = _agent(tmp_path)
    hub = inst.memory_hub
    inst.hippocampus.store_observation("the blue cup is on the shelf")
    inst.hippocampus.flush(timeout=5.0)
    slept: list[bool] = []
    monkeypatch.setattr(hub.hippocampus, "sleep", lambda *a, **k: (slept.append(True), time.sleep(3.0), {})[2])

    with bio_session(memory_hub=hub, hippocampus=inst.hippocampus, owner=object(), consolidation="full"):
        request_stop("test", Deadline.after(1.0))

    assert Path(inst.hippocampus.config.persistence_path).expanduser().exists()
    assert slept == [], "sleep ran past the stop's deadline: the clean exit ended the session unbounded"


# ── G4/G12: maxim serve arms the stop when uvicorn shuts down ──────────────────────────────────────────────────


class _RecordingHandle:
    def __init__(self) -> None:
        self.reasons: list[str | None] = []

    def stop(self, **_kw: Any) -> bool:
        req = current_stop()
        self.reasons.append(req.reason if req is not None else None)
        return True


def _serve_in_thread(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, threading.Thread, _RecordingHandle]:
    import uvicorn

    import maxim.console.server as srv

    assert hasattr(srv, "_MaximServer"), "maxim serve runs plain uvicorn.run: nothing arms the stop on shutdown"
    handle = _RecordingHandle()
    monkeypatch.setattr(srv, "_handle", handle)
    monkeypatch.setattr(srv, "_active_run", {"session_id": None, "thread": None})
    app = srv.build_app(None, auth_token="t" * 32, stop_budget_s=2.0)
    server = srv._MaximServer(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"), stop_budget_s=2.0)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started
    return server, thread, handle


@red
def test_g4_a_sigterm_requests_the_stop_before_the_lifespan_drains(monkeypatch: pytest.MonkeyPatch) -> None:
    server, thread, handle = _serve_in_thread(monkeypatch)
    t0 = time.monotonic()
    server.handle_exit(signal.SIGTERM, None)
    thread.join(timeout=6.0)
    assert not thread.is_alive() and time.monotonic() - t0 < 4.0
    assert handle.reasons and (handle.reasons[0] or "").startswith("signal"), handle.reasons


@red
def test_g12_a_shutdown_with_no_signal_still_requests_the_stop_and_drains(monkeypatch: pytest.MonkeyPatch) -> None:
    server, thread, handle = _serve_in_thread(monkeypatch)
    server.should_exit = True
    thread.join(timeout=6.0)
    assert not thread.is_alive()
    assert handle.reasons == ["server shutdown"]


# ── G5: the lifespan refuses to start while a stop request is still set (the overrun case) ─────────────────────


@red
def test_g5_the_lifespan_refuses_to_start_while_a_stop_request_is_set() -> None:
    from fastapi.testclient import TestClient

    import maxim.console.server as srv

    app = srv.build_app(None, auth_token="t" * 32)
    request_stop("an earlier stop that overran", Deadline.after(60.0))
    started = False
    try:
        with TestClient(app):
            started = True
    except Exception:
        pass
    assert not started, "a console started under a live stop request: its loops and LLM workers are born stopped"


# ── G7: config 1.4 carries console.stop_budget_s ───────────────────────────────────────────────────────────────


@red
def test_g7_config_1_4_carries_the_console_stop_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.runtime import config_loader

    assert config_loader.CONFIG_FORMAT_VERSION == "1.4"
    section = config_loader.ConsoleConfigSection()
    assert getattr(section, "stop_budget_s", None) == 15.0


# ── G9/G10: compare-and-clear, and the backend event follows the request ───────────────────────────────────────


@red
def test_g9_compare_and_clear_leaves_another_owners_request() -> None:
    assert "expected" in inspect.signature(cancellation.clear_stop_request).parameters
    mine = request_stop("mine", Deadline.after(5.0))
    cancellation.clear_stop_request()
    theirs = request_stop("theirs", Deadline.after(5.0))
    assert cancellation.clear_stop_request(expected=mine) is False
    assert current_stop() is theirs


@red
def test_g10_the_backend_event_follows_the_request_even_when_a_request_races_a_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maxim.models.language import cancellation as backend

    try:
        first = request_stop("first", Deadline.after(5.0))
        assert backend.is_shutdown_requested(), "a stop request does not reach the backend's shutdown event"
        assert "expected" in inspect.signature(cancellation.clear_stop_request).parameters

        real_reset = backend.reset_shutdown
        racer: list[threading.Thread] = []

        def _slow_reset() -> None:
            # A new request lands while the clear's on-clear hook is running.
            t = threading.Thread(target=lambda: request_stop("second", Deadline.after(5.0)))
            racer.append(t)
            t.start()
            time.sleep(0.2)
            real_reset()

        monkeypatch.setattr(backend, "reset_shutdown", _slow_reset)
        assert cancellation.clear_stop_request(expected=first) is True
        for t in racer:
            t.join(timeout=5.0)
        assert current_stop() is not None and current_stop().reason == "second"
        assert backend.is_shutdown_requested(), "a clear's reset landed after a newer request's shutdown"
    finally:
        backend.reset_shutdown()


# ── G11: a drain that leaves a campaign running never clears the request ───────────────────────────────────────


@red
def test_g11_a_drain_that_leaves_a_campaign_running_keeps_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.console.server as srv
    from maxim.console.handle import MaximHandle

    assert "deadline" in inspect.signature(srv._drain_and_stop_handle).parameters
    h = MaximHandle.__new__(MaximHandle)
    h.agent_id = "t"
    h._stopped = False
    h._campaign_lock = threading.Lock()
    h._talk_lock = threading.Lock()
    h._talk_bridge = h._talk_thread = h._talk_stop = h._talk_worker = None
    h.instance = MagicMock()
    release = threading.Event()
    held = threading.Event()

    def _campaign() -> None:
        with h._campaign_lock:
            held.set()
            release.wait(10.0)

    threading.Thread(target=_campaign, daemon=True).start()
    assert held.wait(5.0)
    monkeypatch.setattr(srv, "_handle", h)
    monkeypatch.setattr(srv, "_active_run", {"session_id": None, "thread": None})
    try:
        t0 = time.monotonic()
        srv._drain_and_stop_handle(deadline=Deadline.after(1.0))
        assert time.monotonic() - t0 < 2.0
        assert current_stop() is not None, "the request was cleared over a campaign that is still running"
    finally:
        release.set()
