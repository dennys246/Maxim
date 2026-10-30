"""Red gates for bounded stop PR 1c, the sim side (#1438; docs/plans/bounded_stop.md §5).

PR 1b's process stop request ends a campaign's AUT loop, but nothing else in a simulation notices it: the runners
loop on, a direct router call runs to the backend's timeout, a ``reset_shutdown()`` re-arms the backend under a live
request, and ``wait_ready`` waits out a cold load. Each gate below is proven red on the pre-fix code by assertion, for
the reason named (``xfail(strict=True, raises=AssertionError)`` in the red-gate commit), and flipped by PR 1c.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.utils.cancellation import clear_stop_request, request_stop
from maxim.utils.deadline import Deadline

_RED = pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1438 PR 1c: flips with the sim side")


# ── G1: a reset never re-arms the backend under a live stop request ──────────────────────────────────────────────


def test_g1_reset_shutdown_under_a_stop_request_leaves_the_backend_stopped() -> None:
    from maxim.models.language import cancellation as backend

    request_stop("test", Deadline.after(5.0))
    assert backend.is_shutdown_requested()
    backend.reset_shutdown()  # the sim's entry reset and its roundup both call this
    assert backend.is_shutdown_requested(), "reset_shutdown() re-armed the backend while a stop was requested"


def test_a_clear_with_the_backend_hooks_registered_returns_promptly() -> None:
    # Guard for G1's fix: the clear hook runs under the hook lock, so it must never re-take it.
    from maxim.models.language import cancellation as backend

    request_stop("test", Deadline.after(5.0))
    done = threading.Event()
    threading.Thread(target=lambda: (clear_stop_request(), done.set()), daemon=True).start()
    assert done.wait(2.0), "clear_stop_request() did not return: the clear hook deadlocked"
    assert not backend.is_shutdown_requested(), "a finished stop's clear left the backend stopped"


# ── G5: wait_ready honours a stop ────────────────────────────────────────────────────────────────────────────────


def _bare_router() -> Any:
    from maxim.models.language.router import LLMRouter

    router = LLMRouter.__new__(LLMRouter)
    router.cfg = SimpleNamespace(enabled=True)
    router._ready_event = threading.Event()
    router._warmup_failed = False
    return router


def test_g5_wait_ready_returns_promptly_when_a_stop_lands_during_the_load() -> None:
    router = _bare_router()
    result: list[bool] = []
    waiter = threading.Thread(target=lambda: result.append(router.wait_ready(timeout=5.0)), daemon=True)
    waiter.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    request_stop("test", Deadline.after(5.0))
    waiter.join(timeout=6.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.2, f"wait_ready waited {elapsed:.1f}s after the stop (its timeout, not the stop)"
    assert result == [False]


def test_wait_ready_with_only_the_backend_event_set_still_waits_for_readiness() -> None:
    # Guard: a NORMAL sim end sets the backend event too; wait_ready keys on the stop request, never on it.
    from maxim.models.language import cancellation as backend

    router = _bare_router()
    backend.request_shutdown()
    try:
        threading.Timer(0.3, router._ready_event.set).start()
        assert router.wait_ready(timeout=5.0) is True
    finally:
        backend.reset_shutdown()


# ── G6: the router abandons a direct call at a stop ──────────────────────────────────────────────────────────────


def _router_with_blocking_dispatch() -> tuple[Any, threading.Event, threading.Event]:
    router = _bare_router()
    router._routing_policy = SimpleNamespace(max_session_cost=0)
    router._inference_lock = threading.Lock()
    router._pending_auto_drains = []
    router._auto_drain_callback = None
    started, release = threading.Event(), threading.Event()

    def _blocking(*_a: Any, **_k: Any) -> tuple[str, None]:
        started.set()
        release.wait(10.0)
        return "late text", None

    router._complete_text_locked = _blocking
    return router, started, release


def test_g6_a_direct_router_call_returns_promptly_after_a_stop() -> None:
    router, started, release = _router_with_blocking_dispatch()
    result: list[Any] = []
    caller = threading.Thread(
        target=lambda: result.append(router._complete_text("s", "u", temperature=0.0, max_tokens=8)), daemon=True
    )
    caller.start()
    try:
        assert started.wait(5.0)
        t0 = time.monotonic()
        request_stop("test", Deadline.after(5.0))
        caller.join(timeout=8.0)
        elapsed = time.monotonic() - t0
    finally:
        release.set()
    assert elapsed < 1.5, f"the direct call ran {elapsed:.1f}s after the stop (its own end, not the stop)"
    assert result == [("", None)], "an abandoned call returns the dispatch-failure shape"


def test_a_direct_router_call_with_no_stop_returns_its_result() -> None:
    router, started, release = _router_with_blocking_dispatch()
    release.set()
    assert router._complete_text("s", "u", temperature=0.0, max_tokens=8) == ("late text", None)


def test_a_direct_router_call_re_raises_its_dispatch_error_as_itself_and_keeps_the_callers_thread_tag() -> None:
    from maxim.exceptions import LeaderRefusal

    router, _started, _release = _router_with_blocking_dispatch()
    seen: list[str] = []

    def _refuse(*_a: Any, **_k: Any) -> Any:
        seen.append(threading.current_thread().name)
        raise LeaderRefusal("refused")

    router._complete_text_locked = _refuse
    with pytest.raises(LeaderRefusal):
        router._complete_text("s", "u", temperature=0.0, max_tokens=8)
    # sim_log tags [ORCH] / [AUT] by thread name: the helper carries the caller's name.
    assert seen and seen[0].startswith(threading.current_thread().name), seen


def test_a_router_call_under_a_bound_cancel_event_runs_inline() -> None:
    # The worker path (LLMWorker binds a per-call cancel event and abandons on its own) is never double-threaded.
    from maxim.utils.cancellation import reset_cancel_event, set_cancel_event

    router, _started, release = _router_with_blocking_dispatch()
    release.set()
    seen: list[str] = []
    router._complete_text_locked = lambda *a, **k: (seen.append(threading.current_thread().name), ("x", None))[1]
    token = set_cancel_event(threading.Event())
    try:
        assert router._complete_text("s", "u", temperature=0.0, max_tokens=8) == ("x", None)
    finally:
        reset_cancel_event(token)
    assert seen == [threading.current_thread().name]


def test_a_router_call_made_under_a_stop_is_refused_at_once() -> None:
    router, started, _release = _router_with_blocking_dispatch()
    request_stop("test", Deadline.after(5.0))
    assert router._complete_text("s", "u", temperature=0.0, max_tokens=8) == ("", None)
    assert not started.is_set(), "a call made under a stop reached the backend"


# ── G3: the generative runner stops, records no fabricated turn, and says yielded ────────────────────────────────


def test_g3_the_generative_runner_stops_after_the_call_a_stop_lands_in_and_reports_yielded() -> None:
    from maxim.simulation.generative_runner import run_generative_campaign

    sends = [0]
    decision = {"phase": "seed", "scene_type": "x", "notes": "", "narrative": "Scene.", "done": False}

    def _llm_call(*_a: Any, **_k: Any) -> Any:
        if sends[0] >= 1:  # the second turn's narrator call: the stop lands in it, and the router abandons it
            request_stop("test", Deadline.after(5.0))
            raise RuntimeError("abandoned at the stop")
        return dict(decision)

    llm = MagicMock()
    llm.generate_json = MagicMock(side_effect=_llm_call)
    llm.generate_text = MagicMock(side_effect=lambda *a, **k: _llm_call() and "Scene.")
    bridge = MagicMock()

    def _send(*_a: Any, **_k: Any) -> dict[str, Any]:
        sends[0] += 1
        return {"response": "ok", "turn": sends[0], "timed_out": False}

    bridge.send_and_wait = MagicMock(side_effect=_send)
    result = run_generative_campaign(goal="test", bridge=bridge, llm=llm, max_turns=4)
    assert result.finish_reason == "yielded", f"a stopped campaign reported {result.finish_reason!r}"
    assert sends[0] == 1, f"{sends[0] - 1} turn(s) were delivered after the stop, from fallback narrative"


# ── G4: the DM runtime stops, records no fabricated choice, and says yielded ─────────────────────────────────────


def test_g4_the_dm_runtime_stops_at_the_encounter_a_stop_lands_in_and_reports_yielded() -> None:
    from maxim.simulation.dm_runtime import DMRuntime
    from maxim.simulation.dm_schema import Act, CampaignDef, EncounterDef

    loop = EncounterDef(
        name="start",
        scene="A crossroads.",
        choices=["stay", "leave"],
        branches={"stay": "start", "leave": "__END__"},
    )
    campaign = CampaignDef(
        name="stop_test",
        goal="stop",
        seed=0,
        acts=[Act(name="act1", encounters=["start"])],
        encounters={"start": loop},
        pc_spec={},
        npc_specs={},
    )
    bridge = MagicMock()
    runtime = DMRuntime(campaign=campaign, bridge=bridge)
    calls = [0]

    def _send(*_a: Any, **_k: Any) -> dict[str, Any]:
        calls[0] += 1
        if calls[0] == 2:  # the stop lands while the AUT answers the second encounter
            request_stop("test", Deadline.after(5.0))
        if calls[0] >= 5:  # pre-fix the crossroads loops forever on its fallback choice: end the run
            runtime._state.finished = True
        return {"response": "I stay."}

    bridge.send_and_wait = MagicMock(side_effect=_send)
    state = runtime.run()
    assert state.finish_reason == "yielded", f"a stopped campaign reported {state.finish_reason!r}"
    assert len(state.choices_made) == 1, f"{len(state.choices_made) - 1} choice(s) were recorded after the stop"


# ── G4b: the orchestrator reads a runner's yielded as yielded, never error ───────────────────────────────────────


@pytest.mark.parametrize(
    ("runner", "outcome"),
    [
        ("generative_runner", SimpleNamespace(finish_reason="yielded")),
        ("dm_runner", {"campaign": {"finish_reason": "yielded"}}),
    ],
)
def test_g4b_a_runner_that_yielded_is_reported_as_yielded(runner: str, outcome: Any) -> None:
    from maxim.simulation.orchestrator import _runner_outcome

    got = _runner_outcome(runner, outcome)
    assert got is not None and got[0] == "yielded", f"{runner} yielded read as {got!r}"


# ── G2: a simulation refuses to start under a stop request, before anything is built ────────────────────────────


def test_g2_a_simulation_started_under_a_stop_request_refuses_before_building_a_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maxim.runtime.lane_backends as lane_backends
    from maxim.exceptions import StopStillPending
    from maxim.simulation.orchestrator import start_simulation_mode

    def _no_router(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a router was built under a stop request")

    monkeypatch.setattr(lane_backends, "build_primary_router", _no_router)
    request_stop("test", Deadline.after(5.0))
    with pytest.raises(StopStillPending) as raised:
        start_simulation_mode(goal="stop test", max_turns=1)
    assert raised.value.reason == "test"
    assert "simulation not started" in str(raised.value)


class _LoadingRouter:
    """A router whose model load is interrupted by a stop request."""

    def warmup(self) -> bool:
        return True

    def wait_ready(self, timeout: float = 120.0) -> bool:
        request_stop("stopped during the load", Deadline.after(5.0))
        return False


def test_g2b_a_stop_during_the_model_load_refuses_before_the_bridge_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.runtime.lane_backends as lane_backends
    import maxim.simulation.bridge as bridge_mod
    from maxim.exceptions import StopStillPending
    from maxim.simulation.orchestrator import start_simulation_mode

    def _no_bridge(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the bridge was built after the stop landed during the load")

    monkeypatch.setattr(lane_backends, "build_primary_router", lambda **_k: (_LoadingRouter(), None))
    monkeypatch.setattr(bridge_mod, "SimulationBridge", _no_bridge)
    with pytest.raises(StopStillPending) as raised:
        start_simulation_mode(goal="stop test", max_turns=1)
    assert raised.value.reason == "stopped during the load"


# ── G7: a bounded session end keeps a reserve for its waiter (closes 1b's shared-deadline overrun) ──────────────


def test_g7_a_clean_exit_under_a_stop_ends_its_session_inside_the_stops_deadline_minus_a_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maxim.runtime.bio_integration as bio

    seen: list[Any] = []
    monkeypatch.setattr(bio, "end_bio_session", lambda **k: seen.append(k["deadline"]))
    session = bio.BioSession(memory_hub=None, hippocampus=None, ticket=None, consolidation="full")
    bound = Deadline.after(8.0)
    bio._close_session(session, consolidation="full", bound=bound, abandon_if_expired=False)
    [end] = seen
    # The waiter (AgentInstance.shutdown(loop_wait=), the drain) waits on the SAME deadline: the end must leave it
    # time to see the end, persist and after-steps included (25% of the time left, at least 1 s).
    assert end.at <= bound.at - 1.9, f"the session end ran until {bound.at - end.at:.2f}s before its waiter's deadline"


def test_the_abandon_test_keeps_the_full_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1b's rule: an abnormal exit abandons only when the stop's own bound expired, not the reserved one.
    import maxim.runtime.bio_integration as bio

    seen: list[Any] = []
    monkeypatch.setattr(bio, "end_bio_session", lambda **k: seen.append(k["deadline"]))
    session = bio.BioSession(memory_hub=None, hippocampus=None, ticket=None, consolidation="full")
    bio._close_session(session, consolidation="lightweight", bound=Deadline.after(0.5))  # inside the 1 s floor
    assert len(seen) == 1, "an abnormal exit with time left was abandoned because of the reserve"


def test_a_spent_bound_still_flushes_queued_captures() -> None:
    from maxim.runtime.bio_integration import _flush_timeout

    assert _flush_timeout(Deadline.after(0.0)) >= 0.5
    assert _flush_timeout(None) == 5.0
    assert _flush_timeout(Deadline.after(60.0)) == 5.0


def test_ending_early_never_lands_before_now_and_leaves_none_unbounded() -> None:
    assert Deadline.after(0.2).ending_early(5.0).remaining() == pytest.approx(0.0, abs=0.05)
    assert Deadline.none().ending_early(5.0).remaining() == float("inf")
    assert Deadline.after(10.0).ending_early(3.0).remaining() == pytest.approx(7.0, abs=0.1)
