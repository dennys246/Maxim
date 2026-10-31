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

    from maxim.utils import cancellation

    request_stop("test", Deadline.after(5.0))
    done = threading.Event()
    threading.Thread(target=lambda: (clear_stop_request(), done.set()), daemon=True).start()
    if not done.wait(2.0):
        # A deadlocked clear holds the hook lock forever, and the autouse clear would then hang the whole run. Swap in
        # a fresh lock (the module reads it at call time) so the teardown runs, and fail with the reason.
        cancellation._hook_lock = threading.Lock()
        pytest.fail("clear_stop_request() did not return within 2 s: the clear hook deadlocked on the hook lock")
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


def test_a_router_call_from_a_caller_that_abandons_on_the_stop_runs_inline() -> None:
    # The worker path (LLMWorker marks its calls: it abandons on its own registered stop event) is never
    # double-threaded.
    from maxim.utils.cancellation import mark_abandons_on_stop, reset_abandons_on_stop

    router, _started, release = _router_with_blocking_dispatch()
    release.set()
    seen: list[str] = []
    router._complete_text_locked = lambda *a, **k: (seen.append(threading.current_thread().name), ("x", None))[1]
    token = mark_abandons_on_stop()
    try:
        assert router._complete_text("s", "u", temperature=0.0, max_tokens=8) == ("x", None)
    finally:
        reset_abandons_on_stop(token)
    assert seen == [threading.current_thread().name]


def test_a_cancel_event_bound_only_for_a_timeout_does_not_lose_the_stops_abandon() -> None:
    # Binding a cancel event is a documented pattern for timeout-only abandons; only the explicit marker opts out.
    from maxim.utils.cancellation import reset_cancel_event, set_cancel_event

    router, started, release = _router_with_blocking_dispatch()
    result: list[Any] = []

    def _call() -> None:
        token = set_cancel_event(threading.Event())
        try:
            result.append(router._complete_text("s", "u", temperature=0.0, max_tokens=8))
        finally:
            reset_cancel_event(token)

    caller = threading.Thread(target=_call, daemon=True)
    caller.start()
    try:
        assert started.wait(5.0)
        request_stop("test", Deadline.after(5.0))
        caller.join(timeout=3.0)
    finally:
        release.set()
    assert result == [("", None)], "a timeout-only cancel event made the call run inline, past the stop"


def test_an_exception_in_the_callers_wait_cancels_the_orphan(monkeypatch: pytest.MonkeyPatch) -> None:
    # A Ctrl+C lands in the caller's poll: the orphan must be told to unwind, not left holding the inference lock.
    import maxim.utils.cancellation as cancellation

    router = _bare_router()
    router._routing_policy = SimpleNamespace(max_session_cost=0)
    router._inference_lock = threading.Lock()
    router._pending_auto_drains = []
    router._auto_drain_callback = None
    cancelled = threading.Event()

    def _dispatch(*_a: Any, **_k: Any) -> tuple[str, None]:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            ev = cancellation.current_cancel_event()
            if ev is not None and ev.is_set():
                cancelled.set()
                break
            time.sleep(0.01)
        return "", None

    router._complete_text_locked = _dispatch
    calls = [0]

    def _interrupting() -> bool:
        calls[0] += 1
        if calls[0] >= 2:  # the entry check passes; the first poll is "interrupted"
            raise KeyboardInterrupt
        return False

    monkeypatch.setattr(cancellation, "yield_requested", _interrupting)
    with pytest.raises(KeyboardInterrupt):
        router._complete_text("s", "u", temperature=0.0, max_tokens=8)
    assert cancelled.wait(3.0), "the orphan was never cancelled when its caller left"


def test_generate_raises_llm_call_stopped_and_generate_json_returns_none_quietly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from maxim.exceptions import LLMCallStopped, ModelInferenceError

    router, started, _release = _router_with_blocking_dispatch()
    router._get_backend = lambda: None
    request_stop("test", Deadline.after(5.0))
    with pytest.raises(LLMCallStopped) as raised:
        router.generate("p", temperature=0.0)
    assert isinstance(raised.value, ModelInferenceError), "callers catching ModelInferenceError must be unchanged"
    with caplog.at_level("WARNING"):
        assert router.generate_json("p") is None
    assert "non-dict" not in caplog.text, "a stop was logged as a model returning garbage"
    assert not started.is_set()


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
    bio._close_session(session, consolidation="full", bound=bound, abandon_if_expired=False, reserve_for_waiter=True)
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


def test_an_exit_with_no_stop_keeps_its_whole_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    # The reserve exists for a waiter on the SAME stop deadline; an exception exit with no stop has no such waiter.
    import maxim.runtime.bio_integration as bio

    seen: list[Any] = []
    monkeypatch.setattr(bio, "end_bio_session", lambda **k: seen.append(k["deadline"]))
    session = bio.BioSession(memory_hub=None, hippocampus=None, ticket=None, consolidation="full")
    bound = Deadline.after(30.0)
    bio._close_session(session, consolidation="lightweight", bound=bound)
    assert seen == [bound]


def test_an_unbounded_stop_deadline_stays_unbounded_at_the_session_end() -> None:
    from maxim.runtime.bio_integration import _end_bound

    end = _end_bound(Deadline.none())
    assert end is not None and end.remaining() == float("inf"), "inf - inf made the session end's bound 'now'"


def test_a_spent_bound_still_flushes_queued_captures() -> None:
    from maxim.runtime.bio_integration import _flush_timeout

    assert _flush_timeout(Deadline.after(0.0)) >= 0.5
    assert _flush_timeout(None) == 5.0
    assert _flush_timeout(Deadline.after(60.0)) == 5.0


def test_ending_early_never_lands_before_now_and_leaves_none_unbounded() -> None:
    assert Deadline.after(0.2).ending_early(5.0).remaining() == pytest.approx(0.0, abs=0.05)
    assert Deadline.none().ending_early(5.0).remaining() == float("inf")
    assert Deadline.after(10.0).ending_early(3.0).remaining() == pytest.approx(7.0, abs=0.1)


# ── a stopped sim's report: every report-time wait bounded, no skipped check stamped as passed ───────────────────


class _BusyRouter:
    """A role router whose in-flight call never ends (an abandoned orphan holds its lock), with serving probes."""

    call_serial = 9001

    def __init__(self) -> None:
        self.idle_waits: list[float] = []
        self.probed = False

    def wait_idle(self, timeout: float) -> bool:
        self.idle_waits.append(timeout)
        return False

    def serving_probes(self) -> list[dict[str, Any]]:
        self.probed = True
        return []


def test_a_stopped_reports_serving_provenance_caps_its_waits_and_stamps_skipped_probes_as_unknown() -> None:
    import maxim.simulation.serving_provenance as SP

    router = _BusyRouter()
    token = SP.capture_serving_start()
    t0 = time.monotonic()
    prov = SP.serving_provenance(
        token,
        llm_worker=SimpleNamespace(router=router),
        aut_worker=None,
        resume=None,
        adopted=False,
        goal="g",
        deadline=Deadline.after(0.5),
    )
    assert time.monotonic() - t0 < 2.0
    assert router.idle_waits and max(router.idle_waits) <= 0.5, "the idle wait was not capped by the deadline"
    assert not router.probed, "a probe started with less than its own timeout left"
    assert prov["model_changed_during_run"] is None, "a skipped probe was stamped as 'the model did not change'"
    assert prov["sampling_complete"] is False
    assert any("probes skipped" in p for p in prov["sampling_problems"])


def test_with_no_deadline_the_serving_waits_and_probes_are_todays() -> None:
    import maxim.simulation.serving_provenance as SP

    router = _BusyRouter()
    router.wait_idle = lambda timeout: (router.idle_waits.append(timeout), True)[1]  # type: ignore[method-assign]
    token = SP.capture_serving_start()
    prov = SP.serving_provenance(
        token, llm_worker=SimpleNamespace(router=router), aut_worker=None, resume=None, adopted=False, goal="g"
    )
    assert router.idle_waits == [SP.ROUTER_IDLE_TIMEOUT_S] and router.probed
    assert prov["model_changed_during_run"] is False


def test_a_stopped_reports_end_code_stamp_fails_closed_without_running_git(monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.simulation.report as report

    def _no_git(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the end code stamp ran git under a stop")

    start = report.capture_start_provenance()
    monkeypatch.setattr(report, "_code_stamp", _no_git)
    prov = report.run_provenance(
        start, llm_worker=None, aut_worker=None, resume=None, adopted=False, goal="g", deadline=Deadline.after(1.0)
    )
    assert prov["end_executed_git_hash"] == "unknown: stopped"
    assert prov["end_code_tree_sha256"] == "unknown: stopped"
    assert prov["working_tree_dirty_src_scripts"] is True
    assert prov["code_changed_during_run"] is None


# ── a stopped sim's container removal is detached, never killed mid-request ──────────────────────────────────────


def test_under_a_stop_a_container_removal_is_detached_in_its_own_session(monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.simulation.container_runner as cr

    launched: list[dict[str, Any]] = []

    class _Proc:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            launched.append({"argv": argv, **kwargs})

    def _no_wait(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a stopped sim waited on docker rm -f")

    monkeypatch.setattr(cr.subprocess, "Popen", _Proc)
    monkeypatch.setattr(cr, "_DETACHED_REMOVALS", [])
    runner = cr.LocalDockerRunner(docker_cmd="docker")
    monkeypatch.setattr(runner, "_run_docker", _no_wait)
    handle = cr.ContainerHandle(container_id="abc123", name="sim-test", spec=MagicMock())
    cr.LocalDockerRunner._LIVE_HANDLES.add(handle)
    request_stop("test", Deadline.after(5.0))
    runner.stop(handle)
    [call] = launched
    assert call["argv"] == ["docker", "rm", "-f", "abc123"]
    assert call["start_new_session"] is True, "a supervisor killing the process group would kill the removal"
    assert handle not in cr.LocalDockerRunner._LIVE_HANDLES, "the at-exit cleanup would wait on it again"
    assert len(cr._DETACHED_REMOVALS) == 1


# ── the runners' remaining stop checks, each pinned ──────────────────────────────────────────────────────────────


def _crossroads(bridge: Any, llm_router: Any = None) -> Any:
    from maxim.simulation.dm_runtime import DMRuntime
    from maxim.simulation.dm_schema import Act, CampaignDef, EncounterDef

    loop = EncounterDef(
        name="start", scene="A crossroads.", choices=["stay", "leave"], branches={"stay": "start", "leave": "__END__"}
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
    return DMRuntime(campaign=campaign, bridge=bridge, llm_router=llm_router)


def test_a_dm_choice_classified_by_a_call_the_stop_abandoned_is_never_recorded() -> None:
    bridge = MagicMock()
    bridge.send_and_wait = MagicMock(return_value={"response": "Hmm."})  # matches no choice: the LLM classifies it
    llm = MagicMock()

    def _abandoned(*_a: Any, **_k: Any) -> None:
        request_stop("test", Deadline.after(5.0))
        return None  # the router abandoned the call; the DM falls back to the first choice

    llm.generate_json = MagicMock(side_effect=_abandoned)
    runtime = _crossroads(bridge, llm_router=llm)
    state = runtime.run()
    assert llm.generate_json.called, "the test never reached the LLM classification"
    assert state.finish_reason == "yielded"
    assert state.choices_made == [], f"a fallback choice was recorded after the stop: {state.choices_made}"


def test_a_dm_campaign_under_a_stop_delivers_no_encounter() -> None:
    bridge = MagicMock()
    bridge.send_and_wait = MagicMock(return_value={"response": "I stay."})
    request_stop("test", Deadline.after(5.0))
    state = _crossroads(bridge).run()
    assert state.finish_reason == "yielded"
    assert not bridge.send_and_wait.called, "an encounter was composed and delivered under a stop"


def test_a_generative_campaign_under_a_stop_makes_no_narrator_call() -> None:
    from maxim.simulation.generative_runner import run_generative_campaign

    llm = MagicMock()
    bridge = MagicMock()
    request_stop("test", Deadline.after(5.0))
    result = run_generative_campaign(goal="test", bridge=bridge, llm=llm, max_turns=3)
    assert result.finish_reason == "yielded"
    assert not llm.generate_json.called and not llm.generate_text.called, "the narrator was called under a stop"


def test_g2_the_refusal_comes_before_the_start_provenance_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    # The capture runs git and starts model hashing: a refused sim must not spend the stop's budget on it.
    import maxim.simulation.orchestrator as orch
    from maxim.exceptions import StopStillPending

    def _no_capture(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the start provenance was captured under a stop request")

    monkeypatch.setattr(orch, "capture_start_provenance", _no_capture)
    request_stop("test", Deadline.after(5.0))
    with pytest.raises(StopStillPending):
        orch.start_simulation_mode(goal="stop test", max_turns=1)


def test_g2b_a_refusal_after_the_load_releases_the_sims_serving_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.runtime.lane_backends as lane_backends
    import maxim.simulation.serving_provenance as SP
    from maxim.exceptions import StopStillPending
    from maxim.simulation.orchestrator import start_simulation_mode

    monkeypatch.setattr(lane_backends, "build_primary_router", lambda **_k: (_LoadingRouter(), None))
    before = set(SP._ACTIVE)
    with pytest.raises(StopStillPending):
        start_simulation_mode(goal="stop test", max_turns=1)
    assert set(SP._ACTIVE) == before, "a refused sim left its serving registration behind"


def test_an_unmarked_callers_own_cancel_reaches_the_backend_through_the_helper() -> None:
    import maxim.utils.cancellation as cancellation

    router = _bare_router()
    router._routing_policy = SimpleNamespace(max_session_cost=0)
    router._inference_lock = threading.Lock()
    router._pending_auto_drains = []
    router._auto_drain_callback = None
    started, cancelled = threading.Event(), threading.Event()

    def _dispatch(*_a: Any, **_k: Any) -> tuple[str, None]:
        started.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            ev = cancellation.current_cancel_event()
            if ev is not None and ev.is_set():
                cancelled.set()
                break
            time.sleep(0.01)
        return "", None

    router._complete_text_locked = _dispatch
    own = threading.Event()
    token = cancellation.set_cancel_event(own)
    try:
        threading.Timer(0.3, own.set).start()  # the caller's own timeout fires mid-call
        router._complete_text("s", "u", temperature=0.0, max_tokens=8)
    finally:
        cancellation.reset_cancel_event(token)
    assert started.is_set() and cancelled.is_set(), "the caller's own cancel never reached the backend's checkpoint"


# ── G8-G10: the sim's tail under a stop (sim_teardown.end_run) ─────────────────────────────────────────────────────


class _Thread:
    """A joinable stand-in that records each join's timeout into the shared event log."""

    def __init__(self, name: str, events: list[tuple[str, Any]]) -> None:
        self._name, self._events = name, events

    def join(self, timeout: float | None = None) -> None:
        self._events.append((f"{self._name}_join", timeout))

    def is_alive(self) -> bool:
        return False


@pytest.fixture
def _tail_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """A SimTail over stand-ins, with every report writer recorded into one ordered event log."""
    import logging

    import maxim.simulation.report as report_mod
    import maxim.simulation.sim_logger as sim_logger
    import maxim.simulation.sim_teardown as teardown
    import maxim.utils.paths as paths
    from maxim.models.language.cancellation import reset_shutdown

    events: list[tuple[str, Any]] = []
    report = SimpleNamespace(session_id="tail-test", turns=0, total_actions=0, blocked_actions=0, llm_summary="")
    monkeypatch.setattr(report_mod, "build_report", lambda **k: events.append(("build_report", k)) or report)
    monkeypatch.setattr(report_mod, "save_report", lambda *a, **k: events.append(("save_report", None)))
    monkeypatch.setattr(report_mod, "save_action_log", lambda *a, **k: events.append(("save_action_log", None)))
    monkeypatch.setattr(report_mod, "save_aut_state", lambda **k: events.append(("save_aut_state", None)))
    monkeypatch.setattr(report_mod, "analyze_simulation", lambda *a, **k: events.append(("analyze_simulation", None)))
    monkeypatch.setattr(teardown, "run_provenance", lambda *a, **k: {"stamp": "test"})
    monkeypatch.setattr(sim_logger, "display_status", lambda *a, **k: None)
    monkeypatch.setattr(sim_logger, "disable_sim_logging", lambda: None)
    monkeypatch.setattr(paths, "sim_reports", lambda: tmp_path)
    monkeypatch.delenv("MAXIM_REPORT_JSON", raising=False)

    orch_instance = MagicMock()
    orch_instance.shutdown.side_effect = lambda **k: events.append(("orch_shutdown", k))
    adopted = MagicMock()
    adopted.lease.restore.return_value = ([], [])
    bridge = MagicMock()
    bridge.finish_context = None
    bridge.get_all_actions.return_value = []
    fields = {name: None for name in teardown.SimTail.__dataclass_fields__}
    fields.update(
        is_interactive=False,
        stop_event=threading.Event(),
        bridge=bridge,
        orchestrator_source=MagicMock(),
        aut_thread=_Thread("aut", events),
        adopted=adopted,  # an adopted agent's AUT join waits 300 s today
        dm_error=[],
        orch_instance=orch_instance,
        start_time=time.time(),
        operator_stop=threading.Event(),
        goal="tail test",
        mode="test",
        llm_router=MagicMock(session_cost_exceeded=False),
        start_provenance={},
    )
    tail = teardown.SimTail(**fields)
    level = logging.getLogger("maxim").level
    yield SimpleNamespace(tail=tail, events=events, end_run=teardown.end_run)
    logging.getLogger("maxim").setLevel(level)
    reset_shutdown()


def _names(events: list[tuple[str, Any]]) -> list[str]:
    return [name for name, _ in events]


def test_g8_under_a_stop_the_report_is_saved_before_the_tail_waits_on_the_aut(_tail_run: Any) -> None:
    request_stop("test", Deadline.after(10.0))
    _tail_run.end_run(_tail_run.tail)
    names = _names(_tail_run.events)
    assert "save_report" in names and names.index("save_report") < names.index("aut_join"), (
        f"the first report save came after the AUT join, so a stop that expires in the join leaves no report: {names}"
    )


def test_g9_under_a_stop_the_tails_waits_are_bounded_by_the_stops_deadline(_tail_run: Any) -> None:
    request_stop("test", Deadline.after(4.0))
    _tail_run.end_run(_tail_run.tail)
    [join_timeout] = [arg for name, arg in _tail_run.events if name == "aut_join"]
    # The tail's share of the stop's 4 s (_SIM_TAIL_SHARE = 0.35), not the whole deadline: the drain needs the rest.
    assert join_timeout is not None and join_timeout <= 0.35 * 4.0 + 0.05, f"the AUT join waited up to {join_timeout}s"
    [kwargs] = [arg for name, arg in _tail_run.events if name == "orch_shutdown"]
    assert kwargs.get("deadline") is not None, "the orchestrator instance's shutdown ran unbounded under the stop"


def test_g10_under_a_stop_the_tail_skips_the_llm_roundup(_tail_run: Any) -> None:
    request_stop("test", Deadline.after(10.0))
    _tail_run.end_run(_tail_run.tail)
    assert "analyze_simulation" not in _names(_tail_run.events), "the roundup made an LLM call under the stop (D4)"


def test_g10_under_a_stop_an_interactive_sim_skips_the_post_sim_prompt() -> None:
    # The prompt blocks on stdin for a human, and starts a new sim on an answer: neither belongs inside a stop.
    import ast

    from tests.unit._orchestrator_source import orchestrator_tree

    [guard] = [
        n
        for n in ast.walk(orchestrator_tree())
        if isinstance(n, ast.If)
        and any(
            isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_show_report_and_post_sim_prompt"
            for s in n.body
            for c in ast.walk(s)
        )
    ]
    assert "stop" in ast.unparse(guard.test), f"the post-sim prompt's guard ignores the stop: {ast.unparse(guard.test)}"


def test_with_no_stop_the_tail_keeps_todays_order_and_roundup(_tail_run: Any) -> None:
    _tail_run.end_run(_tail_run.tail)
    names = _names(_tail_run.events)
    assert names.index("aut_join") < names.index("build_report") < names.index("save_report"), names
    assert "analyze_simulation" in names, names
    [join_timeout] = [arg for name, arg in _tail_run.events if name == "aut_join"]
    assert join_timeout == 300.0, "an adopted agent's full consolidation lost its 300 s join"
    [kwargs] = [arg for name, arg in _tail_run.events if name == "orch_shutdown"]
    assert kwargs == {}, "with no stop the orchestrator instance's shutdown is called as today"


def test_under_a_stop_provenance_is_computed_once_bounded_and_stamps_each_save(
    _tail_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maxim.simulation.sim_teardown as teardown

    calls: list[Any] = []
    monkeypatch.setattr(teardown, "run_provenance", lambda *a, **k: calls.append(k.get("deadline")) or {"s": 1})
    request_stop("test", Deadline.after(30.0))
    _tail_run.end_run(_tail_run.tail)
    assert len(calls) == 1 and calls[0] is not None, f"provenance ran {len(calls)}x, deadlines {calls}"
    stages = [k["provenance"].get("stop_stage") for name, k in _tail_run.events if name == "build_report"]
    assert stages == ["early", "late"], stages
    assert _tail_run.events[-1][0] != "analyze_simulation"


def test_under_a_spent_stop_the_late_save_is_skipped_and_the_early_report_stands(
    _tail_run: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import maxim.simulation.report as report_mod

    emitted: list[Any] = []
    monkeypatch.setattr(report_mod, "emit_report_json", lambda report, path: emitted.append(path))
    monkeypatch.setenv("MAXIM_REPORT_JSON", str(tmp_path / "r.json"))
    request_stop("test", Deadline.after(0.0))
    _tail_run.end_run(_tail_run.tail)
    names = _names(_tail_run.events)
    assert names.count("build_report") == 1 and "save_aut_state" not in names, names
    assert emitted == [str(tmp_path / "r.json")], "the early report was not emitted when the late save was skipped"


def test_a_stopped_result_skips_the_snapshot_while_the_aut_still_runs(_tail_run: Any) -> None:
    import maxim.simulation.sim_teardown as teardown

    tail = _tail_run.tail
    tail.stop = request_stop("test", Deadline.after(5.0))
    tail.aut_thread = MagicMock(is_alive=lambda: True)
    tail.aut_introspector = MagicMock()
    report = SimpleNamespace(session_id="s", turns=0, total_actions=0, blocked_actions=0, llm_summary="")
    result = teardown.build_simulation_result(tail, report, "/tmp", 0.0, "yielded")
    tail.aut_introspector.benchmark_snapshot.assert_not_called()
    assert result.subsystem_snapshot == {}


def test_under_a_stop_a_failed_early_save_still_waits_releases_and_saves_late(
    _tail_run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maxim.simulation.report as report_mod
    import maxim.simulation.sim_logger as sim_logger

    events = _tail_run.events
    saves = iter([OSError("disk full"), None])

    def _save(*_a: Any, **_k: Any) -> None:
        events.append(("save_report", None))
        failure = next(saves)
        if failure is not None:
            raise failure

    monkeypatch.setattr(report_mod, "save_report", _save)
    monkeypatch.setattr(sim_logger, "disable_sim_logging", lambda: events.append(("disable_sim_logging", None)))
    request_stop("test", Deadline.after(0.0))  # no time left: the late save runs only because the early one failed
    _tail_run.end_run(_tail_run.tail)
    names = _names(events)
    assert "orch_shutdown" in names and "disable_sim_logging" in names, names
    assert names.count("build_report") == 2 and "save_aut_state" in names, names


def test_under_a_stop_a_failed_late_save_keeps_and_emits_the_early_report(
    _tail_run: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import maxim.simulation.report as report_mod

    monkeypatch.setattr(report_mod, "save_aut_state", lambda **k: (_ for _ in ()).throw(OSError("disk full")))
    emitted: list[Any] = []
    monkeypatch.setattr(report_mod, "emit_report_json", lambda report, path: emitted.append(path))
    monkeypatch.setenv("MAXIM_REPORT_JSON", str(tmp_path / "r.json"))
    request_stop("test", Deadline.after(30.0))
    report, *_ = _tail_run.end_run(_tail_run.tail)
    assert report.session_id == "tail-test" and emitted == [str(tmp_path / "r.json")]


def test_under_a_stop_a_live_aut_skips_the_releases_that_act_on_its_state(_tail_run: Any) -> None:
    tail = _tail_run.tail
    tail.aut_thread = MagicMock(is_alive=lambda: True)
    tail.aut_imagination_trigger = MagicMock()
    request_stop("test", Deadline.after(5.0))
    _tail_run.end_run(tail)
    tail.adopted.lease.restore.assert_not_called()
    tail.aut_imagination_trigger.stats.assert_not_called()


def test_with_no_stop_the_releases_run_as_today(_tail_run: Any) -> None:
    tail = _tail_run.tail
    tail.aut_imagination_trigger = MagicMock()
    tail.aut_imagination_trigger.stats.return_value = {}
    _tail_run.end_run(tail)
    tail.adopted.lease.restore.assert_called_once()
    tail.aut_imagination_trigger.stats.assert_called_once()


def test_a_stopped_result_skips_the_basic_analysis_while_the_aut_still_runs(_tail_run: Any) -> None:
    import maxim.simulation.sim_teardown as teardown

    tail = _tail_run.tail
    tail.stop = request_stop("test", Deadline.after(5.0))
    tail.aut_thread = MagicMock(is_alive=lambda: True)
    tail.aut_introspector = MagicMock()
    report = SimpleNamespace(session_id="s", turns=0, total_actions=0, blocked_actions=0, llm_summary="")
    result = teardown.build_simulation_result(tail, report, "/tmp", 0.0, "yielded")
    tail.aut_introspector.full_analysis.assert_not_called()
    assert result.campaign_analysis == {}


def test_a_stopped_interactive_run_prints_its_report_to_the_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.simulation.report as report_mod
    import maxim.simulation.sim_logger as sim_logger
    import maxim.simulation.sim_teardown as teardown

    display = MagicMock()
    sim_logger.set_active_display(display)
    seen: list[Any] = []
    monkeypatch.setattr(report_mod, "print_report", lambda *a, **k: seen.append(sim_logger.get_active_display()))
    try:
        tail = SimpleNamespace(is_interactive=True)
        teardown.print_run_report(tail, SimpleNamespace(session_id="s"), "/tmp")  # type: ignore[arg-type]
    finally:
        sim_logger.set_active_display(None)
    display.stop.assert_called_once()
    assert seen == [None], "the report was routed into the stopped live panel, not the terminal"


@pytest.mark.parametrize(
    "stopped, loop_holds_session, restored", [(True, True, False), (True, False, True), (False, True, True)]
)
def test_the_handle_lease_safety_net_skips_only_under_a_stop_with_a_loop_still_holding_the_session(
    stopped: bool, loop_holds_session: bool, restored: bool
) -> None:
    from maxim.console.handle import _restore_campaign_lease

    instance = MagicMock()
    instance.memory_hub.wait_session_ended.return_value = not loop_holds_session
    lease = MagicMock()
    lease.restore.return_value = ([], [])
    if stopped:
        request_stop("test", Deadline.after(5.0))
    _restore_campaign_lease(instance, lease)
    assert lease.restore.called is restored
