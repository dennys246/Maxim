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


@_RED
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


@_RED
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


@_RED
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


# ── G3: the generative runner stops, records no fabricated turn, and says yielded ────────────────────────────────


@_RED
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


@_RED
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


@_RED
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
