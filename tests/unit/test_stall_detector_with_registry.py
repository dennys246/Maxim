"""Integration-level test for stall detector + LLM call registry suppression.

Pins the load-bearing behavior of the bug fix:
- In-flight + recent bytes → suppress nudge
- In-flight + byte silence > max threshold → fire (wedged-call branch)
- No in-flight + idle past threshold → fire (real stall)
- No in-flight + ping_pong → fire (real stall)
- Tier filtering: large-tier in-flight + medium-tier query → ignored

Doesn't import the orchestrator (avoids pulling the full sim runtime);
drives the orchestrator's REAL decision, ``stall_threshold.stall_suppression``
(#1042: the earlier copy of the decision tree let a router/detector tier
mismatch ship).
"""

from __future__ import annotations

import time

import pytest

from unittest.mock import patch

from maxim.agents.llm_types import PLANNING_LANE
from maxim.runtime import llm_call_registry as registry
from maxim.runtime import stall_threshold as st
from maxim.runtime.stall_threshold import compute_stall_threshold


@pytest.fixture(autouse=True)
def _reset_registry():
    registry._reset_for_tests()
    yield
    registry._reset_for_tests()


def _orchestrator_should_suppress(*, tier: str, max_byte_silence_s: float, turn_in_progress: bool = False) -> bool:
    """The orchestrator's real suppression decision at a given byte-silence threshold."""
    with patch.object(st, "max_byte_silence_threshold_s", lambda: max_byte_silence_s):
        return st.stall_suppression(turn_in_progress=turn_in_progress, lane=tier)[0]


# ─── Suppression scenarios ───────────────────────────────────────────────


def test_in_flight_with_recent_bytes_suppresses_nudge():
    """The load-bearing case: orchestrator's LLM call is in flight, bytes
    flowing (real tokens or keepalives). Stall MUST be suppressed."""
    cid = registry.register_call_start(tier="large")
    registry.register_byte_received(call_id=cid)
    assert _orchestrator_should_suppress(tier="large", max_byte_silence_s=90.0)


def test_in_flight_with_byte_silence_past_threshold_fires_nudge():
    """Connection wedged: in flight per registry but no bytes for >N seconds.
    Stall should fire as a stuck-call warning."""
    registry.register_call_start(tier="large", streams=True)
    # Don't update last_byte_at; let silence accumulate
    # Use a tiny max_byte_silence_s to keep test fast
    time.sleep(0.05)
    assert not _orchestrator_should_suppress(tier="large", max_byte_silence_s=0.01)


def test_no_in_flight_calls_does_not_suppress():
    """No LLM call in flight: existing stall logic (ping_pong / time_stalled)
    should be free to fire its normal nudge."""
    assert not _orchestrator_should_suppress(tier="large", max_byte_silence_s=90.0)


def test_tier_filtered_in_flight_does_not_suppress_other_tier():
    """Large-tier call in flight; orchestrator queries medium-tier — must NOT
    be suppressed (it's a different conversation)."""
    cid = registry.register_call_start(tier="large")
    registry.register_byte_received(call_id=cid)
    # Orchestrator routes via medium (rare but possible failover)
    assert not _orchestrator_should_suppress(tier="medium", max_byte_silence_s=90.0)


def test_suppression_releases_after_call_ends():
    cid = registry.register_call_start(tier="large")
    registry.register_byte_received(call_id=cid)
    assert _orchestrator_should_suppress(tier="large", max_byte_silence_s=90.0)
    registry.register_call_end(cid)
    assert not _orchestrator_should_suppress(tier="large", max_byte_silence_s=90.0)


# ─── End-to-end: derived threshold + suppression compose ─────────────────


def test_derived_threshold_for_long_running_inference():
    """Operator sets lanes.large.timeout_s = 600. Threshold derives 610.
    In-flight call at 300s with bytes flowing: suppression hold.
    In-flight call past threshold + byte silence: wedged-call fires."""
    threshold = compute_stall_threshold(lane_tier="large", lane_timeout_s=600)
    assert threshold == 610.0

    registry.register_call_start(tier="large")
    registry.register_byte_received()
    # Suppression holds with recent bytes — regardless of threshold
    assert _orchestrator_should_suppress(tier="large", max_byte_silence_s=90.0)


# ─── #1042: what counts as busy, and the composition through the router ───


def test_a_silent_non_streaming_call_is_never_wedged():
    """A non-streaming call reports no bytes: its silence is its age, never evidence of a wedged connection."""
    registry.register_call_start(tier="large")  # streams=False
    time.sleep(0.05)
    with patch.object(st, "max_byte_silence_threshold_s", lambda: 0.01):
        assert st.stall_suppression(turn_in_progress=False, lane="large") == (True, None)


def test_a_turn_in_progress_suppresses_with_nothing_in_flight():
    """Waiting on the agent's turn (between its LLM calls: tool runs, the settle window) is busy, not idle."""
    assert st.stall_suppression(turn_in_progress=True) == (True, None)
    assert st.stall_suppression(turn_in_progress=False) == (False, None)


def test_a_routed_planning_call_suppresses_the_detector():
    """The composition: a real router dispatch on the planning lane (what LLMWorker sends) is what the detector's
    default lane asks about."""
    import dataclasses

    from maxim.models.language.config import LLMConfig
    from maxim.models.language.router import LLMRouter

    cfg = dataclasses.replace(
        LLMConfig(),
        enabled=True,
        providers={"a": {"type": "maxim_peer", "base_url": "http://127.0.0.1:1/v1", "model": "m"}},
    )
    router = LLMRouter(cfg)
    seen: dict = {}

    def fake_try_provider(**_kwargs):
        time.sleep(0.02)  # silent: a non-streaming dispatch must still never read as wedged
        with patch.object(st, "max_byte_silence_threshold_s", lambda: 0.0):
            seen["decision"] = st.stall_suppression(turn_in_progress=False)
        return "", None, "failed"

    with patch.object(router, "_try_provider", side_effect=fake_try_provider):
        with patch.object(router, "_candidate_providers", return_value=(["a"], "normal", {})):
            with router._inference_lock:
                router._complete_text_locked(
                    "", "hi", temperature=0.0, max_tokens=1,
                    request_context={"agent_id": "llm_worker", "lane": PLANNING_LANE},
                )  # fmt: skip
    assert seen["decision"] == (True, None), seen
    assert st.stall_suppression(turn_in_progress=False) == (False, None)  # released when the call ends


def test_the_detector_and_the_worker_share_one_planning_lane():
    """Structural pin: the orchestrator's detector and LLMWorker's planning dispatch both read PLANNING_LANE."""
    import inspect

    from maxim.agents import llm_worker
    from maxim.simulation import orchestrator

    assert "_resolved_tier = PLANNING_LANE" in inspect.getsource(orchestrator)
    worker = inspect.getsource(llm_worker)
    assert "lane = PLANNING_LANE" in worker and 'lane = "large"' not in worker


def test_the_bridge_marks_the_whole_turn_in_progress():
    """send_and_wait sets the flag for its whole duration and clears it after, exceptions included."""
    import threading

    from maxim.simulation.bridge import SimulationBridge

    bridge = object.__new__(SimulationBridge)
    bridge._turn_active = threading.Event()
    during: list[bool] = []

    def body(*_a, **_k):
        during.append(bridge.turn_in_progress)
        return {"turn": 1}

    with patch.object(bridge, "_send_and_wait", side_effect=body):
        assert bridge.send_and_wait("hi") == {"turn": 1}
    assert during == [True] and bridge.turn_in_progress is False

    with patch.object(bridge, "_send_and_wait", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            bridge.send_and_wait("hi")
    assert bridge.turn_in_progress is False


def test_a_non_streaming_call_past_its_allowed_time_is_wedged_by_age():
    """No bytes, so age is the evidence: once even the youngest non-streaming call outlives the time it is allowed,
    the detector stops suppressing and reports its age (owner decision 2026-10-01)."""
    registry.register_call_start(tier="large")  # streams=False
    time.sleep(0.05)
    assert st.stall_suppression(turn_in_progress=False, lane="large") == (True, None)  # young: busy
    with patch.object(st, "non_streaming_age_bound_s", lambda *_a, **_k: 0.01):
        suppress, wedged = st.stall_suppression(turn_in_progress=False, lane="large")
    assert suppress is False and wedged is not None and wedged >= 0.05


def test_the_age_bound_is_the_allowed_call_time_plus_margin(monkeypatch):
    monkeypatch.delenv("MAXIM_LLM_CALL_TIMEOUT_S", raising=False)
    monkeypatch.delenv("MAXIM_STALL_MARGIN_S", raising=False)
    assert st.non_streaming_age_bound_s() == 300.0 + st.DEFAULT_STALL_MARGIN_S  # the worker's default call timeout
    assert st.non_streaming_age_bound_s(600.0) == 600.0 + st.DEFAULT_STALL_MARGIN_S  # a longer lane timeout wins


def test_the_orchestrator_wires_the_suppression_decision():
    """Structural pins on the detector's wiring (no unit can drive the sim loop): the turn flag, the lane timeout,
    the activity-clock reset while a turn is in progress, ping-pong exempt, and a reported (not swallowed) failure."""
    import inspect

    from maxim.simulation import orchestrator

    src = inspect.getsource(orchestrator)
    assert "turn_in_progress=bridge.turn_in_progress, lane=_resolved_tier, lane_timeout_s=_lane_timeout_s" in src
    assert "if _suppress and not ping_pong:" in src
    block = src.split("if _suppress and not ping_pong:", 1)[1].split("continue", 1)[0]
    assert "if bridge.turn_in_progress:" in block and "_last_activity_time[0] = time.time()" in block
    assert 'log_swallowed_exception(site="orchestrator.py:_stall_detector:stall_suppression")' in src


def test_a_byte_silent_streaming_call_is_wedged_whatever_its_age():
    """Age is only a non-streaming call's evidence: a young but byte-silent STREAMING call is wedged."""
    registry.register_call_start(tier="large", streams=True)
    time.sleep(0.05)
    with patch.object(st, "max_byte_silence_threshold_s", lambda: 0.01):
        suppress, wedged = st.stall_suppression(turn_in_progress=False, lane="large")
    assert suppress is False and wedged is not None and wedged >= 0.05


def test_the_age_bound_follows_the_configured_call_timeout(monkeypatch):
    monkeypatch.setenv("MAXIM_LLM_CALL_TIMEOUT_S", "600")
    monkeypatch.delenv("MAXIM_STALL_MARGIN_S", raising=False)
    assert st.non_streaming_age_bound_s() == 600.0 + st.DEFAULT_STALL_MARGIN_S


def test_each_call_is_judged_against_the_time_it_was_allowed(monkeypatch):
    """A timeout-retry is allowed twice the time: its own allowed_s, not the default, decides when it is wedged."""
    monkeypatch.setenv("MAXIM_STALL_MARGIN_S", "0")
    assert st.non_streaming_age_bound_s(allowed_s=600.0) == 600.0  # its own allowance, not the 300 s default
    registry.register_call_start(tier="large", allowed_s=0.001)  # long past its (tiny) allowance
    time.sleep(0.02)
    suppress, wedged = st.stall_suppression(turn_in_progress=False, lane="large")
    assert suppress is False and wedged is not None
    registry._reset_for_tests()
    registry.register_call_start(tier="large", allowed_s=1000.0)
    time.sleep(0.02)
    assert st.stall_suppression(turn_in_progress=False, lane="large") == (True, None)


def test_one_live_call_keeps_the_narrator_busy_beside_a_zombie(monkeypatch):
    """The YOUNGEST call decides: a zombie entry past its allowance beside a fresh, healthy call is not a stall."""
    monkeypatch.setenv("MAXIM_STALL_MARGIN_S", "0")
    registry.register_call_start(tier="large", allowed_s=0.001)  # zombie
    time.sleep(0.02)
    registry.register_call_start(tier="large", allowed_s=1000.0)  # fresh
    assert st.stall_suppression(turn_in_progress=False, lane="large") == (True, None)


def test_the_router_registers_the_time_the_caller_allowed():
    """LLMWorker stamps its effective timeout on request_context["allowed_s"]; the router puts it on the entry."""
    import dataclasses
    import inspect

    from maxim.agents import llm_worker
    from maxim.models.language.config import LLMConfig
    from maxim.models.language.router import LLMRouter

    cfg = dataclasses.replace(
        LLMConfig(),
        enabled=True,
        providers={"a": {"type": "maxim_peer", "base_url": "http://127.0.0.1:1/v1", "model": "m"}},
    )
    router = LLMRouter(cfg)
    seen: dict = {}

    def fake_try_provider(**_kwargs):
        seen["calls"] = registry.non_streaming_calls(tier="large")
        return "", None, "failed"

    with patch.object(router, "_try_provider", side_effect=fake_try_provider):
        with patch.object(router, "_candidate_providers", return_value=(["a"], "normal", {})):
            with router._inference_lock:
                router._complete_text_locked(
                    "", "hi", temperature=0.0, max_tokens=1,
                    request_context={"agent_id": "llm_worker", "lane": "large", "allowed_s": 600.0},
                )  # fmt: skip
    assert [a for _age, a in seen["calls"]] == [600.0], seen
    src = inspect.getsource(llm_worker.LLMWorker._call_llm_with_timeout)
    assert '"allowed_s": timeout_override or self._llm_timeout' in src  # the worker stamps its effective timeout
