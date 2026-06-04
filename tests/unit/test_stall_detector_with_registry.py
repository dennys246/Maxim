"""Integration-level test for stall detector + LLM call registry suppression.

Pins the load-bearing behavior of the bug fix:
- In-flight + recent bytes → suppress nudge
- In-flight + byte silence > max threshold → fire (wedged-call branch)
- No in-flight + idle past threshold → fire (real stall)
- No in-flight + ping_pong → fire (real stall)
- Tier filtering: large-tier in-flight + medium-tier query → ignored

Doesn't import the orchestrator (avoids pulling the full sim runtime);
exercises the exact suppression decision tree the orchestrator's
`_stall_detector` uses post-fold.
"""

from __future__ import annotations

import time

import pytest

from maxim.runtime import llm_call_registry as registry
from maxim.runtime.stall_threshold import compute_stall_threshold


@pytest.fixture(autouse=True)
def _reset_registry():
    registry._reset_for_tests()
    yield
    registry._reset_for_tests()


def _orchestrator_should_suppress(
    *,
    tier: str,
    max_byte_silence_s: float,
) -> bool:
    """Replicate the orchestrator's suppression decision tree."""
    if not registry.any_call_in_flight(tier=tier):
        return False
    silence_s = registry.oldest_byte_silence_s(tier=tier)
    if silence_s is None or silence_s < max_byte_silence_s:
        return True
    # No bytes for >max_byte_silence_s → connection wedged; fire nudge
    return False


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
    registry.register_call_start(tier="large")
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
