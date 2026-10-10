"""#1209: PainCircuitBridge books each motion outcome on its own action's NAc event (the #1207 class).

``PainCircuitBridge.record_action_start`` queues an NAc pending event per movement. Before #1209 the bridge
booked its outcomes by SIGNATURE (``NAc.record_outcome(event_id=<signature>)``), so one pain also booked
every other pending event with that signature inside NAc's 300 s window, and nothing retired the pending
event of an action that was replaced, timed out, or completed with learning off. ``movement.py``'s look_at
path starts actions and never completes them, so ``look_at:dy=..:dp=..`` events piled up and one pain booked
NEGATIVE onto all of them (negative inflation).

Driven through the real bridge and a real NAc; the pain arrives through ``_on_pain``, the callback the
detector (or the PainBus) invokes in production.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest

from maxim.bridges.pain_bridge import PainBridgeConfig, PainCircuitBridge
from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc, NACConfig
from maxim.proprioception.pain import PainDetector, PainSignal, PainType

SIG = "look_at:dy=10:dp=0"


def _bridge(**config: Any) -> tuple[PainCircuitBridge, NAc]:
    nac = NAc(NACConfig())
    bridge = PainCircuitBridge(
        nac=nac,
        pain_detector=PainDetector(),
        config=PainBridgeConfig(enable_predictive_harm=False, enable_joint_limit_prediction=False, **config),
    )
    return bridge, nac


def _pain(intensity: float = 0.9) -> PainSignal:
    return PainSignal(pain_type=PainType.EXCESSIVE_VELOCITY, intensity=intensity, timestamp=time.time())


def _observations(nac: NAc, signature: str, valence: Valence) -> int:
    return sum(link.observation_count for link in nac._links.get(signature, []) if link.outcome_valence is valence)


def _pending(nac: NAc, signature: str) -> list[dict[str, Any]]:
    return [e for e in nac._pending_events if e["signature"] == signature]


def test_two_uncompleted_starts_and_one_pain_book_exactly_one_negative() -> None:
    """The issue's gate: look_at starts and never completes, so one pain must not book the stale start."""
    bridge, nac = _bridge()
    bridge.record_action_start(SIG)
    bridge.record_action_start(SIG)
    bridge._on_pain(_pain())
    assert _observations(nac, SIG, Valence.NEGATIVE) == 1


def test_a_pain_never_books_another_producers_same_signature_event() -> None:
    """Pins booking BY ID on its own (retiring a replaced action alone would not pass this): an event some
    other producer queued under the same signature is neither booked nor consumed."""
    bridge, nac = _bridge()
    foreign = nac.record_event("movement", SIG, context={})
    bridge.record_action_start(SIG)
    bridge._on_pain(_pain())
    assert _observations(nac, SIG, Valence.NEGATIVE) == 1
    assert [e["id"] for e in _pending(nac, SIG)] == [foreign]


def test_a_replaced_action_retires_its_event() -> None:
    bridge, nac = _bridge()
    bridge.record_action_start(SIG)
    current = bridge.record_action_start(SIG)
    assert [e["id"] for e in _pending(nac, SIG)] == [current]


def test_a_timed_out_action_retires_its_event() -> None:
    bridge, nac = _bridge(action_timeout_seconds=-1.0)  # every pain arrives too late
    bridge.record_action_start(SIG)
    bridge._on_pain(_pain())
    assert _observations(nac, SIG, Valence.NEGATIVE) == 0
    assert _pending(nac, SIG) == []


def test_completing_with_learning_off_retires_the_action() -> None:
    bridge, nac = _bridge(enable_learning=False)
    bridge.record_action_start(SIG)
    bridge.record_action_complete(success=True)
    assert _pending(nac, SIG) == []
    assert bridge.get_stats()["has_pending_action"] is False


def test_a_completed_action_books_one_positive_and_leaves_nothing_pending() -> None:
    """Guard (green before and after): the turn_around path starts and completes."""
    bridge, nac = _bridge()
    bridge.record_action_start("turn_around")
    bridge.record_action_complete(success=True)
    assert _observations(nac, "turn_around", Valence.POSITIVE) == 1
    assert _pending(nac, "turn_around") == []


def test_the_link_identity_is_unchanged() -> None:
    """Guard (green before and after): booking by id keeps the outcome type, outcome signature and event
    context, so links learned (and persisted) before #1209 are the ones that keep accruing."""
    from maxim.decisions.causal_link import causal_link_id

    bridge, nac = _bridge()
    ctx = {"position": [3, 4], "commanded_6d": {"yaw": 10.0}}
    for _ in range(2):
        bridge.record_action_start(SIG, context=ctx)
        bridge._on_pain(_pain())
    links = nac._links.get(SIG, [])
    assert [(link.id, link.outcome_type, link.observation_count) for link in links] == [
        (causal_link_id(SIG, f"{SIG}:negative", nac._hash_context(ctx)), "result", 2)
    ]


def test_a_failed_start_never_leaves_the_previous_actions_id_behind() -> None:
    """Guard: if NAc refuses the new action's event, a later pain must not be booked on the PREVIOUS
    action's event under the new action's name."""
    bridge, nac = _bridge()
    bridge.record_action_start("look_at:dy=1:dp=1")
    with patch.object(nac, "record_event", side_effect=RuntimeError("probe")):
        with pytest.raises(RuntimeError):
            bridge.record_action_start(SIG)
    assert bridge.get_stats()["has_pending_action"] is False
    bridge._on_pain(_pain())
    assert _observations(nac, "look_at:dy=1:dp=1", Valence.NEGATIVE) == 0
    assert _observations(nac, SIG, Valence.NEGATIVE) == 0


def test_a_late_retire_never_wipes_a_newer_movement() -> None:
    """Pain is handled on the PainBus / detector thread while movements start on the command path: when a
    pain handler finishes with movement A after B has started, B (and its NAc event) must survive."""
    bridge, nac = _bridge()
    bridge.record_action_start("look_at:dy=1:dp=1")
    taken = bridge._pending  # what an in-flight pain handler holds
    current = bridge.record_action_start(SIG)
    bridge._retire(taken)
    assert bridge.get_stats()["has_pending_action"] is True
    assert [e["id"] for e in _pending(nac, SIG)] == [current]


def test_a_pain_whose_event_is_gone_is_not_counted_as_attributed() -> None:
    """When the movement's NAc event was already consumed (another producer's context-similarity booking)
    or aged out, nothing is learned, so the bridge must not count or report an attribution."""
    bridge, nac = _bridge()
    event_id = bridge.record_action_start(SIG)
    nac.discard_pending_event(event_id)  # stands in for a booking by another producer
    bridge._on_pain(_pain())
    assert _observations(nac, SIG, Valence.NEGATIVE) == 0
    assert bridge.get_stats()["total_pain_attributed"] == 0
    assert bridge.get_stats()["has_pending_action"] is False


def test_an_nac_error_while_booking_still_retires_the_movement() -> None:
    bridge, nac = _bridge()
    event_id = bridge.record_action_start(SIG)
    with patch.object(nac, "record_outcome_full", side_effect=RuntimeError("probe")):
        with pytest.raises(RuntimeError):
            bridge._on_pain(_pain())
    assert bridge.get_stats()["has_pending_action"] is False
    assert event_id not in [e["id"] for e in nac._pending_events]
