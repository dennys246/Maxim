"""The drive TemporalEvent emitter constructs a VALID event (bugs ledger D9).

D9: 5 of 6 `TemporalEvent` categories have no producer, and the one drive emitter was
also malformed — it passed `temporal_signature=` and `metadata=`, which the frozen
dataclass does not declare, and omitted the required `event_id`/`event_signature`. The
resulting TypeError went into `except Exception: log.debug`, so a dead path looked alive
at every log level anyone runs.

The path stays DORMANT (no production wiring passes a distributor; roadmap item 13's
1.1.x decision is dormancy, not wiring). This test pins that the dormant contract is
well-formed and that a construction error is now LOUD — the score card's D9 guard
("a test constructing the drive TemporalEvent through body.py's emitter").

Verified to fail on the pre-fix emitter: it raised TypeError into a debug log and
recorded nothing.
"""

from __future__ import annotations

import logging

import pytest

from maxim.embodiment.body import Embodiment
from maxim.time.temporal_event import TemporalEvent


class _RecordingDistributor:
    def __init__(self) -> None:
        self.events: list[TemporalEvent] = []

    def record_event(self, event: TemporalEvent) -> None:
        self.events.append(event)


class _ExplodingDistributor:
    def record_event(self, event: TemporalEvent) -> None:
        raise RuntimeError("distributor is down")


def _embodiment(distributor=None) -> Embodiment:
    """A minimal one-entity body; only the drive emitter is under test."""
    from maxim.embodiment.sem import Entity

    return Embodiment(Entity(name="probe", entity_type="test"), distributor=distributor, agent_id="probe_agent")


@pytest.fixture
def body() -> Embodiment:
    return _embodiment()


def test_emitter_is_a_no_op_without_a_distributor(body) -> None:
    """Dormancy in practice: nothing in production passes a distributor."""
    body._emit_drive_temporal_event("drive:hunger:crossed", "drive:hunger")  # must not raise


def test_emitter_constructs_a_valid_temporal_event() -> None:
    dist = _RecordingDistributor()
    body = _embodiment(dist)
    body._emit_drive_temporal_event("drive:hunger:crossed", "drive:hunger")

    assert len(dist.events) == 1
    event = dist.events[0]
    assert isinstance(event, TemporalEvent)
    assert event.event_type == "drive"
    assert event.event_signature == "drive:hunger"
    assert event.event_id and event.agent_id
    assert event.temporal_sig is not None
    assert event.context["source"] == "drive_protocol"
    assert event.context["transition"] == "drive:hunger:crossed"


def test_a_malformed_event_is_reported_at_warning_not_swallowed_at_debug(monkeypatch, caplog) -> None:
    """The D9 failure mode: a construction error must not hide below WARNING."""
    import maxim.embodiment.body as body_mod

    class _Broken:
        def __init__(self, *args, **kwargs):
            raise TypeError("unexpected keyword argument")

    monkeypatch.setattr("maxim.time.temporal_event.TemporalEvent", _Broken)
    body = _embodiment(_RecordingDistributor())
    with caplog.at_level(logging.WARNING, logger=body_mod.log.name):
        body._emit_drive_temporal_event("drive:hunger:crossed", "drive:hunger")
    assert any("malformed" in r.getMessage() for r in caplog.records), (
        "a malformed drive event was not reported at WARNING — this is exactly how D9 hid"
    )


def test_a_distributor_fault_is_reported_and_does_not_break_the_body(caplog) -> None:
    import maxim.embodiment.body as body_mod

    body = _embodiment(_ExplodingDistributor())
    with caplog.at_level(logging.WARNING, logger=body_mod.log.name):
        body._emit_drive_temporal_event("drive:hunger:crossed", "drive:hunger")
    assert any("delivery failed" in r.getMessage() for r in caplog.records)
