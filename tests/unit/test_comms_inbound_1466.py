"""Inbound comms are pushed by webhook, never polled (#1466, corrected 2026-10-26).

`runtime/bootstrap.py::build_comms_stack` starts the webhook API server; an SMS arrives as
`comms/api.py` -> `TwilioChannel.receive_sms` -> `CommunicationGateway.receive_inbound` -> a percept on
the bus. `MediaLoopMixin._run_headless_loop` still held a poll of `self._gateway` (never assigned)
calling `gateway.poll()` (which does not exist), and `MaximAgent.wire_communication` kept a gateway
nothing read. The positive control pins the live path; the strict red gates pin the dead code's removal.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest


class _Bus:
    def __init__(self) -> None:
        self.published: list = []

    def publish(self, percept) -> None:
        self.published.append(percept)


def test_a_webhook_sms_reaches_the_bus_as_a_percept():
    """The live inbound path (positive control): no polling anywhere on it."""
    from maxim.comms.channels.twilio_channel import TwilioChannel
    from maxim.comms.conversation import ConversationManager
    from maxim.comms.gateway import CommunicationGateway

    bus = _Bus()
    gateway = CommunicationGateway(bus=bus)
    channel = TwilioChannel.__new__(TwilioChannel)  # its constructor needs the twilio SDK; receiving does not
    channel.gateway = gateway
    channel.conv_manager = ConversationManager(bus=bus, nac=None)
    channel.receive_sms("+15550100", "are you there?")

    (percept,) = bus.published
    assert "are you there?" in str(getattr(percept, "content", percept))
    assert percept.metadata.get("external") is True


@pytest.mark.xfail(strict=True, reason="#1466: the headless loop polls a gateway method that does not exist")
def test_the_headless_loop_never_polls_a_gateway(monkeypatch):
    import maxim.embodied_runtime.media_loop as ml
    from maxim.comms.gateway import CommunicationGateway

    swallowed = []
    monkeypatch.setattr(ml, "log_swallowed_exception", lambda e, **k: swallowed.append(e))
    runtime = SimpleNamespace(_gateway=CommunicationGateway(bus=_Bus()))
    stop = threading.Event()

    class _OneTurn(threading.Event):
        def wait(self, timeout=None):
            stop.set()
            return True

    stop.wait = _OneTurn().wait  # one loop pass, then stop
    ml.MediaLoopMixin._run_headless_loop(runtime, stop)
    assert swallowed == [], f"the loop swallowed {swallowed!r}"


@pytest.mark.xfail(strict=True, reason="#1466: wire_communication only stores a gateway nothing reads")
def test_maxim_agent_has_no_dead_gateway_wiring():
    from maxim.agents.maxim_agent import MaximAgent

    assert not hasattr(MaximAgent, "wire_communication")
