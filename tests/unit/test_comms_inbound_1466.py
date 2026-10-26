"""Inbound comms are pushed by webhook, never polled (#1466, corrected 2026-10-26).

`runtime/bootstrap.py::build_comms_stack` starts the webhook API server; an SMS arrives as
`comms/api.py` -> `TwilioChannel.receive_sms` -> `CommunicationGateway.receive_inbound` -> a percept on
the bus. `MediaLoopMixin._run_headless_loop` still held a poll of `self._gateway` (never assigned)
calling `gateway.poll()` (which does not exist), and `MaximAgent.wire_communication` kept a gateway
nothing read. The positive control pins the live path; the strict red gates pin the dead code's removal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _Bus:
    def __init__(self) -> None:
        self.published: list = []

    def publish(self, percept) -> None:
        self.published.append(percept)


def test_a_twilio_channel_sms_reaches_the_bus_as_a_percept():
    """Entered at the channel (the route's callee): channel -> gateway -> bus, no polling anywhere on it.
    The HTTP hop in front of it is pinned by test_a_twilio_webhook_post_reaches_the_bus."""
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


def test_the_headless_loop_never_polls_a_gateway():
    """Even a gateway that HAS a poll() is never polled: inbound is pushed, not polled."""
    import maxim.embodied_runtime.media_loop as ml

    polls = []
    runtime = SimpleNamespace(_gateway=SimpleNamespace(poll=lambda: polls.append(1)))

    class _OnePass:
        def __init__(self):
            self.passes = 0

        def is_set(self):
            return self.passes > 0

        def wait(self, timeout=None):
            self.passes += 1
            return True

    ml.MediaLoopMixin._run_headless_loop(runtime, _OnePass())
    assert polls == []


def test_maxim_agent_has_no_dead_gateway_wiring():
    from maxim.agents.maxim_agent import MaximAgent

    assert not hasattr(MaximAgent, "wire_communication")


# ── The HTTP hop (#1466 review: every comms endpoint answered 422 since comms/api.py was written) ──────────────


def _api(monkeypatch, bus):
    """The real comms app over a real gateway and Twilio channel; only Twilio's signature validator is stubbed."""
    import sys
    import types

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from maxim.comms.api import create_api_app
    from maxim.comms.channels.twilio_channel import TwilioChannel
    from maxim.comms.conversation import ConversationManager
    from maxim.comms.gateway import CommunicationGateway

    validator = types.ModuleType("twilio.request_validator")

    class _AlwaysValid:
        def __init__(self, token):
            pass

        def validate(self, *args):
            return True

    validator.RequestValidator = _AlwaysValid
    monkeypatch.setitem(sys.modules, "twilio", sys.modules.get("twilio") or types.ModuleType("twilio"))
    monkeypatch.setitem(sys.modules, "twilio.request_validator", validator)
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("MAXIM_API_TOKEN", "api-token")

    gateway = CommunicationGateway(bus=bus)
    channel = TwilioChannel.__new__(TwilioChannel)
    channel.gateway = gateway
    channel.conv_manager = ConversationManager(bus=bus, nac=None)
    gateway.register_channel("twilio", channel)
    return TestClient(create_api_app(bus=bus, gateway=gateway))


def test_a_twilio_webhook_post_reaches_the_bus(monkeypatch):
    bus = _Bus()
    client = _api(monkeypatch, bus)
    r = client.post(
        "/api/twilio/sms", data={"From": "+15550100", "Body": "are you there?"}, headers={"X-Twilio-Signature": "s"}
    )
    assert r.status_code == 200, r.text
    (percept,) = bus.published
    assert "are you there?" in str(percept.content)


def test_the_authenticated_command_endpoint_reaches_the_bus(monkeypatch):
    bus = _Bus()
    client = _api(monkeypatch, bus)
    assert client.post("/api/command", json={"command": "status"}).status_code == 401, "no token, no command"
    r = client.post("/api/command", json={"command": "status"}, headers={"Authorization": "Bearer api-token"})
    assert r.status_code == 200, r.text
    assert len(bus.published) == 1
