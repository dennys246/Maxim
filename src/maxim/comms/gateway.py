"""CommunicationGateway — pure transport layer.

Does NOT subscribe to bus events. Does NOT decide when to send messages.
The agent decides that by using the send_message/call_user tools, which
call gateway.send(). Inbound messages are published as Percepts on the bus.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from maxim.agents.autonomy import check_hard_stop
from maxim.agents.bus import Percept

# PerceptContext.channel is a Literal; only these values pass. Unknown
# transports (custom channel adapters) fall back to channel=None on the
# typed context while still setting the free-form ``source`` field.
_KNOWN_CHANNELS = frozenset({"sms", "email", "slack", "speech", "mesh"})

if TYPE_CHECKING:
    from maxim.agents.bus import AgentBus
    from maxim.comms.channels.base import Channel

logger = logging.getLogger(__name__)


class CommunicationGateway:
    """Pure transport layer for external communication.

    - ``send()`` is called by the ``send_message`` tool.
    - ``call()`` is called by the ``call_user`` tool.
    - ``receive_inbound()`` is called by webhook endpoints.

    The gateway never subscribes to bus events — it does NOT decide when
    to communicate. That's the agent's job.
    """

    def __init__(self, bus: "AgentBus") -> None:
        self.bus = bus
        self.channels: dict[str, Channel] = {}

    def register_channel(self, name: str, channel: "Channel") -> None:
        self.channels[name] = channel

    def send(self, channel: str, recipient: str, body: str) -> bool:
        """Called by send_message tool. Just delivers the message."""
        ch = self.channels.get(channel)
        if not ch:
            logger.warning("Channel %r not registered", channel)
            return False
        return ch.send(recipient, body)

    def call(self, recipient: str, message: str) -> bool:
        """Called by call_user tool. Initiates outbound voice call."""
        ch = self.channels.get("twilio")
        if not ch or not hasattr(ch, "call"):
            return False
        return ch.call(recipient, message)

    def receive_inbound(
        self,
        channel: str,
        sender: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Called by webhook endpoint when user texts/calls in.

        SECURITY: Percept control fields are FIXED here — external data
        only populates ``content`` and informational metadata. The gateway
        never trusts external input for salience or hard_override.
        ``hard_override`` is only set if the text matches the existing
        HARD_STOP_TRIGGERS whitelist (returns reason string or None).
        """
        # Typed framing (F0.4). Channel/sender flow through PerceptContext;
        # non-messaging extras (flagged "external" + caller passthrough)
        # stay in the metadata escape hatch.
        from maxim.agents.percept_context import PerceptContext

        ts = time.time()
        percept_context = PerceptContext(
            channel=channel if channel in _KNOWN_CHANNELS else None,
            sender=sender,
            timestamp=ts,
        )
        percept = Percept(
            timestamp=ts,
            source=f"comms:{channel}",
            content=text,
            salience=0.9,  # Fixed — not from external data
            hard_override=None,
            context=percept_context,
            modality="text",
            metadata={
                "external": True,
                **(metadata or {}),
            },
        )
        # Only allow hard overrides via the existing whitelist
        reason = check_hard_stop(text, None)
        if reason:
            percept.hard_override = reason
        self.bus.publish(percept)
