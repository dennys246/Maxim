"""Twilio SMS + Voice channel implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from maxim.comms.channels.base import Channel

if TYPE_CHECKING:
    from maxim.comms.conversation import ConversationManager
    from maxim.comms.gateway import CommunicationGateway

logger = logging.getLogger(__name__)


class TwilioChannel(Channel):
    """Twilio SMS and Voice transport channel.

    Requires the ``twilio`` package (optional dependency).
    """

    def __init__(
        self,
        config: dict,
        conv_manager: "ConversationManager",
        gateway: "CommunicationGateway",
    ) -> None:
        try:
            from twilio.rest import Client  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "twilio package required for TwilioChannel. "
                "Install with: pip install twilio"
            ) from exc

        self.client = Client(config["account_sid"], config["auth_token"])
        self.from_number: str = config["from_number"]
        self.voice_enabled: bool = config.get("voice_enabled", False)
        self.conv_manager = conv_manager
        self.gateway = gateway

    # ── Outbound ──────────────────────────────────────────────────────────

    def send(self, recipient: str, body: str) -> bool:
        """Send an SMS via Twilio."""
        try:
            conv = self.conv_manager.get_or_create("twilio:sms", recipient)
            self.conv_manager.record_outbound(conv, body)
            self.client.messages.create(
                body=body, from_=self.from_number, to=recipient
            )
            return True
        except Exception:
            logger.exception("Twilio SMS send failed")
            return False

    def call(self, recipient: str, message: str) -> bool:
        """Initiate an outbound voice call with TTS."""
        if not self.voice_enabled:
            return False
        try:
            from xml.sax.saxutils import escape

            conv = self.conv_manager.get_or_create("twilio:voice", recipient)
            self.conv_manager.record_outbound(conv, message)
            self.client.calls.create(
                twiml=f"<Response><Say>{escape(message)}</Say></Response>",
                from_=self.from_number,
                to=recipient,
            )
            return True
        except Exception:
            logger.exception("Twilio voice call failed")
            return False

    # ── Inbound (called by webhook endpoints) ─────────────────────────────

    def receive_sms(self, from_number: str, body: str) -> None:
        """Handle inbound SMS from Twilio webhook."""
        conv = self.conv_manager.get_or_create("twilio:sms", from_number)
        self.conv_manager.record_inbound(conv, body)
        self.gateway.receive_inbound(
            channel="twilio:sms",
            sender=from_number,
            text=body,
            metadata={
                "conversation_id": conv.id,
                "conversation_context": self.conv_manager.get_context_for_prompt(conv),
                "message_count": len(conv.messages),
            },
        )

    def receive_voice(self, from_number: str, transcript: str) -> None:
        """Handle inbound voice transcript from Twilio webhook."""
        conv = self.conv_manager.get_or_create("twilio:voice", from_number)
        self.conv_manager.record_inbound(conv, transcript)
        self.gateway.receive_inbound(
            channel="twilio:voice",
            sender=from_number,
            text=transcript,
            metadata={
                "conversation_id": conv.id,
                "conversation_context": self.conv_manager.get_context_for_prompt(conv),
            },
        )
