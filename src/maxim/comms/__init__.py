"""Communication gateway and channel transport layer.

The gateway is a dumb pipe — the agent decides when/what/to whom via tools.
Channels handle transport specifics (Twilio SMS/Voice, webhooks, etc.).
"""

from __future__ import annotations

from maxim.comms.gateway import CommunicationGateway
from maxim.comms.conversation import (
    Conversation,
    ConversationManager,
    ConversationMessage,
)
from maxim.comms.channels.base import Channel

__all__ = [
    "Channel",
    "CommunicationGateway",
    "Conversation",
    "ConversationManager",
    "ConversationMessage",
]
