"""Abstract base class for communication channels."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Channel(ABC):
    """Abstract transport channel.

    Each channel handles the transport specifics for one communication
    medium (SMS, voice, webhook, etc.). The gateway delegates to channels
    for actual message delivery.
    """

    @abstractmethod
    def send(self, recipient: str, body: str) -> bool:
        """Send a text message to the recipient. Returns True on success."""
        ...

    def call(self, recipient: str, message: str) -> bool:
        """Initiate an outbound voice call. Override if supported."""
        return False
