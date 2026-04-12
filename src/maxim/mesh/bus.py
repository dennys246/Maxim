"""LocalMessageBus — in-process message routing between mesh agents.

Thread-safe mailbox-style bus. Each agent registers by nickname and
receives messages via a per-agent queue. No network transport — this
is the in-process mesh used by simulation/research protocols. A networked
transport is deferred to
``docs/plans/deferred/llm_path_multi_peer_dispatch.md`` if/when revived.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Callable

from maxim.mesh.message import MeshMessage, MeshMessageType

logger = logging.getLogger(__name__)

MessageHandler = Callable[[MeshMessage], None]


class LocalMessageBus:
    """In-process pub/sub for mesh agents.

    Agents register handlers by nickname. Messages are delivered
    synchronously (handler called in sender's thread) — fine for
    the research protocol where agents run sequentially.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handlers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._history: list[MeshMessage] = []

    def register(self, nickname: str, handler: MessageHandler) -> None:
        """Register a message handler for an agent nickname."""
        with self._lock:
            self._handlers[nickname].append(handler)
            logger.debug("Bus: registered handler for '%s'", nickname)

    def unregister(self, nickname: str) -> None:
        """Remove all handlers for a nickname."""
        with self._lock:
            self._handlers.pop(nickname, None)

    def send(self, message: MeshMessage) -> None:
        """Deliver a message to its recipient (or broadcast if '*')."""
        with self._lock:
            self._history.append(message)

            if message.recipient == "*":
                # Broadcast — deliver to all except sender
                targets = [h for nick, handlers in self._handlers.items() if nick != message.sender for h in handlers]
            else:
                targets = list(self._handlers.get(message.recipient, []))

        # Deliver outside lock to avoid deadlocks in handlers
        for handler in targets:
            try:
                handler(message)
            except Exception:
                logger.exception(
                    "Bus: handler error for message %s → %s",
                    message.sender,
                    message.recipient,
                )

    def get_history(
        self,
        *,
        sender: str | None = None,
        recipient: str | None = None,
        msg_type: MeshMessageType | None = None,
    ) -> list[MeshMessage]:
        """Query message history with optional filters."""
        with self._lock:
            msgs = list(self._history)

        if sender is not None:
            msgs = [m for m in msgs if m.sender == sender]
        if recipient is not None:
            msgs = [m for m in msgs if m.recipient == recipient]
        if msg_type is not None:
            msgs = [m for m in msgs if m.msg_type == msg_type]
        return msgs

    def clear(self) -> None:
        """Clear all history (for test isolation)."""
        with self._lock:
            self._history.clear()

    @property
    def message_count(self) -> int:
        with self._lock:
            return len(self._history)
