"""Channel-agnostic conversation threading.

Manages conversation lifecycle, history, and context injection.
Each channel maps its transport-specific identifiers (phone number,
webhook ID) to a Conversation object.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from maxim.agents.bus import Percept
from maxim.agents.percept_context import PerceptContext

if TYPE_CHECKING:
    from maxim.agents.bus import AgentBus
    from maxim.decisions.nac import NAc

logger = logging.getLogger(__name__)

# ── SMS Command Shortcuts ────────────────────────────────────────────────────

COMMAND_SHORTCUTS: dict[str, dict[str, Any]] = {
    "status": {
        "source": "comms:command",
        "content": "report current status",
        "hard_override": None,
    },
    "stop": {
        "source": "comms:command",
        "content": "stop",
        "hard_override": "Direct stop command",
    },
    "wake": {
        "source": "comms:command",
        "content": "wake up",
        "hard_override": "User requested wake",
    },
}


@dataclass
class ConversationMessage:
    """A single message in a conversation thread."""

    direction: str  # "inbound" or "outbound"
    content: str
    timestamp: float
    channel: str


@dataclass
class Conversation:
    """A threaded conversation with a participant across a channel."""

    id: str
    channel: str  # "twilio:sms", "twilio:voice", "webhook"
    participant: str  # Phone number, webhook ID, etc.
    messages: deque[ConversationMessage]
    created_at: float
    last_activity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Conversations expire after 30 minutes of inactivity."""
        return time.time() - self.last_activity > 1800


class ConversationManager:
    """Channel-agnostic conversation threading.

    Manages conversation lifecycle, history, and context injection.
    Each channel maps its transport-specific identifiers (phone number,
    webhook ID) to a Conversation object.
    """

    def __init__(
        self,
        bus: "AgentBus",
        nac: "NAc | None" = None,
        max_history: int = 20,
        agent_id: str | None = None,
    ) -> None:
        self.bus = bus
        self.nac = nac
        self.conversations: dict[str, Conversation] = {}
        self.participant_index: dict[str, str] = {}  # key → conv_id
        self.max_history = max_history
        self._agent_id = agent_id

    _KNOWN_CHANNELS = frozenset({"sms", "email", "slack", "speech", "mesh", "narrative", "self", "internal"})

    def _build_context(
        self,
        *,
        channel: str | None = None,
        sender: str | None = None,
    ) -> PerceptContext:
        """Build a PerceptContext stamping this manager's agent_id.

        Channel is passed through when it matches a known literal; unknown
        transport strings (e.g. ``"twilio:sms"``) fall back to ``None`` on
        the typed context while the free-form ``source`` field on Percept
        still carries the raw transport identifier.
        """
        return PerceptContext(
            channel=channel if channel in self._KNOWN_CHANNELS else None,  # type: ignore[arg-type]
            sender=sender,
            timestamp=time.time(),
            agent_id=self._agent_id,
        )

    def get_or_create(self, channel: str, participant: str) -> Conversation:
        """Find existing conversation or start a new one."""
        key = f"{channel}:{participant}"
        conv_id = self.participant_index.get(key)

        if conv_id and conv_id in self.conversations:
            conv = self.conversations[conv_id]
            if not conv.is_expired:
                return conv
            # Expired — archive and start fresh
            self._archive(conv)

        conv = Conversation(
            id=str(uuid.uuid4()),
            channel=channel,
            participant=participant,
            messages=deque(maxlen=self.max_history),
            created_at=time.time(),
            last_activity=time.time(),
            metadata={},
        )
        self.conversations[conv.id] = conv
        self.participant_index[key] = conv.id
        return conv

    def record_inbound(self, conv: Conversation, content: str) -> None:
        conv.messages.append(
            ConversationMessage(
                direction="inbound",
                content=content,
                timestamp=time.time(),
                channel=conv.channel,
            )
        )
        conv.last_activity = time.time()

    def record_outbound(self, conv: Conversation, content: str) -> None:
        conv.messages.append(
            ConversationMessage(
                direction="outbound",
                content=content,
                timestamp=time.time(),
                channel=conv.channel,
            )
        )
        conv.last_activity = time.time()

    def get_context_for_prompt(self, conv: Conversation) -> str:
        """Format conversation history for injection into LLM context."""
        if not conv.messages:
            return ""
        lines = [f"Active conversation via {conv.channel} with {conv.participant}:"]
        for msg in conv.messages:
            prefix = "User" if msg.direction == "inbound" else "Maxim"
            lines.append(f"  [{prefix}] {msg.content}")
        return "\n".join(lines)

    def process_inbound(self, channel: str, sender: str, text: str) -> bool:
        """Check for command shortcuts before normal processing.

        Returns True if a shortcut was handled, False for normal flow.
        """
        cmd = text.strip().lower()
        shortcut = COMMAND_SHORTCUTS.get(cmd)
        if shortcut:
            self.bus.publish(
                Percept(
                    timestamp=time.time(),
                    source=shortcut["source"],
                    content=shortcut["content"],
                    hard_override=shortcut["hard_override"],
                    context=self._build_context(channel=channel, sender=sender),
                    modality="text",
                )
            )
            return True
        return False

    def cleanup_expired(self) -> None:
        """Called by agent loop's periodic maintenance cycle.

        Archives expired conversations and removes them from active tracking.
        """
        for conv_id, conv in list(self.conversations.items()):
            if conv.is_expired:
                self._archive(conv)
                del self.conversations[conv_id]
                # Clean up participant index
                key = f"{conv.channel}:{conv.participant}"
                if self.participant_index.get(key) == conv_id:
                    del self.participant_index[key]

    def _archive(self, conv: Conversation) -> None:
        """Store completed conversation in multi-layer memory.

        1. Hippocampus: episodic record via Percept on bus
        2. ATL (via SemanticPromoter): if conversation reveals preferences
           or patterns, they get promoted during session-end consolidation
        """
        summary = self._summarize(conv)
        self.bus.publish(
            Percept(
                timestamp=time.time(),
                source="conversation:archive",
                content=summary,
                salience=0.6,
                context=self._build_context(channel=conv.channel, sender=conv.participant),
                modality="text",
                metadata={
                    "conversation_id": conv.id,
                    "message_count": len(conv.messages),
                    "duration_seconds": conv.last_activity - conv.created_at,
                },
            )
        )
        # Record conversation-level NAc event for pattern promotion
        if self.nac and len(conv.messages) >= 3:
            self.nac.record_event(
                event_type="conversation",
                event_signature=f"conversation:{conv.channel}",
                context={
                    "message_count": len(conv.messages),
                    "duration_seconds": conv.last_activity - conv.created_at,
                    "channel": conv.channel,
                },
            )

    def _summarize(self, conv: Conversation) -> str:
        """Create a text summary of a conversation for archival."""
        msg_count = len(conv.messages)
        duration = conv.last_activity - conv.created_at
        inbound = sum(1 for m in conv.messages if m.direction == "inbound")
        outbound = msg_count - inbound

        lines = [
            f"Conversation via {conv.channel} with {conv.participant}:",
            f"  {msg_count} messages ({inbound} inbound, {outbound} outbound), duration {duration:.0f}s",
        ]
        for msg in conv.messages:
            prefix = "User" if msg.direction == "inbound" else "Maxim"
            lines.append(f"  [{prefix}] {msg.content}")
        return "\n".join(lines)
