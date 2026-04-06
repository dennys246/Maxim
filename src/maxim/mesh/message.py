"""MeshMessage — typed envelope for inter-agent communication.

Messages flow between agents via the LocalMessageBus (in-process) or
PeerChannel (networked, future). The envelope is transport-agnostic.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class MeshMessageType(Enum):
    """Categories of inter-agent messages.

    Started with the types needed by the research protocol.
    The agent mesh plan (Phase 2) adds HEARTBEAT, GOAL_PROPOSAL,
    EXPERIENCE_OFFER, CAUSAL_LINK_SHARE, etc.
    """

    # Research protocol types
    EXPERIMENT_DATA = auto()  # Researcher → Writer: experiment results
    PAPER_DRAFT = auto()  # Writer → Reviewer: draft ready for review
    REVIEW_RESULT = auto()  # Reviewer → Writer: review feedback
    REVISION_REQUEST = auto()  # Reviewer → Researcher: re-run experiment
    TASK_COMPLETE = auto()  # Any agent → orchestrator: done with phase

    # General mesh types (for future use, defined now for completeness)
    REQUEST = auto()  # Generic request
    RESPONSE = auto()  # Generic response
    ERROR = auto()  # Error notification


@dataclass
class MeshMessage:
    """Typed message envelope for inter-agent communication."""

    sender: str  # Sender nickname
    recipient: str  # Recipient nickname ("*" for broadcast)
    msg_type: MeshMessageType
    payload: dict[str, Any] = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    in_reply_to: str | None = None  # msg_id of the message this replies to

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "msg_type": self.msg_type.name,
            "payload": self.payload,
            "msg_id": self.msg_id,
            "timestamp": self.timestamp,
            "in_reply_to": self.in_reply_to,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MeshMessage:
        return cls(
            sender=d["sender"],
            recipient=d["recipient"],
            msg_type=MeshMessageType[d["msg_type"]],
            payload=d.get("payload", {}),
            msg_id=d.get("msg_id", uuid.uuid4().hex[:12]),
            timestamp=d.get("timestamp", time.time()),
            in_reply_to=d.get("in_reply_to"),
        )
