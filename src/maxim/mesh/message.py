"""MeshMessage — typed envelope for inter-agent communication.

Messages flow between agents via the LocalMessageBus (in-process). The
envelope is transport-agnostic — a future networked transport could reuse it
without schema changes. (The previous PeerChannel transport was deleted as
dead code in R0 of llm_path_foundation.md; see
``docs/plans/deferred/llm_path_multi_peer_dispatch.md`` for the replacement
design.)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


MESH_PROTOCOL_VERSION = 1
"""Current mesh protocol version. Bump on breaking payload changes."""


class MeshMessageType(Enum):
    """Categories of inter-agent messages.

    Uses auto() consistently. Wire format serializes via .name.
    """

    # Research protocol types (existing)
    EXPERIMENT_DATA = auto()  # Researcher → Writer: experiment results
    PAPER_DRAFT = auto()  # Writer → Reviewer: draft ready for review
    REVIEW_RESULT = auto()  # Reviewer → Writer: review feedback
    REVISION_REQUEST = auto()  # Reviewer → Researcher: re-run experiment
    TASK_COMPLETE = auto()  # Any agent → orchestrator: done with phase

    # General mesh types
    REQUEST = auto()  # Generic request
    RESPONSE = auto()  # Generic response
    ERROR = auto()  # Error notification

    # Discovery (Phase 2)
    HEARTBEAT = auto()  # Periodic identity broadcast
    IDENTITY_REQUEST = auto()  # "Tell me about yourself"
    IDENTITY_RESPONSE = auto()

    # Task delegation (Phase 2)
    GOAL_PROPOSAL = auto()  # "Can you do this?"
    GOAL_ACCEPTED = auto()  # "Yes, I'll do it"
    GOAL_REJECTED = auto()  # "No, I can't / won't"
    GOAL_RESULT = auto()  # "Here's what happened"

    # Knowledge sharing (Phase 2)
    EXPERIENCE_OFFER = auto()  # "I learned something relevant to you"
    EXPERIENCE_REQUEST = auto()  # "What do you know about X?"
    EXPERIENCE_RESPONSE = auto()
    CAUSAL_LINK_SHARE = auto()  # "This tool fails in this context"
    REFLECTION_SHARE = auto()  # "Here's why it failed"

    # Prediction queries (Phase 2)
    PREDICT_REQUEST = auto()  # "What would happen if I did X?"
    PREDICT_RESPONSE = auto()

    # Inference routing (Phase 2)
    INFERENCE_REQUEST = auto()
    INFERENCE_RESPONSE = auto()


@dataclass
class MeshMessage:
    """Typed message envelope for inter-agent communication.

    Fields sender/recipient/msg_id/in_reply_to are the original local-bus
    fields. protocol_version and correlation_id were added in Phase 2 for
    network transport — optional with backward-compat defaults.
    """

    sender: str  # Sender nickname or agent_id
    recipient: str  # Recipient nickname/agent_id ("*" for broadcast)
    msg_type: MeshMessageType
    payload: dict[str, Any] = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    in_reply_to: str | None = None  # msg_id of the message this replies to

    # Phase 2 additions — optional with defaults for backward compat
    protocol_version: int = MESH_PROTOCOL_VERSION
    correlation_id: str = ""  # Groups a full request/response exchange

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "msg_type": self.msg_type.name,
            "payload": self.payload,
            "msg_id": self.msg_id,
            "timestamp": self.timestamp,
            "in_reply_to": self.in_reply_to,
            "protocol_version": self.protocol_version,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MeshMessage:
        version = d.get("protocol_version", 1)
        if version > MESH_PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported mesh protocol version {version} (local supports up to {MESH_PROTOCOL_VERSION})"
            )
        return cls(
            sender=d["sender"],
            recipient=d["recipient"],
            msg_type=MeshMessageType[d["msg_type"]],
            payload=d.get("payload", {}),
            msg_id=d.get("msg_id", uuid.uuid4().hex[:12]),
            timestamp=d.get("timestamp", time.time()),
            in_reply_to=d.get("in_reply_to"),
            protocol_version=version,
            correlation_id=d.get("correlation_id", ""),
        )
