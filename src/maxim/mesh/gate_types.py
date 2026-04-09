"""Gate types for the Pecking Order Graph (POG-0 prep).

Defines the domain and direction enums used by PeckingRelation edges.
Created before v0.2.0 for wire format stability — adding these after
publication would cause serialization skew between nodes on different versions.

Post-publication, PeckingNode and PeckingRelation (POG-1) will import from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PeckingDomain(str, Enum):
    """Domains over which pecking order applies."""

    AUTHORITY = "authority"  # Who can push updates, gate actions, override decisions
    COMPUTE = "compute"  # Who handles inference requests (GPU routing)
    MEMORY = "memory"  # Who has richer memory / more RAM for context
    KNOWLEDGE = "knowledge"  # Who has more causal links / concepts (wisdom)
    EMBODIMENT = "embodiment"  # Who has physical sensors / actuators


class PeckingDirection(str, Enum):
    """Direction of authority on a graph edge, per domain."""

    PECKS = "pecks"  # I have authority over peer in this domain
    PECKED_BY = "pecked_by"  # Peer has authority over me in this domain
    MUTUAL = "mutual"  # Equals — route by load/latency (siblings)


class GateReason(str, Enum):
    """Why a request was gated (rate-limited, denied, or queued)."""

    RATE_LIMIT = "rate_limit"  # Per-peer RPM exceeded
    CONCURRENCY = "concurrency"  # Max concurrent requests exceeded
    TRUST = "trust"  # Insufficient trust level for this operation
    CAPACITY = "capacity"  # Target node is overloaded (queue depth / thermal)
    AUTHORITY = "authority"  # Requester lacks authority in this domain
    VERSION = "version"  # Version mismatch (requester too old/new)


@dataclass
class GateDecision:
    """Result of a gate check — allow, deny, or queue with reason."""

    allowed: bool
    reason: GateReason | None = None
    retry_after_s: float = 0.0  # Hint for the requester (0 = don't retry)
    message: str = ""  # Human-readable explanation

    @staticmethod
    def allow() -> GateDecision:
        return GateDecision(allowed=True)

    @staticmethod
    def deny(reason: GateReason, message: str = "", retry_after_s: float = 0.0) -> GateDecision:
        return GateDecision(allowed=False, reason=reason, message=message, retry_after_s=retry_after_s)


@dataclass
class EdgeCapacity:
    """Dynamic capacity constraints on a graph edge."""

    max_rps: float = 0.0  # Requests per second this edge can handle
    current_queue_depth: int = 0  # How backed up is the target?
    permission_flags: set[str] = field(default_factory=set)  # e.g. {"contribute", "query", "delegate"}
    gated_until: float = 0.0  # Monotonic time (0 = not gated)
    gate_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_rps": self.max_rps,
            "current_queue_depth": self.current_queue_depth,
            "permission_flags": sorted(self.permission_flags),
            "gated_until": self.gated_until,
            "gate_reason": self.gate_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EdgeCapacity:
        return cls(
            max_rps=d.get("max_rps", 0.0),
            current_queue_depth=d.get("current_queue_depth", 0),
            permission_flags=set(d.get("permission_flags", [])),
            gated_until=d.get("gated_until", 0.0),
            gate_reason=d.get("gate_reason", ""),
        )
