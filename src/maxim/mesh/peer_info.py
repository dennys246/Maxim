"""PeerInfo — metadata about a known peer in the mesh.

Lightweight dataclass that the PeerRegistry stores per peer. Initially
populated from peer/config.py (tunnel URLs); Phase 0a adds mDNS as an
additional discovery source.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PeerInfo:
    """Metadata about a known mesh peer."""

    peer_id: str  # Stable identifier (node_id from AgentIdentity, or user-assigned)
    base_url: str  # Root URL for /v1/mesh/* endpoints (e.g. "https://maxim.example.com/v1")
    api_key: str = ""  # Bearer token for auth
    trust_level: str = "unknown"  # "verified" | "discovered" | "remote" | "unknown"

    # Optional hardware/identity metadata (populated from HEARTBEAT)
    identity: Any | None = None  # AgentIdentity (typed as Any to avoid circular import)
    models: list[str] = field(default_factory=list)
    device: str = ""  # "gpu" | "cpu" | ""
    vram_gb: float = 0.0

    # Liveness tracking
    last_seen: float = field(default_factory=time.time)
    is_alive: bool = True

    # Heartbeat timeout — peer considered dead after this many seconds of silence
    HEARTBEAT_TIMEOUT_S: float = 60.0

    def check_alive(self) -> bool:
        """Update and return liveness based on last_seen."""
        self.is_alive = (time.time() - self.last_seen) < self.HEARTBEAT_TIMEOUT_S
        return self.is_alive

    def touch(self) -> None:
        """Update last_seen to now."""
        self.last_seen = time.time()
        self.is_alive = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "base_url": self.base_url,
            "trust_level": self.trust_level,
            "models": self.models,
            "device": self.device,
            "vram_gb": self.vram_gb,
            "last_seen": self.last_seen,
            "is_alive": self.is_alive,
        }
