"""AgentProfile — identity for mesh agents.

Each agent in a mesh (local or networked) carries a profile that
describes who it is, what it can do, and how to address it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentProfile:
    """Identity and capabilities for a mesh agent.

    Kept deliberately minimal for Phase 0. The agent mesh plan (Phase 1)
    adds hardware capabilities, knowledge stats, and inference info —
    those fields will be added when needed, not speculatively.
    """

    nickname: str  # Short human-readable name ("researcher", "writer")
    role: str  # Functional role ("researcher", "writer", "reviewer", "aut", "orchestrator")
    capabilities: list[str] = field(default_factory=list)  # Tool names or skill names
    personality: str = ""  # One-line persona hint for LLM prompts
    agent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)

    @property
    def display_name(self) -> str:
        """Truncated name for log prefixes (max 20 chars)."""
        if len(self.nickname) <= 20:
            return self.nickname
        return self.nickname[:17] + "..."

    def to_dict(self) -> dict[str, Any]:
        return {
            "nickname": self.nickname,
            "role": self.role,
            "capabilities": self.capabilities,
            "personality": self.personality,
            "agent_id": self.agent_id,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentProfile:
        return cls(
            nickname=d["nickname"],
            role=d["role"],
            capabilities=d.get("capabilities", []),
            personality=d.get("personality", ""),
            agent_id=d.get("agent_id", uuid.uuid4().hex[:12]),
            started_at=d.get("started_at", time.time()),
        )
