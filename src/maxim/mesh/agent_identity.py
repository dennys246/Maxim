"""AgentIdentity — network-level identity for mesh agents.

Extends AgentProfile with hardware capabilities, knowledge statistics,
and inference info needed for cross-agent coordination. Local agents
keep using AgentProfile; AgentIdentity wraps one and adds networked fields.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maxim.mesh.identity import AgentProfile


def _load_or_create_node_id(base_dir: str | Path | None = None) -> str:
    """Load persistent node ID from disk, or create one.

    The node_id is stable across restarts — peers use it to track each other.
    Stored in ``~/.maxim/util/node_id.txt`` (or under *base_dir* if given).
    """
    if base_dir is None:
        from maxim.utils.paths import resolve_user_state
        path = resolve_user_state("util/node_id.txt")
    else:
        path = Path(base_dir) / "data/util/node_id.txt"
    try:
        text = path.read_text().strip()
        if text:
            return text
    except FileNotFoundError:
        pass
    node_id = uuid.uuid4().hex[:16]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(node_id + "\n")
    except OSError:
        pass  # Non-fatal — use ephemeral ID
    return node_id


@dataclass
class AgentIdentity:
    """What this agent is and what it can do.

    Extends AgentProfile with network-level metadata. Broadcast to peers
    via heartbeat. Read-only externally — only the owning agent updates
    its own identity.
    """

    # Core identity
    profile: AgentProfile  # Nickname, role, capabilities, personality
    # NOTE: profile.agent_id is session-scoped (random on each start).
    # node_id is the persistent node identity, saved to disk and stable
    # across restarts. Peers use node_id to track each other.
    node_id: str  # Persistent (from data/util/node_id.txt)
    agent_name: str  # Human-readable ("reachy-kitchen", "desktop-coder")
    started_at: float = field(default_factory=time.time)

    # Hardware capabilities (serialized RuntimeCapabilities)
    capabilities: dict[str, Any] = field(default_factory=dict)
    available_tools: list[str] = field(default_factory=list)
    available_skills: list[str] = field(default_factory=list)
    embodiment_summary: dict[str, Any] | None = None

    # Learned knowledge summary (statistics, not the data itself)
    episodic_memory_count: int = 0
    causal_link_count: int = 0
    concept_count: int = 0
    top_tools: list[dict[str, Any]] = field(default_factory=list)
    top_causal_patterns: list[dict[str, Any]] = field(default_factory=list)
    recent_domains: list[str] = field(default_factory=list)

    # Inference capabilities
    inference_models: list[dict[str, Any]] = field(default_factory=list)
    inference_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "node_id": self.node_id,
            "agent_name": self.agent_name,
            "started_at": self.started_at,
            "capabilities": self.capabilities,
            "available_tools": self.available_tools,
            "available_skills": self.available_skills,
            "embodiment_summary": self.embodiment_summary,
            "episodic_memory_count": self.episodic_memory_count,
            "causal_link_count": self.causal_link_count,
            "concept_count": self.concept_count,
            "top_tools": self.top_tools,
            "top_causal_patterns": self.top_causal_patterns,
            "recent_domains": self.recent_domains,
            "inference_models": self.inference_models,
            "inference_available": self.inference_available,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentIdentity:
        return cls(
            profile=AgentProfile.from_dict(d["profile"]),
            node_id=d["node_id"],
            agent_name=d["agent_name"],
            started_at=d.get("started_at", 0.0),
            capabilities=d.get("capabilities", {}),
            available_tools=d.get("available_tools", []),
            available_skills=d.get("available_skills", []),
            embodiment_summary=d.get("embodiment_summary"),
            episodic_memory_count=d.get("episodic_memory_count", 0),
            causal_link_count=d.get("causal_link_count", 0),
            concept_count=d.get("concept_count", 0),
            top_tools=d.get("top_tools", []),
            top_causal_patterns=d.get("top_causal_patterns", []),
            recent_domains=d.get("recent_domains", []),
            inference_models=d.get("inference_models", []),
            inference_available=d.get("inference_available", False),
        )

    @classmethod
    def build_from_subsystems(
        cls,
        profile: AgentProfile,
        node_id: str,
        agent_name: str,
        *,
        capabilities: dict[str, Any] | None = None,
        tool_names: list[str] | None = None,
        skill_names: list[str] | None = None,
        hippocampus: Any | None = None,
        nac: Any | None = None,
        atl: Any | None = None,
        embodiment: Any | None = None,
        inference_models: list[dict[str, Any]] | None = None,
        inference_available: bool = False,
    ) -> AgentIdentity:
        """Construct identity from live agent subsystems.

        Pulls statistics from hippocampus, NAc, and ATL without importing
        their full modules at module level. Missing subsystems are safe (zero counts).
        """
        # Memory counts
        episodic_count = 0
        if hippocampus is not None:
            episodic_count = len(hippocampus)

        causal_count = 0
        top_causal: list[dict[str, Any]] = []
        if nac is not None:
            stats = nac.stats()
            causal_count = stats.get("total_links", 0)
            # Top 5 high-confidence links
            try:
                candidates = nac.get_promotion_candidates(min_confidence=0.6, min_observations=3)
                top_causal = [
                    {
                        "event": c.link.event_signature,
                        "outcome": c.link.outcome_signature,
                        "confidence": round(c.link.confidence, 3),
                    }
                    for c in candidates[:5]
                ]
            except Exception:
                pass

        concept_count = 0
        if atl is not None:
            try:
                concept_count = len(atl)
            except Exception:
                pass

        # Embodiment summary
        embodiment_summary = None
        if embodiment is not None:
            try:
                embodiment_summary = {
                    "name": getattr(embodiment, "name", "unknown"),
                    "hardware_backed": getattr(embodiment, "hardware_backed", False),
                }
            except Exception:
                pass

        return cls(
            profile=profile,
            node_id=node_id,
            agent_name=agent_name,
            capabilities=capabilities or {},
            available_tools=tool_names or [],
            available_skills=skill_names or [],
            embodiment_summary=embodiment_summary,
            episodic_memory_count=episodic_count,
            causal_link_count=causal_count,
            concept_count=concept_count,
            top_tools=[],  # Populated by caller from executor stats
            top_causal_patterns=top_causal,
            recent_domains=[],  # Populated by caller from goal history
            inference_models=inference_models or [],
            inference_available=inference_available,
        )
