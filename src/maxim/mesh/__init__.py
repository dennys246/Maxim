"""Agent mesh primitives — identity, naming, messages, and local transport.

These are the foundational abstractions for multi-agent collaboration.
Used by ``simulation/`` and ``create.py`` for research-protocol agent setups.

**What was removed in R0 of archive/llm_path_foundation.md (2026-04-12):**
The peer_registry, peer_info, peer_channel, task_delegation, knowledge,
clock, agent_identity, and admission modules were dead code (zero production
imports) and have been deleted. If you're looking for rate limiting, see
``runtime/rate_limit.py`` which cherry-picked admission.py's logic. If you're
looking for peer-to-peer LLM routing, see ``docs/plans/archive/llm_path_refinement.md``
and the deferred plans in ``docs/plans/deferred/``.
"""

from maxim.mesh.identity import AgentProfile
from maxim.mesh.naming import UMR, parse_umr
from maxim.mesh.message import MESH_PROTOCOL_VERSION, MeshMessage, MeshMessageType
from maxim.mesh.bus import LocalMessageBus

__all__ = [
    "AgentProfile",
    "UMR",
    "parse_umr",
    "MESH_PROTOCOL_VERSION",
    "MeshMessage",
    "MeshMessageType",
    "LocalMessageBus",
]
