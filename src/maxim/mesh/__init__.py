"""Agent mesh primitives — identity, naming, messages, and local bus.

These are the foundational abstractions for multi-agent collaboration.
Built and proven locally (research protocol) before adding network transport.
"""

from maxim.mesh.identity import AgentProfile
from maxim.mesh.naming import UMR, parse_umr
from maxim.mesh.message import MeshMessage, MeshMessageType
from maxim.mesh.bus import LocalMessageBus

__all__ = [
    "AgentProfile",
    "UMR",
    "parse_umr",
    "MeshMessage",
    "MeshMessageType",
    "LocalMessageBus",
]
