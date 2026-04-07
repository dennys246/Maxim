"""Agent mesh primitives — identity, naming, messages, transport, and admission.

These are the foundational abstractions for multi-agent collaboration.
Built and proven locally (research protocol) before adding network transport.
"""

from maxim.mesh.identity import AgentProfile
from maxim.mesh.agent_identity import AgentIdentity
from maxim.mesh.naming import UMR, parse_umr
from maxim.mesh.message import MESH_PROTOCOL_VERSION, MeshMessage, MeshMessageType
from maxim.mesh.bus import LocalMessageBus
from maxim.mesh.peer_info import PeerInfo
from maxim.mesh.peer_registry import PeerRegistry
from maxim.mesh.peer_channel import PeerChannel
from maxim.mesh.admission import MeshAdmissionControl
from maxim.mesh.task_delegation import TaskDelegator, TaskReceiver
from maxim.mesh.clock import PeerClockEstimator
from maxim.mesh.knowledge import (
    CausalLinkProvider,
    CausalLinkReceiver,
    ExperienceBroker,
    KnowledgeProvider,
    KnowledgeReceiver,
    MotorProgramProvider,
    MotorProgramReceiver,
    ReflectionProvider,
    ReflectionReceiver,
)

__all__ = [
    "AgentProfile",
    "AgentIdentity",
    "UMR",
    "parse_umr",
    "MESH_PROTOCOL_VERSION",
    "MeshMessage",
    "MeshMessageType",
    "LocalMessageBus",
    "PeerInfo",
    "PeerRegistry",
    "PeerChannel",
    "MeshAdmissionControl",
    "ExperienceBroker",
    "KnowledgeProvider",
    "KnowledgeReceiver",
    "CausalLinkProvider",
    "CausalLinkReceiver",
    "ReflectionProvider",
    "ReflectionReceiver",
    "MotorProgramProvider",
    "MotorProgramReceiver",
    "TaskDelegator",
    "TaskReceiver",
    "PeerClockEstimator",
]
