"""Maxim's conscience - safety enforcement, self-awareness, and orchestration.

Submodules:
    selfy          - Core Maxim orchestrator (composes all mixins)
    connection     - Connection management and reconnection (ConnectionMixin)
    vision_stream  - Vision event streaming (VisionStreamMixin)
    agentic_runtime - Agentic runtime lifecycle (AgenticRuntimeMixin)
    movement       - Movement & kinematics (MovementMixin)
    workers        - Live loop worker functions (module-level)
    input_handlers - Input & response handling (InputHandlerMixin)
    media_loop     - Live loop & media pipeline (MediaLoopMixin)
"""

from __future__ import annotations

# FearAgent moved to maxim.agents - re-export for backward compatibility
from maxim.agents.fear_agent import (
    DangerCategory,
    FearAgent,
    Finding,
    ReviewResult,
    RiskLevel,
)

# Mixin classes for discoverability
from maxim.conscience.connection import ConnectionMixin
from maxim.conscience.vision_stream import VisionStreamMixin
from maxim.conscience.agentic_runtime import AgenticRuntimeMixin
from maxim.conscience.input_handlers import InputHandlerMixin

__all__ = [
    # FearAgent (backward compat)
    "DangerCategory",
    "FearAgent",
    "Finding",
    "ReviewResult",
    "RiskLevel",
    # Mixins
    "AgenticRuntimeMixin",
    "ConnectionMixin",
    "InputHandlerMixin",
    "VisionStreamMixin",
]
