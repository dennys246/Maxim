"""Maxim's embodied runtime — robot I/O glue and live-loop orchestration.

This package contains the mixin stack composed into the ``Maxim`` robot
class (see :mod:`~maxim.embodied_runtime.selfy`). Each mixin owns one
slice of the robot lifecycle: connection management, vision streaming,
movement/kinematics, agentic runtime bootstrap, input handling, and the
media loop.

It is named "embodied_runtime" to distinguish it from:
  - :mod:`maxim.runtime` — lane/worker pool runtime (no body)
  - :mod:`maxim.embodiment` — SEM entity composition (data model, not
    runtime)

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

# Mixin classes for discoverability
from maxim.embodied_runtime.connection import ConnectionMixin
from maxim.embodied_runtime.vision_stream import VisionStreamMixin
from maxim.embodied_runtime.agentic_runtime import AgenticRuntimeMixin
from maxim.embodied_runtime.input_handlers import InputHandlerMixin

__all__ = [
    "AgenticRuntimeMixin",
    "ConnectionMixin",
    "InputHandlerMixin",
    "VisionStreamMixin",
]
