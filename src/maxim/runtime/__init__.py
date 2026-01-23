"""Agentic runtime orchestration.

This package wires together agents, planning/decision making, tools/executor, environment, state, and memory.

Phase 3: Includes CaptureManager for direct frame/audio access.
"""

from __future__ import annotations

from maxim.runtime.agent_loop import run_agent_loop
from maxim.runtime.bootstrap import (
    build_decision_engine,
    build_environment,
    build_evaluators,
    build_executor,
    build_memory,
    build_state,
    build_tool_registry,
)
from maxim.runtime.capture import CaptureManager, CapturedAudio, CapturedFrame
from maxim.runtime.executor import Executor
from maxim.runtime.state import RuntimeState

__all__ = [
    "CapturedAudio",
    "CapturedFrame",
    "CaptureManager",
    "Executor",
    "RuntimeState",
    "build_decision_engine",
    "build_environment",
    "build_evaluators",
    "build_executor",
    "build_memory",
    "build_state",
    "build_tool_registry",
    "run_agent_loop",
]
