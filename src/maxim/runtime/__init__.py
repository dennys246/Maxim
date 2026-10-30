"""Agentic runtime orchestration.

This package wires together agents, planning/decision making, tools/executor, environment, state, and memory.

Phase 3: Includes CaptureManager for direct frame/audio access.

The names below resolve on first access (PEP 562; #1315 step 0a, V9): importing ``maxim.runtime`` or one of its
submodules no longer imports every subpackage, so a narrow import loads a narrow closure.
``from maxim.runtime import <submodule>`` still works through Python's submodule fallback."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the real types for mypy; at runtime each name is imported on first access
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
    from maxim.runtime.concurrent_executor import (
        ConcurrentExecutor,
        ConflictType,
        partition_for_concurrency,
    )
    from maxim.runtime.executor import Executor
    from maxim.runtime.preemption import (
        ExecutionSnapshot,
        ExecutionTracker,
        GoalDiscarded,
        PreemptionCircuit,
        PreemptionCleared,
        PreemptionEvent,
        PreemptionResponse,
        PreemptionSignal,
        PreemptionState,
        ReversalType,
        RobotState,
        SourceConfig,
        WithdrawalParams,
    )
    from maxim.runtime.skill_matcher import SkillMatcher, SkillPrompt
    from maxim.runtime.state import RuntimeState

# Each public name -> the module that defines it.
_LAZY: dict[str, str] = {
    "run_agent_loop": "maxim.runtime.agent_loop",
    "build_decision_engine": "maxim.runtime.bootstrap",
    "build_environment": "maxim.runtime.bootstrap",
    "build_evaluators": "maxim.runtime.bootstrap",
    "build_executor": "maxim.runtime.bootstrap",
    "build_memory": "maxim.runtime.bootstrap",
    "build_state": "maxim.runtime.bootstrap",
    "build_tool_registry": "maxim.runtime.bootstrap",
    "CaptureManager": "maxim.runtime.capture",
    "CapturedAudio": "maxim.runtime.capture",
    "CapturedFrame": "maxim.runtime.capture",
    "ConcurrentExecutor": "maxim.runtime.concurrent_executor",
    "ConflictType": "maxim.runtime.concurrent_executor",
    "partition_for_concurrency": "maxim.runtime.concurrent_executor",
    "Executor": "maxim.runtime.executor",
    "ExecutionSnapshot": "maxim.runtime.preemption",
    "ExecutionTracker": "maxim.runtime.preemption",
    "GoalDiscarded": "maxim.runtime.preemption",
    "PreemptionCircuit": "maxim.runtime.preemption",
    "PreemptionCleared": "maxim.runtime.preemption",
    "PreemptionEvent": "maxim.runtime.preemption",
    "PreemptionResponse": "maxim.runtime.preemption",
    "PreemptionSignal": "maxim.runtime.preemption",
    "PreemptionState": "maxim.runtime.preemption",
    "ReversalType": "maxim.runtime.preemption",
    "RobotState": "maxim.runtime.preemption",
    "SourceConfig": "maxim.runtime.preemption",
    "WithdrawalParams": "maxim.runtime.preemption",
    "SkillMatcher": "maxim.runtime.skill_matcher",
    "SkillPrompt": "maxim.runtime.skill_matcher",
    "RuntimeState": "maxim.runtime.state",
}

__all__ = [
    "CapturedAudio",
    "CapturedFrame",
    "CaptureManager",
    "ConcurrentExecutor",
    "ConflictType",
    "ExecutionSnapshot",
    "ExecutionTracker",
    "Executor",
    "GoalDiscarded",
    "PreemptionCircuit",
    "PreemptionCleared",
    "PreemptionEvent",
    "PreemptionResponse",
    "PreemptionSignal",
    "PreemptionState",
    "ReversalType",
    "RobotState",
    "RuntimeState",
    "SkillMatcher",
    "SkillPrompt",
    "SourceConfig",
    "WithdrawalParams",
    "build_decision_engine",
    "build_environment",
    "build_evaluators",
    "build_executor",
    "build_memory",
    "build_state",
    "build_tool_registry",
    "partition_for_concurrency",
    "run_agent_loop",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # cache: the next access is a plain module attribute
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
