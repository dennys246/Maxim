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
