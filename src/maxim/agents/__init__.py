"""Agent interfaces and helpers. Defines roles and ownership of decisions, not algorithms.

The names below resolve on first access (PEP 562; #1315 step 0a, V9): importing ``maxim.agents`` or one of its
submodules no longer imports every subpackage, so a narrow import loads a narrow closure.
``from maxim.agents import <submodule>`` still works through Python's submodule fallback."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the real types for mypy; at runtime each name is imported on first access
    from maxim.agents.base import Agent, AgentList, as_agent_list
    from maxim.agents.goal_agent import GoalAgent
    from maxim.agents.llm_agent import ChatLLMAgent, LLMAgent, LLMAgentConfig, TaskLLMAgent
    from maxim.agents.bus import (
        ActionIntent,
        AgentBus,
        DeliberationOutcome,
        DependencyGraph,
        Edge,
        EdgeType,
        FailureStrategy,
        GoalAccepted,
        GoalCompleted,
        GoalPriority,
        MemoryItem,
        MemoryTier,
        NoOpIntent,
        Percept,
        WorkingMemoryEntry,
        ProposedGoal,
        SpeakIntent,
        StatisticalInsight,
        StatisticalSummary,
        StructuredContext,
        SubGoal,
        SubGoalStatus,
        ToolCall,
        ToolResult,
    )
    from maxim.agents.perception_agent import PerceptionAgent
    from maxim.agents.memory_agent import AssociationIndex, MemoryAgent
    from maxim.agents.exec_agent import ExecAgent
    from maxim.agents.agentic_goal_agent import AgenticGoalAgent
    from maxim.agents.maxim_agent import AgenticMaximAgent, MaximAgent
    from maxim.agents.fear_agent import (
        DangerCategory,
        FearAgent,
        Finding,
        ReviewResult,
        RiskLevel,
    )
    from maxim.agents.statistician_agent import PatternState, StatisticianAgent
    from maxim.agents.autonomy import (
        AutonomyController,
        AutonomyLevel,
        AutonomyRequest,
        AuditEntry,
        Proposal,
        ProposalQueue,
        SafetyConstraints,
        SupervisionPolicy,
        Violation,
        check_hard_stop,
    )
    from maxim.agents.llm_worker import (
        FallbackBehavior,
        LLMProposal,
        LLMRequest,
        LLMWorker,
        ModeInfo,
    )
    from maxim.agents.context_pool import (
        AbstractionEntry,
        AgentStateEntry,
        ContextEntry,
        ContextPool,
        ContextPoolConfig,
    )
    from maxim.agents.working_memory import (
        WMEntry,
        WorkingMemoryKind,
        WorkingMemorySet,
    )

# Each public name -> the module that defines it.
_LAZY: dict[str, str] = {
    "Agent": "maxim.agents.base",
    "AgentList": "maxim.agents.base",
    "as_agent_list": "maxim.agents.base",
    "GoalAgent": "maxim.agents.goal_agent",
    "ChatLLMAgent": "maxim.agents.llm_agent",
    "LLMAgent": "maxim.agents.llm_agent",
    "LLMAgentConfig": "maxim.agents.llm_agent",
    "TaskLLMAgent": "maxim.agents.llm_agent",
    "ActionIntent": "maxim.agents.bus",
    "AgentBus": "maxim.agents.bus",
    "DeliberationOutcome": "maxim.agents.bus",
    "DependencyGraph": "maxim.agents.bus",
    "Edge": "maxim.agents.bus",
    "EdgeType": "maxim.agents.bus",
    "FailureStrategy": "maxim.agents.bus",
    "GoalAccepted": "maxim.agents.bus",
    "GoalCompleted": "maxim.agents.bus",
    "GoalPriority": "maxim.agents.bus",
    "MemoryItem": "maxim.agents.bus",
    "MemoryTier": "maxim.agents.bus",
    "NoOpIntent": "maxim.agents.bus",
    "Percept": "maxim.agents.bus",
    "WorkingMemoryEntry": "maxim.agents.bus",
    "ProposedGoal": "maxim.agents.bus",
    "SpeakIntent": "maxim.agents.bus",
    "StatisticalInsight": "maxim.agents.bus",
    "StatisticalSummary": "maxim.agents.bus",
    "StructuredContext": "maxim.agents.bus",
    "SubGoal": "maxim.agents.bus",
    "SubGoalStatus": "maxim.agents.bus",
    "ToolCall": "maxim.agents.bus",
    "ToolResult": "maxim.agents.bus",
    "PerceptionAgent": "maxim.agents.perception_agent",
    "AssociationIndex": "maxim.agents.memory_agent",
    "MemoryAgent": "maxim.agents.memory_agent",
    "ExecAgent": "maxim.agents.exec_agent",
    "AgenticGoalAgent": "maxim.agents.agentic_goal_agent",
    "AgenticMaximAgent": "maxim.agents.maxim_agent",
    "MaximAgent": "maxim.agents.maxim_agent",
    "DangerCategory": "maxim.agents.fear_agent",
    "FearAgent": "maxim.agents.fear_agent",
    "Finding": "maxim.agents.fear_agent",
    "ReviewResult": "maxim.agents.fear_agent",
    "RiskLevel": "maxim.agents.fear_agent",
    "PatternState": "maxim.agents.statistician_agent",
    "StatisticianAgent": "maxim.agents.statistician_agent",
    "AutonomyController": "maxim.agents.autonomy",
    "AutonomyLevel": "maxim.agents.autonomy",
    "AutonomyRequest": "maxim.agents.autonomy",
    "AuditEntry": "maxim.agents.autonomy",
    "Proposal": "maxim.agents.autonomy",
    "ProposalQueue": "maxim.agents.autonomy",
    "SafetyConstraints": "maxim.agents.autonomy",
    "SupervisionPolicy": "maxim.agents.autonomy",
    "Violation": "maxim.agents.autonomy",
    "check_hard_stop": "maxim.agents.autonomy",
    "FallbackBehavior": "maxim.agents.llm_worker",
    "LLMProposal": "maxim.agents.llm_worker",
    "LLMRequest": "maxim.agents.llm_worker",
    "LLMWorker": "maxim.agents.llm_worker",
    "ModeInfo": "maxim.agents.llm_worker",
    "AbstractionEntry": "maxim.agents.context_pool",
    "AgentStateEntry": "maxim.agents.context_pool",
    "ContextEntry": "maxim.agents.context_pool",
    "ContextPool": "maxim.agents.context_pool",
    "ContextPoolConfig": "maxim.agents.context_pool",
    "WMEntry": "maxim.agents.working_memory",
    "WorkingMemoryKind": "maxim.agents.working_memory",
    "WorkingMemorySet": "maxim.agents.working_memory",
}

__all__ = [
    # Base
    "Agent",
    "AgentList",
    "as_agent_list",
    # Core agents
    "ChatLLMAgent",
    "GoalAgent",
    "LLMAgent",
    "LLMAgentConfig",
    "MaximAgent",
    "TaskLLMAgent",
    # Agentic architecture - Bus and types
    "ActionIntent",
    "AgentBus",
    "DependencyGraph",
    "Edge",
    "EdgeType",
    "FailureStrategy",
    "GoalAccepted",
    "GoalCompleted",
    "GoalPriority",
    "DeliberationOutcome",
    "MemoryItem",
    "MemoryTier",
    "NoOpIntent",
    "WorkingMemoryEntry",
    "Percept",
    "ProposedGoal",
    "SpeakIntent",
    "StructuredContext",
    "SubGoal",
    "SubGoalStatus",
    "ToolCall",
    "ToolResult",
    # Statistical
    "StatisticalInsight",
    "StatisticalSummary",
    # Agentic architecture - Agents
    "AgenticGoalAgent",
    "AgenticMaximAgent",
    # Safety - FearAgent
    "DangerCategory",
    "FearAgent",
    "Finding",
    "ReviewResult",
    "RiskLevel",
    "AssociationIndex",
    "ExecAgent",
    "MemoryAgent",
    "PerceptionAgent",
    # Statistician
    "PatternState",
    "StatisticianAgent",
    # Autonomy
    "AutonomyController",
    "AutonomyLevel",
    "AutonomyRequest",
    "AuditEntry",
    "Proposal",
    "ProposalQueue",
    "SafetyConstraints",
    "SupervisionPolicy",
    "Violation",
    "check_hard_stop",
    # LLM Worker
    "FallbackBehavior",
    "LLMProposal",
    "LLMRequest",
    "LLMWorker",
    "ModeInfo",
    # Context Pool
    "AbstractionEntry",
    "AgentStateEntry",
    "ContextEntry",
    "ContextPool",
    "ContextPoolConfig",
    # Working Memory (0.8)
    "WMEntry",
    "WorkingMemoryKind",
    "WorkingMemorySet",
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
