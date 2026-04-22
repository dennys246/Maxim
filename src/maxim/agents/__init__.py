"""Agent interfaces and helpers. Defines roles and ownership of decisions, not algorithms."""

from __future__ import annotations

from maxim.agents.base import Agent, AgentList, as_agent_list
from maxim.agents.goal_agent import GoalAgent
from maxim.agents.llm_agent import ChatLLMAgent, LLMAgent, LLMAgentConfig, TaskLLMAgent

# Agentic architecture components
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

# Autonomy and LLM worker
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

# Context pool for accumulated observations
from maxim.agents.context_pool import (
    AbstractionEntry,
    AgentStateEntry,
    ContextEntry,
    ContextPool,
    ContextPoolConfig,
)

# Working memory (0.8 — Exec-owned active-reference layer)
from maxim.agents.working_memory import (
    WMEntry,
    WorkingMemoryKind,
    WorkingMemorySet,
)

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
