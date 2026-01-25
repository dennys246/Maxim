"""Agent interfaces and helpers. Defines roles and ownership of decisions, not algorithms."""

from __future__ import annotations

from maxim.agents.base import Agent, AgentList, as_agent_list
from maxim.agents.goal_agent import GoalAgent
from maxim.agents.llm_agent import ChatLLMAgent, LLMAgent, LLMAgentConfig, TaskLLMAgent

# Agentic architecture components
from maxim.agents.bus import (
    AgentBus,
    DependencyGraph,
    Edge,
    EdgeType,
    FailureStrategy,
    GoalAccepted,
    GoalCompleted,
    GoalPriority,
    MemoryItem,
    MemoryTier,
    Percept,
    ProposedGoal,
    StructuredContext,
    SubGoal,
    SubGoalStatus,
    ToolCall,
    ToolResult,
)
from maxim.agents.perception_agent import PerceptionAgent
from maxim.agents.memory_agent import AssociationIndex, MemoryAgent, MemoryAssociationGraph
from maxim.agents.exec_agent import ExecAgent
from maxim.agents.agentic_goal_agent import AgenticGoalAgent
from maxim.agents.maxim_agent import AgenticMaximAgent, MaximAgent

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
    StrategyInfo,
)

# Context pool for accumulated observations
from maxim.agents.context_pool import (
    AbstractionEntry,
    AgentStateEntry,
    ContextEntry,
    ContextPool,
    ContextPoolConfig,
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
    "AgentBus",
    "DependencyGraph",
    "Edge",
    "EdgeType",
    "FailureStrategy",
    "GoalAccepted",
    "GoalCompleted",
    "GoalPriority",
    "MemoryItem",
    "MemoryTier",
    "Percept",
    "ProposedGoal",
    "StructuredContext",
    "SubGoal",
    "SubGoalStatus",
    "ToolCall",
    "ToolResult",
    # Agentic architecture - Agents
    "AgenticGoalAgent",
    "AgenticMaximAgent",
    "AssociationIndex",
    "ExecAgent",
    "MemoryAgent",
    "MemoryAssociationGraph",
    "PerceptionAgent",
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
    "StrategyInfo",
    # Context Pool
    "AbstractionEntry",
    "AgentStateEntry",
    "ContextEntry",
    "ContextPool",
    "ContextPoolConfig",
]
