"""Agent interfaces and helpers. Defines roles and ownership of decisions, not algorithms."""

from __future__ import annotations

from maxim.agents.base import Agent, AgentList, as_agent_list
from maxim.agents.goal_agent import GoalAgent
from maxim.agents.llm_agent import ChatLLMAgent, LLMAgent, LLMAgentConfig, TaskLLMAgent
from maxim.agents.reachy_agent import ReachyAgent

__all__ = [
    "Agent",
    "AgentList",
    "ChatLLMAgent",
    "GoalAgent",
    "LLMAgent",
    "LLMAgentConfig",
    "ReachyAgent",
    "TaskLLMAgent",
    "as_agent_list",
]
