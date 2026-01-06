"""Agent interfaces and helpers. Defines roles and ownership of decisions, not algorithms."""

from __future__ import annotations

from maxim.agents.base import Agent, AgentList, as_agent_list
from maxim.agents.goal_agent import GoalAgent
from maxim.agents.reachy_agent import ReachyAgent

__all__ = ["Agent", "AgentList", "GoalAgent", "ReachyAgent", "as_agent_list"]
