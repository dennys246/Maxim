from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from maxim.utils.logging import warn


class Agent:
    """
    Minimal base class for "agents" that plug into Maxim loops.

    Design intent:
    - Keep `on_step()` lightweight (hot-loop safe).
    - Offload heavy work to background threads/processes if needed.
    """

    def __init__(self, name: str | None = None, *, enabled: bool = True) -> None:
        raw = str(name or "").strip()
        self.name = raw or type(self).__name__
        self.enabled = bool(enabled)
        self.log = logging.getLogger(f"maxim.agent.{self.name}")

    def on_start(self, **kwargs: Any) -> None:
        return

    def on_stop(self, **kwargs: Any) -> None:
        return

    def on_step(self, **kwargs: Any) -> None:
        return


class AgentList:
    def __init__(self, agents: Iterable[Agent] | None = None) -> None:
        self.agents: list[Agent] = list(agents or [])
        self._started = False

    def add(self, agent: Agent) -> None:
        self.agents.append(agent)

    def on_start(self, **kwargs: Any) -> None:
        if self._started:
            return
        self._started = True
        for agent in self.agents:
            if not getattr(agent, "enabled", True):
                continue
            try:
                agent.on_start(**kwargs)
            except Exception as e:
                warn("Agent %s.on_start failed: %s", getattr(agent, "name", type(agent).__name__), e)

    def on_stop(self, **kwargs: Any) -> None:
        for agent in self.agents:
            if not getattr(agent, "enabled", True):
                continue
            try:
                agent.on_stop(**kwargs)
            except Exception as e:
                warn("Agent %s.on_stop failed: %s", getattr(agent, "name", type(agent).__name__), e)

    def on_step(self, **kwargs: Any) -> None:
        for agent in self.agents:
            if not getattr(agent, "enabled", True):
                continue
            try:
                agent.on_step(**kwargs)
            except Exception as e:
                warn("Agent %s.on_step failed: %s", getattr(agent, "name", type(agent).__name__), e)


def as_agent_list(agents: Any) -> AgentList | None:
    if agents is None:
        return None
    if isinstance(agents, AgentList):
        return agents
    if isinstance(agents, Agent):
        return AgentList([agents])
    if isinstance(agents, (list, tuple)):
        return AgentList([a for a in agents if a is not None])
    raise TypeError(f"Unsupported agents type: {type(agents).__name__}")
