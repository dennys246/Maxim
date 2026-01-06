from __future__ import annotations

from typing import Any

from maxim.agents.base import Agent


class GoalAgent(Agent):
    agent_name = "goal"

    def __init__(
        self,
        goal: Any,
        *,
        name: str | None = None,
        confidence: float = 1.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, enabled=enabled)
        self.goal = goal
        self.confidence = float(confidence)

    def propose_intent(self, state: Any, memory: Any, **kwargs: Any) -> dict[str, Any] | None:
        return {"goal": self.goal, "confidence": float(self.confidence)}

