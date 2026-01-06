from __future__ import annotations

# planning/base.py
from abc import ABC, abstractmethod
from typing import Any

class Planner(ABC):
    @abstractmethod
    def propose_plans(self, goal: Any, state: Any, memory: Any) -> list[Any]:
        """
        Return candidate plans.

        Canonical schema:
        - plan: list of action dicts
        - action: {"tool_name": str, "params": dict}
        """
        raise NotImplementedError

class Policy(ABC):
    @abstractmethod
    def score(self, plan: Any, state: Any, memory: Any) -> float:
        """
        Higher is better.
        """
        raise NotImplementedError

    @abstractmethod
    def allow(self, action: Any, state: Any) -> bool:
        """
        Hard constraint check.
        """
        raise NotImplementedError
