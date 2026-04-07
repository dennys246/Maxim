from __future__ import annotations

# planning/base.py
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.planning.adaptive_planner import PlanCandidate


class Planner(ABC):
    """Abstract base for plan proposal.

    Concrete implementations:
    - ``TaskPlanner``: static goal→tool mapping (legacy)
    - ``AdaptivePlanner``: deep memory integration (ADaPT + Reflexion)
    """

    @abstractmethod
    def propose_plans(
        self,
        goal: dict | str,
        state: Any,
        memory: Any,
    ) -> list[PlanCandidate | list[dict[str, Any]]]:
        """Return candidate plans.

        May return ``PlanCandidate`` instances (AdaptivePlanner) or raw
        ``list[dict]`` action lists (TaskPlanner).  ``DecisionEngine``
        handles both.
        """
        raise NotImplementedError

    def decompose(
        self,
        failed_goal: dict,
        replan_ctx: Any,
        depth: int,
    ) -> PlanCandidate | None:
        """Recursive decomposition on failure (ADaPT).

        Default: no decomposition.  Override in AdaptivePlanner.
        """
        return None


class Policy(ABC):
    """Abstract base for plan scoring and hard-constraint gating."""

    @abstractmethod
    def score(
        self,
        plan: PlanCandidate | list[dict[str, Any]],
        state: Any,
        memory: Any,
    ) -> float:
        """Score a plan candidate.  Higher is better."""
        raise NotImplementedError

    @abstractmethod
    def allow(self, action: dict[str, Any], state: Any) -> bool:
        """Hard constraint check.  Return False to reject the action."""
        raise NotImplementedError
