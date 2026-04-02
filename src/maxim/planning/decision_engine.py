# planning/decision_engine.py
from __future__ import annotations

from maxim.utils.logging import warn
from .constraints import ConstraintViolation


class DecisionEngine:
    """Select the best action from planner-proposed candidates.

    Supports both PlanCandidate dataclasses (from AdaptivePlanner) and
    raw list[dict] plans (from TaskPlanner / legacy planners).
    """

    def __init__(self, planner, policy, constraints=None):
        self.planner = planner
        self.policy = policy
        self.constraints = constraints or []

    def decide(self, goal, state, memory):
        plans = self.planner.propose_plans(goal, state, memory) or []

        if not plans:
            warn("DecisionEngine: No plans proposed for goal %r", goal)
            return None

        scored = []
        rejected_reasons: list[str] = []
        for plan in plans:
            try:
                # Support both PlanCandidate and raw list[dict]
                if hasattr(plan, "actions"):
                    actions = plan.actions
                elif isinstance(plan, list):
                    actions = plan
                else:
                    rejected_reasons.append("invalid plan format")
                    continue

                if not actions:
                    rejected_reasons.append("empty action list")
                    continue

                for c in self.constraints:
                    c.check(actions, state)
                allowed = True
                for action in actions:
                    if hasattr(self.policy, "allow") and not self.policy.allow(action, state):
                        allowed = False
                        rejected_reasons.append(f"policy rejected action {action.get('tool_name', action)!r}")
                        break
                if not allowed:
                    continue
                score = self.policy.score(plan, state, memory)
                scored.append((score, plan))
            except ConstraintViolation as cv:
                rejected_reasons.append(f"constraint violation: {cv}")
                continue

        if not scored:
            warn(
                "DecisionEngine: All %d plan(s) rejected for goal %r. Reasons: %s",
                len(plans),
                goal,
                "; ".join(rejected_reasons) if rejected_reasons else "unknown",
            )
            return None

        best_score, best_plan = max(scored, key=lambda x: x[0])
        # Extract first action from PlanCandidate or raw list
        if hasattr(best_plan, "actions"):
            next_action = best_plan.actions[0]
        else:
            next_action = best_plan[0]
        return {"action": next_action, "plan": best_plan, "score": best_score}
