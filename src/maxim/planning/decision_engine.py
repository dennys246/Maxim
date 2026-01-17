# planning/decision_engine.py
from __future__ import annotations

from maxim.utils.logging import warn
from .constraints import ConstraintViolation

class DecisionEngine:
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
        rejected_reasons = []
        for plan in plans:
            try:
                if not isinstance(plan, list) or not plan:
                    rejected_reasons.append("invalid plan format")
                    continue
                for c in self.constraints:
                    c.check(plan, state)
                allowed = True
                for action in plan:
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
        next_action = best_plan[0]
        return {"action": next_action, "plan": best_plan, "score": best_score}  # next action + context
