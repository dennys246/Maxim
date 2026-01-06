# planning/decision_engine.py
from __future__ import annotations

from .constraints import ConstraintViolation

class DecisionEngine:
    def __init__(self, planner, policy, constraints=None):
        self.planner = planner
        self.policy = policy
        self.constraints = constraints or []

    def decide(self, goal, state, memory):
        plans = self.planner.propose_plans(goal, state, memory) or []

        scored = []
        for plan in plans:
            try:
                if not isinstance(plan, list) or not plan:
                    continue
                for c in self.constraints:
                    c.check(plan, state)
                allowed = True
                for action in plan:
                    if hasattr(self.policy, "allow") and not self.policy.allow(action, state):
                        allowed = False
                        break
                if not allowed:
                    continue
                score = self.policy.score(plan, state, memory)
                scored.append((score, plan))
            except ConstraintViolation:
                continue

        if not scored:
            return None

        best_score, best_plan = max(scored, key=lambda x: x[0])
        next_action = best_plan[0]
        return {"action": next_action, "plan": best_plan, "score": best_score}  # next action + context
