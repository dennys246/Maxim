from __future__ import annotations

from .base import Evaluator

class PlanEvaluator(Evaluator):
    """
    Evaluates candidate plans before execution.
    """
    def evaluate(self, context):
        plan = context.get("plan")
        if not plan:
            return {"valid": False, "reason": "No plan provided"}

        registered = set(context.get("registered_tools", []))
        steps = None
        if isinstance(plan, dict):
            steps = plan.get("steps")
        if steps is None and isinstance(plan, list):
            steps = plan
        if not isinstance(steps, list) or not steps:
            return {"valid": False, "reason": "Plan has no steps"}

        # Simple heuristic: all steps reference known tools.
        def _tool_name(step):
            if isinstance(step, str):
                return step
            if isinstance(step, dict):
                return step.get("tool_name") or step.get("tool")
            return None

        all_tools_registered = all((_tool_name(step) in registered) for step in steps)

        score = 1.0 if all_tools_registered else 0.0

        return {
            "valid": all_tools_registered,
            "score": score,
            "issues": [] if all_tools_registered else ["Unknown tools in plan"]
        }
