from __future__ import annotations

from .base import Evaluator

class ToolExecutionEvaluator(Evaluator):
    """
    Evaluates the outcome of tool executions.
    """
    def evaluate(self, context):
        result = context.get("tool_result")
        if result is None:
            return {"valid": False, "reason": "No result provided"}

        score = 1.0 if result.success else 0.0

        return {
            "valid": result.success,
            "score": score,
            "error": getattr(result, "error", None),
            "output_summary": getattr(result, "output", None),
        }
