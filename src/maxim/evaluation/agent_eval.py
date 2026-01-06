from __future__ import annotations

from .base import Evaluator

class AgentEvaluator(Evaluator):
    """
    Evaluates agent intents and reasoning.
    """
    def evaluate(self, context):
        intent = context.get("intent")
        if not intent:
            return {"valid": False, "reason": "No intent provided"}

        # Example check: confidence threshold
        confidence = intent.get("confidence", 0)
        is_valid = confidence >= 0.5

        return {
            "valid": is_valid,
            "score": confidence,
            "intent_type": intent.get("intent"),
        }
