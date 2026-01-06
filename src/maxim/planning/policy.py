# planning/policy.py
from __future__ import annotations

from .base import Policy

class DefaultPolicy(Policy):
    def score(self, plan, state, memory):
        steps = plan if isinstance(plan, list) else []
        tool_names = []
        for step in steps:
            if isinstance(step, dict):
                name = step.get("tool_name")
                if isinstance(name, str) and name:
                    tool_names.append(name)

        score = 0.0
        if "read_file" in tool_names:
            score += 1.0
        score -= float(len(steps))  # shorter plans preferred
        return float(score)

    def allow(self, action, state):
        tool_name = action
        if isinstance(action, dict):
            tool_name = action.get("tool_name")
        if tool_name == "delete_files" and not getattr(state, "confirmed", False):
            return False
        return True
