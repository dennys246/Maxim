# planning/planning.py
from __future__ import annotations

from .base import Planner

def _path_from_state(state, key: str) -> str | None:
    try:
        data = getattr(state, "data", None)
        if isinstance(data, dict):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception:
        return None
    return None

class TaskPlanner(Planner):
    def propose_plans(self, goal, state, memory):
        if isinstance(goal, dict) and goal.get("tool_name"):
            return [[{"tool_name": str(goal["tool_name"]), "params": dict(goal.get("params") or {})}]]

        if goal == "read_latest_transcript":
            path = _path_from_state(state, "latest_transcript")
            return [[{"tool_name": "read_file", "params": {"path": path, "tail_lines": 1}}]] if path else []

        if goal == "read_latest_log":
            path = _path_from_state(state, "latest_log")
            return [[{"tool_name": "read_file", "params": {"path": path}}]] if path else []

        if goal == "read_latest_training":
            path = _path_from_state(state, "latest_training")
            return [[{"tool_name": "read_file", "params": {"path": path, "tail_lines": 1}}]] if path else []

        if goal == "read_readme":
            return [[{"tool_name": "read_file", "params": {"path": "README.md"}}]]

        if goal == "run_analysis":
            return [
                [{"tool_name": "read_file", "params": {"path": "README.md"}}],
                [{"tool_name": "read_file", "params": {"path": "DECISIONS.md"}}],
            ]
        return []
