from __future__ import annotations

import json
import os
import time
from typing import Any

from maxim.utils.logging import warn


def _persist_state_json(state: Any, path: str, *, meta: dict[str, Any]) -> None:
    try:
        if hasattr(state, "save_json") and callable(getattr(state, "save_json")):
            try:
                state.save_json(path, meta=meta)
            except TypeError:
                state.save_json(path)
            return
        if hasattr(state, "snapshot") and callable(getattr(state, "snapshot")):
            snap = state.snapshot()
        else:
            snap = {"state": repr(state)}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump({"saved_at": time.time(), **meta, **snap}, fp, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        warn("Failed to persist runtime state: %s", e)


def _get_failure_strategy(intent: dict, action: dict) -> str:
    """Extract failure strategy from intent/action metadata."""
    if isinstance(intent, dict):
        strategy = intent.get("on_failure", "")
        if strategy:
            return str(strategy).lower()
        sub_goals = intent.get("sub_goals", [])
        for sg in sub_goals:
            if isinstance(sg, dict) and sg.get("tool_name") == action.get("tool_name"):
                return str(sg.get("on_failure", "")).lower()
    return ""


def _get_plan_depth(decision: dict) -> int:
    """Extract current plan depth from decision metadata."""
    plan = decision.get("plan")
    if hasattr(plan, "depth"):
        return plan.depth
    return 0


def _build_replan_context(intent: dict, action: dict, result: Any, state: Any):
    """Build a ReplanContext from failure information."""
    from maxim.planning.plan_document import ReplanContext

    return ReplanContext(
        failed_phase=str(intent.get("goal", "")),
        failure_reason=str(getattr(result, "error", "unknown")),
        failure_type=str(getattr(result, "error_kind", "unknown")),
        attempted_sub_goals=[str(action.get("tool_name", ""))],
        attempted_tools=[str(action.get("tool_name", ""))],
        completed_phases=[],
        preserved_results={},
        remaining_phases=[],
        energy_remaining={},
    )
