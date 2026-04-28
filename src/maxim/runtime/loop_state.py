from __future__ import annotations

import logging
import os
import time
from typing import Any

from maxim.utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


def _extract_episode_actions(episode: Any) -> list[dict]:
    """Extract action dicts from an episode for B4 prior-attempt comparison.

    Episodes store their content as dicts with nested state/intent/decision.
    This extracts tool_name + params from each stored step.
    """
    actions: list[dict] = []
    content = getattr(episode, "content", None)
    if isinstance(content, dict):
        # Single-step episode stored by memory.store_raw
        decision = content.get("decision", {})
        action = decision.get("action") if isinstance(decision, dict) else None
        if isinstance(action, dict) and action.get("tool_name"):
            actions.append({"tool_name": action["tool_name"], "params": action.get("params", {})})
    elif isinstance(content, str):
        pass  # text episodes don't have structured actions

    # Also check metadata for plan actions
    metadata = getattr(episode, "metadata", None)
    if isinstance(metadata, dict):
        plan_actions = metadata.get("plan_actions")
        if isinstance(plan_actions, list):
            actions.extend(plan_actions)

    return actions


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
        from maxim.utils.format_version import with_format_version

        abs_path = os.path.abspath(path)
        atomic_write_json(abs_path, with_format_version({"saved_at": time.time(), **meta, **snap}))
    except Exception:
        pass  # Non-critical: state persistence is best-effort


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


def _build_replan_context(
    intent: dict,
    action: dict,
    result: Any,
    state: Any,
    *,
    hippocampus: Any = None,
) -> Any:
    """Build a ReplanContext from failure information.

    The ``hippocampus`` parameter enables B4 prior-attempt retrieval:
    when provided, episodes matching the current goal are retrieved
    and their action sequences are included as ``prior_attempt_actions``
    so the LLM can avoid repeating failed strategies.
    """
    from maxim.planning.plan_document import Phase, PhaseStatus, ReplanContext

    goal_str = str(intent.get("goal", ""))

    # Build a stub Phase so ReplanContext.to_llm_prompt_section works
    # (previously passed a raw string which would crash on .description access).
    stub_phase = Phase(
        id="replan_stub",
        description=goal_str,
        status=PhaseStatus.FAILED,
        plan_id="",
    )

    # B4: retrieve prior plan attempts for the same goal
    prior_attempt_actions: list[list[dict]] = []
    prior_attempt_summaries: list[str] = []
    if hippocampus is not None:
        try:
            prior_episodes = hippocampus.recall(goal=goal_str, limit=10)
            for ep in prior_episodes:
                ep_actions = _extract_episode_actions(ep)
                if ep_actions:
                    prior_attempt_actions.append(ep_actions)
                    raw_content = getattr(ep, "content", "")
                    if isinstance(raw_content, dict):
                        summary = str(raw_content.get("intent", raw_content.get("goal", "")))[:100]
                    elif isinstance(raw_content, str):
                        summary = raw_content[:100] + ("..." if len(raw_content) > 100 else "")
                    else:
                        summary = ""
                    prior_attempt_summaries.append(summary or "prior attempt")
        except Exception:
            logger.debug("B4 prior-attempt retrieval failed", exc_info=True)

    return ReplanContext(
        failed_phase=stub_phase,
        failure_reason=str(getattr(result, "error", "unknown")),
        failure_type=str(getattr(result, "error_kind", "unknown")),
        attempted_sub_goals=[
            {
                "description": goal_str,
                "tool_name": str(action.get("tool_name", "")),
                "params": action.get("params", {}),
                "result": str(getattr(result, "output", ""))[:200],
                "error": str(getattr(result, "error", "")),
            }
        ],
        attempted_tools=[str(action.get("tool_name", ""))],
        completed_phases=[],
        preserved_results={},
        remaining_phases=[],
        original_objective=goal_str,
        energy_remaining={},
        prior_attempt_actions=prior_attempt_actions,
        prior_attempt_summaries=prior_attempt_summaries,
    )
