"""Tool for agent to define its own live mode intent.

This tool allows the agent to modify its behavioral parameters for live mode,
enabling self-directed evolution of behavior through reflection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from maxim.modes.live_intent import (
    LIVE_MODE_PRIMARY_GOAL,
    MAX_INITIATIVE,
    MIN_INITIATIVE,
    LiveModeIntentStore,
)
from maxim.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController

logger = logging.getLogger(__name__)


class DefineLiveModeIntentTool(Tool):
    """Tool for the agent to define its live mode behavioral intent.

    The agent can use this tool to:
    - Extend its context_prompt (behavioral description)
    - Adjust max_initiative within bounds (0.3-0.8)
    - Set preferred/avoided strategies
    - Add goal context

    Constraints:
    - forbidden_tools stays empty (full access)
    - Initiative bounded to [0.3, 0.8]
    - Primary goal "understand reality and help people" is immutable
    """

    name = "define_live_intent"
    description = (
        "Define your behavioral intent for live mode. "
        "Adjust how you want to behave, your initiative level, and strategy preferences. "
        f"Initiative must be between {MIN_INITIATIVE} and {MAX_INITIATIVE}. "
        f"Primary goal '{LIVE_MODE_PRIMARY_GOAL}' cannot be changed."
    )
    input_schema = {
        "context_prompt_extension": (str, ""),  # Optional behavioral description
        "goal_extension": (str, ""),  # Optional additional goal context
        "max_initiative": (float, None),  # Optional, bounded to 0.3-0.8
        "preferred_strategies": (list, None),  # Optional list of strategy names
        "avoid_strategies": (list, None),  # Optional list of strategy names
        "reasoning": (str, ""),  # Why the agent is making this change
    }

    def __init__(
        self,
        intent_store: LiveModeIntentStore,
        autonomy_controller: AutonomyController | None = None,
    ):
        super().__init__()
        self._store = intent_store
        self._autonomy_controller = autonomy_controller

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the intent definition."""
        context_ext = kwargs.get("context_prompt_extension")
        goal_ext = kwargs.get("goal_extension")
        initiative = kwargs.get("max_initiative")
        preferred = kwargs.get("preferred_strategies")
        avoid = kwargs.get("avoid_strategies")
        reasoning = kwargs.get("reasoning", "")

        # Validate initiative bounds
        if initiative is not None:
            try:
                initiative = float(initiative)
                if initiative < MIN_INITIATIVE or initiative > MAX_INITIATIVE:
                    original = initiative
                    initiative = max(MIN_INITIATIVE, min(MAX_INITIATIVE, initiative))
                    logger.warning(
                        f"Initiative {original} clamped to {initiative} (bounds: {MIN_INITIATIVE}-{MAX_INITIATIVE})"
                    )
            except (TypeError, ValueError):
                return ToolResult(
                    success=False,
                    error=f"Invalid initiative value: {initiative}",
                )

        # Validate strategy lists
        if preferred is not None and not isinstance(preferred, list):
            return ToolResult(
                success=False,
                error="preferred_strategies must be a list",
            )
        if avoid is not None and not isinstance(avoid, list):
            return ToolResult(
                success=False,
                error="avoid_strategies must be a list",
            )

        # Update intent
        try:
            intent = self._store.update(
                context_prompt_extension=context_ext,
                goal_extension=goal_ext,
                max_initiative=initiative,
                preferred_strategies=preferred,
                avoid_strategies=avoid,
            )

            # Log to autonomy controller
            if self._autonomy_controller:
                self._autonomy_controller.log_action(
                    action_type="executed",
                    action={"tool_name": "define_live_intent", "params": kwargs},
                    reasoning=reasoning or "Updating live mode intent",
                    mode="live",
                    confidence=1.0,
                )

            logger.info(f"Live mode intent updated (v{intent.version}): initiative={intent.max_initiative}")

            return ToolResult(
                success=True,
                output=f"Live mode intent updated successfully (version {intent.version})",
                metadata={
                    "version": intent.version,
                    "max_initiative": intent.max_initiative,
                    "preferred_strategies": intent.preferred_strategies,
                    "avoid_strategies": intent.avoid_strategies,
                    "has_context_extension": bool(intent.context_prompt_extension),
                    "has_goal_extension": bool(intent.goal_extension),
                },
            )

        except Exception as e:
            logger.error(f"Failed to update live mode intent: {e}")
            return ToolResult(
                success=False,
                error=f"Failed to update intent: {e}",
            )


class ReviewLiveModeIntentTool(Tool):
    """Tool for the agent to review and reflect on its live mode intent.

    Used during reflection mode to evaluate how well the current
    intent configuration is working.
    """

    name = "review_live_intent"
    description = (
        "Review your current live mode intent configuration and historical outcomes. "
        "Use this during reflection to evaluate and consider adjustments."
    )
    input_schema = {
        "include_outcomes": (bool, True),  # Include historical outcomes
        "outcome_limit": (int, 10),  # Number of recent outcomes to include
    }

    def __init__(self, intent_store: LiveModeIntentStore):
        super().__init__()
        self._store = intent_store

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the review."""
        include_outcomes = kwargs.get("include_outcomes", True)
        outcome_limit = kwargs.get("outcome_limit", 10)

        try:
            intent = self._store.load()

            result = {
                "primary_goal": LIVE_MODE_PRIMARY_GOAL,
                "version": intent.version,
                "max_initiative": intent.max_initiative,
                "initiative_bounds": {"min": MIN_INITIATIVE, "max": MAX_INITIATIVE},
                "context_prompt_extension": intent.context_prompt_extension,
                "goal_extension": intent.goal_extension,
                "preferred_strategies": intent.preferred_strategies,
                "avoid_strategies": intent.avoid_strategies,
                "modification_count": intent.modification_count,
                "last_modified_at": intent.last_modified_at,
                "last_reflection_at": intent.last_reflection_at,
                "reflection_insights": intent.reflection_insights[-5:],
            }

            if include_outcomes:
                outcomes = intent.historical_outcomes[-outcome_limit:]
                success_count = sum(1 for o in outcomes if o.get("success"))
                result["recent_outcomes"] = outcomes
                result["success_rate"] = success_count / len(outcomes) if outcomes else 0.0

            return ToolResult(
                success=True,
                output=result,
                metadata={"version": intent.version},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to review intent: {e}",
            )


class RecordLiveIntentInsightTool(Tool):
    """Tool for recording reflection insights about live mode behavior."""

    name = "record_live_insight"
    description = (
        "Record an insight about your live mode behavior during reflection. "
        "These insights help track learning over time."
    )
    input_schema = {
        "insight": str,  # Required: the insight to record
    }

    def __init__(self, intent_store: LiveModeIntentStore):
        super().__init__()
        self._store = intent_store

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the insight recording."""
        insight = kwargs.get("insight", "").strip()

        if not insight:
            return ToolResult(
                success=False,
                error="Insight cannot be empty",
            )

        try:
            self._store.record_reflection_insight(insight)

            return ToolResult(
                success=True,
                output=f"Insight recorded: {insight[:100]}{'...' if len(insight) > 100 else ''}",
                metadata={"insight_length": len(insight)},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to record insight: {e}",
            )


class RecordLiveOutcomeTool(Tool):
    """Tool for recording action outcomes for live mode learning."""

    name = "record_live_outcome"
    description = (
        "Record the outcome of an action taken in live mode. "
        "This helps track what works and informs future behavior adjustments."
    )
    input_schema = {
        "action": str,  # Required: what action was taken
        "success": bool,  # Required: whether it succeeded
        "context": str,  # Required: situation context
        "reasoning": (str, ""),  # Optional: why this outcome occurred
    }

    def __init__(self, intent_store: LiveModeIntentStore):
        super().__init__()
        self._store = intent_store

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the outcome recording."""
        action = kwargs.get("action", "").strip()
        success = kwargs.get("success")
        context = kwargs.get("context", "").strip()
        reasoning = kwargs.get("reasoning", "")

        if not action:
            return ToolResult(
                success=False,
                error="Action cannot be empty",
            )

        if success is None:
            return ToolResult(
                success=False,
                error="Success must be specified (true or false)",
            )

        if not context:
            return ToolResult(
                success=False,
                error="Context cannot be empty",
            )

        try:
            self._store.record_outcome(
                action=action,
                success=bool(success),
                context=context,
                reasoning=reasoning,
            )

            return ToolResult(
                success=True,
                output=f"Outcome recorded: {action} ({'success' if success else 'failure'})",
                metadata={
                    "action": action,
                    "success": success,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to record outcome: {e}",
            )
