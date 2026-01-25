"""Mode switching tool for agent-driven mode transitions.

This tool allows mode changes to flow through the decision engine,
preserving policy gating and evaluator visibility.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable

from maxim.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController

logger = logging.getLogger(__name__)


# Valid modes that can be switched to
VALID_MODES = frozenset(
    {
        "observe",
        "reflection",
        "active-assistance",
        "active_assistance",
        "sleep",
        "live",  # Supports agent-defined intent via LiveModeIntentStore
        "train",
        "exploration",
        "research",
    }
)


class ModeSwitchTool(Tool):
    """Tool for switching between operational modes.

    Mode switches flow through the decision engine, which allows
    autonomy level gating and audit logging.
    """

    name = "mode_switch"
    description = "Switch to a different operational mode"
    input_schema = {
        "mode": str,  # Required: target mode name
        "reason": (str, ""),  # Optional: reason for switch
    }

    def __init__(
        self,
        get_current_mode: Callable[[], str],
        set_mode: Callable[[str], None],
        autonomy_controller: AutonomyController | None = None,
    ):
        super().__init__()
        self._get_current_mode = get_current_mode
        self._set_mode = set_mode
        self._autonomy_controller = autonomy_controller
        self._switch_history: list[dict[str, Any]] = []

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the mode switch."""
        target_mode = kwargs.get("mode", "").lower().strip()
        reason = kwargs.get("reason", "")

        # Validate target mode
        if not target_mode:
            return ToolResult(
                success=False,
                error="No target mode specified",
            )

        # Normalize mode name (allow both hyphen and underscore)
        normalized_mode = target_mode.replace("_", "-")
        if normalized_mode not in VALID_MODES and target_mode not in VALID_MODES:
            return ToolResult(
                success=False,
                error=f"Invalid mode: {target_mode}. Valid modes: {', '.join(sorted(VALID_MODES))}",
            )

        current_mode = self._get_current_mode()

        # Check if already in target mode
        if current_mode.replace("_", "-") == normalized_mode:
            return ToolResult(
                success=True,
                output=f"Already in {target_mode} mode",
                metadata={"mode": target_mode, "was_change": False},
            )

        # Log the switch
        switch_record = {
            "timestamp": time.time(),
            "from_mode": current_mode,
            "to_mode": target_mode,
            "reason": reason,
        }
        self._switch_history.append(switch_record)

        # Log to autonomy controller if available
        if self._autonomy_controller:
            self._autonomy_controller.log_action(
                action_type="executed",
                action={"tool_name": "mode_switch", "params": kwargs},
                reasoning=reason or f"Switching from {current_mode} to {target_mode}",
                mode=current_mode,
                confidence=1.0,
            )

        # Perform the switch
        try:
            self._set_mode(target_mode)
            logger.info(f"Mode switched: {current_mode} -> {target_mode} ({reason})")

            return ToolResult(
                success=True,
                output=f"Switched to {target_mode} mode",
                metadata={
                    "mode": target_mode,
                    "previous_mode": current_mode,
                    "was_change": True,
                },
            )

        except Exception as e:
            logger.error(f"Mode switch failed: {e}")
            return ToolResult(
                success=False,
                error=f"Failed to switch mode: {e}",
                metadata={"target_mode": target_mode},
            )

    def get_switch_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent mode switch history."""
        return self._switch_history[-limit:]


class AutonomyLevelTool(Tool):
    """Tool for requesting autonomy level changes.

    In PLANNING mode, this queues a request for human approval.
    In SUPERVISED mode, escalation is automatic, de-escalation requires approval.
    In AUTONOMOUS mode, self-escalation to PLANNING is always allowed.
    """

    name = "autonomy_level"
    description = "Request a change in autonomy level"
    input_schema = {
        "level": str,  # Required: "planning", "supervised", or "autonomous"
        "duration_seconds": (float, None),  # Optional: duration for timed autonomy
        "reason": (str, ""),  # Optional: justification
    }

    def __init__(self, autonomy_controller: AutonomyController):
        super().__init__()
        self._controller = autonomy_controller

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the autonomy level change request."""
        from maxim.agents.autonomy import AutonomyLevel

        target_level_str = kwargs.get("level", "").lower().strip()
        duration = kwargs.get("duration_seconds")
        reason = kwargs.get("reason", "")

        # Parse target level
        try:
            target_level = AutonomyLevel(target_level_str)
        except ValueError:
            valid = ", ".join(level.value for level in AutonomyLevel)
            return ToolResult(
                success=False,
                error=f"Invalid level: {target_level_str}. Valid levels: {valid}",
            )

        current_level = self._controller.current_level

        # Check if already at target level
        if current_level == target_level:
            return ToolResult(
                success=True,
                output=f"Already at {target_level.value} level",
                metadata={"level": target_level.value, "was_change": False},
            )

        # Escalation (more restrictive) is always allowed
        level_order = {
            AutonomyLevel.AUTONOMOUS: 2,
            AutonomyLevel.SUPERVISED: 1,
            AutonomyLevel.PLANNING: 0,
        }

        is_escalation = level_order[target_level] < level_order[current_level]

        if is_escalation:
            # Immediate escalation
            self._controller.set_level(target_level, reason or "agent-requested escalation")
            return ToolResult(
                success=True,
                output=f"Escalated to {target_level.value} level",
                metadata={
                    "level": target_level.value,
                    "previous_level": current_level.value,
                    "was_change": True,
                    "type": "escalation",
                },
            )

        # De-escalation (more permissive) requires human approval
        request = self._controller.request_autonomy(
            target_level=target_level,
            duration_seconds=duration,
            justification=reason,
        )

        return ToolResult(
            success=True,
            output=f"Requested {target_level.value} level (awaiting human approval)",
            metadata={
                "level": target_level.value,
                "previous_level": current_level.value,
                "was_change": False,
                "type": "request",
                "request_status": request.status,
            },
        )
