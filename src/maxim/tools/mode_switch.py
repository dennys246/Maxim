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


# Valid modes that can be switched to (autonomy levels only)
VALID_MODES = frozenset({"passive", "active", "singularity"})


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

    Gaining autonomy (planning → supervised → autonomous) requires
    human approval — the agent is requesting more power.

    Reducing autonomy (autonomous → supervised → planning) is
    immediate — the agent is voluntarily giving up power.
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

        # Autonomy ordering: planning(0) < supervised(1) < autonomous(2)
        level_order = {
            AutonomyLevel.PLANNING: 0,
            AutonomyLevel.SUPERVISED: 1,
            AutonomyLevel.AUTONOMOUS: 2,
        }

        gaining_autonomy = level_order[target_level] > level_order[current_level]

        if not gaining_autonomy:
            # Reducing autonomy — always safe, immediate
            self._controller.set_level(target_level, reason or "agent voluntarily reduced autonomy")
            return ToolResult(
                success=True,
                output=f"Reduced autonomy to {target_level.value}",
                metadata={
                    "level": target_level.value,
                    "previous_level": current_level.value,
                    "was_change": True,
                    "type": "reduction",
                },
            )

        # Gaining autonomy — requires human approval (or auto-policy if non-interactive)
        from maxim.simulation.sim_logger import should_prompt

        if not should_prompt("autonomy_escalation"):
            # Non-interactive: use auto-escalation policy
            # Default: allow supervised, deny autonomous
            if target_level == AutonomyLevel.SUPERVISED:
                self._controller.set_level(target_level, reason or "auto-approved (supervised)")
                return ToolResult(
                    success=True,
                    output=f"Auto-approved: {target_level.value} (non-interactive mode)",
                    metadata={
                        "level": target_level.value,
                        "previous_level": current_level.value,
                        "was_change": True,
                        "type": "auto_approved",
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"Autonomy escalation to {target_level.value} denied (non-interactive mode). "
                    f"Only supervised level can be auto-approved.",
                    metadata={
                        "level": target_level.value,
                        "previous_level": current_level.value,
                        "type": "auto_denied",
                    },
                )

        # Interactive: queue request for human approval
        request = self._controller.request_autonomy(
            target_level=target_level,
            duration_seconds=duration,
            justification=reason,
        )

        return ToolResult(
            success=True,
            output=f"Requested {target_level.value} level: {reason or 'no reason given'} (awaiting approval)",
            metadata={
                "level": target_level.value,
                "previous_level": current_level.value,
                "was_change": False,
                "type": "request",
                "request_status": request.status,
            },
        )
