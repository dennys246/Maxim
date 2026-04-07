"""Sleep tool — allows the agent to enter low-power mode.

The agent calls this when idle. Wake is automatic on user input
via the existing is_wake_keyword() mechanism in the agent loop.
"""

from __future__ import annotations

import logging
from typing import Any

from maxim.tools.base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class SleepTool(Tool):
    """Enter low-power sleep mode.

    The agent calls this to signal it has nothing to do. While sleeping:
    - LLM processing is skipped (no inference cost)
    - Background tasks continue (memory consolidation, training)
    - The agent wakes automatically when user input arrives

    No separate "wake" tool is needed — wake is triggered by user input
    via the existing wake keyword detection.
    """

    name = "sleep"
    description = (
        "Enter low-power sleep mode. Call this when there is nothing to do. "
        "You will wake automatically when the user speaks or provides input."
    )
    input_schema = {
        "reason": (str, ""),
    }

    def __init__(self, state_manager: Any = None) -> None:
        super().__init__()
        self._state_manager = state_manager

    def execute(self, reason: str = "", **kwargs: Any) -> ToolOutput:
        if self._state_manager is None:
            return ToolOutput(success=False, error="No state manager available")

        if self._state_manager.is_sleeping:
            return ToolOutput(success=True, output="Already sleeping.")

        self._state_manager.set_processing_state("sleep")
        msg = "Entering sleep mode."
        if reason:
            msg += f" Reason: {reason}"
            logger.info("Agent entering sleep: %s", reason)
        else:
            logger.info("Agent entering sleep mode")

        return ToolOutput(success=True, output=msg)
