"""DM-specific tools for campaign encounters.

The ChooseTool is dynamically updated per encounter to list valid choices.
It gives the AUT a structured way to select from offered options instead
of relying on free-text classification.
"""

from __future__ import annotations

import logging
from typing import Any

from maxim.tools.base import Tool, ToolOutput

log = logging.getLogger(__name__)


class ChooseTool(Tool):
    """Let the AUT pick from the current encounter's choices.

    The DM runtime updates ``valid_choices`` before each encounter.
    If the AUT calls choose with an invalid option, it gets an error
    listing the valid ones.

    Example:
        AUT calls: choose(option="accept_job")
        Returns: {"chosen": "accept_job", "valid_choices": ["accept_job", "decline", "negotiate_pay"]}
    """

    name = "choose"
    description = "Pick one of the available choices for the current encounter."
    input_schema: dict[str, Any] = {"option": str}
    timeout = 10.0

    def __init__(self) -> None:
        self.valid_choices: list[str] = []
        self._last_choice: str | None = None
        super().__init__()

    def set_choices(self, choices: list[str]) -> None:
        """Update valid choices for the current encounter."""
        self.valid_choices = [c.lower() for c in choices]
        self._last_choice = None
        # Update description to show current choices
        if self.valid_choices:
            opts = ", ".join(self.valid_choices)
            self.description = f"Pick one of the available choices: {opts}"
        else:
            self.description = "No choices available in current encounter."

    @property
    def last_choice(self) -> str | None:
        """The last valid choice made by the AUT."""
        return self._last_choice

    def execute(self, **kwargs: Any) -> Any:
        option = kwargs.get("option", "")
        if isinstance(option, str):
            option = option.strip().lower()

        if not self.valid_choices:
            return ToolOutput(
                success=False,
                error="No choices available in the current encounter.",
            )

        # Exact match
        if option in self.valid_choices:
            self._last_choice = option
            return {
                "chosen": option,
                "valid_choices": self.valid_choices,
                "success": True,
            }

        # Fuzzy match — underscore/space normalization
        option_normalized = option.replace(" ", "_")
        for choice in self.valid_choices:
            if choice.replace(" ", "_") == option_normalized:
                self._last_choice = choice
                return {
                    "chosen": choice,
                    "valid_choices": self.valid_choices,
                    "success": True,
                }

        # Partial match — choice keyword in option or vice versa
        for choice in self.valid_choices:
            if choice in option or option in choice:
                self._last_choice = choice
                return {
                    "chosen": choice,
                    "valid_choices": self.valid_choices,
                    "success": True,
                }

        # No match
        opts = ", ".join(self.valid_choices)
        return ToolOutput(
            success=False,
            error=f"Invalid choice '{option}'. Valid choices are: {opts}",
        )
