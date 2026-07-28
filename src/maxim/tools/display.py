"""Display and interaction tools — agent-driven output/input switching.

These tools let the agent adjust what the user sees and request
user input at runtime, within the constraints set by the user's
``--display`` and ``--interactive`` flags.
"""

from __future__ import annotations

import logging
from typing import Any

from maxim.tools.base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class DisplayModeTool(Tool):
    """Adjust the display verbosity tier at runtime.

    The agent can escalate display (clean → bio → debug) to surface
    important information, but cannot suppress below the user's
    ``--display`` floor.  Escalations are TEMPORARY: the agent loop's
    ``_maybe_auto_revert_display`` tick drops back to the floor after
    ``sim_logger._ESCALATION_HOLD_S`` seconds (time, not turns — a loop
    iteration runs at 2-30Hz, so no turn count exists at the tick point).
    Both the escalate and the revert emit a ``DISPLAY`` record, which the
    console's EVENT seam carries to the web UI as ``kind="display"``.
    """

    name = "display_mode"
    description = (
        "Adjust what the user sees. Use 'bio' to surface memory and "
        "learning activity when something important is happening. Use "
        "'clean' to return to narrative-only output. Use 'debug' only "
        "when diagnosing issues."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "enum": ["clean", "bio", "debug"],
                "description": "Display tier to switch to",
            },
            "reason": {
                "type": "string",
                "description": "Why the display change is needed",
            },
        },
        "required": ["level"],
    }

    def execute(self, level: str = "bio", reason: str = "", **kwargs: Any) -> ToolOutput:
        from maxim.simulation.sim_logger import (
            DisplayTier,
            agent_escalate_display,
            get_display_tier,
        )

        try:
            target = DisplayTier[level.upper()]
        except KeyError:
            return ToolOutput(
                success=False,
                error=f"Invalid display level '{level}'. Use: clean, bio, debug",
            )

        current = get_display_tier()
        applied = agent_escalate_display(target, reason=reason)

        if applied:
            msg = f"Display changed to {level}"
            if reason:
                msg += f" ({reason})"
            logger.info("Agent adjusted display: %s → %s (%s)", current.name, level, reason or "no reason")
            return ToolOutput(success=True, output=msg)
        else:
            return ToolOutput(
                success=False,
                error=f"Cannot set display below user's floor ({current.name.lower()})",
            )


class RequestInteractionTool(Tool):
    """Request user input for an important decision.

    The agent calls this when it encounters a situation where human
    judgment would be valuable.  Fires when ``--interactive true`` or
    when the context is critical (plan approval, safety escalation).

    Set ``critical=true`` for decisions that should prompt even when
    ``--interactive false`` (e.g., plan approval before implementation).
    """

    name = "request_interaction"
    description = (
        "Request user input for an important decision. Only use when "
        "the decision has significant consequences and the agent is "
        "not confident about the right choice. Set critical=true for "
        "plan approvals or safety-critical decisions."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of choices",
            },
            "reason": {
                "type": "string",
                "description": "Why user input is needed",
            },
            "critical": {
                "type": "boolean",
                "description": "If true, prompt even when interactive mode is off (plan approvals, safety decisions)",
            },
        },
        "required": ["question"],
    }

    def __init__(self, prompt_handler: Any = None) -> None:
        self._handler = prompt_handler

    def execute(
        self,
        question: str = "",
        options: list[str] | None = None,
        reason: str = "",
        critical: bool = False,
        **kwargs: Any,
    ) -> ToolOutput:
        from maxim.simulation.sim_logger import should_prompt

        context = "plan_approval" if critical else "agent_request"
        if not should_prompt(context):
            return ToolOutput(
                success=True,
                output=(
                    "Interaction disabled — make your best judgment. "
                    "You are operating autonomously; no user was consulted."
                ),
                metadata={"was_prompted": False, "reason": "interactive_mode_off"},
            )

        try:
            # Use PromptHandler if available
            if self._handler is not None:
                from maxim.interactive.prompts import PromptRequest, PromptType

                prompt_type = PromptType.SINGLE_CHOICE if options else PromptType.FREEFORM
                request = PromptRequest(
                    prompt_type=prompt_type,
                    question=question,
                    options=tuple(options) if options else None,
                )
                try:
                    response = self._handler.prompt(request)
                    return ToolOutput(
                        success=True,
                        output=f"User responded: {response.value}",
                        metadata={"was_prompted": True},
                    )
                except Exception as e:
                    logger.warning("Prompt handler failed: %s — falling back to default", e)

            # Fallback: print to console but cannot collect response
            from maxim.simulation.sim_logger import display_scene

            display_scene(f"\n  Agent asks: {question}")
            if options:
                for i, opt in enumerate(options, 1):
                    display_scene(f"    [{i}] {opt}")

            return ToolOutput(
                success=True,
                output=(
                    "Question displayed to user but response could not be collected. Proceed with your best judgment."
                ),
                metadata={"was_prompted": False, "reason": "no_handler"},
            )
        finally:
            pass


class SetSceneTool(Tool):
    """Set the scene header in the interactive display.

    The agent calls this to describe the current situation — location,
    objective, atmosphere. The header appears between the status bar
    and the agent log, giving the user narrative context at a glance.

    Update the scene whenever the situation changes meaningfully:
    new location, new objective, shift in tone, new encounter.
    """

    name = "set_scene"
    description = (
        "Set the scene header to describe the current situation. Update when location, objective, or context changes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short scene title (location, encounter name, etc.)",
            },
            "description": {
                "type": "string",
                "description": "One-line description of the current situation or objective",
            },
        },
        "required": ["title"],
    }

    def execute(self, title: str = "", description: str = "", **kwargs: Any) -> ToolOutput:
        from maxim.simulation.sim_logger import get_active_display

        display = get_active_display()
        if display is not None:
            display.set_scene(title=title, description=description)
            return ToolOutput(success=True, output=f"Scene set: {title}")

        return ToolOutput(success=True, output="Scene set (no display active)")
