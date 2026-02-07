"""No-op stub tools for observation-only mode (no live Maxim instance).

These tools allow the agentic loop to run without a connected robot,
returning success with appropriate messages indicating observation-only mode.

Uses a factory pattern to generate stubs from configuration, reducing code duplication.
"""

from __future__ import annotations

from typing import Any

from maxim.tools.base import Tool, ToolResult


# ─────────────────────────────────────────────────────────────────────────────
# Stub Factory
# ─────────────────────────────────────────────────────────────────────────────


def create_stub_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    output_fields: dict[str, Any] | None = None,
    always_allowed: bool = False,
) -> type[Tool]:
    """Factory to create no-op stub tool classes.

    Args:
        name: Tool name (must match the real tool's name).
        description: Tool description for observation-only mode.
        input_schema: Input schema matching the real tool.
        output_fields: Additional fields to include in the output dict.
        always_allowed: Whether this tool is always allowed (for safety tools).

    Returns:
        A Tool subclass configured as a no-op stub.
    """
    extra_output = output_fields or {}

    class StubTool(Tool):
        __doc__ = f"No-op stub for {name} when no live Maxim instance is available."

        def execute(self, **kwargs: Any) -> ToolResult:
            output = {
                "mode": "observation_only",
                "message": f"No live robot connected - '{name}' acknowledged but not executed.",
                **extra_output,
            }
            # Include relevant kwargs in output for debugging
            for key in input_schema:
                if key in kwargs:
                    output[key] = kwargs[key]
            return ToolResult(success=True, output=output)

    StubTool.name = name
    StubTool.description = description
    StubTool.input_schema = input_schema
    if always_allowed:
        StubTool.always_allowed = True

    return StubTool


# ─────────────────────────────────────────────────────────────────────────────
# Stub Definitions
# ─────────────────────────────────────────────────────────────────────────────

# FocusInterestsTool stub
NoOpFocusInterestsTool = create_stub_tool(
    name="focus_interests",
    description="Focus on interesting objects (observation-only mode - no robot connected).",
    input_schema={
        "target_class": (str, None),
        "deadzone_px": (int, 20),
    },
    output_fields={"focused": False},
)

# TrackTargetTool stub
NoOpTrackTargetTool = create_stub_tool(
    name="track_target",
    description="Track and center on a detected object (observation-only mode - no robot connected).",
    input_schema={
        "target_class": (str, None),
        "deadzone_px": (int, 40),
        "duration_s": (float, 0.3),
        "prefer_people": (bool, True),
    },
    output_fields={"tracked": False},
)

# MoveTool stub (always_allowed to match real MoveTool)
NoOpMoveTool = create_stub_tool(
    name="move",
    description="Move Maxim's head to a target position (observation-only mode - no robot connected).",
    input_schema={
        "target_x": (float, None),
        "target_y": (float, None),
        "x": (float, None),
        "y": (float, None),
        "z": (float, None),
        "roll": (float, None),
        "pitch": (float, None),
        "yaw": (float, None),
        "duration": (float, None),
    },
    output_fields={"moved": False},
    always_allowed=True,
)

# MaximCommandTool stub (needs command validation)
class NoOpMaximCommandTool(Tool):
    """No-op stub for maxim_command with command validation."""

    name = "maxim_command"
    description = "Execute a Maxim command (observation-only mode - no robot connected)."

    input_schema = {
        "command": str,
        "params": (dict, None),
        "note": (str, None),
    }

    _ALLOWED: set[str] = {
        "center_vision",
        "goto_pose",
        "look_at_image",
        "mark_trainable_moment",
        "move",
        "move_antenna",
        "turn_around",
        "label_outcome",
        "request_sleep",
        "request_observe",
        "request_shutdown",
        "update_interests",
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command")
        if not isinstance(command, str) or not command:
            return ToolResult(success=False, error="Missing command.")

        command = command.strip()
        if command not in self._ALLOWED:
            return ToolResult(success=False, error=f"Unsupported command: {command}")

        return ToolResult(
            success=True,
            output={
                "mode": "observation_only",
                "command": command,
                "message": f"No live robot connected - '{command}' acknowledged but not executed.",
            },
        )


# NoveltyTrackTool stub (needs action-based behavior)
class NoOpNoveltyTrackTool(Tool):
    """No-op stub for novelty_track with action-based behavior."""

    name = "novelty_track"
    description = "Track novel objects and center vision (observation-only mode - no robot connected)."

    input_schema = {
        "action": (str, "track"),  # "track", "query", "reset"
        "novelty_threshold": (float, 0.5),
        "deadzone_px": (int, 40),
        "duration_s": (float, 0.3),
        "top_k": (int, 5),
        "class_filter": (list, None),
    }

    def __init__(self) -> None:
        super().__init__()
        self._track_memory: dict[int, dict] = {}
        self._frame_count = 0

    def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "track")).lower()

        if action == "query":
            return ToolResult(
                success=True,
                output={
                    "mode": "observation_only",
                    "novelty_rankings": [],
                    "total_tracked": len(self._track_memory),
                    "message": "No live robot connected - novelty query available but no detections.",
                },
            )
        elif action == "reset":
            count = len(self._track_memory)
            self._track_memory.clear()
            self._frame_count = 0
            return ToolResult(
                success=True,
                output={
                    "mode": "observation_only",
                    "reset": True,
                    "cleared_tracks": count,
                },
            )
        else:  # track
            return ToolResult(
                success=True,
                output={
                    "mode": "observation_only",
                    "message": "No live robot connected - novelty track acknowledged but movement disabled.",
                    "tracked": False,
                    "skipped": True,
                    "reason": "observation_only_mode",
                },
            )


__all__ = [
    "NoOpFocusInterestsTool",
    "NoOpTrackTargetTool",
    "NoOpMoveTool",
    "NoOpMaximCommandTool",
    "NoOpNoveltyTrackTool",
    "create_stub_tool",
]
