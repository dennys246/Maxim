"""No-op stub tools for observation-only mode (no live Maxim instance).

These tools allow the agentic loop to run without a connected robot,
returning success with appropriate messages indicating observation-only mode.
"""

from __future__ import annotations

from typing import Any

from maxim.tools.base import Tool, ToolResult


class NoOpFocusInterestsTool(Tool):
    """
    No-op stub for focus_interests when no live Maxim instance is available.

    Returns success to allow the agent loop to continue without errors.
    """

    name = "focus_interests"
    description = "Focus on interesting objects (observation-only mode - no robot connected)."

    input_schema = {
        "deadzone_px": (int, 20),
        "duration_s": (float, None),
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(
            success=True,
            output={
                "mode": "observation_only",
                "message": "No live robot connected - focus request acknowledged but not executed.",
                "focused": False,
            },
        )


class NoOpMaximCommandTool(Tool):
    """
    No-op stub for maxim_command when no live Maxim instance is available.

    Returns success for safe commands, allowing the agent to continue.
    """

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


class NoOpTrackTargetTool(Tool):
    """
    No-op stub for track_target when no live Maxim instance is available.

    Returns success to allow the agent loop to continue without errors.
    """

    name = "track_target"
    description = "Track and center on a detected object (observation-only mode - no robot connected)."

    input_schema = {
        "deadzone_px": (int, 40),
        "duration_s": (float, 0.3),
        "prefer_people": (bool, True),
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(
            success=True,
            output={
                "mode": "observation_only",
                "message": "No live robot connected - track request acknowledged but not executed.",
                "tracked": False,
            },
        )


class NoOpNoveltyTrackTool(Tool):
    """
    No-op stub for novelty_track when no live Maxim instance is available.

    Provides novelty query functionality even without robot movement capabilities.
    In observation-only mode, novelty tracking still works but movement is disabled.
    """

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
        # Even in no-op mode, maintain novelty tracking state for queries
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
