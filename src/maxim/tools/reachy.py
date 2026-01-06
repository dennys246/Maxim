from __future__ import annotations

from typing import Any

import numpy as np

from maxim.inference.observation_passive import passive_observation
from maxim.tools.base import Tool, ToolResult
from maxim.utils.logging import warn


class FocusInterestsTool(Tool):
    """
    Focus Reachy Mini's attention on YOLOv8 detections matching the current interests list.

    Requires a live `Maxim` instance (used for the latest frame, vision model, and motor queue).
    """

    name = "focus_interests"
    description = "Run YOLOv8 on the latest frame and move to center an interesting target."

    input_schema = {
        "deadzone_px": (int, 20),  # optional
        "duration_s": (float, None),  # optional
    }

    def __init__(self, maxim: Any) -> None:
        super().__init__()
        self._maxim = maxim
        self._last_frame_ts: float | None = None

    def execute(self, **kwargs: Any) -> ToolResult:
        maxim = self._maxim
        if maxim is None:
            return ToolResult(success=False, error="No Maxim context available.")

        frame = getattr(maxim, "_last_frame", None)
        frame_ts = getattr(maxim, "_last_frame_ts", None)
        if frame is None or not isinstance(frame, np.ndarray):
            return ToolResult(success=False, error="No camera frame available.")

        ts: float | None = None
        try:
            ts = float(frame_ts) if frame_ts is not None else None
        except Exception:
            ts = None

        if ts is not None and self._last_frame_ts is not None and ts <= float(self._last_frame_ts):
            return ToolResult(success=True, output={"skipped": True, "reason": "no_new_frame"})
        if ts is not None:
            self._last_frame_ts = float(ts)

        if getattr(maxim, "segmenter", None) is None:
            return ToolResult(success=False, error="Vision segmenter not initialized.")

        deadzone_px = int(kwargs.get("deadzone_px", 20) or 20)
        duration_s = kwargs.get("duration_s", None)
        duration: float | None = None
        if duration_s is not None:
            try:
                duration = float(duration_s)
            except Exception:
                duration = None

        paused = getattr(maxim, "_training_paused", None)
        pause_training = bool(getattr(maxim, "train", False)) and paused is not None
        lock = getattr(maxim, "_observation_lock", None)

        try:
            if pause_training:
                try:
                    paused.set()
                except Exception:
                    pass

            if lock is None:
                passive_observation(maxim, frame, duration=duration, deadzone_px=deadzone_px, show=False)
            else:
                with lock:
                    passive_observation(maxim, frame, duration=duration, deadzone_px=deadzone_px, show=False)
        except Exception as e:
            warn("focus_interests failed: %s", e, logger=getattr(maxim, "log", None))
            return ToolResult(success=False, error=str(e))
        finally:
            if pause_training:
                try:
                    paused.clear()
                except Exception:
                    pass

        return ToolResult(
            success=True,
            output={
                "focused": True,
                "frame_ts": ts,
                "deadzone_px": int(deadzone_px),
            },
        )


class MaximCommandTool(Tool):
    """
    Execute a small allowlisted set of actions on a live `Maxim` instance.
    """

    name = "maxim_command"
    description = "Execute an allowlisted Maxim command (side effects on Reachy/runtime)."
    input_schema = {
        "command": str,
        "params": (dict, None),  # optional
        "note": (str, None),  # optional (e.g., for label_outcome)
    }

    _ALLOWED: set[str] = {
        "center_vision",
        "mark_trainable_moment",
        "label_outcome",
        "request_sleep",
        "request_observe",
        "request_shutdown",
    }

    def __init__(self, maxim: Any) -> None:
        super().__init__()
        self._maxim = maxim

    def execute(self, **kwargs: Any) -> ToolResult:
        maxim = self._maxim
        if maxim is None:
            return ToolResult(success=False, error="No Maxim context available.")

        command = kwargs.get("command")
        if not isinstance(command, str) or not command:
            return ToolResult(success=False, error="Missing command.")
        command = command.strip()
        if command not in self._ALLOWED:
            return ToolResult(success=False, error=f"Unsupported command: {command}")

        params = kwargs.get("params") if isinstance(kwargs.get("params"), dict) else {}
        note = kwargs.get("note")

        paused = getattr(maxim, "_training_paused", None)
        pause_training = bool(getattr(maxim, "train", False)) and paused is not None and command in {
            "center_vision",
            "mark_trainable_moment",
            "label_outcome",
        }

        try:
            if pause_training:
                try:
                    paused.set()
                except Exception:
                    pass

            if command == "label_outcome":
                code = params.get("code", 0)
                try:
                    maxim.label_outcome(int(code), source="llm", trigger="maxim", note=str(note) if note else None)
                except Exception as e:
                    return ToolResult(success=False, error=str(e))
                return ToolResult(success=True, output={"command": command, "code": int(code)})

            fn = getattr(maxim, command, None)
            if not callable(fn):
                return ToolResult(success=False, error=f"Maxim missing method: {command}")
            fn()
            return ToolResult(success=True, output={"command": command})
        except Exception as e:
            warn("maxim_command failed: %s", e, logger=getattr(maxim, "log", None))
            return ToolResult(success=False, error=str(e))
        finally:
            if pause_training:
                try:
                    paused.clear()
                except Exception:
                    pass
