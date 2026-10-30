"""Tooling utilities.

Tools are the only layer allowed to perform side effects (I/O, network, etc.).

The names below resolve on first access (PEP 562; #1315 step 0a, V9): importing ``maxim.tools`` or one of its
submodules no longer imports every subpackage, so a narrow import loads a narrow closure.
``from maxim.tools import <submodule>`` still works through Python's submodule fallback."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the real types for mypy; at runtime each name is imported on first access
    from maxim.tools.base import Effect, Tool, ToolErrorKind, ToolOutput, ToolResult
    from maxim.tools.comms import CallUserTool, SendMessageTool
    from maxim.tools.filesystem import (
        ExecuteFileTool,
        PathValidationError,
        ReadFileTool,
        WriteFileTool,
        sanitize_path,
        validate_path_traversal,
    )
    from maxim.tools.response import RespondTool, SpeakTool
    from maxim.tools.reachy import (
        COCO_CLASSES,
        DEFAULT_FRAME_HEIGHT,
        DEFAULT_FRAME_WIDTH,
        MIN_MOVEMENT_THRESHOLD_PX,
        VISION_MARGIN_HORIZONTAL,
        VISION_MARGIN_VERTICAL,
        VISION_MAX_U,
        VISION_MAX_V,
        VISION_MIN_U,
        VISION_MIN_V,
        FocusInterestsTool,
        MaximCommandTool,
        MoveTool,
        NoveltyInfo,
        NoveltyRecord,
        NoveltyTrackTool,
        TrackTargetTool,
        clamp_to_vision_range,
        is_significant_movement,
    )
    from maxim.tools.registry import ToolRegistry

# Each public name -> the module that defines it.
_LAZY: dict[str, str] = {
    "Effect": "maxim.tools.base",
    "Tool": "maxim.tools.base",
    "ToolErrorKind": "maxim.tools.base",
    "ToolOutput": "maxim.tools.base",
    "ToolResult": "maxim.tools.base",
    "CallUserTool": "maxim.tools.comms",
    "SendMessageTool": "maxim.tools.comms",
    "ExecuteFileTool": "maxim.tools.filesystem",
    "PathValidationError": "maxim.tools.filesystem",
    "ReadFileTool": "maxim.tools.filesystem",
    "WriteFileTool": "maxim.tools.filesystem",
    "sanitize_path": "maxim.tools.filesystem",
    "validate_path_traversal": "maxim.tools.filesystem",
    "RespondTool": "maxim.tools.response",
    "SpeakTool": "maxim.tools.response",
    "COCO_CLASSES": "maxim.tools.reachy",
    "DEFAULT_FRAME_HEIGHT": "maxim.tools.reachy",
    "DEFAULT_FRAME_WIDTH": "maxim.tools.reachy",
    "MIN_MOVEMENT_THRESHOLD_PX": "maxim.tools.reachy",
    "VISION_MARGIN_HORIZONTAL": "maxim.tools.reachy",
    "VISION_MARGIN_VERTICAL": "maxim.tools.reachy",
    "VISION_MAX_U": "maxim.tools.reachy",
    "VISION_MAX_V": "maxim.tools.reachy",
    "VISION_MIN_U": "maxim.tools.reachy",
    "VISION_MIN_V": "maxim.tools.reachy",
    "FocusInterestsTool": "maxim.tools.reachy",
    "MaximCommandTool": "maxim.tools.reachy",
    "MoveTool": "maxim.tools.reachy",
    "NoveltyInfo": "maxim.tools.reachy",
    "NoveltyRecord": "maxim.tools.reachy",
    "NoveltyTrackTool": "maxim.tools.reachy",
    "TrackTargetTool": "maxim.tools.reachy",
    "clamp_to_vision_range": "maxim.tools.reachy",
    "is_significant_movement": "maxim.tools.reachy",
    "ToolRegistry": "maxim.tools.registry",
}

__all__ = [
    "CallUserTool",
    "COCO_CLASSES",
    "DEFAULT_FRAME_HEIGHT",
    "DEFAULT_FRAME_WIDTH",
    "ExecuteFileTool",
    "FocusInterestsTool",
    "MIN_MOVEMENT_THRESHOLD_PX",
    "MaximCommandTool",
    "MoveTool",
    "NoveltyInfo",
    "NoveltyRecord",
    "NoveltyTrackTool",
    "PathValidationError",
    "ReadFileTool",
    "RespondTool",
    "SendMessageTool",
    "SpeakTool",
    "Effect",
    "Tool",
    "ToolErrorKind",
    "ToolRegistry",
    "ToolOutput",
    "ToolResult",
    "TrackTargetTool",
    "VISION_MARGIN_HORIZONTAL",
    "VISION_MARGIN_VERTICAL",
    "VISION_MAX_U",
    "VISION_MAX_V",
    "VISION_MIN_U",
    "VISION_MIN_V",
    "WriteFileTool",
    "clamp_to_vision_range",
    "is_significant_movement",
    "sanitize_path",
    "validate_path_traversal",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # cache: the next access is a plain module attribute
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
