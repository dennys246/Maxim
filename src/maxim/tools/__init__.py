"""Tooling utilities.

Tools are the only layer allowed to perform side effects (I/O, network, etc.).
"""

from __future__ import annotations

from maxim.tools.base import Tool, ToolErrorKind, ToolOutput, ToolResult
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
