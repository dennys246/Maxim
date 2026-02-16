

from __future__ import annotations

from typing import Any

from maxim.tools.base import ToolOutput
from maxim.tools.registry import ToolRegistry


class Executor:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.registry = tool_registry

    def execute(self, action: dict[str, Any]) -> ToolOutput:
        """Execute a tool action, returning raw ToolOutput.

        The caller (agent loop) is responsible for converting this to a
        bus ToolResult (agents.bus.ToolResult) with tool_call_id/tool_name/params
        before publishing on the bus.
        """
        tool_name = action.get("tool_name")
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        if not isinstance(tool_name, str) or not tool_name:
            return ToolOutput(success=False, error=f"Invalid action: {action!r}")

        try:
            tool = self.registry.get(tool_name)
        except KeyError:
            return ToolOutput(success=False, error=f"Tool not registered: {tool_name!r}")

        try:
            return tool.run(**params)
        except Exception as e:
            return ToolOutput(success=False, error=f"Tool {tool_name!r} execution failed: {e}")
