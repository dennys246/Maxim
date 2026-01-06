

from __future__ import annotations

from typing import Any

from maxim.tools.base import ToolResult
from maxim.tools.registry import ToolRegistry


class Executor:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.registry = tool_registry

    def execute(self, action: dict[str, Any]) -> ToolResult:
        tool_name = action.get("tool_name")
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        if not isinstance(tool_name, str) or not tool_name:
            return ToolResult(success=False, error=f"Invalid action: {action!r}")

        tool = self.registry.get(tool_name)
        return tool.run(**params)
