"""Protocol management tools for the agentic runtime."""

from __future__ import annotations

from typing import Any

from maxim.tools.base import Tool, ToolOutput

__all__ = ["RunProtocolTool", "StopProtocolTool", "ListProtocolsTool"]


class RunProtocolTool(Tool):
    name = "run_protocol"
    description = (
        "Activate a named protocol. Protocols bundle skills with "
        "workspace constraints. Use list_protocols to see available options."
    )
    input_schema = {"name": str}

    def __init__(self, protocol_registry: Any) -> None:
        super().__init__()
        self._registry = protocol_registry

    def execute(self, **kwargs: Any) -> ToolOutput:
        name = kwargs["name"]
        result = self._registry.activate(name)
        success = "activated" in result.lower() or "already active" in result.lower()
        return ToolOutput(success=success, output=result)


class StopProtocolTool(Tool):
    name = "stop_protocol"
    description = "Deactivate a running protocol."
    input_schema = {"name": str}

    def __init__(self, protocol_registry: Any) -> None:
        super().__init__()
        self._registry = protocol_registry

    def execute(self, **kwargs: Any) -> ToolOutput:
        name = kwargs["name"]
        result = self._registry.deactivate(name)
        success = "deactivated" in result.lower() or "not active" in result.lower()
        return ToolOutput(success=success, output=result)


class ListProtocolsTool(Tool):
    name = "list_protocols"
    description = "List available and active protocols."

    def __init__(self, protocol_registry: Any) -> None:
        super().__init__()
        self._registry = protocol_registry

    def execute(self, **kwargs: Any) -> ToolOutput:
        available = self._registry.get_available()
        active = [p.name for p in self._registry.get_active()]
        lines = ["Available protocols:"]
        for name in available:
            status = " (ACTIVE)" if name in active else ""
            lines.append(f"  - {name}{status}")
        return ToolOutput(success=True, output="\n".join(lines))
