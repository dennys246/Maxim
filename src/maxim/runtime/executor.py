

from __future__ import annotations

import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

from maxim.tools.base import ToolOutput
from maxim.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from maxim.bridges.tool_pain_bridge import ToolPainBridge
    from maxim.proprioception.pain import PainDetector


class Executor:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        pain_detector: "PainDetector | None" = None,
        tool_pain_bridge: "ToolPainBridge | None" = None,
    ) -> None:
        self.registry = tool_registry
        self._pain_detector = pain_detector
        self._tool_pain_bridge = tool_pain_bridge
        self._lock = threading.Lock()
        # (tool_name, start_time, invocation_id) or None
        self._running: tuple[str, float, str] | None = None

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

        invocation_id = str(uuid.uuid4())

        with self._lock:
            self._running = (tool_name, time.time(), invocation_id)

        if self._tool_pain_bridge is not None:
            self._tool_pain_bridge.record_tool_start(
                tool_name, invocation_id, context={"params": params}
            )

        try:
            tool = self.registry.get(tool_name)
        except KeyError:
            with self._lock:
                self._running = None
            result = ToolOutput(success=False, error=f"Tool not registered: {tool_name!r}")
            self._report_failure(tool_name, invocation_id, result, params)
            return result

        try:
            result = tool.run(**params)
        except Exception as e:
            with self._lock:
                self._running = None
            result = ToolOutput(success=False, error=f"Tool {tool_name!r} execution failed: {e}")
            self._report_failure(tool_name, invocation_id, result, params)
            return result

        with self._lock:
            self._running = None

        if result.success:
            if self._tool_pain_bridge is not None:
                self._tool_pain_bridge.record_tool_complete(
                    tool_name, invocation_id, success=True
                )
        else:
            self._report_failure(tool_name, invocation_id, result, params)

        return result

    def _report_failure(
        self,
        tool_name: str,
        invocation_id: str,
        result: ToolOutput,
        params: dict[str, Any],
    ) -> None:
        """Report a tool failure to pain detector and bridge."""
        if self._pain_detector is not None:
            from maxim.agents.bus import ToolErrorKind

            self._pain_detector.record_tool_error(
                tool_name=tool_name,
                error=result.error or "unknown",
                error_kind=result.error_kind or ToolErrorKind.EXTERNAL_FAILURE,
                context={
                    "params": params,
                    "invocation_id": invocation_id,
                    "metadata": result.metadata,
                },
            )

    def get_running_tool(self) -> tuple[str, float, str] | None:
        """Get the currently running tool info.

        Returns:
            Tuple of (tool_name, start_time, invocation_id) or None.
        """
        with self._lock:
            return self._running
