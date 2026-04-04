"""InstrumentedExecutor — wraps Executor to record all actions to a sink.

Captures every tool execution (success, failure, and autonomy rejections)
as ActionRecords in a RecordingSink. Transparently wraps an existing
Executor without changing its interface.
"""

from __future__ import annotations

import time
from typing import Any

from maxim.simulation.sinks import ActionRecord, ActionSink
from maxim.tools.base import ToolOutput


class InstrumentedExecutor:
    """Wraps an Executor to record all actions to an ActionSink.

    Drop-in replacement for Executor — same execute() interface.
    All calls are forwarded to the wrapped executor, and results
    are recorded in the sink.

    Example:
        sink = RecordingSink()
        instrumented = InstrumentedExecutor(real_executor, sink)
        result = instrumented.execute({"tool_name": "read_file", "params": {...}})
        # result is the real ToolOutput
        # sink.actions now contains the ActionRecord
    """

    def __init__(self, executor: Any, sink: ActionSink) -> None:
        self._executor = executor
        self._sink = sink

    def execute(self, action: dict[str, Any]) -> ToolOutput:
        """Execute a tool action and record the result."""
        tool_name = action.get("tool_name", "unknown")
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}

        result = self._executor.execute(action)

        # Detect FearAgent blocks from metadata
        metadata = getattr(result, "metadata", None) or {}
        is_blocked = metadata.get("fear_agent_blocked", False)

        self._sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name=tool_name,
                tool_args=params,
                result_success=result.success,
                result_output=result.output,
                result_error=result.error,
                blocked=is_blocked,
                block_reason=result.error if is_blocked else None,
            )
        )

        return result

    def record_block(
        self, tool_name: str, reason: str, params: dict[str, Any] | None = None
    ) -> None:
        """Record that an action was blocked (e.g., by FearAgent or autonomy)."""
        self._sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name=tool_name,
                tool_args=params or {},
                blocked=True,
                block_reason=reason,
            )
        )

    # Forward all other attributes to the wrapped executor
    def __getattr__(self, name: str) -> Any:
        return getattr(self._executor, name)
