"""ActionSink protocol and implementations for capturing tool outputs.

The ActionSink captures every tool execution (including FearAgent blocks)
for post-run validation in scenario testing.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ActionRecord:
    """Captured output action from the agent pipeline."""

    timestamp: float
    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    result_success: bool = False
    result_output: Any = None
    result_error: str | None = None
    blocked: bool = False
    block_reason: str | None = None


@runtime_checkable
class ActionSink(Protocol):
    """Captures tool outputs and motor commands."""

    def record(self, action: ActionRecord) -> None:
        """Record an executed action."""
        ...

    @property
    def actions(self) -> list[ActionRecord]:
        """All recorded actions in order."""
        ...


class RecordingSink:
    """Stores all actions for post-run validation."""

    def __init__(self) -> None:
        self._actions: list[ActionRecord] = []
        self._lock = threading.Lock()

    def record(self, action: ActionRecord) -> None:
        with self._lock:
            self._actions.append(action)

    @property
    def actions(self) -> list[ActionRecord]:
        with self._lock:
            return list(self._actions)

    def find_blocked(self, tool_pattern: str | None = None, reason_contains: str | None = None) -> list[ActionRecord]:
        """Find actions that were blocked by FearAgent."""
        import re

        results = []
        for action in self.actions:
            if not action.blocked:
                continue
            if tool_pattern and not re.search(tool_pattern, action.tool_name):
                continue
            if reason_contains and (
                action.block_reason is None or reason_contains.lower() not in action.block_reason.lower()
            ):
                continue
            results.append(action)
        return results

    def find_actions(self, tool: str | None = None, output_matches: str | None = None) -> list[ActionRecord]:
        """Find actions matching criteria."""
        import re

        results = []
        for action in self.actions:
            if action.blocked:
                continue
            if tool and action.tool_name != tool:
                continue
            if output_matches:
                output_str = str(action.result_output or "")
                if not re.search(output_matches, output_str, re.IGNORECASE):
                    continue
            results.append(action)
        return results
