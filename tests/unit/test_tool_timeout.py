"""Unit tests for Tool.timeout class variable (PERF-1 Tier 1)."""

from __future__ import annotations

from typing import Any

from maxim.tools.base import Tool


class _DummyTool(Tool):
    """Minimal Tool subclass using the default timeout."""

    name = "dummy"

    def execute(self, **kwargs: Any) -> Any:
        return "ok"


class _CustomTimeoutTool(Tool):
    """Tool subclass that overrides timeout."""

    name = "custom"
    timeout: float = 60.0

    def execute(self, **kwargs: Any) -> Any:
        return "ok"


class TestToolTimeout:
    """Test Tool.timeout class variable."""

    def test_default_timeout(self):
        """A plain Tool subclass inherits the 30.0s default."""
        tool = _DummyTool()
        assert tool.timeout == 30.0

    def test_override_timeout(self):
        """A subclass can override timeout to a custom value."""
        tool = _CustomTimeoutTool()
        assert tool.timeout == 60.0

    def test_execute_file_tool_timeout(self):
        """ExecuteFileTool declares a 10.0s timeout."""
        from maxim.tools.filesystem import ExecuteFileTool

        tool = ExecuteFileTool()
        assert tool.timeout == 10.0

    def test_bash_tool_timeout(self):
        """BashTool inherits the default 30.0s timeout."""
        from maxim.tools.filesystem import BashTool

        tool = BashTool()
        assert tool.timeout == 30.0
