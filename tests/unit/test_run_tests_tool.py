"""Tests for RunTestsTool (Phase 3 #11c)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from maxim.agents.bus import ToolErrorKind
from maxim.tools.code_tools import RunTestsTool


class TestRunTestsToolSuccess:
    """Successful test run returns success=True with stdout."""

    def test_success(self) -> None:
        tool = RunTestsTool()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "5 passed in 1.2s"
        mock_result.stderr = ""

        with patch("maxim.tools.code_tools.subprocess.run", return_value=mock_result):
            output = tool.run(command="python -m pytest", test_path="tests/")

        assert output.success is True
        assert "5 passed" in output.output


class TestRunTestsToolFailure:
    """Failed test run returns success=False with EXTERNAL_FAILURE."""

    def test_failure(self) -> None:
        tool = RunTestsTool()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "FAILED test_foo.py"
        mock_result.stderr = "1 failed"

        with patch("maxim.tools.code_tools.subprocess.run", return_value=mock_result):
            output = tool.run(command="python -m pytest")

        assert output.success is False
        assert output.error_kind == ToolErrorKind.EXTERNAL_FAILURE


class TestRunTestsToolTimeout:
    """TimeoutExpired should map to TIMEOUT error_kind."""

    def test_timeout(self) -> None:
        tool = RunTestsTool()

        with patch(
            "maxim.tools.code_tools.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=120),
        ):
            output = tool.run(command="python -m pytest")

        assert output.success is False
        assert output.error_kind == ToolErrorKind.TIMEOUT


class TestRunTestsToolCommandNotFound:
    """FileNotFoundError should map to FILE_NOT_FOUND."""

    def test_command_not_found(self) -> None:
        tool = RunTestsTool()

        with patch(
            "maxim.tools.code_tools.subprocess.run",
            side_effect=FileNotFoundError("No such file"),
        ):
            output = tool.run(command="nonexistent_runner")

        assert output.success is False
        assert output.error_kind == ToolErrorKind.FILE_NOT_FOUND


class TestRunTestsToolTimeoutClassVar:
    """RunTestsTool.timeout should be 300.0 (5 minutes)."""

    def test_timeout_value(self) -> None:
        assert RunTestsTool.timeout == 300.0
