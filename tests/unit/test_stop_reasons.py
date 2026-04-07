"""Unit tests for StopReason, ToolErrorKind enums and LoopTerminated dataclass."""

from __future__ import annotations

import time
from typing import Any

from maxim.agents.bus import LoopTerminated, StopReason, ToolErrorKind
from maxim.tools.base import Tool, ToolOutput


class TestStopReason:
    """Verify StopReason enum members and values."""

    def test_all_members_exist(self):
        expected = {
            "COMPLETED": "completed",
            "MODE_SHUTDOWN": "mode_shutdown",
            "MAX_TURNS": "max_turns",
            "TOKEN_BUDGET": "token_budget",
            "ENERGY_BUDGET": "energy_budget",
            "USER_INTERRUPT": "user_interrupt",
            "PLAN_FAILED": "plan_failed",
            "NO_INTENT": "no_intent",
            "LLM_TIMEOUT": "llm_timeout",
            "SAFETY_GATE": "safety_gate",
        }
        for name, value in expected.items():
            member = StopReason[name]
            assert member.value == value

    def test_member_count(self):
        assert len(StopReason) == 10


class TestToolErrorKind:
    """Verify ToolErrorKind enum members and values."""

    def test_all_members_exist(self):
        expected = {
            "FILE_NOT_FOUND": "file_not_found",
            "PERMISSION_DENIED": "permission_denied",
            "SYNTAX_ERROR": "syntax_error",
            "TIMEOUT": "timeout",
            "INVALID_INPUT": "invalid_input",
            "EXTERNAL_FAILURE": "external_failure",
            "VALIDATION": "validation",
        }
        for name, value in expected.items():
            member = ToolErrorKind[name]
            assert member.value == value

    def test_member_count(self):
        assert len(ToolErrorKind) == 7


class TestToolOutputErrorKind:
    """Verify error_kind field on ToolOutput."""

    def test_with_error_kind(self):
        out = ToolOutput(success=False, error="timed out", error_kind=ToolErrorKind.TIMEOUT)
        assert out.success is False
        assert out.error == "timed out"
        assert out.error_kind is ToolErrorKind.TIMEOUT

    def test_without_error_kind(self):
        out = ToolOutput(success=True)
        assert out.error_kind is None
        assert out.error is None


class TestLoopTerminated:
    """Verify LoopTerminated dataclass."""

    def test_creation_and_auto_timestamp(self):
        before = time.time()
        lt = LoopTerminated(
            run_id="run-1",
            stop_reason=StopReason.COMPLETED,
            steps_taken=5,
        )
        after = time.time()

        assert lt.run_id == "run-1"
        assert lt.stop_reason is StopReason.COMPLETED
        assert lt.steps_taken == 5
        assert before <= lt.timestamp <= after

    def test_explicit_timestamp(self):
        lt = LoopTerminated(
            run_id="run-2",
            stop_reason=StopReason.MAX_TURNS,
            steps_taken=10,
            timestamp=1000.0,
        )
        assert lt.timestamp == 1000.0


# ── Tool subclass that returns error_kind ────────────────────────────────


class _FailingTool(Tool):
    """Minimal tool that returns a ToolOutput with error_kind."""

    name = "failing"

    def execute(self, **kwargs: Any) -> Any:
        return ToolOutput(
            success=False,
            error="file missing",
            error_kind=ToolErrorKind.FILE_NOT_FOUND,
        )


class TestToolErrorKindPropagation:
    """Verify error_kind propagates through Tool.run()."""

    def test_error_kind_survives_run(self):
        tool = _FailingTool()
        result = tool.run()
        assert result.success is False
        assert result.error_kind is ToolErrorKind.FILE_NOT_FOUND
        assert result.error == "file missing"
