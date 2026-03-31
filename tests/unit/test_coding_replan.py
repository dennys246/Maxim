"""Unit tests for CodingReplanContext, ReplanContext coding integration, and _extract_coding_context."""

from __future__ import annotations

from maxim.agents.bus import FailureStrategy, SubGoal, SubGoalStatus
from maxim.planning.plan_document import (
    CodingReplanContext,
    Phase,
    PhaseStatus,
    ReplanContext,
)
from maxim.planning.plan_manager import _extract_coding_context


# ─────────────────────────────────────────────────────────────────────────────
# CodingReplanContext
# ─────────────────────────────────────────────────────────────────────────────


class TestCodingReplanContext:
    def test_creation_with_test_failures(self):
        failures = [{"command": "pytest", "returncode": 1, "error": "assert False"}]
        ctx = CodingReplanContext(
            test_failures=failures,
            build_errors=[],
            lint_errors=[],
            files_modified=["src/foo.py"],
        )
        assert ctx.test_failures == failures
        assert ctx.build_errors == []
        assert ctx.lint_errors == []
        assert ctx.files_modified == ["src/foo.py"]

    def test_empty_defaults(self):
        ctx = CodingReplanContext()
        assert ctx.test_failures == []
        assert ctx.build_errors == []
        assert ctx.lint_errors == []
        assert ctx.files_modified == []

    def test_to_dict_from_dict_roundtrip(self):
        ctx = CodingReplanContext(
            test_failures=[{"command": "pytest", "returncode": 1, "error": "fail"}],
            build_errors=[{"command": "make", "error": "missing dep"}],
            lint_errors=[{"command": "ruff", "error": "E501"}],
            files_modified=["a.py", "b.py"],
        )
        d = ctx.to_dict()
        restored = CodingReplanContext.from_dict(d)
        assert restored.test_failures == ctx.test_failures
        assert restored.build_errors == ctx.build_errors
        assert restored.lint_errors == ctx.lint_errors
        assert restored.files_modified == ctx.files_modified


# ─────────────────────────────────────────────────────────────────────────────
# ReplanContext with coding_context
# ─────────────────────────────────────────────────────────────────────────────


def _make_stub_phase(description: str = "stub phase") -> Phase:
    return Phase(
        id="phase-stub",
        description=description,
        status=PhaseStatus.FAILED,
        plan_id="plan-stub",
    )


class TestReplanContextCoding:
    def test_with_coding_context(self):
        coding = CodingReplanContext(
            test_failures=[{"command": "pytest", "returncode": 1, "error": "fail"}],
        )
        ctx = ReplanContext(
            failed_phase=_make_stub_phase(),
            failure_reason="tests failed",
            failure_type="tool_error",
            coding_context=coding,
        )
        assert ctx.coding_context is coding
        assert ctx.coding_context.test_failures[0]["command"] == "pytest"

    def test_without_coding_context(self):
        ctx = ReplanContext(
            failed_phase=_make_stub_phase(),
            failure_reason="timeout",
            failure_type="timeout",
        )
        assert ctx.coding_context is None

    def test_to_llm_prompt_section_includes_test_failures(self):
        coding = CodingReplanContext(
            test_failures=[
                {"command": "pytest tests/", "returncode": 1, "error": "AssertionError"},
            ],
        )
        ctx = ReplanContext(
            failed_phase=_make_stub_phase("run tests phase"),
            failure_reason="tests failed",
            failure_type="tool_error",
            coding_context=coding,
        )
        prompt = ctx.to_llm_prompt_section()
        assert "Test Failures" in prompt
        assert "pytest tests/" in prompt
        assert "AssertionError" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# _extract_coding_context
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractCodingContext:
    def test_no_coding_tools_returns_none(self):
        sg = SubGoal(
            id="sg-1",
            description="respond to user",
            tool_name="respond",
            status=SubGoalStatus.COMPLETED,
        )
        sg.result = {"tool_name": "respond", "output": "hello"}
        phase = Phase(
            id="phase-1",
            description="respond phase",
            status=PhaseStatus.FAILED,
            plan_id="plan-1",
            sub_goals=[sg],
        )
        result = _extract_coding_context(phase)
        assert result is None

    def test_failed_run_tests_populates_test_failures(self):
        sg = SubGoal(
            id="sg-1",
            description="run tests",
            tool_name="run_tests",
            status=SubGoalStatus.FAILED,
        )
        sg.result = {
            "tool_name": "run_tests",
            "error": "2 tests failed",
            "metadata": {
                "command": "pytest tests/unit/",
                "returncode": 1,
            },
        }
        phase = Phase(
            id="phase-1",
            description="testing phase",
            status=PhaseStatus.FAILED,
            plan_id="plan-1",
            sub_goals=[sg],
        )
        ctx = _extract_coding_context(phase)
        assert ctx is not None
        assert len(ctx.test_failures) == 1
        assert ctx.test_failures[0]["command"] == "pytest tests/unit/"
        assert ctx.test_failures[0]["returncode"] == 1
        assert ctx.test_failures[0]["error"] == "2 tests failed"
