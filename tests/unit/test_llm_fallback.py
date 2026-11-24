"""Comprehensive tests for maxim.agents.llm_fallback module."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from maxim.agents.llm_fallback import (
    FallbackBehavior,
    ReasoningCarryover,
    ReasoningEntry,
    evaluate_simple_arithmetic,
    evaluate_unary_math,
)
from maxim.agents.llm_types import LLMProposal


@pytest.fixture(autouse=True)
def _mock_fallback_responses(monkeypatch):
    """Mock get_fallback_responses to return an empty dict by default."""
    monkeypatch.setattr("maxim.agents.llm_fallback.get_fallback_responses", lambda: {})


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_simple_arithmetic
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluateSimpleArithmetic:
    def test_addition(self):
        assert evaluate_simple_arithmetic("2 + 3") == "2 + 3 = 5"

    def test_subtraction(self):
        assert evaluate_simple_arithmetic("10 - 4") == "10 - 4 = 6"

    def test_multiplication(self):
        assert evaluate_simple_arithmetic("3 * 7") == "3 * 7 = 21"

    def test_x_as_multiply(self):
        result = evaluate_simple_arithmetic("3 x 7")
        assert result == "3 * 7 = 21"

    def test_division(self):
        assert evaluate_simple_arithmetic("10 / 2") == "10 / 2 = 5"

    def test_exponent(self):
        assert evaluate_simple_arithmetic("2 ^ 10") == "2 ^ 10 = 1024"

    def test_modulo(self):
        assert evaluate_simple_arithmetic("10 % 3") == "10 % 3 = 1"

    def test_division_by_zero(self):
        result = evaluate_simple_arithmetic("5 / 0")
        assert result == "That operation is undefined (division by zero)."

    def test_modulo_by_zero(self):
        result = evaluate_simple_arithmetic("5 % 0")
        assert result == "That operation is undefined (division by zero)."

    def test_chained_expression_rejected(self):
        assert evaluate_simple_arithmetic("1 + 2 + 3") is None

    def test_negative_numbers(self):
        result = evaluate_simple_arithmetic("-3 + 5")
        assert result == "-3 + 5 = 2"

    def test_decimals(self):
        result = evaluate_simple_arithmetic("1.5 + 2.5")
        assert result == "1.5 + 2.5 = 4"

    def test_int_result_no_decimal(self):
        result = evaluate_simple_arithmetic("4 + 6")
        assert result == "4 + 6 = 10"
        assert ".0" not in result

    def test_decimal_result_shown(self):
        result = evaluate_simple_arithmetic("7 / 2")
        assert result == "7 / 2 = 3.5"

    def test_embedded_in_text(self):
        result = evaluate_simple_arithmetic("what is 5 + 3")
        assert result == "5 + 3 = 8"

    def test_no_spaces(self):
        result = evaluate_simple_arithmetic("2+3")
        assert result == "2 + 3 = 5"

    def test_no_match(self):
        assert evaluate_simple_arithmetic("hello world") is None

    def test_empty_string(self):
        assert evaluate_simple_arithmetic("") is None


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_unary_math
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluateUnaryMath:
    def test_square_root(self):
        assert evaluate_unary_math("square root of 25") == "√25 = 5"

    def test_sqrt_keyword(self):
        assert evaluate_unary_math("sqrt 16") == "√16 = 4"

    def test_cube_root(self):
        assert evaluate_unary_math("cube root of 8") == "∛8 = 2"

    def test_squared(self):
        assert evaluate_unary_math("5 squared") == "5² = 25"

    def test_cubed(self):
        assert evaluate_unary_math("3 cubed") == "3³ = 27"

    def test_negative_sqrt(self):
        result = evaluate_unary_math("square root of -4")
        assert result == "The square root of a negative number is not a real number."

    def test_sqrt_with_trailing_plus(self):
        result = evaluate_unary_math("square root of 25 plus 3")
        assert result == "√25 + 3 = 8"

    def test_squared_with_trailing_minus(self):
        result = evaluate_unary_math("5 squared minus 10")
        assert result == "5² - 10 = 15"

    def test_cubed_with_trailing_times(self):
        result = evaluate_unary_math("2 cubed times 3")
        assert result == "2³ * 3 = 24"

    def test_sqrt_with_trailing_divided_by(self):
        result = evaluate_unary_math("square root of 100 divided by 2")
        assert result == "√100 / 2 = 5"

    def test_trailing_division_by_zero(self):
        result = evaluate_unary_math("square root of 9 divided by 0")
        assert result == "That operation is undefined (division by zero)."

    def test_decimal_input(self):
        result = evaluate_unary_math("sqrt 2.25")
        assert result == "√2.25 = 1.5"

    def test_no_match(self):
        assert evaluate_unary_math("hello world") is None

    def test_empty_string(self):
        assert evaluate_unary_math("") is None


# ─────────────────────────────────────────────────────────────────────────────
# ReasoningEntry
# ─────────────────────────────────────────────────────────────────────────────


class TestReasoningEntry:
    def test_to_prompt_line_success(self):
        entry = ReasoningEntry(
            action_tool="search",
            reasoning="Looking for files",
            success=True,
            result_summary="Found 3 files",
        )
        line = entry.to_prompt_line()
        assert line == "- search: Looking for files [OK: Found 3 files]"

    def test_to_prompt_line_failure(self):
        entry = ReasoningEntry(
            action_tool="api_call",
            reasoning="Fetching data",
            success=False,
            result_summary="Timeout",
        )
        line = entry.to_prompt_line()
        assert line == "- api_call: Fetching data [FAIL: Timeout]"

    def test_to_prompt_line_truncates_long_reasoning(self):
        entry = ReasoningEntry(
            action_tool="tool",
            reasoning="A" * 100,
            success=True,
            result_summary="done",
        )
        line = entry.to_prompt_line()
        # reasoning truncated to 80 chars
        assert "A" * 80 in line
        assert "A" * 81 not in line

    def test_to_prompt_line_truncates_long_summary(self):
        entry = ReasoningEntry(
            action_tool="tool",
            reasoning="reason",
            success=True,
            result_summary="B" * 150,
        )
        line = entry.to_prompt_line()
        assert "B" * 100 in line
        assert "B" * 101 not in line

    def test_to_prompt_line_empty_fields(self):
        entry = ReasoningEntry(
            action_tool="t",
            reasoning="",
            success=False,
            result_summary="",
        )
        line = entry.to_prompt_line()
        assert line == "- t:  [FAIL: ]"

    def test_default_timestamp(self):
        before = time.time()
        entry = ReasoningEntry(action_tool="t", reasoning="r", success=True, result_summary="s")
        after = time.time()
        assert before <= entry.timestamp <= after


# ─────────────────────────────────────────────────────────────────────────────
# ReasoningCarryover
# ─────────────────────────────────────────────────────────────────────────────


class TestReasoningCarryover:
    def test_record_and_len(self):
        rc = ReasoningCarryover(max_entries=5)
        assert len(rc) == 0
        rc.record("tool1", "reason1", True, "ok")
        assert len(rc) == 1

    def test_max_eviction(self):
        rc = ReasoningCarryover(max_entries=3)
        for i in range(5):
            rc.record(f"tool{i}", f"reason{i}", True, f"result{i}")
        assert len(rc) == 3

    def test_eviction_keeps_most_recent(self):
        rc = ReasoningCarryover(max_entries=2)
        rc.record("old", "old_reason", True, "old_result")
        rc.record("mid", "mid_reason", True, "mid_result")
        rc.record("new", "new_reason", True, "new_result")
        text = rc.get_prompt_body()
        assert "old" not in text
        assert "mid" in text
        assert "new" in text

    def test_get_prompt_body_empty(self):
        rc = ReasoningCarryover()
        assert rc.get_prompt_body() == ""

    def test_get_prompt_body_format(self):
        rc = ReasoningCarryover()
        rc.record("search", "Looking", True, "Found items")
        text = rc.get_prompt_body()
        assert rc.PROMPT_HEADER == "=== Recent Decisions (Working Memory) ==="
        assert "- search: Looking [OK: Found items]" in text

    def test_clear(self):
        rc = ReasoningCarryover()
        rc.record("tool", "r", True, "s")
        assert len(rc) == 1
        rc.clear()
        assert len(rc) == 0
        assert rc.get_prompt_body() == ""

    def test_thread_safety(self):
        rc = ReasoningCarryover(max_entries=100)
        errors = []

        def writer(thread_id):
            try:
                for i in range(50):
                    rc.record(f"t{thread_id}", f"r{i}", True, f"s{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(rc) == 100  # 4 threads * 50 = 200, capped to 100


# ─────────────────────────────────────────────────────────────────────────────
# FallbackBehavior
# ─────────────────────────────────────────────────────────────────────────────


class TestFallbackBehavior:
    def test_get_fallback_action_none_percept(self):
        result = FallbackBehavior.get_fallback_action("observe", None)
        assert result is None

    def test_get_fallback_action_hard_override(self):
        percept = MagicMock()
        percept.hard_override = "shutdown"
        result = FallbackBehavior.get_fallback_action("observe", percept)
        assert result == {
            "tool_name": "maxim_command",
            "params": {"command": "shutdown"},
        }

    def test_get_fallback_action_observe_mode(self):
        percept = MagicMock()
        percept.hard_override = None
        result = FallbackBehavior.get_fallback_action("observe", percept)
        assert result == {"tool_name": "focus_interests", "params": {}}

    def test_get_fallback_action_no_maxim_keyword(self):
        percept = MagicMock()
        percept.hard_override = None
        percept.has_maxim_keyword = False
        result = FallbackBehavior.get_fallback_action("active_assistance", percept)
        assert result is None

    def test_get_fallback_action_reflection_mode(self):
        percept = MagicMock()
        percept.hard_override = None
        percept.has_maxim_keyword = True
        result = FallbackBehavior.get_fallback_action("reflection", percept)
        assert result is not None
        assert result["tool_name"] == "speak"
        assert "reflection" in result["params"]["text"].lower()

    def test_get_fallback_action_active_assistance_mode(self):
        percept = MagicMock()
        percept.hard_override = None
        percept.has_maxim_keyword = True
        result = FallbackBehavior.get_fallback_action("active_assistance", percept)
        assert result is not None
        assert result["tool_name"] == "speak"
        assert "processing" in result["params"]["text"].lower() or "delayed" in result["params"]["text"].lower()

    def test_get_fallback_action_unknown_mode_returns_none(self):
        percept = MagicMock()
        percept.hard_override = None
        percept.has_maxim_keyword = True
        result = FallbackBehavior.get_fallback_action("unknown_mode", percept)
        assert result is None

    def test_get_fallback_proposal_returns_llm_proposal(self):
        percept = MagicMock()
        percept.hard_override = None
        percept.has_maxim_keyword = True
        result = FallbackBehavior.get_fallback_proposal("reflection", percept)
        assert isinstance(result, LLMProposal)
        assert result.request_id.startswith("fallback-")
        assert result.reasoning == "LLM unavailable, using fallback behavior"
        assert result.strategy_used == "fallback"
        assert result.error == "LLM unavailable"
        assert result.mode_goal_achieved is False
        assert result.action is not None
        assert result.confidence == 0.5

    def test_get_fallback_proposal_no_action(self):
        result = FallbackBehavior.get_fallback_proposal("some_mode", None)
        assert isinstance(result, LLMProposal)
        assert result.action is None
        assert result.confidence == 0.0

    def test_get_fallback_action_internet_access_param(self):
        """Verify internet_access param is accepted (future use)."""
        percept = MagicMock()
        percept.hard_override = None
        result = FallbackBehavior.get_fallback_action("observe", percept, internet_access=True)
        assert result == {"tool_name": "focus_interests", "params": {}}

    def test_get_fallback_action_with_custom_llm_unavailable(self, monkeypatch):
        """Test that custom llm_unavailable messages from config are used."""
        monkeypatch.setattr(
            "maxim.agents.llm_fallback.get_fallback_responses",
            lambda: {
                "llm_unavailable": {
                    "reflection": "Custom reflection message",
                }
            },
        )
        percept = MagicMock()
        percept.hard_override = None
        percept.has_maxim_keyword = True
        result = FallbackBehavior.get_fallback_action("reflection", percept)
        assert result == {
            "tool_name": "speak",
            "params": {"text": "Custom reflection message"},
        }
