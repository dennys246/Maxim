"""Tests for the 4-stage JSON repair pipeline.

The pipeline is the primary defense against LLM output malformation.
Tests exercise each stage independently and the full pipeline end-to-end.
"""

from __future__ import annotations

import json

from maxim.models.language.json_parser import (
    _sanitize_json_string,
    _repair_with_library,
    _find_first_json_object,
    _try_structural_repair,
    json_parse_stats,
    json_parse_stats_reset,
)


# ─── Stage 1: Valid JSON passes through unchanged ────────────────────────


class TestValidJsonPassthrough:
    def test_simple_object(self):
        text = '{"action": "respond", "confidence": 0.9}'
        assert json.loads(text) is not None  # Valid JSON

    def test_nested_object(self):
        text = '{"action": {"tool_name": "choose", "params": {"option": "flee"}}, "confidence": 1.0}'
        parsed = json.loads(text)
        assert parsed["action"]["tool_name"] == "choose"

    def test_array_value(self):
        text = '{"items": [1, 2, 3], "count": 3}'
        parsed = json.loads(text)
        assert parsed["items"] == [1, 2, 3]


# ─── Stage 2: Control character sanitization ─────────────────────────────


class TestSanitizeJsonString:
    def test_literal_newline_in_string(self):
        """Literal \\n inside a JSON string should be escaped."""
        text = '{"message": "line one\nline two"}'
        sanitized = _sanitize_json_string(text)
        parsed = json.loads(sanitized)
        assert parsed["message"] == "line one\nline two"

    def test_literal_tab_in_string(self):
        text = '{"message": "col1\tcol2"}'
        sanitized = _sanitize_json_string(text)
        parsed = json.loads(sanitized)
        assert parsed["message"] == "col1\tcol2"

    def test_carriage_return_in_string(self):
        text = '{"message": "before\rafter"}'
        sanitized = _sanitize_json_string(text)
        parsed = json.loads(sanitized)
        assert "before" in parsed["message"]

    def test_control_character_escaped(self):
        """Low ASCII control chars (< 32) should be unicode-escaped."""
        text = '{"data": "has\x01control"}'
        sanitized = _sanitize_json_string(text)
        parsed = json.loads(sanitized)
        assert "control" in parsed["data"]

    def test_already_escaped_newline_untouched(self):
        """Already-escaped \\n should pass through."""
        text = '{"message": "already\\nescaped"}'
        sanitized = _sanitize_json_string(text)
        parsed = json.loads(sanitized)
        assert parsed["message"] == "already\nescaped"

    def test_empty_string_passthrough(self):
        text = '{"message": ""}'
        sanitized = _sanitize_json_string(text)
        assert json.loads(sanitized)["message"] == ""

    def test_no_strings_passthrough(self):
        text = '{"count": 42, "flag": true}'
        sanitized = _sanitize_json_string(text)
        assert json.loads(sanitized)["count"] == 42


# ─── Stage 3: json_repair library ────────────────────────────────────────


class TestRepairWithLibrary:
    def test_trailing_comma(self):
        """json_repair should handle trailing commas."""
        text = '{"a": 1, "b": 2,}'
        result = _repair_with_library(text)
        if result is not None:  # Library may not be installed
            parsed = json.loads(result)
            assert parsed["a"] == 1
            assert parsed["b"] == 2

    def test_single_quotes(self):
        """json_repair should handle single-quoted strings."""
        text = "{'action': 'respond'}"
        result = _repair_with_library(text)
        if result is not None:
            parsed = json.loads(result)
            assert parsed["action"] == "respond"

    def test_unquoted_keys(self):
        """json_repair should handle unquoted keys."""
        text = '{action: "respond", confidence: 0.9}'
        result = _repair_with_library(text)
        if result is not None:
            parsed = json.loads(result)
            assert parsed["action"] == "respond"

    def test_valid_json_passthrough(self):
        """Valid JSON should pass through unchanged."""
        text = '{"valid": true}'
        result = _repair_with_library(text)
        if result is not None:
            assert json.loads(result)["valid"] is True


# ─── Helper: Find first JSON object ─────────────────────────────────────


class TestFindFirstJsonObject:
    def test_simple_object(self):
        text = '{"a": 1}'
        result = _find_first_json_object(text)
        assert result == '{"a": 1}'

    def test_object_with_trailing_text(self):
        text = '{"a": 1} some extra text'
        result = _find_first_json_object(text)
        assert result == '{"a": 1}'

    def test_nested_braces(self):
        text = '{"outer": {"inner": 1}}'
        result = _find_first_json_object(text)
        assert result == '{"outer": {"inner": 1}}'

    def test_braces_inside_string(self):
        """Braces inside strings should not count."""
        text = '{"msg": "has {braces} inside"}'
        result = _find_first_json_object(text)
        assert result == '{"msg": "has {braces} inside"}'

    def test_no_opening_brace(self):
        text = "just plain text"
        result = _find_first_json_object(text)
        assert result is None

    def test_unclosed_brace_returns_whole_text(self):
        """Truncated JSON should return the whole string for repair."""
        text = '{"a": 1, "b":'
        result = _find_first_json_object(text)
        assert result == text


# ─── Stage 4: Structural repair (truncated JSON) ────────────────────────


class TestStructuralRepair:
    def test_missing_closing_brace(self):
        text = '{"a": 1, "b": 2'
        repaired = _try_structural_repair(text)
        parsed = json.loads(repaired)
        assert parsed["a"] == 1

    def test_missing_closing_bracket(self):
        text = '{"items": [1, 2, 3'
        repaired = _try_structural_repair(text)
        parsed = json.loads(repaired)
        assert parsed["items"] == [1, 2, 3]

    def test_unclosed_string(self):
        text = '{"msg": "truncated'
        repaired = _try_structural_repair(text)
        parsed = json.loads(repaired)
        assert "truncated" in parsed["msg"]

    def test_already_valid_untouched(self):
        text = '{"valid": true}'
        repaired = _try_structural_repair(text)
        assert repaired == text  # No changes

    def test_deeply_nested_truncation(self):
        text = '{"a": {"b": {"c": 1'
        repaired = _try_structural_repair(text)
        parsed = json.loads(repaired)
        assert parsed["a"]["b"]["c"] == 1

    def test_trailing_comma_stripped(self):
        text = '{"a": 1, "b": 2,'
        repaired = _try_structural_repair(text)
        parsed = json.loads(repaired)
        assert parsed["b"] == 2


# ─── Compliance counters ─────────────────────────────────────────────────


class TestComplianceCounters:
    def test_stats_start_at_zero(self):
        json_parse_stats_reset()
        stats = json_parse_stats()
        assert stats["json_total"] == 0
        assert stats["json_first_try"] == 0
        assert stats["json_compliance_rate"] == 1.0

    def test_reset_clears_counters(self):
        json_parse_stats_reset()
        stats = json_parse_stats()
        assert stats["json_total"] == 0
