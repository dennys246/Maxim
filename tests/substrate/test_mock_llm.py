"""Tests for MockLLMBackend (S2 — simulator_upgrades_plan)."""

from __future__ import annotations

import pytest

from maxim.models.language.backend_protocol import LLMBackend
from maxim.models.language.types import LLMResponse
from tests.substrate.mock_llm import MockLLMBackend


# ── Protocol conformance ─────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_satisfies_protocol(self):
        """MockLLMBackend is a structural match for LLMBackend Protocol."""
        mock = MockLLMBackend(canned="hello")
        assert isinstance(mock, LLMBackend)

    def test_has_required_attributes(self):
        mock = MockLLMBackend()
        assert hasattr(mock, "requires_prompt_formatting")
        assert hasattr(mock, "supports_model_override")
        assert hasattr(mock, "supports_tool_use")
        assert hasattr(mock, "supports_streaming")

    def test_attribute_defaults(self):
        mock = MockLLMBackend()
        assert mock.requires_prompt_formatting is False
        assert mock.supports_model_override is False
        assert mock.supports_tool_use is True
        assert mock.supports_streaming is False


# ── Canned mode ──────────────────────────────────────────────────────────────


class TestCannedMode:
    def test_complete_returns_canned(self):
        mock = MockLLMBackend(canned="fixed response")
        result = mock.complete("any prompt", max_tokens=100, temperature=0.0, stop=())
        assert result == "fixed response"

    def test_complete_with_usage_returns_canned(self):
        mock = MockLLMBackend(canned="fixed response")
        resp = mock.complete_with_usage(system="sys", user="any", max_tokens=100, temperature=0.0)
        assert isinstance(resp, LLMResponse)
        assert resp.content == "fixed response"
        assert resp.model == "mock"
        assert resp.provider == "mock"

    def test_canned_ignores_prompt_content(self):
        mock = MockLLMBackend(canned="always this")
        assert mock.complete("hello", max_tokens=10, temperature=0.0, stop=()) == "always this"
        assert mock.complete("goodbye", max_tokens=10, temperature=0.0, stop=()) == "always this"
        assert mock.complete("anything", max_tokens=10, temperature=0.0, stop=()) == "always this"


# ── Policy mode ──────────────────────────────────────────────────────────────


class TestPolicyMode:
    def test_matches_substring(self):
        mock = MockLLMBackend(
            policy={
                "delete": "I cannot delete files.",
                "hello": "Hi there!",
            }
        )
        assert (
            mock.complete("Please delete my files", max_tokens=100, temperature=0.0, stop=())
            == "I cannot delete files."
        )
        assert mock.complete("hello world", max_tokens=100, temperature=0.0, stop=()) == "Hi there!"

    def test_case_insensitive(self):
        mock = MockLLMBackend(policy={"DELETE": "blocked"})
        assert mock.complete("delete stuff", max_tokens=100, temperature=0.0, stop=()) == "blocked"

    def test_regex_pattern(self):
        mock = MockLLMBackend(policy={r"file\s*system": "filesystem matched"})
        assert mock.complete("access the file system", max_tokens=100, temperature=0.0, stop=()) == "filesystem matched"

    def test_falls_back_to_default(self):
        mock = MockLLMBackend(
            policy={"specific": "matched"},
            default_response="fallback",
        )
        assert mock.complete("unrelated prompt", max_tokens=100, temperature=0.0, stop=()) == "fallback"

    def test_first_match_wins(self):
        mock = MockLLMBackend(
            policy={
                "hello": "first",
                "hello world": "second",
            }
        )
        # "hello" matches first
        assert mock.complete("hello world", max_tokens=100, temperature=0.0, stop=()) == "first"


# ── Scripted mode ────────────────────────────────────────────────────────────


class TestScriptedMode:
    def test_returns_in_order(self):
        mock = MockLLMBackend(scripted=["first", "second", "third"])
        assert mock.complete("a", max_tokens=100, temperature=0.0, stop=()) == "first"
        assert mock.complete("b", max_tokens=100, temperature=0.0, stop=()) == "second"
        assert mock.complete("c", max_tokens=100, temperature=0.0, stop=()) == "third"

    def test_raises_when_exhausted(self):
        mock = MockLLMBackend(scripted=["only one"])
        mock.complete("first call", max_tokens=100, temperature=0.0, stop=())
        with pytest.raises(IndexError, match="scripted mode exhausted"):
            mock.complete("second call", max_tokens=100, temperature=0.0, stop=())

    def test_reset_resets_index(self):
        mock = MockLLMBackend(scripted=["a", "b"])
        mock.complete("x", max_tokens=100, temperature=0.0, stop=())
        mock.reset()
        assert mock.complete("y", max_tokens=100, temperature=0.0, stop=()) == "a"


# ── complete_with_usage details ──────────────────────────────────────────────


class TestCompleteWithUsage:
    def test_returns_llm_response(self):
        mock = MockLLMBackend(canned="response text")
        resp = mock.complete_with_usage(system="sys", user="user", max_tokens=100, temperature=0.0)
        assert isinstance(resp, LLMResponse)
        assert resp.content == "response text"
        assert resp.stop_reason == "end_turn"
        assert resp.input_tokens > 0
        assert resp.output_tokens > 0

    def test_tool_call_parsing(self):
        tool_json = '{"tool_name": "bash", "params": {"command": "ls"}}'
        mock = MockLLMBackend(canned=tool_json)
        resp = mock.complete_with_usage(
            system="sys",
            user="run ls",
            max_tokens=100,
            temperature=0.0,
            tools=[{"name": "bash", "description": "run shell commands"}],
        )
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["tool_name"] == "bash"

    def test_no_tool_calls_without_tools_param(self):
        tool_json = '{"tool_name": "bash", "params": {"command": "ls"}}'
        mock = MockLLMBackend(canned=tool_json)
        resp = mock.complete_with_usage(system="sys", user="run ls", max_tokens=100, temperature=0.0)
        # No tools passed → no tool_call parsing
        assert resp.tool_calls is None

    def test_accepts_optional_kwargs(self):
        mock = MockLLMBackend(canned="ok")
        resp = mock.complete_with_usage(
            system="sys",
            user="user",
            max_tokens=100,
            temperature=0.0,
            model_override="claude-sonnet",
            tools=[{"name": "test"}],
            thinking={"type": "enabled", "budget_tokens": 1000},
            stream=False,
        )
        assert resp.content == "ok"


# ── Call history ─────────────────────────────────────────────────────────────


class TestCallHistory:
    def test_complete_records_history(self):
        mock = MockLLMBackend(canned="x")
        mock.complete("prompt text", max_tokens=50, temperature=0.5, stop=())
        assert len(mock.call_history) == 1
        assert mock.call_history[0]["method"] == "complete"
        assert mock.call_history[0]["prompt"] == "prompt text"

    def test_complete_with_usage_records_history(self):
        mock = MockLLMBackend(canned="x")
        mock.complete_with_usage(system="s", user="u", max_tokens=50, temperature=0.1)
        assert len(mock.call_history) == 1
        assert mock.call_history[0]["method"] == "complete_with_usage"
        assert mock.call_history[0]["system"] == "s"

    def test_reset_clears_history(self):
        mock = MockLLMBackend(canned="x")
        mock.complete("a", max_tokens=10, temperature=0.0, stop=())
        mock.complete("b", max_tokens=10, temperature=0.0, stop=())
        assert len(mock.call_history) == 2
        mock.reset()
        assert len(mock.call_history) == 0


# ── Warmup ───────────────────────────────────────────────────────────────────


class TestWarmup:
    def test_warmup_returns_true(self):
        mock = MockLLMBackend()
        assert mock.warmup() is True


# ── Construction validation ──────────────────────────────────────────────────


class TestConstruction:
    def test_default_construction(self):
        mock = MockLLMBackend()
        result = mock.complete("test", max_tokens=100, temperature=0.0, stop=())
        assert "Mock response" in result

    def test_rejects_multiple_modes(self):
        with pytest.raises(ValueError, match="at most one"):
            MockLLMBackend(canned="a", policy={"b": "c"})

        with pytest.raises(ValueError, match="at most one"):
            MockLLMBackend(canned="a", scripted=["b"])

        with pytest.raises(ValueError, match="at most one"):
            MockLLMBackend(policy={"a": "b"}, scripted=["c"])
