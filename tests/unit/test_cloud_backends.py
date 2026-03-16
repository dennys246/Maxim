"""Tests for cloud backend Phase 3 features: prompt caching, tool use, thinking, streaming."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock


from maxim.models.language.router import LLMConfig


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_cfg(**overrides: Any) -> LLMConfig:
    defaults: dict[str, Any] = {
        "enabled": True,
        "cloud_enabled": True,
        "backend": "anthropic",
        "providers": {
            "anthropic": {
                "type": "anthropic",
                "api_key_env": "ANTHROPIC_API_KEY",
                "model": "claude-sonnet-4-5-20250514",
                "prompt_cache": {"enabled": True},
                "thinking": {"enabled": True, "budget_tokens": 5000},
            }
        },
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic Backend Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAnthropicBackend:
    """Tests for _AnthropicBackend Phase 3 features."""

    def _make_backend(self, **provider_overrides: Any):
        from maxim.models.language.anthropic_backend import _AnthropicBackend

        providers: dict[str, Any] = {
            "type": "anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
            "model": "claude-sonnet-4-5-20250514",
        }
        providers.update(provider_overrides)
        cfg = _make_cfg(providers={"anthropic": providers})
        return _AnthropicBackend(cfg, provider_key="anthropic")

    # ── Prompt caching ──────────────────────────────────────────────────

    def test_system_blocks_no_cache(self):
        backend = self._make_backend(prompt_cache=None)
        blocks = backend._build_system_blocks("You are a helpful assistant.")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "cache_control" not in blocks[0]

    def test_system_blocks_with_cache(self):
        backend = self._make_backend(prompt_cache={"enabled": True})
        blocks = backend._build_system_blocks("You are a helpful assistant.")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_system_blocks_empty(self):
        backend = self._make_backend(prompt_cache={"enabled": True})
        blocks = backend._build_system_blocks("")
        assert blocks == []

    # ── Thinking config ─────────────────────────────────────────────────

    def test_thinking_config_disabled(self):
        backend = self._make_backend(thinking={"enabled": False, "budget_tokens": 5000})
        assert backend._thinking_config() is None

    def test_thinking_config_enabled(self):
        backend = self._make_backend(thinking={"enabled": True, "budget_tokens": 8000})
        cfg = backend._thinking_config()
        assert cfg is not None
        assert cfg["budget_tokens"] == 8000

    def test_thinking_config_clamp_max(self):
        backend = self._make_backend(thinking={"enabled": True, "budget_tokens": 99999})
        cfg = backend._thinking_config()
        assert cfg["budget_tokens"] == 20000

    def test_thinking_config_clamp_min(self):
        backend = self._make_backend(thinking={"enabled": True, "budget_tokens": 100})
        cfg = backend._thinking_config()
        assert cfg["budget_tokens"] == 1024

    def test_thinking_config_missing(self):
        backend = self._make_backend()
        # No thinking key in provider config
        assert backend._thinking_config() is None

    # ── Tool use ────────────────────────────────────────────────────────

    def test_proposed_goal_tool_schema(self):
        from maxim.models.language.anthropic_backend import PROPOSED_GOAL_TOOL

        assert PROPOSED_GOAL_TOOL["name"] == "propose_goal"
        schema = PROPOSED_GOAL_TOOL["input_schema"]
        assert "goal_description" in schema["properties"]
        assert "priority" in schema["properties"]
        assert "tool_name" in schema["properties"]
        assert "sub_goals" in schema["properties"]
        assert set(schema["required"]) == {"goal_description", "priority"}

    # ── Response parsing ────────────────────────────────────────────────

    def test_parse_text_response(self):
        backend = self._make_backend()

        mock_resp = MagicMock()
        mock_resp.usage.input_tokens = 100
        mock_resp.usage.output_tokens = 50
        mock_resp.usage.cache_read_input_tokens = 30
        mock_resp.stop_reason = "end_turn"

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '{"goal_description": "test"}'
        mock_resp.content = [text_block]

        result = backend._parse_response(mock_resp, "claude-sonnet-4-5-20250514", 0.0)
        assert result.content == '{"goal_description": "test"}'
        assert result.input_tokens == 100
        assert result.cached_input_tokens == 30
        assert result.uncached_input_tokens == 70

    def test_parse_tool_use_response(self):
        backend = self._make_backend()

        mock_resp = MagicMock()
        mock_resp.usage.input_tokens = 200
        mock_resp.usage.output_tokens = 80
        mock_resp.usage.cache_read_input_tokens = 0
        mock_resp.stop_reason = "tool_use"

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_123"
        tool_block.name = "propose_goal"
        tool_block.input = {"goal_description": "test", "priority": "HIGH"}
        mock_resp.content = [tool_block]

        result = backend._parse_response(mock_resp, "claude-sonnet-4-5-20250514", 0.0)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "propose_goal"
        assert result.tool_calls[0]["input"]["priority"] == "HIGH"
        # Content should be JSON-serialized tool input for compatibility
        parsed = json.loads(result.content)
        assert parsed["goal_description"] == "test"

    def test_parse_thinking_response(self):
        backend = self._make_backend()

        mock_resp = MagicMock()
        mock_resp.usage.input_tokens = 150
        mock_resp.usage.output_tokens = 200
        mock_resp.usage.cache_read_input_tokens = 50
        mock_resp.stop_reason = "end_turn"

        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Let me reason about this..."

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '{"goal_description": "reasoned goal"}'
        mock_resp.content = [thinking_block, text_block]

        result = backend._parse_response(mock_resp, "claude-sonnet-4-5-20250514", 0.0)
        assert result.content == '{"goal_description": "reasoned goal"}'
        # Thinking text is captured but not included in content
        assert "Let me reason" not in result.content

    # ── Capability flags ────────────────────────────────────────────────

    def test_capability_flags(self):
        backend = self._make_backend()
        assert backend.supports_tool_use is True
        assert backend.supports_streaming is True
        assert backend.requires_prompt_formatting is False


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Backend Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenAIBackend:
    """Tests for _OpenAIBackend streaming support."""

    def _make_backend(self):
        from maxim.models.language.openai_backend import _OpenAIBackend

        cfg = LLMConfig(
            enabled=True,
            providers={"openai": {"type": "openai", "api_key_env": "OPENAI_API_KEY", "model": "gpt-4o"}},
        )
        return _OpenAIBackend(cfg, provider_key="openai")

    def test_parse_response(self):
        backend = self._make_backend()

        mock_resp = MagicMock()
        choice = MagicMock()
        choice.message.content = '{"answer": "hello"}'
        choice.finish_reason = "stop"
        mock_resp.choices = [choice]
        mock_resp.usage.prompt_tokens = 50
        mock_resp.usage.completion_tokens = 20
        mock_resp.usage.prompt_tokens_details = None

        result = backend._parse_response(mock_resp, "gpt-4o", 0.0)
        assert result.content == '{"answer": "hello"}'
        assert result.input_tokens == 50
        assert result.output_tokens == 20

    def test_streaming_flag(self):
        backend = self._make_backend()
        assert backend.supports_streaming is True


# ─────────────────────────────────────────────────────────────────────────────
# Router Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRouterPhase3:
    """Tests that router correctly passes Phase 3 kwargs to backends."""

    def test_generate_json_accepts_tools_and_thinking(self):
        """Verify generate_json signature accepts tools, thinking, stream."""
        import inspect
        from maxim.models.language.router import LLMRouter

        sig = inspect.signature(LLMRouter.generate_json)
        params = sig.parameters
        assert "tools" in params
        assert "thinking" in params
        assert "stream" in params

    def test_complete_text_accepts_tools_and_thinking(self):
        """Verify _complete_text signature accepts tools, thinking, stream."""
        import inspect
        from maxim.models.language.router import LLMRouter

        sig = inspect.signature(LLMRouter._complete_text)
        params = sig.parameters
        assert "tools" in params
        assert "thinking" in params
        assert "stream" in params


# ─────────────────────────────────────────────────────────────────────────────
# ExecAgent Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExecAgentPhase3:
    """Tests that ExecAgent correctly uses Phase 3 features."""

    def test_get_tool_definitions_returns_none_for_local(self):
        """Tool definitions should only be returned for cloud Anthropic providers."""
        from maxim.agents.exec_agent import ExecAgent

        mock_router = MagicMock()
        mock_router.cloud_allowed.return_value = True
        mock_router.preview_provider.return_value = {"is_cloud": False, "provider": "local"}
        mock_router.get_provider_configs.return_value = {"local": {"type": "llama_cpp"}}

        # Create a minimal ExecAgent to test the helper
        exec_agent = ExecAgent.__new__(ExecAgent)
        exec_agent._system_prompt = "test"

        result = exec_agent._get_tool_definitions(mock_router)
        assert result is None

    def test_get_tool_definitions_returns_schema_for_anthropic(self):
        """Tool definitions should return PROPOSED_GOAL_TOOL for Anthropic."""
        from maxim.agents.exec_agent import ExecAgent

        mock_router = MagicMock()
        mock_router.cloud_allowed.return_value = True
        mock_router.preview_provider.return_value = {"is_cloud": True, "provider": "anthropic"}
        mock_router.get_provider_configs.return_value = {
            "anthropic": {"type": "anthropic", "model": "claude-sonnet-4-5-20250514"}
        }

        exec_agent = ExecAgent.__new__(ExecAgent)
        exec_agent._system_prompt = "test"

        result = exec_agent._get_tool_definitions(mock_router)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "propose_goal"

    def test_get_thinking_config_returns_none_when_disabled(self):
        from maxim.agents.exec_agent import ExecAgent

        mock_router = MagicMock()
        mock_router.get_provider_configs.return_value = {
            "anthropic": {"type": "anthropic", "thinking": {"enabled": False}}
        }

        exec_agent = ExecAgent.__new__(ExecAgent)
        result = exec_agent._get_thinking_config(mock_router)
        assert result is None

    def test_get_thinking_config_returns_budget_when_enabled(self):
        from maxim.agents.exec_agent import ExecAgent

        mock_router = MagicMock()
        mock_router.get_provider_configs.return_value = {
            "anthropic": {
                "type": "anthropic",
                "thinking": {"enabled": True, "budget_tokens": 7000},
            }
        }

        exec_agent = ExecAgent.__new__(ExecAgent)
        result = exec_agent._get_thinking_config(mock_router)
        assert result is not None
        assert result["budget_tokens"] == 7000


# ─────────────────────────────────────────────────────────────────────────────
# LLMWorker Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMWorkerPhase3:
    """Tests that LLMWorker correctly passes Phase 3 kwargs."""

    def test_generate_json_direct_accepts_tools_and_thinking(self):
        """Verify generate_json_direct signature accepts tools, thinking, stream."""
        import inspect
        from maxim.agents.llm_worker import LLMWorker

        sig = inspect.signature(LLMWorker.generate_json_direct)
        params = sig.parameters
        assert "tools" in params
        assert "thinking" in params
        assert "stream" in params

    def test_call_llm_with_timeout_accepts_tools_and_thinking(self):
        """Verify _call_llm_with_timeout signature accepts tools, thinking, stream."""
        import inspect
        from maxim.agents.llm_worker import LLMWorker

        sig = inspect.signature(LLMWorker._call_llm_with_timeout)
        params = sig.parameters
        assert "tools" in params
        assert "thinking" in params
        assert "stream" in params
