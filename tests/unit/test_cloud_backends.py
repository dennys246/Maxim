"""Tests for cloud backend Phase 3 features: prompt caching, tool use, thinking, streaming."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock


from maxim.models.language.config import LLMConfig


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
                "model": "claude-sonnet-4-6",
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
            "model": "claude-sonnet-4-6",
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

    def test_prompt_cache_bool_shorthand(self):
        backend = self._make_backend(prompt_cache=True)
        assert backend._prompt_cache_enabled() is True

    def test_prompt_cache_disabled_default(self):
        backend = self._make_backend(prompt_cache=None)
        # No provider flag and cfg.prompt_cache defaults False.
        assert backend._prompt_cache_enabled() is False

    def test_prompt_cache_cfg_fallback(self):
        """Default cloud path: provider entry has no prompt_cache key, but the
        top-level cfg flag (set from the profile) enables it."""
        from maxim.models.language.anthropic_backend import _AnthropicBackend

        cfg = _make_cfg(
            prompt_cache=True,
            providers={"local": {"type": "anthropic", "model": "claude-sonnet-4-6"}},
        )
        backend = _AnthropicBackend(cfg, provider_key="local")
        assert backend._prompt_cache_enabled() is True

    def test_provider_entry_overrides_cfg_fallback(self):
        """An explicit provider-level prompt_cache=False wins over cfg=True."""
        from maxim.models.language.anthropic_backend import _AnthropicBackend

        cfg = _make_cfg(
            prompt_cache=True,
            providers={"local": {"type": "anthropic", "model": "claude-sonnet-4-6", "prompt_cache": False}},
        )
        backend = _AnthropicBackend(cfg, provider_key="local")
        assert backend._prompt_cache_enabled() is False

    def test_parse_response_emits_cache_event(self, caplog):
        import logging

        backend = self._make_backend(prompt_cache=True)
        usage = MagicMock(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=900,
            cache_creation_input_tokens=0,
        )
        resp = MagicMock(usage=usage, content=[], stop_reason="end_turn", id="msg_abc")
        with caplog.at_level(logging.INFO, logger="maxim.models.language.anthropic_backend"):
            backend._parse_response(resp, "claude-sonnet-4-6", 0.0)
        events = [r for r in caplog.records if r.getMessage() == "anthropic_cache"]
        assert len(events) == 1
        data = events[0].data
        assert data["cache_read_tokens"] == 900
        assert data["cache_write_tokens"] == 0
        assert data["input_tokens_uncached"] == 100
        assert abs(data["cache_hit_ratio"] - 0.9) < 1e-6
        assert data["request_id"] == "msg_abc"

    def test_parse_response_no_cache_event_when_disabled(self, caplog):
        import logging

        backend = self._make_backend(prompt_cache=None)
        usage = MagicMock(input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        resp = MagicMock(usage=usage, content=[], stop_reason="end_turn", id="msg_x")
        with caplog.at_level(logging.INFO, logger="maxim.models.language.anthropic_backend"):
            backend._parse_response(resp, "claude-sonnet-4-6", 0.0)
        assert not [r for r in caplog.records if r.getMessage() == "anthropic_cache"]

    # ── Capacity-error backoff (429 / 529) ──────────────────────────────

    def test_overloaded_error_detection(self):
        from maxim.models.language.anthropic_backend import (
            _is_overloaded_error,
            _is_rate_limit_error,
            _is_capacity_error,
        )

        ov = Exception("Error code: 529 - {'type': 'overloaded_error'}")
        rl = Exception("Error code: 429 - rate_limit_error")
        other = Exception("Error code: 500 - internal")
        assert _is_overloaded_error(ov) is True
        assert _is_overloaded_error(rl) is False  # 529-specific
        assert _is_rate_limit_error(rl) is True
        assert _is_capacity_error(ov) is True and _is_capacity_error(rl) is True
        assert _is_capacity_error(other) is False

    def test_retry_after_seconds_parsing(self):
        from maxim.models.language.anthropic_backend import _retry_after_seconds

        err = Exception("429")
        err.response = MagicMock()
        err.response.headers = {"retry-after": "12"}
        assert _retry_after_seconds(err) == 12.0
        # absent / no response
        assert _retry_after_seconds(Exception("nope")) is None

    def test_capacity_backoff_bounded_and_honors_retry_after(self):
        from maxim.models.language.anthropic_backend import _AnthropicBackend

        # exponential, jittered, capped at 30 (no retry-after)
        for attempt in range(8):
            b = _AnthropicBackend._capacity_backoff(attempt, Exception("529"))
            assert 0 < b <= 30 * 1.3 + 0.01
        # retry-after raises the floor
        err = Exception("429")
        err.response = MagicMock()
        err.response.headers = {"retry-after": "45"}
        b = _AnthropicBackend._capacity_backoff(0, err)
        assert b >= 45 * 0.7  # honored (with jitter band), capped at 60*1.3

    def test_capacity_max_retries_default(self):
        backend = self._make_backend()
        assert backend._get_capacity_max_retries() == 6

    def test_529_retries_then_succeeds(self):
        """A 529 on the first attempt must be retried (not exhaust the tiny
        generic budget) and succeed on the next — the core 2026-06-08 fix."""
        from unittest.mock import patch

        backend = self._make_backend(capacity_max_retries=3)
        # Build a fake client whose .messages.create raises 529 once then returns.
        good = MagicMock()
        good.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        good.content = []
        good.stop_reason = "end_turn"
        good.id = "msg_ok"
        calls = {"n": 0}

        def _create(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Exception("Error code: 529 - overloaded_error")
            return good

        fake_client = MagicMock()
        fake_client.messages.create.side_effect = _create
        backend._client = fake_client

        # Patch backoff sleep so the test is instant.
        with patch("maxim.models.language.anthropic_backend.shutdown_wait", return_value=False):
            resp = backend.complete_with_usage(system="s", user="u", max_tokens=16, temperature=0.0)
        assert calls["n"] == 2  # retried once
        assert resp.input_tokens == 10  # succeeded on retry

    def test_generic_error_uses_small_budget(self):
        """A non-capacity error exhausts the small generic budget (2) — does NOT
        get the capacity budget."""
        from unittest.mock import patch

        backend = self._make_backend()  # generic max_retries=2
        calls = {"n": 0}

        def _create(**kw):
            calls["n"] += 1
            raise Exception("Error code: 500 - internal_server_error")

        fake_client = MagicMock()
        fake_client.messages.create.side_effect = _create
        backend._client = fake_client
        with patch("maxim.models.language.anthropic_backend.shutdown_wait", return_value=False):
            resp = backend.complete_with_usage(system="s", user="u", max_tokens=16, temperature=0.0)
        assert resp.content == ""  # failed
        assert calls["n"] == 3  # 1 initial + 2 generic retries (NOT 7)

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

        result = backend._parse_response(mock_resp, "claude-sonnet-4-6", 0.0)
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

        result = backend._parse_response(mock_resp, "claude-sonnet-4-6", 0.0)
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

        result = backend._parse_response(mock_resp, "claude-sonnet-4-6", 0.0)
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
        mock_resp.usage.prompt_cache_hit_tokens = 0  # real OpenAI lacks DeepSeek's field

        result = backend._parse_response(mock_resp, "gpt-4o", 0.0)
        assert result.content == '{"answer": "hello"}'
        assert result.input_tokens == 50
        assert result.output_tokens == 20

    def test_streaming_flag(self):
        backend = self._make_backend()
        assert backend.supports_streaming is True

    # ── Prompt caching (Phase 2) ────────────────────────────────────────

    def _make_backend_with_cache(self, prompt_cache=True):
        from maxim.models.language.openai_backend import _OpenAIBackend

        cfg = LLMConfig(
            enabled=True,
            providers={
                "openai": {
                    "type": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-4o",
                    "prompt_cache": prompt_cache,
                }
            },
        )
        return _OpenAIBackend(cfg, provider_key="openai")

    def test_prompt_cache_enabled_provider_bool(self):
        assert self._make_backend_with_cache(True)._prompt_cache_enabled() is True

    def test_prompt_cache_disabled_default(self):
        assert self._make_backend()._prompt_cache_enabled() is False

    def test_prompt_cache_cfg_fallback(self):
        from maxim.models.language.openai_backend import _OpenAIBackend

        cfg = LLMConfig(
            enabled=True,
            prompt_cache=True,
            providers={"local": {"type": "openai", "model": "gpt-4o"}},
        )
        backend = _OpenAIBackend(cfg, provider_key="local")
        assert backend._prompt_cache_enabled() is True

    def test_parse_response_emits_openai_cache_event(self, caplog):
        import logging

        backend = self._make_backend_with_cache(True)
        mock_resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "{}"
        choice.finish_reason = "stop"
        mock_resp.choices = [choice]
        mock_resp.id = "chatcmpl-xyz"
        mock_resp.usage.prompt_tokens = 1000
        mock_resp.usage.completion_tokens = 20
        details = MagicMock()
        details.cached_tokens = 768
        mock_resp.usage.prompt_tokens_details = details

        with caplog.at_level(logging.INFO, logger="maxim.models.language.openai_backend"):
            backend._parse_response(mock_resp, "gpt-4o", 0.0)
        events = [r for r in caplog.records if r.getMessage() == "openai_cache"]
        assert len(events) == 1
        data = events[0].data
        assert data["cache_read_tokens"] == 768
        assert data["input_tokens_uncached"] == 232  # 1000 - 768 (OpenAI prompt_tokens is inclusive)
        assert data["prompt_tokens_total"] == 1000
        assert abs(data["cache_hit_ratio"] - 0.768) < 1e-6
        assert data["request_id"] == "chatcmpl-xyz"

    def test_parse_response_no_event_when_disabled(self, caplog):
        import logging

        backend = self._make_backend()  # no prompt_cache
        mock_resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "{}"
        choice.finish_reason = "stop"
        mock_resp.choices = [choice]
        mock_resp.usage.prompt_tokens = 1000
        mock_resp.usage.completion_tokens = 20
        mock_resp.usage.prompt_tokens_details = None
        mock_resp.usage.prompt_cache_hit_tokens = 0  # real OpenAI lacks DeepSeek's field
        with caplog.at_level(logging.INFO, logger="maxim.models.language.openai_backend"):
            backend._parse_response(mock_resp, "gpt-4o", 0.0)
        assert not [r for r in caplog.records if r.getMessage() == "openai_cache"]

    def test_parse_response_reads_deepseek_cache_field(self, caplog):
        """DeepSeek reports cache hits via usage.prompt_cache_hit_tokens, not
        prompt_tokens_details.cached_tokens."""
        import logging

        backend = self._make_backend_with_cache(True)
        mock_resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "{}"
        choice.finish_reason = "stop"
        mock_resp.choices = [choice]
        mock_resp.id = "ds-1"
        mock_resp.usage.prompt_tokens = 2000
        mock_resp.usage.completion_tokens = 30
        mock_resp.usage.prompt_tokens_details = None  # DeepSeek omits this
        mock_resp.usage.prompt_cache_hit_tokens = 1536
        with caplog.at_level(logging.INFO, logger="maxim.models.language.openai_backend"):
            result = backend._parse_response(mock_resp, "deepseek-chat", 0.0)
        events = [r for r in caplog.records if r.getMessage() == "openai_cache"]
        assert len(events) == 1
        assert events[0].data["cache_read_tokens"] == 1536
        assert events[0].data["input_tokens_uncached"] == 464
        assert result.cached_input_tokens == 1536


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
            "anthropic": {"type": "anthropic", "model": "claude-sonnet-4-6"}
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


class TestPromptCacheConfigWiring:
    """Profile → cfg.prompt_cache → synthesized provider entry wiring
    (prompt_caching_for_cloud_backends.md Phase 1b)."""

    def test_builtin_claude_profiles_enable_prompt_cache(self):
        from maxim.models.language.config import _BUILTIN_PROFILES

        for name in ("claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-6"):
            assert _BUILTIN_PROFILES[name].get("prompt_cache") is True, name

    def test_builtin_gpt4o_profile_enables_prompt_cache(self):
        from maxim.models.language.config import _BUILTIN_PROFILES

        assert _BUILTIN_PROFILES["gpt-4o"].get("prompt_cache") is True

    # ── DeepSeek / OpenAI-compatible default-path dispatch ──────────────

    def test_deepseek_has_default_pricing(self):
        from maxim.models.language.cloud_dispatch import DEFAULT_PRICING

        assert "deepseek-chat" in DEFAULT_PRICING

    def test_load_llm_config_deepseek_threads_base_url_and_key_env(self, monkeypatch):
        from maxim.models.language.config import load_llm_config

        monkeypatch.setenv("MAXIM_LLM_CONFIG", "/nonexistent/llm.json")
        cfg = load_llm_config(profile_override="deepseek-chat")
        assert cfg.base_url == "https://api.deepseek.com/v1"
        assert cfg.api_key_env == "DEEPSEEK_API_KEY"
        assert cfg.prompt_cache is True

    def test_normalize_providers_surfaces_base_url_and_key_env(self):
        from maxim.models.language.cloud_dispatch import normalize_providers

        cfg = _make_cfg(
            providers={},
            backend="openai",
            base_url="https://api.deepseek.com/v1",
            api_key_env="DEEPSEEK_API_KEY",
        )
        entry = normalize_providers(cfg)["local"]
        assert entry["base_url"] == "https://api.deepseek.com/v1"
        assert entry["api_key_env"] == "DEEPSEEK_API_KEY"

    def test_normalize_providers_omits_blank_base_url(self):
        from maxim.models.language.cloud_dispatch import normalize_providers

        cfg = _make_cfg(providers={}, backend="openai")  # no base_url / api_key_env
        entry = normalize_providers(cfg)["local"]
        assert "base_url" not in entry
        assert "api_key_env" not in entry

    def test_openai_backend_api_key_env_cfg_fallback(self, monkeypatch):
        """Default cloud path: provider entry (cfg.providers) is empty, so the
        backend must read api_key_env from the top-level LLMConfig."""
        from maxim.models.language.openai_backend import _OpenAIBackend

        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = _make_cfg(providers={}, backend="openai", api_key_env="DEEPSEEK_API_KEY")
        backend = _OpenAIBackend(cfg, provider_key="local")
        assert backend._get_api_key() == "sk-deepseek-test"

    def test_openai_backend_base_url_cfg_fallback(self):
        from maxim.models.language.openai_backend import _OpenAIBackend

        cfg = _make_cfg(providers={}, backend="openai", base_url="https://api.deepseek.com/v1")
        backend = _OpenAIBackend(cfg, provider_key="local")
        assert backend._get_base_url() == "https://api.deepseek.com/v1"

    def test_normalize_providers_surfaces_prompt_cache(self):
        from maxim.models.language.cloud_dispatch import normalize_providers

        cfg = _make_cfg(prompt_cache=True, providers={})
        synthesized = normalize_providers(cfg)
        assert synthesized["local"]["prompt_cache"] is True

    def test_normalize_providers_default_prompt_cache_false(self):
        from maxim.models.language.cloud_dispatch import normalize_providers

        cfg = _make_cfg(providers={})  # prompt_cache defaults False
        synthesized = normalize_providers(cfg)
        assert synthesized["local"]["prompt_cache"] is False

    def test_load_llm_config_claude_profile_enables_prompt_cache(self, monkeypatch):
        from maxim.models.language.config import load_llm_config

        # Isolate from any repo-local llm.json / env profile override.
        monkeypatch.setenv("MAXIM_LLM_CONFIG", "/nonexistent/llm.json")
        monkeypatch.delenv("MAXIM_LLM_PROMPT_CACHE", raising=False)
        cfg = load_llm_config(profile_override="claude-sonnet")
        assert cfg.prompt_cache is True

    def test_load_llm_config_local_profile_no_prompt_cache(self, monkeypatch):
        from maxim.models.language.config import load_llm_config

        monkeypatch.setenv("MAXIM_LLM_CONFIG", "/nonexistent/llm.json")
        monkeypatch.delenv("MAXIM_LLM_PROMPT_CACHE", raising=False)
        cfg = load_llm_config(profile_override="mistral-7b")
        assert cfg.prompt_cache is False
