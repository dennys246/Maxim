"""Anthropic Claude backend for cloud LLM inference (optional dependency)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from maxim.utils.logging import warn
from maxim.utils.optional_deps import require_optional_dependency
from maxim.utils.structured_logging import log_structured
from maxim.models.language.cancellation import is_shutdown_requested, shutdown_wait
from maxim.models.language.config import LLMConfig
from maxim.models.language.types import LLMResponse

logger = logging.getLogger(__name__)


def _is_auth_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg


def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "429" in msg or "rate" in msg


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema for ExecAgent's ProposedGoal
# ─────────────────────────────────────────────────────────────────────────────

PROPOSED_GOAL_TOOL: dict[str, Any] = {
    "name": "propose_goal",
    "description": (
        "Propose a goal with an associated tool action. Return null goal_description if no action is needed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "goal_description": {
                "type": ["string", "null"],
                "description": "What you want to achieve, or null if no goal needed.",
            },
            "priority": {
                "type": "string",
                "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "IDLE"],
                "description": "Goal priority level.",
            },
            "tool_name": {
                "type": "string",
                "description": "The tool to invoke (e.g. track_target, maxim_command, math, internet_search).",
            },
            "tool_params": {
                "type": "object",
                "description": "Parameters for the tool.",
                "additionalProperties": True,
            },
            "reasoning": {
                "type": "string",
                "description": "How this goal serves the root goal.",
            },
            "sub_goals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "tool_name": {"type": "string"},
                        "tool_params": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["tool_name"],
                },
                "description": "Optional sub-goals for multi-step plans.",
            },
        },
        "required": ["goal_description", "priority"],
    },
}


class _AnthropicBackend:
    """Anthropic backend using the official SDK."""

    def __init__(self, cfg: LLMConfig, provider_key: str = "anthropic") -> None:
        self.cfg = cfg
        self._provider_key = provider_key
        self._client: Any | None = None
        self.requires_prompt_formatting = False
        self.supports_model_override = True
        self.supports_tool_use = True
        self.supports_streaming = True

    def _provider_cfg(self) -> dict[str, Any]:
        providers = getattr(self.cfg, "providers", {}) or {}
        raw = providers.get(self._provider_key, {})
        return raw if isinstance(raw, dict) else {}

    def _get_api_key(self) -> str:
        cfg = self._provider_cfg()
        # Provider entry wins; fall back to the top-level LLMConfig.api_key_env
        # (set from the active profile) for the default cloud path where the
        # provider entry is synthesized in the router — same shape as the
        # OpenAI backend. Default stays ANTHROPIC_API_KEY.
        env_key = str(cfg.get("api_key_env") or getattr(self.cfg, "api_key_env", "") or "ANTHROPIC_API_KEY")
        return str(os.getenv(env_key, "")).strip()

    def _get_timeout(self) -> float:
        cfg = self._provider_cfg()
        try:
            return float(cfg.get("timeout_s", 60.0))
        except Exception:
            return 60.0

    def _get_max_retries(self) -> int:
        cfg = self._provider_cfg()
        try:
            return int(cfg.get("max_retries", 2))
        except Exception:
            return 2

    def _prompt_cache_enabled(self) -> bool:
        cfg = self._provider_cfg()
        cache_cfg = cfg.get("prompt_cache")
        if isinstance(cache_cfg, dict):
            return bool(cache_cfg.get("enabled", False))
        if cache_cfg is not None:
            return bool(cache_cfg)
        # Fall back to the top-level LLMConfig flag (set from the active
        # profile by load_llm_config) for the default cloud path where the
        # provider entry was synthesized without an explicit prompt_cache key.
        return bool(getattr(self.cfg, "prompt_cache", False))

    def _thinking_config(self) -> dict[str, Any] | None:
        cfg = self._provider_cfg()
        thinking = cfg.get("thinking")
        if not isinstance(thinking, dict):
            return None
        if not thinking.get("enabled", False):
            return None
        budget = int(thinking.get("budget_tokens", 5000))
        budget = max(1024, min(budget, 20000))
        return {"budget_tokens": budget}

    def _ensure_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        # Requested-but-missing dependency is a SETUP error, not a transient
        # failure: raise loudly (aborts the run with an actionable hint) rather
        # than returning None and letting the router mask it as "no eligible
        # providers". This is the 2026-06-05 cloud-dispatch incident fix.
        anthropic_mod = require_optional_dependency("anthropic", feature="Anthropic backend")
        Anthropic = anthropic_mod.Anthropic
        api_key = self._get_api_key()
        if not api_key:
            warn("Anthropic API key missing. Fix: export ANTHROPIC_API_KEY=<your-key>")
            return None
        try:
            self._client = Anthropic(api_key=api_key, timeout=self._get_timeout())
            return self._client
        except Exception as e:
            warn("Failed to init Anthropic client: %s", e)
            return None

    def warmup(self) -> bool:
        """Validate API key presence (no billable call)."""
        return bool(self._get_api_key())

    def unload(self) -> None:
        self._client = None

    def _build_system_blocks(self, system: str) -> list[dict[str, Any]]:
        """Build system parameter with optional cache_control markers."""
        if not system:
            return []
        if not self._prompt_cache_enabled():
            return [{"type": "text", "text": system}]
        # Mark the system prompt as cacheable (ephemeral = server-managed TTL)
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _log_cache_usage(
        self,
        *,
        input_tokens: int,
        cache_read: int,
        cache_write: int,
        request_id: str | None,
    ) -> None:
        """Emit a per-call ``anthropic_cache`` structured event.

        Only fires when prompt caching is enabled so cached runs produce one
        event per turn (cache_read=0 on a miss is signal, not noise). Pairs with
        ``MAXIM_LOG_FILE`` for JSONL analysis. See
        docs/plans/prompt_caching_for_cloud_backends.md Phase 1c.
        """
        if not self._prompt_cache_enabled():
            return
        # Anthropic reports cache_read / cache_write / (uncached) input_tokens as
        # three disjoint counts; their sum is the full prompt size.
        total = cache_read + cache_write + input_tokens
        ratio = (cache_read / total) if total > 0 else 0.0
        log_structured(
            logger,
            logging.INFO,
            event="anthropic_cache",
            data={
                "provider": self._provider_key,
                "cache_read_tokens": int(cache_read),
                "cache_write_tokens": int(cache_write),
                "input_tokens_uncached": int(input_tokens),
                "cache_hit_ratio": round(ratio, 4),
                "request_id": request_id,
            },
        )

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: tuple[str, ...],
        system: str | None = None,
    ) -> str:
        resp = self.complete_with_usage(
            system=system or "",
            user=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return resp.content

    def complete_with_usage(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        stop: tuple[str, ...] = (),
        model_override: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        start = time.time()
        client = self._ensure_client()
        if client is None:
            return LLMResponse(content="")

        model = str(model_override or self._provider_cfg().get("model") or getattr(self.cfg, "model", "") or "").strip()

        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        system_blocks = self._build_system_blocks(system)

        # Build kwargs for the API call
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system_blocks:
            create_kwargs["system"] = system_blocks

        # Extended thinking support
        thinking_cfg = thinking or self._thinking_config()
        if thinking_cfg:
            budget = thinking_cfg.get("budget_tokens", 5000)
            create_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
            # Extended thinking requires temperature=1 per Anthropic API
            create_kwargs["temperature"] = 1.0
        else:
            create_kwargs["temperature"] = temperature

        # Tool use support
        if tools:
            create_kwargs["tools"] = tools

        last_err: Exception | None = None
        for attempt in range(self._get_max_retries() + 1):
            # Abort if the process is shutting down. Previously a user Ctrl+C
            # mid-retry would run the loop to completion (up to ~8s of
            # uninterruptible backoff sleeps) while burning cloud credits.
            if is_shutdown_requested():
                warn("Anthropic call aborted: shutdown requested during retry loop")
                return LLMResponse(content="")
            try:
                if stream:
                    return self._stream_response(create_kwargs, start)

                resp = client.messages.create(**create_kwargs)
                return self._parse_response(resp, model, start)
            except Exception as e:
                last_err = e
                if _is_auth_error(e):
                    self._client = None
                if attempt < self._get_max_retries():
                    backoff = 0.5 * (attempt + 1)
                    if _is_rate_limit_error(e):
                        backoff = min(backoff * 4, 30.0)
                    # shutdown_wait returns True early if shutdown fires
                    # mid-backoff; we bail out on the next iteration check.
                    if shutdown_wait(backoff):
                        warn("Anthropic call aborted: shutdown requested during backoff")
                        return LLMResponse(content="")
                    continue
                break

        warn("Anthropic call failed: %s", last_err)
        return LLMResponse(content="")

    def _parse_response(self, resp: Any, model: str, start: float) -> LLMResponse:
        """Parse an Anthropic API response into LLMResponse."""
        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage is not None else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0
        cached = 0
        cache_creation = 0
        if usage is not None:
            cached = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            # Also count cache_creation_input_tokens as uncached
        uncached = max(0, input_tokens - cached) if input_tokens else 0

        self._log_cache_usage(
            input_tokens=int(input_tokens or 0),
            cache_read=int(cached or 0),
            cache_write=int(cache_creation or 0),
            request_id=getattr(resp, "id", None),
        )

        text_blocks: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in getattr(resp, "content", []) or []:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text = getattr(block, "text", None)
                if text:
                    text_blocks.append(str(text))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "input": getattr(block, "input", {}),
                    }
                )
            elif block_type == "thinking":
                str(getattr(block, "thinking", "") or "")

        content = "\n".join(text_blocks).strip()

        # If the response is purely tool_use with no text, serialize tool input as content
        # so callers that expect text-based JSON still work
        if not content and tool_calls and len(tool_calls) == 1:
            import json

            content = json.dumps(tool_calls[0].get("input", {}))

        return LLMResponse(
            content=content,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            model=model,
            latency_ms=(time.time() - start) * 1000,
            provider=self._provider_key,
            stop_reason=str(getattr(resp, "stop_reason", "")),
            tool_calls=tool_calls if tool_calls else None,
            cached_input_tokens=int(cached or 0),
            uncached_input_tokens=int(uncached or 0),
        )

    def _stream_response(self, create_kwargs: dict[str, Any], start: float) -> LLMResponse:
        """Stream a response, collecting text incrementally."""
        client = self._client
        if client is None:
            return LLMResponse(content="")

        model = create_kwargs.get("model", "")
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        cache_creation_tokens = 0
        request_id: str | None = None
        stop_reason = ""

        # Track tool_use blocks being built from deltas
        current_tool: dict[str, Any] | None = None
        current_tool_json = ""

        with client.messages.stream(**create_kwargs) as stream:
            for event in stream:
                event_type = getattr(event, "type", "")

                if event_type == "message_start":
                    msg = getattr(event, "message", None)
                    if msg:
                        request_id = getattr(msg, "id", None)
                        usage = getattr(msg, "usage", None)
                        if usage:
                            input_tokens = getattr(usage, "input_tokens", 0) or 0
                            cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
                            cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0

                elif event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", "") == "tool_use":
                        current_tool = {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            "input": {},
                        }
                        current_tool_json = ""

                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        delta_type = getattr(delta, "type", "")
                        if delta_type == "text_delta":
                            text_parts.append(str(getattr(delta, "text", "")))
                        elif delta_type == "input_json_delta":
                            current_tool_json += str(getattr(delta, "partial_json", ""))

                elif event_type == "content_block_stop":
                    if current_tool is not None:
                        import json as _json

                        try:
                            current_tool["input"] = _json.loads(current_tool_json) if current_tool_json else {}
                        except Exception:
                            current_tool["input"] = {}
                        tool_calls.append(current_tool)
                        current_tool = None
                        current_tool_json = ""

                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        stop_reason = str(getattr(delta, "stop_reason", "") or "")
                    usage = getattr(event, "usage", None)
                    if usage:
                        output_tokens = getattr(usage, "output_tokens", 0) or 0

        uncached = max(0, input_tokens - cached_tokens) if input_tokens else 0
        content = "".join(text_parts).strip()

        self._log_cache_usage(
            input_tokens=int(input_tokens or 0),
            cache_read=int(cached_tokens or 0),
            cache_write=int(cache_creation_tokens or 0),
            request_id=request_id,
        )

        if not content and tool_calls and len(tool_calls) == 1:
            import json as _json

            content = _json.dumps(tool_calls[0].get("input", {}))

        return LLMResponse(
            content=content,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            model=model,
            latency_ms=(time.time() - start) * 1000,
            provider=self._provider_key,
            stop_reason=stop_reason,
            tool_calls=tool_calls if tool_calls else None,
            cached_input_tokens=int(cached_tokens or 0),
            uncached_input_tokens=int(uncached or 0),
        )

    def complete_with_tools(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        model_override: str | None = None,
        thinking: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        """Complete with native tool use — returns structured tool calls."""
        return self.complete_with_usage(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            model_override=model_override,
            tools=tools,
            thinking=thinking,
            stream=stream,
        )
