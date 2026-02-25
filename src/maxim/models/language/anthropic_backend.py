"""Anthropic Claude backend for cloud LLM inference (optional dependency)."""

from __future__ import annotations

import os
import time
from typing import Any

from maxim.utils.logging import warn
from maxim.models.language.router import LLMConfig, LLMResponse


def _is_auth_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg


class _AnthropicBackend:
    """Anthropic backend using the official SDK."""

    def __init__(self, cfg: LLMConfig, provider_key: str = "anthropic") -> None:
        self.cfg = cfg
        self._provider_key = provider_key
        self._client: Any | None = None
        self.requires_prompt_formatting = False
        self.supports_model_override = True

    def _provider_cfg(self) -> dict[str, Any]:
        providers = getattr(self.cfg, "providers", {}) or {}
        raw = providers.get(self._provider_key, {})
        return raw if isinstance(raw, dict) else {}

    def _get_api_key(self) -> str:
        cfg = self._provider_cfg()
        env_key = str(cfg.get("api_key_env") or "ANTHROPIC_API_KEY")
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

    def _ensure_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as e:
            warn("Anthropic backend unavailable (install `anthropic`): %s", e)
            return None
        api_key = self._get_api_key()
        if not api_key:
            warn("Anthropic API key missing")
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
    ) -> LLMResponse:
        start = time.time()
        client = self._ensure_client()
        if client is None:
            return LLMResponse(content="")

        model = str(model_override or self._provider_cfg().get("model") or getattr(self.cfg, "model", "") or "").strip()

        messages = [{"role": "user", "content": user}]
        last_err: Exception | None = None
        for attempt in range(self._get_max_retries() + 1):
            try:
                resp = client.messages.create(
                    model=model,
                    system=system,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                usage = getattr(resp, "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0) if usage is not None else 0
                output_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0
                cached = 0
                if usage is not None:
                    cached = getattr(usage, "cache_read_input_tokens", 0) or 0
                uncached = max(0, input_tokens - cached) if input_tokens else 0

                text_blocks = []
                for block in getattr(resp, "content", []) or []:
                    text = getattr(block, "text", None)
                    if text:
                        text_blocks.append(str(text))
                content = "\n".join(text_blocks).strip()

                return LLMResponse(
                    content=content,
                    input_tokens=int(input_tokens or 0),
                    output_tokens=int(output_tokens or 0),
                    model=model,
                    latency_ms=(time.time() - start) * 1000,
                    provider=self._provider_key,
                    stop_reason=str(getattr(resp, "stop_reason", "")),
                    cached_input_tokens=int(cached or 0),
                    uncached_input_tokens=int(uncached or 0),
                )
            except Exception as e:
                last_err = e
                if _is_auth_error(e):
                    self._client = None
                if attempt < self._get_max_retries():
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break

        warn("Anthropic call failed: %s", last_err)
        return LLMResponse(content="")
