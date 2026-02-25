"""OpenAI backend for cloud LLM inference (optional dependency)."""

from __future__ import annotations

import ipaddress
import os
import socket
import time
from typing import Any
from urllib.parse import urlparse

from maxim.utils.logging import warn
from maxim.models.language.router import LLMConfig, LLMResponse


def _is_auth_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except Exception:
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    )


def _validate_base_url(base_url: str, allow_local: bool) -> str | None:
    try:
        parsed = urlparse(base_url)
    except Exception:
        return None
    if parsed.scheme != "https":
        return None
    host = parsed.hostname
    if not host:
        return None
    try:
        addrinfos = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except Exception:
        return None
    for info in addrinfos:
        ip = info[4][0]
        if _is_private_ip(ip) and not allow_local:
            return None
    return base_url


class _OpenAIBackend:
    """OpenAI backend using the official SDK."""

    def __init__(self, cfg: LLMConfig, provider_key: str = "openai") -> None:
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
        env_key = str(cfg.get("api_key_env") or "OPENAI_API_KEY")
        return str(os.getenv(env_key, "")).strip()

    def _get_base_url(self) -> str | None:
        cfg = self._provider_cfg()
        base_url = cfg.get("base_url")
        if not base_url:
            return None
        allow_local = bool(cfg.get("allow_local_endpoints", False))
        validated = _validate_base_url(str(base_url).strip(), allow_local)
        if not validated:
            warn("Blocked base_url for provider %s (SSRF protection)", self._provider_key)
            return None
        return validated

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
            from openai import OpenAI  # type: ignore
        except Exception as e:
            warn("OpenAI backend unavailable (install `openai`): %s", e)
            return None
        api_key = self._get_api_key()
        if not api_key:
            warn("OpenAI API key missing")
            return None
        try:
            base_url = self._get_base_url()
            if self._provider_key in ("openai_compatible", "openai_compat") and not base_url:
                warn("OpenAI-compatible provider requires a valid base_url")
                return None
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self._get_timeout(),
            )
            return self._client
        except Exception as e:
            warn("Failed to init OpenAI client: %s", e)
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

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_err: Exception | None = None
        for attempt in range(self._get_max_retries() + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=list(stop) if stop else None,
                )
                choice = resp.choices[0] if getattr(resp, "choices", None) else None
                content = ""
                if choice and getattr(choice, "message", None) is not None:
                    content = str(getattr(choice.message, "content", "") or "").strip()

                usage = getattr(resp, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
                output_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0
                cached = 0
                details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
                if details is not None:
                    cached = getattr(details, "cached_tokens", 0) or 0
                uncached = max(0, input_tokens - cached) if input_tokens else 0

                return LLMResponse(
                    content=content,
                    input_tokens=int(input_tokens or 0),
                    output_tokens=int(output_tokens or 0),
                    model=model,
                    latency_ms=(time.time() - start) * 1000,
                    provider=self._provider_key,
                    stop_reason=str(getattr(choice, "finish_reason", "")) if choice else "",
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

        warn("OpenAI call failed: %s", last_err)
        return LLMResponse(content="")
