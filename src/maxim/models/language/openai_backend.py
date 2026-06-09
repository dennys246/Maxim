"""OpenAI backend for cloud LLM inference (optional dependency)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from maxim.utils.logging import warn
from maxim.utils.optional_deps import require_optional_dependency
from maxim.utils.structured_logging import log_structured

log = logging.getLogger(__name__)
from maxim.models.language.cancellation import is_shutdown_requested, shutdown_wait
from maxim.models.language.config import LLMConfig
from maxim.models.language.types import LLMResponse

# Fix #6 (R2 review): Plan 2 R2d moved the SSRF check from this module
# to maxim.utils.net. The alias preserves backward compatibility for
# anyone importing _validate_base_url from openai_backend. Moved to the
# imports block so the E402 noqa suppression is no longer needed.
from maxim.utils.net import validate_base_url as _validate_base_url

# Stall-detector integration: warn-once shim around the registry import.
# Architecture review I5: silent-swallow re-introduces the bug class
# llm_call_registry exists to close.
_register_byte_received_warned: bool = False


def _register_byte_received_safe() -> None:
    global _register_byte_received_warned
    try:
        from maxim.runtime.llm_call_registry import register_byte_received

        register_byte_received()
    except Exception as exc:  # pragma: no cover — defensive
        if not _register_byte_received_warned:
            _register_byte_received_warned = True
            warn(
                "Stall registry instrumentation failed (further occurrences suppressed): %s",
                exc,
            )


def _is_auth_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg


def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "429" in msg or "rate" in msg


def _sanitize_error_message(err: BaseException | None, *, max_len: int = 500) -> str:
    """Collapse HTML error bodies to a short summary; truncate long strings.

    Cloudflare (and other reverse proxies) return ~4KB HTML bodies on 5xx
    errors that the OpenAI client surfaces via ``str(err)``. Logging those
    verbatim buries the terminal. This helper detects HTML anywhere in the
    first 200 characters (covers exception-chain prefixes like
    ``"RuntimeError: <!DOCTYPE html>..."``) and replaces the body with a
    one-line ``<title, N bytes>`` summary.

    Plain-text error messages over ``max_len`` are truncated with an
    ellipsis marker. Short plain strings pass through unchanged.
    """
    if err is None:
        return ""
    import re

    msg = str(err)
    # Check the first 200 chars so HTML bodies wrapped by an exception
    # chain (e.g. "RuntimeError: <!DOCTYPE ...") are still caught.
    head = msg[:200]
    head_lower = head.lower()
    if "<!doctype" in head_lower or "<html" in head_lower:
        title_match = re.search(r"<title>([^<]+)</title>", msg, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "HTML error body"
        return f"<{title}, {len(msg)} bytes>"
    if len(msg) > max_len:
        return msg[:max_len] + "… (truncated)"
    return msg


def _http_status_of(err: Exception | None) -> int | None:
    """Best-effort extraction of HTTP status code from an openai client error."""
    if err is None:
        return None
    code = getattr(err, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(err, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    # Last-resort string scan
    msg = str(err)
    for candidate in ("401", "403", "404", "429", "500", "502", "503", "504"):
        if candidate in msg:
            return int(candidate)
    return None


def _is_bad_gateway_error(err: Exception) -> bool:
    """Detect 502/503/504 — transient upstream errors (server restarting)."""
    status = _http_status_of(err)
    return status in (502, 503, 504)


def _parse_processing_ms(headers: Any) -> float | None:
    """Extract server-side processing time from openai-processing-ms header."""
    val = None
    if hasattr(headers, "get"):
        val = headers.get("openai-processing-ms")
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


class _OpenAIBackend:
    """OpenAI backend using the official SDK."""

    def __init__(self, cfg: LLMConfig, provider_key: str = "openai") -> None:
        self.cfg = cfg
        self._provider_key = provider_key
        self._client: Any | None = None
        self.requires_prompt_formatting = False
        self.supports_model_override = True
        self.supports_streaming = True

    def _provider_cfg(self) -> dict[str, Any]:
        providers = getattr(self.cfg, "providers", {}) or {}
        raw = providers.get(self._provider_key, {})
        return raw if isinstance(raw, dict) else {}

    def _get_api_key(self) -> str:
        cfg = self._provider_cfg()
        # Provider entry wins; fall back to the top-level LLMConfig.api_key_env
        # (set from the active profile by load_llm_config) for the default cloud
        # path where the provider entry is synthesized in the router and the
        # backend's cfg.providers is empty — same fallback shape as model /
        # prompt_cache. Without this, an OpenAI-compatible profile with a custom
        # key env (e.g. DeepSeek → DEEPSEEK_API_KEY) silently reads OPENAI_API_KEY.
        env_key = str(cfg.get("api_key_env") or getattr(self.cfg, "api_key_env", "") or "OPENAI_API_KEY")
        return str(os.getenv(env_key, "")).strip()

    def _prompt_cache_enabled(self) -> bool:
        """Whether to emit ``openai_cache`` instrumentation for this provider.

        NOTE: unlike Anthropic, OpenAI Chat Completions has NO ``cache_control``
        marker — prompt caching is fully automatic for prompts >1024 tokens and
        keys off the stable message PREFIX (which Phase 1's system/user split
        already provides). This flag therefore does NOT enable caching (it is
        always on, server-side); it only gates the ``openai_cache`` measurement
        event so cache-relevant runs produce telemetry without noise elsewhere.
        Mirrors _AnthropicBackend._prompt_cache_enabled for config symmetry.
        """
        cfg = self._provider_cfg()
        cache_cfg = cfg.get("prompt_cache")
        if isinstance(cache_cfg, dict):
            return bool(cache_cfg.get("enabled", False))
        if cache_cfg is not None:
            return bool(cache_cfg)
        return bool(getattr(self.cfg, "prompt_cache", False))

    def _log_cache_usage(
        self,
        *,
        input_tokens: int,
        cache_read: int,
        request_id: str | None,
    ) -> None:
        """Emit a per-call ``openai_cache`` structured event when caching is
        enabled. OpenAI reports cached tokens via
        ``usage.prompt_tokens_details.cached_tokens``; ``input_tokens``
        (prompt_tokens) is INCLUSIVE of cached on OpenAI, so uncached =
        input_tokens - cache_read. See
        docs/plans/prompt_caching_for_cloud_backends.md Phase 2.
        """
        if not self._prompt_cache_enabled():
            return
        uncached = max(0, input_tokens - cache_read)
        ratio = (cache_read / input_tokens) if input_tokens > 0 else 0.0
        log_structured(
            log,
            logging.INFO,
            event="openai_cache",
            data={
                "provider": self._provider_key,
                "cache_read_tokens": int(cache_read),
                "input_tokens_uncached": int(uncached),
                "prompt_tokens_total": int(input_tokens),
                "cache_hit_ratio": round(ratio, 4),
                "request_id": request_id,
            },
        )

    def _get_base_url(self) -> str | None:
        cfg = self._provider_cfg()
        # Provider entry wins; fall back to the top-level LLMConfig.base_url (set
        # from the active profile) for the default cloud path — see _get_api_key.
        base_url = cfg.get("base_url") or getattr(self.cfg, "base_url", "")
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
        # Requested-but-missing dependency is a SETUP error: raise loudly with
        # an actionable hint rather than returning None and letting the router
        # mask it as "no eligible providers" (the 2026-06-05 incident class).
        openai_mod = require_optional_dependency("openai", feature="OpenAI backend")
        OpenAI = openai_mod.OpenAI
        api_key = self._get_api_key()
        if not api_key:
            warn("OpenAI API key missing. Fix: export OPENAI_API_KEY=<your-key>")
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
        stream: bool = False,
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

        # Stage A observability: generate a request-id for cross-machine
        # correlation. Attached as X-Maxim-Request-Id header on every call;
        # also logged via maxim.mesh.trace when MAXIM_LANE_TRACE or
        # MAXIM_PEER_LOG_REQUESTS is set.
        from maxim.models.language.mesh_trace import (
            REQUEST_ID_HEADER,
            TraceRecord,
            emit_trace,
            enrich_trace_with_leader_status,
            generate_request_id,
        )

        request_id = generate_request_id()
        base_url = self._get_base_url() or ""
        extra_headers = {REQUEST_ID_HEADER: request_id}

        last_err: Exception | None = None
        # Extra retries for transient upstream errors (502/503/504 during
        # leader restart).  Normal retries: 2.  Gateway retries: up to 4
        # additional attempts with longer backoff, giving the LLM server
        # time to finish loading the model.
        max_retries = self._get_max_retries()
        max_gateway_retries = 4
        gateway_retries_used = 0

        attempt = 0
        while attempt <= max_retries:
            # Abort if the process is shutting down. Without this check a
            # Ctrl+C mid-retry would run the full chain (up to 2 normal + 4
            # gateway retries with backoffs totalling ~50s), burning cloud
            # credits and holding up shutdown even though the user asked to
            # stop. See models/language/cancellation.py for the primitive.
            if is_shutdown_requested():
                log.warning("OpenAI call aborted: shutdown requested during retry loop [req=%s]", request_id)
                last_err = last_err or RuntimeError("shutdown requested")
                break
            try:
                if stream:
                    return self._stream_response(client, model, messages, temperature, max_tokens, stop, start)

                # Use with_raw_response to capture server-side timing headers
                raw_resp = client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=list(stop) if stop else None,
                    extra_headers=extra_headers,
                )
                resp = raw_resp.parse()
                server_ms = _parse_processing_ms(raw_resp.headers)

                parsed = self._parse_response(resp, model, start)
                trace = TraceRecord(
                    request_id=request_id,
                    provider=self._provider_key,
                    base_url=base_url,
                    model=model,
                    status="ok",
                    http_status=200,
                    latency_ms=parsed.latency_ms,
                    input_tokens=parsed.input_tokens,
                    output_tokens=parsed.output_tokens,
                    server_processing_ms=server_ms,
                )
                enrich_trace_with_leader_status(
                    trace,
                    base_url,
                    self._get_api_key(),
                    response_headers=raw_resp.headers,
                )
                emit_trace(trace)
                return parsed
            except Exception as e:
                last_err = e
                if _is_auth_error(e):
                    self._client = None

                # 502/503/504: upstream server restarting — use longer backoff
                # and grant extra retries beyond the normal limit.
                if _is_bad_gateway_error(e) and gateway_retries_used < max_gateway_retries:
                    gateway_retries_used += 1
                    backoff = min(5.0 * gateway_retries_used, 20.0)
                    log.warning(
                        "Upstream unavailable (502/503/504), retry %d/%d in %.0fs [req=%s]",
                        gateway_retries_used,
                        max_gateway_retries,
                        backoff,
                        request_id,
                    )
                    # shutdown_wait returns early when shutdown fires; the
                    # next iteration's is_shutdown_requested() check then
                    # exits the loop cleanly.
                    if shutdown_wait(backoff):
                        log.warning("OpenAI call aborted during gateway backoff [req=%s]", request_id)
                        break
                    # Don't consume a normal retry for gateway errors
                    continue

                if attempt < max_retries:
                    backoff = 0.5 * (attempt + 1)
                    if _is_rate_limit_error(e):
                        backoff = min(backoff * 4, 30.0)
                    if shutdown_wait(backoff):
                        log.warning(
                            "OpenAI call aborted during rate-limit backoff [req=%s]",
                            request_id,
                        )
                        break
                    attempt += 1
                    continue
                break
            attempt += 1

        # Final failure — emit a trace record with error details before returning
        sanitized_error = _sanitize_error_message(last_err)
        emit_trace(
            TraceRecord(
                request_id=request_id,
                provider=self._provider_key,
                base_url=base_url,
                model=model,
                status="error",
                http_status=_http_status_of(last_err),
                latency_ms=(time.time() - start) * 1000,
                error=sanitized_error[:200] if sanitized_error else None,
            )
        )
        warn("OpenAI call failed [req=%s]: %s", request_id[:8], sanitized_error)
        return LLMResponse(content="")

    def _parse_response(self, resp: Any, model: str, start: float) -> LLMResponse:
        """Parse an OpenAI chat completion response."""
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
        if not cached and usage is not None:
            # DeepSeek (OpenAI-compatible) reports cache hits at the usage top
            # level as prompt_cache_hit_tokens rather than in
            # prompt_tokens_details.cached_tokens. Read it as a fallback so the
            # instrumentation is accurate across OpenAI-compatible providers.
            cached = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        uncached = max(0, input_tokens - cached) if input_tokens else 0

        self._log_cache_usage(
            input_tokens=int(input_tokens or 0),
            cache_read=int(cached or 0),
            request_id=getattr(resp, "id", None),
        )

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

    def _stream_response(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        stop: tuple[str, ...],
        start: float,
    ) -> LLMResponse:
        """Stream a response, collecting text incrementally."""
        text_parts: list[str] = []
        stop_reason = ""

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=list(stop) if stop else None,
            stream=True,
            stream_options={"include_usage": True},
        )

        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        request_id: str | None = None

        for chunk in stream:
            # Stall-detector integration: update last_byte_at on every
            # chunk arrival so the orchestrator's stall detector sees
            # the call as alive. Mirrors _MaximPeerBackend._stream_response.
            # See _register_byte_received_safe helper at module top
            # (architecture review I5).
            _register_byte_received_safe()
            if request_id is None:
                request_id = getattr(chunk, "id", None)
            if getattr(chunk, "usage", None):
                usage = chunk.usage
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
                details = getattr(usage, "prompt_tokens_details", None)
                if details is not None:
                    cached_tokens = getattr(details, "cached_tokens", 0) or 0
                if not cached_tokens:
                    # DeepSeek-style top-level cache-hit field (see _parse_response).
                    cached_tokens = getattr(usage, "prompt_cache_hit_tokens", 0) or 0

            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta:
                text = getattr(delta, "content", None)
                if text:
                    text_parts.append(str(text))
            finish = getattr(choices[0], "finish_reason", None)
            if finish:
                stop_reason = str(finish)

        content = "".join(text_parts).strip()
        uncached = max(0, input_tokens - cached_tokens) if input_tokens else 0

        self._log_cache_usage(
            input_tokens=int(input_tokens or 0),
            cache_read=int(cached_tokens or 0),
            request_id=request_id,
        )

        return LLMResponse(
            content=content,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            model=model,
            latency_ms=(time.time() - start) * 1000,
            provider=self._provider_key,
            stop_reason=stop_reason,
            cached_input_tokens=int(cached_tokens or 0),
            uncached_input_tokens=int(uncached or 0),
        )
