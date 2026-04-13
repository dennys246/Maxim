"""Purpose-built backend for self-hosted Maxim peers (Plan 3 R2.5).

This backend replaces ``_OpenAIBackend`` for self-hosted peer traffic. The
single invariant that matters here is:

    ``_MaximPeerBackend.complete_with_usage()`` makes EXACTLY ONE HTTP call.

No repeated-call loop, no upstream cooldown, no empty-content fallback. If
the call fails, the backend raises a typed :class:`BackendError` subclass
and the :class:`LLMRouter`'s provider-fallback loop decides what to do
next. Adding a ``try: ... except: <call again>`` block anywhere in this
file re-introduces the 52-second fail-slow incident from 2026-04-12 and is
expressly forbidden — the Plan 3 CI invariant scans the file for the
pattern words (the only permitted match is the ``retry_after_s``
parameter name, which is the :class:`BackendOverloaded` constructor shape
fixed by Plan 2 R2b and cannot be renamed).

Design principles (see docs/plans/llm_path_fast_failover.md for full
context):

- **Single HTTP call per call-site.** No internal repeat, no internal
  cooldown. The router handles failover.
- **Typed exceptions, not empty content.** Every failure raises a
  :class:`BackendError` subclass with a ``.fix_hint``. The router's
  ``_try_provider`` bridges ``except BackendOverloaded`` → provider
  cooldown without string-matching.
- **Uses utils/http.py exclusively.** No openai SDK, no raw httpx, no
  raw urllib. Registry-backed endpoints give us pool management,
  sanitized headers, X-Maxim-\\* propagation, and automatic metrics.
- **Streaming support.** Mid-stream failures raise :class:`BackendDown`
  — this is the intentional contract difference versus
  :class:`_OpenAIBackend._stream_response` which collects partial output.
  The router's fallback loop handles partial-stream aborts by failing
  over to a different provider. See architecture doc "Behaviors not
  obvious".
- **Shutdown-aware.** Checks :func:`is_shutdown_requested` before the
  HTTP call. In-flight streams are cancelled when the shared httpx
  client closes on process shutdown.
- **Multi-agent context propagation.** The ``request_context`` kwarg is
  normalised via the canonical
  :func:`maxim.agents.llm_worker._normalize_request_context` shim
  (Plan 2 R2b). The resulting typed :class:`RequestContext` flows to
  ``utils/http.py``, which sets the ``X-Maxim-*`` headers on the
  outbound call.

NOT this backend's job:

- Cost tracking (zero cost on self-hosted peers).
- PII redaction (self-infrastructure — the router's cloud-redaction
  filter is a no-op for self-hosted providers).
- Repeat-on-failure logic (router does this).
- Internal cooldown (router's ``ProviderState`` tracks it).
- Persisted credentials (``_get_api_key`` reads the env var directly).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Mapping

from maxim.models.language.cancellation import is_shutdown_requested
from maxim.models.language.config import LLMConfig
from maxim.models.language.types import (
    INFERENCE_BROKEN_BACKOFF_S,  # noqa: F401  — re-exported for callers
    BackendAuthFailed,
    BackendDown,
    BackendError,
    BackendInferenceBroken,
    BackendModelMissing,
    BackendOverloaded,
    BackendTimeout,
    LLMResponse,
)
from maxim.utils import http as _http
from maxim.utils.http import (
    HTTPAuthError,
    HTTPClientError,
    HTTPConnectionError,
    HTTPEndpoint,
    HTTPError,
    HTTPRateLimited,
    HTTPServerError,
    HTTPTimeout,
    RequestContext,
    TimeoutPolicy,
)
from maxim.utils.logging import warn
from maxim.utils.net import validate_base_url
from maxim.utils.structured_logging import log_structured

logger = logging.getLogger(__name__)


def _backend_trace_enabled() -> bool:
    """``MAXIM_BACKEND_TRACE=1`` bumps per-call logs from DEBUG to INFO."""
    return os.environ.get("MAXIM_BACKEND_TRACE", "").strip() in ("1", "true", "yes")


class _MaximPeerBackend:
    """OpenAI-compatible backend purpose-built for self-hosted Maxim peers.

    Selection is driven by :data:`maxim.runtime.lane_backends.BACKEND_CLASSES`.
    The default resolution for self-hosted URLs is ``"maxim_peer"`` →
    this class; cloud URLs stay on ``"openai"`` →
    :class:`_OpenAIBackend`.

    Instances are cached per-provider inside :class:`LLMRouter` — one
    endpoint gets registered with ``utils/http.py`` on first use and the
    registered name is reused for every subsequent call.
    """

    requires_prompt_formatting = False
    supports_model_override = True
    supports_streaming = True
    supports_tool_use = True

    def __init__(self, cfg: LLMConfig, provider_key: str = "maxim_peer") -> None:
        self.cfg = cfg
        self._provider_key = provider_key
        self._endpoint_name = f"peer-{provider_key}"
        self._endpoint_registered = False

    # ─── Interface methods ──────────────────────────────────────────────

    def warmup(self) -> bool:
        """Validate auth + base URL. No billable call."""
        if not self._get_api_key():
            return False
        if not self._resolve_base_url():
            return False
        return True

    def unload(self) -> None:
        """The shared httpx client is registry-owned. Nothing to unload."""
        return None

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: tuple[str, ...],
        system: str | None = None,
    ) -> str:
        """Legacy prompt-formatting path. Delegates to ``complete_with_usage``."""
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
        request_context: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> LLMResponse:
        """ONE HTTP call. Raises typed ``BackendError`` on failure."""
        if is_shutdown_requested():
            raise BackendDown(
                self._provider_key,
                fix_hint="Process shutdown requested before peer call",
            )

        # Ensure the endpoint is registered with utils/http. Lazy so a
        # backend that is created but never called doesn't touch the
        # registry (keeps startup cheap).
        if not self._ensure_endpoint_registered():
            raise BackendDown(
                self._provider_key,
                fix_hint="Peer endpoint has no valid base_url (SSRF check or missing config)",
            )

        # Plan 2 R2b canonical shim — the ONLY path from legacy dict to
        # typed RequestContext. Do not duplicate.
        from maxim.agents.llm_worker import _normalize_request_context

        context = _normalize_request_context(request_context)

        model = self._resolve_model(model_override)
        payload = self._build_payload(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            model=model,
            tools=tools,
            thinking=thinking,
        )

        t0 = time.monotonic()
        try:
            if stream:
                return self._stream_response(payload, context, model, t0)
            resp = _http.post(
                self._endpoint_name,
                path="/chat/completions",
                json=payload,
                context=context,
            )
        except HTTPAuthError as e:
            self._log_failure("auth_rejected", e, context, t0)
            raise BackendAuthFailed(
                self._provider_key,
                status=e.status,
            ) from e
        except HTTPRateLimited as e:
            self._log_failure("overloaded", e, context, t0)
            raise BackendOverloaded(
                self._provider_key,
                status=e.status,
                retry_after_s=e.retry_after_s,
                suggested_peer=e.suggested_peer,
                queue_depth=self._parse_queue_depth(e.response.headers if e.response is not None else None),
            ) from e
        except HTTPClientError as e:
            # 404 on /v1/chat/completions with a model body means the
            # listener is alive but the requested model isn't loaded.
            self._log_failure("model_missing" if e.status == 404 else "client_error", e, context, t0)
            if e.status == 404:
                raise BackendModelMissing(
                    self._provider_key,
                    status=e.status,
                    requested_model=model,
                ) from e
            err = BackendError(self._provider_key, status=e.status, fix_hint=e.fix_hint)
            raise err from e
        except HTTPServerError as e:
            self._log_failure("down", e, context, t0)
            raise BackendDown(
                self._provider_key,
                status=e.status,
            ) from e
        except HTTPTimeout as e:
            elapsed_s = time.monotonic() - t0
            self._log_failure("timeout", e, context, t0)
            raise BackendTimeout(
                self._provider_key,
                elapsed_s=elapsed_s,
            ) from e
        except HTTPConnectionError as e:
            self._log_failure("down", e, context, t0)
            raise BackendDown(
                self._provider_key,
            ) from e
        except HTTPError as e:
            # Generic catch-all for HTTPError subclasses we didn't
            # handle above. Preserves status + fix_hint to the router.
            self._log_failure("generic_http_error", e, context, t0)
            err = BackendError(self._provider_key, status=e.status, fix_hint=e.fix_hint)
            raise err from e

        try:
            raw = resp.json()
        except Exception as e:
            self._log_failure("parse_error", e, context, t0)
            raise BackendInferenceBroken(
                self._provider_key,
                fix_hint=f"Peer returned non-JSON body: {type(e).__name__}",
            ) from e

        parsed = self._parse_llm_response(raw, model=model, start=t0)
        self._log_success(parsed, context, t0)
        return parsed

    def health_check(self):
        """Two-stage probe. Returns a
        :class:`maxim.runtime.llm_server.ProbeResult`.

        Stage 1 (liveness): ``GET /v1/models`` with a short timeout.
        Stage 2 (readiness): a 1-token ``POST /v1/chat/completions``.

        Consolidates the three parallel probe implementations that used to
        live in :mod:`runtime.llm_server`. Delegates to
        :func:`runtime.llm_server.probe_llm_server` in R2.5; R2.6 will
        move the implementation inside this class and delete the
        standalone function.
        """
        # R2.5: delegate to the existing two-stage probe. R2.6 folds the
        # implementation into this method and deletes the free function.
        from maxim.runtime.llm_server import probe_llm_server

        base_url = self._resolve_base_url()
        if not base_url:
            from maxim.runtime.llm_server import ProbeResult

            return ProbeResult(
                url=self._raw_base_url() or "",
                outcome="other",
                detail="SSRF check or missing config",
                latency_ms=None,
            )
        return probe_llm_server(
            base_url,
            api_key=self._get_api_key() or None,
            model_name=self._resolve_model(None),
            enable_stage2=True,
        )

    # ─── Private helpers ────────────────────────────────────────────────

    def _provider_cfg(self) -> dict[str, Any]:
        providers = getattr(self.cfg, "providers", {}) or {}
        raw = providers.get(self._provider_key, {})
        return raw if isinstance(raw, dict) else {}

    def _get_api_key(self) -> str:
        cfg = self._provider_cfg()
        env_key = str(cfg.get("api_key_env") or "MAXIM_PEER_API_KEY")
        return str(os.getenv(env_key, "")).strip()

    def _raw_base_url(self) -> str | None:
        cfg = self._provider_cfg()
        base = cfg.get("base_url")
        if not base:
            return None
        return str(base).strip()

    def _resolve_base_url(self) -> str | None:
        """Apply SSRF validation from :func:`maxim.utils.net.validate_base_url`."""
        raw = self._raw_base_url()
        if not raw:
            return None
        cfg = self._provider_cfg()
        allow_local = bool(cfg.get("allow_local_endpoints", True))  # self-hosted default
        validated = validate_base_url(raw, allow_local)
        if not validated:
            warn(
                "Blocked base_url for peer backend %s (SSRF protection)",
                self._provider_key,
            )
            return None
        return validated

    def _resolve_model(self, override: str | None) -> str:
        if override:
            return str(override)
        cfg = self._provider_cfg()
        return str(cfg.get("model") or getattr(self.cfg, "model", "") or "default")

    def _get_timeout_policy(self) -> TimeoutPolicy:
        cfg = self._provider_cfg()
        try:
            total = float(cfg.get("timeout_s", 60.0))
        except (TypeError, ValueError):
            total = 60.0
        return TimeoutPolicy(
            connect_s=3.0,
            read_s=min(total, 120.0),
            total_s=total + 5.0,
        )

    def _ensure_endpoint_registered(self) -> bool:
        if self._endpoint_registered:
            return True
        base = self._resolve_base_url()
        if not base:
            return False
        # Capture snapshots so the auth_provider callable doesn't re-read
        # the provider config on every call (amortises the SSRF check).
        api_key = self._get_api_key()
        auth_provider: Any = None
        if api_key:

            def _provide_key(_captured: str = api_key) -> str:
                return _captured

            auth_provider = _provide_key

        _http.register_endpoint(
            HTTPEndpoint(
                name=self._endpoint_name,
                base_url=base,
                default_headers={
                    "Content-Type": "application/json",
                    "User-Agent": _http.DEFAULT_USER_AGENT,
                },
                auth_provider=auth_provider,
                timeouts=self._get_timeout_policy(),
                internal=True,
            )
        )
        self._endpoint_registered = True
        return True

    def _build_payload(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        stop: tuple[str, ...],
        model: str,
        tools: list[dict[str, Any]] | None,
        thinking: dict[str, Any] | None,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if stop:
            payload["stop"] = list(stop)
        if tools:
            payload["tools"] = tools
        if thinking:
            payload["thinking"] = thinking
        return payload

    def _parse_llm_response(
        self,
        raw: dict[str, Any],
        *,
        model: str,
        start: float,
    ) -> LLMResponse:
        """Parse an OpenAI-compatible chat completion body."""
        choices = raw.get("choices") or []
        content = ""
        stop_reason = ""
        if choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            msg = first.get("message") or {}
            content = str(msg.get("content") or "").strip()
            stop_reason = str(first.get("finish_reason") or "")

        usage = raw.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
        uncached = max(0, input_tokens - cached) if input_tokens else 0

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=str(raw.get("model") or model),
            latency_ms=(time.monotonic() - start) * 1000,
            provider=self._provider_key,
            stop_reason=stop_reason,
            cached_input_tokens=cached,
            uncached_input_tokens=uncached,
        )

    def _stream_response(
        self,
        payload: dict[str, Any],
        context: RequestContext,
        model: str,
        start: float,
    ) -> LLMResponse:
        """Stream a chat completion. ONE HTTP call, strict-fail on errors.

        Mid-stream failures raise :class:`BackendDown`. Unlike
        :class:`_OpenAIBackend._stream_response`, partial output is
        deliberately NOT returned — the router's failover loop handles
        re-dispatch against a different provider. See architecture doc
        "Behaviors not obvious".
        """
        payload_with_stream = {
            **payload,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        text_parts: list[str] = []
        input_tokens = 0
        output_tokens = 0
        cached = 0
        stop_reason = ""
        chunks_seen = 0

        log_structured(
            logger,
            logging.DEBUG,
            event="peer_stream_start",
            data={
                "provider": self._provider_key,
                "model": model,
                "request_id": context.request_id,
                "agent_id": context.agent_id,
                "session_id": context.session_id,
            },
        )

        try:
            with _http.stream_post(
                self._endpoint_name,
                path="/chat/completions",
                json=payload_with_stream,
                context=context,
            ) as response:
                for line in response.iter_lines():
                    if not line:
                        continue
                    # httpx iter_lines strips trailing newline, does not
                    # strip the ``data: `` SSE prefix.
                    if line.startswith("data: "):
                        data = line[6:]
                    elif line.startswith("data:"):
                        data = line[5:]
                    else:
                        continue
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except Exception:
                        # A malformed chunk mid-stream is a broken peer.
                        # Do NOT silently accept partial output.
                        raise BackendDown(
                            self._provider_key,
                            fix_hint="Peer streamed non-JSON chunk (broken upstream)",
                        )
                    chunks_seen += 1
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        input_tokens = int(usage.get("prompt_tokens") or input_tokens)
                        output_tokens = int(usage.get("completion_tokens") or output_tokens)
                        details = usage.get("prompt_tokens_details") or {}
                        if isinstance(details, dict):
                            cached = int(details.get("cached_tokens") or cached)
                    choices = chunk.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        delta = choices[0].get("delta") or {}
                        text = delta.get("content")
                        if text:
                            text_parts.append(str(text))
                        fr = choices[0].get("finish_reason")
                        if fr:
                            stop_reason = str(fr)
                    if is_shutdown_requested():
                        raise BackendDown(
                            self._provider_key,
                            fix_hint="Process shutdown requested during stream",
                        )
        except HTTPAuthError as e:
            self._log_failure("auth_rejected", e, context, start)
            raise BackendAuthFailed(self._provider_key, status=e.status) from e
        except HTTPRateLimited as e:
            self._log_failure("overloaded", e, context, start)
            raise BackendOverloaded(
                self._provider_key,
                status=e.status,
                retry_after_s=e.retry_after_s,
                suggested_peer=e.suggested_peer,
                queue_depth=self._parse_queue_depth(e.response.headers if e.response is not None else None),
            ) from e
        except HTTPClientError as e:
            self._log_failure("model_missing" if e.status == 404 else "client_error", e, context, start)
            if e.status == 404:
                raise BackendModelMissing(
                    self._provider_key,
                    status=e.status,
                    requested_model=model,
                ) from e
            err = BackendError(self._provider_key, status=e.status, fix_hint=e.fix_hint)
            raise err from e
        except HTTPServerError as e:
            self._log_failure("down", e, context, start)
            raise BackendDown(self._provider_key, status=e.status) from e
        except HTTPTimeout as e:
            self._log_failure("timeout", e, context, start)
            raise BackendTimeout(
                self._provider_key,
                elapsed_s=time.monotonic() - start,
            ) from e
        except HTTPConnectionError as e:
            # A mid-stream read failure from httpx surfaces here.
            self._log_failure("down", e, context, start)
            raise BackendDown(self._provider_key) from e
        except BackendError:
            # Already typed — re-raise unchanged.
            raise
        except HTTPError as e:
            self._log_failure("generic_http_error", e, context, start)
            err = BackendError(self._provider_key, status=e.status, fix_hint=e.fix_hint)
            raise err from e

        content = "".join(text_parts).strip()
        if not content:
            # Empty stream with no error is still a failure for this
            # backend's strict contract. Cloud backends treat this as
            # "try next provider"; we surface it the same way via
            # BackendDown so the router falls over.
            raise BackendDown(
                self._provider_key,
                fix_hint="Peer streamed zero content (upstream likely broken)",
            )

        uncached = max(0, input_tokens - cached) if input_tokens else 0
        elapsed_ms = (time.monotonic() - start) * 1000
        log_structured(
            logger,
            logging.DEBUG,
            event="peer_stream_complete",
            data={
                "provider": self._provider_key,
                "model": model,
                "request_id": context.request_id,
                "chunks": chunks_seen,
                "latency_ms": round(elapsed_ms, 1),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            latency_ms=elapsed_ms,
            provider=self._provider_key,
            stop_reason=stop_reason,
            cached_input_tokens=cached,
            uncached_input_tokens=uncached,
        )

    @staticmethod
    def _parse_queue_depth(headers: Mapping[str, str] | None) -> int:
        if not headers:
            return 0
        raw = headers.get("X-Maxim-Queue-Depth") or headers.get("x-maxim-queue-depth")
        if not raw:
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def _log_success(
        self,
        resp: LLMResponse,
        context: RequestContext,
        start: float,
    ) -> None:
        level = logging.INFO if _backend_trace_enabled() else logging.DEBUG
        log_structured(
            logger,
            level,
            event="peer_backend_call",
            data={
                "provider": self._provider_key,
                "model": resp.model,
                "status": 200,
                "latency_ms": round(resp.latency_ms, 1),
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "request_id": context.request_id,
                "agent_id": context.agent_id,
                "session_id": context.session_id,
                "lane": context.lane,
            },
        )

    def _log_failure(
        self,
        outcome: str,
        exc: Exception,
        context: RequestContext,
        start: float,
    ) -> None:
        elapsed_ms = (time.monotonic() - start) * 1000
        log_structured(
            logger,
            logging.WARNING,
            event="peer_backend_failed",
            data={
                "provider": self._provider_key,
                "error": type(exc).__name__,
                "outcome": outcome,
                "status": getattr(exc, "status", None),
                "fix_hint": getattr(exc, "fix_hint", "") or str(exc),
                "latency_ms": round(elapsed_ms, 1),
                "request_id": context.request_id,
                "agent_id": context.agent_id,
                "session_id": context.session_id,
                "lane": context.lane,
            },
        )


__all__ = ["_MaximPeerBackend"]
