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

Design principles (see docs/plans/archive/llm_path_fast_failover.md for full
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
import socket
import ssl
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


# Stall-detector integration: register_byte_received is a registry probe
# called from the stream loop on every chunk arrival. Hoisted out of the
# loop (architecture review I5) and wrapped with a warn-once shim so a
# silently broken instrumentation surfaces in logs instead of silently
# re-introducing the bug class llm_call_registry exists to close.
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


# ─── Probe primitives (moved from runtime/llm_server.py) ─────────────────
#
# These are the actual probe implementations. ``health_check()`` calls them
# directly. Previously they lived in ``llm_server.py`` with lazy imports in
# both directions to avoid circular deps; moving them here eliminates the
# cycle entirely. ``ProbeResult`` stays in ``llm_server.py`` as the public
# type used by callers outside ``models/language/``.


def _build_probe_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base + "/models" if base.endswith("/v1") else base + "/v1/models"


def _classify_probe_cause(exc: BaseException, fallback_detail: str) -> str:
    """Walk the ``__cause__`` chain looking for a socket-level error class."""
    cur: BaseException | None = exc
    seen = 0
    while cur is not None and seen < 6:
        if isinstance(cur, socket.gaierror):
            return "dns_fail"
        if isinstance(cur, ssl.SSLError):
            return "tls_error"
        if isinstance(cur, TimeoutError) or isinstance(cur, socket.timeout):
            return "timeout"
        if isinstance(cur, (ConnectionRefusedError, ConnectionResetError)):
            return "connection_refused"
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return "other"


def _probe_once(url: str, api_key: str | None, timeout_s: float):
    """Single liveness probe attempt — ``GET /v1/models``.

    Specific-before-general ``except`` ordering — ``HTTPAuthError``
    before ``HTTPError`` — guarantees auth rejection is classified as
    ``auth_rejected`` rather than ``other``.
    """
    from maxim.runtime.llm_server import ProbeResult

    probe_url = _build_probe_url(url)
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.monotonic()
    try:
        resp = _http.fetch_url(
            probe_url,
            method="GET",
            headers=headers,
            timeout=_http.TimeoutPolicy(
                connect_s=min(timeout_s, 2.0),
                read_s=timeout_s,
                total_s=timeout_s + 1.0,
            ),
        )
    except _http.HTTPAuthError as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(url, "auth_rejected", f"HTTP {e.status}", round(latency_ms, 1))
    except _http.HTTPServerError as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(url, "http_5xx", f"HTTP {e.status}", round(latency_ms, 1))
    except _http.HTTPClientError as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(url, "other", f"HTTP {e.status}", round(latency_ms, 1))
    except _http.HTTPRateLimited as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(url, "other", f"HTTP {e.status}", round(latency_ms, 1))
    except _http.HTTPTimeout:
        return ProbeResult(url, "timeout", f"{timeout_s}s", None)
    except _http.HTTPConnectionError as e:
        outcome = _classify_probe_cause(e, "connection failure")
        return ProbeResult(url, outcome, e.fix_hint or str(e), None)
    except _http.HTTPError as e:
        return ProbeResult(url, "other", f"{type(e).__name__}: {e.fix_hint}", None)
    except Exception as e:  # noqa: BLE001 — defensive catch-all
        return ProbeResult(url, "other", f"{type(e).__name__}: {e}", None)

    latency_ms = (time.monotonic() - start) * 1000
    return ProbeResult(url, "ok", f"HTTP {resp.status}", round(latency_ms, 1))


def _probe_stage2_readiness(
    url: str,
    api_key: str | None,
    model_name: str,
    timeout_s: float = 3.0,
):
    """Stage-2 readiness probe: micro-completion via ``POST /v1/chat/completions``."""
    from maxim.runtime.llm_server import ProbeResult

    base = url.rstrip("/")
    chat_url = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model_name or "default",
        "messages": [{"role": "user", "content": "."}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    start = time.monotonic()
    try:
        resp = _http.fetch_url(
            chat_url,
            method="POST",
            headers=headers,
            json=body,
            timeout=_http.TimeoutPolicy(
                connect_s=min(timeout_s, 2.0),
                read_s=timeout_s,
                total_s=timeout_s + 1.0,
            ),
        )
    except _http.HTTPAuthError as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(url, "auth_rejected", f"stage2 HTTP {e.status}", round(latency_ms, 1))
    except _http.HTTPRateLimited as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(url, "other", f"stage2 HTTP {e.status} (rate limited)", round(latency_ms, 1))
    except (_http.HTTPServerError, _http.HTTPClientError, _http.HTTPTimeout) as e:
        latency_ms = (time.monotonic() - start) * 1000
        return ProbeResult(
            url, "inference_broken", f"stage2: {type(e).__name__} {getattr(e, 'status', '')}", round(latency_ms, 1)
        )
    except _http.HTTPConnectionError as e:
        return ProbeResult(url, "inference_broken", f"stage2: {e.fix_hint or str(e)}", None)
    except _http.HTTPError as e:
        return ProbeResult(url, "inference_broken", f"stage2: {type(e).__name__}", None)
    except Exception as e:  # noqa: BLE001 — fallback safety
        log_structured(
            logger,
            logging.WARNING,
            event="probe_stage2_fallback",
            data={"endpoint": url, "reason": f"{type(e).__name__}: {e}"},
        )
        return ProbeResult(url, "ok", "stage1-ok (stage2 fallback)", None)

    latency_ms = (time.monotonic() - start) * 1000
    if resp.status == 200:
        return ProbeResult(url, "ok", "stage2 HTTP 200", round(latency_ms, 1))
    return ProbeResult(url, "inference_broken", f"stage2 HTTP {resp.status}", round(latency_ms, 1))


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
    # Plan 4 A.1: declare to the router that this backend wants the
    # request_context kwarg forwarded. Cloud backends
    # (_OpenAIBackend, _AnthropicBackend) do not set this flag, so the
    # router's capability-flag forwarding pattern only wires the kwarg
    # for self-hosted peer calls — avoids "unexpected keyword argument"
    # on cloud backends that don't accept it.
    supports_request_context = True

    def __init__(self, cfg: LLMConfig, provider_key: str = "maxim_peer") -> None:
        self.cfg = cfg
        self._provider_key = provider_key
        self._endpoint_name = f"peer-{provider_key}"
        self._endpoint_registered = False
        # R3 review fix: ``_MaximPeerBackend.for_url`` previously stored
        # the probe API key in ``os.environ["MAXIM_PEER_PROBE_KEY"]``
        # which races under concurrent probes (lane-A's key leaks into
        # lane-B's request). Instance-level override consulted by
        # ``_get_api_key`` first, which keeps the factory safe to call
        # from parallel code paths.
        self._api_key_override: str | None = None
        # Ground-truth model name as reported by the leader's ``/v1/models``
        # endpoint. Populated lazily on first successful call. llama-cpp-server
        # echoes the request's ``model`` field back unchanged in
        # ``/v1/chat/completions`` responses, so without this we'd surface the
        # requested name (e.g. ``claude-sonnet-4-6``) as the served name —
        # corrupting display AND cost tracking (CostTracker would apply
        # Anthropic Sonnet pricing for local Qwen inference). The discovery
        # call hits a registry-cached endpoint and never raises; failure leaves
        # the field empty and falls back to existing behaviour.
        self._served_model: str = ""
        # One-shot: don't re-probe ``/v1/models`` on every response if the
        # first attempt failed (avoids repeated network calls per request on
        # a leader that doesn't expose ``/v1/models``).
        self._served_model_discovery_attempted: bool = False

    # ─── Interface methods ──────────────────────────────────────────────

    def warmup(self) -> bool:
        """Validate auth + base URL + register the HTTP endpoint.

        Calling ``_ensure_endpoint_registered()`` here (rather than deferring
        it to the first ``complete_with_usage()`` call) surfaces DNS failures
        at startup where they're expected and visible, instead of on the first
        user-facing inference request where they produce a silent
        ``dispatch_exhausted`` with ``total_elapsed_ms~0.2ms``.  The
        ``_endpoint_registered`` flag ensures subsequent calls are a no-op.
        """
        if not self._get_api_key():
            return False
        if not self._ensure_endpoint_registered():
            return False
        self._try_discover_served_model()
        return True

    @property
    def served_model(self) -> str:
        """The model the leader is actually serving (from ``/v1/models``).

        Empty string until first successful discovery. Treat as advisory:
        callers fall back to the requested model name when this is empty.
        """
        return self._served_model

    def _try_discover_served_model(self) -> None:
        """Probe ``/v1/models`` once and cache ``data[0]['id']``.

        Best effort — any failure silently leaves ``_served_model`` empty so
        the existing requested-model-name path stays the fallback. Discovery
        is one-shot per backend instance (success or failure both flip the
        ``_served_model_discovery_attempted`` flag so we don't re-probe on
        every response); callers that need re-resolution after a leader model
        swap should construct a fresh backend (which happens on every router
        reload).
        """
        if self._served_model or self._served_model_discovery_attempted:
            return
        self._served_model_discovery_attempted = True
        base = self._resolve_base_url()
        if not base:
            return
        probe_url = _build_probe_url(base)
        headers: dict[str, str] = {}
        api_key = self._get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            resp = _http.fetch_url(
                probe_url,
                method="GET",
                headers=headers,
                timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=3.0, total_s=4.0),
            )
            body = json.loads(resp.content or b"{}")
        except Exception:  # noqa: BLE001 — discovery is best-effort
            return
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or not data:
            return
        first = data[0]
        if not isinstance(first, dict):
            return
        served = first.get("id")
        if isinstance(served, str) and served:
            self._served_model = served

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

        # Plan 3.5 R4: cooperative cancellation checkpoint. If the
        # agent-level timeout in LLMWorker has fired while we were
        # waiting to execute, raise BackendDown immediately so the
        # router unwinds _inference_lock cleanly without queuing
        # another HTTP call.
        from maxim.utils.cancellation import is_cancelled

        if is_cancelled():
            raise BackendDown(
                self._provider_key,
                fix_hint="Request cancelled by caller before peer call",
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

        try:
            parsed = self._parse_llm_response(raw, model=model, start=t0)
        except BackendInferenceBroken as e:
            # _parse_llm_response raises for empty choices (context overflow,
            # model error, truncated body).  _log_failure must be called here
            # because _parse_llm_response has no access to context/t0 and
            # the exception propagates past the HTTP-error handlers above
            # without triggering any peer_backend_failed event otherwise.
            # This ensures the fix_hint ("prompt exceeded context length",
            # "non-JSON body", etc.) appears in JSONL alongside the http_request
            # so operators can diagnose without cross-referencing router logs.
            self._log_failure("inference_broken", e, context, t0)
            raise
        self._log_success(parsed, context, t0)
        return parsed

    def health_check(
        self,
        *,
        first_timeout_s: float = 0.8,
        retry_timeout_s: float = 2.5,
        enable_stage2: bool = True,
    ):
        """Two-stage probe. Returns a
        :class:`maxim.runtime.llm_server.ProbeResult`.

        Stage 1 (liveness): ``GET /v1/models`` with an aggressive first
        timeout. On any unreachable outcome, one re-attempt with a
        longer budget catches slow-starting leaders without slowing the
        happy path. Short-circuits on ``ok`` and ``auth_rejected`` —
        both mean the HTTP listener is alive.

        Stage 2 (readiness): a 1-token ``POST /v1/chat/completions`` to
        confirm the model is actually loaded and the chat endpoint is
        wired. Opt-out via ``enable_stage2=False`` for callers that only
        need liveness.

        **Canonical probe entry point** for all Maxim peer URLs. Probe
        primitives (``_probe_once``, ``_probe_stage2_readiness``) live in
        this module.
        """
        import uuid

        from maxim.runtime.llm_server import ProbeResult

        # Probe path uses the raw base_url WITHOUT the SSRF check —
        # callers have already vetted the URL (operator typed it into
        # peer.yml, or it came from a trusted tier config). The SSRF
        # check only applies to actual LLM calls via
        # :meth:`_ensure_endpoint_registered`, which is reached via
        # :meth:`complete_with_usage` — not via :meth:`health_check`.
        # This keeps the probe surface compatible with test mocks that
        # patch ``_probe_once`` on non-DNS-resolvable fake hostnames.
        base_url = self._raw_base_url()
        if not base_url:
            return ProbeResult(
                url="",
                outcome="other",
                detail="missing base_url",
                latency_ms=None,
            )
        api_key = self._get_api_key() or None
        model_name = self._resolve_model(None)

        probe_id = uuid.uuid4().hex[:8]
        log_structured(
            logger,
            logging.INFO,
            event="probe_started",
            data={
                "probe_id": probe_id,
                "endpoint": base_url,
                "stage": "liveness",
                "cached": False,
                "provider": self._provider_key,
            },
        )

        stage1_start = time.monotonic()
        result = _probe_once(base_url, api_key, first_timeout_s)
        if not result.is_reachable:
            result = _probe_once(base_url, api_key, retry_timeout_s)
        stage1_ms = round((time.monotonic() - stage1_start) * 1000, 1)

        if enable_stage2 and result.outcome == "ok":
            stage2 = _probe_stage2_readiness(base_url, api_key, model_name)
            log_structured(
                logger,
                logging.INFO,
                event="probe_completed",
                data={
                    "probe_id": probe_id,
                    "endpoint": base_url,
                    "stage": "both",
                    "outcome": stage2.outcome,
                    "liveness_ms": stage1_ms,
                    "readiness_ms": stage2.latency_ms,
                    "provider": self._provider_key,
                },
            )
            return stage2

        log_structured(
            logger,
            logging.INFO,
            event="probe_completed",
            data={
                "probe_id": probe_id,
                "endpoint": base_url,
                "stage": "liveness",
                "outcome": result.outcome,
                "liveness_ms": stage1_ms,
                "readiness_ms": None,
                "provider": self._provider_key,
            },
        )
        return result

    @classmethod
    def for_url(
        cls,
        url: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> "_MaximPeerBackend":
        """Lightweight factory for callers that only need
        :meth:`health_check` against a URL (no real LLMConfig).

        Used by :mod:`maxim.runtime.lane_backends._validate_remote_urls`
        and :mod:`maxim.doctor.checks` — places that probe a peer URL
        without holding a full router-scoped config.

        **Concurrency-safe:** R3 review fix. The earlier implementation
        wrote ``MAXIM_PEER_PROBE_KEY`` into ``os.environ`` which races
        across concurrent probes (lane-A's key leaked into lane-B's
        request under parallel validation). This version captures the
        key on the returned instance's ``_api_key_override`` so
        ``_get_api_key`` returns the correct value per-instance without
        touching global state. Safe to call from threads / doctor
        re-probe loops / future parallel validation paths.
        """
        import dataclasses as _dc

        cfg = _dc.replace(
            LLMConfig(),
            providers={
                "probe": {
                    "type": "maxim_peer",
                    "base_url": url,
                    # ``api_key_env`` is a fallback only; the real key is
                    # stored on the instance via ``_api_key_override``.
                    "api_key_env": "",
                    "model": model or "default",
                    "allow_local_endpoints": True,
                    "pricing_required": False,
                },
            },
        )
        instance = cls(cfg, provider_key="probe")
        instance._api_key_override = api_key
        return instance

    # ─── Private helpers ────────────────────────────────────────────────

    def _provider_cfg(self) -> dict[str, Any]:
        providers = getattr(self.cfg, "providers", {}) or {}
        raw = providers.get(self._provider_key, {})
        return raw if isinstance(raw, dict) else {}

    def _get_api_key(self) -> str:
        # Per-instance override takes precedence so ``for_url(api_key=k)``
        # is safe under concurrent probes (no env-var race).
        if self._api_key_override is not None:
            return self._api_key_override
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
            total = float(cfg.get("timeout_s", 300.0))
        except (TypeError, ValueError):
            total = 300.0
        return TimeoutPolicy(
            connect_s=3.0,
            read_s=total,
            total_s=total + 60.0,
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
        """Parse an OpenAI-compatible chat completion body.

        Raises :class:`BackendInferenceBroken` when ``choices`` is empty —
        this surfaces as a typed router exception so the provider gets a
        15-second cooldown window and the router tries the next provider.
        Returning an empty :class:`LLMResponse` silently would bypass the
        typed exception path entirely (no cooldown, no router-level
        re-dispatch, empty content propagates to the agent).
        """
        # Plan 3.5 R6 review: cancellation checkpoint AFTER the HTTP call
        # returned but BEFORE we do JSON parsing + return success. Closes
        # the race window in non-streaming where cancellation fired
        # mid-HTTP — the orphan completed the call but the caller already
        # gave up. Bailing here means we don't write success bookkeeping
        # via _note_provider_success for an abandoned request.
        from maxim.utils.cancellation import is_cancelled

        if is_cancelled():
            raise BackendDown(
                self._provider_key,
                fix_hint="Request cancelled by caller after peer call returned",
            )

        choices = raw.get("choices") or []
        if not choices:
            raise BackendInferenceBroken(
                self._provider_key,
                fix_hint=(
                    "Peer returned a completion with no choices — "
                    "model may be overloaded or the prompt exceeded context length"
                ),
            )
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

        # Served-model substitution: llama-cpp-server echoes the request's
        # ``model`` field unchanged, so ``raw.get("model")`` is NOT
        # authoritative. Prefer the leader-discovered ``_served_model``
        # (populated once during ``warmup()`` via ``/v1/models``) when
        # available; otherwise fall back to the upstream's echo (which may
        # come from a different upstream that doesn't echo), then to the
        # requested name. The ``EXACTLY one HTTP call`` invariant is
        # preserved — discovery happens in ``warmup()``, not here.
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._served_model or str(raw.get("model") or model),
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
                    # Stall-detector integration: update the registry's
                    # last_byte_at on EVERY arriving line, including PR #320
                    # TTFT keepalive frames (``: keepalive``) which are SSE
                    # comments filtered out by the data-line check below.
                    # The orchestrator's stall detector reads
                    # ``oldest_byte_silence_s`` to distinguish "call alive
                    # but slow" from "connection wedged" — keepalive bytes
                    # are exactly what make this distinction work.
                    # Module-level import was hoisted out of the loop and
                    # the silent-swallow is replaced by a warn-once flag
                    # (architecture review I5) so a silently-broken
                    # instrumentation surfaces in logs instead of
                    # re-introducing the bug class this module exists
                    # to close.
                    _register_byte_received_safe()
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
                    except Exception as parse_exc:
                        # A malformed chunk mid-stream is a broken peer.
                        # Do NOT silently accept partial output. R3
                        # review fix: emit a per-call
                        # ``peer_backend_failed`` WARN before raising so
                        # the operator sees the malformed-chunk signal
                        # in the per-call log, not just in the router's
                        # aggregated ``dispatch_exhausted`` line.
                        err = BackendDown(
                            self._provider_key,
                            fix_hint=(
                                f"Peer streamed non-JSON chunk "
                                f"(broken upstream, model={model}): "
                                f"{type(parse_exc).__name__}"
                            ),
                        )
                        self._log_failure("malformed_chunk", err, context, start)
                        raise err
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
                    # Plan 3.5 R4: cooperative cancellation checkpoint
                    # between SSE chunks. The agent-level timeout in
                    # LLMWorker may have fired while we were generating;
                    # unwind now rather than accumulating more partial
                    # output that will be discarded.
                    from maxim.utils.cancellation import is_cancelled

                    if is_cancelled():
                        raise BackendDown(
                            self._provider_key,
                            fix_hint="Request cancelled by caller during stream",
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
            # BackendDown so the router falls over. R3 review fix: emit
            # a per-call WARN before raising so the empty-stream case
            # is visible outside the aggregated dispatch WARN.
            err = BackendDown(
                self._provider_key,
                fix_hint=f"Peer streamed zero content (upstream likely broken, model={model})",
            )
            self._log_failure("empty_stream", err, context, start)
            raise err

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
                # CC12: standard-name token telemetry on JSONL emissions.
                "cached_tokens": cached,
            },
        )

        # Served-model substitution: streaming chunks don't carry an
        # authoritative model name (chunks just echo the request payload).
        # Prefer leader-discovered ``_served_model`` (populated once during
        # ``warmup()`` via ``/v1/models``) so display + cost tracking see
        # ground truth instead of the requested name. The ``EXACTLY one
        # HTTP call`` invariant is preserved — discovery happens in
        # ``warmup()``, not here.
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._served_model or model,
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
                # CC12: standard-name token telemetry on JSONL emissions.
                "cached_tokens": resp.cached_input_tokens,
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
