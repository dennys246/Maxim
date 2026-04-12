"""Unified HTTP client for Maxim (Plan 1 R1).

Replaces 11 scattered urllib call sites with a single registry-backed module.
Key properties:

- **Registry pattern**: endpoints register once with headers, auth, timeouts,
  connection pool limits. Call sites reference them by name. The Cloudflare
  User-Agent incident (commit 8b52cbd on 2026-04-12) becomes structurally
  impossible because User-Agent is set once at registration.

- **Typed error taxonomy**: every HTTP failure raises a specific subclass of
  :class:`HTTPError` with a ``fix_hint``. No more string-matching on
  exception messages.

- **Multi-agent context propagation**: :class:`RequestContext` is set via
  :data:`contextvars.ContextVar` at the request boundary and automatically
  flows into ``X-Maxim-*`` headers on every outbound call to *internal*
  endpoints. External endpoints (user tools, HuggingFace downloads) do not
  leak request IDs.

- **Dual-format logging**: human-readable stdout is unchanged. When
  ``MAXIM_LOG_FILE`` is set, a JSONL file handler backed by
  :class:`maxim.utils.structured_logging.StructuredFormatter` captures
  structured events. Wired via :func:`maxim.utils.logging.configure_logging`.

- **httpx backend**: already a transitive dep via the openai SDK. Offers
  better error taxonomy, native connection pooling, and streaming support
  than raw urllib.

This module is load-bearing for Plans 2, 3, and 4. Do not add
``urllib.request`` call sites in ``src/maxim/`` — the CI grep enforces this.
"""

from __future__ import annotations

import json as _json
import logging
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import httpx

from maxim.utils.structured_logging import log_structured

logger = logging.getLogger(__name__)

# ─────────────────────────── Named constants ────────────────────────────

MAX_HEADER_VALUE_LEN = 256
"""Upper bound on sanitized header values. Prevents log injection + DoS
via absurdly long headers from user-controlled content."""

DEFAULT_POOL_PER_ENDPOINT = 10
"""Per-endpoint connection pool ceiling. Concurrent-dispatch ceiling across
a node = N peers × DEFAULT_POOL_PER_ENDPOINT. Metric
``http_pool_exhausted_total`` fires when a call waits for a pool slot."""

PROTOCOL_VERSION = "1"
"""Wire protocol version for ``X-Maxim-Protocol-Version`` header. Bump when
the X-Maxim-* header contract changes in a breaking way."""

DEFAULT_USER_AGENT = "maxim-peer/1.0"
"""Default User-Agent for outbound calls. Set once at endpoint registration.
The 2026-04-12 Cloudflare Bot Fight Mode incident was caused by missing
this header in one call site."""


# ─────────────────────────── Typed exceptions ───────────────────────────


class HTTPError(Exception):
    """Base class for all HTTP failures raised by this module.

    Every subclass sets a ``fix_hint`` with actionable guidance. Catching
    ``HTTPError`` broadly is fine; catching bare ``Exception`` is not.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        status: int | None = None,
        fix_hint: str = "",
        message: str = "",
    ) -> None:
        self.endpoint = endpoint
        self.status = status
        self.fix_hint = fix_hint
        self.response: Response | None = None
        super().__init__(message or f"{type(self).__name__}[{endpoint}]: {fix_hint}")


class HTTPTimeout(HTTPError):
    """Connect or read timeout exceeded the configured TimeoutPolicy."""


class HTTPConnectionError(HTTPError):
    """Low-level connection failure (DNS, TCP reset, pool exhaustion, TLS)."""


class HTTPAuthError(HTTPError):
    """401 or 403 — the server rejected our credentials."""


class HTTPServerError(HTTPError):
    """5xx — the server had an internal error."""


class HTTPClientError(HTTPError):
    """4xx (non-auth, non-rate-limit) — malformed request or header sanitation failure."""


class HTTPRateLimited(HTTPError):
    """429 — server is rate-limiting. Carries ``retry_after_s`` + optional
    ``suggested_peer`` hint (for the deferred multi-peer dispatch plan)."""

    def __init__(
        self,
        endpoint: str,
        *,
        status: int | None = 429,
        retry_after_s: float = 0.0,
        suggested_peer: str | None = None,
        fix_hint: str = "",
    ) -> None:
        self.retry_after_s = retry_after_s
        self.suggested_peer = suggested_peer
        super().__init__(endpoint, status=status, fix_hint=fix_hint)


# ─────────────────────────── Config dataclasses ─────────────────────────


@dataclass(frozen=True)
class TimeoutPolicy:
    """Layered timeouts — named fields, no single ``timeout`` kwarg.

    ``connect_s`` bounds DNS + TCP handshake. ``read_s`` bounds per-chunk
    read latency. ``total_s`` is the pool acquisition + total request
    budget. Use the provided factory helpers for common cases.
    """

    connect_s: float = 3.0
    read_s: float = 30.0
    total_s: float = 60.0

    def to_httpx(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect_s,
            read=self.read_s,
            write=self.read_s,
            pool=self.total_s,
        )

    @classmethod
    def fast(cls) -> TimeoutPolicy:
        """Short budget — use for probes + health checks."""
        return cls(connect_s=1.0, read_s=2.0, total_s=5.0)

    @classmethod
    def long(cls) -> TimeoutPolicy:
        """Extended budget — use for inference + large downloads."""
        return cls(connect_s=5.0, read_s=120.0, total_s=600.0)


@dataclass(frozen=True)
class HTTPEndpoint:
    """A named HTTP endpoint registered once, referenced many times.

    ``internal=True`` endpoints receive ``X-Maxim-*`` header propagation.
    ``internal=False`` endpoints (user tools, third-party downloads) do not
    — this prevents leaking request IDs / agent IDs to external services.
    """

    name: str
    base_url: str | None
    default_headers: Mapping[str, str] = field(default_factory=dict)
    auth_provider: Callable[[], str | None] | None = None
    timeouts: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    max_pool_connections: int = DEFAULT_POOL_PER_ENDPOINT
    internal: bool = True


# ─────────────────────────── RequestContext ─────────────────────────────


@dataclass(frozen=True)
class RequestContext:
    """Multi-agent routing + observability bundle.

    Set via :func:`set_context` at the request boundary. Read automatically
    by :func:`_build_headers` and the logging formatter. Callers don't have
    to thread this through function signatures — use the contextvar.
    """

    request_id: str
    agent_id: str | None = None
    session_id: str | None = None
    lane: str | None = None
    parent_request_id: str | None = None


_current_context: ContextVar[RequestContext | None] = ContextVar(
    "maxim_request_context",
    default=None,
)


def generate_request_id() -> str:
    """Re-exports :func:`maxim.models.language.mesh_trace.generate_request_id`.

    Lazy import to avoid a circular dependency at module init.
    """
    from maxim.models.language.mesh_trace import generate_request_id as _gen

    return _gen()


def new_request_context(
    *,
    request_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    lane: str | None = None,
    parent_request_id: str | None = None,
) -> RequestContext:
    """Build a RequestContext, auto-generating request_id if absent."""
    return RequestContext(
        request_id=request_id or generate_request_id(),
        agent_id=agent_id,
        session_id=session_id,
        lane=lane,
        parent_request_id=parent_request_id,
    )


def set_context(ctx: RequestContext) -> Any:
    """Bind a context for the current async/sync scope. Returns the token
    produced by :meth:`contextvars.ContextVar.set` so callers can restore."""
    return _current_context.set(ctx)


def reset_context(token: Any) -> None:
    _current_context.reset(token)


def current_context() -> RequestContext | None:
    return _current_context.get()


class _ContextScope:
    """Context manager that binds and restores a RequestContext."""

    def __init__(self, ctx: RequestContext) -> None:
        self._ctx = ctx
        self._token: Any = None

    def __enter__(self) -> RequestContext:
        self._token = set_context(self._ctx)
        return self._ctx

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._token is not None:
            reset_context(self._token)


def context_scope(ctx: RequestContext) -> _ContextScope:
    """Usage: ``with http.context_scope(ctx): http.get(...)``."""
    return _ContextScope(ctx)


# ─────────────────────────── Metrics ────────────────────────────────────


class HttpMetrics:
    """HTTP client metrics — sibling to ``lane_metrics.MetricsRegistry``.

    Endpoint-scoped (not lane-scoped), thread-safe, in-memory only. Snapshot
    via :func:`metrics_snapshot` for doctor / admin API consumption.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests_total: dict[tuple[str, str], int] = {}
        self._latencies_ms: dict[str, list[float]] = {}
        self._pool_exhausted_total: dict[str, int] = {}
        self._header_rejected_total: dict[str, int] = {}
        self._startup_phases: dict[str, float] = {}

    def record_request(self, endpoint: str, status: int | str, latency_ms: float) -> None:
        with self._lock:
            key = (endpoint, str(status))
            self._requests_total[key] = self._requests_total.get(key, 0) + 1
            bucket = self._latencies_ms.setdefault(endpoint, [])
            bucket.append(latency_ms)
            if len(bucket) > 200:
                del bucket[: len(bucket) - 200]

    def record_header_rejected(self, field_name: str) -> None:
        with self._lock:
            self._header_rejected_total[field_name] = self._header_rejected_total.get(field_name, 0) + 1

    def record_pool_exhausted(self, endpoint: str) -> None:
        with self._lock:
            self._pool_exhausted_total[endpoint] = self._pool_exhausted_total.get(endpoint, 0) + 1

    def record_startup_phase(self, phase: str, duration_ms: float) -> None:
        with self._lock:
            self._startup_phases[phase] = duration_ms

    def snapshot(self) -> dict[str, Any]:
        with self._lock:

            def pct(vals: list[float], p: float) -> float:
                if not vals:
                    return 0.0
                s = sorted(vals)
                idx = min(len(s) - 1, int(len(s) * p))
                return round(s[idx], 1)

            latency = {
                ep: {
                    "count": len(vals),
                    "p50_ms": pct(vals, 0.5),
                    "p99_ms": pct(vals, 0.99),
                }
                for ep, vals in self._latencies_ms.items()
            }
            return {
                "requests_total": {f"{ep}/{st}": c for (ep, st), c in self._requests_total.items()},
                "latency": latency,
                "pool_exhausted_total": dict(self._pool_exhausted_total),
                "header_rejected_total": dict(self._header_rejected_total),
                "startup_phases": dict(self._startup_phases),
            }


_metrics = HttpMetrics()


def metrics_snapshot() -> dict[str, Any]:
    """Public snapshot of HTTP metrics — consumed by doctor + admin API."""
    return _metrics.snapshot()


def record_startup_phase(phase: str, duration_ms: float) -> None:
    """Record a named startup phase duration. Emitted as a structured event
    so ``jq 'select(.event=="startup_phase")'`` on the JSONL log reveals
    where cold-start time is spent."""
    _metrics.record_startup_phase(phase, duration_ms)
    log_structured(
        logger,
        logging.INFO,
        event="startup_phase",
        data={"phase": phase, "duration_ms": round(duration_ms, 1)},
    )


# ─────────────────────────── Header sanitization ────────────────────────


def _sanitize_header_value(value: Any, field_name: str = "") -> str:
    """Reject headers that could enable log injection or HTTP smuggling.

    Rejects: CR/LF, control characters, non-ASCII bytes, values exceeding
    :data:`MAX_HEADER_VALUE_LEN`. Raises :class:`HTTPClientError` with a
    ``fix_hint`` pointing at the offending field.
    """
    if value is None:
        _metrics.record_header_rejected(field_name or "unknown")
        raise HTTPClientError(
            field_name or "<header>",
            fix_hint=f"Header '{field_name}' is None",
        )
    if not isinstance(value, str):
        value = str(value)
    if len(value) > MAX_HEADER_VALUE_LEN:
        _metrics.record_header_rejected(field_name or "unknown")
        raise HTTPClientError(
            field_name or "<header>",
            fix_hint=(f"Header '{field_name}' too long: {len(value)} > {MAX_HEADER_VALUE_LEN}"),
        )
    for ch in value:
        o = ord(ch)
        if ch in ("\r", "\n"):
            _metrics.record_header_rejected(field_name or "unknown")
            raise HTTPClientError(
                field_name or "<header>",
                fix_hint=f"Header '{field_name}' contains CR/LF (log injection risk)",
            )
        if o < 0x20 or o == 0x7F:
            _metrics.record_header_rejected(field_name or "unknown")
            raise HTTPClientError(
                field_name or "<header>",
                fix_hint=f"Header '{field_name}' contains control character (0x{o:02x})",
            )
        if o > 0x7E:
            _metrics.record_header_rejected(field_name or "unknown")
            raise HTTPClientError(
                field_name or "<header>",
                fix_hint=f"Header '{field_name}' contains non-ASCII byte",
            )
    return value


# ─────────────────────────── Registry ───────────────────────────────────


class _EndpointRegistry:
    """Thread-safe store of registered endpoints + their httpx clients."""

    def __init__(self) -> None:
        self._endpoints: dict[str, HTTPEndpoint] = {}
        self._clients: dict[str, httpx.Client] = {}
        self._lock = threading.RLock()

    def register(self, endpoint: HTTPEndpoint) -> None:
        with self._lock:
            old = self._clients.pop(endpoint.name, None)
            if old is not None:
                try:
                    old.close()
                except Exception:  # pragma: no cover — best effort
                    pass
            self._endpoints[endpoint.name] = endpoint

    def get(self, name: str) -> HTTPEndpoint:
        with self._lock:
            ep = self._endpoints.get(name)
            if ep is None:
                raise KeyError(f"HTTP endpoint '{name}' not registered — call http.register_endpoint() first")
            return ep

    def get_client(self, name: str) -> httpx.Client:
        with self._lock:
            if name not in self._clients:
                ep = self._endpoints.get(name)
                if ep is None:
                    raise KeyError(f"HTTP endpoint '{name}' not registered")
                limits = httpx.Limits(
                    max_connections=ep.max_pool_connections,
                    max_keepalive_connections=ep.max_pool_connections,
                )
                self._clients[name] = httpx.Client(
                    base_url=ep.base_url or "",
                    timeout=ep.timeouts.to_httpx(),
                    limits=limits,
                )
            return self._clients[name]

    def close_all(self) -> None:
        with self._lock:
            for client in self._clients.values():
                try:
                    client.close()
                except Exception:  # pragma: no cover
                    pass
            self._clients.clear()

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._endpoints.keys())


_registry = _EndpointRegistry()


def register_endpoint(endpoint: HTTPEndpoint) -> None:
    """Register an endpoint. Idempotent — later calls replace earlier ones."""
    _registry.register(endpoint)


def get_endpoint(name: str) -> HTTPEndpoint:
    return _registry.get(name)


def is_registered(name: str) -> bool:
    try:
        _registry.get(name)
        return True
    except KeyError:
        return False


def close_all() -> None:
    """Close every registered client. Called from process shutdown hooks."""
    _registry.close_all()


# ─────────────────────────── External endpoint bootstrap ────────────────

_EXTERNAL_ENDPOINT = "_external"
"""Reserved endpoint name for ad-hoc URL fetches (user tools, HF downloads,
peer-cli remote admin calls). Headers are minimal and non-internal —
no X-Maxim-* propagation."""

_external_registered = False
_external_lock = threading.Lock()


def _ensure_external_endpoint() -> None:
    global _external_registered
    with _external_lock:
        if _external_registered:
            return
        register_endpoint(
            HTTPEndpoint(
                name=_EXTERNAL_ENDPOINT,
                base_url=None,
                default_headers={"User-Agent": DEFAULT_USER_AGENT},
                auth_provider=None,
                timeouts=TimeoutPolicy.long(),
                max_pool_connections=DEFAULT_POOL_PER_ENDPOINT,
                internal=False,
            )
        )
        _external_registered = True


# ─────────────────────────── Response wrapper ───────────────────────────


@dataclass
class Response:
    """Light wrapper around httpx.Response — consistent shape, decoupled
    from the underlying library so callers can swap backends later."""

    status: int
    headers: dict[str, str]
    content: bytes
    elapsed_ms: float
    endpoint: str
    request_id: str

    def json(self) -> Any:
        return _json.loads(self.content)

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


# ─────────────────────────── Header + error plumbing ────────────────────


def _build_headers(
    endpoint: HTTPEndpoint,
    context: RequestContext | None,
    extra: Mapping[str, str] | None,
) -> dict[str, str]:
    """Combine endpoint defaults + context + caller extras. Sanitize all.

    Order: defaults → auth → X-Maxim-* (internal only) → caller overrides.
    Later entries override earlier ones on name collision.
    """
    headers: dict[str, str] = {}
    for k, v in endpoint.default_headers.items():
        headers[k] = _sanitize_header_value(v, k)

    if endpoint.auth_provider is not None:
        try:
            token = endpoint.auth_provider()
        except Exception as e:  # pragma: no cover — defensive
            raise HTTPAuthError(
                endpoint.name,
                fix_hint=f"auth_provider raised {type(e).__name__}: {e}",
            ) from e
        if token:
            headers["Authorization"] = "Bearer " + _sanitize_header_value(token, "Authorization")

    if endpoint.internal:
        ctx = context or current_context() or new_request_context()
        headers["X-Maxim-Request-Id"] = _sanitize_header_value(ctx.request_id, "X-Maxim-Request-Id")
        headers["X-Maxim-Protocol-Version"] = PROTOCOL_VERSION
        if ctx.agent_id:
            headers["X-Maxim-Agent-Id"] = _sanitize_header_value(ctx.agent_id, "X-Maxim-Agent-Id")
        if ctx.session_id:
            headers["X-Maxim-Session-Id"] = _sanitize_header_value(ctx.session_id, "X-Maxim-Session-Id")
        if ctx.lane:
            headers["X-Maxim-Lane"] = _sanitize_header_value(ctx.lane, "X-Maxim-Lane")
        if ctx.parent_request_id:
            headers["X-Maxim-Parent-Request-Id"] = _sanitize_header_value(
                ctx.parent_request_id, "X-Maxim-Parent-Request-Id"
            )

    if extra:
        for k, v in extra.items():
            headers[k] = _sanitize_header_value(v, k)
    return headers


def _classify_status(
    endpoint: str,
    status: int,
    headers: Mapping[str, str],
) -> HTTPError | None:
    if 200 <= status < 400:
        return None
    if status == 429:
        retry_after = 0.0
        ra = headers.get("Retry-After") or headers.get("retry-after") or ""
        if ra:
            try:
                retry_after = float(ra)
            except (ValueError, TypeError):
                pass
        return HTTPRateLimited(
            endpoint,
            status=status,
            retry_after_s=retry_after,
            suggested_peer=headers.get("X-Maxim-Suggested-Peer"),
            fix_hint=f"Rate limited (429), retry_after_s={retry_after}",
        )
    if status in (401, 403):
        return HTTPAuthError(
            endpoint,
            status=status,
            fix_hint=f"Auth rejected ({status}) — check API key / cluster key",
        )
    if 400 <= status < 500:
        return HTTPClientError(
            endpoint,
            status=status,
            fix_hint=f"Client error ({status})",
        )
    if 500 <= status < 600:
        return HTTPServerError(
            endpoint,
            status=status,
            fix_hint=f"Server error ({status}) — upstream may be restarting",
        )
    return HTTPError(
        endpoint,
        status=status,
        fix_hint=f"Unexpected HTTP status: {status}",
    )


def _classify_httpx_error(endpoint: str, exc: BaseException) -> HTTPError:
    if isinstance(exc, httpx.TimeoutException):
        return HTTPTimeout(
            endpoint,
            fix_hint="Request exceeded TimeoutPolicy — widen timeouts or check network",
        )
    if isinstance(exc, httpx.PoolTimeout):
        _metrics.record_pool_exhausted(endpoint)
        return HTTPConnectionError(
            endpoint,
            fix_hint=(
                f"Connection pool exhausted for endpoint '{endpoint}' — "
                "increase max_pool_connections or reduce concurrency"
            ),
        )
    if isinstance(exc, httpx.ConnectError):
        return HTTPConnectionError(
            endpoint,
            fix_hint="Could not connect — check URL + network reachability",
        )
    if isinstance(exc, httpx.RequestError):
        return HTTPConnectionError(
            endpoint,
            fix_hint=f"httpx RequestError: {type(exc).__name__}: {exc}",
        )
    return HTTPError(
        endpoint,
        fix_hint=f"Unexpected HTTP error: {type(exc).__name__}: {exc}",
    )


def _http_trace_enabled() -> bool:
    return os.environ.get("MAXIM_HTTP_TRACE", "").strip() in ("1", "true", "yes")


# ─────────────────────────── Core request API ───────────────────────────


def _resolve_timeout(
    endpoint: HTTPEndpoint,
    timeout: TimeoutPolicy | float | None,
) -> httpx.Timeout:
    if isinstance(timeout, TimeoutPolicy):
        return timeout.to_httpx()
    if isinstance(timeout, (int, float)):
        return httpx.Timeout(float(timeout))
    return endpoint.timeouts.to_httpx()


def request(
    method: str,
    endpoint_name: str,
    path: str = "",
    *,
    context: RequestContext | None = None,
    headers: Mapping[str, str] | None = None,
    json: Any = None,
    content: bytes | None = None,
    params: Mapping[str, Any] | None = None,
    timeout: TimeoutPolicy | float | None = None,
) -> Response:
    """Low-level request primitive. Consumers typically call
    :func:`get` / :func:`post`."""
    ep = _registry.get(endpoint_name)
    client = _registry.get_client(endpoint_name)
    final_headers = _build_headers(ep, context, headers)
    used_ctx = context or current_context() or new_request_context()
    timeout_obj = _resolve_timeout(ep, timeout)

    t0 = time.monotonic()
    try:
        resp = client.request(
            method.upper(),
            path,
            headers=final_headers,
            json=json,
            content=content,
            params=params,
            timeout=timeout_obj,
        )
    except httpx.HTTPError as e:
        elapsed_ms = (time.monotonic() - t0) * 1000
        _metrics.record_request(endpoint_name, "error", elapsed_ms)
        log_structured(
            logger,
            logging.WARNING,
            event="http_request_failed",
            data={
                "endpoint": endpoint_name,
                "method": method.upper(),
                "request_id": used_ctx.request_id,
                "error": type(e).__name__,
                "latency_ms": round(elapsed_ms, 1),
            },
        )
        raise _classify_httpx_error(endpoint_name, e) from e

    elapsed_ms = (time.monotonic() - t0) * 1000
    _metrics.record_request(endpoint_name, resp.status_code, elapsed_ms)

    level = logging.INFO if _http_trace_enabled() else logging.DEBUG
    log_structured(
        logger,
        level,
        event="http_request",
        data={
            "endpoint": endpoint_name,
            "method": method.upper(),
            "status": resp.status_code,
            "latency_ms": round(elapsed_ms, 1),
            "request_id": used_ctx.request_id,
            "agent_id": used_ctx.agent_id,
            "session_id": used_ctx.session_id,
            "lane": used_ctx.lane,
        },
    )

    response = Response(
        status=resp.status_code,
        headers=dict(resp.headers),
        content=resp.content,
        elapsed_ms=elapsed_ms,
        endpoint=endpoint_name,
        request_id=used_ctx.request_id,
    )
    err = _classify_status(endpoint_name, resp.status_code, resp.headers)
    if err is not None:
        err.response = response
        raise err
    return response


def get(endpoint_name: str, path: str = "", **kwargs: Any) -> Response:
    return request("GET", endpoint_name, path, **kwargs)


def post(endpoint_name: str, path: str = "", **kwargs: Any) -> Response:
    return request("POST", endpoint_name, path, **kwargs)


# ─────────────────────────── Ad-hoc URL fetch ───────────────────────────


def fetch_url(
    url: str,
    *,
    method: str = "GET",
    context: RequestContext | None = None,
    headers: Mapping[str, str] | None = None,
    content: bytes | None = None,
    json: Any = None,
    timeout: TimeoutPolicy | float | None = None,
) -> Response:
    """One-off fetch of a full URL via the shared ``_external`` endpoint.

    Used by user tools (http_fetch, internet_search), model downloads,
    and peer-cli remote admin calls where the URL isn't known at
    registration time. Does NOT propagate X-Maxim-* headers (external
    endpoint).
    """
    _ensure_external_endpoint()
    ep = _registry.get(_EXTERNAL_ENDPOINT)
    client = _registry.get_client(_EXTERNAL_ENDPOINT)
    final_headers = _build_headers(ep, context, headers)
    used_ctx = context or current_context() or new_request_context()
    timeout_obj = _resolve_timeout(ep, timeout)

    t0 = time.monotonic()
    try:
        resp = client.request(
            method.upper(),
            url,
            headers=final_headers,
            json=json,
            content=content,
            timeout=timeout_obj,
        )
    except httpx.HTTPError as e:
        elapsed_ms = (time.monotonic() - t0) * 1000
        _metrics.record_request(_EXTERNAL_ENDPOINT, "error", elapsed_ms)
        log_structured(
            logger,
            logging.WARNING,
            event="http_request_failed",
            data={
                "endpoint": _EXTERNAL_ENDPOINT,
                "url": url,
                "method": method.upper(),
                "request_id": used_ctx.request_id,
                "error": type(e).__name__,
                "latency_ms": round(elapsed_ms, 1),
            },
        )
        raise _classify_httpx_error(_EXTERNAL_ENDPOINT, e) from e

    elapsed_ms = (time.monotonic() - t0) * 1000
    _metrics.record_request(_EXTERNAL_ENDPOINT, resp.status_code, elapsed_ms)

    level = logging.INFO if _http_trace_enabled() else logging.DEBUG
    log_structured(
        logger,
        level,
        event="http_request",
        data={
            "endpoint": _EXTERNAL_ENDPOINT,
            "url": url,
            "method": method.upper(),
            "status": resp.status_code,
            "latency_ms": round(elapsed_ms, 1),
            "request_id": used_ctx.request_id,
        },
    )

    response = Response(
        status=resp.status_code,
        headers=dict(resp.headers),
        content=resp.content,
        elapsed_ms=elapsed_ms,
        endpoint=_EXTERNAL_ENDPOINT,
        request_id=used_ctx.request_id,
    )
    err = _classify_status(_EXTERNAL_ENDPOINT, resp.status_code, resp.headers)
    if err is not None:
        err.response = response
        raise err
    return response


# ─────────────────────────── Streaming ──────────────────────────────────


@dataclass
class StreamingResponse:
    status: int
    headers: dict[str, str]
    endpoint: str
    request_id: str
    _raw: httpx.Response
    _client: httpx.Client | None = None
    _owns_client: bool = False

    def iter_bytes(self, chunk_size: int = 65536) -> Iterator[bytes]:
        yield from self._raw.iter_bytes(chunk_size=chunk_size)

    def close(self) -> None:
        try:
            self._raw.close()
        except Exception:  # pragma: no cover
            pass
        if self._owns_client and self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self) -> StreamingResponse:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def download_to_file(
    url: str,
    dest_path: Path | str,
    *,
    context: RequestContext | None = None,
    progress_hook: Callable[[int, int, int], None] | None = None,
    timeout: TimeoutPolicy | float | None = None,
    expected_bytes: int | None = None,
    headers: Mapping[str, str] | None = None,
    chunk_size: int = 1 << 16,
) -> int:
    """Stream ``url`` to ``dest_path``. Returns the number of bytes written.

    Atomic semantics mirror the pre-R1 ``urlretrieve`` behavior in
    ``maxim.models.download``: caller is responsible for ``.partial`` tmp
    paths + ``os.replace``. This function writes bytes only.

    ``progress_hook`` matches the legacy ``reporthook`` signature:
    ``hook(block_num, block_size, total_size)``. Called after each chunk.

    ``expected_bytes`` is advisory — if set, the final total is compared
    and :class:`HTTPError` is raised on mismatch with the partial file
    preserved for caller cleanup.
    """
    _ensure_external_endpoint()
    ep = _registry.get(_EXTERNAL_ENDPOINT)
    client = _registry.get_client(_EXTERNAL_ENDPOINT)
    final_headers = _build_headers(ep, context, headers)
    timeout_obj = _resolve_timeout(ep, timeout or TimeoutPolicy.long())

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    block_num = 0
    total_written = 0
    total_size = 0

    try:
        with client.stream(
            "GET",
            url,
            headers=final_headers,
            timeout=timeout_obj,
            follow_redirects=True,
        ) as resp:
            if resp.status_code >= 400:
                err = _classify_status(_EXTERNAL_ENDPOINT, resp.status_code, resp.headers)
                if err is not None:
                    raise err
            cl = resp.headers.get("content-length") or "0"
            try:
                total_size = int(cl)
            except (ValueError, TypeError):
                total_size = 0

            with dest.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    total_written += len(chunk)
                    block_num += 1
                    if progress_hook is not None:
                        try:
                            progress_hook(block_num, len(chunk), total_size)
                        except Exception:  # pragma: no cover — defensive
                            pass
    except httpx.HTTPError as e:
        elapsed_ms = (time.monotonic() - t0) * 1000
        _metrics.record_request(_EXTERNAL_ENDPOINT, "error", elapsed_ms)
        raise _classify_httpx_error(_EXTERNAL_ENDPOINT, e) from e

    elapsed_ms = (time.monotonic() - t0) * 1000
    _metrics.record_request(_EXTERNAL_ENDPOINT, 200, elapsed_ms)

    if expected_bytes is not None and total_written != expected_bytes:
        raise HTTPError(
            _EXTERNAL_ENDPOINT,
            fix_hint=(f"Download size mismatch: got {total_written} bytes, expected {expected_bytes}"),
        )
    return total_written


# ─────────────────────────── Raw proxy forward ──────────────────────────


def raw_proxy_forward(
    url: str,
    method: str,
    *,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float = 60.0,
) -> Response:
    """Escape hatch for :mod:`runtime.leader_proxy` — forwards a request
    with caller-supplied headers verbatim (no sanitization, no X-Maxim-*
    injection) to an upstream URL.

    Used exclusively by the reverse-proxy path where client headers must
    be forwarded unchanged to the upstream llama-cpp-server. Do NOT use
    this for anything else — it bypasses the registry's safety net.
    """
    _ensure_external_endpoint()
    client = _registry.get_client(_EXTERNAL_ENDPOINT)
    t0 = time.monotonic()
    try:
        resp = client.request(
            method.upper(),
            url,
            headers=dict(headers),
            content=body,
            timeout=httpx.Timeout(float(timeout)),
        )
    except httpx.HTTPError as e:
        elapsed_ms = (time.monotonic() - t0) * 1000
        _metrics.record_request(_EXTERNAL_ENDPOINT, "error", elapsed_ms)
        raise _classify_httpx_error(_EXTERNAL_ENDPOINT, e) from e

    elapsed_ms = (time.monotonic() - t0) * 1000
    _metrics.record_request(_EXTERNAL_ENDPOINT, resp.status_code, elapsed_ms)
    return Response(
        status=resp.status_code,
        headers=dict(resp.headers),
        content=resp.content,
        elapsed_ms=elapsed_ms,
        endpoint=_EXTERNAL_ENDPOINT,
        request_id=headers.get("X-Maxim-Request-Id", ""),
    )


__all__ = [
    # Constants
    "MAX_HEADER_VALUE_LEN",
    "DEFAULT_POOL_PER_ENDPOINT",
    "PROTOCOL_VERSION",
    "DEFAULT_USER_AGENT",
    # Types
    "HTTPEndpoint",
    "TimeoutPolicy",
    "RequestContext",
    "Response",
    "StreamingResponse",
    # Exceptions
    "HTTPError",
    "HTTPTimeout",
    "HTTPConnectionError",
    "HTTPAuthError",
    "HTTPServerError",
    "HTTPClientError",
    "HTTPRateLimited",
    # Registry
    "register_endpoint",
    "get_endpoint",
    "is_registered",
    "close_all",
    # Context
    "set_context",
    "reset_context",
    "current_context",
    "new_request_context",
    "context_scope",
    "generate_request_id",
    # Requests
    "request",
    "get",
    "post",
    "fetch_url",
    "download_to_file",
    "raw_proxy_forward",
    # Metrics
    "metrics_snapshot",
    "record_startup_phase",
]
