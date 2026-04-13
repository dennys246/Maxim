"""LLM type definitions — pure data types with no business logic.

Extracted from router.py for modularity. These types are used by
LLMRouter, backends (openai, anthropic), and external callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ─────────────────────────── Plan 2 R2b constants ──────────────────────────

INFERENCE_BROKEN_BACKOFF_S: float = 15.0
"""Single source of truth for how long the router backs off on
``BackendInferenceBroken`` and how long ``probe_cache`` treats an
``inference_broken`` probe outcome as authoritative. Plan 2 R2c's
``probe_cache.CACHE_TTL_BY_OUTCOME`` imports this; Plan 3's router
integration must import it too. Do not duplicate the value."""


@dataclass(slots=True)
class LLMResponse:
    """Structured response from any LLM backend."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    provider: str = ""
    stop_reason: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    cached_input_tokens: int = 0
    uncached_input_tokens: int = 0


@dataclass(slots=True)
class RoutingPolicy:
    """Governs routing and budget enforcement for providers."""

    provider_priority: list[str] = field(default_factory=list)
    fallback_on_rate_limit: bool = True
    fallback_on_timeout: bool = True
    fallback_on_budget_exceeded: str = "local"
    require_cloud_opt_in: bool = True
    context_window_routing: bool = True
    max_cost_per_request: float = 0.50
    max_cost_per_hour: float = 1.00
    max_cost_per_day: float = 10.00
    max_cost_per_month: float = 100.00
    max_session_cost: float = 5.00  # Hard ceiling — rejects ALL requests when hit
    cost_warning_threshold: float = 0.80
    cost_critical_threshold: float = 0.95


# ─────────────────────────── Plan 2 R2b: BackendError taxonomy ─────────────


class BackendError(Exception):
    """Base for all backend-raised errors.

    Mirrors the shape of :class:`maxim.utils.http.HTTPError` exactly — same
    three access patterns (``status``, ``response``, ``fix_hint``) so Plan 3's
    router integration can bridge HTTP → Backend exceptions with a simple
    ``except/raise`` pair instead of string-matching on exception messages.

    The three access patterns:

    1. ``e.status`` — HTTP status code when the failure came from an HTTP
       response. ``None`` for transport-layer failures (DNS, connect).
    2. ``e.response`` — parsed response object attached by the raiser (usually
       Plan 3's router ``_try_provider`` bridge). Use ``e.response`` only; do
       NOT add a parallel ``raw_body`` attribute. Matches how
       ``utils/http.py::_classify_status`` attaches ``response`` to raised
       ``HTTPError`` subclasses.
    3. ``e.fix_hint`` — human-actionable repair string. Class attribute with
       optional per-instance override. Never user-controllable (prevents log
       injection via exception-body interpolation).

    Fix #8 (R2 review): the base ``__init__`` no longer accepts
    ``**kwargs`` with ``setattr`` fallback. The old shape silently
    swallowed typos like ``retry_after=2.5`` (missing ``_s``) and set
    an unused attribute while the correctly-named class default
    remained in place. Every subclass with extra fields now declares
    an explicit ``__init__`` with named parameters so typos are
    ``TypeError``\\s at call time, not silent hot-path bugs in Plan 3.
    Subclasses without extra fields (``BackendDown``, ``BackendAuthFailed``,
    ``BackendInferenceBroken``) inherit the base ``__init__`` directly.
    """

    provider_key: str = ""
    fix_hint: str = ""

    def __init__(
        self,
        provider_key: str,
        *,
        status: int | None = None,
        fix_hint: str = "",
    ) -> None:
        self.provider_key = provider_key
        self.status = status
        if fix_hint:
            self.fix_hint = fix_hint
        self.response: Any | None = None  # attached by the router on body-bearing errors
        super().__init__(f"{type(self).__name__}[{provider_key}]: {self.fix_hint}")


class BackendOverloaded(BackendError):
    """Peer is at or near capacity (429, admission-control throttle)."""

    fix_hint = "Peer is at capacity. Try a different peer or wait."

    def __init__(
        self,
        provider_key: str,
        *,
        status: int | None = None,
        fix_hint: str = "",
        retry_after_s: float = 0.0,
        suggested_peer: str | None = None,
        queue_depth: int = 0,
    ) -> None:
        self.retry_after_s = retry_after_s
        self.suggested_peer = suggested_peer
        self.queue_depth = queue_depth
        super().__init__(provider_key, status=status, fix_hint=fix_hint)


class BackendDown(BackendError):
    """Peer is not responding — transport-layer failure (DNS, connect, reset)."""

    fix_hint = "Peer is not responding. Run `maxim peer --node <name> status`."


class BackendTimeout(BackendError):
    """Peer exceeded configured timeout without returning a response."""

    fix_hint = "Peer exceeded timeout. Check network or MAXIM_LANE_*_REMOTE_TIMEOUT_S."

    def __init__(
        self,
        provider_key: str,
        *,
        status: int | None = None,
        fix_hint: str = "",
        elapsed_s: float = 0.0,
    ) -> None:
        self.elapsed_s = elapsed_s
        super().__init__(provider_key, status=status, fix_hint=fix_hint)


class BackendAuthFailed(BackendError):
    """401/403 — the cluster key was rejected."""

    fix_hint = "Cluster key rejected. Verify mesh.yml::cluster_key matches peer config."


class BackendModelMissing(BackendError):
    """Peer is up but the requested model is not loaded."""

    fix_hint = "Run `maxim peer --node <name> install <model>`."

    def __init__(
        self,
        provider_key: str,
        *,
        status: int | None = None,
        fix_hint: str = "",
        requested_model: str = "",
    ) -> None:
        self.requested_model = requested_model
        super().__init__(provider_key, status=status, fix_hint=fix_hint)


class BackendInferenceBroken(BackendError):
    """Stage-2 probe failed: listener alive, chat endpoint broken.

    Raised when the liveness probe (``GET /v1/models``) succeeds but the
    readiness probe (micro-completion) fails. Router backs off for
    :data:`INFERENCE_BROKEN_BACKOFF_S` seconds before retrying.
    """

    fix_hint = "Model loading, llama-cpp crashed, or chat template broken. Check peer logs."


@dataclass
class ProviderState:
    """Tracks provider health and backoff state."""

    backoff_until: float = 0.0
    consecutive_errors: int = 0
    last_error: str = ""
    last_success: float = 0.0
