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

    Subclasses MUST NOT override ``__init__`` unless they have a documentation
    reason like :class:`BackendOverloaded` (explicit named kwargs for the
    retry-after path). Instead, set class-level defaults and let the base
    class handle kwargs.
    """

    provider_key: str = ""
    fix_hint: str = ""

    def __init__(
        self,
        provider_key: str,
        *,
        status: int | None = None,
        fix_hint: str = "",
        **kwargs: Any,
    ) -> None:
        self.provider_key = provider_key
        self.status = status
        if fix_hint:
            self.fix_hint = fix_hint
        self.response: Any | None = None  # attached by the router on body-bearing errors
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__(f"{type(self).__name__}[{provider_key}]: {self.fix_hint}")


class BackendOverloaded(BackendError):
    """Peer is at or near capacity (429, admission-control throttle)."""

    retry_after_s: float = 0.0
    suggested_peer: str | None = None
    queue_depth: int = 0
    fix_hint = "Peer is at capacity. Try a different peer or wait."


class BackendDown(BackendError):
    """Peer is not responding — transport-layer failure (DNS, connect, reset)."""

    fix_hint = "Peer is not responding. Run `maxim peer --node <name> status`."


class BackendTimeout(BackendError):
    """Peer exceeded configured timeout without returning a response."""

    elapsed_s: float = 0.0
    fix_hint = "Peer exceeded timeout. Check network or MAXIM_LANE_*_REMOTE_TIMEOUT_S."


class BackendAuthFailed(BackendError):
    """401/403 — the cluster key was rejected."""

    fix_hint = "Cluster key rejected. Verify mesh.yml::cluster_key matches peer config."


class BackendModelMissing(BackendError):
    """Peer is up but the requested model is not loaded."""

    requested_model: str = ""
    fix_hint = "Run `maxim peer --node <name> install <model>`."


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
