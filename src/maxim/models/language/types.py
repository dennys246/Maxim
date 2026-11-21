"""LLM type definitions — pure data types with no business logic.

Extracted from router.py for modularity. These types are used by
LLMRouter, backends (openai, anthropic), and external callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from maxim.exceptions import ConfigurationError

# ─────────────────────────── Plan 2 R2b constants ──────────────────────────

INFERENCE_BROKEN_BACKOFF_S: float = 15.0
"""Single source of truth for how long the router backs off on
``BackendInferenceBroken`` and how long ``probe_cache`` treats an
``inference_broken`` probe outcome as authoritative. Plan 2 R2c's
``probe_cache.CACHE_TTL_BY_OUTCOME`` imports this; Plan 3's router
integration must import it too. Do not duplicate the value."""


@dataclass(slots=True)
class LLMResponse:
    """Structured response from any LLM backend.

    Token telemetry contract (CC12, frozen at 1.0):

    - ``input_tokens`` — total prompt tokens (cached + uncached)
    - ``output_tokens`` — generated tokens
    - ``cached_tokens`` (property) — alias for ``cached_input_tokens``;
      exposed under the de-facto-standard name used by Anthropic and
      OpenAI usage payloads
    - ``cached_input_tokens`` — cached portion of input (legacy name,
      kept for ``CostTracker`` / router internals)
    - ``uncached_input_tokens`` — non-cached portion of input
      (cost-calculation detail; not part of the public contract)

    External callers should prefer ``cached_tokens`` over
    ``cached_input_tokens``.
    """

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

    @property
    def cached_tokens(self) -> int:
        """Standard-name alias for :attr:`cached_input_tokens` (CC12, 1.0)."""
        return self.cached_input_tokens


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

    SHAPE-FROZEN at 1.0 (CC3). The exception hierarchy is a typed
    contract bridged by the router on a per-subclass basis (specific-
    before-general); a free-form ``**kwargs`` / ``extra`` would re-open
    the silent-typo class of bugs Fix #8 closed. New keyword-only
    parameters with defaults are non-breaking and may be added per
    subclass; adding a *new subclass* requires the matching
    ``LLMRouter._try_provider`` branch in specific-before-general
    order. The one exception is a build-time subclass that no provider
    raises (``LocalModelUnavailable``, raised while a router is being
    built, before any provider runs): it has no ``_try_provider``
    branch, and its docstring says so (#1396 bullet 2). Adding a
    *required* field post-1.0 is a major-version-bump change.
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


class LocalModelUnavailable(BackendError, ConfigurationError):
    """An explicitly named local model is not on disk and was not downloaded, so the run refuses up front (#1396).

    Raised at router build time, never by a provider, so it has no ``LLMRouter._try_provider`` branch (the build-time
    exception :class:`BackendError` names): when the caller named the model (``build_primary_router(profile=)``,
    ``--llm``, ``--aut-model``, research's ``language_model``) and the consent gate
    (``models/download.py::ensure_available``) did not put its GGUF on disk. A run on some other model than the one
    asked for would be weak evidence, and a router on the missing file refuses every call. The CLI renders ``fix_hint``
    and exits 2 (``cli.py::_main_guarded``).

    Also a :class:`~maxim.exceptions.ConfigurationError` (owner decision 2026-11-14): ``maxim.run()``,
    ``maxim.create.router()`` and the other API entry points document ``ConfigurationError`` for a model they cannot
    serve, so an existing ``except ConfigurationError`` still catches it.

    ``profile`` is the model and ``cause`` why it is unusable (one of :attr:`CAUSES`); the fix hint is chosen by the
    cause from static text, and the profile is kept out of it (``fix_hint`` is never user-controllable).
    """

    NOT_DOWNLOADED = "not_downloaded"  # a registry model the gate did not download (declined, no grant, failed)
    NOT_IN_REGISTRY = "not_in_registry"  # a profile the download registry has no file for: placed by hand
    MODEL_PATH_MISSING = "model_path_missing"  # the router's own model_path is not where the registry file landed
    _FIX_HINTS = {
        NOT_DOWNLOADED: (
            "The model is not on disk and was not downloaded. Download it with `python -m maxim.models.download --llm "
            "<profile>`, or grant downloads with --auto-download or `maxim config set llm.auto_download true`."
        ),
        NOT_IN_REGISTRY: (
            "The model is not in the download registry, so it cannot be downloaded for you. Place its GGUF by hand, "
            "or register it with `maxim model add`."
        ),
        MODEL_PATH_MISSING: (
            "The router's model_path does not exist. Fix model_path in ~/.maxim/config/llm.json (or the quantization "
            "it is built from) to point at the model file."
        ),
    }
    CAUSES = tuple(_FIX_HINTS)
    fix_hint = _FIX_HINTS[NOT_DOWNLOADED]

    def __init__(self, profile: str, *, cause: str = NOT_DOWNLOADED, fix_hint: str = "") -> None:
        if cause not in self._FIX_HINTS:
            raise ValueError(f"unknown LocalModelUnavailable cause {cause!r}; expected one of {self.CAUSES}")
        self.profile = profile
        self.cause = cause
        super().__init__(profile, status=None, fix_hint=fix_hint or self._FIX_HINTS[cause])


class BackendInferenceBroken(BackendError):
    """Stage-2 probe failed: listener alive, chat endpoint broken.

    Raised when the liveness probe (``GET /v1/models``) succeeds but the
    readiness probe (micro-completion) fails. Router backs off for
    :data:`INFERENCE_BROKEN_BACKOFF_S` seconds before retrying.
    """

    fix_hint = "Model loading, llama-cpp crashed, or chat template broken. Check peer logs."


class BackendContextOverflow(BackendError):
    """The REQUEST overflowed the model's context window; the provider is healthy (#1369).

    Raised from a 400/413 whose body says so: the leader proxy's admission gate, llama-cpp-python, vLLM and the
    llama.cpp server each have a shape (``maxim_peer_backend.py::_context_overflow_counts``), and from a mid-stream
    error chunk of one of those shapes. The router gives it no cooldown, because the next, smaller prompt should
    still reach this provider. ``requested_tokens`` (prompt plus completion, or the prompt alone when the server
    reports only that) and ``context_window`` are ``None`` when the body did not carry them.
    """

    fix_hint = (
        "The prompt exceeds the model's context window. Shrink the prompt or max_tokens, or raise llm.n_ctx; "
        "`maxim doctor`'s context-window check reports the n_ctx the running server uses."
    )

    def __init__(
        self,
        provider_key: str,
        *,
        status: int | None = None,
        fix_hint: str = "",
        requested_tokens: int | None = None,
        context_window: int | None = None,
    ) -> None:
        self.requested_tokens = requested_tokens
        self.context_window = context_window
        super().__init__(provider_key, status=status, fix_hint=fix_hint)


RESOLVED_KEY_MARKER = "api_key_resolved"


def resolved_entry_key(entry: Any) -> str | None:
    """The key a provider entry carries already resolved, or None (#1371).

    A lane or placement entry built by ``runtime/lane_backends.py`` (``_build_remote_backend``,
    ``_placement_entry_to_provider`` for PEER and url-CLOUD) carries the key it resolved at the build, where an
    operator's ``MAXIM_LANE_<TIER>_REMOTE_API_KEY`` already beat config, and marks it ``api_key_resolved: True``. Every
    backend sends that key and consults no env var for it. Any other entry (a cloud profile, a hand-written llm.json
    provider) keeps its backend's env-first rule (CLI > env > config): the marker, never the absence of an
    ``api_key_env``, is what says "already resolved", so an entry that gains an ``api_key_env`` later cannot silently
    flip back."""
    if not isinstance(entry, dict) or entry.get(RESOLVED_KEY_MARKER) is not True:
        return None
    key = entry.get("api_key")
    return str(key).strip() if key else None


@dataclass
class ProviderState:
    """Tracks provider health and backoff state."""

    backoff_until: float = 0.0
    consecutive_errors: int = 0
    last_error: str = ""
    last_success: float = 0.0
