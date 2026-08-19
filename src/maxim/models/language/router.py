from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from collections.abc import Callable
from typing import Any

from maxim.utils.logging import info, warn
from maxim.utils.optional_deps import OptionalDependencyError

logger = logging.getLogger(__name__)
from maxim.utils.structured_logging import log_agentic, log_structured
from maxim.models.language.cost_tracker import CostTracker
from maxim.utils.cloud_redaction import CloudRedactionFilter, RedactionResult
from maxim.utils.cloud_audit import CloudAuditEntry, CloudAuditLogger

# Internal imports — only symbols actually used by LLMRouter
from maxim.models.language.token_counter import (
    TokenCounter,
    _LazyTokenCounter,
)
from maxim.models.language.prompt_formats import _build_prompt
from maxim.models.language.json_parser import _extract_json_object
from maxim.models.language.llama_backend import _LlamaCppBackend
from maxim.models.language.config import LLMConfig, load_llm_config
from maxim.models.language.types import (
    LLMResponse,
    RoutingPolicy,
    ProviderState,
    INFERENCE_BROKEN_BACKOFF_S,
    BackendError,
    BackendOverloaded,
    BackendDown,
    BackendTimeout,
    BackendAuthFailed,
    BackendModelMissing,
    BackendInferenceBroken,
)
from maxim.models.language.cloud_dispatch import (
    MODEL_DOWNGRADE_MAP as _MODEL_DOWNGRADE_MAP,
    SYSTEM_TOOL_RESPONSE as _SYSTEM_TOOL_RESPONSE,
    SYSTEM_JSON_ONLY as _SYSTEM_JSON_ONLY,
    SYSTEM_ROUTE as _SYSTEM_ROUTE,
    normalize_providers as _normalize_providers_fn,
    load_routing_policy as _load_routing_policy_fn,
    load_pricing_table as _load_pricing_table_fn,
    load_cost_config as _load_cost_config_fn,
)


def _inference_lock_timeout_s() -> float:
    """Ceiling on waiting for ``_inference_lock`` (bugs ledger D12).

    Default 600s: far above any sane single inference (including
    reasoning-model TTFTs), far below eternity. Clamped to [60, 3600] so a
    typo cannot re-introduce the unbounded wait or make the lock
    trigger-happy. Env: ``MAXIM_INFERENCE_LOCK_TIMEOUT_S``.
    """
    import os as _os

    raw = _os.environ.get("MAXIM_INFERENCE_LOCK_TIMEOUT_S", "").strip()
    if not raw:
        return 600.0
    try:
        return max(60.0, min(3600.0, float(raw)))
    except ValueError:
        return 600.0


def _record_unclassified_backend_error(provider_key: str) -> None:
    """Thin wrapper around
    :func:`maxim.models.language.lane_metrics.record_backend_unclassified_error`.

    Indirection lets the router catch the rare case where
    ``lane_metrics`` itself fails to import (optional dependency path) so
    the safety net never raises from inside a safety net. **R3 review
    fix:** import failures are now logged once via
    ``_UNCLASSIFIED_IMPORT_WARNED`` so a silent zero counter never masks
    a broken ``lane_metrics`` module — the whole point of the safety
    net is to be the canary for missing typed-exception classes, and
    silent import failures would defeat it.
    """
    global _UNCLASSIFIED_IMPORT_WARNED  # noqa: PLW0603
    try:
        from maxim.models.language.lane_metrics import record_backend_unclassified_error

        record_backend_unclassified_error(provider_key)
    except Exception as exc:  # pragma: no cover — defensive
        if not _UNCLASSIFIED_IMPORT_WARNED:
            _UNCLASSIFIED_IMPORT_WARNED = True
            warn(
                "backend_unclassified_errors_total counter unavailable: %s — "
                "safety-net metric will not record missing typed exception classes",
                exc,
            )


_UNCLASSIFIED_IMPORT_WARNED = False


def _maybe_dispatch_backend_class(provider_type: str) -> type | None:
    """Look up a backend class via ``lane_backends.BACKEND_CLASSES``.

    Isolated helper so the router's ``_get_backend_for_provider``
    branch that consults the dispatch table stays a single expression.
    Returns ``None`` on any failure (unknown identifier, optional
    dependency missing, import-time error) so the router falls through
    to its ``unknown provider type`` warning path.
    """
    try:
        from maxim.runtime.lane_backends import resolve_backend_class

        return resolve_backend_class(provider_type)
    except Exception:  # pragma: no cover — defensive
        return None


class LLMRouter:
    """
    Small, swappable transcript → action router.

    Returns a single Maxim action dict in the canonical schema:
    {"tool_name": str, "params": dict}
    """

    # Sentinel value to distinguish "init failed" from "not yet initialized"
    _INIT_FAILED = object()

    def __init__(
        self,
        cfg: LLMConfig | None = None,
        *,
        drain_constraint: Callable[[str], bool] | None = None,
        auto_drain_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.cfg = cfg or load_llm_config()
        self._backend: Any | None = None
        self._backends: dict[str, Any] = {}
        self._backend_lock = threading.Lock()
        self._inference_lock = threading.Lock()  # Serializes inference calls (llama-cpp not thread-safe)
        self._ready_event = threading.Event()  # Set when warmup completes
        self._warmup_failed = False
        self._drain_constraint = drain_constraint
        self._auto_drain_callback = auto_drain_callback
        self._pending_auto_drains: list[tuple[str, str]] = []
        self._providers = self._normalize_providers(self.cfg)
        self._provider_states: dict[str, ProviderState] = {key: ProviderState() for key in self._providers.keys()}
        self._routing_policy = self._load_routing_policy(self.cfg.routing)
        self._cost_tracker = CostTracker(
            pricing=self._load_pricing_table(self.cfg),
            config=self._load_cost_config(self.cfg),
        )
        self._audit_logger = CloudAuditLogger()
        self._cloud_allowed, self._cloud_block_reason = self._validate_cloud_config()
        self._session_cost: float = 0.0  # Accumulates cost for hard ceiling check
        self._session_cost_exceeded = False  # Once True, all requests rejected
        # (provider, model) pairs whose CostTracker.record already WARNed this
        # session — repeat failures drop to DEBUG so a metered lane with a
        # broken pricing table doesn't flood the log once per call.
        self._cost_record_warned: set[tuple[str, str]] = set()
        # Track last successfully-used model across providers. ``model_name``
        # returns the config's default; ``last_used_model`` returns what
        # actually ran (important when the router spans local + cloud backends).
        self._last_used_model: str = ""
        self._last_used_provider: str = ""
        # Backend type + base URL captured at success time for routing-path
        # diagnostics. Without these, the report cannot distinguish
        # "claude-sonnet-4-6 via Anthropic-direct" from
        # "claude-sonnet-4-6 via leader proxy" — both show identical model
        # names. Empty until the first successful dispatch.
        self._last_used_backend_class: str = ""
        self._last_used_endpoint: str = ""
        # Plan 3 R2.5: per-dispatch attempt log, drained into a single
        # ``dispatch_exhausted`` WARN when all providers fail. Populated
        # by ``_record_attempt_outcome`` from inside ``_try_provider``.
        # Safe as a plain list because ``_complete_text`` holds
        # ``_inference_lock`` for the full duration of a dispatch.
        self._dispatch_attempts: list[dict[str, Any]] = []
        self._last_drained_keys: list[str] = []
        self._last_total_provider_count: int = 0
        self._last_suggested_peer: str | None = None
        try:
            import atexit

            atexit.register(self._cost_tracker.flush)
        except Exception:
            pass

    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "enabled", False))

    @property
    def session_cost(self) -> float:
        """Current session cost in USD."""
        return self._session_cost

    @property
    def session_cost_exceeded(self) -> bool:
        """Whether the hard session ceiling has been hit."""
        return self._session_cost_exceeded

    def reset_session_cost(self) -> None:
        """Reset session cost accumulator and provider backoff state.

        Called between simulation runs.  Clears both the cost ceiling and the
        per-provider exponential backoff so that failures from a prior run
        (e.g., a stress-test phase that timed out) don't silently silence
        providers at the start of the next run.  ``_provider_states`` is
        instance-scoped and never auto-resets, so without this the backoff from
        run N bleeds into run N+1 for up to 60s.
        """
        self._session_cost = 0.0
        self._session_cost_exceeded = False
        for state in self._provider_states.values():
            state.backoff_until = 0.0
            state.consecutive_errors = 0
            state.last_error = ""

    def set_session_cost_limit(self, limit: float) -> None:
        """Override the session cost ceiling at runtime."""
        self._routing_policy.max_session_cost = limit

    @property
    def n_ctx(self) -> int:
        """Return the model's context window size in tokens."""
        return int(getattr(self.cfg, "n_ctx", 4096))

    @property
    def model_name(self) -> str:
        """Return the configured default model name (if any)."""
        return str(getattr(self.cfg, "model", "") or "")

    @property
    def last_used_model(self) -> str:
        """Return the model name that most recently served a request.

        When the router routes across multiple providers (local GPU,
        Anthropic, OpenAI, etc.), ``model_name`` returns only the
        config default — which is misleading in reports. This property
        reflects what actually executed.
        """
        return self._last_used_model or self.model_name

    @property
    def last_used_provider(self) -> str:
        """Return the provider key that most recently served a request."""
        return self._last_used_provider

    @property
    def last_used_backend_class(self) -> str:
        """Backend class name that served the most recent successful call.

        E.g., ``"_OpenAIBackend"`` (Anthropic or OpenAI cloud), or
        ``"_MaximPeerBackend"`` (self-hosted leader). Empty until the
        first successful dispatch.
        """
        return self._last_used_backend_class

    @property
    def last_used_endpoint(self) -> str:
        """Base URL that served the most recent successful call.

        E.g., ``"https://api.anthropic.com/v1"`` (Anthropic direct) or
        ``"https://maxim.big-mac-mini.org/v1"`` (leader-proxied) — lets
        the report distinguish cloud-direct from leader-proxied even
        when both serve the same model name. Empty when the provider's
        cfg has no base_url (most cloud providers use the SDK's default).
        """
        return self._last_used_endpoint

    def get_token_counter(self) -> TokenCounter:
        """Return a token counter that lazily upgrades to the real tokenizer.

        At startup the model may still be loading in a background thread.
        The returned counter starts with CharEstimateCounter and transparently
        switches to the actual tokenizer once it's available.
        """
        return _LazyTokenCounter(self)

    def _get_tokenizer_backend(self) -> Any | None:
        if self._backend is not None:
            return self._backend
        for backend in self._backends.values():
            if backend is not None and hasattr(backend, "_llm") and backend._llm is not None:
                return backend
        return None

    # Static config loaders — delegated to cloud_dispatch.py
    _normalize_providers = staticmethod(_normalize_providers_fn)
    _load_routing_policy = staticmethod(_load_routing_policy_fn)
    _load_pricing_table = staticmethod(_load_pricing_table_fn)
    _load_cost_config = staticmethod(_load_cost_config_fn)

    def _validate_cloud_config(self) -> tuple[bool, str]:
        if not self.cfg.cloud_enabled:
            return False, "cloud_enabled false"

        redaction_present = False
        redaction_cfg = self.cfg.redaction if isinstance(self.cfg.redaction, dict) else {}
        if redaction_cfg.get("policy"):
            redaction_present = True
        if not redaction_present:
            for provider in self._providers.values():
                if isinstance(provider, dict) and provider.get("redaction_policy"):
                    redaction_present = True
                    break
        if not redaction_present:
            warn("Cloud enabled but no redaction policy configured; cloud dispatch disabled")
            return False, "redaction missing"

        policy = self._routing_policy
        if (
            policy.max_cost_per_request <= 0
            and policy.max_cost_per_hour <= 0
            and policy.max_cost_per_day <= 0
            and policy.max_cost_per_month <= 0
        ):
            warn("Cloud enabled but all cost limits are zero; cloud dispatch disabled")
            return False, "cost limits missing"

        return True, ""

    def _provider_type(self, provider_cfg: dict[str, Any]) -> str:
        raw = provider_cfg.get("type") or provider_cfg.get("backend") or self.cfg.backend
        return str(raw or "").strip().lower().replace("-", "_")

    def _provider_is_cloud(self, provider_cfg: dict[str, Any]) -> bool:
        provider_type = self._provider_type(provider_cfg)
        return provider_type in ("anthropic", "claude", "openai", "openai_compatible", "openai_compat")

    def _provider_n_ctx(self, provider_cfg: dict[str, Any]) -> int:
        try:
            return int(provider_cfg.get("n_ctx", self.cfg.n_ctx))
        except Exception:
            return int(self.cfg.n_ctx)

    def update_provider_n_ctx(self, provider_key: str, n_ctx: int) -> None:
        """Update a provider's cached context window size after hot-swap.

        Called by swap_llm_server() so the router's context_window_routing
        filter uses the new model's n_ctx, not the startup value.
        """
        cfg = self._providers.get(provider_key)
        if cfg is not None:
            cfg["n_ctx"] = n_ctx

    def _provider_model(self, provider_cfg: dict[str, Any]) -> str:
        model = provider_cfg.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
        return str(self.cfg.model or "")

    def _provider_cost_visible(self, provider_cfg: dict[str, Any]) -> bool:
        if "cost_visible" in provider_cfg:
            return bool(provider_cfg.get("cost_visible"))
        return self._provider_is_cloud(provider_cfg)

    def _provider_pricing_required(self, provider_cfg: dict[str, Any]) -> bool:
        if "pricing_required" in provider_cfg:
            return bool(provider_cfg.get("pricing_required"))
        return self._provider_is_cloud(provider_cfg)

    def _get_backend_for_provider(self, provider_key: str) -> Any | None:
        backend = self._backends.get(provider_key)
        if backend is not None:
            return None if backend is LLMRouter._INIT_FAILED else backend

        provider_cfg = self._providers.get(provider_key, {})
        provider_type = self._provider_type(provider_cfg)

        # Local backends
        if provider_type in ("llama", "llama_cpp", "llamacpp"):
            cfg = self._build_provider_config(provider_cfg)
            backend = _LlamaCppBackend(cfg)
        elif provider_type in ("pytorch", "torch", "transformers", "huggingface", "hf"):
            from maxim.models.language.transformers_backend import _PyTorchTransformersBackend

            cfg = self._build_provider_config(provider_cfg)
            backend = _PyTorchTransformersBackend(cfg)
        # Cloud backends
        elif provider_type in ("anthropic", "claude"):
            from maxim.models.language.anthropic_backend import _AnthropicBackend

            backend = _AnthropicBackend(self.cfg, provider_key=provider_key)
        elif provider_type in ("openai", "gpt"):
            from maxim.models.language.openai_backend import _OpenAIBackend

            backend = _OpenAIBackend(self.cfg, provider_key=provider_key)
        elif provider_type in ("openai_compatible", "openai_compat"):
            from maxim.models.language.openai_backend import _OpenAIBackend

            backend = _OpenAIBackend(self.cfg, provider_key=provider_key)
        elif _maybe_dispatch_backend_class(provider_type) is not None:
            # Plan 3 R2.5 / R3 review fix: the
            # ``lane_backends.BACKEND_CLASSES`` dispatch table is now
            # authoritative. Self-hosted peers
            # (``provider_type="maxim_peer"`` / ``"maxim-peer"``)
            # resolve to ``_MaximPeerBackend``; future backend types
            # only need a new entry in the dispatch table plus any
            # ``_classify_backend`` wiring — no router edit.
            backend_cls = _maybe_dispatch_backend_class(provider_type)
            assert backend_cls is not None  # narrowed by the check above
            backend = backend_cls(self.cfg, provider_key=provider_key)
        else:
            warn("Unknown LLM provider type: %s (%s)", provider_type, provider_key)
            backend = LLMRouter._INIT_FAILED

        # Validate that the backend initialized successfully
        if backend is not LLMRouter._INIT_FAILED and backend is not None:
            try:
                # Quick sanity check — ensure the backend has the expected interface
                if not hasattr(backend, "complete") and not hasattr(backend, "complete_with_usage"):
                    warn(
                        "Provider %s (%s) backend missing complete method — marking as failed",
                        provider_key,
                        provider_type,
                    )
                    backend = LLMRouter._INIT_FAILED
            except Exception as e:
                warn(
                    "Provider %s (%s) init validation failed: %s. Check API key and provider configuration.",
                    provider_key,
                    provider_type,
                    e,
                )
                backend = LLMRouter._INIT_FAILED

        self._backends[provider_key] = backend
        return None if backend is LLMRouter._INIT_FAILED else backend

    def _build_provider_config(self, provider_cfg: dict[str, Any]) -> LLMConfig:
        """Override base config for local provider-specific settings."""
        model = provider_cfg.get("model", self.cfg.model)
        model_base = provider_cfg.get("model_base", self.cfg.model_base)
        model_path = provider_cfg.get("model_path", self.cfg.model_path)
        n_ctx = provider_cfg.get("n_ctx", self.cfg.n_ctx)
        backend = provider_cfg.get("type", self.cfg.backend)
        prompt_style = provider_cfg.get("prompt_style", self.cfg.prompt_style)
        stop_val = provider_cfg.get("stop", self.cfg.stop)
        stop: tuple[str, ...]
        if isinstance(stop_val, (list, tuple)) and stop_val:
            stop = tuple(str(s) for s in stop_val if isinstance(s, (str, int, float)) and str(s).strip())
        elif isinstance(stop_val, str) and stop_val.strip():
            stop = tuple(s.strip() for s in stop_val.split(",") if s.strip())
        else:
            stop = tuple(self.cfg.stop)
        return replace(
            self.cfg,
            backend=str(backend or self.cfg.backend),
            model=str(model or self.cfg.model),
            model_base=str(model_base or self.cfg.model_base),
            model_path=str(model_path or self.cfg.model_path),
            n_ctx=int(n_ctx or self.cfg.n_ctx),
            prompt_style=str(prompt_style or self.cfg.prompt_style),
            stop=stop,
        )

    def _default_provider(self) -> str:
        if self._routing_policy.provider_priority:
            for name in self._routing_policy.provider_priority:
                if name in self._providers:
                    return name
        return next(iter(self._providers.keys()))

    def _provider_order(self) -> list[str]:
        if self._routing_policy.provider_priority:
            return [p for p in self._routing_policy.provider_priority if p in self._providers]
        return list(self._providers.keys())

    def _local_providers(self) -> list[str]:
        locals_only = []
        for key, cfg in self._providers.items():
            if not self._provider_is_cloud(cfg):
                locals_only.append(key)
        return locals_only

    def _estimate_prompt_tokens(self, system: str, user: str) -> int:
        counter = self.get_token_counter()
        combined = f"{system}\n{user}".strip()
        try:
            return int(counter.count_tokens(combined))
        except Exception:
            return max(1, len(combined) // 3)

    def _budget_status(self, now: float) -> tuple[str, dict[str, float]]:
        policy = self._routing_policy
        totals = self._cost_tracker.get_totals(now)
        ratios: dict[str, float] = {}
        if policy.max_cost_per_hour > 0:
            ratios["hourly"] = totals["hourly"] / policy.max_cost_per_hour
        if policy.max_cost_per_day > 0:
            ratios["daily"] = totals["daily"] / policy.max_cost_per_day
        if policy.max_cost_per_month > 0:
            ratios["monthly"] = totals["monthly"] / policy.max_cost_per_month
        if not ratios:
            return "normal", totals
        max_ratio = max(ratios.values())
        if max_ratio >= 1.0:
            return "blocked", totals
        if max_ratio >= policy.cost_critical_threshold:
            return "critical", totals
        if max_ratio >= policy.cost_warning_threshold:
            return "warning", totals
        return "normal", totals

    def _model_for_tier(self, current_model: str, provider_cfg: dict[str, Any], tier: str) -> str:
        if tier not in ("warning", "critical"):
            return current_model
        tiers = provider_cfg.get("model_tiers")
        if isinstance(tiers, list) and tiers:
            model_list = [str(m) for m in tiers if isinstance(m, str)]
            if current_model in model_list:
                idx = model_list.index(current_model)
                if tier == "warning":
                    return model_list[min(idx + 1, len(model_list) - 1)]
                return model_list[-1]
            if tier == "critical":
                return model_list[-1]
        if tier == "warning":
            return _MODEL_DOWNGRADE_MAP.get(current_model, current_model)
        # critical: downgrade until stable
        model = current_model
        while model in _MODEL_DOWNGRADE_MAP and _MODEL_DOWNGRADE_MAP[model] != model:
            model = _MODEL_DOWNGRADE_MAP[model]
        return model

    def _candidate_providers(
        self,
        prompt_tokens: int,
        max_tokens: int,
        now: float,
    ) -> tuple[list[str], str, dict[str, float]]:
        policy = self._routing_policy
        budget_tier, totals = self._budget_status(now)

        providers = self._provider_order()
        filtered: list[str] = []
        drained_keys: list[str] = []
        for key in providers:
            cfg = self._providers.get(key, {})
            # Plan 4 C4: skip drained providers before any other filter.
            # Drained providers are recorded separately (not in
            # _dispatch_attempts) because they were never tried.
            if self._drain_constraint is not None and self._drain_constraint(key):
                drained_keys.append(key)
                continue
            # Self-hosted openai-compatible servers (llama-cpp-server, Ollama,
            # vLLM on a private IP) opt out of the cloud gate by setting
            # allow_local_endpoints=True — the SSRF check in _OpenAIBackend
            # still enforces that the URL actually resolves locally.
            is_self_hosted = bool(cfg.get("allow_local_endpoints", False))
            if self._provider_is_cloud(cfg) and not is_self_hosted:
                if not self._cloud_allowed or (policy.require_cloud_opt_in and not self.cfg.cloud_enabled):
                    continue
            state = self._provider_states.get(key)
            if state and state.backoff_until > now:
                continue
            if policy.context_window_routing:
                n_ctx = self._provider_n_ctx(cfg)
                if prompt_tokens + max_tokens > n_ctx:
                    continue
            filtered.append(key)
        self._last_drained_keys = drained_keys
        self._last_total_provider_count = len(providers)

        if budget_tier == "blocked":
            if policy.fallback_on_budget_exceeded == "local":
                locals_only = []
                for key in self._local_providers():
                    # Plan 4 C4: drain filter applies to local fallback too.
                    if self._drain_constraint is not None and self._drain_constraint(key):
                        continue
                    cfg = self._providers.get(key, {})
                    if policy.context_window_routing:
                        n_ctx = self._provider_n_ctx(cfg)
                        if prompt_tokens + max_tokens > n_ctx:
                            continue
                    locals_only.append(key)
                return locals_only, budget_tier, totals
            return [], budget_tier, totals

        return filtered, budget_tier, totals

    def _note_provider_success(
        self,
        provider_key: str,
        model: str = "",
        backend: Any | None = None,
    ) -> None:
        # Plan 3.5 R6 review: guard success bookkeeping behind the
        # cancellation check too. If an orphan thread completes the HTTP
        # call after the agent-level timeout fired, we don't want it to
        # write last_success/last_used_provider for a request the caller
        # already abandoned. The plan's invariant is "cancelled requests
        # leave _provider_states untouched" — applies to success AND
        # failure paths.
        if self._is_cancelled():
            return
        state = self._provider_states.get(provider_key)
        if state is None:
            return
        state.consecutive_errors = 0
        state.backoff_until = 0.0
        state.last_error = ""
        state.last_success = time.time()
        self._last_used_provider = provider_key
        if model:
            self._last_used_model = model
        # Capture routing-path diagnostics: backend class + base URL.
        # Without these the simulation report cannot distinguish
        # Anthropic-direct from leader-proxied from local-llama. Empty
        # values fall back silently — display layer renders only what's set.
        if backend is not None:
            try:
                self._last_used_backend_class = type(backend).__name__
            except Exception:
                pass
        # base_url lives on the provider config dict, not the backend
        # instance — read it from the same _providers cfg the dispatcher
        # used so the captured value reflects what actually ran.
        try:
            provider_cfg = self._providers.get(provider_key, {})
            self._last_used_endpoint = str(provider_cfg.get("base_url") or "")
        except Exception:
            pass

    def _is_cancelled(self) -> bool:
        """Plan 3.5 R4: check if the current request has been cancelled
        by ``LLMWorker._call_llm_with_timeout``'s agent-level timeout.

        When True, all ``_note_provider_*``, ``_set_*_backoff``, AND
        ``_note_provider_success`` helpers become no-ops so the cancelled
        request's orphan-thread unwind does NOT pollute ``_provider_states``
        with phantom state (failure OR success). Without this guard, a
        cancelled call whose orphan eventually raises
        ``BackendDown(fix_hint="cancelled")`` or completes successfully
        would mutate ``consecutive_errors`` / ``last_success`` /
        ``_last_used_provider`` for a request the caller already abandoned.

        Lazy-imports ``maxim.utils.cancellation`` for cycle-free import
        ordering. The primitive lives in ``utils/`` (Plan 3.5 R6 review)
        precisely so router → utils is the only direction in the graph.
        """
        from maxim.utils.cancellation import is_cancelled

        return is_cancelled()

    def _note_provider_failure(self, provider_key: str, error: str) -> None:
        if self._is_cancelled():
            return
        state = self._provider_states.get(provider_key)
        if state is None:
            return
        state.consecutive_errors += 1
        state.last_error = error[:200]
        backoff = min(60.0, 1.0 * (2 ** max(state.consecutive_errors - 1, 0)))
        state.backoff_until = time.time() + backoff

    # ─── Plan 3 R2.5: typed-failure provider helpers ────────────────────
    def _note_provider_overload(
        self,
        provider_key: str,
        *,
        retry_after_s: float = 0.0,
    ) -> None:
        """Apply cooldown for a :class:`BackendOverloaded` outcome.

        Honors the upstream-supplied ``retry_after_s`` hint (clamped to a
        sane range) when it is set; otherwise falls back to the exponential
        cooldown applied by :meth:`_note_provider_failure`.
        """
        if self._is_cancelled():
            return
        state = self._provider_states.get(provider_key)
        if state is None:
            return
        state.consecutive_errors += 1
        state.last_error = "overloaded"
        if retry_after_s > 0:
            delay = max(1.0, min(float(retry_after_s), 300.0))
        else:
            delay = min(60.0, 1.0 * (2 ** max(state.consecutive_errors - 1, 0)))
        state.backoff_until = time.time() + delay

    def _set_long_backoff(self, provider_key: str, seconds: float) -> None:
        """Hard-set a long cooldown for hints that don't self-heal
        (auth rejection, model missing). Bypasses the exponential ramp."""
        if self._is_cancelled():
            return
        state = self._provider_states.get(provider_key)
        if state is None:
            return
        state.consecutive_errors += 1
        state.backoff_until = time.time() + max(1.0, float(seconds))
        # Emit a structured event so operators can see which providers are
        # silenced and why without waiting for the next dispatch_exhausted.
        # Hard cooldowns (300s auth, 60s model_missing) require operator
        # intervention and should be immediately visible in logs.
        log_structured(
            logger,
            logging.WARNING,
            event="provider_silenced",
            data={
                "provider": provider_key,
                "backoff_s": round(seconds, 1),
                "reason": state.last_error or "unknown",
                "consecutive_errors": state.consecutive_errors,
            },
        )

    def _set_short_backoff(self, provider_key: str, seconds: float) -> None:
        """Short cooldown matching the probe-cache TTL for
        :data:`INFERENCE_BROKEN_BACKOFF_S`."""
        if self._is_cancelled():
            return
        state = self._provider_states.get(provider_key)
        if state is None:
            return
        state.consecutive_errors += 1
        state.backoff_until = time.time() + max(0.5, float(seconds))
        log_structured(
            logger,
            logging.WARNING,
            event="provider_silenced",
            data={
                "provider": provider_key,
                "backoff_s": round(seconds, 1),
                "reason": state.last_error or "unknown",
                "consecutive_errors": state.consecutive_errors,
            },
        )

    def _record_suggested_peer_hint(self, suggested_peer: str | None) -> None:
        """No-op stub for the deferred multi-peer dispatch plan.

        :class:`BackendOverloaded` carries a ``suggested_peer`` hint that
        the leader sets via ``X-Maxim-Suggested-Peer``. Plan 3's scope is
        single-peer failover; the hint is recorded here as a no-op so the
        deferred multi-peer plan can light it up without refactoring the
        router's error-handling branches."""
        if suggested_peer:
            self._last_suggested_peer = suggested_peer

    def _record_attempt_outcome(
        self,
        provider_key: str,
        *,
        outcome: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Buffer a per-attempt outcome so
        :meth:`_emit_dispatch_exhausted` can log one aggregated WARN line
        per failed dispatch (instead of one-per-attempt).

        **R3 review fix:** ``_dispatch_attempts`` is a plain list and is
        only safe because ``_complete_text`` holds ``_inference_lock``
        for the full duration of a dispatch. The invariant is now
        enforced at call time — any future code path that calls
        ``_try_provider`` outside the lock (e.g., a warmup probe, a
        benchmark hook) trips this assertion loudly instead of silently
        corrupting the attempt buffer across dispatches.
        """
        assert self._inference_lock.locked(), (
            "_record_attempt_outcome must be called under _inference_lock — _dispatch_attempts is not thread-safe"
        )
        attempt: dict[str, Any] = {
            "provider": provider_key,
            "outcome": outcome,
        }
        if extra:
            attempt.update(extra)
        self._dispatch_attempts.append(attempt)

    def _emit_dispatch_exhausted(
        self,
        *,
        request_context: dict[str, Any] | None,
        total_elapsed_ms: float,
        drained_keys: list[str] | None = None,
    ) -> None:
        """Emit one ``dispatch_exhausted`` WARN per failed dispatch.

        Preserves multi-agent context (``agent_id``, ``session_id``,
        ``request_id``, ``lane``) and lists every per-provider attempt
        with its typed outcome. Grep-friendly via
        ``jq 'select(.e=="dispatch_exhausted")'``.

        Plan 4 C4: ``drained_keys`` lists providers skipped due to drain
        state. They are NOT in ``_dispatch_attempts`` (never tried).

        **R3 review fix:** delegates to
        :func:`maxim.agents.llm_worker._normalize_request_context` (the
        canonical Plan 2 R2b shim) instead of reading ``ctx.get("agent")``
        inline. The shim is the only sanctioned bridge between the
        legacy dict-shape ``request_context`` and typed
        :class:`RequestContext`; duplicating its logic here would be
        the same architectural drift the Plan 2 R2b commit forbid.
        """
        attempts = list(self._dispatch_attempts)
        self._dispatch_attempts.clear()

        # Lazy import avoids a circular dependency via
        # agents.llm_worker → router.
        from maxim.agents.llm_worker import _normalize_request_context

        ctx = _normalize_request_context(request_context)
        data: dict[str, Any] = {
            "request_id": ctx.request_id,
            "agent_id": ctx.agent_id,
            "session_id": ctx.session_id,
            "lane": ctx.lane,
            "total_elapsed_ms": round(total_elapsed_ms, 1),
            "attempts": attempts,
        }
        if drained_keys:
            data["drained_providers"] = sorted(drained_keys)
        log_structured(
            logger,
            logging.WARNING,
            event="dispatch_exhausted",
            data=data,
        )

    def _emit_dispatch_exhausted_all_drained(
        self,
        *,
        request_context: dict[str, Any] | None,
        drained_keys: list[str],
    ) -> None:
        """All candidates were drained. Distinct event with fix hint.

        Plan 4 C4: fires IFF all candidates were eliminated by drain
        AND there were candidates before drain.
        """
        from maxim.agents.llm_worker import _normalize_request_context

        ctx = _normalize_request_context(request_context)
        log_structured(
            logger,
            logging.WARNING,
            event="dispatch_exhausted_all_drained",
            data={
                "request_id": ctx.request_id,
                "agent_id": ctx.agent_id,
                "session_id": ctx.session_id,
                "lane": ctx.lane,
                "drained_providers": sorted(drained_keys),
                "fix": "all providers are drained -- run 'maxim peer list-drained'",
            },
        )

    def _maybe_schedule_auto_drain(self, provider_key: str, outcome: str) -> None:
        """Schedule an auto-drain if consecutive errors cross the threshold.

        Plan 4 C4.5: appends to ``_pending_auto_drains`` which is flushed
        OUTSIDE ``_inference_lock`` by ``_complete_text``. The actual write
        (filelock + disk I/O) never runs inside the inference critical section.
        """
        if self._auto_drain_callback is None:
            return
        state = self._provider_states.get(provider_key)
        if state is None:
            return
        from maxim.peer.drain_routing import auto_drain_threshold

        threshold = auto_drain_threshold(outcome)
        if state.consecutive_errors >= threshold:
            self._pending_auto_drains.append((provider_key, outcome))

    def _emit_cloud_audit(
        self,
        *,
        provider_key: str,
        model: str,
        usage: dict[str, Any],
        cost_usd: float,
        redaction: RedactionResult,
        request_context: dict[str, Any] | None,
    ) -> None:
        ctx = request_context or {}
        entry = CloudAuditEntry(
            timestamp=time.time(),
            provider=provider_key,
            model=model,
            data_categories_sent=redaction.categories_sent,
            data_categories_redacted=redaction.categories_redacted,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            estimated_cost_usd=float(cost_usd or 0.0),
            request_id=str(ctx.get("request_id", "")),
            agent=str(ctx.get("agent", "")),
            redaction_policy=redaction.policy,
        )
        try:
            self._audit_logger.write(entry)
        except Exception as e:
            warn("Failed to write cloud audit entry: %s", e)

    def _complete_text(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
        provider_hint: str | None = None,
        request_context: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> tuple[str, dict[str, Any] | None]:
        """Complete text with optional usage metadata."""
        if not self.enabled():
            return "", None

        # Hard session cost ceiling — once exceeded, ALL requests rejected
        policy = self._routing_policy
        if policy.max_session_cost > 0 and self._session_cost_exceeded:
            return "", None
        if policy.max_session_cost > 0 and self._session_cost >= policy.max_session_cost:
            self._session_cost_exceeded = True
            warn(
                "Session cost ceiling reached ($%.2f >= $%.2f). "
                "All further LLM requests will be rejected. "
                "Increase max_session_cost in routing policy or llm.json to raise the limit.",
                self._session_cost,
                policy.max_session_cost,
            )
            return "", None

        # Serialize inference — llama-cpp is not thread-safe for concurrent
        # calls on the same model.  In simulation mode two LLMWorkers share
        # one router; without this lock the second call segfaults.
        #
        # BOUNDED acquire (bugs ledger D12, root-caused 2026-08-18): the
        # llm_worker timeout path abandons an orphan thread that may still
        # be INSIDE the locked region; its executor-replacement fallback
        # gives new calls a fresh thread but nothing ever frees this lock,
        # so with an untimed acquire every subsequent call parks forever —
        # observed live as 2.4h 'planning' hangs with ~75 lock-waiter
        # threads, zero network activity, and a blind stall detector (the
        # call registry is entered inside the lock, so queued calls are
        # invisible to it). A bounded acquire converts the eternal silent
        # hang into a loud per-call failure the retry/stall machinery can
        # actually see. The ceiling is generous (default 600s — far above
        # any sane single inference incl. reasoning-model TTFT, far below
        # eternity); tune via MAXIM_INFERENCE_LOCK_TIMEOUT_S.
        if not self._inference_lock.acquire(timeout=_inference_lock_timeout_s()):
            logger.error(
                "inference lock held > %.0fs — an orphaned/wedged call is holding it "
                "(bugs ledger D12). Failing this call loudly instead of queuing forever.",
                _inference_lock_timeout_s(),
            )
            return "", None
        try:
            result = self._complete_text_locked(
                system,
                user,
                temperature=temperature,
                max_tokens=max_tokens,
                provider_hint=provider_hint,
                request_context=request_context,
                tools=tools,
                thinking=thinking,
                stream=stream,
            )
        finally:
            self._inference_lock.release()

        # Plan 4 C4.5: flush pending auto-drains OUTSIDE _inference_lock.
        # The threshold checks in _try_provider populated the buffer under
        # the lock; the actual filelock + disk writes happen here.
        if self._pending_auto_drains and self._auto_drain_callback is not None:
            pending = list(self._pending_auto_drains)
            self._pending_auto_drains.clear()
            for key, reason in pending:
                try:
                    self._auto_drain_callback(key, reason)
                except Exception:
                    pass  # AutoDrainWriter logs internally

        return result

    def _complete_text_locked(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
        provider_hint: str | None = None,
        request_context: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> tuple[str, dict[str, Any] | None]:
        """Actual inference — called under _inference_lock."""
        prompt_tokens = self._estimate_prompt_tokens(system, user)
        now = time.time()

        providers, budget_tier, _totals = self._candidate_providers(prompt_tokens, max_tokens, now)
        if provider_hint and provider_hint in providers:
            providers = [provider_hint] + [p for p in providers if p != provider_hint]

        if not providers:
            # Plan 4 C4: distinguish "all drained" from "no providers".
            # Only fire all_drained if EVERY provider was eliminated by
            # drain (not by backoff, cloud gate, or context window).
            if self._last_drained_keys and len(self._last_drained_keys) == self._last_total_provider_count:
                self._emit_dispatch_exhausted_all_drained(
                    request_context=request_context,
                    drained_keys=self._last_drained_keys,
                )
            else:
                warn(
                    "No eligible LLM providers for request. "
                    "Run `maxim --list-models` to check model status, "
                    "or set an API key (e.g. ANTHROPIC_API_KEY)"
                )
            return "", None

        # Plan 3 R2.5: reset the per-dispatch attempt log at the top of
        # every dispatch. Populated by ``_record_attempt_outcome`` inside
        # ``_try_provider`` and drained into a single
        # ``dispatch_exhausted`` WARN if all providers fail.
        self._dispatch_attempts.clear()
        dispatch_t0 = time.monotonic()

        # Stall-detector integration: register the dispatch as an in-flight
        # LLM call BEFORE entering the provider-fallback loop, so the
        # registry sees one continuous in-flight window across provider
        # retries. Per consolidated review of stall_detector_timeout_awareness.md
        # v2, wrapping at the per-backend ``complete_with_usage`` site would
        # leave gaps between provider-A end and provider-B start where the
        # orchestrator's stall detector would incorrectly fire mid-failover.
        # The call_id is threaded into ``_try_provider`` so backend stream
        # consumers can call ``register_byte_received(call_id)`` on chunk
        # arrival (including PR #320 TTFT keepalive frames).
        from maxim.runtime.llm_call_registry import register_call_end, register_call_start

        # call_id flows to backend stream consumers via ContextVar
        # (see llm_call_registry.py); no signature threading needed.
        call_id = register_call_start(tier=str(budget_tier) if budget_tier else "unknown")
        try:
            for provider_key in providers:
                outcome = self._try_provider(
                    provider_key=provider_key,
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    prompt_tokens=prompt_tokens,
                    budget_tier=budget_tier,
                    tools=tools,
                    thinking=thinking,
                    stream=stream,
                    request_context=request_context,
                    now=now,
                )
                if outcome is None:
                    # provider was skipped (cloud-disabled, backend init failure, etc.)
                    continue
                text, usage, status = outcome
                if status == "success":
                    # Clear the attempt buffer; a successful dispatch doesn't
                    # need an aggregated WARN even if earlier providers in
                    # the priority list failed.
                    self._dispatch_attempts.clear()
                    return text, usage
                if status == "budget_reject":
                    self._dispatch_attempts.clear()
                    return "", None
                # status == "failed" → try next provider

            # All providers failed — emit one aggregated WARN with the
            # full per-attempt breakdown and multi-agent context.
            total_elapsed_ms = (time.monotonic() - dispatch_t0) * 1000
            self._emit_dispatch_exhausted(
                request_context=request_context,
                total_elapsed_ms=total_elapsed_ms,
                drained_keys=self._last_drained_keys or None,
            )
            return "", None
        finally:
            register_call_end(call_id)

    def _try_provider(
        self,
        *,
        provider_key: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        prompt_tokens: int,
        budget_tier: Any,
        tools: list[dict[str, Any]] | None,
        thinking: dict[str, Any] | None,
        stream: bool,
        request_context: dict[str, Any] | None,
        now: float,
    ) -> tuple[str, dict[str, Any] | None, str] | None:
        """Try a single provider for one inference attempt.

        Returns:
            None if the provider was skipped (not eligible, no backend),
            ``(text, usage, status)`` otherwise where status is one of:
              - ``"success"`` — return the text/usage to caller
              - ``"budget_reject"`` — caller should reject the request entirely
              - ``"failed"`` — caller should try the next provider
        """
        provider_cfg = self._providers.get(provider_key, {})
        is_self_hosted = bool(provider_cfg.get("allow_local_endpoints", False))
        if self._provider_is_cloud(provider_cfg) and not is_self_hosted and not self._cloud_allowed:
            return None

        backend = self._get_backend_for_provider(provider_key)
        if backend is None:
            self._note_provider_failure(provider_key, "backend_init_failed")
            return None

        model = self._provider_model(provider_cfg)
        model_override = self._model_for_tier(model, provider_cfg, budget_tier)

        # Cost checks + PII redaction only apply to real cloud providers,
        # not to self-hosted openai-compatible servers (llama-cpp-server,
        # Ollama, vLLM) running on a private IP.
        treat_as_cloud = self._provider_is_cloud(provider_cfg) and not is_self_hosted
        budget_status = self._check_provider_budget(
            provider_key=provider_key,
            provider_cfg=provider_cfg,
            model_override=model_override,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            treat_as_cloud=treat_as_cloud,
        )
        if budget_status == "skip":
            return None
        if budget_status == "reject":
            return "", None, "budget_reject"

        redacted_system, redacted_user, redaction_result = self._redact_for_provider(
            system=system,
            user=user,
            provider_cfg=provider_cfg,
            treat_as_cloud=treat_as_cloud,
        )

        # Plan 3 R2.5: typed exception handlers BEFORE the generic
        # catch-all. Order is specific-before-general because Python's
        # except dispatch is first-match-wins by source order — the
        # 2026-04-12 stage-2-probe review round found the inverse bug
        # (R2c initially classified HTTPAuthError as inference_broken
        # because the generic catch came first).
        try:
            text, usage = self._invoke_backend(
                backend=backend,
                provider_key=provider_key,
                redacted_system=redacted_system,
                redacted_user=redacted_user,
                model=model,
                model_override=model_override,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                thinking=thinking,
                stream=stream,
                redaction_result=redaction_result,
                request_context=request_context,
                now=now,
                treat_as_cloud=treat_as_cloud,
            )
            if text:
                return text, usage, "success"
        except OptionalDependencyError:
            # A requested backend's optional dependency is NOT installed. This
            # is a SETUP error, categorically different from the transient
            # BackendError failures below (which all mean "try the next
            # provider"). Silently falling through to another provider is
            # exactly the 2026-06-05 cloud-dispatch incident: the sim completed
            # with cost=$0 and every action an _llm_unavailable fallback, and
            # the missing `anthropic` package was invisible. Re-raise so it
            # propagates out of the dispatch loop and aborts the run loudly
            # with the actionable pip-install hint. MUST stay BEFORE the
            # generic `except Exception` (which would otherwise swallow it as
            # an "unclassified" provider failure).
            raise
        except BackendOverloaded as e:
            self._note_provider_overload(
                provider_key,
                retry_after_s=getattr(e, "retry_after_s", 0.0),
            )
            self._record_suggested_peer_hint(getattr(e, "suggested_peer", None))
            self._record_attempt_outcome(
                provider_key,
                outcome="overloaded",
                extra={"retry_after_s": getattr(e, "retry_after_s", 0.0)},
            )
            self._maybe_schedule_auto_drain(provider_key, "overloaded")
            return "", None, "failed"
        except BackendAuthFailed:
            # 300s hard cooldown — rotating a cluster key doesn't
            # self-heal; operator intervention required. Do NOT call
            # ``_note_provider_failure`` in this branch: it runs the
            # exponential ramp ``min(60, 2**n)`` which would clobber
            # the 300s hard value with at most a 60s cooldown. The
            # 300s value is load-bearing: it stops the router from
            # hammering a key-rotated peer every few seconds.
            state = self._provider_states.get(provider_key)
            if state is not None:
                state.last_error = "auth_failed"
            self._set_long_backoff(provider_key, 300.0)
            self._record_attempt_outcome(provider_key, outcome="auth_failed")
            self._maybe_schedule_auto_drain(provider_key, "auth_failed")
            return "", None, "failed"
        except BackendModelMissing as e:
            # 60s cooldown — operator needs to install the model; don't
            # hammer the peer while it's not going to answer. Same
            # load-bearing rule as the auth branch above: no
            # ``_note_provider_failure`` call — it would overwrite the
            # hard 60s value with the exponential ramp.
            requested = getattr(e, "requested_model", "")
            state = self._provider_states.get(provider_key)
            if state is not None:
                state.last_error = f"model_missing:{requested}"[:200]
            self._set_long_backoff(provider_key, 60.0)
            self._record_attempt_outcome(
                provider_key,
                outcome="model_missing",
                extra={"requested_model": requested},
            )
            self._maybe_schedule_auto_drain(provider_key, "model_missing")
            return "", None, "failed"
        except BackendInferenceBroken:
            # Stage-2 probe outcome — match the probe-cache TTL.
            state = self._provider_states.get(provider_key)
            if state is not None:
                state.last_error = "inference_broken"
            self._set_short_backoff(provider_key, INFERENCE_BROKEN_BACKOFF_S)
            self._record_attempt_outcome(provider_key, outcome="inference_broken")
            self._maybe_schedule_auto_drain(provider_key, "inference_broken")
            return "", None, "failed"
        except BackendTimeout as e:
            elapsed = getattr(e, "elapsed_s", 0.0) or 0.0
            self._note_provider_failure(
                provider_key,
                f"timeout_{elapsed:.1f}s"[:200],
            )
            self._record_attempt_outcome(
                provider_key,
                outcome="timeout",
                extra={"elapsed_s": round(elapsed, 2)},
            )
            self._maybe_schedule_auto_drain(provider_key, "timeout")
            return "", None, "failed"
        except BackendDown as e:
            http_status = getattr(e, "status", None)
            self._note_provider_failure(
                provider_key,
                f"down_{http_status}"[:200] if http_status is not None else "down",
            )
            self._record_attempt_outcome(
                provider_key,
                outcome="down",
                extra={"http_status": http_status},
            )
            self._maybe_schedule_auto_drain(provider_key, "down")
            return "", None, "failed"
        except BackendError as e:
            # Unmapped BackendError subclass or generic base. Records as
            # generic_backend_error (still typed, still actionable — the
            # fix_hint propagates through _log_provider_error below).
            self._log_provider_error(provider_key, e)
            self._note_provider_failure(provider_key, "generic_backend_error")
            self._record_attempt_outcome(provider_key, outcome="generic_backend_error")
            self._maybe_schedule_auto_drain(provider_key, "generic_backend_error")
            return "", None, "failed"
        except Exception as e:
            # Safety net for non-typed exceptions. Non-zero count of
            # ``backend_unclassified_errors_total`` means a backend
            # raised something outside the BackendError hierarchy —
            # file a bug and either extend the hierarchy or wrap the
            # call site. See docs/plans/archive/llm_path_fast_failover.md.
            self._log_provider_error(provider_key, e)
            self._note_provider_failure(provider_key, "unclassified")
            _record_unclassified_backend_error(provider_key)
            self._record_attempt_outcome(
                provider_key,
                outcome="unclassified",
                extra={"error": type(e).__name__},
            )
            self._maybe_schedule_auto_drain(provider_key, "unclassified")
            return "", None, "failed"

        self._note_provider_failure(provider_key, "call_failed")
        self._record_attempt_outcome(provider_key, outcome="empty_response")
        self._maybe_schedule_auto_drain(provider_key, "empty_response")
        return "", None, "failed"

    def _check_provider_budget(
        self,
        *,
        provider_key: str,
        provider_cfg: dict[str, Any],
        model_override: str | None,
        prompt_tokens: int,
        max_tokens: int,
        treat_as_cloud: bool,
    ) -> str:
        """Apply per-request cost ceiling. Returns ``"ok"``/``"skip"``/``"reject"``."""
        if not treat_as_cloud:
            return "ok"
        policy = self._routing_policy
        if policy.max_cost_per_request <= 0:
            return "ok"

        estimate = self._cost_tracker.estimate_cost(model_override, prompt_tokens, max_tokens)
        if estimate is None and self._provider_pricing_required(provider_cfg):
            warn("Missing pricing for model %s; skipping provider %s", model_override, provider_key)
            return "skip"
        if estimate is not None and estimate > policy.max_cost_per_request:
            warn("Estimated cost %.4f exceeds per-request limit", estimate)
            if policy.fallback_on_budget_exceeded == "reject":
                return "reject"
            return "skip"
        return "ok"

    def _redact_for_provider(
        self,
        *,
        system: str,
        user: str,
        provider_cfg: dict[str, Any],
        treat_as_cloud: bool,
    ) -> tuple[str, str, RedactionResult | None]:
        """Apply PII redaction for cloud providers; pass-through for self-hosted."""
        if not treat_as_cloud:
            return system, user, None

        redactor = CloudRedactionFilter.from_config(
            provider_cfg=provider_cfg,
            global_cfg=self.cfg.redaction,
        )
        redaction_result = redactor.redact(system, user)
        return redaction_result.system, redaction_result.user, redaction_result

    def _invoke_backend(
        self,
        *,
        backend: Any,
        provider_key: str,
        redacted_system: str,
        redacted_user: str,
        model: str,
        model_override: str | None,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        thinking: dict[str, Any] | None,
        stream: bool,
        redaction_result: RedactionResult | None,
        request_context: dict[str, Any] | None,
        now: float,
        treat_as_cloud: bool,
    ) -> tuple[str, dict[str, Any] | None]:
        """Dispatch to a backend's complete()/complete_with_usage() and bookkeep.

        Records cost + emits cloud-audit on success. Returns ``(text, usage)``;
        ``usage`` is ``None`` for the legacy prompt-formatting path.

        ``treat_as_cloud`` is the caller's (``_try_provider``) already-computed
        "genuinely metered cloud" predicate — ``_provider_is_cloud(cfg) and not
        allow_local_endpoints``. It, and ONLY it, decides whether a missing
        pricing entry at record time is loud. Do NOT substitute the provider
        config's ``pricing_required`` flag here: lane_backends hardcodes
        ``pricing_required: False`` on every remote lane entry (including
        kind=="cloud" URL lanes) to opt out of the pre-dispatch estimate gate,
        so reusing that flag at record time silently unmeters real cloud spend.
        """
        # Path A: backend wants pre-formatted prompt (legacy llama-cpp style)
        if getattr(backend, "requires_prompt_formatting", True):
            prompt_cfg = getattr(backend, "cfg", self.cfg)
            stop = tuple(getattr(prompt_cfg, "stop", ("</s>",)))
            prompt = _build_prompt(prompt_cfg, redacted_system, redacted_user)
            text = backend.complete(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            if text:
                self._note_provider_success(provider_key, model=model_override or model, backend=backend)
            return text, None

        # Path B: backend takes structured system/user (cloud providers)
        if hasattr(backend, "complete_with_usage"):
            kwargs: dict[str, Any] = {
                "system": redacted_system,
                "user": redacted_user,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": tuple(getattr(self.cfg, "stop", ("</s>",))),
            }
            if model_override and getattr(backend, "supports_model_override", False):
                kwargs["model_override"] = model_override
            if tools and getattr(backend, "supports_tool_use", False):
                kwargs["tools"] = tools
            if thinking:
                kwargs["thinking"] = thinking
            if stream and getattr(backend, "supports_streaming", False):
                kwargs["stream"] = True
            # Plan 4 A.1: forward request_context to backends that accept
            # it (capability flag, matches existing supports_* pattern).
            # _MaximPeerBackend needs this to populate agent_id/session_id/
            # request_id/lane fields on its peer_backend_call and
            # peer_backend_failed structured log events — the Phase D
            # report flagged these as agent_id=null because this kwarg
            # was being dropped on the floor here. Cloud backends
            # (_OpenAIBackend, _AnthropicBackend) do not declare this
            # capability and continue to receive the same kwargs as before.
            if request_context is not None and getattr(backend, "supports_request_context", False):
                kwargs["request_context"] = request_context
            resp = backend.complete_with_usage(**kwargs)
            if isinstance(resp, LLMResponse) and resp.content:
                self._note_provider_success(
                    provider_key,
                    model=resp.model or model_override or model,
                    backend=backend,
                )
                usage = {
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "model": resp.model,
                    "provider": resp.provider,
                    # CC12: cached_tokens is the public-contract field
                    # name; cached_input_tokens is the legacy alias kept
                    # for CostTracker / router internals.
                    "cached_tokens": resp.cached_input_tokens,
                    "cached_input_tokens": resp.cached_input_tokens,
                    "uncached_input_tokens": resp.uncached_input_tokens,
                }
                try:
                    cost_usd = self._cost_tracker.record(
                        provider=resp.provider or provider_key,
                        model=resp.model or model_override,
                        input_tokens=resp.input_tokens,
                        output_tokens=resp.output_tokens,
                        cached_input_tokens=resp.cached_input_tokens,
                        uncached_input_tokens=resp.uncached_input_tokens,
                        timestamp=now,
                        # Metering is decided by treat_as_cloud (genuinely
                        # metered cloud call), NOT by the provider config's
                        # pricing_required flag — that flag opts remote lanes
                        # out of the pre-dispatch estimate gate and is False
                        # even on kind=="cloud" URL lanes, so keying record()
                        # on it would silently unmeter real cloud spend.
                        # Local/self-hosted (treat_as_cloud=False): a missing
                        # pricing entry is BY DESIGN, record quietly as $0
                        # (2026-08-03: conflating the two produced a WARNING
                        # on every single local LLM call).
                        pricing_required=treat_as_cloud,
                    )
                except Exception as e:
                    # Pricing data corruption or type error on a METERED
                    # call — the session cost ceiling will not trigger.
                    # WARN once per (provider, model) so operators see it
                    # without a per-call flood; repeats drop to DEBUG.
                    warn_key = (
                        str(resp.provider or provider_key),
                        str(resp.model or model_override or ""),
                    )
                    if warn_key not in self._cost_record_warned:
                        self._cost_record_warned.add(warn_key)
                        logger.warning(
                            "CostTracker.record failed (provider=%s model=%s): %s — cost ceiling will not trigger for these calls (further failures for this provider/model logged at DEBUG)",
                            warn_key[0],
                            warn_key[1],
                            e,
                        )
                    else:
                        logger.debug(
                            "CostTracker.record failed (provider=%s model=%s): %s",
                            warn_key[0],
                            warn_key[1],
                            e,
                        )
                    cost_usd = 0.0
                usage["cost_usd"] = cost_usd
                self._session_cost += cost_usd

                # Cloud audit is for CLOUD calls — data leaving to a third
                # party, with cost attached. Peer-tunnel/self-hosted traffic
                # was filling data/logs/cloud_audit.jsonl (13 MB in a day)
                # and emitting a console "cloud_audit" INFO per call
                # (2026-08-04 live session). Gate on the same metering
                # predicate record() uses.
                if treat_as_cloud:
                    if redaction_result is None:
                        redaction_result = RedactionResult(
                            system=redacted_system,
                            user=redacted_user,
                            categories_sent=[],
                            categories_redacted=[],
                            policy="unknown",
                        )
                    self._emit_cloud_audit(
                        provider_key=provider_key,
                        model=resp.model or model_override,
                        usage=usage,
                        cost_usd=cost_usd,
                        redaction=redaction_result,
                        request_context=request_context,
                    )
                return resp.content, usage

        # Path C: fallback — treat user string as prompt
        text = backend.complete(
            redacted_user,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=tuple(getattr(self.cfg, "stop", ("</s>",))),
        )
        if text:
            self._note_provider_success(provider_key, model=model_override or model, backend=backend)
        return text, None

    @staticmethod
    def _log_provider_error(provider_key: str, exc: Exception) -> None:
        """Classify a backend exception and emit an actionable warning."""
        err_msg = str(exc).lower()
        is_auth = "401" in err_msg or "403" in err_msg or "unauthorized" in err_msg
        is_not_found = "404" in err_msg or "not found" in err_msg
        if is_auth:
            warn(
                "LLM auth failed (%s): %s. Check API key for provider '%s'.",
                provider_key,
                exc,
                provider_key,
            )
        elif is_not_found:
            warn(
                "LLM model not found (%s): %s. Verify model name in config.",
                provider_key,
                exc,
            )
        else:
            warn("LLM complete failed (%s): %s", provider_key, exc)

    def preview_provider(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Preview which provider would likely be used for this prompt."""
        prompt_tokens = self._estimate_prompt_tokens(system, user)
        now = time.time()
        providers, budget_tier, totals = self._candidate_providers(prompt_tokens, max_tokens, now)
        if not providers:
            return {
                "provider": "",
                "is_cloud": False,
                "model": "",
                "budget_tier": "blocked",
                "prompt_tokens": prompt_tokens,
                "n_ctx": 0,
            }
        provider_key = providers[0]
        provider_cfg = self._providers.get(provider_key, {})
        model = self._provider_model(provider_cfg)
        model_override = self._model_for_tier(model, provider_cfg, budget_tier)
        return {
            "provider": provider_key,
            "is_cloud": self._provider_is_cloud(provider_cfg),
            "model": model_override,
            "budget_tier": budget_tier,
            "prompt_tokens": prompt_tokens,
            "n_ctx": self._provider_n_ctx(provider_cfg),
            "cost_visible": self._provider_cost_visible(provider_cfg),
            "totals": totals,
        }

    def get_cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    def cloud_allowed(self) -> bool:
        return bool(self._cloud_allowed)

    def get_routing_policy(self) -> RoutingPolicy:
        return self._routing_policy

    def cloud_block_reason(self) -> str:
        return self._cloud_block_reason

    def has_cost_visible_provider(self) -> bool:
        for cfg in self._providers.values():
            if self._provider_is_cloud(cfg) and self._provider_cost_visible(cfg):
                return True
        return False

    def get_provider_configs(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self._providers.items()}

    def warmup(self) -> bool:
        """
        Pre-load the LLM model at startup to avoid first-request latency.

        Call this after other initialization is complete. The model loading
        happens in a background thread so it doesn't block startup.

        Returns True if warmup was initiated, False if LLM is disabled.
        """
        if not self.enabled():
            return False

        import threading
        import time

        def _warmup_thread():
            start_time = time.time()
            log_agentic(
                "llm_router", "startup", {"status": "loading", "model": str(getattr(self.cfg, "model_path", ""))[-50:]}
            )
            backend = self._get_backend()
            success = False
            if backend is not None and hasattr(backend, "warmup"):
                info("Warming up LLM backend...")
                if backend.warmup():
                    elapsed = time.time() - start_time
                    info("LLM model loaded and ready")
                    log_agentic("llm_router", "startup", {"status": "ready", "load_time_s": round(elapsed, 1)})
                    success = True
                else:
                    warn("LLM warmup failed")
                    log_agentic(
                        "llm_router", "error", {"context": "warmup", "error": "warmup returned false"}, level="WARNING"
                    )
            elif backend is not None:
                # Backend doesn't have warmup, try a minimal completion to load
                info("Warming up LLM backend (no warmup method, using test prompt)...")
                try:
                    backend.complete(
                        prompt="Hello",
                        max_tokens=1,
                        temperature=0.0,
                        stop=(),
                    )
                    elapsed = time.time() - start_time
                    info("LLM model loaded and ready")
                    log_agentic("llm_router", "startup", {"status": "ready", "load_time_s": round(elapsed, 1)})
                    success = True
                except Exception as e:
                    warn("LLM warmup failed: %s", e)
                    log_agentic("llm_router", "error", {"context": "warmup", "error": str(e)[:50]}, level="WARNING")
            self._warmup_failed = not success
            self._ready_event.set()

        thread = threading.Thread(target=_warmup_thread, daemon=True, name="LLMWarmup")
        thread.start()
        return True

    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Block until the LLM is loaded and ready, or timeout expires.

        Call after warmup() to ensure the model is available before
        submitting requests. In robot mode you typically don't call this
        (warmup is async, fallback handles the gap). In simulation mode
        or multi-LLM setups, call this to guarantee readiness.

        Args:
            timeout: Maximum seconds to wait. Default 120s for large models on CPU.

        Returns:
            True if LLM is ready, False if warmup failed or timed out.
        """
        if not self.enabled():
            return False
        if self._ready_event.wait(timeout=timeout):
            return not self._warmup_failed
        warn("LLM wait_ready timed out after %.1fs", timeout)
        return False

    @property
    def is_ready(self) -> bool:
        """Check if the LLM has finished loading (non-blocking)."""
        return self._ready_event.is_set() and not self._warmup_failed

    def _get_backend(self) -> Any | None:
        # Fast path: already initialized
        if self._backend is not None:
            return None if self._backend is LLMRouter._INIT_FAILED else self._backend

        # Thread-safe initialization
        with self._backend_lock:
            if self._backend is not None:
                return None if self._backend is LLMRouter._INIT_FAILED else self._backend

            if not self.enabled():
                self._backend = LLMRouter._INIT_FAILED
                return None

            provider_key = self._default_provider()
            backend = self._get_backend_for_provider(provider_key)
            self._backend = backend if backend is not None else LLMRouter._INIT_FAILED
            return backend

    def route(
        self,
        transcript_text: str,
        *,
        allowed_tools: set[str],
        allowed_commands: set[str],
    ) -> dict[str, Any] | None:
        if not self.enabled():
            return None

        tools = ", ".join(sorted(allowed_tools))
        commands = ", ".join(sorted(allowed_commands))

        system = _SYSTEM_ROUTE
        user = f"""
Transcript:
{transcript_text}

Allowed tools: {tools}
If tool_name == "maxim_command", command must be one of: {commands}

Return JSON exactly like:
{{"tool_name":"...","params":{{...}}}}
""".strip()
        text, _usage = self._complete_text(
            system,
            user,
            temperature=float(self.cfg.temperature),
            max_tokens=int(self.cfg.max_tokens),
        )

        obj = _extract_json_object(text)
        if not isinstance(obj, dict):
            return None

        tool_name = obj.get("tool_name")
        params = obj.get("params") if isinstance(obj.get("params"), dict) else {}
        if not isinstance(tool_name, str) or not tool_name or tool_name not in allowed_tools:
            return None

        if tool_name == "maxim_command":
            cmd = params.get("command")
            if not isinstance(cmd, str) or not cmd or cmd not in allowed_commands:
                return None

        return {"tool_name": tool_name, "params": dict(params)}

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        *,
        provider_hint: str | None = None,
        request_context: dict[str, Any] | None = None,
        system_override: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | None:
        """Generate a JSON response from a prompt.

        Used by LLMWorker for agentic goal proposal.

        Args:
            prompt: The full prompt to send to the LLM (used as user message).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            provider_hint: Optional provider key to prefer.
            request_context: Metadata for audit logs (agent, request_id, lane).
            system_override: Override the default JSON-only system prompt.
            tools: Optional tool definitions for native tool use (Claude/OpenAI).
            thinking: Optional extended thinking config (e.g. {"budget_tokens": 5000}).
            stream: Whether to stream the response.

        Returns:
            Parsed JSON dict or None if generation failed.
        """
        if not self.enabled():
            return None
        backend = self._get_backend()

        # Two-stage approach for ANSWER_ONLY prompts
        # Stage 1: Get plain text answer (easier for small models)
        # Stage 2: Wrap in JSON programmatically
        if prompt.startswith("ANSWER_ONLY|"):
            question = prompt[len("ANSWER_ONLY|") :].strip()
            return self._generate_answer_only(
                backend,
                question,
                temperature,
                max_tokens,
                provider_hint=provider_hint,
                request_context=request_context,
            )

        # Tool-aware prompt with full context
        if prompt.startswith("TOOL_PROMPT|"):
            tool_prompt = prompt[len("TOOL_PROMPT|") :].strip()
            return self._generate_tool_response(
                backend,
                tool_prompt,
                temperature,
                max_tokens,
                provider_hint=provider_hint,
                request_context=request_context,
            )

        # Standard JSON generation path
        # Use a strict system prompt
        system = system_override or _SYSTEM_JSON_ONLY

        text, usage = self._complete_text(
            system,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_hint=provider_hint,
            request_context=request_context,
            tools=tools,
            thinking=thinking,
            stream=stream,
        )
        if text:
            logger.debug("LLM raw response (first 200 chars): %s", text[:200] if len(text) > 200 else text)

        obj = _extract_json_object(text)
        if not isinstance(obj, dict):
            warn("LLM returned non-dict: %s (raw text was: %s)", type(obj), text[:100] if text else "empty")
            return None

        if usage:
            if "usage" not in obj:
                obj["usage"] = usage
            else:
                obj["_usage"] = usage

        return obj

    def _generate_answer_only(
        self,
        backend: Any,
        question: str,
        temperature: float,
        max_tokens: int,
        *,
        provider_hint: str | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Two-stage answer generation: get plain text, then wrap in JSON.

        This approach works better with small local models that struggle
        with JSON formatting instructions.
        """
        # Stage 1: Ask for a plain text answer (no JSON complexity)
        system = (
            "You are Maxim, a helpful robot assistant. "
            "Answer the user's question directly and concisely. "
            "Give only the answer, no explanations or preamble."
        )
        user = f"Question: {question}\nAnswer:"
        text, usage = self._complete_text(
            system,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_hint=provider_hint,
            request_context=request_context,
        )

        if not text or not text.strip():
            warn("LLM returned empty answer for question: %s", question)
            return None

        # Clean up the answer
        answer = text.strip()

        # Remove common prefixes the LLM might add
        prefixes_to_remove = [
            "answer:",
            "the answer is:",
            "response:",
            "here is the answer:",
            "a:",
            "q:",
            "the answer:",
        ]
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix) :].strip()
                break

        # Note: Response length is controlled by max_tokens in config
        # No truncation here - let the full response through

        logger.debug("LLM answer_only response: %s", answer[:500] if len(answer) > 500 else answer)

        # Stage 2: Wrap in JSON programmatically (no LLM needed)
        # Escape the answer for JSON
        escaped_answer = answer.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        # Field order: action → confidence → reasoning (most important first)
        result = {
            "action": {
                "tool_name": "respond",
                "params": {"message": escaped_answer},
            },
            "confidence": 0.85,
            "reasoning": "answer_only",
        }
        if usage:
            result["usage"] = usage
        return result

    def _generate_tool_response(
        self,
        backend: Any,
        tool_prompt: str,
        temperature: float,
        max_tokens: int,
        *,
        provider_hint: str | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Generate a tool-aware response with full context.

        This method handles complex prompts that include:
        - Mode context and goals
        - Available tools
        - Percepts and observations
        - Agent states

        Returns a structured response with action, reasoning, and optional next_actions.
        """
        # Detect PLANNING mode from the prompt banner. Match against key phrase
        # that appears in both old and new banner formats. Detect on the FULL
        # payload BEFORE splitting — the planning banner / instructions are in
        # the cacheable (stable) segment, so detecting on the dynamic remainder
        # alone would miss it.
        is_planning_mode = "PLANNING MODE" in tool_prompt and "APPROVAL" in tool_prompt

        # Prompt-caching split (prompt_caching_for_cloud_backends.md Phase 1):
        # PromptBuilder emits ``<stable_prefix>\x1e<dynamic_remainder>``. Route
        # the byte-stable prefix into the system message (where the cloud
        # backends' cache_control breakpoint lives) and the dynamic remainder
        # into the user message. Payloads without the delimiter (followup
        # prompts, legacy callers) carry no stable prefix — everything stays in
        # the user message exactly as before.
        from maxim.agents.prompt_builder import PROMPT_SEGMENT_DELIMITER

        stable_prefix = ""
        if PROMPT_SEGMENT_DELIMITER in tool_prompt:
            stable_prefix, tool_prompt = tool_prompt.split(PROMPT_SEGMENT_DELIMITER, 1)

        if is_planning_mode:
            # PLANNING MODE: Allow proposal text followed by JSON
            system = (
                "You are Maxim, an intelligent robot assistant. "
                "You are in PLANNING mode and need user approval before acting. "
                "Follow the response format instructions in the prompt exactly."
            )
            user = f"""{tool_prompt}

IMPORTANT: Follow the PLANNING MODE format exactly as shown above.
First write your proposal in plain text, then <|action_json|>, then the JSON."""
        else:
            # NORMAL MODE: JSON-only response
            system = _SYSTEM_TOOL_RESPONSE
            user = f"""{tool_prompt}

Respond with ONLY a valid JSON object. No text before or after the JSON."""

        # Append the stable prefix AFTER the base system instruction so the
        # whole system block (static constant + stable prefix) is byte-identical
        # across turns and the single cache_control breakpoint at its end caches
        # all of it.
        if stable_prefix.strip():
            system = f"{system}\n\n{stable_prefix}"

        if is_planning_mode:
            info("PLANNING mode detected - expecting proposal + <|action_json|> + JSON format")

        text, usage = self._complete_text(
            system,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_hint=provider_hint,
            request_context=request_context,
        )
        if text:
            logger.debug("LLM tool_response raw (first 300 chars): %s", text[:300] if len(text) > 300 else text)

        if not text or not text.strip():
            warn("LLM returned empty tool response")
            return None

        # Check for planning mode delimiter: plan text followed by <|action_json|> followed by JSON
        plan_text = None
        action_json_delimiter = "<|action_json|>"
        if action_json_delimiter in text:
            parts = text.split(action_json_delimiter, 1)
            plan_text = parts[0].strip()
            text = parts[1].strip() if len(parts) > 1 else ""
            info("Planning mode: extracted plan_text (%d chars) and JSON", len(plan_text))
        elif is_planning_mode:
            # PLANNING MODE FALLBACK: LLM output raw JSON without the delimiter
            # We'll generate a synthetic proposal after extracting the JSON
            info("PLANNING mode but no delimiter found - will generate synthetic proposal")

        # Extract JSON from response
        obj = _extract_json_object(text)
        if not isinstance(obj, dict):
            warn("LLM tool_response returned non-dict: %s", type(obj))
            # Try to salvage by extracting message from JSON-like text
            clean_text = text.strip()

            # If text looks like JSON with a respond action, extract the message
            # Pattern: "message": "actual message content"
            if '"message"' in clean_text:
                import re

                msg_match = re.search(r'"message"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', clean_text)
                if msg_match:
                    # Extract and unescape the message
                    extracted_msg = msg_match.group(1).replace('\\"', '"')
                    if extracted_msg and len(extracted_msg) > 10:
                        info("Salvaged message from failed JSON parse: %s", extracted_msg[:50])
                        return {
                            "action": {
                                "tool_name": "respond",
                                "params": {"message": extracted_msg},
                            },
                            "confidence": 0.5,
                            "reasoning": "salvaged_from_json",
                        }

            # Last resort: wrap non-JSON text (but NOT raw JSON)
            if clean_text and not clean_text.startswith("{"):
                return {
                    "action": {
                        "tool_name": "respond",
                        "params": {"message": clean_text[:500]},
                    },
                    "confidence": 0.5,
                    "reasoning": "fallback_parse",
                }
            return None

        # Validate action structure
        action = obj.get("action")
        if action and isinstance(action, dict):
            if "tool_name" not in action:
                # Try to extract tool_name from flat structure
                if "tool_name" in obj:
                    action = {"tool_name": obj["tool_name"], "params": obj.get("params", {})}
                    obj["action"] = action

        # Ensure required fields
        if "confidence" not in obj:
            obj["confidence"] = 0.7
        if "reasoning" not in obj:
            obj["reasoning"] = ""
        if "mode_goal_achieved" not in obj:
            obj["mode_goal_achieved"] = False

        # Add planning mode fields if present
        if plan_text:
            obj["_plan_text"] = plan_text
            obj["_requires_approval"] = True
        elif is_planning_mode:
            # PLANNING MODE FALLBACK: Generate synthetic proposal from the action
            # This handles cases where the LLM outputs raw JSON despite instructions
            action = obj.get("action", {})
            tool_name = action.get("tool_name", "unknown") if isinstance(action, dict) else "unknown"
            params = action.get("params", {}) if isinstance(action, dict) else {}

            # Generate a user-friendly proposal based on the tool
            if tool_name == "internet_search":
                query = params.get("query", "information")
                synthetic_plan = f"I'd like to search the internet for: {query}\n\nMay I proceed with this search?"
            elif tool_name == "read_file":
                path = params.get("path", "a file")
                synthetic_plan = f"I'd like to read the file: {path}\n\nMay I proceed?"
            elif tool_name == "write_file":
                path = params.get("path", "a file")
                synthetic_plan = f"I'd like to write to the file: {path}\n\nMay I proceed?"
            elif tool_name == "http_fetch":
                url = params.get("url", "a URL")
                synthetic_plan = f"I'd like to fetch content from: {url}\n\nMay I proceed?"
            elif tool_name == "respond":
                # For respond, no approval needed - just pass through
                synthetic_plan = None
            else:
                synthetic_plan = f"I'd like to execute the '{tool_name}' action.\n\nMay I proceed?"

            if synthetic_plan:
                info("Generated synthetic proposal for PLANNING mode: %s", synthetic_plan[:100])
                obj["_plan_text"] = synthetic_plan
                obj["_requires_approval"] = True

        if usage:
            if "usage" not in obj:
                obj["usage"] = usage
            else:
                obj["_usage"] = usage

        return obj
