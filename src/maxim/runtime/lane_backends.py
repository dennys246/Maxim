"""Per-lane LLM backend construction (multi-LLM Phase 3 + 4).

LaneBackendManager creates and caches one backend per WorkerPool lane, driven
by the lane's LaneConfig. Backends are loaded lazily on first access.

Safety gates (env-driven, all defaulted to safe values):
- MAXIM_MAX_CONCURRENT_BACKENDS (default 2) — hard cap on cached backends,
  refusing to create more when exceeded. Protects VRAM/RAM.
- MAXIM_MAX_CLOUD_LANES (default 0) — hard cap on lanes using cloud endpoints.
  Zero by default forces users to opt in; the session cost ceiling in
  LLMRouter provides a second layer.

Remote/cloud classification:
- remote_url with private-IP / localhost host → "self-hosted" (no cloud gate)
- remote_url with public host → "cloud" (gated by MAXIM_MAX_CLOUD_LANES
  AND the existing LLMRouter cloud_enabled flag)
"""

from __future__ import annotations

import dataclasses
import os
import socket
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from maxim.runtime.worker_pool import LaneConfig


# ─── active server tracking (for hot-swap) ────────────────────────────────
_active_spawner: Any | None = None
_active_model: str | None = None
_swap_lock = threading.Lock()

# ─── env var plumbing ─────────────────────────────────────────────────────

_DEFAULT_MAX_BACKENDS = 2
_DEFAULT_MAX_CLOUD_LANES = 0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


# ─── URL classification (cloud vs self-hosted) ────────────────────────────


def _host_is_private(host: str) -> bool:
    """True if host is localhost, a private IP, or resolves to one.

    Mirrors the SSRF logic in openai_backend.py so classification stays
    consistent. Resolution failure is treated as "not private" (fail safe —
    assume cloud if we can't tell).
    """
    if not host:
        return False
    try:
        import ipaddress

        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
    except ValueError:
        pass
    if host.lower() in ("localhost", "local.home", "local"):
        return True
    try:
        addrinfos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except Exception:
        return False
    for info in addrinfos:
        ip = info[4][0]
        try:
            import ipaddress

            addr = ipaddress.ip_address(ip)
            if not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved):
                return False
        except ValueError:
            return False
    return True


def _is_cloud_url(remote_url: str | None) -> bool:
    """Classify a remote_url as cloud (True) or self-hosted (False).

    None/empty → False (handled elsewhere). Unparseable → True (fail safe:
    treat as cloud so gates engage).
    """
    if not remote_url:
        return False
    try:
        parsed = urlparse(remote_url)
    except Exception:
        return True
    host = parsed.hostname or ""
    return not _host_is_private(host)


# ─── exceptions ───────────────────────────────────────────────────────────


class BackendGateError(RuntimeError):
    """Raised when a safety gate refuses to construct a backend."""


# ─── manager ──────────────────────────────────────────────────────────────


class LaneBackendManager:
    """Owns the lane → backend mapping for multi-LLM operation."""

    def __init__(
        self,
        lane_configs: dict[str, LaneConfig],
        *,
        max_backends: int | None = None,
        max_cloud_lanes: int | None = None,
        peer_owned: bool = False,
    ) -> None:
        self._configs: dict[str, LaneConfig] = dict(lane_configs)
        self._backends: dict[str, Any] = {}
        self._cloud_lanes: set[str] = set()
        self._peer_owned = peer_owned
        self._lock = threading.Lock()
        self._max_backends = (
            max_backends
            if max_backends is not None
            else _env_int("MAXIM_MAX_CONCURRENT_BACKENDS", _DEFAULT_MAX_BACKENDS)
        )
        self._max_cloud_lanes = (
            max_cloud_lanes
            if max_cloud_lanes is not None
            else _env_int("MAXIM_MAX_CLOUD_LANES", _DEFAULT_MAX_CLOUD_LANES)
        )
        # Phase 8: per-lane metrics (shared with LeaderProxy via singleton)
        from maxim.models.language.lane_metrics import get_metrics_registry

        self._metrics_registry = get_metrics_registry()
        # Pre-create metrics for all configured lanes
        for lane_name in self._configs:
            self._metrics_registry.get(lane_name)

    @property
    def lanes(self) -> tuple[str, ...]:
        return tuple(self._configs.keys())

    @property
    def max_backends(self) -> int:
        return self._max_backends

    @property
    def max_cloud_lanes(self) -> int:
        return self._max_cloud_lanes

    def get_lane_kind(self, lane: str) -> str:
        """Return the classification of a lane: 'cloud', 'self-hosted', or 'local'."""
        cfg = self._configs.get(lane)
        if cfg is None:
            return "local"
        return self._classify(cfg)

    def metrics_snapshot(self) -> dict[str, dict[str, Any]]:
        """Thread-safe snapshot of all per-lane metrics."""
        return self._metrics_registry.snapshot()

    def get_metrics(self, lane: str) -> Any:
        """Get the LaneMetrics instance for a lane."""
        return self._metrics_registry.get(lane)

    def describe(self) -> dict[str, dict[str, Any]]:
        """Snapshot of lane assignments, for logging/diagnostics."""
        out: dict[str, dict[str, Any]] = {}
        for lane, cfg in self._configs.items():
            out[lane] = {
                "profile": cfg.model_profile,
                "device": cfg.device,
                "n_gpu_layers": cfg.n_gpu_layers,
                "remote_url": cfg.remote_url,
                "kind": self._classify(cfg),
                "loaded": lane in self._backends,
            }
        return out

    def get_backend(self, lane: str) -> Any | None:
        """Return the backend for `lane`, constructing it on first call."""
        cfg = self._configs.get(lane)
        if cfg is None:
            return None
        if not cfg.model_profile and not cfg.remote_url:
            return None

        with self._lock:
            cached = self._backends.get(lane)
            if cached is not None:
                return cached
            self._check_backend_cap()
            kind = self._classify(cfg)
            if kind == "cloud":
                self._check_cloud_lane_cap(lane)
            backend = self._build_backend(cfg)
            if backend is not None:
                self._backends[lane] = backend
                if kind == "cloud":
                    self._cloud_lanes.add(lane)
            return backend

    def unload_all(self) -> None:
        """Release all cached backends. Best-effort; swallows exceptions."""
        with self._lock:
            for backend in self._backends.values():
                unload = getattr(backend, "unload", None)
                if callable(unload):
                    try:
                        unload()
                    except Exception:
                        pass
            self._backends.clear()
            self._cloud_lanes.clear()

    # ─── classification ───────────────────────────────────────────────────

    def _classify(self, cfg: LaneConfig) -> str:
        if cfg.remote_url:
            # Peer-owned URLs (from `maxim peer connect`) are your own
            # infrastructure behind a tunnel — not cloud providers.
            # The flag is set by build_primary_router when peer config is loaded.
            if self._peer_owned:
                return "self-hosted"
            return "cloud" if _is_cloud_url(cfg.remote_url) else "self-hosted"
        return "local"

    # ─── gates ────────────────────────────────────────────────────────────

    def _check_backend_cap(self) -> None:
        if len(self._backends) >= self._max_backends:
            raise BackendGateError(
                f"Refusing to create backend: already caching {len(self._backends)} "
                f"(limit MAXIM_MAX_CONCURRENT_BACKENDS={self._max_backends}). "
                "Raise the env var or call unload_all() first."
            )

    def _check_cloud_lane_cap(self, lane: str) -> None:
        if lane in self._cloud_lanes:
            return  # already counted
        if len(self._cloud_lanes) >= self._max_cloud_lanes:
            raise BackendGateError(
                f"Refusing to create cloud backend for lane '{lane}': "
                f"cloud-lane count would exceed MAXIM_MAX_CLOUD_LANES="
                f"{self._max_cloud_lanes}. Raise the env var to permit cloud use."
            )

    # ─── backend construction ─────────────────────────────────────────────

    def _build_backend(self, cfg: LaneConfig) -> Any | None:
        if cfg.remote_url:
            return self._build_remote_backend(cfg)
        return self._build_local_backend(cfg)

    def _build_local_backend(self, cfg: LaneConfig) -> Any | None:
        """Construct a local LLMRouter for a lane."""
        try:
            from maxim.models.language.config import load_llm_config
            from maxim.models.language.router import LLMRouter
        except Exception:
            return None

        env_profile = os.environ.get("MAXIM_LLM_PROFILE", "").strip()
        profile = env_profile or cfg.model_profile
        try:
            llm_config = load_llm_config(profile_override=profile)
        except Exception:
            return None

        if "MAXIM_LLM_N_GPU_LAYERS" not in os.environ:
            try:
                llm_config = dataclasses.replace(llm_config, n_gpu_layers=cfg.n_gpu_layers)
            except Exception:
                pass

        # Inject cloud fallback provider if configured via --cloud-fallback
        llm_config = self._maybe_inject_cloud_fallback(cfg, llm_config)

        try:
            return LLMRouter(llm_config)
        except Exception:
            return None

    def _maybe_inject_cloud_fallback(self, cfg: "LaneConfig", llm_config: Any) -> Any:
        """Add a cloud fallback provider to the infer lane's LLMConfig.

        When --cloud-fallback is set, the CLI stores the resolved model name
        in MAXIM_CLOUD_FALLBACK_MODEL.  We inject it as a secondary provider
        so the router tries the primary (local/self-hosted) first and falls
        back to cloud on failure, rate-limit, or timeout.
        """
        if cfg.name != "infer":
            return llm_config
        fallback_model = os.environ.get("MAXIM_CLOUD_FALLBACK_MODEL", "").strip()
        if not fallback_model:
            return llm_config

        try:
            from maxim.models.language.config import _BUILTIN_PROFILES
        except Exception:
            return llm_config

        profile_data = _BUILTIN_PROFILES.get(fallback_model, {})
        if not profile_data.get("cloud"):
            return llm_config

        backend_type = profile_data.get("backend", "anthropic")
        api_key_env = profile_data.get("api_key_env", "ANTHROPIC_API_KEY")
        n_ctx = profile_data.get("n_ctx", 200000)

        # Build providers dict — ensure local provider exists first
        providers = dict(llm_config.providers or {})
        if not providers:
            providers["local"] = {
                "type": llm_config.backend,
                "model": llm_config.model,
                "n_ctx": llm_config.n_ctx,
            }
        providers["cloud-fallback"] = {
            "type": backend_type,
            "model": fallback_model,
            "api_key_env": api_key_env,
            "n_ctx": n_ctx,
        }

        # Append cloud-fallback to priority (after primary)
        routing = dict(llm_config.routing or {})
        priority = list(routing.get("provider_priority", list(providers.keys())))
        if "cloud-fallback" not in priority:
            priority.append("cloud-fallback")
        routing["provider_priority"] = priority

        # Apply session budget override if set
        budget = os.environ.get("MAXIM_CLOUD_SESSION_BUDGET", "").strip()
        if budget:
            try:
                routing["max_session_cost"] = float(budget)
            except ValueError:
                pass

        return dataclasses.replace(
            llm_config,
            cloud_enabled=True,
            providers=providers,
            routing=routing,
        )

    def _build_remote_backend(self, cfg: LaneConfig) -> Any | None:
        """Construct an OpenAI-compatible remote backend via LLMRouter.

        Wraps _OpenAIBackend in an LLMRouter so downstream consumers get the
        same interface as local lanes (warmup, complete_with_usage, etc.).

        Self-hosted targets (llama-cpp-server, Ollama, vLLM on private IP)
        skip the cloud_enabled requirement. Cloud providers (Anthropic, OpenAI)
        require cloud_enabled=True and count against MAXIM_MAX_CLOUD_LANES.
        """
        try:
            from maxim.models.language.config import load_llm_config
            from maxim.models.language.router import LLMRouter
        except Exception:
            return None

        kind = self._classify(cfg)
        try:
            base = load_llm_config()
        except Exception:
            return None

        provider_key = f"lane-{cfg.name}"
        api_key_env = f"MAXIM_LANE_{cfg.name.upper()}_API_KEY"
        if cfg.remote_api_key:
            os.environ.setdefault(api_key_env, cfg.remote_api_key)
        else:
            os.environ.setdefault(api_key_env, "not-needed")

        providers = dict(base.providers or {})
        providers[provider_key] = {
            "base_url": cfg.remote_url,
            "api_key_env": api_key_env,
            "model": cfg.remote_model or cfg.model_profile or base.model,
            "allow_local_endpoints": (kind == "self-hosted"),
            # Peer-tunnel URLs resolve to Cloudflare IPs (classified as cloud)
            # but actually serve a self-hosted model with no metered pricing.
            # Skip the cost-estimation gate that would reject the provider
            # when no pricing entry exists for the model name.
            "pricing_required": False,
        }

        # Ensure the lane provider appears in routing.provider_priority so
        # _provider_order() doesn't silently exclude it when the user's
        # llm.json defines an explicit priority list.
        routing = dict(base.routing or {})
        priority = list(routing.get("provider_priority", []))
        if provider_key not in priority:
            priority.insert(0, provider_key)
        routing["provider_priority"] = priority

        try:
            remote_cfg = dataclasses.replace(
                base,
                enabled=True,
                backend="openai",
                model=cfg.remote_model or cfg.model_profile or base.model,
                providers=providers,
                routing=routing,
                # Self-hosted is not cloud; cloud lanes still require cloud_enabled.
                cloud_enabled=(True if kind == "cloud" else base.cloud_enabled),
            )
        except Exception:
            return None

        try:
            return LLMRouter(remote_cfg)
        except Exception:
            return None


def _apply_cloud_cli_overrides(
    lane_configs: dict[str, Any],
    logger: Any | None,
) -> dict[str, Any]:
    """Apply --cloud-lane env vars set by cli.py.

    Rewrites the named lane to use the cloud model profile as its primary.
    The cloud fallback (--cloud-fallback) is handled later in
    _build_local_backend, which injects the fallback provider into the
    LLMRouter's provider dict.
    """
    out = dict(lane_configs)
    for lane_name, cfg in list(out.items()):
        cloud_model = os.environ.get(f"MAXIM_CLOUD_LANE_{lane_name.upper()}_MODEL", "").strip()
        if not cloud_model:
            continue
        try:
            from maxim.models.language.config import _BUILTIN_PROFILES
        except Exception:
            continue
        profile_data = _BUILTIN_PROFILES.get(cloud_model, {})
        if not profile_data.get("cloud"):
            if logger is not None:
                logger.warning(
                    "MAXIM_CLOUD_LANE_%s_MODEL=%s is not a cloud profile — ignoring",
                    lane_name.upper(),
                    cloud_model,
                )
            continue
        # Rewrite the lane to use the cloud profile as primary
        out[lane_name] = dataclasses.replace(cfg, model_profile=cloud_model)
        if logger is not None:
            logger.info("Lane '%s' assigned cloud model: %s", lane_name, cloud_model)
    return out


def build_primary_router(
    capabilities: Any | None = None,
    *,
    logger: Any | None = None,
) -> tuple[Any | None, "LaneBackendManager"]:
    """Build the infer-lane LLMRouter + its owning LaneBackendManager.

    Single source of truth for primary LLM construction. Both the agentic
    runtime and the simulation orchestrator call this so they share exactly
    the same lane/remote/gating logic — no more duplicate model loads.

    Pipeline:
      1. Detect RuntimeCapabilities if not provided.
      2. Clone DEFAULT_LANES + apply capability-driven profile assignments
         (Phase 2).
      3. Apply MAXIM_LANE_{NAME}_REMOTE_URL / _MODEL / _API_KEY env overrides
         (Phase 4).
      4. Construct LaneBackendManager with the gates (Phase 3 + 4 safety).
      5. Return (manager.get_backend("infer"), manager).

    Returns (None, manager) if the infer lane resolves to nothing — e.g. the
    lane has no profile AND no remote URL, the backend build fails, or the
    LLMConfig ends up disabled. Callers should fall back to their own default
    LLMRouter construction in that case.

    Raises:
        BackendGateError: if the safety gates (MAXIM_MAX_CONCURRENT_BACKENDS
        or MAXIM_MAX_CLOUD_LANES) refuse the request. Surface this to the
        user; it means their configuration exceeds the declared limits.
    """
    from maxim.runtime.capabilities import RuntimeCapabilities, detect_compute_resources
    from maxim.runtime.lane_models import apply_lane_env_overrides, build_lane_model_config
    from maxim.runtime.worker_pool import DEFAULT_LANES

    # Peer-config auto-load: if ~/.config/maxim/peer.yml exists and env vars
    # aren't already set, populate them from the file. Set by
    # `maxim peer connect`. Env wins over file for per-session overrides.
    _has_peer_config = False
    try:
        from maxim.peer.config import apply_peer_config_to_env, read_peer_config

        peer_cfg = read_peer_config()
        if peer_cfg is not None:
            apply_peer_config_to_env(peer_cfg)
            _has_peer_config = True
    except Exception:
        pass

    # ── Reconcile --language-model with peer remote config ────────────
    # When peer config sets a remote URL for the infer lane AND the user
    # specified --language-model (MAXIM_LLM_PROFILE), the intent is
    # "run this model on the remote server" — not "load it locally."
    # Redirect: copy the profile name into MAXIM_LANE_INFER_REMOTE_MODEL
    # and clear MAXIM_LLM_PROFILE so the local backend machinery doesn't
    # activate and try to load a local pytorch/llama_cpp backend.
    if _has_peer_config:
        _remote_url = os.environ.get("MAXIM_LANE_INFER_REMOTE_URL", "").strip()
        _llm_profile = os.environ.get("MAXIM_LLM_PROFILE", "").strip()
        if _remote_url and _llm_profile:
            os.environ.setdefault("MAXIM_LANE_INFER_REMOTE_MODEL", _llm_profile)
            os.environ.pop("MAXIM_LLM_PROFILE", None)

    if capabilities is None:
        has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()
        capabilities = RuntimeCapabilities(
            has_gpu=has_gpu,
            gpu_type=gpu_type,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
        )

    lane_configs = {name: dataclasses.replace(cfg) for name, cfg in DEFAULT_LANES.items()}

    try:
        lane_models = build_lane_model_config(
            capabilities,
            profile_available=_profile_has_local_file,
        )
        for lane_name, lm in lane_models.items():
            base = lane_configs.get(lane_name)
            if base is None:
                continue
            lane_configs[lane_name] = dataclasses.replace(
                base,
                model_profile=lm.profile,
                device=lm.device,
                n_gpu_layers=lm.n_gpu_layers,
            )
    except Exception as e:
        if logger is not None:
            logger.warning("Lane model assignment failed (using defaults): %s", e)

    lane_configs = apply_lane_env_overrides(lane_configs)
    lane_configs = _apply_cloud_cli_overrides(lane_configs, logger)

    # Health-check any user-supplied remote_url before wiring a lane to it.
    # Catches stale env vars pointing at servers that have since shut down —
    # drops the remote_url so auto-spawn can take over instead of failing
    # every request with a connection error.
    lane_configs = _validate_remote_urls(lane_configs, logger)

    # Auto-spawn a local llama-cpp-server if:
    #  - GPU is available
    #  - infer lane has a local profile (not already remote)
    #  - MAXIM_AUTO_SPAWN_LLM_SERVER != 0
    # When it fires, the infer lane is rewritten to point at the spawned server.
    lane_configs = _maybe_auto_spawn_server(capabilities, lane_configs, logger)

    manager = LaneBackendManager(lane_configs, peer_owned=_has_peer_config)

    if logger is not None:
        logger.info("Lane assignments: %s", manager.describe())

    _print_lane_banner(manager)

    # Start heartbeat monitor if enabled via env flags (peer or solo mode)
    try:
        from maxim.runtime.heartbeat import get_heartbeat_monitor, should_enable_heartbeat

        if should_enable_heartbeat():
            get_heartbeat_monitor().start()
    except Exception:
        pass

    router = manager.get_backend("infer")
    return router, manager


def _validate_remote_urls(lane_configs: dict[str, Any], logger: Any | None) -> dict[str, Any]:
    """Drop remote_url on lanes whose target server isn't responding.

    Only applies to loopback / private-IP URLs (where we can probe safely
    without external network calls). Public/cloud URLs are trusted as-is —
    probing them at startup would add latency + cost.
    """
    out = dict(lane_configs)
    for name, cfg in lane_configs.items():
        url = getattr(cfg, "remote_url", None)
        if not url:
            continue
        if _is_cloud_url(url):
            continue  # don't probe public endpoints at startup
        if _llm_server_responding_at(url):
            continue
        if logger is not None:
            logger.warning(
                "Lane '%s' remote_url=%s is unreachable — dropping it so auto-spawn "
                "can take over. Unset MAXIM_LANE_%s_REMOTE_URL to silence this.",
                name,
                url,
                name.upper(),
            )
        out[name] = dataclasses.replace(cfg, remote_url=None, remote_model=None, remote_api_key=None)
    return out


def _llm_server_responding_at(url: str, *, timeout_s: float = 1.5) -> bool:
    """Return True iff an OpenAI-compatible server answers GET /v1/models at `url`.

    Used for auto-discovery (reuse an already-running server) and for
    validating env-supplied remote URLs before wiring a lane to them.

    Treats both 200 and 401 as "server is up": 401 means an HTTP listener
    with auth enabled is answering, which is still a valid signal the port
    is in use by a real llama-cpp-server (we're just not authenticated).
    """
    if not url:
        return False
    # Normalize: strip trailing slashes, ensure /v1 path
    base = url.rstrip("/")
    probe = base + "/models" if base.endswith("/v1") else base + "/v1/models"
    try:
        import urllib.error
        import urllib.request

        with urllib.request.urlopen(probe, timeout=timeout_s) as resp:  # noqa: S310
            return resp.status == 200
    except urllib.error.HTTPError as e:
        return e.code == 401  # server reachable, auth-gated
    except Exception:
        return False


def _profile_has_local_file(profile_name: str) -> bool:
    """Return True iff the profile's GGUF file actually exists on disk.

    Used to filter the capability-driven tier table down to models the user
    has actually downloaded. Returns False on any error (missing profile,
    unbuildable path, etc.) — fail-closed so we fall back safely.
    """
    if not profile_name:
        return False
    try:
        from maxim.models.language.config import load_llm_config

        cfg = load_llm_config(profile_override=profile_name)
        model_path = getattr(cfg, "model_path", "") or ""
        return bool(model_path) and Path(model_path).is_file()
    except Exception:
        return False


def _maybe_auto_spawn_server(
    capabilities: Any,
    lane_configs: dict[str, Any],
    logger: Any | None,
) -> dict[str, Any]:
    """Auto-spawn a llama-cpp-server subprocess and rewrite the infer lane.

    No-op (returns input unchanged) if any of:
    - MAXIM_AUTO_SPAWN_LLM_SERVER is explicitly '0'/'false'/'off'
    - no GPU detected (spawner only speeds up GPU inference in practice)
    - infer lane already has a remote_url set (user opted out via override)
    - infer lane has no profile (no model to serve)
    - llama_cpp.server isn't importable (user didn't install [llm-server])
    - resolving the profile to a GGUF file path fails or file doesn't exist
    """
    auto_raw = os.environ.get("MAXIM_AUTO_SPAWN_LLM_SERVER", "").strip().lower()
    if auto_raw in ("0", "false", "f", "no", "n", "off"):
        return lane_configs

    if capabilities is None or not getattr(capabilities, "has_gpu", False):
        return lane_configs

    infer_cfg = lane_configs.get("infer")
    if infer_cfg is None or infer_cfg.remote_url or not infer_cfg.model_profile:
        return lane_configs

    try:
        import llama_cpp.server  # noqa: F401
    except ImportError:
        if logger is not None:
            logger.warning(
                "Auto-spawn skipped: llama_cpp.server not installed. "
                "Run: pip install -e '.[llm-server]' to enable local server spawning."
            )
        return lane_configs

    # Resolve the profile's GGUF path
    try:
        from maxim.models.language.config import load_llm_config

        profile_cfg = load_llm_config(profile_override=infer_cfg.model_profile)
        model_path = getattr(profile_cfg, "model_path", "")
    except Exception:
        return lane_configs
    if not model_path or not Path(model_path).is_file():
        if logger is not None:
            logger.warning(
                "Auto-spawn skipped: GGUF file not found at %s (profile=%s). "
                "Run ./scripts/download_models.sh --llm or adjust MAXIM_LLM_PROFILE.",
                model_path,
                infer_cfg.model_profile,
            )
        return lane_configs

    # Decide bind host + port from role
    try:
        from maxim.runtime.leader_mode import detect_role
    except Exception:

        class _Fallback:
            bind_host = "127.0.0.1"
            role = "solo"
            reason = "detect_role import failed"

        role_decision = _Fallback()
    else:
        role_decision = detect_role()

    try:
        port = int(os.environ.get("MAXIM_AUTO_SPAWN_PORT", "8100"))
    except ValueError:
        port = 8100

    # Load API key first — leader mode uses it both for spawning (--api_key
    # flag) and for wiring the leader's own client. Auto-discovery below
    # also needs it so reused servers get auth-wired into the lane.
    api_key: str | None = None
    if role_decision.role == "leader":
        try:
            from maxim.tunnel.keys import read_key

            api_key = read_key()
        except Exception:
            api_key = None

    # Auto-discovery: if something already answers at port, reuse it and skip spawn.
    # This makes "run two maxim terminals" transparent — the second finds the
    # first's server and wires its infer lane to it, no duplicate model load.
    existing_url = f"http://127.0.0.1:{port}/v1"
    if _llm_server_responding_at(existing_url):
        if logger is not None:
            logger.info(
                "Auto-discovery: found existing llama-cpp-server on port %d, reusing it",
                port,
            )
        out = dict(lane_configs)
        out["infer"] = dataclasses.replace(
            infer_cfg,
            remote_url=existing_url,
            remote_model=infer_cfg.model_profile,
            remote_api_key=infer_cfg.remote_api_key or api_key,
        )
        return out

    try:
        from maxim.runtime.local_server_spawner import LocalServerSpawner
    except Exception:
        return lane_configs

    # api_key was already loaded above (before auto-discovery) — reuse here
    # when spawning the subprocess. Solo mode (127.0.0.1 bind) has api_key=None
    # which means the server spawns without --api_key and accepts all requests.

    spawner = LocalServerSpawner(
        model_path=model_path,
        port=port,
        bind_host=role_decision.bind_host,
        n_ctx=int(os.environ.get("MAXIM_AUTO_SPAWN_N_CTX", "8192")),
        n_gpu_layers=infer_cfg.n_gpu_layers,
        api_key=api_key,
    )
    try:
        timeout_s = float(os.environ.get("MAXIM_AUTO_SPAWN_TIMEOUT_S", "120.0"))
    except ValueError:
        timeout_s = 120.0
    if logger is not None:
        auth_note = " (auth=on)" if api_key else ""
        logger.info(
            "Auto-spawning llama-cpp-server (model=%s port=%d host=%s role=%s timeout=%ds)%s",
            Path(model_path).name,
            port,
            role_decision.bind_host,
            role_decision.role,
            int(timeout_s),
            auth_note,
        )
    url = spawner.start(timeout_s=timeout_s)
    if url is None:
        if logger is not None:
            logger.warning("Auto-spawn failed; falling back to in-process inference")
        return lane_configs

    # Track active spawner for hot-swap via `maxim peer llm <model>`
    global _active_spawner, _active_model  # noqa: PLW0603
    _active_spawner = spawner
    _active_model = infer_cfg.model_profile

    # Rewrite the infer lane to point at the spawned server. Auto-wire the
    # API key for the leader's own client so local inference doesn't 401.
    out = dict(lane_configs)
    infer_api_key = infer_cfg.remote_api_key or api_key
    out["infer"] = dataclasses.replace(
        infer_cfg,
        remote_url=url,
        remote_model=infer_cfg.model_profile,
        remote_api_key=infer_api_key,
    )

    # Leader mode: also auto-spawn the cloudflared daemon alongside the LLM
    # server, so `maxim` on the leader brings up the full stack in one
    # command. No-op when daemon is already running (systemd service, etc.).
    if role_decision.role == "leader":
        # Auto-enable remote update for leaders unless explicitly disabled.
        # Leader already auth-gates all requests via Bearer token — remote
        # update is just another auth-gated action.
        os.environ.setdefault("MAXIM_ALLOW_REMOTE_UPDATE", "1")
        _maybe_auto_spawn_tunnel_daemon(logger)
        _maybe_start_leader_proxy(role_decision.bind_host, api_key, logger)
    return out


def swap_llm_server(profile: str, logger: Any | None = None) -> dict[str, Any]:
    """Hot-swap the llama-cpp-server to a different model.

    Called by LeaderProxy's /v1/admin/llm-swap endpoint.  Stops the current
    server, resolves the new profile to a GGUF path, and starts a fresh
    LocalServerSpawner.

    Returns a result dict consumed by the admin endpoint.
    Raises ValueError for client errors (400-level) and RuntimeError for
    server errors (500-level).
    """
    import time as _time

    from maxim.models.language.config import (
        _BUILTIN_PROFILES,
        list_llm_profiles,
        load_llm_config,
        normalize_llm_profile,
    )
    from maxim.runtime.local_server_spawner import LocalServerSpawner

    global _active_spawner, _active_model  # noqa: PLW0603

    # Resolve profile name
    resolved = normalize_llm_profile(profile)
    if not resolved:
        raise ValueError("Empty model name")

    profile_data = _BUILTIN_PROFILES.get(resolved, {})

    # Block cloud profiles
    if profile_data.get("cloud"):
        raise ValueError(f"Profile '{resolved}' is a cloud provider — cannot run on llama-cpp-server")

    # Block non-llama_cpp backends
    backend = profile_data.get("backend", "llama_cpp")
    if backend not in ("llama_cpp", "llamacpp", "llama"):
        raise ValueError(f"Profile '{resolved}' uses {backend} backend — not compatible with llama-cpp-server")

    # Validate profile exists
    available = list_llm_profiles()
    if available and resolved not in available:
        raise ValueError(f"Unknown model profile: {profile}. Available: {', '.join(sorted(available))}")

    # Acquire swap lock (non-blocking — reject concurrent swaps)
    if not _swap_lock.acquire(blocking=False):
        raise RuntimeError("LLM swap already in progress")

    try:
        # Same model = no-op
        if _active_model and _active_model == resolved and _active_spawner is not None:
            if _active_spawner.is_running:
                return {
                    "status": "already_running",
                    "model": resolved,
                }

        # Resolve GGUF path
        try:
            cfg = load_llm_config(profile_override=resolved)
            model_path = getattr(cfg, "model_path", "")
        except Exception as e:
            raise ValueError(f"Failed to resolve profile '{resolved}': {e}") from e

        if not model_path or not Path(model_path).is_file():
            raise FileNotFoundError(
                f"GGUF not found: {model_path}|Run on leader: python -m maxim.models.download --llm {resolved}"
            )

        # Determine port and API key (reuse current config)
        try:
            port = int(os.environ.get("MAXIM_AUTO_SPAWN_PORT", "8100"))
        except ValueError:
            port = 8100

        api_key: str | None = None
        try:
            from maxim.tunnel.keys import read_key

            api_key = read_key()
        except Exception:
            pass

        previous_model = _active_model or "none"

        # Drain in-flight requests (best-effort, 5s max)
        try:
            from maxim.models.language.lane_metrics import get_metrics_registry

            metrics = get_metrics_registry().get("infer")
            deadline = _time.time() + 5.0
            while metrics.current_in_flight > 0 and _time.time() < deadline:
                _time.sleep(0.25)
        except Exception:
            pass

        # Stop current server
        if _active_spawner is not None:
            if logger is not None:
                logger.info("LLM swap: stopping current server (model=%s)", _active_model)
            _active_spawner.stop()
            _active_spawner = None
            _active_model = None

        # Detect bind host
        bind_host = "0.0.0.0"  # noqa: S104 — leader always binds all interfaces
        try:
            from maxim.runtime.leader_mode import detect_role

            role_decision = detect_role()
            bind_host = role_decision.bind_host
        except Exception:
            pass

        # Start new server
        n_ctx = cfg.n_ctx if cfg.n_ctx > 0 else 8192
        spawner = LocalServerSpawner(
            model_path=model_path,
            port=port,
            bind_host=bind_host,
            n_ctx=n_ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            api_key=api_key,
        )

        if logger is not None:
            logger.info(
                "LLM swap: starting %s (port=%d, n_ctx=%d)",
                Path(model_path).name,
                port,
                n_ctx,
            )

        t0 = _time.time()
        url = spawner.start(timeout_s=120.0)
        startup_s = round(_time.time() - t0, 1)

        if url is None:
            raise RuntimeError(
                f"Server failed to start for model '{resolved}'. Check GPU memory — the model may be too large."
            )

        _active_spawner = spawner
        _active_model = resolved

        return {
            "status": "swapped",
            "model": resolved,
            "model_path": model_path,
            "port": port,
            "startup_s": startup_s,
            "previous_model": previous_model,
        }
    finally:
        _swap_lock.release()


def _maybe_auto_spawn_tunnel_daemon(logger: Any | None) -> None:
    """Spawn the cloudflared tunnel daemon if leader mode + config + no daemon running.

    Opt out with MAXIM_AUTO_SPAWN_TUNNEL=0.
    """
    if os.environ.get("MAXIM_AUTO_SPAWN_TUNNEL", "").strip().lower() in (
        "0",
        "false",
        "f",
        "no",
        "n",
        "off",
    ):
        return
    try:
        from maxim.tunnel.daemon_spawner import (
            TunnelDaemonSpawner,
            cloudflared_already_running,
            resolve_config_path,
        )
    except Exception:
        return
    if cloudflared_already_running():
        if logger is not None:
            logger.info(
                "Cloudflared daemon already running — skipping auto-spawn (managed elsewhere, e.g. systemd service)"
            )
        return
    config_path = resolve_config_path()
    if config_path is None:
        if logger is not None:
            logger.debug(
                "Tunnel auto-spawn skipped: no ~/.cloudflared/config.yml or /etc/cloudflared/config.yml found."
            )
        return
    spawner = TunnelDaemonSpawner(config_path=config_path)
    if logger is not None:
        logger.info("Auto-spawning cloudflared tunnel daemon (config=%s)", config_path)
    if not spawner.start():
        if logger is not None:
            logger.warning(
                "Cloudflared daemon auto-spawn failed — tunnel will not be active "
                "until you start it manually: cloudflared --config %s tunnel run",
                config_path,
            )


def _maybe_start_leader_proxy(
    bind_host: str,
    api_key: str | None,
    logger: Any | None,
) -> None:
    """Start the LeaderProxy reverse proxy in front of llama-cpp-server.

    Handles auth, per-request logging, GPU debug endpoints, and injects
    X-Maxim-* headers for peer-side trace enrichment. Tunnel ingress
    should point at the proxy port (8099) instead of the inference
    server (8100).
    """
    try:
        from maxim.runtime.leader_proxy import start_leader_proxy
    except Exception:
        return
    start_leader_proxy(api_key=api_key, bind_host=bind_host)

    # Start heartbeat monitor in leader mode (always on for leaders)
    try:
        from maxim.runtime.heartbeat import get_heartbeat_monitor

        get_heartbeat_monitor().start()
    except Exception:
        pass


def _print_lane_banner(manager: "LaneBackendManager") -> None:
    """Emit a compact, always-visible banner showing per-lane backend assignments.

    Printed directly so it bypasses the application logger (which may be
    configured to filter INFO). Keeps users oriented about which backend each
    lane is actually using — answers "is it really talking to my server?"
    at a glance.
    """
    import sys

    info = manager.describe()
    lines = [" " + "─" * 62, "  Maxim LLM lanes"]
    for lane, data in info.items():
        kind = data["kind"]
        if kind == "local":
            profile = data["profile"] or "(no LLM)"
            device = data["device"]
            descr = f"local   {profile} ({device})" if data["profile"] else "(no LLM)"
        else:
            url = data["remote_url"] or "?"
            profile = data["profile"] or data.get("remote_url", "")
            descr = f"{kind:<7} {url}"
        lines.append(f"  {lane:<7} {descr}")
    lines.append(" " + "─" * 62)
    for line in lines:
        print(line, file=sys.stderr)


__all__ = ["LaneBackendManager", "BackendGateError", "build_primary_router"]
