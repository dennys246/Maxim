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
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

from maxim.runtime.worker_pool import LaneConfig, Origin, ProviderPlacement

# ─── Plan 3 R2.5: backend-class dispatch table ─────────────────────────
#
# Maps a short string identifier to a lazy import path for the backend
# class that will serve a remote-URL lane. ``_build_remote_backend``
# writes the selected identifier into the provider config's ``type``
# field, and ``LLMRouter._get_backend_for_provider`` imports +
# instantiates the corresponding class via
# :func:`resolve_backend_class` at call time.
#
# **R3 review fix:** the original R2.5 shipped this as an identity map
# (``{"openai": "openai", "maxim_peer": "maxim_peer"}``) with
# ``_get_backend_for_provider`` hard-coding the ``"maxim_peer"`` /
# ``"maxim-peer"`` branch as a string literal. That meant the dispatch
# table provided zero indirection — adding a new backend required edits
# in two places, and a typo in either would silently fall through to
# the router's "unknown provider" warn. The table is now authoritative:
# the router imports ``resolve_backend_class`` and dispatches through
# it, so adding a new backend is exactly one change here.
#
# Values are ``(module_path, class_name)`` tuples rather than class
# objects so optional-dependency backends (e.g., Anthropic SDK, future
# Google Gemini peers) don't pay an import cost on every startup.
BACKEND_CLASSES: dict[str, tuple[str, str]] = {
    "openai": ("maxim.models.language.openai_backend", "_OpenAIBackend"),
    "maxim_peer": ("maxim.models.language.maxim_peer_backend", "_MaximPeerBackend"),
}


def resolve_backend_class(identifier: str) -> type | None:
    """Lazy-import the backend class for a ``BACKEND_CLASSES`` identifier.

    Accepts the canonical ``"maxim_peer"`` spelling plus the
    hyphenated variant ``"maxim-peer"`` so operator-written configs
    that use either spelling resolve to the same class. Returns
    ``None`` if the identifier is unknown or the import fails — the
    router then treats it as an unclassified provider type and emits a
    warning.
    """
    canonical = identifier.strip().lower().replace("-", "_")
    entry = BACKEND_CLASSES.get(canonical)
    if entry is None:
        return None
    module_path, class_name = entry
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name, None)
    except Exception:  # pragma: no cover — optional dep missing
        return None
    if not isinstance(cls, type):
        return None
    return cls


def _classify_backend(kind: str) -> str:
    """Pick a backend identifier for a classified lane.

    ``kind`` is the output of :meth:`LaneBackendManager._classify` —
    ``"self-hosted"`` or ``"cloud"``. Self-hosted lanes get the Plan 3
    ``_MaximPeerBackend`` (single HTTP call, typed failure, no internal
    repeat); cloud lanes keep the existing ``_OpenAIBackend`` with its
    full retry + cost tracking + PII redaction shape.
    """
    if kind == "self-hosted":
        return "maxim_peer"
    return "openai"


def _safe_int_env(name: str, default: int) -> int:
    """Parse an integer env var, returning *default* on invalid input."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid %s=%r, using default %d", name, raw, default)
        return default


def _safe_float_env(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Parse a float env var with optional clamping. Falls back on bad input.

    Used by P6's probe-tuning knobs (``MAXIM_REMOTE_PROBE_*``) where a
    typo or out-of-range value should fall back to the documented default
    instead of, say, a 0-second timeout that bricks every probe.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default
    if min_value is not None and value < min_value:
        return min_value
    if max_value is not None and value > max_value:
        return max_value
    return value


# ─── Pre-upgrade tier snapshot (peer_leader_flexibility_plan F0.3) ──────
#
# Frozen snapshot of `_INFER_VRAM_TIERS` as of the commit that introduced
# the pinning-migration logic. Used ONLY by `_maybe_pin_pre_upgrade_profile`
# to compute what an existing leader would have picked before the tier
# table expanded — so we can pin it to that choice and prevent silent
# model swaps on upgrade.
#
# When `_INFER_VRAM_TIERS` changes (e.g., P4a adds a qwen2.5-14b row),
# THIS snapshot stays frozen. That's the whole point of the migration
# safety: the pinning decision is made against the OLD table, not
# whatever the new table says.
#
# DO NOT update this table when adding new profiles to
# `_INFER_VRAM_TIERS`. This is a historical record.
_PRE_P4A_INFER_VRAM_TIERS: tuple[tuple[float, str], ...] = (
    (14.0, "llama-2-13b-chat"),
    (8.0, "llama-3-8b-instruct"),
    (4.0, "mistral-7b-instruct-v0.2"),
    (0.0, "smollm-1.7b-instruct"),
)


def _pick_pre_upgrade_infer_profile(vram_gb: float) -> str:
    """Walk the frozen pre-P4a tier table for a given VRAM budget.

    Returns the profile a pre-P4a leader would have picked for its
    large/infer tier. Used only by the pinning migration path.
    """
    for min_vram, profile in _PRE_P4A_INFER_VRAM_TIERS:
        if vram_gb >= min_vram:
            return profile
    return _PRE_P4A_INFER_VRAM_TIERS[-1][1]


def classify_self_hosted_lane(remote_url: str | None, source: str) -> bool:
    """Classify a resolved ``lanes.<tier>.remote_url`` as self-hosted infra.

    IM5 audit-note fold (C4 of config_unification.md): the pre-C4
    ``apply_peer_config_to_env`` set ``MAXIM_MAX_CLOUD_LANES=1`` as a
    side effect to mark peer-leader URLs as self-hosted infrastructure
    (so the cloud-lane gate didn't refuse them). Post-C4 the same
    classification happens here, gated on the resolve_setting source:

    - ``"config"`` or ``"env"`` with a non-null URL → self-hosted
    - ``"default"`` (no URL) → not applicable
    - any value → cloud opt-in stays explicit via
      ``config.json::cloud.enabled``; this function does NOT make
      cloud-routing decisions, only "is this remote_url a self-hosted
      peer URL?"
    """
    if not remote_url:
        return False
    return source in ("config", "env")


# Post-implementation Executor C1 fold: idempotency flag.
# _apply_lane_config_to_env populates env vars from config.json. Without
# this flag, a second invocation (e.g. from another build_primary_router
# call) sees the env vars set by the first call and re-attributes the
# field source from "config" to "env" — silently breaking C5's doctor
# source attribution. The cached `_lane_env_has_remote` lets callers
# get the original result without re-running the side-effect path.
_lane_env_applied: bool = False
_lane_env_has_remote: bool = False


def _reset_lane_env_applied() -> None:
    """Test-only helper: clear the lane-env once-per-startup flag.

    Pairs with :func:`maxim.runtime.config_loader.reset_config_cache`
    in the ``_isolate_config_json_env`` autouse fixture.
    """
    global _lane_env_applied, _lane_env_has_remote
    _lane_env_applied = False
    _lane_env_has_remote = False


def _apply_lane_config_to_env(logger_obj: Any | None) -> bool:
    """Populate ``MAXIM_LANE_<TIER>_REMOTE_*`` env vars from the unified
    config_loader precedence chain.

    Returns True iff at least one tier resolved a non-default
    ``remote_url`` (i.e., this Maxim is routing inference to a remote
    endpoint — leader or otherwise). Callers use this flag to suppress
    the persisted-model restore so a stale local profile doesn't
    silently override a self-hosted leader URL.

    Self-hosted classification (IM5 audit-note fold) sets
    ``MAXIM_MAX_CLOUD_LANES=1`` when a non-cloud URL is resolved, so
    the cloud-lane gate doesn't refuse self-hosted peer URLs.
    Cloud-provider opt-in stays explicit via
    ``config.json::cloud.enabled = true``.

    **Idempotent** (post-implementation Executor C1 fold): the first
    call performs the env-population work and caches the result; later
    calls return the cached ``has_remote`` value without re-running
    ``resolve_setting`` or re-firing side effects. Without this guard,
    the env vars populated by the first call would be re-attributed
    as ``source="env"`` by ``resolve_setting`` on the second call,
    silently corrupting C5's doctor "Resolved Config" section.
    """
    import os as _os

    global _lane_env_applied, _lane_env_has_remote
    if _lane_env_applied:
        return _lane_env_has_remote
    _lane_env_applied = True

    # Cloud-dispatch escape hatch (Exp 37 fix 2026-06-07): when the harness
    # spawns a sub-sim with a cloud LARGE-tier profile (claude-*, gpt-*,
    # etc.), it sets MAXIM_DISABLE_PEER_CONFIG=1 so the sub-sim doesn't
    # silently re-populate MAXIM_LANE_LARGE_REMOTE_URL from peer.yml or
    # config.json — which would route LARGE through ``_MaximPeerBackend``
    # to whatever local-model leader the peer was previously configured
    # for, defeating the cloud-dispatch intent. The harness's PR #337 fix
    # explicitly blanked the env var, but ``setdefault`` semantics here
    # treat empty-string-set as functionally-equivalent-to-unset and
    # re-populated from config. This gate is the targeted fix: when the
    # env var is set (truthy), short-circuit to no-remote-routing so the
    # router falls back to the profile's default backend (e.g.,
    # ``_AnthropicBackend`` for ``claude-sonnet-*``).
    _disable_peer = _os.environ.get("MAXIM_DISABLE_PEER_CONFIG", "").strip().lower()
    if _disable_peer in ("1", "true", "yes", "on"):
        if logger_obj is not None:
            logger_obj.info(
                "MAXIM_DISABLE_PEER_CONFIG=%s — skipping peer.yml + config.json "
                "lane routing. Router will use profile-default backend for each tier.",
                _disable_peer,
            )
        _lane_env_has_remote = False
        return False

    try:
        from maxim.runtime.config_loader import (
            load_config,
            resolve_api_key_ref,
            resolve_setting,
        )
    except Exception as e:
        if logger_obj is not None:
            logger_obj.warning("Failed to import config_loader for lane config: %s", e)
        return False

    # Load config.json so resolve_setting can consult it. The IM5
    # auto-migration shim fires inside load_config() if peer.yml →
    # config.json migration conditions match (config.json absent +
    # peer.yml present + cloudflared absent).
    try:
        cfg = load_config()
    except Exception as e:
        if logger_obj is not None:
            logger_obj.warning("Failed to load config.json: %s", e)
        cfg = None

    # Post-implementation field-coverage fold (2026-06-02): C4's
    # original _apply_lane_config_to_env only populated the
    # MAXIM_LANE_<TIER>_REMOTE_* env vars from config.json. Other
    # absorbed fields — MAXIM_LLM_PROFILE, MAXIM_LLM_ENABLED,
    # MAXIM_LLM_N_CTX, MAXIM_AUTO_DOWNLOAD_MODELS — were resolved by
    # ``resolve_setting`` (so they appeared in C5's "Resolved Config"
    # section) but never reached the legacy env-var-reading code at
    # detect_tiers / _maybe_pin_pre_upgrade_profile / llama-cpp-server
    # spawn. Result: a Mac Mini with config.json::llm.profile=
    # mistral-small-24b would still spawn qwen2.5-14b because tier
    # detection picked the hardware-best heuristic. Fix here mirrors
    # the lane-routing pattern: setdefault from config.json for the
    # full llm.* + auto_spawn.* surface so env values still win and
    # the legacy reader paths pick up config.json values seamlessly.
    _llm_env_map: tuple[tuple[str, str], ...] = (
        ("llm.profile", "MAXIM_LLM_PROFILE"),
        ("llm.n_ctx", "MAXIM_LLM_N_CTX"),
        ("llm.backend", "MAXIM_LLM_BACKEND"),
    )
    for field_path, env_name in _llm_env_map:
        try:
            value, _ = resolve_setting(field_path, config=cfg)
        except Exception:
            continue
        if value is None or value == "":
            continue
        if not _os.environ.get(env_name, "").strip():
            _os.environ[env_name] = str(value)

    # Booleans need str("1") / str("0") rendering since the legacy
    # readers use the truthy/falsy set.
    for field_path, env_name in (
        ("llm.enabled", "MAXIM_LLM_ENABLED"),
        ("llm.auto_download", "MAXIM_AUTO_DOWNLOAD_MODELS"),
    ):
        try:
            value, source = resolve_setting(field_path, config=cfg)
        except Exception:
            continue
        # Only propagate when config or env explicitly set the value
        # — defaults shouldn't write env vars (that would shadow a
        # subsequent operator opt-out via shell unset).
        if source == "default":
            continue
        if not _os.environ.get(env_name, "").strip():
            _os.environ[env_name] = "1" if value else "0"

    has_remote = False
    for tier in ("large", "medium", "small"):
        env_url = f"MAXIM_LANE_{tier.upper()}_REMOTE_URL"
        env_model = f"MAXIM_LANE_{tier.upper()}_REMOTE_MODEL"
        env_key = f"MAXIM_LANE_{tier.upper()}_REMOTE_API_KEY"
        env_timeout = f"MAXIM_LANE_{tier.upper()}_TIMEOUT_S"

        try:
            url, url_source = resolve_setting(f"lanes.{tier}.remote_url", config=cfg)
            model, _ = resolve_setting(f"lanes.{tier}.remote_model", config=cfg)
            key_ref, _ = resolve_setting(f"lanes.{tier}.remote_api_key_ref", config=cfg)
            timeout_s, timeout_source = resolve_setting(f"lanes.{tier}.timeout_s", config=cfg)
        except Exception as e:
            if logger_obj is not None:
                logger_obj.warning("Failed to resolve lanes.%s.* from config: %s", tier, e)
            continue

        # llm_timeout_scalability.md Stage 2: per-tier timeout passthrough.
        # Populate the env var so downstream backends + the leader proxy
        # see a single source of truth. Only set when source is non-default
        # (operator explicitly configured) so backend defaults remain in
        # play for unconfigured tiers.
        if timeout_s is not None and timeout_source != "default":
            if not _os.environ.get(env_timeout, "").strip():
                _os.environ[env_timeout] = str(float(timeout_s))

        if url:
            # Set env var only if not already populated — preserves the
            # "env wins" precedence for downstream code.
            if not _os.environ.get(env_url, "").strip():
                _os.environ[env_url] = url
            if model and not _os.environ.get(env_model, "").strip():
                _os.environ[env_model] = model
            if key_ref:
                # I-1 / I-2 fold: resolve the api_key_ref lazily here.
                # File path / keyring lookup. If unresolvable, leave the
                # env var unset — downstream lane construction will
                # surface BackendAuthFailed if the lane is actually used.
                resolved_key = resolve_api_key_ref(key_ref)
                if resolved_key and not _os.environ.get(env_key, "").strip():
                    _os.environ[env_key] = resolved_key
            has_remote = True

            # Self-hosted classification (IM5 audit note)
            if classify_self_hosted_lane(url, url_source):
                _os.environ.setdefault("MAXIM_MAX_CLOUD_LANES", "1")

    # peer.yml fallback (deprecated compat-read): if no lane resolved
    # a URL via env or config.json, AND peer.yml is still present on
    # disk (e.g., cloudflared-present case where IM5 auto-migration
    # didn't fire), populate from peer.yml with a once-per-startup
    # INFO deprecation log. Keeps the leader-with-stale-peer.yml case
    # working while the operator migrates.
    #
    # Post-implementation Executor C2 fold: replicate the controlled
    # in-function env-population pattern instead of calling the legacy
    # ``apply_peer_config_to_env`` shim. The shim writes env vars +
    # ``MAXIM_MAX_CLOUD_LANES=1`` via ``_setdefault_nonempty`` (which
    # is functionally equivalent), but routing through it makes the
    # idempotency contract harder to reason about and re-creates the
    # source-attribution corruption pattern this function's own gate
    # guards against. Doing the work inline keeps the contract local.
    if not has_remote:
        try:
            from maxim.peer.config import read_peer_config, register_leader_endpoint

            peer_cfg = read_peer_config()
            if peer_cfg is not None:
                if peer_cfg.url and not _os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL", "").strip():
                    _os.environ["MAXIM_LANE_LARGE_REMOTE_URL"] = peer_cfg.url
                if peer_cfg.api_key and not _os.environ.get("MAXIM_LANE_LARGE_REMOTE_API_KEY", "").strip():
                    _os.environ["MAXIM_LANE_LARGE_REMOTE_API_KEY"] = peer_cfg.api_key
                if peer_cfg.model and not _os.environ.get("MAXIM_LANE_LARGE_REMOTE_MODEL", "").strip():
                    _os.environ["MAXIM_LANE_LARGE_REMOTE_MODEL"] = peer_cfg.model
                # peer.yml URL is self-hosted infrastructure by definition
                _os.environ.setdefault("MAXIM_MAX_CLOUD_LANES", "1")
                # Preserve the legacy register_leader_endpoint side
                # effect so downstream http callers can resolve the
                # "leader" named endpoint.
                register_leader_endpoint(peer_cfg.url)
                if logger_obj is not None:
                    logger_obj.info(
                        "config: lanes.large.* resolved from peer.yml "
                        "(deprecated — run `maxim peer connect <url>` to "
                        "migrate to config.json)."
                    )
                has_remote = True
        except Exception as e:
            if logger_obj is not None:
                logger_obj.warning("Failed to load peer.yml fallback: %s", e)

    _lane_env_has_remote = has_remote
    return has_remote


def _maybe_pin_pre_upgrade_profile(capabilities: Any | None, logger_obj: Any | None) -> None:
    """Pin the leader to its pre-upgrade profile on first post-upgrade run.

    Fires exactly once per machine — writes a marker file after completing
    so subsequent startups skip this code path. Triggers only when all of
    these are true:

    1. The user has NOT set MAXIM_LLM_PROFILE (no explicit current-session
       override).
    2. The persisted-model file is absent (nothing has pinned it yet).
    3. The pin marker file is absent (this migration has not run before).
    4. The data_home has at least one non-hidden child (indicates this
       machine has run Maxim before — new installs are skipped).

    Best-effort: swallows all exceptions because pinning failure must
    not block startup. If the pin doesn't happen, the worst case is the
    next run picks whatever the current tier table says, which matches
    existing behavior before this migration was added.
    """
    try:
        from maxim.utils.paths import data_home

        # (3) Skip if the pin marker already exists
        home = data_home()
        util_dir = home / "util"
        marker = util_dir / "tier_pinned_at_upgrade"
        if marker.exists():
            return

        # (1) Skip if the user set an explicit profile via env
        if os.environ.get("MAXIM_LLM_PROFILE", "").strip():
            return

        # (2) Skip if a persisted model already exists (existing pin)
        persisted = _read_persisted_model()
        if persisted:
            # Still record the marker so we don't re-check this path.
            _write_pin_marker(marker)
            return

        # (4) Skip fresh installs. Heuristic: util/ must exist with at
        # least one file. A brand-new install has no util/ directory at
        # all (utils/paths creates it lazily on first write), so the
        # absence of util/ is our proxy for "never run before".
        if not util_dir.exists():
            return
        try:
            has_content = any(util_dir.iterdir())
        except OSError:
            has_content = False
        if not has_content:
            return

        # Compute what the pre-P4a tier table would have picked.
        if capabilities is None:
            from maxim.runtime.capabilities import detect_compute_resources

            has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()
        else:
            has_gpu = getattr(capabilities, "has_gpu", False)
            vram_gb = getattr(capabilities, "vram_gb", 0.0)

        # Before P4a, MPS was hard-excluded from the large-tier branch,
        # so MPS systems would not have had a large-tier pin at all.
        # Only CUDA systems with vram_gb >= 4.0 got pinned. Mirror that
        # exactly — don't pin MPS systems retroactively.
        gpu_type = getattr(capabilities, "gpu_type", None) if capabilities is not None else None
        if not (has_gpu and gpu_type not in ("mps", None) and vram_gb >= 4.0):
            # Pre-P4a would have left this machine without a large tier.
            # Nothing to pin. Still record the marker so we don't re-check.
            _write_pin_marker(marker)
            return

        pre_profile = _pick_pre_upgrade_infer_profile(vram_gb)

        # Write BOTH the persisted-model file AND the env var. The file
        # pins all future startups; the env var makes sure the CURRENT
        # run also respects the pin rather than using whatever the new
        # tier table selects. Without the env var set here, the pin
        # fires one run too late — the machine would use the new
        # default on the first post-upgrade run and the old default
        # on every subsequent run, which is confusing.
        _write_persisted_model(pre_profile)
        os.environ["MAXIM_LLM_PROFILE"] = pre_profile
        _write_pin_marker(marker)

        if logger_obj is not None:
            logger_obj.info(
                "Pinned leader to pre-upgrade profile %r for safe migration. "
                "Run `maxim peer llm <model>` to change, or delete "
                "~/.maxim/util/tier_pinned_at_upgrade to re-run the pin.",
                pre_profile,
            )
    except Exception as e:
        # Pinning failure must never block startup. Log at debug so the
        # absence of a pin is visible to anyone spelunking, but don't
        # surface a warning that'd confuse users.
        if logger_obj is not None:
            logger_obj.debug("Pre-upgrade profile pinning failed (non-fatal): %s", e)


def _write_pin_marker(marker_path: Any) -> None:
    """Touch the marker file so the pin-migration code path is skipped
    on subsequent startups. Swallows errors — failure to write the
    marker just means the pin may re-run next time, which is harmless
    (conditions 1-4 in `_maybe_pin_pre_upgrade_profile` still apply).
    """
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.touch()
    except OSError as e:
        logger.debug("Could not write pin marker %s: %s", marker_path, e)


# Extracted to llm_server.py — mutable globals accessed via module reference
# to avoid Python's import-by-value semantics splitting the state.
import maxim.runtime.llm_server as _server_mod  # noqa: E402

# Re-export functions for backward compatibility (functions are safe to re-export)
from maxim.runtime.llm_server import (  # noqa: F401, E402
    stop_active_spawner,
    register_router,
    _find_active_routers,
    _model_state_file,
    read_persisted_model as _read_persisted_model,
    write_persisted_model as _write_persisted_model,
    profile_has_local_file as _profile_has_local_file,
)

# Mutable globals — access via _server_mod to keep a single source of truth.
# DO NOT import these by name (e.g. `from llm_server import _active_spawner`)
# because Python binds by value at import time and assignments diverge.
_swap_lock = _server_mod._swap_lock  # Lock is safe (same object reference, never reassigned)


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
        logger.warning("Could not parse remote URL %r — treating as private (fail-closed)", remote_url)
        return False
    host = parsed.hostname or ""
    return not _host_is_private(host)


# ─── placement derivation (lane_capability_placement_split.md, Phase 1) ─────
#
# The placement axis (Origin/ProviderPlacement) is introduced additively. In
# Phase 1 it is NOT yet wired into dispatch — ``get_backend`` / ``_build_*`` /
# ``_classify`` still drive routing off the legacy LaneConfig fields. These
# helpers are the one-way back-compat shim + resolver scaffolding: they derive
# a placement tuple from the legacy fields and reproduce ``_classify``'s exact
# kind classification, so Phase 2 can switch dispatch to read ``origin`` with a
# proven before/after equivalence. Derivation direction is one-way: env /
# tier-detection / auto-spawn keep landing in the legacy fields; placement is
# computed from them when ``cfg.placement == ()``.


def derive_placement(cfg: LaneConfig, *, peer_owned: bool) -> tuple[ProviderPlacement, ...]:
    """Derive the ordered placement for a lane from its legacy fields.

    Reproduces :meth:`LaneBackendManager._classify`'s exact origin
    classification so that, until Phase 2 wires placement into dispatch,
    every existing config yields identical routing:

    - ``remote_url`` set + ``peer_owned`` → ``PEER`` (own infra behind a tunnel)
    - ``remote_url`` set, not peer-owned → ``CLOUD`` if the host is public else
      ``PEER`` (the ``_is_cloud_url`` split)
    - no ``remote_url`` but a ``model_profile`` → ``LOCAL`` — **even when the
      profile names a cloud model** (the ``--cloud-lane`` case). This matches
      ``_classify`` returning ``"local"`` for a cloud-profile-with-no-URL lane,
      which is load-bearing: such lanes must NOT count against
      ``MAXIM_MAX_CLOUD_LANES`` (they are gated by ``_validate_cloud_config``
      instead). Phase 3 re-expresses ``--cloud-lane`` as a real CLOUD placement
      and preserves the cap accounting via the ``cli_utils`` cap bump.
    - neither profile nor URL → ``()`` (the ``get_backend`` None-gate case).

    If ``cfg.placement`` is already populated it is returned verbatim — an
    explicit producer wins over derivation.
    """
    if cfg.placement:
        return cfg.placement
    if cfg.remote_url:
        if peer_owned:
            origin = Origin.PEER
        else:
            origin = Origin.CLOUD if _is_cloud_url(cfg.remote_url) else Origin.PEER
        return (
            ProviderPlacement(
                origin=origin,
                model=cfg.remote_model or cfg.model_profile,
                url=cfg.remote_url,
                api_key=cfg.remote_api_key,
                timeout_s=cfg.remote_timeout_s,
            ),
        )
    if cfg.model_profile:
        return (
            ProviderPlacement(
                origin=Origin.LOCAL,
                model=cfg.model_profile,
                timeout_s=cfg.remote_timeout_s,
            ),
        )
    return ()


def primary_placement(cfg: LaneConfig, *, peer_owned: bool) -> ProviderPlacement | None:
    """The primary (first / index-0) placement entry, or ``None`` for an empty lane.

    Purely positional — NOT a health/eligibility resolution. First-healthy
    failover across an ordered placement rides on ``LLMRouter``'s
    ``provider_priority`` per the plan's scope decision; do NOT bolt selection
    logic onto this helper.
    """
    placement = derive_placement(cfg, peer_owned=peer_owned)
    return placement[0] if placement else None


def placement_kind(placement: tuple[ProviderPlacement, ...]) -> str:
    """Map a placement tuple to a legacy ``_classify`` kind string.

    Keys off the PRIMARY (first) entry's origin so the result matches
    ``_classify`` exactly: ``PEER`` → ``"self-hosted"``, ``CLOUD`` →
    ``"cloud"``, ``LOCAL`` → ``"local"``. Empty → ``"local"`` (matches
    ``get_lane_kind``'s None-cfg / no-backend default).
    """
    if not placement:
        return "local"
    origin = placement[0].origin
    if origin is Origin.PEER:
        return "self-hosted"
    if origin is Origin.CLOUD:
        return "cloud"
    return "local"


# ─── exceptions ───────────────────────────────────────────────────────────


class BackendGateError(RuntimeError):
    """Raised when a safety gate refuses to construct a backend."""


# ─── manager ──────────────────────────────────────────────────────────────


class LaneBackendManager:
    """Owns the lane → backend mapping for multi-LLM operation.

    Lanes are now tier-based (large/medium/small). The optional
    ``function_router`` maps function names to tiers with fallback chains.
    """

    def __init__(
        self,
        lane_configs: dict[str, LaneConfig],
        *,
        max_backends: int | None = None,
        max_cloud_lanes: int | None = None,
        peer_owned: bool = False,
        function_router: Any | None = None,
    ) -> None:
        self._configs: dict[str, LaneConfig] = dict(lane_configs)
        self._backends: dict[str, Any] = {}
        self._cloud_lanes: set[str] = set()
        self._peer_owned = peer_owned
        self._function_router = function_router
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

    def has_cloud_placement(self) -> bool:
        """True if ANY lane names a CLOUD origin anywhere in its placement.

        Unlike ``get_lane_kind``/``_classify`` (which key off the PRIMARY
        placement only), this counts cloud PRESENCE — primary or fallback
        tail. NOTE: a cloud profile with no URL derives as LOCAL and is
        invisible here — anything deciding whether cloud spend is possible
        must use :meth:`has_cloud_billing_surface`, which closes that
        blind spot.
        """
        return any(
            entry.origin is Origin.CLOUD
            for cfg in self._configs.values()
            for entry in derive_placement(cfg, peer_owned=self._peer_owned)
        )

    def has_cloud_billing_surface(self) -> bool:
        """True if ANYTHING in this manager's config could bill a cloud call.

        The startup "Cost tracking disabled (local models only)" banner
        keys off this — so it must fail CLOSED (return True when unsure):
        wrongly printing "disabled" on a billing config is the expensive
        direction. Three surfaces, any one of which means cloud spend is
        possible:

        1. A CLOUD origin anywhere in any lane's placement (primary or
           fallback tail) — remote_url cloud lanes, explicit placements,
           ``--cloud-lane`` / ``--cloud-fallback`` (both materialized into
           placement before the banner prints).
        2. A cloud PROFILE with no URL (``--llm claude-sonnet``) — derives
           a LOCAL placement by design (the MAXIM_MAX_CLOUD_LANES cap
           exemption) but dispatches to a real metered cloud backend.
           This is the highest-spend headline path; ``has_cloud_placement``
           alone would print "disabled" on it.
        3. Cloud dispatch globally enabled (``cloud.enabled`` /
           MAXIM_LLM_CLOUD_ENABLED / llm.json) — a user llm.json
           ``providers`` table can carry cloud entries in
           ``provider_priority`` that the lane placement view can't see.
        """
        if self.has_cloud_placement():
            return True
        try:
            from maxim.models.language.config import _BUILTIN_PROFILES, _normalize_profile

            for cfg in self._configs.values():
                for entry in derive_placement(cfg, peer_owned=self._peer_owned):
                    # Normalize first: the lane may carry an alias
                    # ("claude-sonnet") rather than the canonical profile
                    # key ("claude-sonnet-4-6") depending on which path
                    # populated model_profile.
                    key = _normalize_profile(entry.model or "") or (entry.model or "")
                    if _BUILTIN_PROFILES.get(key, {}).get("cloud"):
                        return True
        except Exception:
            return True  # can't inspect → don't claim cost tracking is inert
        try:
            from maxim.models.language.config import load_llm_config

            if load_llm_config().cloud_enabled:
                return True
        except Exception:
            return True  # fail closed, same reasoning
        return False

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
        # Phase 2: gate on the resolved placement (honors an explicit
        # cfg.placement; equivalent to the legacy "no profile and no remote_url"
        # check for every derived placement, since derive_placement returns ()
        # iff both legacy fields are empty).
        placement = derive_placement(cfg, peer_owned=self._peer_owned)
        if not placement:
            return None

        with self._lock:
            cached = self._backends.get(lane)
            if cached is not None:
                return cached
            self._check_backend_cap()
            kind = placement_kind(placement)
            if kind == "cloud":
                self._check_cloud_lane_cap(lane)
            backend = self._build_backend(cfg, placement)
            if backend is not None:
                self._backends[lane] = backend
                if kind == "cloud":
                    self._cloud_lanes.add(lane)
            return backend

    @property
    def function_router(self) -> Any | None:
        """The FunctionRouter attached to this manager (if any)."""
        return self._function_router

    def set_function_router(self, router: Any) -> None:
        """Attach or replace the FunctionRouter."""
        self._function_router = router

    def get_backend_for_function(self, function: str) -> tuple[Any | None, str]:
        """Resolve a function name to a tier and return (backend, tier_name).

        Uses the attached ``FunctionRouter`` to map function → tier.
        Falls back to ``get_backend(function)`` if no router is set (treats
        the function name as a literal lane name — backward compat).

        Returns:
            (backend, tier_name) tuple. Backend may be None if the tier
            has no model configured.

        Raises:
            TierRestrictionError / TierUnavailableError from FunctionRouter
            if the function's tier isn't available.
        """
        if self._function_router is not None:
            tier, _boost = self._function_router.resolve(function)
            return self.get_backend(tier), tier
        return self.get_backend(function), function

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
        """Classify a lane as ``"local"`` / ``"self-hosted"`` / ``"cloud"``.

        Phase 2 (lane_capability_placement_split.md): reads the **primary
        placement's origin** instead of URL-sniffing ``cfg.remote_url`` here.
        ``derive_placement`` honors an explicit ``cfg.placement`` (Phase 3
        producers) and otherwise derives a 1-element placement from the legacy
        fields — reproducing the prior URL-sniff classification **exactly** for
        every existing config (the URL-sniff now lives once inside
        ``derive_placement``'s legacy path, not at every dispatch site). The
        equivalence to the pre-Phase-2 logic is pinned by the frozen legacy
        oracle in ``tests/unit/test_lane_placement.py``.
        """
        return placement_kind(derive_placement(cfg, peer_owned=self._peer_owned))

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

    def _build_backend(self, cfg: LaneConfig, placement: tuple[ProviderPlacement, ...] | None = None) -> Any | None:
        # Phase 2: dispatch on the primary placement's origin rather than
        # sniffing cfg.remote_url. LOCAL → profile-driven local router (which
        # also serves --cloud-lane cloud profiles); CLOUD / PEER → remote router.
        # Equivalent to the prior `if cfg.remote_url` for every derived
        # placement (LOCAL ⟺ no remote_url; CLOUD/PEER ⟺ remote_url set).
        #
        # ``placement`` is the already-resolved tuple threaded from
        # ``get_backend`` so we don't re-derive (re-derivation re-runs the
        # ``_is_cloud_url`` DNS probe). It's optional for standalone/test
        # callers, who get a fresh derivation.
        if placement is None:
            placement = derive_placement(cfg, peer_owned=self._peer_owned)
        if not placement:
            return None
        # Phase 3: the builders compile the backend from the resolved primary
        # PLACEMENT (model / url / api_key / timeout_s), not the legacy
        # LaneConfig fields. For a DERIVED placement these mirror the legacy
        # fields exactly (byte-for-byte equivalent); for an EXPLICIT
        # cfg.placement (config.json / re-expressed CLI flags) the placement is
        # authoritative — which is what makes capability × placement
        # independent. (This replaces the Phase-2 fence that raised on explicit
        # placements.) Lane-level hardware tuning (n_gpu_layers, device, n_ctx,
        # kv_quant_mode) stays on LaneConfig — it is not a placement preference.
        #
        # Coherence note: removing the fence traded loud construction-time
        # failure for runtime failure on a MALFORMED explicit placement. Derived
        # placements are always coherent; explicit placements are validated at
        # their producer boundary — config.json at load (Phase 3b, via
        # worker_pool.validate_placement_coherence: PEER needs url, LOCAL needs
        # model, CLOUD needs url-or-model) and CLI edits in Phase 3c.
        primary = placement[0]
        # Dispatch on the primary origin:
        # - LOCAL → profile-driven local router.
        # - CLOUD *profile* (no url, e.g. claude-sonnet) → ALSO the profile-
        #   driven path: load_llm_config resolves the cloud profile to the right
        #   backend (anthropic/openai). _build_remote_backend assumes an
        #   openai-compatible base_url and would mis-build a cloud profile.
        #   (Derived CLOUD always carries a url, so this only affects explicit
        #   cloud-profile placements — no regression.)
        # - CLOUD with url / PEER → explicit remote URL via _build_remote_backend.
        if primary.origin is Origin.LOCAL or (primary.origin is Origin.CLOUD and not primary.url):
            return self._build_local_backend(cfg, primary)
        return self._build_remote_backend(cfg, primary, kind=placement_kind(placement))

    def _build_local_backend(self, cfg: LaneConfig, primary: ProviderPlacement | None = None) -> Any | None:
        """Construct a local LLMRouter for a lane.

        Phase 3: the served profile comes from the resolved placement
        (``primary.model``); for a derived placement that equals
        ``cfg.model_profile``. Hardware tuning (``n_gpu_layers`` etc.) stays on
        ``LaneConfig``. ``primary`` is optional for standalone/test callers,
        who get a fresh derivation.
        """
        try:
            from maxim.models.language.config import load_llm_config
            from maxim.models.language.router import LLMRouter
        except Exception as e:
            logger.warning("Language model modules not available: %s", e)
            return None

        if primary is None:
            primary = primary_placement(cfg, peer_owned=self._peer_owned)
        env_profile = os.environ.get("MAXIM_LLM_PROFILE", "").strip()
        profile = env_profile or (primary.model if primary else cfg.model_profile)
        try:
            llm_config = load_llm_config(profile_override=profile)
        except Exception as e:
            logger.warning("Failed to load LLM config for profile %r: %s", profile, e)
            return None

        # An explicit CLOUD-origin primary (cloud profile, no url) routes here
        # for correct backend selection; the operator explicitly chose cloud, so
        # enable cloud dispatch (the cloud-lane cap + _validate_cloud_config key/
        # cost gates still apply at the get_backend + router layers).
        if primary is not None and primary.origin is Origin.CLOUD:
            # Carry the placement's resolved key IN CONFIG (LLMConfig.api_key)
            # so the backend reads it directly — never via os.environ mutation.
            # The previous setdefault made a re-keyed cloud setup unrecoverable
            # in-process: the first build's stale key was indistinguishable
            # from an operator export and won forever (post-merge review
            # Exec #3; also removes that fold's silent except-pass, Arch #2).
            # An operator-exported env key still wins at the backend (env >
            # config precedence).
            replace_kwargs: dict[str, Any] = {"cloud_enabled": True}
            if primary.api_key:
                replace_kwargs["api_key"] = primary.api_key
            llm_config = dataclasses.replace(llm_config, **replace_kwargs)

        if "MAXIM_LLM_N_GPU_LAYERS" not in os.environ:
            try:
                llm_config = dataclasses.replace(llm_config, n_gpu_layers=cfg.n_gpu_layers)
            except Exception:
                pass  # Non-fatal config tweak; original config is still usable

        # Fallback injection — EXACTLY ONE path (the Phase-3c double-injection
        # guard, keyed on the cfg.placement FIELD):
        # - explicit cfg.placement → tail-inject placement[1:] (placement-driven
        #   multi-element compile; the placement is authoritative).
        # - derived (cfg.placement == ()) → legacy env-driven --cloud-fallback,
        #   unchanged behavior.
        if cfg.placement:
            llm_config = _inject_placement_tail(llm_config, cfg.placement[1:], lane_name=cfg.name, logger=logger)
        else:
            llm_config = self._maybe_inject_cloud_fallback(cfg, llm_config)

        # Plan 4 C4: inject drain constraint so drained mesh nodes are
        # skipped during dispatch. Returns None (no-op) for non-mesh
        # installs or when no provider URL matches any mesh node.
        drain_constraint = None
        auto_drain_callback = None
        try:
            from maxim.peer.mesh_config import read_or_synthesize_mesh_config
            from maxim.peer.drain_routing import (
                build_auto_drain_writer,
                build_auto_undrain_prober,
                build_drain_constraint,
            )

            mesh_cfg = read_or_synthesize_mesh_config()
            if mesh_cfg is not None:
                providers = llm_config.providers if isinstance(llm_config.providers, dict) else {}
                dc = build_drain_constraint(mesh_cfg, providers)
                if dc is not None:
                    drain_constraint = dc.is_drained
                    # C4.5: auto-drain writer shares the same URL→node mapping.
                    writer = build_auto_drain_writer(dc)
                    auto_drain_callback = writer.maybe_auto_drain
                    # C4.6: auto-undrain prober — background thread that
                    # probes auto-drained nodes and clears them when healthy.
                    prober = build_auto_undrain_prober(mesh_cfg, mesh_cfg.cluster_key)
                    prober.start()
        except ImportError:
            logger.debug("drain constraint init skipped (optional deps)", exc_info=True)
        except Exception:
            logger.warning("drain constraint init failed", exc_info=True)

        try:
            return LLMRouter(
                llm_config,
                drain_constraint=drain_constraint,
                auto_drain_callback=auto_drain_callback,
            )
        except Exception as e:
            logger.warning("Failed to create LLM router: %s", e)
            return None

    def _maybe_inject_cloud_fallback(self, cfg: "LaneConfig", llm_config: Any) -> Any:
        """Add a cloud fallback provider to the infer lane's LLMConfig.

        When --cloud-fallback is set, the CLI stores the resolved model name
        in MAXIM_CLOUD_FALLBACK_MODEL.  We inject it as a secondary provider
        so the router tries the primary (local/self-hosted) first and falls
        back to cloud on failure, rate-limit, or timeout.
        """
        if cfg.name != "large":
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

    def _build_remote_backend(
        self, cfg: LaneConfig, primary: ProviderPlacement | None = None, kind: str | None = None
    ) -> Any | None:
        """Construct an OpenAI-compatible remote backend via LLMRouter.

        Wraps _OpenAIBackend in an LLMRouter so downstream consumers get the
        same interface as local lanes (warmup, complete_with_usage, etc.).

        Self-hosted targets (llama-cpp-server, Ollama, vLLM on private IP)
        skip the cloud_enabled requirement. Cloud providers (Anthropic, OpenAI)
        require cloud_enabled=True and count against MAXIM_MAX_CLOUD_LANES.

        Phase 3: ``base_url`` / ``model`` / ``api_key`` / ``timeout_s`` are
        compiled from the resolved ``primary`` placement — for a derived
        placement these equal the legacy ``remote_*`` fields. ``kind`` is the
        pre-resolved classification threaded from ``_build_backend`` (avoids a
        re-run of the ``_is_cloud_url`` DNS probe). Both are optional for
        standalone/test callers, who get fresh derivations.
        """
        try:
            from maxim.models.language.config import load_llm_config
            from maxim.models.language.router import LLMRouter
        except Exception as e:
            logger.warning("LLM modules not available for remote lane %s: %s", cfg.name, e)
            return None

        if primary is None:
            primary = primary_placement(cfg, peer_owned=self._peer_owned)
        if kind is None:
            kind = self._classify(cfg)
        try:
            base = load_llm_config()
        except Exception as e:
            logger.warning("Failed to load LLM config for remote lane %s: %s", cfg.name, e)
            return None

        # Compile the connection from the resolved placement; for a derived
        # placement these mirror the legacy remote_* fields exactly.
        remote_url = primary.url if primary else cfg.remote_url
        remote_api_key = primary.api_key if primary else cfg.remote_api_key
        remote_model = (primary.model if primary else None) or cfg.remote_model or cfg.model_profile or base.model
        remote_timeout_s = primary.timeout_s if primary else cfg.remote_timeout_s

        provider_key = f"lane-{cfg.name}"
        api_key_env = f"MAXIM_LANE_{cfg.name.upper()}_API_KEY"
        if remote_api_key:
            os.environ.setdefault(api_key_env, remote_api_key)
        else:
            os.environ.setdefault(api_key_env, "not-needed")

        providers = dict(base.providers or {})
        # Plan 3 R2.5: dispatch to _MaximPeerBackend for self-hosted
        # lanes, keep _OpenAIBackend for cloud. The "type" field drives
        # LLMRouter._get_backend_for_provider's branch selection.
        backend_type = _classify_backend(kind)
        provider_entry: dict[str, Any] = {
            "type": backend_type,
            "base_url": remote_url,
            "api_key_env": api_key_env,
            "model": remote_model,
            "allow_local_endpoints": (kind == "self-hosted"),
            # Peer-tunnel URLs resolve to Cloudflare IPs (classified as cloud)
            # but actually serve a self-hosted model with no metered pricing.
            # Skip the cost-estimation gate that would reject the provider
            # when no pricing entry exists for the model name.
            "pricing_required": False,
        }
        # llm_timeout_scalability.md Stage 2: thread per-tier timeout_s
        # into the provider config. Both backends (_MaximPeerBackend
        # _get_timeout_policy, _OpenAIBackend _get_timeout) already read
        # cfg.get("timeout_s", <default>) — populating the key here makes
        # the operator's per-tier override flow naturally to the read
        # site without backend-side wiring changes.
        if remote_timeout_s is not None:
            provider_entry["timeout_s"] = float(remote_timeout_s)
        providers[provider_key] = provider_entry

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
                model=remote_model,
                providers=providers,
                routing=routing,
                # Self-hosted is not cloud; cloud lanes still require cloud_enabled.
                cloud_enabled=(True if kind == "cloud" else base.cloud_enabled),
            )
        except Exception as e:
            logger.warning("Failed to build remote lane config for %s: %s", cfg.name, e)
            return None

        # Phase 3c: a remote PRIMARY can carry a placement tail (e.g.
        # [PEER, CLOUD-fallback]); append the tail as fallback providers after
        # the primary in provider_priority. Empty tail (derived placements) →
        # no-op.
        if cfg.placement:
            remote_cfg = _inject_placement_tail(remote_cfg, cfg.placement[1:], lane_name=cfg.name, logger=logger)

        try:
            return LLMRouter(remote_cfg)
        except Exception as e:
            logger.warning("Failed to create remote LLM router for %s: %s", cfg.name, e)
            return None


def _config_placement_to_runtime(entry: Any, *, where: str) -> ProviderPlacement:
    """Resolve one declarative ``LaneTierPlacement`` into a runtime
    ``ProviderPlacement``: ``origin`` str → :class:`Origin`, ``api_key_ref`` →
    resolved ``api_key``, coherence re-checked (belt-and-suspenders — the loader
    already validated it). See lane_capability_placement_split.md Phase 3c.
    """
    from maxim.runtime.config_loader import resolve_api_key_ref
    from maxim.runtime.worker_pool import validate_placement_coherence

    p = ProviderPlacement(
        origin=Origin(entry.origin),
        model=entry.model,
        url=entry.url,
        api_key=resolve_api_key_ref(entry.api_key_ref),
        timeout_s=entry.timeout_s,
        extra=dict(entry.extra),
    )
    validate_placement_coherence(p, where=where)
    return p


def _apply_config_placements(lane_configs: dict[str, Any], logger: Any | None) -> dict[str, Any]:
    """Set ``LaneConfig.placement`` from ``config.json::lanes.<tier>.placement``.

    The declarative BASE for the placement axis (Phase 3c). Only tiers with a
    non-empty config placement are touched; everything else keeps ``placement
    == ()`` and derives from the legacy fields (unchanged behavior). An explicit
    config placement opts the tier out of auto-spawn / legacy-field derivation —
    it is the operator's authoritative choice (CLI flags still edit it on top).
    """
    try:
        # Cached singleton (review fold): avoid a second uncached disk read +
        # JSON re-parse during router build. get_config() reuses the load that
        # _apply_lane_config_to_env already performed.
        from maxim.runtime.config_loader import get_config

        cfg = get_config()
    except Exception as e:
        if logger is not None:
            logger.debug("config.json placement read skipped: %s", e)
        return lane_configs

    lanes = getattr(cfg, "lanes", None)
    if lanes is None:
        return lane_configs

    out = dict(lane_configs)
    cloud_primary_count = 0
    for tier in ("large", "medium", "small"):
        tier_cfg = getattr(lanes, tier, None)
        if tier_cfg is None or not getattr(tier_cfg, "placement", ()):
            continue
        try:
            placement = tuple(
                _config_placement_to_runtime(e, where=f"config.json: lanes.{tier}.placement[{i}]")
                for i, e in enumerate(tier_cfg.placement)
            )
        except Exception as e:
            if logger is not None:
                logger.warning("Lane '%s' config placement skipped (resolve failed): %s", tier, e)
            continue
        existing = out.get(tier)
        if existing is None:
            existing = LaneConfig(name=tier, max_workers=1, requires_gpu=False)
        out[tier] = dataclasses.replace(existing, placement=placement)
        # A CLOUD *primary* consumes a MAX_CLOUD_LANES slot (placement_kind keys
        # off placement[0]); a cloud *fallback* does not.
        if placement[0].origin is Origin.CLOUD:
            cloud_primary_count += 1
        if logger is not None:
            origins = ", ".join(p.origin.value for p in placement)
            logger.info("Lane '%s' placement from config.json: [%s]", tier, origins)

    # Review fold: writing a CLOUD placement in config.json IS an explicit cloud
    # opt-in, so raise the cloud-lane cap to cover it (bump-if-lower, mirroring
    # cli_utils' --cloud-lane handling). Without this, the canonical declarative
    # way to express a cloud lane hits BackendGateError(cap=0) at first dispatch
    # with no guidance.
    if cloud_primary_count > 0:
        try:
            current = int(os.environ.get("MAXIM_MAX_CLOUD_LANES", "0"))
        except ValueError:
            current = 0
        if current < cloud_primary_count:
            os.environ["MAXIM_MAX_CLOUD_LANES"] = str(cloud_primary_count)
            if logger is not None:
                logger.info(
                    "Raised MAXIM_MAX_CLOUD_LANES to %d to admit config.json CLOUD placement(s)",
                    cloud_primary_count,
                )
    return out


def _placement_provider_api_key_env(key: str) -> str:
    """Per-fallback-provider api-key env var name."""
    return f"MAXIM_PLACEMENT_{key.upper().replace('-', '_')}_API_KEY"


# Review fold: one-shot deprecation INFO for the --cloud-lane / --cloud-fallback
# CLI aliases (now re-expressed as placement edits, kept working through 1.x,
# drop in 1.2). Mirrors config_unification's `_warned_envs` once-per-startup set.
_warned_cloud_flag_deprecation: set[str] = set()


def _warn_cloud_flag_deprecated(flag: str, logger: Any | None) -> None:
    if logger is None or flag in _warned_cloud_flag_deprecation:
        return
    _warned_cloud_flag_deprecation.add(flag)
    logger.info(
        "%s is deprecated (kept working through 1.x, removed in 1.2) — it is now "
        "re-expressed as a placement edit. Prefer the declarative "
        "config.json::lanes.<tier>.placement to express capability × placement.",
        flag,
    )


def _placement_entry_to_provider(entry: ProviderPlacement, key: str) -> tuple[dict[str, Any] | None, bool]:
    """Compile one (non-primary) placement entry into an LLMRouter provider dict.

    Returns ``(provider_dict, is_cloud)``; ``(None, False)`` if the entry can't
    be compiled (logged + skipped by the caller). This is the per-entry mapper
    the tail-injection unification rides on (lane_capability_placement_split.md
    Phase 3c). Origin semantics:

    - ``PEER`` → ``_MaximPeerBackend`` (self-hosted URL; not cloud).
    - ``CLOUD`` with ``url`` → ``_OpenAIBackend`` (openai-compatible base_url).
    - ``CLOUD`` with a model that is a cloud *profile* (``_BUILTIN_PROFILES``) →
      that profile's backend (anthropic/openai) + api_key_env — the path that
      makes ``--cloud-fallback claude-sonnet`` pick the Anthropic backend.
    - ``LOCAL`` → resolve the profile via ``load_llm_config`` into a llama-cpp /
      transformers provider entry (model_path/n_ctx from the profile).
    """
    if entry.origin is Origin.PEER:
        api_key_env = _placement_provider_api_key_env(key)
        os.environ.setdefault(api_key_env, entry.api_key or "not-needed")
        prov: dict[str, Any] = {
            "type": "maxim_peer",
            "base_url": entry.url,
            "api_key_env": api_key_env,
            "model": entry.model,
            "allow_local_endpoints": True,
            "pricing_required": False,
        }
        if entry.timeout_s is not None:
            prov["timeout_s"] = float(entry.timeout_s)
        return prov, False

    if entry.origin is Origin.CLOUD:
        if entry.url:
            api_key_env = _placement_provider_api_key_env(key)
            if entry.api_key:
                os.environ.setdefault(api_key_env, entry.api_key)
            prov = {
                "type": "openai",
                "base_url": entry.url,
                "api_key_env": api_key_env,
                "model": entry.model,
                "pricing_required": False,
            }
            if entry.timeout_s is not None:
                prov["timeout_s"] = float(entry.timeout_s)
            return prov, True
        # cloud profile (claude-sonnet, gpt-4o, …): use the profile's backend.
        # Normalize the alias to its canonical profile key first
        # (``claude-sonnet`` → ``claude-sonnet-4-6``), mirroring how cli_utils
        # resolves --cloud-fallback before storing the model name.
        try:
            from maxim.models.language.config import _BUILTIN_PROFILES, normalize_llm_profile
        except Exception:
            return None, False
        resolved_model = normalize_llm_profile(entry.model or "")
        profile = _BUILTIN_PROFILES.get(resolved_model, {})
        if not profile.get("cloud"):
            return None, False
        api_key_env = profile.get("api_key_env", "ANTHROPIC_API_KEY")
        # Carry the placement's resolved key in the PROVIDER ENTRY (backends
        # prefer env, then the entry's ``api_key``) instead of mutating
        # os.environ — a setdefault'd key made re-keyed setup unrecoverable
        # in-process (post-merge review Exec #3). An explicit
        # ``export ANTHROPIC_API_KEY`` still wins at the backend.
        prov = {
            "type": profile.get("backend", "anthropic"),
            "model": resolved_model,
            "api_key_env": api_key_env,
            "n_ctx": profile.get("n_ctx", 200000),
        }
        if entry.api_key:
            prov["api_key"] = entry.api_key
        if entry.timeout_s is not None:
            prov["timeout_s"] = float(entry.timeout_s)
        return prov, True

    # LOCAL fallback: resolve the profile to a concrete local provider entry.
    try:
        from maxim.models.language.config import load_llm_config

        lc = load_llm_config(profile_override=entry.model)
    except Exception:
        return None, False
    prov = {
        "type": lc.backend,
        "model": lc.model,
        "model_path": lc.model_path,
        "n_ctx": lc.n_ctx,
        "prompt_style": lc.prompt_style,
    }
    if entry.timeout_s is not None:
        prov["timeout_s"] = float(entry.timeout_s)
    return prov, False


def _inject_placement_tail(
    llm_config: Any, tail: tuple[ProviderPlacement, ...], *, lane_name: str, logger: Any | None
) -> Any:
    """Append each tail placement entry as a fallback provider, in order.

    Generalizes ``_maybe_inject_cloud_fallback`` from the single env-driven
    cloud-fallback to an arbitrary ordered placement tail — the multi-element
    "ride on LLMRouter's provider_priority" unification. The primary backend is
    already built into ``llm_config``; this adds the fallback providers after it
    in ``routing.provider_priority`` so the router fails over in placement order.
    Returns ``llm_config`` unchanged when ``tail`` is empty.

    ``lane_name`` qualifies each fallback provider's api-key env var so two
    tiers' credentialed fallbacks at the same tail index don't collide on one
    ``MAXIM_PLACEMENT_*`` name (mirrors the per-lane ``MAXIM_LANE_<NAME>_API_KEY``
    of the primary remote path).
    """
    if not tail:
        return llm_config
    providers = dict(getattr(llm_config, "providers", None) or {})
    if not providers:
        providers["primary"] = {
            "type": llm_config.backend,
            "model": llm_config.model,
            "model_path": getattr(llm_config, "model_path", ""),
            "n_ctx": llm_config.n_ctx,
        }
    routing = dict(getattr(llm_config, "routing", None) or {})
    priority = list(routing.get("provider_priority", list(providers.keys())))
    any_cloud = False
    for i, entry in enumerate(tail):
        key = f"{lane_name}-placement-fallback-{i + 1}"
        prov, is_cloud = _placement_entry_to_provider(entry, key)
        if prov is None:
            if logger is not None:
                logger.warning(
                    "Placement fallback #%d (origin=%s) could not be compiled — skipped", i + 1, entry.origin
                )
            continue
        providers[key] = prov
        if key not in priority:
            priority.append(key)
        any_cloud = any_cloud or is_cloud
    routing["provider_priority"] = priority
    budget = os.environ.get("MAXIM_CLOUD_SESSION_BUDGET", "").strip()
    if budget:
        try:
            routing["max_session_cost"] = float(budget)
        except ValueError:
            pass
    return dataclasses.replace(
        llm_config,
        providers=providers,
        routing=routing,
        cloud_enabled=(llm_config.cloud_enabled or any_cloud),
    )


def _apply_cloud_cli_overrides(
    lane_configs: dict[str, Any],
    logger: Any | None,
) -> dict[str, Any]:
    """Re-express ``--cloud-lane`` (``MAXIM_CLOUD_LANE_<TIER>_MODEL``) as a
    placement edit (Phase 3c).

    PREPENDS a CLOUD placement entry as the tier's PRIMARY, on top of any
    config.json/derived placement base (CLI > config). For a tier with no prior
    explicit placement this yields ``[CLOUD]`` — cloud-only, the legacy
    ``--cloud-lane`` behavior; on top of a config placement it makes cloud the
    primary and keeps the rest as fallback. Legacy ``remote_*`` / ``model_profile``
    fields are cleared so the placement is unambiguous. ``MAXIM_CLOUD_LANE_*``
    stays a deprecated-but-working alias through 1.x. The model name is already
    normalized to its canonical cloud-profile key by ``cli_utils``.
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
        if not _BUILTIN_PROFILES.get(cloud_model, {}).get("cloud"):
            if logger is not None:
                logger.warning(
                    "MAXIM_CLOUD_LANE_%s_MODEL=%s is not a cloud profile — ignoring",
                    lane_name.upper(),
                    cloud_model,
                )
            continue
        _warn_cloud_flag_deprecated("--cloud-lane", logger)
        cloud_primary = ProviderPlacement(origin=Origin.CLOUD, model=cloud_model)
        existing = tuple(cfg.placement)
        out[lane_name] = dataclasses.replace(
            cfg,
            placement=(cloud_primary,) + existing,
            model_profile=None,
            remote_url=None,
            remote_model=None,
            remote_api_key=None,
        )
        if logger is not None:
            logger.info("Lane '%s' --cloud-lane → CLOUD placement primary: %s", lane_name, cloud_model)
    return out


def _apply_cloud_fallback_override(
    lane_configs: dict[str, Any],
    logger: Any | None,
    peer_owned: bool,
) -> dict[str, Any]:
    """Re-express ``--cloud-fallback`` (``MAXIM_CLOUD_FALLBACK_MODEL``) as an
    APPENDED CLOUD placement on the large tier (Phase 3c).

    Materializes the current primary (explicit config placement, or the derived
    1-element placement from the legacy fields) so the cloud model becomes a
    FALLBACK *after* it — i.e. ``[<primary>, CLOUD-fallback]`` → the
    multi-element tail-injection compile builds "local primary → cloud
    fallback" as one ordered list. This is the unification that subsumes the
    old env-driven ``_maybe_inject_cloud_fallback`` injection; because placement
    is now set, the build takes the tail-injection path (not the legacy env
    path), so the cloud provider is added exactly once. Deprecated-but-working
    alias through 1.x; model name already normalized by ``cli_utils``.
    """
    model = os.environ.get("MAXIM_CLOUD_FALLBACK_MODEL", "").strip()
    if not model:
        return lane_configs
    try:
        from maxim.models.language.config import _BUILTIN_PROFILES
    except Exception:
        return lane_configs
    if not _BUILTIN_PROFILES.get(model, {}).get("cloud"):
        if logger is not None:
            logger.warning("MAXIM_CLOUD_FALLBACK_MODEL=%s is not a cloud profile — ignoring", model)
        return lane_configs
    _warn_cloud_flag_deprecated("--cloud-fallback", logger)

    out = dict(lane_configs)
    cfg = out.get("large")
    if cfg is None:
        return lane_configs
    base = tuple(cfg.placement) if cfg.placement else derive_placement(cfg, peer_owned=peer_owned)
    if not base:
        # Nothing to fall back FROM (no primary) — skip rather than make cloud
        # the silent primary.
        if logger is not None:
            logger.warning("--cloud-fallback ignored: large lane has no primary to fall back from")
        return lane_configs
    fallback = ProviderPlacement(origin=Origin.CLOUD, model=model)
    out["large"] = dataclasses.replace(cfg, placement=base + (fallback,))
    if logger is not None:
        logger.info("Lane 'large' --cloud-fallback → appended CLOUD placement: %s", model)
    return out


def _ensure_lane_profiles_available(
    lane_configs: dict[str, Any],
    capabilities: Any | None,
    logger: Any | None,
) -> dict[str, Any]:
    """Make sure each local lane's GGUF is on disk; auto-download if missing.

    For each lane that runs locally (no ``remote_url``, profile is a
    llama_cpp profile), call :func:`maxim.models.download.ensure_available`.
    On failure, re-walk the tier table with the missing profile filtered
    out so we land on the next-smaller model the user already has, rather
    than failing startup with a cryptic load error.

    No-op for cloud lanes, remote lanes, and lanes with no profile.
    """
    try:
        from maxim.models.download import LLM_MODELS, ensure_available
        from maxim.models.language.config import _BUILTIN_PROFILES
        from maxim.runtime.lane_models import detect_tiers
    except Exception as e:
        if logger is not None:
            logger.debug("ensure_available unavailable, skipping preflight: %s", e)
        return lane_configs

    out = dict(lane_configs)
    re_walked = False
    excluded: set[str] = set()

    for lane_name, cfg in list(out.items()):
        if cfg.remote_url:
            continue
        profile = cfg.model_profile
        if not profile:
            continue
        # Skip cloud profiles outright — they have no GGUF.
        profile_data = _BUILTIN_PROFILES.get(profile, {})
        if profile_data.get("cloud"):
            continue
        # Skip profiles that aren't in our download registry. They're either
        # custom user profiles (handled out-of-band) or non-llama_cpp
        # backends like transformers (HF auto-downloads on first use).
        if profile not in LLM_MODELS:
            continue
        if _profile_has_local_file(profile):
            continue

        if logger is not None:
            logger.info(
                "Lane '%s' profile '%s' missing on disk — running ensure_available",
                lane_name,
                profile,
            )
        ok = ensure_available(profile, logger=logger)
        if ok:
            continue

        # Download refused / failed / declined — re-walk the tier table
        # with this profile filtered out so we land on a smaller model
        # the user already has. Only re-walk once per build call to keep
        # the cost bounded.
        excluded.add(profile)
        if re_walked or capabilities is None:
            # Already re-walked once or we have no caps to walk against —
            # just clear the broken profile so downstream lane assembly
            # falls back to its own defaults.
            out[lane_name] = dataclasses.replace(cfg, model_profile=None)
            continue
        try:
            new_tiers = detect_tiers(
                capabilities,
                profile_available=lambda p, _excl=excluded: _profile_has_local_file(p) and p not in _excl,
            )
        except Exception as e:
            if logger is not None:
                logger.warning("Tier re-walk after failed download failed: %s", e)
            out[lane_name] = dataclasses.replace(cfg, model_profile=None)
            continue
        re_walked = True
        # Replace lanes with their re-walked counterparts. We deliberately
        # only carry over fields that the re-walk knows about — preserving
        # remote_url overrides etc. that came from upstream stages.
        for new_name, new_cfg in new_tiers.items():
            existing = out.get(new_name)
            if existing is None:
                out[new_name] = new_cfg
                continue
            out[new_name] = dataclasses.replace(
                existing,
                model_profile=new_cfg.model_profile,
                device=new_cfg.device,
                n_gpu_layers=new_cfg.n_gpu_layers,
                n_ctx=new_cfg.n_ctx,
                kv_quant_mode=new_cfg.kv_quant_mode,
                requires_gpu=new_cfg.requires_gpu,
            )

    return out


def _apply_local_llm_override(
    lane_configs: dict[str, Any],
    logger: Any | None,
) -> dict[str, Any]:
    """Clear the large lane's remote_url when --llm names a local profile.

    When a user passes ``--llm mistral-7b`` on the command line, they
    expect that model to run **on this machine**. If peer config is
    also active (``~/.config/maxim/peer.yml`` exists), the old
    "reconcile" block in build_primary_router would rewrite this into
    "ask the leader for mistral-7b" — which the leader ignores because
    it has whatever model IS loaded, regardless of the name in the
    request. The user's explicit CLI flag was silently translated
    into something that didn't match their intent.

    This helper is the counterpart to ``_apply_cloud_cli_overrides``.
    It runs AFTER env overrides + cloud overrides so:

    - Cloud profiles land via ``_apply_cloud_cli_overrides`` and exit
      that function without touching remote_url semantics here.
    - Local profiles named via ``--llm`` reach this function and
      clear the large lane's remote fields, forcing the auto-spawn
      path (or the local backend) to serve the requested model.
    - If ``MAXIM_LLM_PROFILE`` is unset, this function is a no-op —
      peer config's remote_url stays intact and routing is unchanged.

    Precedence (all three scenarios now resolve correctly):

    +---------------------------+-------------------------------------+
    | User config               | Outcome                             |
    +===========================+=====================================+
    | ``--llm <local>``         | Local wins. remote_url cleared.     |
    | + peer config             | Large lane spawns locally.          |
    +---------------------------+-------------------------------------+
    | ``--llm <cloud>``         | Cloud wins (handled upstream by     |
    | + peer config             | ``_apply_cloud_cli_overrides``).    |
    +---------------------------+-------------------------------------+
    | No ``--llm``, peer config | Peer wins (unchanged behavior).     |
    +---------------------------+-------------------------------------+

    Scoped to the ``large`` tier by design. Medium/small lanes do not
    have remote_url set today; if a future feature distributes them
    across machines, that routing will need its own override path
    (via the ``MAXIM_LANE_{NAME}_REMOTE_URL`` env var family).
    """
    profile_name = os.environ.get("MAXIM_LLM_PROFILE", "").strip()
    if not profile_name:
        return lane_configs

    try:
        from maxim.models.language.config import _BUILTIN_PROFILES
    except Exception:
        return lane_configs

    profile_data = _BUILTIN_PROFILES.get(profile_name, {})
    if profile_data.get("cloud"):
        # Cloud profiles are handled by _apply_cloud_cli_overrides and
        # also by the existing cloud-backend construction path. This
        # override is for LOCAL profiles only — "run this model here".
        return lane_configs

    out = dict(lane_configs)
    target_lane = "large"
    cfg = out.get(target_lane)
    if cfg is None:
        return lane_configs

    # Phase 3c: if the lane carries an explicit placement (config.json /
    # --cloud-lane / --cloud-fallback), --llm <local> overrides its PRIMARY with
    # the local profile while KEEPING any fallback tail (so `--llm mistral
    # --cloud-fallback claude-sonnet` = local primary → cloud fallback). CLI >
    # config.
    if cfg.placement:
        local_primary = ProviderPlacement(origin=Origin.LOCAL, model=profile_name)
        new_placement = (local_primary,) + tuple(cfg.placement[1:])
        out[target_lane] = dataclasses.replace(
            cfg,
            placement=new_placement,
            remote_url=None,
            remote_model=None,
            remote_api_key=None,
        )
        if logger is not None:
            logger.info(
                "Lane 'large' --llm=%s → LOCAL placement primary (keeping %d fallback(s))",
                profile_name,
                len(cfg.placement) - 1,
            )
        return out

    if not cfg.remote_url:
        return lane_configs

    if logger is not None:
        logger.info(
            "Lane '%s' local override: --llm=%s clears remote_url=%s (local wins over peer)",
            target_lane,
            profile_name,
            cfg.remote_url,
        )

    out[target_lane] = dataclasses.replace(
        cfg,
        model_profile=profile_name,
        remote_url=None,
        remote_model=None,
        remote_api_key=None,
    )
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
      5. Return (manager.get_backend("large"), manager).

    Returns (None, manager) if the large tier resolves to nothing — e.g. the
    tier has no profile AND no remote URL, the backend build fails, or the
    LLMConfig ends up disabled. Callers should fall back to their own default
    LLMRouter construction in that case.

    Raises:
        BackendGateError: if the safety gates (MAXIM_MAX_CONCURRENT_BACKENDS
        or MAXIM_MAX_CLOUD_LANES) refuse the request. Surface this to the
        user; it means their configuration exceeds the declared limits.
    """
    from maxim.runtime.capabilities import RuntimeCapabilities, detect_compute_resources
    from maxim.runtime.lane_models import (
        apply_lane_env_overrides,
        apply_tier_config_overrides,
        detect_tiers,
        load_function_overrides,
    )

    # Lane-routing auto-load (C4 of config_unification.md):
    #
    # 1. The IM5 fold auto-migration shim in config_loader.load_config()
    #    has already moved peer.yml → config.json on first startup when
    #    conditions matched (config.json absent + peer.yml present +
    #    cloudflared absent), so subsequent startups read from
    #    config.json::lanes.large.* via the unified resolve_setting
    #    precedence chain.
    # 2. For ongoing back-compat with downstream code that reads the
    #    MAXIM_LANE_LARGE_REMOTE_* env vars directly, populate them
    #    from resolve_setting here when they aren't already set. This
    #    is the compat shim that lets the rest of the runtime not
    #    care whether the value came from env, config.json, or peer.yml.
    # 3. Self-hosted classification preservation (IM5 audit note):
    #    when lanes.large.remote_url resolves from env or config (not
    #    from cloud config), set MAXIM_MAX_CLOUD_LANES=1 so the cloud-
    #    lane gate doesn't refuse the request. Cloud-provider opt-in
    #    stays explicit via config.json::cloud.enabled.
    # 4. peer.yml fallback (deprecated, compat-read): if neither
    #    config.json nor env has the data but peer.yml still does
    #    (e.g., cloudflared-present case where auto-migration didn't
    #    fire), populate from peer.yml with a once-per-startup INFO
    #    deprecation log.
    #
    # This MUST run before the persisted model restore below — the
    # lane config signals "my large tier lives on the leader", and
    # restoring a stale local profile would cause
    # _apply_local_llm_override to clear the remote_url and silently
    # route everything locally.
    _has_peer_config = _apply_lane_config_to_env(logger)

    # Restore persisted model preference (from --llm or maxim.run(model=...))
    # when no explicit MAXIM_LLM_PROFILE is set for this session.
    #
    # Skip when peer config is active: the peer config means "my large-tier
    # LLM lives on the leader". Restoring a persisted local profile would
    # cause _apply_local_llm_override to clear the remote_url, silently
    # routing everything locally instead of to the leader's GPU. The user
    # can still override with an explicit --llm flag (which sets
    # MAXIM_LLM_PROFILE before we get here), and _apply_local_llm_override
    # will correctly clear the remote for that session.
    if not os.environ.get("MAXIM_LLM_PROFILE", "").strip() and not _has_peer_config:
        _persisted = _read_persisted_model()
        if _persisted:
            os.environ["MAXIM_LLM_PROFILE"] = _persisted
            if logger is not None:
                logger.info("Restored persisted model preference: %s", _persisted)

    # ── Migration safety: pin existing leaders to pre-upgrade profile ──
    # Before the tier table expands (planned P4a), any leader that was
    # running on the OLD tier table's defaults gets its choice frozen
    # to the persisted-model file so a subsequent `maxim peer restart`
    # does NOT silently swap its model. See peer_leader_flexibility_plan
    # F0.3 for the full rationale.
    #
    # This code path only fires on the FIRST startup on a machine that:
    #   1. Has run Maxim before (util/ exists with content)
    #   2. Has NO persisted model file (nothing was explicitly pinned)
    #   3. Has NO explicit MAXIM_LLM_PROFILE in env (user didn't override)
    #   4. Has NOT already run the pin (marker file absent)
    # The marker file prevents repeat execution across future restarts.
    _maybe_pin_pre_upgrade_profile(capabilities, logger)

    # Note: the old "reconcile --language-model with peer remote config"
    # block used to live here. It rewrote `--llm <local>` into
    # "send <local> as model name to leader" on peer configs, which
    # was the wrong default — the leader ignores the requested model
    # name and serves whatever is loaded. The correct precedence is
    # now applied after tier detection via _apply_local_llm_override,
    # which clears remote_url on the affected lane so the user's
    # local profile actually runs locally. See peer_leader_flexibility
    # plan P1 for the rationale and precedence table.

    if capabilities is None:
        has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()
        capabilities = RuntimeCapabilities(
            has_gpu=has_gpu,
            gpu_type=gpu_type,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
        )

    try:
        lane_configs = detect_tiers(
            capabilities,
            profile_available=_profile_has_local_file,
        )
    except Exception as e:
        if logger is not None:
            logger.warning("Tier detection failed (using defaults): %s", e)
        from maxim.runtime.worker_pool import DEFAULT_TIERS

        lane_configs = {name: dataclasses.replace(cfg) for name, cfg in DEFAULT_TIERS.items()}

    # If peer config provides a remote URL for the large tier but local
    # hardware didn't create one (e.g., Mac with no CUDA GPU), inject a
    # large tier placeholder so apply_lane_env_overrides() can wire it
    # to the leader. The remote URL IS the large tier — the leader's GPU.
    if "large" not in lane_configs:
        _large_url = os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL", "").strip()
        if _large_url:
            lane_configs["large"] = LaneConfig(
                name="large",
                max_workers=1,
                requires_gpu=False,  # Remote — no local GPU needed
            )

    # Apply tier/function config from llm.json if present
    _llm_json_tier_config: dict = {}
    _llm_json_func_config: dict = {}
    try:
        from maxim.models.language.config import load_llm_config as _load_cfg

        _raw_cfg = _load_cfg()
        _raw = getattr(_raw_cfg, "_raw", {}) or {}
        _llm_json_tier_config = _raw.get("tiers", {})
        _llm_json_func_config = _raw.get("functions", {})
    except Exception:
        pass
    if _llm_json_tier_config:
        lane_configs = apply_tier_config_overrides(lane_configs, _llm_json_tier_config)

    lane_configs = apply_lane_env_overrides(lane_configs)
    # lane_capability_placement_split.md Phase 3c: apply the declarative
    # config.json placement as the BASE (config → LaneConfig.placement). CLI
    # flags (--cloud-lane / --cloud-fallback / --llm) edit placement on top of
    # this base below, preserving the CLI > config precedence chain.
    lane_configs = _apply_config_placements(lane_configs, logger)
    lane_configs = _apply_cloud_cli_overrides(lane_configs, logger)
    # Local --llm <profile> override: runs AFTER cloud overrides so that
    # when both --llm and --cloud-lane are set, cloud wins for the
    # lane it targets while local wins for anything else. See P1 in
    # docs/plans/archive/peer_leader_flexibility_plan.md for the full precedence
    # table and the rationale for clearing remote_url here.
    lane_configs = _apply_local_llm_override(lane_configs, logger)

    # P5: ensure each local-lane profile has an on-disk GGUF. If the user
    # is offline + non-tty + has not opted in to auto-download, this is a
    # no-op (the lane keeps its missing profile, llm_server load fails,
    # and the existing fallback paths take over). When the file is
    # missing AND the user is interactive (or set --auto-download), this
    # downloads the GGUF before auto-spawn tries to load it. On failure,
    # we re-walk the tier table with the missing profile filtered out so
    # the lane lands on the next-smaller model the user already has.
    lane_configs = _ensure_lane_profiles_available(lane_configs, capabilities, logger)

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
    # ``_auto_spawn_sources`` captures whether each tier reused an existing
    # server (singleton check) or spawned a fresh one — fed into the decision
    # log so the Exp 37 harness preflight can distinguish a safe leader-local
    # reuse from the harness-on-leader cascade.
    _auto_spawn_sources: dict[str, str] = {}
    lane_configs = _maybe_auto_spawn_server(capabilities, lane_configs, logger, sources_out=_auto_spawn_sources)

    # --cloud-fallback: append a CLOUD fallback AFTER the primary is finalized.
    # Runs post-auto-spawn (review fold) so it materializes the FINAL primary —
    # incl. the auto-spawned-server remote_url — into the placement, rather than
    # freezing a pre-spawn in-process primary and silently bypassing the spawned
    # llama-cpp-server. For a derived lane this yields [<spawned/derived primary>,
    # CLOUD]; the tail-injection then builds the fallback once.
    lane_configs = _apply_cloud_fallback_override(lane_configs, logger, _has_peer_config)

    # Build FunctionRouter with available tiers derived from lane_configs
    from maxim.runtime.function_router import DEFAULT_FUNCTIONS, FunctionRouter

    available_tiers = set(lane_configs.keys())
    functions = dict(DEFAULT_FUNCTIONS)
    if _llm_json_func_config:
        functions.update(load_function_overrides(_llm_json_func_config))
    fn_router = FunctionRouter(functions=functions, available_tiers=available_tiers)

    manager = LaneBackendManager(
        lane_configs,
        peer_owned=_has_peer_config,
        function_router=fn_router,
    )

    if logger is not None:
        logger.info("Lane assignments: %s", manager.describe())

    _print_lane_banner(manager)

    # P9: persist a post-mortem record of this routing decision so the
    # user can run `maxim doctor --last-decision` later and see exactly
    # which env vars + capabilities + probe outcomes produced this
    # configuration. Best-effort — never fails the build.
    try:
        from maxim.runtime import decision_log

        record = decision_log.build_record(
            caps=capabilities,
            final_lane_configs=lane_configs,
            decision_sources=_derive_decision_sources(lane_configs, _auto_spawn_sources),
            peer_config_loaded=_has_peer_config,
        )
        decision_log.append_record(record)
    except Exception as e:
        if logger is not None:
            logger.debug("decision_log append failed (non-fatal): %s", e)

    # Start heartbeat monitor if enabled via env flags (peer or solo mode)
    try:
        from maxim.runtime.heartbeat import get_heartbeat_monitor, should_enable_heartbeat

        if should_enable_heartbeat():
            get_heartbeat_monitor().start()
    except Exception:
        pass

    # Primary inference backend: "large" tier.
    router = manager.get_backend("large")
    if router is not None and hasattr(router, "update_provider_n_ctx"):
        register_router(router)
    return router, manager


_PROBE_FIX_HINTS: dict[str, str] = {
    "ok": "",
    # Key-drift detection (post-config-unification UX fold, 2026-06-03):
    # canonical 401 cause is "your stored key differs from leader's
    # current key". Direct the operator at the rotate-and-re-paste
    # path. See peer/probe_classify.py for the long-form message.
    "auth_rejected": (
        "Stored API key likely doesn't match the leader's current key. "
        "On the leader: `maxim tunnel key show` (or `maxim tunnel key rotate` for "
        "a fresh key), then on this node: `maxim peer connect <leader-url>` and "
        "paste the fresh key"
    ),
    "dns_fail": "Check the hostname spelling in peer.yml or $MAXIM_LANE_*_REMOTE_URL",
    "tls_error": "Check the TLS certificate validity on the leader",
    "connection_refused": "Leader is not accepting connections — is `maxim` running there?",
    "timeout": "Leader did not respond in time — is it cold-loading a model?",
    "http_5xx": "Leader returned a server error — check `maxim peer logs`",
    "other": "Unexpected error — check `maxim peer logs`",
}


def _log_probe_failure(
    log: Any | None,
    lane_name: str,
    url: str,
    result: Any,  # ProbeResult
) -> None:
    """Emit a structured warning with an outcome-specific fix hint."""
    if log is None:
        return
    hint = _PROBE_FIX_HINTS.get(result.outcome, "unknown outcome")
    log.warning(
        "Lane '%s' probe failed (outcome=%s, url=%s): %s. Fix: %s. Falling back to local model selection.",
        lane_name,
        result.outcome,
        url,
        result.detail,
        hint,
    )


def _validate_remote_urls(lane_configs: dict[str, Any], logger: Any | None) -> dict[str, Any]:
    """Probe each lane's remote_url and drop unreachable ones.

    Replaces the older "skip public URLs" heuristic with a structured
    probe via :meth:`_MaximPeerBackend.for_url(...).health_check` backed by a
    short-TTL on-disk cache (:mod:`maxim.runtime.probe_cache`). Catches
    dead Cloudflare tunnels that the previous "loopback only" guard
    would have left wired to the lane, generating retry storms on every
    real request.

    Behavior:
      * Empty remote_url → skip.
      * ``MAXIM_SKIP_REMOTE_PROBE=1`` → skip entirely (CI escape hatch).
      * Cache hit fresher than ``MAXIM_REMOTE_PROBE_CACHE_TTL_S`` (default
        60 s) → use cached outcome.
      * Otherwise probe with two-attempt retry, persist the result.
      * ``ok`` / ``auth_rejected`` → keep the lane wired. ``auth_rejected``
        emits a warning so the user knows to rotate the key, but the
        leader is still alive so we don't fall back locally.
      * Anything else → log a structured warning + drop ``remote_url`` so
        auto-spawn or local fallback takes over.
    """
    if os.environ.get("MAXIM_SKIP_REMOTE_PROBE", "").strip().lower() in (
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    ):
        return dict(lane_configs)

    from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
    from maxim.runtime import probe_cache
    from maxim.runtime.llm_server import ProbeResult

    ttl_s = _safe_float_env("MAXIM_REMOTE_PROBE_CACHE_TTL_S", 60.0, min_value=0.0, max_value=600.0)
    # First-attempt timeout: 1.5s handles a warm-but-cold httpx connection
    # (measured ~710ms on Mac→Cloudflare→RTX5080). Previous default of 0.8s
    # was within 90ms of the measured cold-connection latency — any
    # system/DNS hiccup caused a false negative.
    first_timeout = _safe_float_env("MAXIM_REMOTE_PROBE_FIRST_TIMEOUT_S", 1.5, min_value=0.2, max_value=5.0)
    # Retry timeout: 8.0s handles a dormant Cloudflare tunnel that needs to
    # re-establish its origin connection (observed >2.5s after 8+ minutes of
    # inactivity). Previous default of 2.5s fell in the gap between
    # "warm tunnel fast" and "dormant tunnel re-establishing".
    retry_timeout = _safe_float_env("MAXIM_REMOTE_PROBE_RETRY_TIMEOUT_S", 8.0, min_value=0.5, max_value=10.0)

    cache = probe_cache.load_cache()
    cache_dirty = False
    out = dict(lane_configs)

    for name, cfg in lane_configs.items():
        url = getattr(cfg, "remote_url", None)
        if not url:
            continue

        cached = cache.get(url)
        if cached and probe_cache.is_fresh(cached, ttl_s):
            result = ProbeResult(
                url=url,
                outcome=cached.get("outcome", "other"),
                detail=cached.get("detail", ""),
                latency_ms=cached.get("latency_ms"),
            )
        else:
            result = _MaximPeerBackend.for_url(
                url,
                api_key=cfg.remote_api_key,
            ).health_check(
                first_timeout_s=first_timeout,
                retry_timeout_s=retry_timeout,
            )
            cache[url] = {
                "outcome": result.outcome,
                "detail": result.detail,
                "probed_at": time.time(),
                "latency_ms": result.latency_ms,
            }
            cache_dirty = True

        if result.outcome == "auth_rejected":
            # Leader is alive but rejected the key. KEEP the lane wired —
            # the user needs to rotate the key, not fall back to local.
            if logger is not None:
                logger.warning(
                    "Lane '%s' auth_rejected by %s. Fix: %s",
                    name,
                    url,
                    _PROBE_FIX_HINTS["auth_rejected"],
                )
            continue
        if result.outcome == "ok":
            continue

        _log_probe_failure(logger, name, url, result)
        out[name] = dataclasses.replace(cfg, remote_url=None, remote_model=None, remote_api_key=None)

    if cache_dirty:
        probe_cache.save_cache(cache)
    return out


# Decision-source value marking a lane that reused an already-running
# llama-cpp-server (the singleton-check path). Distinct from "tier_table"
# (the cascade signature: a sub-sim spawned its OWN local server) so the
# Exp 37 harness preflight can tell a safe leader-local reuse apart from
# the failure class. See decision_log.py / benchmark_cross_session.py.
_REUSED_SERVER_SOURCE = "reused_server"


def _derive_decision_sources(
    lane_configs: dict[str, Any],
    auto_spawn_sources: dict[str, str],
) -> dict[str, str]:
    """Attribute a routing ``source`` to each lane for the decision log.

    Precedence per tier:
      1. An explicit source recorded by ``_maybe_auto_spawn_server``
         (``"reused_server"`` for the singleton-reuse path, ``"tier_table"``
         for a fresh local spawn).
      2. ``"env"`` when ``MAXIM_LANE_<TIER>_REMOTE_URL`` is set non-empty
         (also covers config.json values, which ``_apply_lane_config_to_env``
         setdefaults into the same env var).
      3. ``"config"`` when the lane carries a ``remote_url`` from another
         resolution path.
      4. ``"tier_table"`` otherwise (in-process local inference — the
         sub-sim is doing its own large-tier work).

    The Exp 37 harness preflight rejects ``"tier_table"`` on the large tier
    as the harness-on-leader cascade signature; ``"reused_server"`` and
    ``"env"`` are the safe states.
    """
    sources: dict[str, str] = {}
    for tier, cfg in lane_configs.items():
        if tier in auto_spawn_sources:
            sources[tier] = auto_spawn_sources[tier]
            continue
        env_key = f"MAXIM_LANE_{tier.upper()}_REMOTE_URL"
        if os.environ.get(env_key, "").strip():
            sources[tier] = "env"
        elif getattr(cfg, "remote_url", None):
            sources[tier] = "config"
        else:
            sources[tier] = "tier_table"
    return sources


def _maybe_auto_spawn_server(
    capabilities: Any,
    lane_configs: dict[str, Any],
    logger: Any | None,
    *,
    sources_out: dict[str, str] | None = None,
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
    # Mutable globals live in _server_mod (llm_server.py) — single source of truth

    auto_raw = os.environ.get("MAXIM_AUTO_SPAWN_LLM_SERVER", "").strip().lower()
    if auto_raw in ("0", "false", "f", "no", "n", "off"):
        return lane_configs

    if capabilities is None or not getattr(capabilities, "has_gpu", False):
        return lane_configs

    # Tier-based: primary inference tier is "large"; fall back to "infer"
    # for backward compat with custom lane_configs.
    _infer_tier = "large" if "large" in lane_configs else "infer"
    infer_cfg = lane_configs.get(_infer_tier)
    if infer_cfg is None or infer_cfg.remote_url or not infer_cfg.model_profile:
        return lane_configs

    # If the lane's assigned profile is a cloud profile (anthropic, openai,
    # gemini, etc.), there's no local GGUF to serve and auto-spawn is not
    # applicable. Skip BEFORE consulting MAXIM_LLM_PROFILE / persisted model
    # — the lane config is the explicit, current decision about what this
    # lane should do. Previously a stale persisted model from a prior
    # `maxim peer llm mistral-7b` swap could resurrect local auto-spawn
    # even after `--cloud-lane large claude-sonnet` explicitly routed to
    # Anthropic, wasting 120s of spawn startup on a model the lane won't
    # use. See cancellation.py / cli.py for the cloud-lane override path.
    try:
        from maxim.models.language.config import _BUILTIN_PROFILES

        _lane_profile_data = _BUILTIN_PROFILES.get(infer_cfg.model_profile, {})
        if _lane_profile_data.get("cloud"):
            if logger is not None:
                logger.info(
                    "Auto-spawn skipped: lane '%s' is configured for cloud profile '%s' (no local GGUF needed)",
                    _infer_tier,
                    infer_cfg.model_profile,
                )
            return lane_configs
    except Exception as e:
        logger.debug("Cloud-profile detection failed, continuing auto-spawn check: %s", e)

    try:
        import llama_cpp.server  # noqa: F401
    except ImportError:
        if logger is not None:
            logger.warning(
                "Auto-spawn skipped: llama_cpp.server not installed. "
                "Run: pip install 'pymaxim[llm-server]' to enable local server spawning."
            )
        return lane_configs

    # Check for a persisted model from a previous `maxim peer llm` swap.
    # This survives os.execv restarts so the leader comes back with the
    # same model the user selected, not the default profile.
    # Explicit --language-model / MAXIM_LLM_PROFILE takes priority over
    # persisted model — the user's CLI flag is the strongest signal.
    cli_profile = os.environ.get("MAXIM_LLM_PROFILE", "").strip()
    persisted = _read_persisted_model()
    if cli_profile:
        effective_profile = cli_profile
    else:
        effective_profile = persisted or infer_cfg.model_profile
    if persisted and not cli_profile and logger is not None:
        logger.info("Auto-spawn: using persisted model '%s' from previous swap", persisted)

    # Resolve the profile's GGUF path
    try:
        from maxim.models.language.config import load_llm_config

        profile_cfg = load_llm_config(profile_override=effective_profile)
        model_path = getattr(profile_cfg, "model_path", "")
    except Exception as e:
        logger.warning("Auto-spawn: failed to resolve model profile %r: %s", effective_profile, e)
        return lane_configs
    if not model_path or not Path(model_path).is_file():
        if logger is not None:
            logger.warning(
                "Auto-spawn skipped: GGUF file not found at %s (profile=%s). "
                "Run: maxim --list-models to see download status, or set MAXIM_LLM_PROFILE.",
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
        except Exception as e:
            logger.warning("Failed to read tunnel API key: %s", e)
            api_key = None

    # Singleton check: if a llama-cpp-server is already serving the right
    # model on this port, REUSE it instead of kill+respawn. This is the
    # structural fix for the harness-on-leader cascade (CLAUDE.md "Don't run
    # the benchmark harness on the same machine as the leader"): a sub-sim
    # that auto-detects leader role on the leader machine must NOT kill the
    # leader's live server (port collision → dead upstream → 502s → operator
    # kills maxim → cloudflared dies → tunnel down → 530s). Probes via the
    # canonical entry point and fails loud on a wrong-model server.
    from maxim.runtime.llm_server import check_existing_llm_server

    existing_url = f"http://127.0.0.1:{port}/v1"
    decision = check_existing_llm_server(
        url=existing_url,
        api_key=api_key,
        expected_model_path=model_path,
        expected_profile=effective_profile,
        logger=logger,
    )
    if decision == "reuse":
        if _server_mod._active_spawner is not None and _server_mod._active_spawner.is_running:
            # We spawned this server in this process — refresh tracking.
            _server_mod._active_model = effective_profile
            _server_mod._llm_start_time = time.time()
        if sources_out is not None:
            sources_out[_infer_tier] = _REUSED_SERVER_SOURCE
        out = dict(lane_configs)
        out[_infer_tier] = dataclasses.replace(
            infer_cfg,
            remote_url=existing_url,
            remote_model=effective_profile,
            remote_api_key=infer_cfg.remote_api_key or api_key,
        )
        return out

    # decision == "spawn": port not responding — may have a stale process
    # holding VRAM.  Only kill+sleep if something was actually found.
    try:
        from maxim.runtime.local_server_spawner import kill_stale_llm_servers

        n_killed = kill_stale_llm_servers(port)
        if n_killed:
            if logger is not None:
                logger.info(
                    "Killed %d stale llama-cpp-server process(es) on port %d before spawn",
                    n_killed,
                    port,
                )
            time.sleep(0.5)  # Brief pause for port release
    except Exception:
        pass

    try:
        from maxim.runtime.local_server_spawner import LocalServerSpawner
    except Exception:
        return lane_configs

    # api_key was already loaded above (before auto-discovery) — reuse here
    # when spawning the subprocess. Solo mode (127.0.0.1 bind) has api_key=None
    # which means the server spawns without --api_key and accepts all requests.

    # n_ctx precedence (peer_leader_flexibility_plan P4c):
    #   1. MAXIM_LLM_N_CTX  — explicit user override (--llm-n-ctx flag)
    #   2. MAXIM_AUTO_SPAWN_N_CTX — legacy env (kept for in-place upgrades)
    #   3. infer_cfg.n_ctx — value computed by detect_tiers/estimate_max_ctx
    #   4. 8192 — final fallback if none of the above resolved
    _cli_n_ctx = _safe_int_env("MAXIM_LLM_N_CTX", 0)
    _legacy_n_ctx = _safe_int_env("MAXIM_AUTO_SPAWN_N_CTX", 0)
    _tier_n_ctx = int(infer_cfg.n_ctx or 0)
    resolved_n_ctx = _cli_n_ctx or _legacy_n_ctx or _tier_n_ctx or 8192

    # Plan 3.6 R5: warn loudly if this (profile, n_ctx) combo will push VRAM
    # past the 95% spillover ratio. Observability only — does not override
    # the operator's n_ctx choice.
    _check_vram_spillover_risk(
        profile_name=effective_profile,
        resolved_n_ctx=resolved_n_ctx,
        vram_gb=float(getattr(capabilities, "vram_gb", 0.0) or 0.0),
        kv_quant_mode=str(getattr(infer_cfg, "kv_quant_mode", "") or "f16"),
        logger=logger,
    )

    spawner = LocalServerSpawner(
        model_path=model_path,
        port=port,
        bind_host=role_decision.bind_host,
        n_ctx=resolved_n_ctx,
        n_gpu_layers=infer_cfg.n_gpu_layers,
        api_key=api_key,
        kv_quant_mode=infer_cfg.kv_quant_mode,
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
    _server_mod._active_spawner = spawner
    _server_mod._active_model = effective_profile
    _server_mod._llm_start_time = time.time()

    # A FRESH local spawn (vs the singleton-reuse path above) is the
    # cascade signature the harness preflight watches for: a sub-sim that
    # spawned its own large-tier server on the leader machine. Mark it
    # "tier_table" so _derive_decision_sources doesn't mislabel the
    # localhost remote_url as "config".
    if sources_out is not None:
        sources_out[_infer_tier] = "tier_table"

    # Rewrite the primary inference tier to point at the spawned server.
    # Auto-wire the API key for the leader's own client so local inference doesn't 401.
    out = dict(lane_configs)
    infer_api_key = infer_cfg.remote_api_key or api_key
    out[_infer_tier] = dataclasses.replace(
        infer_cfg,
        remote_url=url,
        remote_model=effective_profile,
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


def _validate_swap_profile(profile: str) -> tuple[str, dict[str, Any]]:
    """Validate that ``profile`` can be swapped onto llama-cpp-server.

    Resolves the profile name, blocks cloud and non-llama_cpp backends,
    and confirms the profile is in the registry. Returns
    ``(resolved_name, profile_data)``. Raises ``ValueError`` on any
    client-side issue.
    """
    from maxim.models.language.config import (
        _BUILTIN_PROFILES,
        list_llm_profiles,
        normalize_llm_profile,
    )

    resolved = normalize_llm_profile(profile)
    if not resolved:
        raise ValueError("Empty model name")

    profile_data = _BUILTIN_PROFILES.get(resolved, {})

    if profile_data.get("cloud"):
        raise ValueError(f"Profile '{resolved}' is a cloud provider — cannot run on llama-cpp-server")

    backend = profile_data.get("backend", "llama_cpp")
    if backend not in ("llama_cpp", "llamacpp", "llama"):
        raise ValueError(f"Profile '{resolved}' uses {backend} backend — not compatible with llama-cpp-server")

    available = list_llm_profiles()
    if available and resolved not in available:
        raise ValueError(f"Unknown model profile: {profile}. Available: {', '.join(sorted(available))}")

    return resolved, profile_data


def _resolve_swap_target(resolved: str) -> tuple[Any, str]:
    """Resolve the LLMConfig + GGUF path for ``resolved`` profile.

    Raises ``ValueError`` if the profile fails to load and
    ``FileNotFoundError`` (with a copy-paste fix line) if the GGUF is
    not on disk.
    """
    from maxim.models.language.config import load_llm_config

    try:
        cfg = load_llm_config(profile_override=resolved)
        model_path = getattr(cfg, "model_path", "")
    except Exception as e:
        raise ValueError(f"Failed to resolve profile '{resolved}': {e}") from e

    if not model_path or not Path(model_path).is_file():
        raise FileNotFoundError(
            f"GGUF not found: {model_path}|Run on leader: python -m maxim.models.download --llm {resolved}"
        )
    return cfg, model_path


def _estimate_swap_n_ctx(cfg: Any, *, fallback: int = 8192) -> int:
    """Recompute n_ctx for a hot-swap target using current capabilities.

    ``cfg`` is the freshly-loaded ``LLMConfig`` for the swap target. We
    detect the running machine's VRAM, look up the profile's arch metadata,
    and ask :func:`estimate_max_ctx` for the largest viable context. Falls
    back to the profile's declared ``n_ctx`` (or ``fallback``) when the
    capability path can't produce a number — never returns 0.
    """
    try:
        from maxim.models.language.config import _BUILTIN_PROFILES
        from maxim.runtime.capabilities import detect_compute_resources
        from maxim.runtime.lane_models import _lookup_fallback_nctx, estimate_max_ctx
    except Exception:
        return int(getattr(cfg, "n_ctx", 0) or fallback)

    profile_name = str(getattr(cfg, "profile", "") or "").strip()
    profile_data = _BUILTIN_PROFILES.get(profile_name, {}) if profile_name else {}
    try:
        _, _, vram_gb, _ = detect_compute_resources()
    except Exception:
        vram_gb = 0.0

    formula_ctx = estimate_max_ctx(profile_data, vram_gb) if vram_gb > 0 else 0
    fallback_ctx = _lookup_fallback_nctx(profile_name, vram_gb) if profile_name else 0
    if formula_ctx > 0 and fallback_ctx > 0:
        n_ctx = min(formula_ctx, fallback_ctx)
    else:
        n_ctx = formula_ctx or fallback_ctx
    if n_ctx > 0:
        return n_ctx
    # Last resort: trust the profile's declared value, then the static fallback.
    declared = int(getattr(cfg, "n_ctx", 0) or 0)
    return declared if declared > 0 else fallback


def _check_vram_spillover_risk(
    profile_name: str,
    resolved_n_ctx: int,
    vram_gb: float,
    kv_quant_mode: str,
    logger: Any | None,
) -> None:
    """Emit a loud WARN if the spawning model will push VRAM past the spillover ratio.

    Plan 3.6 R5. Pure observability: does NOT lower ``resolved_n_ctx``, does
    NOT abort spawn. The operator made a choice (CLI flag, env var, tier
    estimator); our job is to make the consequence visible BEFORE they hit a
    30-50x slowdown.

    The threshold is deliberately static (95% of physical VRAM). If the math
    says the model fits but observed VRAM later shows otherwise, that's a
    signal the ``estimate_max_ctx`` headroom assumption (1.5 GB) is too low
    for this model class — capture it in lessons-learned, don't paper over
    it here.
    """
    if logger is None:
        return
    if not profile_name or resolved_n_ctx <= 0 or vram_gb <= 0:
        return
    try:
        from maxim.models.language.config import _BUILTIN_PROFILES
        from maxim.runtime.lane_models import project_vram_usage
    except Exception:
        return

    profile_meta = _BUILTIN_PROFILES.get(profile_name, {})
    projection = project_vram_usage(
        profile_name,
        profile_meta,
        resolved_n_ctx,
        vram_gb,
        kv_quant_mode=kv_quant_mode or "f16",
    )
    if projection is None or not projection.spillover_risk:
        return

    # StructuredFormatter reads `extra={"event": ..., "data": {...}}` — flat
    # extra fields outside `data` are silently dropped from MAXIM_LOG_FILE
    # JSONL. Same class as feedback_structured_formatter_short_keys.md.
    logger.warning(
        "vram_spillover_risk: profile=%s n_ctx=%d projected=%.1fGB "
        "(weights=%.1f + kv=%.1f + headroom=%.1f) vs physical=%.1fGB. "
        "Recommended n_ctx=%d. Expect 5-30x slowdown once KV cache fills. "
        "Lower MAXIM_LLM_N_CTX or switch to a smaller profile.",
        projection.profile,
        projection.n_ctx,
        projection.projected_total_gb,
        projection.weights_gb,
        projection.kv_cache_gb,
        projection.headroom_gb,
        projection.physical_vram_gb,
        projection.recommended_n_ctx,
        extra={
            "event": "vram_spillover_risk",
            "data": {
                "profile": projection.profile,
                "n_ctx": projection.n_ctx,
                "projected_gb": projection.projected_total_gb,
                "weights_gb": projection.weights_gb,
                "kv_cache_gb": projection.kv_cache_gb,
                "headroom_gb": projection.headroom_gb,
                "physical_vram_gb": projection.physical_vram_gb,
                "recommended_n_ctx": projection.recommended_n_ctx,
            },
        },
    )


def _drain_in_flight_requests(deadline_s: float = 5.0) -> None:
    """Wait up to ``deadline_s`` seconds for in-flight large-lane requests to finish.

    Best-effort: silently no-ops if the metrics registry is unavailable.
    """
    import time as _time

    try:
        from maxim.models.language.lane_metrics import get_metrics_registry

        metrics = get_metrics_registry().get("large")
        deadline = _time.time() + deadline_s
        while metrics.current_in_flight > 0 and _time.time() < deadline:
            _time.sleep(0.25)
    except Exception as e:
        logger.debug("In-flight drain skipped (metrics unavailable): %s", e)


def _stop_active_server(logger_obj: Any | None) -> None:
    """Stop the currently-active llama-cpp-server (if any) and clear globals."""
    if _server_mod._active_spawner is not None:
        if logger_obj is not None:
            logger_obj.info("LLM swap: stopping current server (model=%s)", _server_mod._active_model)
        _server_mod._active_spawner.stop()
        _server_mod._active_spawner = None
        _server_mod._active_model = None


def _start_swap_server(
    *,
    cfg: Any,
    model_path: str,
    api_key: str | None,
    port: int,
    logger_obj: Any | None,
) -> tuple[Any, float, int]:
    """Spawn a fresh LocalServerSpawner for the swap target.

    Returns ``(spawner, startup_s, n_ctx)``. Raises ``RuntimeError`` if
    the server fails to become ready in 120s.
    """
    import time as _time

    from maxim.runtime.local_server_spawner import LocalServerSpawner

    bind_host = "0.0.0.0"  # noqa: S104 — leader always binds all interfaces
    try:
        from maxim.runtime.leader_mode import detect_role

        bind_host = detect_role().bind_host
    except Exception as e:
        logger.debug("Could not detect leader role for bind_host: %s", e)

    # P4c: re-run estimate_max_ctx for the swap target so a hot-swap from
    # qwen-14b → mistral-7b on a 16 GB card doesn't accidentally inherit
    # the previous model's tight 4096 budget (or vice versa). The CLI
    # override (MAXIM_LLM_N_CTX) wins if set; otherwise we recompute from
    # current capabilities + the swap target's arch metadata.
    _cli_n_ctx = _safe_int_env("MAXIM_LLM_N_CTX", 0)
    if _cli_n_ctx > 0:
        n_ctx = _cli_n_ctx
    else:
        n_ctx = _estimate_swap_n_ctx(cfg, fallback=8192)
    spawner = LocalServerSpawner(
        model_path=model_path,
        port=port,
        bind_host=bind_host,
        n_ctx=n_ctx,
        n_gpu_layers=cfg.n_gpu_layers,
        api_key=api_key,
        kv_quant_mode=getattr(cfg, "kv_quant_mode", "f16"),
    )

    if logger_obj is not None:
        logger_obj.info(
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
            f"Server failed to start for model '{Path(model_path).name}'. "
            "Check GPU memory — the model may be too large."
        )
    return spawner, startup_s, n_ctx


def swap_llm_server(profile: str, logger: Any | None = None) -> dict[str, Any]:
    """Hot-swap the llama-cpp-server to a different model.

    Called by LeaderProxy's /v1/admin/llm-swap endpoint.  Stops the current
    server, resolves the new profile to a GGUF path, and starts a fresh
    LocalServerSpawner.

    Returns a result dict consumed by the admin endpoint.
    Raises ValueError for client errors (400-level) and RuntimeError for
    server errors (500-level).
    """
    # Validate + resolve target before touching any globals
    resolved, _profile_data = _validate_swap_profile(profile)

    # Acquire swap lock (non-blocking — reject concurrent swaps)
    if not _swap_lock.acquire(blocking=False):
        raise RuntimeError("LLM swap already in progress")

    try:
        # Same model = no-op
        if (
            _server_mod._active_model
            and _server_mod._active_model == resolved
            and _server_mod._active_spawner is not None
            and _server_mod._active_spawner.is_running
        ):
            return {"status": "already_running", "model": resolved}

        cfg, model_path = _resolve_swap_target(resolved)

        # Determine port and API key (reuse current config)
        try:
            port = int(os.environ.get("MAXIM_AUTO_SPAWN_PORT", "8100"))
        except ValueError:
            port = 8100

        api_key: str | None = None
        try:
            from maxim.tunnel.keys import read_key

            api_key = read_key()
        except Exception as e:
            if logger is not None:
                logger.warning("Failed to read tunnel API key for swap: %s", e)

        previous_model = _server_mod._active_model or "none"

        _drain_in_flight_requests(deadline_s=5.0)
        _stop_active_server(logger)

        spawner, startup_s, n_ctx = _start_swap_server(
            cfg=cfg,
            model_path=model_path,
            api_key=api_key,
            port=port,
            logger_obj=logger,
        )

        _server_mod._active_spawner = spawner
        _server_mod._active_model = resolved
        _server_mod._llm_start_time = time.time()
        _write_persisted_model(resolved)

        # Update the LLM router's cached n_ctx so context_window_routing
        # uses the new model's context window, not the startup value.
        # Without this, hot-swapping from a 4k model to a 32k model
        # still rejects requests that would fit in the new window.
        try:
            for router in _find_active_routers():
                router.update_provider_n_ctx("local", n_ctx)
                if logger is not None:
                    logger.info("LLM swap: updated router n_ctx to %d", n_ctx)
        except Exception as e:
            logger.debug("Could not update router n_ctx after swap: %s", e)

        return {
            "status": "swapped",
            "model": resolved,
            "model_path": model_path,
            "port": port,
            "n_ctx": n_ctx,
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
    except Exception as e:
        if logger is not None:
            logger.warning("Tunnel daemon spawner not available: %s", e)
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
    except Exception as e:
        if logger is not None:
            logger.warning("Leader proxy module not available: %s", e)
        return
    start_leader_proxy(api_key=api_key, bind_host=bind_host)

    # Start heartbeat monitor in leader mode (always on for leaders)
    try:
        from maxim.runtime.heartbeat import get_heartbeat_monitor

        get_heartbeat_monitor().start()
    except Exception:
        pass


def _print_lane_banner(manager: "LaneBackendManager") -> None:
    """Emit a compact banner showing per-lane backend assignments.

    Logged at INFO level so users see which backend each lane is using —
    answers "is it really talking to my server?" at a glance. CLI users
    with default verbosity (>=1) see this; library users can suppress
    via ``logging.getLogger("maxim").setLevel(logging.WARNING)``.
    """
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
    # Owner-requested startup truth (2026-08-03): when NOTHING in the config
    # can bill a cloud call, say once that cost tracking is inert — instead
    # of a "Missing pricing" WARNING on every local call. MUST use the
    # fail-closed billing-surface predicate, not has_cloud_placement():
    # a cloud profile with no URL (--llm claude-sonnet) derives a LOCAL
    # placement, and printing "disabled" on the highest-spend path is the
    # exact lie the two-lens review caught.
    if not manager.has_cloud_billing_surface():
        lines.append("  Cost tracking disabled (local models only)")
    lines.append(" " + "─" * 62)
    logger.info("\n".join(lines))


__all__ = ["LaneBackendManager", "BackendGateError", "build_primary_router"]
