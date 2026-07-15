"""Individual diagnostic checks run by `maxim doctor`.

Each check is a pure function: takes `PlatformInfo`, returns a `CheckResult`.
Results include a status, message, and platform-specific fix hint. The fix
hint is rendered verbatim in the doctor output — it should be copy-pasteable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from maxim.doctor.platform_detect import PlatformInfo, detect_wsl_ip

logger = logging.getLogger(__name__)

Status = Literal["ok", "warn", "fail", "info"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    fix: str | None = None  # platform-specific fix hint
    retry_id: str | None = None  # only set when retry is meaningful

    @property
    def symbol(self) -> str:
        return {"ok": "✓", "warn": "⚠", "fail": "✗", "info": "ℹ"}[self.status]


# ─── environment checks ────────────────────────────────────────────────────


def check_gpu() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult(
            name="GPU / CUDA",
            status="warn",
            message="torch not installed — CPU-only mode",
            fix="Install GPU stack: pip install -e '.[llm-torch]'",
        )
    if not torch.cuda.is_available():
        import os

        cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_vis == "":
            return CheckResult(
                name="GPU / CUDA",
                status="warn",
                message="CUDA hidden (CUDA_VISIBLE_DEVICES=''). Likely the Blackwell workaround.",
                fix=(
                    "If you have a Blackwell GPU (RTX 50xx), current behavior keeps CUDA\n"
                    "  enabled by default. If it's hidden, set: unset MAXIM_BLACKWELL_HIDE_CUDA"
                ),
            )
        return CheckResult(
            name="GPU / CUDA",
            status="warn",
            message="No CUDA device available — inference will be CPU-only",
        )
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    return CheckResult(
        name="GPU / CUDA",
        status="ok",
        message=f"{props.name} ({vram_gb:.1f} GB VRAM)",
    )


def check_tier_detection(caps=None) -> CheckResult:
    """Report which capability tiers are available on this hardware."""
    try:
        from maxim.runtime.capabilities import RuntimeCapabilities, detect_compute_resources
        from maxim.runtime.lane_models import detect_tiers

        if caps is None:
            has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()
            caps = RuntimeCapabilities(
                has_gpu=has_gpu,
                gpu_type=gpu_type,
                vram_gb=vram_gb,
                ram_gb=ram_gb,
            )
        tiers = detect_tiers(caps)
    except ImportError as e:
        return CheckResult(
            name="LLM Tiers",
            status="warn",
            message=f"Tier detection unavailable: {e}",
            fix="pip install -e '.[llm-llama]'  # or install 'pymaxim[llm-llama]'",
        )
    except Exception as e:
        return CheckResult(
            name="LLM Tiers",
            status="warn",
            message=f"Tier detection failed: {e}",
        )

    tier_names = sorted(tiers.keys())
    profiles = {name: cfg.model_profile for name, cfg in tiers.items()}

    if "large" in tiers or "medium" in tiers:
        return CheckResult(
            name="LLM Tiers",
            status="ok",
            message=f"Tiers: {', '.join(tier_names)}. Profiles: {profiles}",
        )
    return CheckResult(
        name="LLM Tiers",
        status="warn",
        message=f"Only 'small' tier detected ({caps.ram_gb:.0f}GB RAM, GPU: {caps.gpu_type or 'none'})",
        fix=(
            "Agent inference needs a large or medium tier. Options:\n"
            "  --language-model mistral-7b          # if you have 8+ GB RAM\n"
            "  --cloud-fallback claude-sonnet       # use cloud for inference\n"
            "  --tier-model large=<remote-url>      # point to a remote leader"
        ),
        retry_id="tier_detection",
    )


def check_llama_cpp_server_installed() -> CheckResult:
    try:
        import llama_cpp.server  # noqa: F401
    except ImportError:
        return CheckResult(
            name="llama-cpp-server installed",
            status="fail",
            message="llama_cpp.server not installed — auto-spawn disabled",
            fix="pip install -e '.[llm-server]'",
        )
    return CheckResult(
        name="llama-cpp-server installed",
        status="ok",
        message="ready for auto-spawn",
    )


def check_server_reachable(port: int = 8100) -> CheckResult:
    """Probe the local auto-spawn port."""
    from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

    url = f"http://127.0.0.1:{port}/v1"
    if (
        _MaximPeerBackend.for_url(url)
        .health_check(
            first_timeout_s=1.0,
            retry_timeout_s=2.0,
            enable_stage2=False,
        )
        .is_reachable
    ):
        return CheckResult(
            name="Auto-spawn server",
            status="ok",
            message=f"responding at {url}",
            retry_id="server",
        )
    return CheckResult(
        name="Auto-spawn server",
        status="warn",
        message=f"nothing responding at {url}",
        fix="Run `maxim` in another terminal — auto-spawn will launch one.",
        retry_id="server",
    )


def check_llm_model_active() -> CheckResult:
    """Check if an LLM model is loaded and report which one.

    Post-implementation field-coverage fold: reads ``llm.profile`` via
    the unified ``resolve_setting`` chain so config.json values are
    surfaced alongside the live ``_active_model`` and persisted-model
    sources. Pre-fold the check only saw env / persisted-model state
    and silently drifted from C5's "Resolved Config" section when
    config.json was the actual source.
    """
    try:
        # Must read `_active_model` through `_server_mod` — importing it by
        # name binds the value at import time and diverges from the live
        # state (CLAUDE.md "Mutable globals + module extraction" lesson).
        # This was a silent bug: the check only ever reported the persisted
        # model name, never the live one.
        import maxim.runtime.llm_server as _server_mod
        from maxim.runtime.config_loader import resolve_setting
        from maxim.runtime.lane_backends import _read_persisted_model

        active = _server_mod._active_model
        try:
            configured, configured_source = resolve_setting("llm.profile")
        except Exception:
            configured, configured_source = None, "default"
        persisted = _read_persisted_model()

        if active:
            return CheckResult(
                name="LLM model",
                status="ok",
                message=f"active model: {active}",
            )
        if configured and configured_source in ("config", "env", "cli"):
            return CheckResult(
                name="LLM model",
                status="warn",
                message=f"configured model: {configured} (not yet loaded) [source={configured_source}]",
                fix=f"maxim peer llm {configured}",
                retry_id="llm_model",
            )
        if persisted:
            return CheckResult(
                name="LLM model",
                status="warn",
                message=f"persisted model: {persisted} (not yet loaded)",
                fix=f"maxim peer llm {persisted}",
                retry_id="llm_model",
            )
        return CheckResult(
            name="LLM model",
            status="warn",
            message="no model configured — use `maxim config set llm.profile <name>` to set one",
            fix="maxim config set llm.profile qwen2.5-14b",
            retry_id="llm_model",
        )
    except Exception:
        return CheckResult(
            name="LLM model",
            status="info",
            message="model tracking not available (solo mode)",
        )


# ─── environment variable config ───────────────────────────────────────────


def check_resolved_config() -> list["CheckResult"]:
    """C5 of config_unification.md: "Resolved Config" doctor section.

    Walks every absorbed field in :data:`maxim.runtime.config_loader._FIELD_TO_ENV`
    and surfaces the effective value + source marker. This is the
    single-place answer to "what does this instance think it's configured
    as?" — collapsing what previously required cross-referencing ~96 env
    vars + 4 config files + 2 role detectors.

    The N2 fold from the pre-implementation review pinned the
    catch-everything semantics: shadow, convergence, role=client
    coercion, missing key file, mode != 0600, inline-string-pre-
    migration, keyring not installed, peer.yml deprecated,
    format-version forward-compat warning all surface as WARN rows
    in this section.

    Each absorbed field becomes one :class:`CheckResult`. Status:

    - ``ok`` — field resolved cleanly (any source)
    - ``warn`` — env shadows config with a different value, OR an
      ancillary condition fires (missing api_key file, peer.yml
      present-but-deprecated, etc.)
    - ``fail`` — parse/coerce error on the field
    - ``info`` — default or convergence (both env and config set the
      same value)
    """
    from maxim.runtime.config_loader import (
        _FIELD_TO_ENV,
        _env_is_set,
        config_path,
        load_config,
        resolve_setting,
    )

    results: list[CheckResult] = []

    # Load config once for reuse across all field resolutions
    try:
        cfg = load_config()
    except Exception as e:
        results.append(
            CheckResult(
                name="config.json",
                status="fail",
                message=f"failed to load: {e}",
                fix=(f"Inspect the file directly:\n  cat {config_path()}\nThen `maxim config edit` to fix syntax."),
            )
        )
        return results

    # Show file location at the top of the section
    if config_path().is_file():
        results.append(
            CheckResult(
                name="config.json path",
                status="ok",
                message=str(config_path()),
            )
        )
    else:
        results.append(
            CheckResult(
                name="config.json path",
                status="info",
                message=f"{config_path()} (file absent — using defaults)",
            )
        )

    # Walk every absorbed field
    import os as _os

    for field_path in sorted(_FIELD_TO_ENV.keys()):
        env_name = _FIELD_TO_ENV[field_path]
        env_present = _env_is_set(env_name)

        try:
            value, source = resolve_setting(field_path, config=cfg)
        except Exception as e:
            results.append(
                CheckResult(
                    name=field_path,
                    status="fail",
                    message=f"{type(e).__name__}: {e}",
                )
            )
            continue

        # Render the value compactly
        display_value = _format_doctor_value(value)

        # Decide status + message based on source + env shadow
        if source == "env":
            env_raw = _os.environ[env_name]
            cfg_value = _read_config_for_doctor(cfg, field_path)
            if cfg_value is not None:
                if cfg_value == value:
                    # Convergence (CR3 fold)
                    results.append(
                        CheckResult(
                            name=field_path,
                            status="info",
                            message=f"{display_value}  [source=env, config.json also sets identically]",
                        )
                    )
                else:
                    # Divergence (CR3 fold)
                    results.append(
                        CheckResult(
                            name=field_path,
                            status="warn",
                            message=f"{display_value}  [source=env, shadows config.json={_format_doctor_value(cfg_value)!s}]",
                            fix=(
                                f"To make config.json win, unset the env var:\n"
                                f"  unset {env_name}\n"
                                f"Or to make env explicit, update config.json:\n"
                                f"  maxim config set {field_path} {env_raw}"
                            ),
                        )
                    )
            else:
                results.append(
                    CheckResult(
                        name=field_path,
                        status="ok",
                        message=f"{display_value}  [source=env]",
                    )
                )
        elif source == "config":
            results.append(
                CheckResult(
                    name=field_path,
                    status="ok",
                    message=f"{display_value}  [source=config.json]",
                )
            )
        else:  # default
            # If env is set-but-empty (i.e. POSIX `export FOO=`), surface it
            # as a WARN so the operator sees the leaked-empty-export case.
            raw_env = _os.environ.get(env_name)
            if raw_env is not None and raw_env.strip() == "":
                results.append(
                    CheckResult(
                        name=field_path,
                        status="warn",
                        message=f"{display_value}  [source=default; {env_name} is set to empty string — treated as unset per C-1 fold]",
                        fix=f"  unset {env_name}",
                    )
                )
            else:
                # Quietly surface the default — info status, no fix
                results.append(
                    CheckResult(
                        name=field_path,
                        status="info",
                        message=f"{display_value}  [source=default]",
                    )
                )
        del env_present  # consumed via env_present check above

    # Ancillary checks: api_key_ref file health, peer.yml deprecation
    results.extend(_check_lane_api_key_refs_health(cfg))
    results.extend(_check_peer_yml_deprecation())

    # Legacy env-var migration: operators upgrading from Maxim ≤0.9.1
    # likely have several MAXIM_* env vars exported in their shell rc /
    # launchd plist. Each fires its own per-startup deprecation INFO
    # (PR #318's I-4 fold), but that's easy to miss in the log scroll.
    # This row aggregates the situation into a single doctor row with
    # a copy/paste migration script so operators can clean up in one
    # pass.
    results.extend(_check_legacy_env_migration(cfg))

    # llm_timeout_scalability.md follow-up: surface the proxy
    # context-overflow admission gate state so operators see at-a-glance
    # whether oversize prompts will be rejected cleanly (413) or quietly
    # hang the upstream model. ON when llm.n_ctx is resolvable AND
    # MAXIM_PROXY_CONTEXT_ADMISSION isn't set to a disable value.
    results.extend(_check_proxy_context_admission())

    # Adjacent VRAM projection: at the configured (llm.n_ctx, llm.profile)
    # combination, will the model fit comfortably in available VRAM? The
    # existing `check_vram_pressure` only works on nvidia-smi systems
    # (and ALSO does live-monitoring) — this is a cross-platform
    # projection that works on Apple Silicon + AMD + NVIDIA via
    # `detect_compute_resources`, surfaced here so the admission row
    # and the VRAM-fit row appear together in the Resolved Config view.
    # Answers "will this config fit?" rather than "is VRAM full now?".
    results.extend(_check_vram_context_pressure())

    # lane_capability_placement_split.md Phase 3c: surface the per-tier
    # capability × placement axis. For each tier with an explicit config.json
    # placement, show the ordered origins; WARN on legacy remote_* + placement
    # coexistence (placement wins via derive_placement's short-circuit, so the
    # remote_* fields are silently ignored — an easy operator foot-gun).
    for tier in ("large", "medium", "small"):
        tier_cfg = getattr(getattr(cfg, "lanes", None), tier, None)
        placement = getattr(tier_cfg, "placement", ()) if tier_cfg is not None else ()
        if not placement:
            continue
        chain = " → ".join(f"{p.origin}:{p.model or p.url or '?'}" for p in placement)
        results.append(
            CheckResult(
                name=f"lanes.{tier}.placement",
                status="ok",
                message=f"capability '{tier}' → placement [{chain}]",
            )
        )
        legacy_set = any(getattr(tier_cfg, f, None) for f in ("remote_url", "remote_model", "remote_api_key_ref"))
        if legacy_set:
            results.append(
                CheckResult(
                    name=f"lanes.{tier}.placement",
                    status="warn",
                    message=(
                        f"lanes.{tier} sets BOTH an explicit placement AND legacy "
                        f"remote_* fields — placement wins, the remote_* fields are ignored"
                    ),
                    fix=(
                        f"Remove the redundant remote_* fields from lanes.{tier}, or drop the "
                        f"placement to use them:\n  maxim config edit"
                    ),
                )
            )

    return results


def _check_proxy_context_admission() -> list["CheckResult"]:
    """Surface the proxy context-overflow admission gate's effective state.

    Returns one ``CheckResult`` row showing whether the gate is enabled
    (with the active context window + overhead) or disabled (with the
    reason: missing ``llm.n_ctx`` vs explicit operator opt-out).

    The row is intentionally ``info`` (not ``warn``) in both states —
    enabled is the recommended path but disabled is a legitimate choice
    for operators running a model where the proxy can't determine n_ctx
    (custom servers, etc.). The startup WARNING in
    ``leader_proxy._get_proxy_context_window`` is the louder signal.
    """
    import os as _os

    from maxim.runtime.leader_proxy import (
        _PROXY_CHAR_TO_TOKEN_RATIO,
        _is_admission_enabled,
        _resolve_context_overhead_tokens,
        _resolve_proxy_context_window,
    )

    if not _is_admission_enabled():
        return [
            CheckResult(
                name="proxy.context_admission",
                status="warn",
                message="disabled  [source=MAXIM_PROXY_CONTEXT_ADMISSION]",
                fix=(
                    "Oversize prompts will hang the upstream model with no "
                    "actionable error. Remove the opt-out to re-enable:\n"
                    "  unset MAXIM_PROXY_CONTEXT_ADMISSION"
                ),
            )
        ]

    n_ctx = _resolve_proxy_context_window()
    if n_ctx is None:
        return [
            CheckResult(
                name="proxy.context_admission",
                status="warn",
                message="disabled  [reason=llm.n_ctx not configured]",
                fix=(
                    "Oversize prompts will hang the upstream model with no "
                    "actionable error. Set llm.n_ctx to the upstream's actual "
                    "runtime context window:\n"
                    "  maxim config set llm.n_ctx 8192\n"
                    "Or via env:\n"
                    "  export MAXIM_LLM_N_CTX=8192"
                ),
            )
        ]

    overhead = _resolve_context_overhead_tokens()
    overhead_env = _os.environ.get("MAXIM_PROXY_CONTEXT_OVERHEAD_TOKENS")
    overhead_src = "env" if overhead_env and overhead_env.strip() else "default"
    return [
        CheckResult(
            name="proxy.context_admission",
            status="ok",
            message=(
                f"enabled  [context_window={n_ctx}, "
                f"overhead={overhead} tokens (source={overhead_src}), "
                f"char_to_token_ratio={_PROXY_CHAR_TO_TOKEN_RATIO}]"
            ),
        )
    ]


def _check_vram_context_pressure() -> list["CheckResult"]:
    """Project whether (llm.n_ctx, llm.profile) fits comfortably in VRAM.

    Cross-platform companion to ``check_vram_pressure``:

    - ``check_vram_pressure`` uses ``nvidia-smi`` and ALSO does live
      monitoring of current VRAM utilization. It silently no-ops on
      Apple Silicon / AMD systems because ``nvidia-smi`` is absent.
    - This check uses ``detect_compute_resources`` (cross-platform) +
      ``project_vram_usage`` (pure-data projection). Surfaces the
      same ``vram_spillover_risk`` warning the leader emits at spawn
      time, but in the ``maxim doctor`` flow so the operator sees it
      AT DIAGNOSTIC time rather than only AT STARTUP.

    Positioned in the Resolved Config section directly after the
    proxy context-admission row so the operator reads "your gate
    rejects oversize at 16384 tokens" right next to "your 16384
    setting projects 34.8 GB of 36 GB physical VRAM — recommended
    13312." Both rows answer adjacent questions about the same
    context budget.

    Returns ``[]`` (no row) when n_ctx or profile is unresolvable
    (the admission row already noted the absence) — keeps the doctor
    output tight.
    """
    try:
        from maxim.models.language.config import _BUILTIN_PROFILES
        from maxim.runtime.capabilities import detect_compute_resources
        from maxim.runtime.config_loader import load_config, resolve_setting
        from maxim.runtime.lane_models import project_vram_usage
    except Exception as e:
        return [
            CheckResult(
                name="proxy.vram_context_fit",
                status="info",
                message=f"VRAM projection unavailable: {type(e).__name__}",
            )
        ]

    try:
        cfg = load_config()
        n_ctx_value, n_ctx_src = resolve_setting("llm.n_ctx", config=cfg)
        profile_value, profile_src = resolve_setting("llm.profile", config=cfg)
    except Exception as e:
        return [
            CheckResult(
                name="proxy.vram_context_fit",
                status="info",
                message=f"VRAM projection skipped: config resolution failed ({type(e).__name__})",
            )
        ]

    if n_ctx_value is None or profile_value is None:
        return []  # admission row already noted the absence

    try:
        _, _, vram_gb, _ = detect_compute_resources()
    except Exception:
        vram_gb = 0.0

    if not vram_gb or vram_gb <= 0:
        return [
            CheckResult(
                name="proxy.vram_context_fit",
                status="info",
                message=("VRAM budget unknown (no GPU / unified memory detected) — projection skipped"),
            )
        ]

    profile_meta = _BUILTIN_PROFILES.get(str(profile_value), {})
    try:
        projection = project_vram_usage(
            str(profile_value),
            profile_meta,
            int(n_ctx_value),
            float(vram_gb),
        )
    except Exception as e:
        return [
            CheckResult(
                name="proxy.vram_context_fit",
                status="info",
                message=f"VRAM projection failed: {type(e).__name__}",
            )
        ]
    if projection is None:
        return [
            CheckResult(
                name="proxy.vram_context_fit",
                status="info",
                message=(
                    f"VRAM projection unavailable for profile {profile_value!r} (no arch metadata in _BUILTIN_PROFILES)"
                ),
            )
        ]

    msg_base = (
        f"projected {projection.projected_total_gb:.1f} GB "
        f"(weights {projection.weights_gb:.1f} + KV {projection.kv_cache_gb:.1f} "
        f"+ {projection.headroom_gb:.1f} headroom) vs physical "
        f"{projection.physical_vram_gb:.1f} GB at n_ctx={projection.n_ctx}"
    )

    if projection.spillover_risk:
        # Pick the fix narrative based on where the operator is setting
        # the value. The mesh.yml two-layer-split invariant prefers
        # config.json for declarative settings; if env is the source,
        # we also recommend unsetting it.
        fix_lines: list[str] = [
            "# Lower n_ctx to the projection's recommended value:",
            f"maxim config set llm.n_ctx {projection.recommended_n_ctx}",
        ]
        if n_ctx_src == "env":
            fix_lines.extend(
                [
                    "# Then unset the env shadow so config.json wins:",
                    "unset MAXIM_LLM_N_CTX",
                    "# (also remove the export from ~/.zshrc / ~/.bashrc / launchd plist)",
                ]
            )
        fix_lines.append("# Restart the leader to apply (kill + relaunch, or `maxim peer restart` from a peer)")
        return [
            CheckResult(
                name="proxy.vram_context_fit",
                status="warn",
                message=f"{msg_base} — KV cache will spill to shared memory once full (5-30x slowdown)",
                fix="\n".join(fix_lines),
            )
        ]

    return [
        CheckResult(
            name="proxy.vram_context_fit",
            status="ok",
            message=f"{msg_base} — fits comfortably",
        )
    ]


def _format_doctor_value(value: object) -> str:
    if value is None:
        return "<unset>"
    if isinstance(value, str):
        return value if value else "<empty>"
    return repr(value)


def _read_config_for_doctor(cfg, field_path: str):
    """Walk the dot path on a MaximConfig to return the raw config value,
    or ``None`` if the field equals the default (i.e., no operator-set value).
    Used by check_resolved_config to detect shadow/convergence states."""
    from maxim.runtime.config_loader import MaximConfig

    parts = field_path.split(".")
    obj = cfg
    default_obj = MaximConfig()
    for part in parts:
        obj = getattr(obj, part, None)
        default_obj = getattr(default_obj, part, None)
        if obj is None:
            return None
    if obj == default_obj:
        return None
    return obj


def _check_lane_api_key_refs_health(cfg) -> list["CheckResult"]:
    """Eager probe on every lane's api_key_ref — file missing /
    mode != 0600 / keyring not installed / inline pre-migration."""
    from pathlib import Path

    from maxim.runtime.config_loader import resolve_setting

    out: list[CheckResult] = []
    for tier in ("large", "medium", "small"):
        try:
            ref, source = resolve_setting(f"lanes.{tier}.remote_api_key_ref", config=cfg)
        except Exception:
            continue
        if not ref:
            continue
        # Path-mode ref: check file existence + mode
        if ref.startswith("/") or ref.startswith("~"):
            path = Path(ref).expanduser()
            if not path.is_file():
                out.append(
                    CheckResult(
                        name=f"lanes.{tier}.remote_api_key_ref",
                        status="warn",
                        message=f"file missing: {path}",
                        fix=(
                            f"Write the leader's API key to the referenced file (mode 0600):\n"
                            f"  echo $LEADER_API_KEY > {path}\n"
                            f"  chmod 0600 {path}"
                        ),
                    )
                )
                continue
            try:
                mode = path.stat().st_mode & 0o777
            except OSError:
                mode = None
            if mode is not None and mode != 0o600:
                out.append(
                    CheckResult(
                        name=f"lanes.{tier}.remote_api_key_ref",
                        status="warn",
                        message=f"mode is {oct(mode)} (want 0o600): {path}",
                        fix=f"  chmod 0600 {path}",
                    )
                )
        # Keyring URI: check keyring availability
        elif ref.startswith("keyring:"):
            try:
                import keyring  # noqa: F401
            except ImportError:
                out.append(
                    CheckResult(
                        name=f"lanes.{tier}.remote_api_key_ref",
                        status="warn",
                        message=f"keyring URI {ref} but keyring package not installed",
                        fix="  pip install keyring",
                    )
                )
        # Inline string from env (legacy MAXIM_LANE_<TIER>_REMOTE_API_KEY)
        elif source == "env":
            out.append(
                CheckResult(
                    name=f"lanes.{tier}.remote_api_key_ref",
                    status="warn",
                    message="inline key from env (legacy — should migrate to file path)",
                    fix=(
                        f"Migrate to a mode-0600 file:\n"
                        f"  echo $MAXIM_LANE_{tier.upper()}_REMOTE_API_KEY > ~/.config/maxim/api_key\n"
                        f"  chmod 0600 ~/.config/maxim/api_key\n"
                        f"  maxim config set lanes.{tier}.remote_api_key_ref ~/.config/maxim/api_key\n"
                        f"  unset MAXIM_LANE_{tier.upper()}_REMOTE_API_KEY"
                    ),
                )
            )
    return out


def _format_value_for_set(value: object) -> str:
    """Format a resolved value for use in a ``maxim config set`` command.

    Strings with whitespace or special chars get single-quoted; other
    scalars render bare. Helpers like booleans use the canonical lower-
    case form the coercer accepts (``true`` / ``false``).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    text = str(value)
    if not text:
        return "''"
    # Single-quote anything that contains spaces, glob chars, or shell
    # metacharacters; otherwise emit bare for cleanest copy/paste.
    needs_quote = any(c in text for c in " \t\"'$\\`*?[](){}|&;<>#")
    if needs_quote:
        # Single-quote, escape embedded single quotes via the
        # POSIX ``'\''`` sequence.
        return "'" + text.replace("'", "'\\''") + "'"
    return text


def _check_legacy_env_migration(cfg: object) -> list["CheckResult"]:
    """Detect pre-config-unification env-var-heavy setups + suggest migration.

    Operators upgrading from Maxim ≤0.9.1 likely have several MAXIM_*
    env vars exported in their shell rc / launchd plist from the era
    when env vars were the only configuration surface. Each one now
    fires its own once-per-startup deprecation INFO via PR #318's I-4
    fold, but those scroll past in the log and don't tell the operator
    "here's the full migration command list."

    This check aggregates the situation:

    - Counts how many ``_ABSORBED_ENV_VARS`` are currently set
    - Below threshold (< 2): returns empty (one stray var is not the
      "pre-migration era" pattern; could be a deliberate per-process
      override)
    - At or above threshold: emits one WARN row with a copy/paste
      migration script that includes:
        1. ``maxim config set <field> <value>`` commands for env-only
           values (skips ones already convergent with config.json — no
           point re-setting what's there)
        2. ``unset MAXIM_*`` for the current shell
        3. ``grep`` commands to find persistent exports in shell rc files
        4. Platform-aware lookup of launchd plists (macOS) or systemd
           units (Linux) where env vars may be set globally

    The check is informational — it never fails. Operators on a healthy
    pre-1.1 env-heavy setup will see this row exactly once per doctor
    invocation and can clean up at their leisure.
    """
    from maxim.runtime.config_loader import (
        _ABSORBED_ENV_VARS,
        _FIELD_TO_ENV,
        _env_is_set,
        resolve_setting,
    )

    set_envs = sorted(name for name in _ABSORBED_ENV_VARS if _env_is_set(name))
    if len(set_envs) < 2:
        return []

    # Reverse lookup: env var name → field_path
    env_to_field: dict[str, str] = {v: k for k, v in _FIELD_TO_ENV.items()}

    config_set_lines: list[str] = []
    convergent_only_envs: list[str] = []

    for env_name in set_envs:
        field_path = env_to_field.get(env_name)
        if field_path is None:
            continue
        try:
            env_value, env_source = resolve_setting(field_path, config=cfg)
        except Exception:
            continue
        if env_source != "env":
            # Resolve says source isn't env even though it's set — shouldn't
            # normally happen (env wins over config), but skip defensively.
            continue
        # Check if config.json has the same value (convergent case) —
        # then we don't need maxim config set, just unset the env.
        try:
            from maxim.runtime.config_loader import _read_from_config

            config_value = _read_from_config(cfg, field_path)
        except Exception:
            config_value = None
        if config_value is not None and config_value == env_value:
            convergent_only_envs.append(env_name)
        else:
            # Don't write secrets (API keys) into the migration script in
            # plaintext — recommend using the file-ref pattern instead.
            if env_name.endswith("_REMOTE_API_KEY"):
                config_set_lines.append(f"# {env_name}: write key to a 0600 file and reference by path")
                config_set_lines.append(
                    "echo -n '<your-key>' > ~/.config/maxim/api_key && chmod 0600 ~/.config/maxim/api_key"
                )
                config_set_lines.append(
                    f"maxim config set {field_path.replace('_REMOTE_API_KEY', '_remote_api_key_ref').lower()} ~/.config/maxim/api_key"
                )
            else:
                config_set_lines.append(f"maxim config set {field_path} {_format_value_for_set(env_value)}")

    # Build the full fix script
    fix_lines: list[str] = []
    if config_set_lines:
        fix_lines.append("# 1. Persist current env values to config.json (skips ones already convergent):")
        fix_lines.extend(config_set_lines)
        fix_lines.append("")
    fix_lines.append("# 2. Unset the env vars for the current shell:")
    fix_lines.append(f"unset {' '.join(set_envs)}")
    fix_lines.append("")
    fix_lines.append("# 3. Find + remove persistent exports from shell rc files:")
    fix_lines.append(
        "grep -nE '^export MAXIM_' ~/.zshrc ~/.zprofile ~/.zshenv ~/.bashrc ~/.bash_profile /etc/profile 2>/dev/null"
    )
    fix_lines.append("# Then edit each matching file to remove the export line(s).")

    # Platform-specific persistent-env lookup
    import platform as _platform_mod

    system = _platform_mod.system().lower()
    if system == "darwin":
        fix_lines.append("")
        fix_lines.append("# 4. macOS: also check launchd plists for MAXIM_* keys:")
        fix_lines.append(
            "grep -rl 'MAXIM_' ~/Library/LaunchAgents/ /Library/LaunchAgents/ /Library/LaunchDaemons/ 2>/dev/null"
        )
        fix_lines.append("# For each matching plist, remove the <key>MAXIM_*</key><string>...</string> pair,")
        fix_lines.append("# then `launchctl unload <plist>; launchctl load <plist>` to apply.")
    elif system == "linux":
        fix_lines.append("")
        fix_lines.append("# 4. Linux: also check systemd unit files for Environment= lines:")
        fix_lines.append("grep -rnE '^Environment.*MAXIM_' /etc/systemd/ ~/.config/systemd/ 2>/dev/null")
        fix_lines.append("# For each matching unit, remove the Environment= line,")
        fix_lines.append("# then `systemctl daemon-reload && systemctl restart <unit>` to apply.")

    fix_lines.append("")
    fix_lines.append("# 5. Restart the leader to pick up the migrated config:")
    fix_lines.append("#    kill the running `maxim` and re-launch (or `maxim peer restart` from a peer)")

    # Compose the message
    convergent_note = (
        f" ({len(convergent_only_envs)} already match config.json — those just need unset)"
        if convergent_only_envs
        else ""
    )
    message = (
        f"detected {len(set_envs)} MAXIM_* env vars from pre-0.10 configuration"
        f"{convergent_note}. Migrate to config.json for cleaner setup "
        f"(env vars deprecated in 1.1)."
    )

    return [
        CheckResult(
            name="legacy_env_migration",
            status="warn",
            message=message,
            fix="\n".join(fix_lines),
        )
    ]


def _check_peer_yml_deprecation() -> list["CheckResult"]:
    """C4 IM5 fold: surface peer.yml presence as a deprecated compat
    signal so the operator sees the migration path."""
    try:
        from maxim.peer.config import peer_config_path
    except Exception:
        return []
    path = peer_config_path()
    try:
        if not path.is_file():
            return []
    except OSError:
        return []
    return [
        CheckResult(
            name="peer.yml",
            status="warn",
            message=f"{path} present (deprecated as of 1.0, retired in 2.0)",
            fix=(
                "If config.json::lanes.large.* is set, peer.yml is now redundant.\n"
                "  - To migrate: `maxim peer connect <url>` rewrites both files,\n"
                "    or `maxim peer forget` removes both.\n"
                "  - To check sources: `maxim config list`."
            ),
        )
    ]


def check_env_config(info: PlatformInfo, role: str | None = None) -> list["CheckResult"]:
    """Validate critical Maxim environment variables.

    Returns a list (may be empty if everything is fine) so the caller can
    splice the results into the Environment section without a fixed slot.
    Checks are cross-platform — only uses ``os.environ``, no shell calls.

    Covers:
    - MAXIM_ROLE missing or set to an invalid value
    - MAXIM_LLM_ENABLED not set on a leader/solo machine
    - MAXIM_LLM_PROFILE missing when LLM is enabled
    - MAXIM_LLM_N_CTX not set (context overflow risk on 14B+ models)
    - Stale MAXIM_PEER_PROBE_KEY from pre-R2.5 (now ignored, misleading)
    - MAXIM_SKIP_REMOTE_PROBE left set from debugging (silently disables health probes)
    - MAXIM_ROLE=peer set on a machine running as leader (role_divergence trigger)
    """
    import os

    results: list[CheckResult] = []
    is_peer = (role == "peer") or (os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL", "").strip() != "")

    # ── MAXIM_ROLE ────────────────────────────────────────────────────────────
    maxim_role = os.environ.get("MAXIM_ROLE", "").strip().lower()
    valid_roles = {"leader", "peer", "solo"}
    if not maxim_role:
        # Role is inferred from heuristics — warn so the user knows
        # which heuristic fired and can make it explicit.
        inferred = "peer" if is_peer else "leader"
        # MAXIM_ROLE is normally exported by detect_and_apply_role() at startup —
        # this branch fires only if that path was bypassed (e.g. direct import).
        # Don't suggest .zshrc for peer (auto-detected from peer.yml); for leader
        # the systemd unit Environment= is the right persistent location.
        if info.os == "macos":
            export_cmd = f"export MAXIM_ROLE={inferred}  # current session; auto-detected on normal startup"
        else:
            export_cmd = f"export MAXIM_ROLE={inferred}  # or set in systemd unit Environment="
        results.append(
            CheckResult(
                name="MAXIM_ROLE",
                status="warn",
                message=f"not set — role inferred as '{inferred}' from heuristics. Normally auto-exported at startup; check that cli.py::main ran before this check.",
                fix=export_cmd,
            )
        )
    elif maxim_role not in valid_roles:
        results.append(
            CheckResult(
                name="MAXIM_ROLE",
                status="fail",
                message=f"invalid value '{maxim_role}' — must be leader, peer, or solo",
                fix="export MAXIM_ROLE=leader  # or peer / solo",
            )
        )
    else:
        results.append(
            CheckResult(
                name="MAXIM_ROLE",
                status="ok",
                message=f"MAXIM_ROLE={maxim_role}",
            )
        )

    # ── LLM enablement (leader/solo only) ────────────────────────────────────
    #
    # Post-implementation field-coverage fold (2026-06-02): the three
    # llm.* checks below now route through ``resolve_setting`` so
    # config.json values silence the warnings the same way env values
    # do. Pre-fold a leader with config.json::llm.profile=qwen-32b
    # still saw "MAXIM_LLM_PROFILE: not set" — silent drift from C5's
    # "Resolved Config" section. The status string includes the
    # resolved source so the operator sees the precedence chain.
    if not is_peer:
        try:
            from maxim.runtime.config_loader import resolve_setting as _rs
        except Exception:
            _rs = None

        # ── llm.enabled ──────────────────────────────────────────────────────
        if _rs is not None:
            try:
                llm_enabled_value, _ = _rs("llm.enabled")
            except Exception:
                llm_enabled_value = None
        else:
            llm_enabled_value = None
        if not llm_enabled_value:
            results.append(
                CheckResult(
                    name="llm.enabled",
                    status="warn",
                    message="not enabled — LLM inference will be skipped",
                    fix="maxim config set llm.enabled true",
                )
            )

        # ── llm.profile ──────────────────────────────────────────────────────
        if _rs is not None:
            try:
                profile_value, profile_source = _rs("llm.profile")
            except Exception:
                profile_value, profile_source = None, "default"
        else:
            profile_value, profile_source = None, "default"
        if profile_value and profile_source in ("config", "env", "cli"):
            results.append(
                CheckResult(
                    name="llm.profile",
                    status="ok",
                    message=f"{profile_value}  [source={profile_source}]",
                )
            )
        else:
            # Fallback: check persisted-model file (legacy state)
            try:
                from maxim.runtime.lane_backends import _read_persisted_model

                persisted = _read_persisted_model()
            except Exception:
                persisted = None
            if persisted:
                results.append(
                    CheckResult(
                        name="llm.profile",
                        status="info",
                        message=f"not set — using persisted model '{persisted}'",
                        fix=f"maxim config set llm.profile {persisted}  # makes it explicit",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name="llm.profile",
                        status="warn",
                        message="not set and no persisted model — LLM will not start",
                        fix="maxim config set llm.profile qwen2.5-14b  # or your model name",
                    )
                )

        # ── llm.n_ctx ────────────────────────────────────────────────────────
        if _rs is not None:
            try:
                n_ctx_value, n_ctx_source = _rs("llm.n_ctx")
            except Exception:
                n_ctx_value, n_ctx_source = None, "default"
        else:
            n_ctx_value, n_ctx_source = None, "default"
        if n_ctx_source == "default":
            # Default (8192) — surface a soft INFO so operators on 14B+
            # know to raise it explicitly, but not a WARN since the
            # default is sensible for sub-14B.
            results.append(
                CheckResult(
                    name="llm.n_ctx",
                    status="info",
                    message=(
                        f"using default ({n_ctx_value}). Long prompts on 14B+ "
                        "models risk KV-cache overflow; consider raising to 16384."
                    ),
                    fix="maxim config set llm.n_ctx 16384",
                )
            )
        elif isinstance(n_ctx_value, int):
            if n_ctx_value < 8192:
                results.append(
                    CheckResult(
                        name="llm.n_ctx",
                        status="warn",
                        message=f"{n_ctx_value} [source={n_ctx_source}] — below 8192 risks context overflow on multi-turn sims",
                        fix="maxim config set llm.n_ctx 16384",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name="llm.n_ctx",
                        status="ok",
                        message=f"{n_ctx_value}  [source={n_ctx_source}]",
                    )
                )
        else:
            results.append(
                CheckResult(
                    name="llm.n_ctx",
                    status="fail",
                    message=f"{n_ctx_value!r} is not a valid integer [source={n_ctx_source}]",
                    fix="maxim config set llm.n_ctx 16384",
                )
            )

    # ── Stale debugging vars that cause silent failures ───────────────────────
    if os.environ.get("MAXIM_SKIP_REMOTE_PROBE", "").strip().lower() in ("1", "true", "yes"):
        results.append(
            CheckResult(
                name="MAXIM_SKIP_REMOTE_PROBE",
                status="warn",
                message="set — all health probes are bypassed. Inference will fire before the server is ready.",
                fix="unset MAXIM_SKIP_REMOTE_PROBE",
            )
        )

    if os.environ.get("MAXIM_PEER_PROBE_KEY", "").strip():
        results.append(
            CheckResult(
                name="MAXIM_PEER_PROBE_KEY",
                status="warn",
                message="set — this variable was removed in Plan 3 R2.5 (probe key moved to instance level). It is ignored but may indicate a stale environment.",
                fix="unset MAXIM_PEER_PROBE_KEY",
            )
        )

    return results


# ─── context window ─────────────────────────────────────────────────────────


def check_context_window(port: int = 8100) -> CheckResult:
    """Detect the context window actually in use by the running llama-cpp server.

    Queries ``/v1/models`` first (cheap). If that doesn't carry n_ctx, falls
    back to inspecting the process command-line on Linux (``/proc/*/cmdline``)
    and macOS/Windows (``ps``/``wmic``). Returns info if server isn't running.

    This check exists specifically to catch the silent inference_broken cascade
    where llama-cpp runs at n_ctx=4096 and long prompts overflow the KV cache,
    returning empty ``choices`` with HTTP 200.
    """
    import platform
    import subprocess

    url = f"http://127.0.0.1:{port}/v1"

    # Step 1: is the server up at all?
    try:
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

        backend = _MaximPeerBackend.for_url(url)
        healthy = backend.health_check(enable_stage2=False)
        if not healthy:
            return CheckResult(
                name="Context window (n_ctx)",
                status="info",
                message="llama-cpp server not reachable — start it first",
            )
    except Exception:
        return CheckResult(
            name="Context window (n_ctx)",
            status="info",
            message="context window check unavailable (server not reachable)",
        )

    # Step 2: check /v1/models for n_ctx in metadata
    n_ctx: int | None = None
    try:
        import socket

        req_bytes = (f"GET /v1/models HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n").encode()
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
            s.sendall(req_bytes)
            raw = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                raw += chunk
        body = raw.split(b"\r\n\r\n", 1)[-1]
        import json as _json

        data = _json.loads(body)
        for model in data.get("data", []):
            ctx = model.get("context_length") or model.get("n_ctx") or model.get("max_context_length")
            if ctx:
                n_ctx = int(ctx)
                break
    except Exception:
        pass

    # Step 3: inspect process args (cross-platform, best-effort)
    if n_ctx is None:
        try:
            system = platform.system().lower()
            if system == "linux":
                # /proc is authoritative and doesn't require ps
                import glob

                for cmdline_path in glob.glob("/proc/*/cmdline"):
                    try:
                        with open(cmdline_path, "rb") as f:
                            args = f.read().split(b"\x00")
                        args_str = [a.decode("utf-8", errors="ignore") for a in args]
                        if any("llama" in a or "llama-server" in a for a in args_str):
                            for i, arg in enumerate(args_str):
                                if arg in ("--ctx-size", "-c", "--n-ctx") and i + 1 < len(args_str):
                                    n_ctx = int(args_str[i + 1])
                                    break
                                if arg.startswith("--ctx-size="):
                                    n_ctx = int(arg.split("=", 1)[1])
                                    break
                    except Exception:
                        continue
                    if n_ctx is not None:
                        break
            elif system == "darwin":
                out = subprocess.check_output(
                    ["ps", "aux"],
                    timeout=3.0,
                    stderr=subprocess.DEVNULL,
                ).decode("utf-8", errors="ignore")
                for line in out.splitlines():
                    if "llama" not in line:
                        continue
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p in ("--ctx-size", "-c", "--n-ctx") and i + 1 < len(parts):
                            n_ctx = int(parts[i + 1])
                            break
                        if p.startswith("--ctx-size="):
                            n_ctx = int(p.split("=", 1)[1])
                            break
                    if n_ctx is not None:
                        break
        except Exception:
            pass

    if n_ctx is None:
        # Post-implementation field-coverage fold: surface the
        # CONFIGURED n_ctx from resolve_setting when the running
        # server's introspection failed. Pre-fold the operator only
        # saw "could not be determined" with no way to confirm what
        # config.json had asked for.
        configured_n_ctx, configured_source = None, "default"
        try:
            from maxim.runtime.config_loader import resolve_setting as _rs

            configured_n_ctx, configured_source = _rs("llm.n_ctx")
        except Exception:
            pass
        if configured_n_ctx and configured_source in ("config", "env", "cli"):
            return CheckResult(
                name="Context window (n_ctx)",
                status="info",
                message=(
                    f"server running but did not expose n_ctx; configured value "
                    f"{configured_n_ctx} [source={configured_source}] should be in effect."
                ),
            )
        return CheckResult(
            name="Context window (n_ctx)",
            status="warn",
            message=(
                "server is running but n_ctx could not be determined and no config value set. "
                "Set llm.n_ctx via `maxim config set llm.n_ctx 16384` to make it explicit."
            ),
            fix="maxim config set llm.n_ctx 16384",
        )

    if n_ctx < 8192:
        return CheckResult(
            name="Context window (n_ctx)",
            status="warn",
            message=(
                f"n_ctx={n_ctx} — too small for multi-turn sims with memory summaries. "
                f"Long prompts will overflow the KV cache and return empty choices "
                f"(inference_broken cascade)."
            ),
            fix="maxim config set llm.n_ctx 16384  # then restart: maxim peer restart",
            retry_id="ctx_window",
        )

    return CheckResult(
        name="Context window (n_ctx)",
        status="ok",
        message=f"n_ctx={n_ctx}",
    )


# ─── VRAM spillover (Plan 3.6 R5) ──────────────────────────────────────────


def _current_llama_server_n_ctx(port: int) -> int | None:
    """Return the n_ctx the running llama-cpp-server is configured for, or None.

    Delegates to the canonical implementation in ``leader_proxy`` -- both
    this module and ``_handle_debug_vram`` share the same probe logic.
    """
    from maxim.runtime.leader_proxy import (
        _current_llama_server_n_ctx as _probe_n_ctx,
    )

    return _probe_n_ctx(port)


def check_vram_pressure(port: int = 8100) -> CheckResult:
    """Detect KV-cache spillover risk — the 2026-04-13 ~125s-latency class bug.

    Two signals:

    1. **Live ratio** from ``nvidia-smi``: ``vram_used / vram_total > 0.95``
       means the UMA driver is already spilling KV pages into shared system
       memory, and inference will run at PCIe speeds (5-30x slower than
       on-GPU). This is the primary signal — it catches the state the user
       actually hit on 2026-04-13.
    2. **Predictive KV-math**: given the currently loaded profile's ``arch``
       + the running server's ``n_ctx``, project weights + KV cache +
       headroom against physical VRAM. Catches misconfiguration BEFORE the
       operator notices a slowdown.

    Either signal alone produces a WARN. Both together → FAIL with a specific
    ``MAXIM_LLM_N_CTX=<recommended>`` fix string, interpolating real numbers
    (not ``<your-vram>`` placeholders).
    """
    # Lazy imports — keep the check fast on systems without a GPU.
    try:
        from maxim.runtime.leader_proxy import _query_nvidia_smi
    except Exception:
        return CheckResult(
            name="VRAM pressure",
            status="info",
            message="VRAM pressure check unavailable (leader_proxy import failed)",
        )

    gpu = _query_nvidia_smi()
    if gpu is None:
        return CheckResult(
            name="VRAM pressure",
            status="info",
            message="nvidia-smi unavailable — skipping VRAM pressure check",
        )

    vram_used_gb = float(gpu.get("vram_used_gb") or 0.0)
    vram_total_gb = float(gpu.get("vram_total_gb") or 0.0)
    if vram_total_gb <= 0:
        return CheckResult(
            name="VRAM pressure",
            status="info",
            message="nvidia-smi reported zero total VRAM — skipping",
        )

    ratio = vram_used_gb / vram_total_gb

    # Resolve active profile (if any) so we can do the predictive projection.
    # Must read through the llm_server module reference — `_active_model` is
    # a mutable global and `from lane_backends import _active_model` binds by
    # value (see CLAUDE.md "Mutable globals + module extraction" lesson).
    active_profile: str | None = None
    try:
        import maxim.runtime.llm_server as _server_mod

        active_profile = _server_mod._active_model
    except Exception:
        active_profile = None

    projection = None
    running_n_ctx: int | None = None
    if active_profile:
        running_n_ctx = _current_llama_server_n_ctx(port)
        if running_n_ctx and running_n_ctx > 0:
            try:
                from maxim.models.language.config import _BUILTIN_PROFILES
                from maxim.runtime.lane_models import project_vram_usage

                profile_meta = _BUILTIN_PROFILES.get(active_profile, {})
                projection = project_vram_usage(
                    active_profile,
                    profile_meta,
                    running_n_ctx,
                    vram_total_gb,
                )
            except Exception:
                projection = None

    # Import the named thresholds from lane_models rather than re-declaring
    # literals here — single source of truth for spawn-time projection AND
    # the live nvidia-smi ratio check. Drift between the two values is
    # exactly the "future bug" pattern flagged in Plan 3.6 R5 review.
    from maxim.runtime.lane_models import _SPILLOVER_RATIO, _SPILLOVER_WARN_RATIO

    live_spillover = ratio > _SPILLOVER_RATIO
    live_warning = ratio > _SPILLOVER_WARN_RATIO

    # Build a fix hint that names REAL values — profile, n_ctx, VRAM. The
    # operator should be able to copy-paste it.
    def _build_fix(rec_n_ctx: int | None) -> str:
        lines: list[str] = []
        if rec_n_ctx and rec_n_ctx > 0:
            lines.append(f"export MAXIM_LLM_N_CTX={rec_n_ctx}  # then: maxim peer restart")
        else:
            lines.append("export MAXIM_LLM_N_CTX=4096  # then: maxim peer restart")
        if active_profile and active_profile.startswith("qwen2.5-14b"):
            lines.append("# or switch to a smaller profile: maxim peer llm qwen2-7b-instruct")
        else:
            lines.append("# or switch to a smaller profile via `maxim peer llm <model>`")
        lines.append("# or close other GPU consumers (browser hw accel, extra models)")
        return "\n".join(lines)

    # Case 1: live spillover confirmed — worst case. Inference is already slow.
    if live_spillover:
        msg = (
            f"VRAM {vram_used_gb:.1f}/{vram_total_gb:.0f} GB "
            f"({ratio * 100:.0f}% used) — KV cache likely spilled to shared GPU "
            f"memory. Expect 5-30x slowdown (PCIe-bound inference)."
        )
        rec = projection.recommended_n_ctx if projection else None
        return CheckResult(
            name="VRAM pressure",
            status="fail",
            message=msg,
            fix=_build_fix(rec),
            retry_id="vram_pressure",
        )

    # Case 2: predictive projection says we'll spill even if we're currently fine.
    # Can happen right after spawn before KV cache fills.
    if projection is not None and projection.spillover_risk:
        msg = (
            f"Projected {projection.projected_total_gb:.1f} GB "
            f"(weights {projection.weights_gb:.1f} + KV {projection.kv_cache_gb:.1f} "
            f"+ {projection.headroom_gb:.1f} headroom) exceeds "
            f"{_SPILLOVER_RATIO * 100:.0f}% of "
            f"{projection.physical_vram_gb:.0f} GB VRAM at n_ctx={projection.n_ctx}. "
            f"Will spill to shared memory once KV cache fills."
        )
        return CheckResult(
            name="VRAM pressure",
            status="fail",
            message=msg,
            fix=_build_fix(projection.recommended_n_ctx),
            retry_id="vram_pressure",
        )

    # Case 3: in the warning band — no headroom for KV growth.
    if live_warning:
        return CheckResult(
            name="VRAM pressure",
            status="warn",
            message=(
                f"VRAM {vram_used_gb:.1f}/{vram_total_gb:.0f} GB "
                f"({ratio * 100:.0f}% used) — no headroom for KV cache growth. "
                f"Long prompts may spill."
            ),
            fix=_build_fix(projection.recommended_n_ctx if projection else None),
            retry_id="vram_pressure",
        )

    # All clear.
    suffix = f", n_ctx={running_n_ctx}" if running_n_ctx else ""
    return CheckResult(
        name="VRAM pressure",
        status="ok",
        message=f"VRAM {vram_used_gb:.1f}/{vram_total_gb:.0f} GB ({ratio * 100:.0f}% used{suffix})",
    )


# ─── LAN access ────────────────────────────────────────────────────────────


def check_lan_access(info: PlatformInfo, port: int = 8100) -> CheckResult:
    """Platform-specific LAN-access guidance for exposing the local server.

    Doesn't test reachability from an external host (can't, from here). Just
    prints what the user needs to do to make the server LAN-reachable, with
    their actual IPs filled in.
    """
    from maxim.runtime.leader_mode import detect_role

    # Use resolved role (respects both MAXIM_ROLE env var and implicit
    # cloudflared-config detection), not just the raw env var.
    leader = detect_role().is_leader

    if info.runtime == "wsl2":
        wsl_ip = detect_wsl_ip() or "<wsl-ip>"
        host_ip = info.windows_host_ip or "<windows-host-ip>"
        fix = (
            f"WSL2 peers can't reach http://127.0.0.1:{port} directly — the\n"
            f"  Windows host needs to forward the port.\n\n"
            f"  1. In PowerShell on Windows (as admin):\n"
            f"       netsh interface portproxy add v4tov4 listenport={port} listenaddress=0.0.0.0 connectport={port} connectaddress={wsl_ip}\n"
            f'       netsh advfirewall firewall add rule name="Maxim LLM" dir=in action=allow protocol=TCP localport={port}\n\n'
            f"  2. Run Maxim as leader:\n"
            f"       MAXIM_ROLE=leader maxim\n\n"
            f"  3. Peers connect via your Windows LAN IP:\n"
            f"       http://{host_ip}:{port}/v1\n"
        )
        return CheckResult(
            name="LAN access (WSL2)",
            status="warn" if not leader else "ok",
            message=(
                "WSL2 port-forwarding required for LAN peers"
                if not leader
                else f"leader mode + port-forwarded at {host_ip}:{port}"
            ),
            fix=fix if not leader else None,
        )

    if info.runtime in ("wsl1", "docker"):
        return CheckResult(
            name=f"LAN access ({info.runtime})",
            status="warn",
            message=f"LAN access untested on {info.runtime}",
            fix=(
                f"{info.runtime} has unusual networking. Consider switching to WSL2\n"
                f"  (mirrored networking mode) or running Maxim in the native host."
            ),
        )

    if info.os == "linux":
        from maxim.doctor.platform_detect import detect_lan_ip

        lan_ip = detect_lan_ip() or "<your-lan-ip>"
        firewall_hint = {
            "ubuntu": f"sudo ufw allow {port}/tcp",
            "debian": f"sudo ufw allow {port}/tcp",
            "fedora": f"sudo firewall-cmd --add-port={port}/tcp --permanent && sudo firewall-cmd --reload",
            "rhel": f"sudo firewall-cmd --add-port={port}/tcp --permanent && sudo firewall-cmd --reload",
            "arch": f"sudo ufw allow {port}/tcp  # (or your preferred firewall)",
        }.get(info.distro, f"Open TCP port {port} in your firewall")
        fix = (
            f"Linux peers reach you directly via your LAN IP:\n"
            f"  1. Run Maxim as leader:\n"
            f"       MAXIM_ROLE=leader maxim\n\n"
            f"  2. Allow port {port} in your firewall (if active):\n"
            f"       {firewall_hint}\n\n"
            f"  3. Peers connect:\n"
            f"       http://{lan_ip}:{port}/v1\n"
        )
        return CheckResult(
            name="LAN access (Linux)",
            status="warn" if not leader else "ok",
            message="native Linux — peers reach LAN IP directly",
            fix=fix if not leader else None,
        )

    if info.os == "macos":
        from maxim.doctor.platform_detect import detect_lan_ip

        lan_ip = detect_lan_ip() or "<your-lan-ip>"
        fix = (
            f"macOS peers reach you directly via your LAN IP:\n"
            f"  1. Run Maxim as leader:\n"
            f"       MAXIM_ROLE=leader maxim\n\n"
            f"  2. If macOS firewall is enabled, allow Python in:\n"
            f"       System Settings → Network → Firewall → Options\n\n"
            f"  3. Peers connect:\n"
            f"       http://{lan_ip}:{port}/v1\n"
        )
        return CheckResult(
            name="LAN access (macOS)",
            status="warn" if not leader else "ok",
            message="native macOS — peers reach LAN IP directly",
            fix=fix if not leader else None,
        )

    if info.os == "windows":
        fix = (
            f"Windows native:\n"
            f"  1. In PowerShell as admin, allow inbound:\n"
            f'       New-NetFirewallRule -DisplayName "Maxim LLM" -Direction Inbound -Protocol TCP -LocalPort {port} -Action Allow\n\n'
            f"  2. Run Maxim as leader:\n"
            f"       $env:MAXIM_ROLE='leader'\n"
            f"       maxim\n"
        )
        return CheckResult(
            name="LAN access (Windows)",
            status="warn" if not leader else "ok",
            message="native Windows",
            fix=fix if not leader else None,
        )

    return CheckResult(
        name="LAN access",
        status="warn",
        message="unknown platform",
    )


# ─── cloudflared / tunnel ──────────────────────────────────────────────────


def check_cloudflared(info: PlatformInfo) -> CheckResult:
    from maxim.tunnel.cloudflared import cloudflared_version, find_cloudflared

    path = find_cloudflared()
    if path is None:
        install = {
            (
                "linux",
                "ubuntu",
            ): "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb\n"
            "  sudo dpkg -i cloudflared.deb",
            (
                "linux",
                "debian",
            ): "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb\n"
            "  sudo dpkg -i cloudflared.deb",
            (
                "linux",
                "fedora",
            ): "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm -o cloudflared.rpm\n"
            "  sudo rpm -i cloudflared.rpm",
            (
                "linux",
                "rhel",
            ): "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm -o cloudflared.rpm\n"
            "  sudo rpm -i cloudflared.rpm",
            ("linux", "arch"): "yay -S cloudflared  # or your preferred AUR helper",
            ("macos", "unknown"): "brew install cloudflare/cloudflare/cloudflared",
            (
                "windows",
                "unknown",
            ): "Download https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe\n"
            "  (add to PATH)",
        }
        key = (info.os, info.distro if info.os == "linux" else "unknown")
        install_cmd = install.get(
            key,
            "See https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/",
        )
        return CheckResult(
            name="cloudflared",
            status="warn",
            message="not installed — required for external tunnel access",
            fix=f"Install:\n  {install_cmd}\n  Then run: maxim tunnel setup",
            retry_id="cloudflared",
        )
    version = cloudflared_version() or "(unknown version)"
    return CheckResult(
        name="cloudflared",
        status="ok",
        message=f"{path} — {version}",
    )


def check_tunnel_config() -> CheckResult:
    from maxim.tunnel.config import CONFIG_PATH, read_config_summary

    summary = read_config_summary()
    if summary is None:
        return CheckResult(
            name="Tunnel config",
            status="warn",
            message=f"no config at {CONFIG_PATH}",
            fix="Run: maxim tunnel setup",
            retry_id="tunnel-config",
        )
    hostname = summary.get("hostname", "?")
    service = summary.get("service", "")

    # Warn if tunnel points directly at llama-cpp-server (8100) instead of
    # LeaderProxy (8099). Without the proxy, peers bypass auth enforcement,
    # logging, GPU metrics, admission control, and admin endpoints.
    if ":8100" in service:
        return CheckResult(
            name="Tunnel config",
            status="warn",
            message=f"hostname={hostname} — tunnel points at port 8100 (llama-cpp-server directly)",
            fix=(
                "Update your tunnel config to route through the LeaderProxy (port 8099):\n"
                f"  Edit {CONFIG_PATH}\n"
                "  Change: service: http://localhost:8100\n"
                "  To:     service: http://localhost:8099\n"
                "  Then restart cloudflared:\n"
                "    sudo systemctl restart cloudflared   # Linux\n"
                "    cloudflared --config ~/.cloudflared/config.yml tunnel run   # manual"
            ),
        )

    return CheckResult(
        name="Tunnel config",
        status="ok",
        message=f"hostname={hostname} service={service}",
    )


def check_tunnel_config_sync() -> CheckResult:
    """Warn if the systemd cloudflared config differs from the user config.

    When cloudflared is installed as a systemd service, it reads from
    /etc/cloudflared/config.yml — NOT ~/.cloudflared/config.yml. After
    editing or regenerating the user config, the systemd copy must be
    updated too, or the tunnel will use stale settings.
    """
    import platform

    if platform.system() != "Linux":
        return CheckResult(
            name="Tunnel config sync",
            status="ok",
            message="skipped (non-Linux)",
        )

    from pathlib import Path

    user_cfg = Path.home() / ".cloudflared" / "config.yml"
    system_cfg = Path("/etc/cloudflared/config.yml")

    if not user_cfg.is_file():
        return CheckResult(
            name="Tunnel config sync",
            status="ok",
            message="no user config — nothing to compare",
        )
    if not system_cfg.is_file():
        return CheckResult(
            name="Tunnel config sync",
            status="ok",
            message="no systemd config — cloudflared may not be a service",
        )

    try:
        user_text = user_cfg.read_text()
        system_text = system_cfg.read_text()
    except PermissionError:
        return CheckResult(
            name="Tunnel config sync",
            status="warn",
            message="cannot read /etc/cloudflared/config.yml (permission denied)",
            fix="Run: sudo cat /etc/cloudflared/config.yml",
        )

    if user_text.strip() == system_text.strip():
        return CheckResult(
            name="Tunnel config sync",
            status="ok",
            message="~/.cloudflared/config.yml and /etc/cloudflared/config.yml are in sync",
        )

    return CheckResult(
        name="Tunnel config sync",
        status="warn",
        message=(
            "~/.cloudflared/config.yml and /etc/cloudflared/config.yml DIFFER — "
            "systemd service may be using stale settings"
        ),
        fix=(
            "The systemd cloudflared service reads /etc/cloudflared/config.yml,\n"
            "not ~/.cloudflared/config.yml. After editing the user config, sync it:\n"
            "  sudo cp ~/.cloudflared/config.yml /etc/cloudflared/config.yml\n"
            "  sudo systemctl restart cloudflared"
        ),
        retry_id="tunnel-config-sync",
    )


# ─── API key ───────────────────────────────────────────────────────────────


def check_api_key() -> CheckResult:
    from maxim.tunnel.keys import key_exists, key_file_path, truncate_for_display, read_key

    if not key_exists():
        return CheckResult(
            name="API key",
            status="warn",
            message="no key set — server accepts all requests (localhost is fine)",
            fix=(
                "Generate one before exposing to LAN/tunnel:\n"
                "  maxim tunnel key rotate\n"
                "  maxim tunnel key export   # shell snippets for peers"
            ),
        )
    key = read_key() or ""
    return CheckResult(
        name="API key",
        status="ok",
        message=f"{truncate_for_display(key)} at {key_file_path()}",
    )


def check_key_age() -> CheckResult:
    """Warn when the API key file is older than 90 days."""
    import time

    from maxim.tunnel.keys import key_exists, key_file_path

    if not key_exists():
        return CheckResult(
            name="Key age",
            status="info",
            message="no key file — skipped",
        )
    path = key_file_path()
    try:
        age_days = (time.time() - path.stat().st_mtime) / 86400
    except OSError:
        return CheckResult(
            name="Key age",
            status="warn",
            message=f"cannot stat {path}",
        )
    if age_days > 90:
        return CheckResult(
            name="Key age",
            status="warn",
            message=f"key is {age_days:.0f} days old — consider rotating",
            fix="maxim tunnel key rotate && maxim tunnel key export",
            retry_id="key-age",
        )
    return CheckResult(
        name="Key age",
        status="ok",
        message=f"key is {age_days:.0f} days old",
    )


def check_key_permissions() -> CheckResult:
    """Fail if the key file is world-readable (POSIX only)."""
    import platform
    import stat

    if platform.system() == "Windows":
        return CheckResult(
            name="Key permissions",
            status="ok",
            message="skipped (Windows uses ACLs)",
        )

    from maxim.tunnel.keys import key_exists, key_file_path

    if not key_exists():
        return CheckResult(
            name="Key permissions",
            status="info",
            message="no key file — skipped",
        )
    path = key_file_path()
    try:
        mode = path.stat().st_mode
    except OSError:
        return CheckResult(
            name="Key permissions",
            status="warn",
            message=f"cannot stat {path}",
        )
    # Check group/other read bits
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        return CheckResult(
            name="Key permissions",
            status="fail",
            message=f"key file is world-readable (mode {oct(mode & 0o777)})",
            fix=f"chmod 600 {path}",
            retry_id="key-permissions",
        )
    return CheckResult(
        name="Key permissions",
        status="ok",
        message=f"mode {oct(mode & 0o777)}",
    )


def check_key_auth_smoke(port: int = 8100) -> CheckResult:
    """Verify the local server enforces the configured API key."""
    from maxim.tunnel.keys import key_exists, read_key
    from maxim.utils import http as _http

    if not key_exists():
        return CheckResult(
            name="Key auth smoke",
            status="info",
            message="no key configured — skipped",
        )
    key = read_key() or ""
    url = f"http://127.0.0.1:{port}/v1/models"
    fast = _http.TimeoutPolicy(connect_s=1.0, read_s=3.0, total_s=4.0)
    # Test with correct key
    try:
        _http.fetch_url(
            url,
            method="GET",
            headers={"Authorization": f"Bearer {key}"},
            timeout=fast,
        )
    except _http.HTTPAuthError:
        return CheckResult(
            name="Key auth smoke",
            status="fail",
            message="server rejected our key (401)",
            fix="Key mismatch — regenerate: maxim tunnel key rotate",
            retry_id="key-auth",
        )
    except _http.HTTPError as e:
        return CheckResult(
            name="Key auth smoke",
            status="warn",
            message=f"server returned HTTP {e.status}",
        )
    except Exception:
        return CheckResult(
            name="Key auth smoke",
            status="info",
            message="server not reachable — skipped",
        )
    # Test with bogus key — should get 401 if auth is enforced
    try:
        _http.fetch_url(
            url,
            method="GET",
            headers={"Authorization": "Bearer BOGUS"},
            timeout=fast,
        )
        return CheckResult(
            name="Key auth smoke",
            status="warn",
            message="server accepts ANY key — auth not enforced",
            fix=(
                "The server accepts requests without verifying the API key.\n"
                "  Fine for localhost, but insecure if exposed via tunnel.\n"
                "  Route through LeaderProxy (port 8099) to enforce auth."
            ),
        )
    except _http.HTTPAuthError:
        return CheckResult(
            name="Key auth smoke",
            status="ok",
            message="auth enforced — valid key accepted, bogus key rejected",
        )
    except _http.HTTPError as e:
        return CheckResult(
            name="Key auth smoke",
            status="warn",
            message=f"bogus-key probe returned HTTP {e.status} (expected 401)",
        )
    except Exception:
        return CheckResult(
            name="Key auth smoke",
            status="ok",
            message="valid key accepted (bogus-key probe inconclusive)",
        )


# ─── disk + memory ────────────────────────────────────────────────────────


def check_disk_space() -> CheckResult:
    """Warn when free disk space is low."""
    import shutil
    from pathlib import Path

    from maxim.utils.paths import data_home

    try:
        check_path = data_home()
    except Exception:
        check_path = Path.home()
    try:
        usage = shutil.disk_usage(check_path)
    except OSError as e:
        return CheckResult(
            name="Disk space",
            status="warn",
            message=f"cannot check: {e}",
        )
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    if free_gb < 2:
        return CheckResult(
            name="Disk space",
            status="fail",
            message=f"{free_gb:.1f} GB free of {total_gb:.0f} GB — critically low",
            fix="Free disk space. Sim reports, model files, and logs consume storage over time.",
        )
    if free_gb < 10:
        return CheckResult(
            name="Disk space",
            status="warn",
            message=f"{free_gb:.1f} GB free of {total_gb:.0f} GB — model downloads may fail",
        )
    return CheckResult(
        name="Disk space",
        status="ok",
        message=f"{free_gb:.1f} GB free of {total_gb:.0f} GB",
    )


def check_ram_headroom() -> CheckResult:
    """Report available system RAM."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
    except ImportError:
        # Fallback: read /proc/meminfo on Linux, sysctl on macOS
        import platform

        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    info = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            info[parts[0].rstrip(":")] = int(parts[1])
                avail_gb = info.get("MemAvailable", 0) / (1024**2)
                total_gb = info.get("MemTotal", 0) / (1024**2)
            except Exception:
                return CheckResult(name="RAM", status="info", message="cannot read /proc/meminfo")
        elif platform.system() == "Darwin":
            import subprocess

            try:
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=5, text=True)
                total_gb = int(out.strip()) / (1024**3)
                # macOS doesn't expose "available" easily without psutil
                avail_gb = -1
            except Exception:
                return CheckResult(name="RAM", status="info", message="cannot read sysctl")
        else:
            return CheckResult(
                name="RAM",
                status="info",
                message="install psutil for RAM check: pip install psutil",
            )

    if avail_gb < 0:
        return CheckResult(
            name="RAM",
            status="ok",
            message=f"{total_gb:.1f} GB total (install psutil for available RAM)",
        )
    if avail_gb < 2:
        return CheckResult(
            name="RAM",
            status="warn",
            message=f"{avail_gb:.1f} GB available of {total_gb:.1f} GB — tight for LLM inference",
            fix="Close other applications or choose a smaller model (e.g., smollm-1.7b)",
        )
    return CheckResult(
        name="RAM",
        status="ok",
        message=f"{avail_gb:.1f} GB available of {total_gb:.1f} GB",
    )


# ─── inference coherence ──────────────────────────────────────────────────


def check_inference_coherence(port: int = 8100) -> CheckResult:
    """Send a fixed prompt and verify the LLM returns a sensible answer."""
    import time

    from maxim.utils import http as _http

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    try:
        t0 = time.time()
        resp = _http.fetch_url(
            url,
            method="POST",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=30.0, total_s=32.0),
        )
        data = resp.json()
        dt_ms = (time.time() - t0) * 1000
    except Exception:
        return CheckResult(
            name="Inference coherence",
            status="info",
            message="server not reachable — skipped",
        )

    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return CheckResult(
            name="Inference coherence",
            status="fail",
            message="response has unexpected structure",
        )

    # Check if response contains "4" somewhere
    if "4" in text:
        return CheckResult(
            name="Inference coherence",
            status="ok",
            message=f"correct ({dt_ms:.0f} ms): {text!r}",
        )
    return CheckResult(
        name="Inference coherence",
        status="warn",
        message=f"unexpected answer ({dt_ms:.0f} ms): {text!r} — model may be misconfigured",
    )


# ─── peer-mode checks ────────────────────────────────────────────────────


def check_peer_url_reachable(url: str) -> CheckResult:
    """Resolve DNS and probe /v1/models on a remote leader."""
    import socket
    import time
    import urllib.parse

    from maxim.utils import http as _http

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if not host:
        return CheckResult(
            name="Remote URL",
            status="fail",
            message=f"no hostname in URL: {url}",
        )
    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        return CheckResult(
            name="Remote URL",
            status="fail",
            message=f"DNS resolution failed for {host}",
            fix=f"Check the hostname. Is the leader URL correct?\n  URL: {url}",
            retry_id="peer-url",
        )
    # HTTP probe
    models_url = url.rstrip("/")
    if not models_url.endswith("/v1"):
        models_url += "/v1"
    models_url += "/models"
    try:
        t0 = time.time()
        _http.fetch_url(
            models_url,
            method="GET",
            timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=10.0, total_s=12.0),
        )
        dt_ms = (time.time() - t0) * 1000
    except _http.HTTPAuthError:
        return CheckResult(
            name="Remote URL",
            status="ok",
            message=f"{host} reachable (auth required — checked separately)",
        )
    except _http.HTTPError as e:
        return CheckResult(
            name="Remote URL",
            status="warn",
            message=f"{host} returned HTTP {e.status}",
            retry_id="peer-url",
        )
    except Exception as e:
        return CheckResult(
            name="Remote URL",
            status="fail",
            message=f"cannot reach {host}: {e}",
            fix=f"Is the leader running? Try: curl -v {models_url}",
            retry_id="peer-url",
        )
    return CheckResult(
        name="Remote URL",
        status="ok",
        message=f"{host} reachable ({dt_ms:.0f} ms)",
    )


def check_peer_key_set() -> CheckResult:
    """Check that the peer has a remote API key configured."""
    import os

    key = os.environ.get("MAXIM_LANE_LARGE_REMOTE_API_KEY")
    if not key:
        try:
            from maxim.peer.config import read_peer_config

            cfg = read_peer_config()
            if cfg is not None:
                key = cfg.api_key
        except Exception as e:
            logger.debug("Could not read peer config for API key check: %s", e)
    if not key:
        return CheckResult(
            name="Peer API key",
            status="warn",
            message="no API key configured for remote access",
            fix=(
                "Get the key from the leader:\n"
                "  On leader: maxim tunnel key export\n"
                "  Then paste the export snippet here."
            ),
            retry_id="peer-key",
        )
    from maxim.tunnel.keys import truncate_for_display

    return CheckResult(
        name="Peer API key",
        status="ok",
        message=f"key set: {truncate_for_display(key)}",
    )


def check_peer_auth(url: str, key: str | None) -> CheckResult:
    """Send an authenticated request to the leader and verify 200."""
    from maxim.utils import http as _http

    if not key:
        return CheckResult(
            name="Peer auth",
            status="info",
            message="no key — skipped",
        )
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    models_url = f"{base}/models"
    try:
        _http.fetch_url(
            models_url,
            method="GET",
            headers={"Authorization": f"Bearer {key}"},
            timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=10.0, total_s=12.0),
        )
    except _http.HTTPAuthError:
        return CheckResult(
            name="Peer auth",
            status="fail",
            message="key rejected by leader (401)",
            fix=(
                "Key mismatch. Ask the leader to regenerate:\n"
                "  On leader: maxim tunnel key rotate\n"
                "  Then: maxim tunnel key export"
            ),
            retry_id="peer-auth",
        )
    except _http.HTTPError as e:
        return CheckResult(
            name="Peer auth",
            status="warn",
            message=f"leader returned HTTP {e.status}",
        )
    except Exception as e:
        return CheckResult(
            name="Peer auth",
            status="warn",
            message=f"connection failed: {e}",
        )
    return CheckResult(
        name="Peer auth",
        status="ok",
        message="authenticated successfully",
    )


def check_peer_model(url: str, key: str | None, expected_model: str | None = None) -> CheckResult:
    """Check if the leader advertises the expected model."""
    from maxim.utils import http as _http

    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    models_url = f"{base}/models"
    headers: dict[str, str] = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        resp = _http.fetch_url(
            models_url,
            method="GET",
            headers=headers,
            timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=10.0, total_s=12.0),
        )
        data = resp.json()
        model_ids = [m.get("id", "") for m in data.get("data", [])]
    except Exception:
        return CheckResult(
            name="Remote model",
            status="info",
            message="cannot query /v1/models — skipped",
        )
    if not model_ids:
        return CheckResult(
            name="Remote model",
            status="warn",
            message="leader reports no models loaded",
            fix="On the leader, load a model: maxim peer llm mistral-7b",
        )
    if expected_model and expected_model not in model_ids:
        return CheckResult(
            name="Remote model",
            status="warn",
            message=f"expected {expected_model!r}, leader has: {', '.join(model_ids)}",
            fix=f"On leader: maxim peer llm {expected_model}",
        )
    return CheckResult(
        name="Remote model",
        status="ok",
        message=f"leader model: {', '.join(model_ids)}",
    )


def check_peer_latency(url: str, key: str | None) -> CheckResult:
    """Measure round-trip latency to the leader (5 pings, report p50)."""
    import time

    from maxim.utils import http as _http

    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    ping_url = f"{base}/models"
    headers: dict[str, str] = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    timings: list[float] = []
    fast = _http.TimeoutPolicy(connect_s=2.0, read_s=10.0, total_s=12.0)
    for probe_idx in range(5):
        try:
            t0 = time.time()
            _http.fetch_url(ping_url, method="GET", headers=headers, timeout=fast)
            timings.append((time.time() - t0) * 1000)
        except Exception as e:
            logger.debug("Latency probe %d failed: %s", probe_idx, e)
    if not timings:
        return CheckResult(
            name="Peer latency",
            status="warn",
            message="all latency probes failed",
        )
    timings.sort()
    p50 = timings[len(timings) // 2]
    p95 = timings[int(len(timings) * 0.95)] if len(timings) >= 2 else timings[-1]
    if p50 > 200:
        return CheckResult(
            name="Peer latency",
            status="warn",
            message=f"p50={p50:.0f} ms, p95={p95:.0f} ms — high latency may slow real-time inference",
        )
    return CheckResult(
        name="Peer latency",
        status="ok",
        message=f"p50={p50:.0f} ms, p95={p95:.0f} ms ({len(timings)}/5 probes)",
    )


# ─── leader role ───────────────────────────────────────────────────────────


def check_role() -> CheckResult:
    from maxim.runtime.leader_mode import detect_role as _legacy_detect
    from maxim.runtime.role import detect_role as _new_detect

    decision = _legacy_detect()
    new_role, new_source = _new_detect()

    # Normalise legacy "client" → "peer" for comparison
    legacy_role = "peer" if decision.role == "client" else decision.role

    if legacy_role != new_role:
        # Two role-detection systems disagree — surface this in doctor output
        # rather than burying it as a WARNING-only log event.  Operators need
        # to set MAXIM_ROLE explicitly to resolve it.
        return CheckResult(
            name="Role",
            status="warn",
            message=(
                f"role_divergence: leader_mode says '{legacy_role}' "
                f"({decision.reason}), role.py says '{new_role}' ({new_source}). "
                f"Set MAXIM_ROLE explicitly to resolve."
            ),
            fix=f"export MAXIM_ROLE={new_role}  # or 'leader' / 'solo' as appropriate",
        )

    if decision.is_leader:
        return CheckResult(
            name="Role",
            status="ok",
            message=f"leader (bind={decision.bind_host}) — {decision.reason}",
        )
    return CheckResult(
        name="Role",
        status="ok",
        message=f"{decision.role} (bind={decision.bind_host}) — {decision.reason}",
    )


def _check_lane_metrics() -> list["CheckResult"]:
    """Report per-lane performance metrics if available (Phase 8)."""
    try:
        from maxim.models.language.lane_metrics import get_metrics_registry
    except Exception:
        return []
    registry = get_metrics_registry()
    results: list[CheckResult] = []
    for name, metrics in registry.all_metrics().items():
        snap = metrics.snapshot()
        total = snap["jobs_completed"] + snap["jobs_failed"]
        if total == 0:
            continue
        msg = metrics.format_compact()
        status: Status = "ok"
        if snap["failure_rate"] > 0.2:
            status = "warn"
        if snap["failure_rate"] > 0.5:
            status = "fail"
        results.append(CheckResult(name=f"Lane: {name}", status=status, message=msg))
    return results


def _detect_doctor_role(explicit: str | None = None, peer_url: str | None = None) -> tuple[str, str | None]:
    """Detect whether this machine is peer/leader/solo for doctor purposes.

    Returns ``(role, peer_url)`` where role is ``"peer"``, ``"leader"``,
    ``"solo"``, or ``"auto"`` (fall through to existing behaviour).
    """
    import os
    from urllib.parse import urlparse

    if explicit == "peer":
        url = peer_url or os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL")
        return "peer", url
    if explicit in ("leader", "solo"):
        return explicit, None
    # Auto-detect from env: URL wins (most specific signal)
    url = os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL")
    if url:
        host = urlparse(url).hostname or ""
        if host not in ("127.0.0.1", "localhost", "::1"):
            return "peer", url
    # Fallback: MAXIM_ROLE is set by detect_and_apply_role() before any subcommand
    # runs. If it says "peer", trust it and pull the URL from peer.yml.
    maxim_role = os.environ.get("MAXIM_ROLE", "").strip().lower()
    if maxim_role == "peer":
        try:
            from maxim.peer.config import read_peer_config

            cfg = read_peer_config()
            if cfg is not None:
                return "peer", cfg.url
        except Exception:
            pass
        return "peer", None
    if maxim_role in ("leader", "solo"):
        return maxim_role, None
    return "auto", None


# ─── P8: Routing-decision visibility checks ─────────────────────────────────


def check_tier_effectiveness(caps=None) -> CheckResult:
    """Compare what tier detection actually picked vs. what the hardware allows.

    On a 24 GB Mac that hasn't downloaded qwen2.5-14b yet, ``detect_tiers``
    walks past qwen and lands on the next-best profile that IS on disk
    (e.g. mistral-7b). The tier check still says "ok", which is correct
    but hides the gap. This check runs detect_tiers TWICE — once
    respecting availability, once ignoring it — and reports the gap with
    the exact download command.
    """
    try:
        from maxim.runtime.capabilities import RuntimeCapabilities, detect_compute_resources
        from maxim.runtime.lane_models import detect_tiers
        from maxim.runtime.llm_server import profile_has_local_file
    except ImportError as e:
        return CheckResult(
            name="LLM Tier headroom",
            status="warn",
            message=f"Tier-effectiveness check unavailable: {e}",
        )
    if caps is None:
        try:
            has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()
            caps = RuntimeCapabilities(has_gpu=has_gpu, gpu_type=gpu_type, vram_gb=vram_gb, ram_gb=ram_gb)
        except Exception as e:
            return CheckResult(
                name="LLM Tier headroom",
                status="warn",
                message=f"Capability detection failed: {e}",
            )

    try:
        actual_tiers = detect_tiers(caps, profile_available=profile_has_local_file)
        ideal_tiers = detect_tiers(caps, profile_available=lambda _name: True)
    except Exception as e:
        return CheckResult(
            name="LLM Tier headroom",
            status="warn",
            message=f"Tier walk failed: {e}",
        )

    actual_large = actual_tiers.get("large") or actual_tiers.get("medium")
    ideal_large = ideal_tiers.get("large") or ideal_tiers.get("medium")
    if actual_large is None or ideal_large is None:
        return CheckResult(
            name="LLM Tier headroom",
            status="info",
            message="No large/medium tier detected",
        )
    actual_profile = actual_large.model_profile or "(none)"
    ideal_profile = ideal_large.model_profile or "(none)"
    if actual_profile == ideal_profile:
        return CheckResult(
            name="LLM Tier headroom",
            status="ok",
            message=f"Hardware-best profile selected: {actual_profile}",
        )
    return CheckResult(
        name="LLM Tier headroom",
        status="warn",
        message=(f"Running '{actual_profile}' but hardware ({caps.vram_gb:.0f} GB VRAM) could run '{ideal_profile}'."),
        fix=f"python -m maxim.models.download --llm {ideal_profile}",
        retry_id="tier_effectiveness",
    )


def check_peer_vs_local_conflict() -> CheckResult:
    """Inform the user when --llm + peer config will run locally.

    P1's ``_apply_local_llm_override`` clears the large lane's
    ``remote_url`` whenever ``MAXIM_LLM_PROFILE`` names a local
    (non-cloud) profile. This is the right behavior, but invisible — a
    user with a peer config and ``--llm mistral-7b`` may be expecting
    the leader to serve mistral. This check surfaces the override as an
    informational message so they're not surprised.
    """
    import os

    profile = os.environ.get("MAXIM_LLM_PROFILE", "").strip()
    if not profile:
        return CheckResult(
            name="--llm vs peer config",
            status="ok",
            message="no local --llm override active",
        )
    try:
        from maxim.models.language.config import _BUILTIN_PROFILES
        from maxim.peer.config import read_peer_config

        cfg = read_peer_config()
    except Exception:
        return CheckResult(
            name="--llm vs peer config",
            status="info",
            message=f"--llm={profile} active (peer config check unavailable)",
        )
    if cfg is None:
        return CheckResult(
            name="--llm vs peer config",
            status="ok",
            message=f"--llm={profile} (no peer config)",
        )
    profile_data = _BUILTIN_PROFILES.get(profile, {})
    if profile_data.get("cloud"):
        return CheckResult(
            name="--llm vs peer config",
            status="ok",
            message=f"--llm={profile} is cloud (handled by cloud overrides)",
        )
    return CheckResult(
        name="--llm vs peer config",
        status="info",
        message=(
            f"--llm={profile} will run LOCALLY despite peer config at {cfg.url}. "
            f"P1 clears remote_url for local profiles. To use the leader "
            f"instead, run: maxim peer llm {profile}"
        ),
    )


def check_remote_reachability(url: str | None = None, api_key: str | None = None) -> CheckResult:
    """Probe a remote leader URL with the structured probe (P6).

    Reports outcome-specific fix hints. Distinct from
    ``check_peer_url_reachable`` (the existing peer-mode check) which
    just returns ok/fail; this one surfaces the auth_rejected vs
    dns_fail vs timeout distinction so the user gets a directly
    actionable hint.
    """
    if url is None:
        try:
            from maxim.peer.config import read_peer_config

            cfg = read_peer_config()
            if cfg is None:
                return CheckResult(
                    name="Remote leader probe",
                    status="info",
                    message="No peer config — skipping remote probe",
                )
            url = cfg.url
            api_key = api_key or cfg.api_key
        except Exception as e:
            return CheckResult(
                name="Remote leader probe",
                status="warn",
                message=f"Could not load peer config: {e}",
            )
    try:
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
    except ImportError as e:
        return CheckResult(
            name="Remote leader probe",
            status="warn",
            message=f"Probe unavailable: {e}",
        )
    result = _MaximPeerBackend.for_url(url, api_key=api_key).health_check()
    if result.outcome == "ok":
        return CheckResult(
            name="Remote leader probe",
            status="ok",
            message=f"{url} responding ({result.latency_ms:.0f} ms)",
        )
    if result.outcome == "auth_rejected":
        # Key-drift detection (post-config-unification UX fold,
        # 2026-06-03): route the message + fix through the canonical
        # classifier in peer/probe_classify.py rather than overriding
        # locally. CLAUDE.md C1 invariant: "callers do NOT override
        # the returned message" — pass richer `detail` instead.
        from maxim.peer.probe_classify import classify_probe_outcome

        classification = classify_probe_outcome(
            outcome=result.outcome,
            detail=result.detail,
            latency_ms=result.latency_ms,
            url=url,
        )
        return CheckResult(
            name="Remote leader probe",
            status="warn",  # auth_rejected is recoverable (re-paste key)
            message=classification.message,
            fix=classification.fix,
            retry_id="remote_probe",
        )
    fix_hints = {
        "dns_fail": f"Check the hostname in peer.yml or $MAXIM_LANE_LARGE_REMOTE_URL ({url})",
        "tls_error": "Check the leader's TLS certificate validity",
        "connection_refused": "Leader is not accepting connections — start `maxim` on the leader",
        "timeout": "Leader did not respond in time — is it cold-loading a model?",
        "http_5xx": "Leader returned a server error — check `maxim peer logs`",
        "other": "Unexpected error — check `maxim peer logs`",
    }
    return CheckResult(
        name="Remote leader probe",
        status="fail",
        message=f"{url} unreachable: {result.outcome} ({result.detail})",
        fix=fix_hints.get(result.outcome, "check `maxim peer logs`"),
        retry_id="remote_probe",
    )


def check_storage_footprint() -> CheckResult:
    """Surface ~/.maxim disk usage with a top-N subdir breakdown.

    Distinct from check_disk_space (which is a fs-level free check):
    this one tells the user where their Maxim footprint is going so
    they can decide what to delete. Fails when free space drops below
    10 GB, warns below 20 GB.
    """
    try:
        from maxim.utils.storage import format_report, report_storage
    except ImportError as e:
        return CheckResult(
            name="Storage footprint",
            status="warn",
            message=f"Storage report unavailable: {e}",
        )
    try:
        report = report_storage(force=True)
    except Exception as e:
        return CheckResult(
            name="Storage footprint",
            status="warn",
            message=f"Storage walk failed: {e}",
        )
    summary = format_report(report)
    if report.fs_free_gb == float("inf"):
        return CheckResult(
            name="Storage footprint",
            status="info",
            message=summary,
        )
    if report.fs_free_gb < 10.0:
        return CheckResult(
            name="Storage footprint",
            status="fail",
            message=f"Only {report.fs_free_gb:.1f} GB free on {report.data_home}",
            fix=summary + "\n  Delete unused models: maxim --delete-model NAME",
        )
    if report.fs_free_gb < 20.0:
        return CheckResult(
            name="Storage footprint",
            status="warn",
            message=f"{report.fs_free_gb:.1f} GB free on {report.data_home} (Maxim using {report.total_maxim_gb:.1f} GB)",
            fix=summary,
        )
    return CheckResult(
        name="Storage footprint",
        status="ok",
        message=f"{report.fs_free_gb:.1f} GB free, Maxim using {report.total_maxim_gb:.1f} GB",
    )


# ─── mesh-mode checks (Plan 4 Stage C1) ──────────────────────────────────


def check_mesh_nodes() -> list[CheckResult]:
    """Probe every node declared in ``mesh.yml`` via
    :meth:`maxim.models.language.maxim_peer_backend._MaximPeerBackend.health_check`.

    Returns one :class:`CheckResult` per node. When ``mesh.yml`` is
    absent (and no ``peer.yml`` fallback), returns ``[]`` so the caller
    can cleanly skip the section via a normal truthy check. Schema
    errors in ``mesh.yml`` surface as a single ``fail`` result.

    Retry ids are per-node: ``mesh_node_<name>``. The retry loop in
    ``doctor/cli.py`` decides whether each id's status warrants a
    re-probe — the producer just sets the stable identity.

    **Shared classifier:** probe-outcome → (status, message, fix_hint)
    classification routes through
    :func:`maxim.peer.probe_classify.classify_probe_outcome` — the
    single source of truth for this mapping across mesh_cli, doctor,
    and future C2/C3 callers. Callers pass a caller-specific ``detail``
    string through the classifier rather than post-processing the
    returned message.
    """
    from maxim.peer.drain_state import read_drained_nodes
    from maxim.peer.mesh_config import MeshConfigError, read_or_synthesize_mesh_config

    try:
        mesh = read_or_synthesize_mesh_config()
    except MeshConfigError as e:
        return [
            CheckResult(
                name="Mesh config",
                status="fail",
                message=str(e),
                fix="Edit ~/.config/maxim/mesh.yml to fix the schema error above.",
            )
        ]
    if mesh is None:
        return []

    # Plan 4 C2: drained nodes render as info (not probed) and orphan
    # drain entries (drain state names that no longer match any
    # mesh.yml node) surface as a warn with a resume hint — operator
    # may be mid-edit, so this is not a hard fail.
    known_names = {n.name for n in mesh.nodes}
    drain_result = read_drained_nodes(known_names)

    results: list[CheckResult] = []
    for node in mesh.nodes:
        if node.name in drain_result.active:
            results.append(_drained_mesh_node_check(node))
        else:
            results.append(_probe_mesh_node_to_check(node, mesh.cluster_key))

    for orphan in sorted(drain_result.orphans):
        results.append(
            CheckResult(
                name=f"Drain orphan {orphan}",
                status="warn",
                message=(
                    f"drain state entry {orphan!r} has no matching node in mesh.yml "
                    "— operator mid-edit, or a stale entry"
                ),
                fix=(
                    f"Run `maxim peer --node {orphan} resume` to clean up the drain\n"
                    f"state, or re-add {orphan!r} to mesh.yml::nodes."
                ),
                retry_id=f"mesh_drain_orphan_{orphan}",
            )
        )
    return results


def _drained_mesh_node_check(node: "MeshNode") -> CheckResult:  # noqa: F821
    """Render a drained node as an ``info`` CheckResult without
    probing. Plan 4 C2: drain is operator-intentional, not a failure
    signal, and doctor MUST NOT make a network call for drained
    nodes — regression-guarded in ``test_doctor.py``.
    """
    return CheckResult(
        name=f"Node {node.name}",
        status="info",
        message=f"drained (not probed) — {node.role}, {node.url}",
        fix=None,
        retry_id=None,
    )


def _probe_mesh_node_to_check(node: "MeshNode", cluster_key: str) -> CheckResult:  # noqa: F821
    """Run one ``health_check`` and render a :class:`CheckResult`.

    Uses ``_MaximPeerBackend.for_url(url, api_key=k)`` — the canonical
    probe entry point (Plan 3 R2.6) that does NOT mutate
    ``os.environ`` and is concurrency-safe via instance-level
    ``_api_key_override``. Routes classification through
    :func:`maxim.peer.probe_classify.classify_probe_outcome`.

    Doctor passes ``detail = f"{node.role}, {node.url}"`` so the
    classifier's "reachable" message includes the operator-relevant
    context without any post-hoc override — round 2 review A1R2
    flagged message rewriting as a shared-helper leak.
    """
    from maxim.peer.probe_classify import classify_probe_outcome

    retry_id = f"mesh_node_{node.name}"

    try:
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
    except ImportError as e:
        return CheckResult(
            name=f"Node {node.name}",
            status="warn",
            message=f"peer backend import failed: {e}",
            fix="Install the llm-server extra: pip install -e '.[llm-server]'",
            retry_id=retry_id,
        )

    # health_check is contractually required to return a ProbeResult
    # rather than raise (Plan 3 R2.6). Catching Exception broadly
    # here would hide backend regressions as silent warnings — let
    # them propagate up to the doctor runner.
    backend = _MaximPeerBackend.for_url(node.url, api_key=cluster_key)
    result = backend.health_check(enable_stage2=True)

    outcome = getattr(result, "outcome", "other")
    latency = getattr(result, "latency_ms", None)
    # Doctor's richer detail flows through the classifier rather than
    # being patched in afterwards. For non-ok outcomes we still want
    # the raw detail so operators see probe-level context.
    if outcome == "ok":
        detail = f"{node.role}, {node.url}"
    else:
        detail = getattr(result, "detail", "")
    classification = classify_probe_outcome(outcome, detail, latency, node.url)

    # retry_id is always set; the retry loop in doctor/cli.py filters on
    # status to decide which checks to re-run. Coupling status and
    # retry_id at the producer (round 1 fold) was redundant and meant
    # future info-status results couldn't be re-probed (round 2 A2R2).
    return CheckResult(
        name=f"Node {node.name}",
        status=classification.status,
        message=classification.message,
        fix=classification.fix,
        retry_id=retry_id,
    )


def check_embodiment_ref(ref: str) -> CheckResult:
    """Validate that an SEM component ref resolves in the registry.

    Stage 4 of sem_execution_hook. When the user passes
    ``maxim doctor --embodiment <REF>`` (mirroring the
    ``maxim --llm X --embodiment <REF>`` agent flag), this check
    confirms the ref is loadable BEFORE they start the agent — a typo
    becomes a doctor-level fail with the available refs as the fix
    hint, instead of a runtime ``ComponentNotFoundError`` after the
    agent has spent time spinning up.

    If the ref does not resolve, the fix hint includes a sorted list
    of available refs (slicing to the first 20 to keep the doctor
    output bounded; longer registries get a "...and N more" suffix).

    The function is only called from `run_all_checks` when
    `embodiment_ref` is truthy — there is no empty-ref branch.
    Pre-merge review (A-arch-3) caught the original empty-ref `info`
    return as dead code that lied in the function contract.
    """
    try:
        from maxim.embodiment.component_registry import ComponentRegistry
    except ImportError as e:
        return CheckResult(
            name="Embodiment ref",
            status="warn",
            message=f"ComponentRegistry import failed: {e}",
            fix="Install the embodiment extras (typically core install).",
        )

    try:
        registry = ComponentRegistry()
    except Exception as e:
        return CheckResult(
            name="Embodiment ref",
            status="fail",
            message=f"ComponentRegistry construction failed: {e}",
            fix=(
                "Check that ~/.maxim/components/ exists or that the bundled "
                "_data/components/ tree is intact. Run `maxim --list-models` "
                "to verify the bundled data is present."
            ),
        )

    try:
        spec = registry.get(ref)
    except Exception:
        # ComponentNotFoundError already includes a sorted available
        # list in its message — but doctor output is bounded, so
        # rebuild the hint with a length cap. Prefer same-category
        # matches so a typo in `weapons/X` surfaces weapons options
        # first instead of alphabetically-first categories.
        # Use the PUBLIC list_refs() API rather than reaching into
        # `_index` — pre-merge review I5 cross-confirmed: the private
        # attribute coupling silently fell back to "registry has no
        # components" if `_index` was ever renamed.
        try:
            available = sorted(registry.list_refs())
        except Exception:
            available = []
        if not available:
            return CheckResult(
                name="Embodiment ref",
                status="fail",
                message=f"Component ref {ref!r} not found",
                fix="ComponentRegistry has no components — check your installation.",
            )

        # Prefix-aware preview: if the user typed `weapons/foo`,
        # show all weapons/* refs first (likely typo), then a
        # category sample of the rest.
        category = ref.split("/", 1)[0] if "/" in ref else ""
        same_category = [r for r in available if category and r.startswith(f"{category}/")]
        other = [r for r in available if r not in same_category]

        preview_lines: list[str] = []
        if same_category:
            preview_lines.append(f"Components in {category!r}:")
            preview_lines.extend(f"  {r}" for r in same_category[:20])
            if len(same_category) > 20:
                preview_lines.append(f"  ...and {len(same_category) - 20} more in {category!r}")
            preview_lines.append("")
        preview_lines.append("Other available components:")
        preview_lines.extend(f"  {r}" for r in other[:15])
        if len(other) > 15:
            preview_lines.append(f"  ...and {len(other) - 15} more")

        return CheckResult(
            name="Embodiment ref",
            status="fail",
            message=f"Component ref {ref!r} not found",
            fix="\n".join(preview_lines),
        )

    # Resolution succeeded — surface the entity name + a short stat
    # so the operator sees what would be loaded.
    entity_name = spec.get("name") if isinstance(spec, dict) else None
    return CheckResult(
        name="Embodiment ref",
        status="ok",
        message=f"{ref!r} resolves" + (f" → entity={entity_name!r}" if entity_name else ""),
    )


def check_user_profiles() -> CheckResult:
    """Surface ``~/.config/maxim/profiles.yml`` state so operators see
    user-profile counts AND any schema-validation errors at doctor time,
    not at first ``maxim`` invocation.

    Four outcomes:

    - File missing → ``info`` ("0 user profiles defined")
    - File parses + all entries validate → ``ok`` with count
    - YAML syntax broken → ``fail`` with parse error
    - YAML parses but a profile entry fails per-profile schema
      validation (missing required field, bad enum, broken ``arch:``
      block, intra-user alias collision, etc.) → ``fail`` with the
      specific schema error

    A malformed ``profiles.yml`` blocks ``maxim`` startup (the loader
    raises ``ConfigurationError`` from module import). Pre-merge
    executor review caught the original implementation only running
    ``load_user_profiles`` (top-level YAML parse) and missing the
    per-profile schema layer — leaving operators with a misleading
    "OK / N loaded" doctor report for files that would crash startup.
    The fix routes through the full ``apply_user_profiles`` pipeline
    against throwaway dicts so every schema error surfaces.
    """
    try:
        from maxim.models.language.profile_loader import (
            apply_user_profiles,
            load_user_profiles,
            profiles_config_path,
        )
    except ImportError as exc:
        return CheckResult(
            name="User profiles",
            status="info",
            message=f"profile loader unavailable: {exc}",
        )

    path = profiles_config_path()
    if not path.is_file():
        return CheckResult(
            name="User profiles",
            status="info",
            message=f"no user profiles file at {path} (0 user profiles)",
        )
    try:
        entries = load_user_profiles(path=path)
        # Run the full validate-and-merge pipeline against throwaway
        # dicts so per-profile schema errors surface (not just YAML
        # syntax). Discarded dicts mean no mutation of the runtime
        # state — this is a pure validation pass.
        apply_user_profiles({}, {}, {}, path=path)
    except Exception as exc:
        return CheckResult(
            name="User profiles",
            status="fail",
            message=f"{path}: {exc}",
            fix=(
                f"Fix the schema error in {path} (or move it aside) — `maxim` will refuse to start until this resolves."
            ),
        )
    count = len(entries)
    return CheckResult(
        name="User profiles",
        status="ok" if count else "info",
        message=f"{count} user profile{'s' if count != 1 else ''} loaded from {path}",
    )


# ─── robot checks (docs/plans/archive/doctor_robot_reachable.md) ────────────────────


def _parse_major_minor(version: str) -> tuple[int, int] | None:
    try:
        parts = version.split(".")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError, AttributeError):
        return None


def _robot_sdk_version() -> tuple[int, int] | None:
    """(major, minor) of the locally installed reachy-mini SDK, or None.

    Package metadata, not ``__version__`` — pre-1.5 (zenoh-era) SDKs have no
    ``__version__`` attribute, so metadata is the only cross-era probe.
    """
    try:
        from importlib.metadata import version

        return _parse_major_minor(version("reachy-mini"))
    except Exception:  # noqa: BLE001 - not installed / metadata unreadable
        return None


def _check_one_robot(info: PlatformInfo, robot: Any) -> CheckResult:
    """Reachability + era coherence for one configured robot.

    Probe order mirrors the hardware-validated sequence
    (docs/embodiment/reachy_mini/troubleshooting.md): resolve → TCP :8000 →
    GET /api/daemon/status → era/pin coherence.
    """
    import socket

    name = f"robot:{robot.robot_id}"

    # 1. Resolve — config host: verbatim; else IPv4-only gethostbyname of
    # <robot_name>.local (sidesteps the getaddrinfo/IPv6-first flake).
    host = robot.config.get("host")
    if not host:
        mdns_name = str(robot.config.get("robot_name", "reachy_mini")).replace("_", "-") + ".local"
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(2.0)
        try:
            host = socket.gethostbyname(mdns_name)
        except OSError:
            fix = f"set host: <robot-ip> in ~/.maxim/robots.yaml (DHCP reservation recommended); {mdns_name} did not resolve"
            if info.os == "macos":
                fix = (
                    "grant this terminal Local Network permission (System Settings → "
                    "Privacy & Security → Local Network) and/or " + fix
                )
            return CheckResult(name, "fail", f"{mdns_name} does not resolve", fix=fix, retry_id="robot")
        finally:
            socket.setdefaulttimeout(old_timeout)

    # 2. TCP :8000 — the single WS-era control port (FastAPI + /ws/sdk).
    try:
        with socket.create_connection((host, 8000), timeout=2.0):
            pass
    except OSError:
        return CheckResult(
            name,
            "fail",
            f"daemon port {host}:8000 unreachable",
            fix=(
                f"ssh pollen@{host} → systemctl status reachy-mini-daemon; after a Wi-Fi "
                "change, reboot the robot (the daemon binds at startup)"
                + ("; on macOS also check Local Network permission" if info.os == "macos" else "")
            ),
            retry_id="robot",
        )

    # 3. Daemon status (version + backend state).
    daemon_version: str | None = None
    daemon_state: str | None = None
    try:
        from maxim.utils import http as maxim_http

        status = maxim_http.fetch_url(f"http://{host}:8000/api/daemon/status", timeout=2.0).json()
        daemon_version = str(status.get("version", "")) or None
        daemon_state = str(status.get("state", "")) or None
    except Exception:  # noqa: BLE001
        return CheckResult(
            name,
            "warn",
            f"{host}:8000 open but /api/daemon/status unreadable (pre-1.5 daemon or endpoint moved)",
            fix="check the daemon era: docs/embodiment/reachy_mini/troubleshooting.md",
            retry_id="robot",
        )

    # 4. Era / pin coherence — daemon vs local SDK (v1.5.0 zenoh→WS pivot).
    sdk_mm = _robot_sdk_version()
    daemon_mm = _parse_major_minor(daemon_version or "")
    state_note = f", state={daemon_state}" if daemon_state else ""
    if sdk_mm is None:
        return CheckResult(
            name,
            "warn",
            f"reachable (daemon {daemon_version or '?'}{state_note}) but no local reachy-mini SDK found",
            fix="pip install 'reachy-mini[gstreamer]>=1.8.3,<2.0' (in the interpreter you run Maxim from)",
            retry_id="robot",
        )
    if daemon_mm is not None and ((sdk_mm < (1, 5)) != (daemon_mm < (1, 5)) or sdk_mm < (1, 5)):
        return CheckResult(
            name,
            "fail",
            f"era mismatch: daemon {daemon_version} vs local SDK {sdk_mm[0]}.{sdk_mm[1]}.x "
            "(opposite sides of the v1.5.0 zenoh→WebSocket pivot)",
            fix=(f'pip install "reachy_mini=={daemon_version}" — see docs/embodiment/reachy_mini/troubleshooting.md'),
            retry_id="robot",
        )
    if daemon_mm is not None and daemon_mm != sdk_mm:
        return CheckResult(
            name,
            "warn",
            f"reachable (daemon {daemon_version}{state_note}) but SDK minor drift (local {sdk_mm[0]}.{sdk_mm[1]}.x)",
            fix=f'pip install "reachy_mini=={daemon_version}" to match exactly',
            retry_id="robot",
        )
    if daemon_state and daemon_state != "running":
        return CheckResult(
            name,
            "warn",
            f"daemon up (version {daemon_version}) but state={daemon_state} — WS connects will be refused (1013)",
            fix=f"journalctl -u reachy-mini-daemon on the robot (ssh pollen@{host})",
            retry_id="robot",
        )
    return CheckResult(
        name,
        "ok",
        f"reachable at {host}:8000 (daemon {daemon_version or '?'}{state_note}, SDK match)",
    )


def check_robot_reachable(info: PlatformInfo) -> list[CheckResult]:
    """Per-robot reachability + era coherence for ~/.maxim/robots.yaml entries.

    Full spec: docs/plans/archive/doctor_robot_reachable.md. The gate NEVER fails —
    a machine without robots must not fail doctor because of this check.
    """
    try:
        from maxim.hardware.config import load_robots_config

        cfg = load_robots_config()
    except Exception as e:  # noqa: BLE001 - malformed yaml etc.
        return [
            CheckResult(
                "robots.yaml",
                "warn",
                f"could not load robots config: {e}",
                fix="fix ~/.maxim/robots.yaml (YAML syntax / schema)",
            )
        ]
    if not cfg.robots:
        return [
            CheckResult(
                "robots",
                "info",
                "no robots configured — skipping robot checks (add ~/.maxim/robots.yaml to enable)",
            )
        ]
    return [_check_one_robot(info, robot) for robot in cfg.robots]


def run_all_checks(
    info: PlatformInfo,
    *,
    role: str | None = None,
    peer_url: str | None = None,
    peer_key: str | None = None,
    embodiment_ref: str | None = None,
) -> list[tuple[str, list[CheckResult]]]:
    """Return ordered ``[(section_name, results)]`` for ``maxim doctor``.

    Parameters
    ----------
    role
        Explicit role override: ``"peer"``, ``"leader"``, ``"solo"``.
        ``None`` triggers auto-detection.
    peer_url
        Remote leader URL (used when role is ``"peer"``).
    peer_key
        API key for the remote leader.
    """
    detected_role, detected_url = _detect_doctor_role(role, peer_url)
    peer_url = peer_url or detected_url

    # ── environment (always) ──────────────────────────────────────────────
    env_checks: list[CheckResult] = [
        CheckResult(name="Platform", status="ok", message=info.display_name),
        CheckResult(name="Architecture", status="ok", message=info.arch),
        check_gpu(),
        check_tier_detection(),
        check_tier_effectiveness(),
        check_disk_space(),
        check_ram_headroom(),
        check_storage_footprint(),
        check_user_profiles(),
    ]
    # Splice env-var checks in after the hardware checks so operators see
    # misconfigurations right alongside the hardware context.
    env_checks.extend(check_env_config(info, role=detected_role))
    sections: list[tuple[str, list[CheckResult]]] = [
        ("Environment", env_checks),
    ]

    # C5 of config_unification.md: "Resolved Config" section surfaces
    # every absorbed field with its effective value + source marker.
    # Single answer to "what does this instance think it's configured
    # as?" — collapses what previously required cross-referencing 96
    # env vars + 4 config files + 2 role detectors.
    sections.append(("Resolved Config", check_resolved_config()))

    if detected_role == "peer" and peer_url:
        # ── peer-mode sections ────────────────────────────────────────────
        # Resolve key from arg → env → peer config
        if not peer_key:
            import os

            peer_key = os.environ.get("MAXIM_LANE_LARGE_REMOTE_API_KEY")
        if not peer_key:
            try:
                from maxim.peer.config import read_peer_config

                cfg = read_peer_config()
                if cfg is not None:
                    peer_key = cfg.api_key
            except Exception as e:
                logger.debug("Could not read peer config in run_all_checks: %s", e)
        sections.append(
            (
                "Peer Connectivity",
                [
                    check_peer_url_reachable(peer_url),
                    check_peer_key_set(),
                    check_peer_auth(peer_url, peer_key),
                    check_remote_reachability(peer_url, peer_key),
                    check_peer_vs_local_conflict(),
                    check_peer_model(peer_url, peer_key),
                    check_peer_latency(peer_url, peer_key),
                ],
            )
        )
    else:
        # ── leader / solo sections ────────────────────────────────────────
        sections += [
            ("Robots", check_robot_reachable(info)),
            (
                "Local LLM",
                [
                    check_llama_cpp_server_installed(),
                    check_server_reachable(),
                    check_llm_model_active(),
                    check_context_window(),
                    check_vram_pressure(),
                    check_inference_coherence(),
                ],
            ),
            (
                "Role & Access",
                [
                    check_role(),
                    check_lan_access(info),
                ],
            ),
            (
                "Tunnel (Cloudflare)",
                [
                    check_cloudflared(info),
                    check_tunnel_config(),
                    check_tunnel_config_sync(),
                ],
            ),
            (
                "API key",
                [
                    check_api_key(),
                    check_key_age(),
                    check_key_permissions(),
                    check_key_auth_smoke(),
                ],
            ),
        ]

    # Mesh topology (Plan 4 Stage C1): only shown when mesh.yml or
    # peer.yml exists. ``check_mesh_nodes`` returns ``[]`` for the
    # unconfigured case so the section is skipped naturally.
    mesh_results = check_mesh_nodes()
    if mesh_results:
        sections.append(("Mesh Topology", mesh_results))

    # Embodiment ref validation (sem_execution_hook Stage 4): only
    # shown when ``maxim doctor --embodiment <REF>`` was passed.
    # Hidden by default so the doctor command behaves identically to
    # before for users who aren't using SEM bodies.
    if embodiment_ref:
        sections.append(
            ("Embodiment", [check_embodiment_ref(embodiment_ref)]),
        )

    # Lane metrics (only show if any calls have been recorded)
    lane_results = _check_lane_metrics()
    if lane_results:
        sections.append(("Lane Metrics", lane_results))
    return sections


__all__ = [
    "CheckResult",
    "Status",
    "check_gpu",
    "check_tier_detection",
    "check_llama_cpp_server_installed",
    "check_server_reachable",
    "check_llm_model_active",
    "check_env_config",
    "check_context_window",
    "check_vram_pressure",
    "check_lan_access",
    "check_cloudflared",
    "check_tunnel_config",
    "check_tunnel_config_sync",
    "check_api_key",
    "check_key_age",
    "check_key_permissions",
    "check_key_auth_smoke",
    "check_disk_space",
    "check_ram_headroom",
    "check_inference_coherence",
    "check_peer_url_reachable",
    "check_peer_key_set",
    "check_peer_auth",
    "check_peer_model",
    "check_peer_latency",
    "check_role",
    "check_tier_effectiveness",
    "check_peer_vs_local_conflict",
    "check_remote_reachability",
    "check_storage_footprint",
    "check_user_profiles",
    "check_mesh_nodes",
    "run_all_checks",
]
