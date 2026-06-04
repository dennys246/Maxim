"""Capability-driven per-lane LLM profile assignment (multi-LLM Phase 2).

Maps detected compute resources (VRAM, RAM) to concrete LLM profile choices for
each WorkerPool lane. Used by LaneBackendManager (Phase 3) to decide which
backend to instantiate per lane.

This module only builds the *configuration* — it does not load models or create
backends. That happens in Phase 3.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from maxim.runtime.capabilities import RuntimeCapabilities
from maxim.runtime.worker_pool import LaneConfig

ProfileAvailabilityCheck = Callable[[str], bool]


@dataclass(frozen=True)
class LaneModelConfig:
    """Resolved LLM assignment for a single WorkerPool lane."""

    profile: str  # Matches a key in BUILTIN_PROFILES
    device: str = "auto"  # "gpu" | "cpu" | "auto"
    n_gpu_layers: int = -1  # -1 = all on GPU, 0 = CPU only


# VRAM tiers for the "large" tier (the hot-path LLM that benefits most from GPU).
# Each tier names an existing profile from BUILTIN_PROFILES. Extend this table
# when new profiles are added (e.g., a 14B / 24B GGUF for >16GB cards).
#
# Tiers are inclusive-lower, exclusive-upper: a 15.9GB card lands in the >=8 tier.
_INFER_VRAM_TIERS: tuple[tuple[float, str], ...] = (
    (16.0, "qwen2.5-14b-instruct"),  # 14B Q4 ~9.5GB weights, 32K max ctx, strong tool-use (P4a)
    (14.0, "llama-2-13b-chat"),  # 13B Q4 ~8GB, 4K ctx
    (8.0, "llama-3-8b-instruct"),  # 8B Q4 ~5GB, 8K ctx
    (4.0, "mistral-7b-instruct-v0.2"),  # 7B Q4 ~4.5GB, 4K ctx
    (0.0, "smollm-1.7b-instruct"),  # 1.7B Q4 ~1GB, CPU fallback
)
# NOTE: The 16 GB threshold covers both "CUDA discrete 16 GB cards
# (RTX 5080 class)" and "Apple Silicon 24 GB effective VRAM at default
# headroom". qwen2.5-14b's declared profile n_ctx of 32K produces a KV
# cache cost that only fits in budgets above ~20 GB with f16 KV, so
# 16 GB cards need a reduced runtime n_ctx. That's P4c's responsibility
# (estimate_max_ctx) — the static tier-table placement here just says
# "qwen is the best model the hardware CAN run at some context size",
# not "qwen fits at its advertised 32K context".

# RAM threshold (GB) below which the review lane shares the infer backend
# instead of loading its own CPU model. SmolLM Q4 needs ~2GB resident + overhead.
_REVIEW_MIN_RAM_GB = 4.0

# RAM tiers for the medium tier (CPU/MPS). Picks the best model that fits
# in available RAM with headroom for the OS + small tier (~2GB overhead).
# Format: (min_ram_gb, profile)
# Walk-down: first match wins. Uses the profile's default quantization
# (Q4_K_M) — different RAM levels get different-sized models rather than
# the same model at different quantizations (avoids GGUF download complexity).
_MEDIUM_RAM_TIERS: tuple[tuple[float, str], ...] = (
    (16.0, "mistral-7b-instruct-v0.2"),  # 7B Q4 ~4.5GB — plenty of headroom
    (8.0, "phi-3-mini-4k-instruct"),  # 3.8B Q4 ~2.5GB — fits on 8GB with headroom
    # Below 8GB: no medium tier (only small)
)


def _pick_medium_profile(
    ram_gb: float,
    profile_available: ProfileAvailabilityCheck | None = None,
) -> str | None:
    """Pick the best medium-tier model for available RAM.

    Returns profile name or None if RAM is too low for any medium model.
    """
    check = profile_available or (lambda _: True)
    for min_ram, profile in _MEDIUM_RAM_TIERS:
        if ram_gb >= min_ram and check(profile):
            return profile
    return None


# Lightweight profile used for CPU-side review/reflection when RAM permits.
_REVIEW_PROFILE = "smollm-1.7b-instruct"


# ─── P4c: dynamic n_ctx sizing ───────────────────────────────────────────────
#
# Bytes-per-KV-token map. The formula
#     kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * kv_type_bytes
# (the leading 2 covers K and V) gets multiplied by n_ctx to compute total KV
# cache cost. The "2" coefficient and arch-field meanings are documented in
# the llama.cpp memory math whitepaper.
_KV_TYPE_BYTES: dict[str, float] = {
    "f16": 2.0,
    "q8_0": 1.0,
    "q4_0": 0.5,
}
# Hard floor for "useful for agent work". Below this, tool prompts won't fit.
_MIN_VIABLE_CTX = 2048
# Default safety headroom (GB) reserved on top of weights + KV before we
# declare a context size viable. Covers KV growth during decode + activation
# scratchpads + the OS keeping a few hundred MB resident.
_DEFAULT_SAFETY_MARGIN_GB = 1.5


def estimate_max_ctx(
    profile_meta: dict,
    vram_budget_gb: float,
    *,
    safety_margin_gb: float = _DEFAULT_SAFETY_MARGIN_GB,
    kv_quant_mode: str = "f16",
) -> int:
    """Largest n_ctx (multiple of 1024) that fits weights + KV cache + margin.

    Returns 0 if the weights alone exceed the budget — caller should fall
    through to a smaller profile. Caps the returned value at the profile's
    declared ``n_ctx`` so we never request more context than the model was
    trained for.

    Requires ``profile_meta["arch"]`` populated by P4b. Custom profiles
    without arch metadata return 0; callers must consult
    :data:`_NCTX_SAFE_FALLBACK` (or the env-var default) instead.
    """
    arch = profile_meta.get("arch")
    if not arch:
        return 0
    try:
        weights_gb = float(arch["weights_gb"])
        n_layers = int(arch["n_layers"])
        n_kv_heads = int(arch["n_kv_heads"])
        head_dim = int(arch["head_dim"])
    except (KeyError, TypeError, ValueError):
        return 0

    kv_type_bytes = _KV_TYPE_BYTES.get(kv_quant_mode, 2.0)
    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * kv_type_bytes
    if kv_bytes_per_token <= 0:
        return 0

    available_kv_gb = vram_budget_gb - weights_gb - safety_margin_gb
    if available_kv_gb <= 0:
        return 0

    max_tokens = int(available_kv_gb * (1024**3) / kv_bytes_per_token)
    profile_max = profile_meta.get("n_ctx")
    if isinstance(profile_max, int) and profile_max > 0:
        max_tokens = min(max_tokens, profile_max)
    # Round down to a multiple of 1024 — llama.cpp doesn't require it but
    # rounded values produce nicer logs and avoid spurious "5121" surprises.
    rounded = (max_tokens // 1024) * 1024
    return max(rounded, 0)


# Measured "known good" context sizes per (profile, vram). Walk-largest-first
# tuples. Used as a sanity floor over the formula: when a card sits below the
# safety_margin assumption (e.g. background OS apps competing on a Mac), the
# fallback table catches it before the formula picks an OOM-prone value.
_NCTX_SAFE_FALLBACK: dict[str, tuple[tuple[float, int], ...]] = {
    "qwen2.5-14b-instruct": (
        (24.0, 16384),
        (18.0, 8192),
        (16.0, 4096),
    ),
    "llama-2-13b-chat": (
        (16.0, 4096),
        (14.0, 2048),
    ),
    "llama-3-8b-instruct": (
        (12.0, 8192),
        (8.0, 4096),
    ),
    "mistral-7b-instruct-v0.2": (
        (8.0, 4096),
        (4.0, 2048),
    ),
}


# Plan 3.6 R5: VRAM spillover projection.
#
# When projected (weights + KV cache + headroom) > 95% of physical VRAM, NVIDIA
# drivers silently spill KV cache into shared system memory on Windows (and
# some Linux configurations). The model still loads cleanly — it just runs at
# PCIe speeds, roughly 5-30x slower than on-GPU inference. llama-cpp-server
# has no log line for this; it's invisible from the outside.
#
# This ratio is a deliberate floor. The user's 2026-04-13 incident had a
# 16 GB RTX 5080 at vram_used/vram_total ≈ 0.97 running Qwen-14B Q4 at
# n_ctx=12380 and measured 0.9 tok/s (vs the 30-50 tok/s healthy baseline).
# 95% leaves room for llama-cpp's ~1.5 GB of activation scratchpads +
# driver overhead before the UMA fallback kicks in.
_SPILLOVER_RATIO = 0.95
# Warning band: below spillover but with no headroom for KV growth during
# long prompts. Doctor emits WARN (not FAIL) in this band. Single source of
# truth for both spawn-time projection and the live nvidia-smi ratio check
# in doctor/checks.py::check_vram_pressure.
_SPILLOVER_WARN_RATIO = 0.85

# Dynamic headroom for Plan 3.6's spillover projection. The _DEFAULT_SAFETY_MARGIN_GB
# above (1.5 GB) is calibrated for 7B-class models and underestimates activation
# scratchpads + CUDA allocator overhead + driver reserved memory for 14B+ models.
#
# The 2026-04-13 incident observation: qwen2.5-14b-instruct (8.4 GB weights) at
# n_ctx=12380 (KV ≈ 2.27 GB) measured 15.3 GB VRAM used on a 16 GB card. That's
# 15.3 - 8.4 - 2.27 = 4.63 GB of non-weights-non-KV overhead. The 1.5 GB floor
# would have projected 12.17 GB and declared the config safe — silently wrong.
#
# Empirical ratio: 4.63 / 8.4 ≈ 0.55. Applied as floor:
#     headroom_gb = max(1.5, weights_gb * 0.55)
#
# This is OBSERVABILITY, not a routing change. estimate_max_ctx() continues to
# use the 1.5 GB floor because most of its callers take `min(formula, fallback)`
# against _NCTX_SAFE_FALLBACK (measured-good n_ctx per profile × VRAM). Lowering
# estimate_max_ctx's headroom would ripple through auto-spawn on every install.
#
# KNOWN GAP (Plan 3.6 R5 review finding B2): detect_tiers()'s MAXIM_LLM_PROFILE
# env-profile branch below (around line 444) takes the formula value directly
# and only consults the fallback when the formula returns ≤ 0. An
# optimistic-but-positive formula value (e.g., qwen2.5-14b @ 16 GB) bypasses
# the fallback table. That's the exact user-facing path most likely to hit
# the 14B-on-16GB spillover this plan exists to detect, and the spawn-time
# _check_vram_spillover_risk warn is currently the only safety net for it.
# Deferred to a follow-up: either (a) extend the env-profile branch to also
# take min(formula, fallback), or (b) raise estimate_max_ctx's headroom floor
# with hardware validation across 7B/14B/24B+ tiers.
_ACTIVATION_OVERHEAD_RATIO = 0.55


def _dynamic_headroom_gb(weights_gb: float) -> float:
    """Plan 3.6 R5: activation+allocator+driver overhead scales with model size."""
    return max(_DEFAULT_SAFETY_MARGIN_GB, weights_gb * _ACTIVATION_OVERHEAD_RATIO)


@dataclass(frozen=True)
class VRAMProjection:
    """Projected VRAM footprint for a (profile, n_ctx) pair.

    Pure data object — no side effects. Callers decide how to surface the
    result (doctor WARN, startup log, refusal to spawn, …).

    SHAPE-FROZEN at 1.0 (CC3). Surfaces in ``maxim doctor --json`` output
    consumed by external tooling. Adding new optional fields with defaults
    at the end is non-breaking; an ``extra`` dict would dilute the typed
    diagnostic surface. Adding a *required* field post-1.0 is a
    major-version-bump change for the doctor JSON contract.
    """

    profile: str
    n_ctx: int
    physical_vram_gb: float
    weights_gb: float
    kv_cache_gb: float
    headroom_gb: float  # activation scratchpads + driver overhead + OS
    projected_total_gb: float
    spillover_risk: bool  # projected_total_gb > physical_vram_gb * _SPILLOVER_RATIO
    recommended_n_ctx: int  # largest n_ctx that fits; 0 if profile too big for card


def project_vram_usage(
    profile_name: str,
    profile_meta: dict,
    n_ctx: int,
    physical_vram_gb: float,
    *,
    kv_quant_mode: str = "f16",
    headroom_gb: float | None = None,
) -> VRAMProjection | None:
    """Compute projected VRAM footprint for the given profile + n_ctx.

    Returns ``None`` when the profile has no ``arch`` block (custom profiles
    or cloud profiles) — callers should treat the result as "cannot project"
    rather than "safe". Same contract as :func:`estimate_max_ctx`.

    The returned :class:`VRAMProjection` is pure data; it never raises.
    ``recommended_n_ctx`` is the largest value that fits under
    ``_SPILLOVER_RATIO * physical_vram_gb``, rounded to a multiple of 1024 —
    the same rounding :func:`estimate_max_ctx` applies.
    """
    arch = profile_meta.get("arch")
    if not arch:
        return None
    try:
        weights_gb = float(arch["weights_gb"])
        n_layers = int(arch["n_layers"])
        n_kv_heads = int(arch["n_kv_heads"])
        head_dim = int(arch["head_dim"])
    except (KeyError, TypeError, ValueError):
        return None

    effective_headroom_gb = headroom_gb if headroom_gb is not None else _dynamic_headroom_gb(weights_gb)

    kv_type_bytes = _KV_TYPE_BYTES.get(kv_quant_mode, 2.0)
    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * kv_type_bytes
    kv_cache_gb = (kv_bytes_per_token * max(n_ctx, 0)) / (1024**3)
    projected_total_gb = weights_gb + kv_cache_gb + effective_headroom_gb
    spillover_threshold_gb = physical_vram_gb * _SPILLOVER_RATIO
    spillover_risk = projected_total_gb > spillover_threshold_gb

    # Recommended n_ctx: use the allowed KV budget (threshold - weights - headroom)
    # not the absolute budget — that's the value that actually fits UNDER the
    # spillover ratio, not right up against it. estimate_max_ctx uses the raw
    # VRAM budget which is what causes the silent spillover in the first place.
    allowed_kv_gb = spillover_threshold_gb - weights_gb - effective_headroom_gb
    if allowed_kv_gb <= 0 or kv_bytes_per_token <= 0:
        recommended_n_ctx = 0
    else:
        max_tokens = int(allowed_kv_gb * (1024**3) / kv_bytes_per_token)
        profile_max = profile_meta.get("n_ctx")
        if isinstance(profile_max, int) and profile_max > 0:
            max_tokens = min(max_tokens, profile_max)
        recommended_n_ctx = max((max_tokens // 1024) * 1024, 0)

    return VRAMProjection(
        profile=profile_name,
        n_ctx=int(n_ctx),
        physical_vram_gb=float(physical_vram_gb),
        weights_gb=weights_gb,
        kv_cache_gb=round(kv_cache_gb, 2),
        headroom_gb=round(effective_headroom_gb, 2),
        projected_total_gb=round(projected_total_gb, 2),
        spillover_risk=spillover_risk,
        recommended_n_ctx=recommended_n_ctx,
    )


def _lookup_fallback_nctx(profile_name: str, vram_gb: float) -> int:
    """Return the measured-safe n_ctx for ``profile_name`` at this VRAM, or 0."""
    rows = _NCTX_SAFE_FALLBACK.get(profile_name)
    if not rows:
        return 0
    for min_vram, n_ctx in rows:
        if vram_gb >= min_vram:
            return n_ctx
    return 0


def _pick_infer_profile(
    vram_gb: float,
    profile_available: ProfileAvailabilityCheck | None = None,
) -> str:
    """Walk the VRAM tier table from largest to smallest, returning the first
    profile that fits the VRAM budget AND passes the availability check.

    Note: this is the legacy "name only" picker, retained because callers
    that don't yet thread n_ctx (e.g. ``maxim doctor``) only need the
    profile name. New code should prefer :func:`_pick_infer_profile_with_ctx`.
    """
    check = profile_available or (lambda _: True)
    for min_vram, profile in _INFER_VRAM_TIERS:
        if vram_gb >= min_vram and check(profile):
            return profile
    # Last tier (smollm) is the universal fallback — return it even if the
    # availability check fails, so callers always get a profile name.
    return _INFER_VRAM_TIERS[-1][1]


def _pick_infer_profile_with_ctx(
    vram_gb: float,
    profile_available: ProfileAvailabilityCheck | None = None,
    *,
    kv_quant_mode: str = "f16",
    min_viable_ctx: int = _MIN_VIABLE_CTX,
) -> tuple[str, int]:
    """Walk the tier table picking the largest profile that produces a viable
    n_ctx for the given VRAM budget.

    Returns ``(profile_name, n_ctx)``. If the formula says 0 for every tier
    above the smollm floor, falls through to the smollm row with n_ctx=0
    (the spawner's default). Callers should treat n_ctx<=0 as "use the
    spawner default" so this function never blocks startup.
    """
    from maxim.models.language.config import _BUILTIN_PROFILES

    check = profile_available or (lambda _: True)
    for min_vram, profile_name in _INFER_VRAM_TIERS:
        if vram_gb < min_vram:
            continue
        if not check(profile_name):
            continue
        profile_data = _BUILTIN_PROFILES.get(profile_name, {})
        formula_ctx = estimate_max_ctx(profile_data, vram_gb, kv_quant_mode=kv_quant_mode)
        fallback_ctx = _lookup_fallback_nctx(profile_name, vram_gb)
        # Be conservative: when both speak, take the SMALLER of the two so
        # measured reality bounds the formula's optimism.
        if formula_ctx > 0 and fallback_ctx > 0:
            n_ctx = min(formula_ctx, fallback_ctx)
        else:
            n_ctx = formula_ctx or fallback_ctx
        if n_ctx >= min_viable_ctx:
            return profile_name, n_ctx
        # Profile fits the tier table but produces an unusable context size
        # at this budget — fall through to the next-smaller tier.
    # Universal fallback (smollm); n_ctx=0 means "spawner picks default".
    return _INFER_VRAM_TIERS[-1][1], 0


def detect_tiers(
    caps: RuntimeCapabilities | None = None,
    *,
    profile_available: ProfileAvailabilityCheck | None = None,
) -> dict[str, LaneConfig]:
    """Auto-detect available tiers based on hardware.

    Delegates VRAM-based profile selection to :func:`_pick_infer_profile`
    (reuses ``_INFER_VRAM_TIERS``). Does NOT duplicate hardware detection —
    :class:`RuntimeCapabilities` is the single source of truth.

    Called at startup by ``LaneBackendManager``. Also called by
    ``maxim doctor`` to report tier availability (``check_tier_detection``).

    Returns:
        Dict of tier name → LaneConfig.  Always includes ``"small"``.
        Includes ``"large"`` when a CUDA GPU is available (VRAM ≥ 4GB).
        Includes ``"medium"`` on Mac MPS or CPU-only boxes with >= 8GB RAM.
    """
    if caps is None:
        from maxim.runtime.capabilities import RuntimeCapabilities as _RC
        from maxim.runtime.capabilities import detect_compute_resources

        has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()
        caps = _RC(has_gpu=has_gpu, gpu_type=gpu_type, vram_gb=vram_gb, ram_gb=ram_gb)

    tiers: dict[str, LaneConfig] = {}

    # Always available: small (CPU, ~2GB RAM)
    tiers["small"] = LaneConfig(
        name="small",
        max_workers=2,
        model_profile="smollm-1.7b-instruct",
        device="cpu",
        n_gpu_layers=0,
    )

    # GPU with enough VRAM → large tier (profile selected by VRAM).
    # Respect --language-model / MAXIM_LLM_PROFILE if set by the user.
    #
    # MPS systems are now accepted into the large tier as long as they
    # report a non-zero vram_gb (which they do via `_mps_effective_vram_gb`
    # on Apple Silicon — see capabilities.py). Previously MPS was hard-
    # excluded here and routed to the medium tier regardless of available
    # memory, which capped a 24GB Mac at mistral-7b. The fallback path
    # below still creates a medium tier for Macs without enough effective
    # VRAM (<4GB) or Intel Macs where MPS reports vram_gb=0.
    env_profile = os.environ.get("MAXIM_LLM_PROFILE", "").strip()
    # gpu_type=None is the "detected a GPU but couldn't identify it" edge
    # case (e.g., torch exception mid-detection). Treat that as
    # unconfigured and fall through to the medium/CPU path rather than
    # routing to the large tier blindly. MPS is an explicit, known type
    # so it passes this check.
    if caps.has_gpu and caps.gpu_type is not None and caps.vram_gb >= 4.0:
        # When the user names a profile via --llm / MAXIM_LLM_PROFILE we
        # honor it but still run estimate_max_ctx() so the resulting n_ctx
        # respects the user's hardware. The "with_ctx" picker is only used
        # for auto-selection.
        if env_profile:
            from maxim.models.language.config import _BUILTIN_PROFILES

            profile = env_profile
            n_ctx = estimate_max_ctx(_BUILTIN_PROFILES.get(profile, {}), caps.vram_gb)
            if n_ctx <= 0:
                n_ctx = _lookup_fallback_nctx(profile, caps.vram_gb)
        else:
            profile, n_ctx = _pick_infer_profile_with_ctx(caps.vram_gb, profile_available=profile_available)
        # MPS uses `device="auto"` so llama.cpp's Metal backend picks the
        # right offload strategy. `requires_gpu=False` on MPS so the
        # worker pool doesn't reserve a CUDA worker that doesn't exist
        # (requires_gpu is historically used for CUDA-specific gating).
        is_mps = caps.gpu_type == "mps"
        tiers["large"] = LaneConfig(
            name="large",
            max_workers=1,
            requires_gpu=not is_mps,
            model_profile=profile,
            device="auto" if is_mps else "gpu",
            n_gpu_layers=-1,
            n_ctx=n_ctx if n_ctx > 0 else None,
        )

    # No large tier (no GPU, unknown GPU type, or effective VRAM < 4GB)
    # → pick a CPU/medium model by RAM. This branch catches Intel Macs
    # with MPS but vram_gb=0.0, low-VRAM discrete cards, and headless
    # servers without a GPU at all.
    if "large" not in tiers:
        m_profile = env_profile or _pick_medium_profile(caps.ram_gb, profile_available)
        if m_profile is not None:
            is_mps = caps.gpu_type == "mps"
            tiers["medium"] = LaneConfig(
                name="medium",
                max_workers=1,
                model_profile=m_profile,
                device="auto" if is_mps else "cpu",
                n_gpu_layers=-1 if is_mps else 0,
            )

    if len(tiers) == 1:
        logger.warning(
            "Only 'small' tier detected (no GPU, %.0fGB RAM). "
            "Agent inference requires --language-model or --cloud-fallback.",
            caps.ram_gb,
        )

    return tiers


def apply_lane_env_overrides(lane_configs: dict[str, LaneConfig]) -> dict[str, LaneConfig]:
    """Patch lane configs with env-var overrides for remote URLs.

    Recognized env vars, per lane name:
      MAXIM_LANE_{NAME}_REMOTE_URL      → remote_url
      MAXIM_LANE_{NAME}_REMOTE_MODEL    → remote_model
      MAXIM_LANE_{NAME}_REMOTE_API_KEY  → remote_api_key
      MAXIM_LANE_{NAME}_TIMEOUT_S       → remote_timeout_s
        (llm_timeout_scalability.md Stage 2 — per-tier inference read
        timeout, strictly positive seconds; falsy/unparseable values are
        ignored and the backend default applies)

    If REMOTE_URL is set on a lane, it supersedes that lane's local profile
    assignment (the backend becomes a remote HTTP client). A missing or empty
    REMOTE_URL leaves the lane unchanged. Names are case-folded (large →
    MAXIM_LANE_LARGE_REMOTE_URL).

    Returns a new dict — does not mutate the input.
    """
    out: dict[str, LaneConfig] = {}
    for name, cfg in lane_configs.items():
        key = name.upper()
        url = os.environ.get(f"MAXIM_LANE_{key}_REMOTE_URL", "").strip()
        if not url:
            out[name] = cfg
            continue
        model = os.environ.get(f"MAXIM_LANE_{key}_REMOTE_MODEL", "").strip() or None
        api_key = os.environ.get(f"MAXIM_LANE_{key}_REMOTE_API_KEY", "").strip() or None
        raw_timeout = os.environ.get(f"MAXIM_LANE_{key}_TIMEOUT_S", "").strip()
        timeout_s: float | None = None
        if raw_timeout:
            try:
                parsed = float(raw_timeout)
                if parsed > 0:
                    timeout_s = parsed
            except (ValueError, TypeError):
                # Silent ignore — the config_loader-side coercion already
                # raises ConfigurationError with a clear message. If the
                # env var is set directly (bypassing the loader), fall
                # back to the backend default rather than bricking the
                # lane.
                timeout_s = None
        out[name] = dataclasses.replace(
            cfg,
            remote_url=url,
            remote_model=model,
            remote_api_key=api_key,
            remote_timeout_s=timeout_s,
        )
    return out


def apply_tier_config_overrides(
    tiers: dict[str, LaneConfig],
    tier_config: dict[str, dict],
) -> dict[str, LaneConfig]:
    """Apply tier overrides from llm.json ``"tiers"`` section.

    Each key in ``tier_config`` is a tier name (``large``, ``medium``, ``small``).
    Values are dicts with optional keys: ``model_profile``, ``device``,
    ``max_workers``, ``n_gpu_layers``.

    Only overrides fields that are present in the config — missing fields
    keep their hardware-detected defaults. Creates new tiers if the key
    doesn't exist yet (e.g., adding a ``medium`` tier via config on a
    two-tier deployment).

    Returns a new dict — does not mutate the input.
    """
    out = dict(tiers)
    for name, overrides in tier_config.items():
        if not isinstance(overrides, dict):
            continue
        base = out.get(name) or LaneConfig(name=name, max_workers=1)
        replacements = {}
        if "model_profile" in overrides:
            replacements["model_profile"] = overrides["model_profile"]
        if "device" in overrides:
            replacements["device"] = overrides["device"]
        if "max_workers" in overrides:
            replacements["max_workers"] = int(overrides["max_workers"])
        if "n_gpu_layers" in overrides:
            replacements["n_gpu_layers"] = int(overrides["n_gpu_layers"])
        if replacements:
            out[name] = dataclasses.replace(base, **replacements)
        elif name not in out:
            out[name] = base
    return out


def load_function_overrides(
    function_config: dict[str, dict],
) -> dict:
    """Load function routing overrides from llm.json ``"functions"`` section.

    Returns a dict of function name → FunctionSpec. Merged with
    DEFAULT_FUNCTIONS by the caller (overrides take precedence).
    """
    from maxim.runtime.function_router import FunctionSpec

    overrides: dict[str, FunctionSpec] = {}
    for name, cfg in function_config.items():
        if not isinstance(cfg, dict):
            continue
        tier = cfg.get("tier")
        if not tier:
            continue
        fallback_raw = cfg.get("fallback", ())
        if isinstance(fallback_raw, list):
            fallback_raw = tuple(fallback_raw)
        overrides[name] = FunctionSpec(
            name=name,
            tier=tier,
            fallback=fallback_raw if isinstance(fallback_raw, tuple) else (),
            description=cfg.get("description", ""),
        )
    return overrides


__all__ = [
    "LaneModelConfig",
    "apply_lane_env_overrides",
    "apply_tier_config_overrides",
    "detect_tiers",
    "load_function_overrides",
]
