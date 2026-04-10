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
    (14.0, "llama-2-13b-chat"),  # 13B Q4 ~8GB, leaves headroom on 16GB cards
    (8.0, "llama-3-8b-instruct"),  # 8B Q4 ~5GB
    (4.0, "mistral-7b-instruct-v0.2"),  # 7B Q4 ~4.5GB
    (0.0, "smollm-1.7b-instruct"),  # 1.7B Q4 ~1GB, CPU fallback
)

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


def _pick_infer_profile(
    vram_gb: float,
    profile_available: ProfileAvailabilityCheck | None = None,
) -> str:
    """Walk the VRAM tier table from largest to smallest, returning the first
    profile that fits the VRAM budget AND passes the availability check."""
    check = profile_available or (lambda _: True)
    for min_vram, profile in _INFER_VRAM_TIERS:
        if vram_gb >= min_vram and check(profile):
            return profile
    # Last tier (smollm) is the universal fallback — return it even if the
    # availability check fails, so callers always get a profile name.
    return _INFER_VRAM_TIERS[-1][1]


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

    # CUDA GPU with enough VRAM → large tier (profile selected by VRAM)
    # Respect --language-model / MAXIM_LLM_PROFILE if set by the user.
    env_profile = os.environ.get("MAXIM_LLM_PROFILE", "").strip()
    if caps.has_gpu and caps.gpu_type not in ("mps", None) and caps.vram_gb >= 4.0:
        profile = env_profile or _pick_infer_profile(caps.vram_gb, profile_available=profile_available)
        tiers["large"] = LaneConfig(
            name="large",
            max_workers=1,
            requires_gpu=True,
            model_profile=profile,
            device="gpu",
            n_gpu_layers=-1,
        )
    elif caps.has_gpu and caps.gpu_type == "mps":
        # Mac with MPS: unified memory — pick model by RAM (shared with GPU)
        m_profile = env_profile or _pick_medium_profile(caps.ram_gb, profile_available)
        if m_profile is not None:
            tiers["medium"] = LaneConfig(
                name="medium",
                max_workers=1,
                model_profile=m_profile,
                device="auto",
            )

    # No GPU (or GPU with unknown type) — pick CPU model by RAM
    if "medium" not in tiers and "large" not in tiers:
        m_profile = env_profile or _pick_medium_profile(caps.ram_gb, profile_available)
        if m_profile is not None:
            tiers["medium"] = LaneConfig(
                name="medium",
                max_workers=1,
                model_profile=m_profile,
                device="cpu",
                n_gpu_layers=0,
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
        out[name] = dataclasses.replace(
            cfg,
            remote_url=url,
            remote_model=model,
            remote_api_key=api_key,
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
