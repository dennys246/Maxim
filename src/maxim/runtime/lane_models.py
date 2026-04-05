"""Capability-driven per-lane LLM profile assignment (multi-LLM Phase 2).

Maps detected compute resources (VRAM, RAM) to concrete LLM profile choices for
each WorkerPool lane. Used by LaneBackendManager (Phase 3) to decide which
backend to instantiate per lane.

This module only builds the *configuration* — it does not load models or create
backends. That happens in Phase 3.
"""
from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from dataclasses import dataclass

from maxim.runtime.capabilities import RuntimeCapabilities
from maxim.runtime.worker_pool import LaneConfig

ProfileAvailabilityCheck = Callable[[str], bool]


@dataclass(frozen=True)
class LaneModelConfig:
    """Resolved LLM assignment for a single WorkerPool lane."""

    profile: str           # Matches a key in BUILTIN_PROFILES
    device: str = "auto"   # "gpu" | "cpu" | "auto"
    n_gpu_layers: int = -1  # -1 = all on GPU, 0 = CPU only


# VRAM tiers for the "infer" lane (the hot-path LLM that benefits most from GPU).
# Each tier names an existing profile from BUILTIN_PROFILES. Extend this table
# when new profiles are added (e.g., a 14B / 24B GGUF for >16GB cards).
#
# Tiers are inclusive-lower, exclusive-upper: a 15.9GB card lands in the >=8 tier.
_INFER_VRAM_TIERS: tuple[tuple[float, str], ...] = (
    (14.0, "llama-2-13b-chat"),        # 13B Q4 ~8GB, leaves headroom on 16GB cards
    (8.0, "llama-3-8b-instruct"),      # 8B Q4 ~5GB
    (4.0, "mistral-7b-instruct-v0.2"),  # 7B Q4 ~4.5GB
    (0.0, "smollm-1.7b-instruct"),      # 1.7B Q4 ~1GB, CPU fallback
)

# RAM threshold (GB) below which the review lane shares the infer backend
# instead of loading its own CPU model. SmolLM Q4 needs ~2GB resident + overhead.
_REVIEW_MIN_RAM_GB = 4.0

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


def build_lane_model_config(
    caps: RuntimeCapabilities,
    *,
    profile_available: ProfileAvailabilityCheck | None = None,
) -> dict[str, LaneModelConfig]:
    """Map WorkerPool lanes to LLM profiles based on detected hardware.

    Returns a dict keyed by lane name (`infer`, `review`). The `record` lane
    does no LLM work and is intentionally omitted — it stays with whatever
    backend the caller provides as default.

    Decisions:
    - GPU present  → infer lane gets a GPU profile sized to VRAM.
    - No GPU       → infer lane falls back to SmolLM on CPU.
    - Enough RAM   → review lane gets its own CPU-side SmolLM.
    - Tight RAM    → review lane falls back to the infer profile (shared backend).

    The effective policy respects CUDA_VISIBLE_DEVICES: if CUDA was hidden by
    the Blackwell workaround, `caps.has_gpu` should already be False and we'll
    pick the CPU fallback.

    Args:
        profile_available: optional callback (profile_name) -> bool that filters
            the tier search. Used to skip profiles whose GGUF files aren't
            downloaded. Tests can omit it for a "trust-all" behavior.
    """
    check = profile_available or (lambda _: True)
    assignments: dict[str, LaneModelConfig] = {}

    if caps.has_gpu and caps.vram_gb >= _INFER_VRAM_TIERS[-1][0]:
        infer_profile = _pick_infer_profile(caps.vram_gb, profile_available=check)
        assignments["infer"] = LaneModelConfig(
            profile=infer_profile,
            device="gpu",
            n_gpu_layers=-1,
        )
    else:
        assignments["infer"] = LaneModelConfig(
            profile="smollm-1.7b-instruct",
            device="cpu",
            n_gpu_layers=0,
        )

    if caps.ram_gb >= _REVIEW_MIN_RAM_GB and check(_REVIEW_PROFILE):
        assignments["review"] = LaneModelConfig(
            profile=_REVIEW_PROFILE,
            device="cpu",
            n_gpu_layers=0,
        )
    else:
        # RAM is tight OR review profile unavailable — share the infer backend
        # rather than loading a second model (or one that doesn't exist).
        assignments["review"] = assignments["infer"]

    return assignments


def apply_lane_env_overrides(lane_configs: dict[str, LaneConfig]) -> dict[str, LaneConfig]:
    """Patch lane configs with env-var overrides for remote URLs.

    Recognized env vars, per lane name:
      MAXIM_LANE_{NAME}_REMOTE_URL      → remote_url
      MAXIM_LANE_{NAME}_REMOTE_MODEL    → remote_model
      MAXIM_LANE_{NAME}_REMOTE_API_KEY  → remote_api_key

    If REMOTE_URL is set on a lane, it supersedes that lane's local profile
    assignment (the backend becomes a remote HTTP client). A missing or empty
    REMOTE_URL leaves the lane unchanged. Names are case-folded (infer →
    MAXIM_LANE_INFER_REMOTE_URL).

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


__all__ = ["LaneModelConfig", "build_lane_model_config", "apply_lane_env_overrides"]
