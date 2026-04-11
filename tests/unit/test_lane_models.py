"""Tests for capability-driven per-lane LLM profile assignment."""

from __future__ import annotations

from maxim.runtime.capabilities import RuntimeCapabilities
from maxim.runtime.lane_models import (
    apply_lane_env_overrides,
)
from maxim.runtime.worker_pool import LaneConfig


def _caps(has_gpu: bool, vram_gb: float, ram_gb: float) -> RuntimeCapabilities:
    return RuntimeCapabilities(has_gpu=has_gpu, vram_gb=vram_gb, ram_gb=ram_gb)


class TestApplyLaneEnvOverrides:
    def _lanes(self) -> dict[str, LaneConfig]:
        return {
            "large": LaneConfig(name="large", max_workers=1, model_profile="mistral-7b-instruct-v0.2"),
            "small": LaneConfig(name="small", max_workers=1, model_profile="smollm-1.7b-instruct"),
        }

    def test_no_env_vars_returns_unchanged(self, monkeypatch):
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        monkeypatch.delenv("MAXIM_LANE_SMALL_REMOTE_URL", raising=False)
        lanes = self._lanes()
        out = apply_lane_env_overrides(lanes)
        assert out["large"].remote_url is None
        assert out["large"].model_profile == "mistral-7b-instruct-v0.2"
        assert out["small"].remote_url is None

    def test_remote_url_populates_lane(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://127.0.0.1:8000/v1")
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_MODEL", "mistral-7b")
        monkeypatch.delenv("MAXIM_LANE_SMALL_REMOTE_URL", raising=False)
        out = apply_lane_env_overrides(self._lanes())
        assert out["large"].remote_url == "http://127.0.0.1:8000/v1"
        assert out["large"].remote_model == "mistral-7b"
        # Other lanes untouched
        assert out["small"].remote_url is None

    def test_remote_api_key_optional(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://127.0.0.1:8000/v1")
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", raising=False)
        out = apply_lane_env_overrides(self._lanes())
        assert out["large"].remote_api_key is None

    def test_remote_api_key_propagated(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "https://api.anthropic.com/v1")
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", "sk-ant-test")
        out = apply_lane_env_overrides(self._lanes())
        assert out["large"].remote_api_key == "sk-ant-test"

    def test_does_not_mutate_input(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://127.0.0.1:8000/v1")
        lanes = self._lanes()
        apply_lane_env_overrides(lanes)
        # Original dict values are unchanged
        assert lanes["large"].remote_url is None

    def test_empty_url_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "   ")
        out = apply_lane_env_overrides(self._lanes())
        assert out["large"].remote_url is None


# ─── detect_tiers ────────────────────────────────────────────────────────────

from maxim.runtime.lane_models import detect_tiers


class TestDetectTiers:
    """Tests for hardware-based tier auto-detection."""

    def test_rtx_5080_gets_large_and_small(self):
        """RTX 5080: 16GB VRAM → large (GPU) + small (CPU)."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA RTX 5080", vram_gb=15.9, ram_gb=16.0)
        tiers = detect_tiers(caps)
        assert "large" in tiers
        assert "small" in tiers
        assert "medium" not in tiers
        assert tiers["large"].requires_gpu is True
        assert tiers["large"].device == "gpu"
        assert tiers["small"].max_workers == 2
        assert tiers["small"].device == "cpu"

    def test_large_tier_profile_selected_by_vram(self):
        """VRAM tier table should be used to pick the profile."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA RTX 3080", vram_gb=10.0, ram_gb=16.0)
        tiers = detect_tiers(caps)
        assert tiers["large"].model_profile == "llama-3-8b-instruct"

    def test_mac_mps_gets_medium_and_small(self):
        """Mac with MPS: unified memory → medium (auto) + small (CPU)."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=0.0, ram_gb=24.0)
        tiers = detect_tiers(caps)
        assert "medium" in tiers
        assert "small" in tiers
        assert "large" not in tiers
        assert tiers["medium"].device == "auto"
        assert tiers["medium"].model_profile == "mistral-7b-instruct-v0.2"

    def test_cpu_only_16gb_gets_medium_and_small(self):
        """CPU-only box with 16GB RAM → medium (CPU) + small (CPU)."""
        caps = RuntimeCapabilities(has_gpu=False, vram_gb=0.0, ram_gb=16.0)
        tiers = detect_tiers(caps)
        assert "medium" in tiers
        assert "small" in tiers
        assert "large" not in tiers
        assert tiers["medium"].device == "cpu"

    def test_low_ram_no_gpu_only_small(self):
        """Raspberry Pi: 4GB RAM, no GPU → only small."""
        caps = RuntimeCapabilities(has_gpu=False, vram_gb=0.0, ram_gb=4.0)
        tiers = detect_tiers(caps)
        assert list(tiers.keys()) == ["small"]

    def test_small_gpu_below_4gb_no_large(self):
        """GPU with < 4GB VRAM → no large tier (not enough for any model)."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA GTX 1050", vram_gb=3.5, ram_gb=8.0)
        tiers = detect_tiers(caps)
        assert "large" not in tiers

    def test_small_tier_always_present(self):
        """Small tier should always be present regardless of hardware."""
        for caps in [
            RuntimeCapabilities(has_gpu=True, vram_gb=16.0, ram_gb=32.0),
            RuntimeCapabilities(has_gpu=False, vram_gb=0.0, ram_gb=2.0),
            RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=0.0, ram_gb=24.0),
        ]:
            tiers = detect_tiers(caps)
            assert "small" in tiers
            assert tiers["small"].model_profile == "smollm-1.7b-instruct"

    def test_profile_available_filter(self):
        """Profile availability filter should be passed through to _pick_infer_profile."""
        available = {"mistral-7b-instruct-v0.2", "smollm-1.7b-instruct"}
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA RTX 5080", vram_gb=15.9, ram_gb=16.0)
        tiers = detect_tiers(caps, profile_available=lambda p: p in available)
        # Should pick mistral-7b (next tier down) since llama-2-13b isn't available
        assert tiers["large"].model_profile == "mistral-7b-instruct-v0.2"

    def test_dual_gpu_large_tier(self):
        """Dual GPU server: high VRAM → large with best profile.

        After P4a, qwen2.5-14b-instruct is the top of the VRAM tier
        table (at the 16 GB threshold), so a 40 GB A100 picks it.
        Previously this test asserted llama-2-13b-chat.
        """
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA A100", vram_gb=40.0, ram_gb=64.0)
        tiers = detect_tiers(caps)
        assert "large" in tiers
        assert tiers["large"].model_profile == "qwen2.5-14b-instruct"  # top of VRAM tiers

    # ── Boundary value tests ──

    def test_boundary_vram_exactly_4gb(self):
        """VRAM exactly 4.0GB → large tier (inclusive boundary)."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA GTX 1650", vram_gb=4.0, ram_gb=8.0)
        tiers = detect_tiers(caps)
        assert "large" in tiers

    def test_boundary_ram_exactly_8gb(self):
        """RAM exactly 8GB → medium tier (>= 8, inclusive)."""
        caps = RuntimeCapabilities(has_gpu=False, vram_gb=0.0, ram_gb=8.0)
        tiers = detect_tiers(caps)
        assert "medium" in tiers

    def test_boundary_ram_just_below_8gb(self):
        """RAM 7.9GB → only small (below 8GB threshold)."""
        caps = RuntimeCapabilities(has_gpu=False, vram_gb=0.0, ram_gb=7.9)
        tiers = detect_tiers(caps)
        assert list(tiers.keys()) == ["small"]

    # ── gpu_type edge cases ──

    def test_gpu_type_none_no_large_tier(self):
        """has_gpu=True but gpu_type=None → no large tier (unknown GPU type)."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type=None, vram_gb=16.0, ram_gb=16.0)
        tiers = detect_tiers(caps)
        assert "large" not in tiers
        # Falls through to CPU path: has_gpu=True but no large/medium yet,
        # ram >= 8 check fires because neither large nor medium exists
        assert "medium" in tiers

    def test_mps_with_high_ram_no_duplicate_medium(self):
        """MPS + 24GB RAM → only ONE medium tier (no double creation)."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=0.0, ram_gb=24.0)
        tiers = detect_tiers(caps)
        tier_names = list(tiers.keys())
        assert tier_names.count("medium") == 1  # dict keys are unique anyway
        assert tiers["medium"].device == "auto"  # MPS path, not CPU path

    # ── RAM-based medium tier model selection ──

    def test_mac_8gb_gets_phi3_medium(self):
        """Mac with 8GB → medium tier uses phi-3-mini (fits in RAM)."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=0.0, ram_gb=8.0)
        tiers = detect_tiers(caps)
        assert "medium" in tiers
        assert tiers["medium"].model_profile == "phi-3-mini-4k-instruct"

    def test_mac_16gb_gets_mistral_medium(self):
        """Mac with 16GB → medium tier uses mistral-7b (plenty of headroom)."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=0.0, ram_gb=16.0)
        tiers = detect_tiers(caps)
        assert "medium" in tiers
        assert tiers["medium"].model_profile == "mistral-7b-instruct-v0.2"

    def test_cpu_8gb_gets_phi3_medium(self):
        """CPU-only 8GB → medium tier uses phi-3-mini."""
        caps = RuntimeCapabilities(has_gpu=False, vram_gb=0.0, ram_gb=8.0)
        tiers = detect_tiers(caps)
        assert "medium" in tiers
        assert tiers["medium"].model_profile == "phi-3-mini-4k-instruct"

    def test_cpu_12gb_gets_phi3_medium(self):
        """CPU-only 12GB → phi-3-mini (below 16GB threshold for mistral)."""
        caps = RuntimeCapabilities(has_gpu=False, vram_gb=0.0, ram_gb=12.0)
        tiers = detect_tiers(caps)
        assert "medium" in tiers
        assert tiers["medium"].model_profile == "phi-3-mini-4k-instruct"

    def test_mac_6gb_no_medium(self):
        """Mac with 6GB → no medium tier (too little RAM for any model)."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=0.0, ram_gb=6.0)
        tiers = detect_tiers(caps)
        assert "medium" not in tiers
        assert list(tiers.keys()) == ["small"]

    def test_small_gpu_high_ram_gets_medium(self):
        """GPU with < 4GB VRAM + 16GB RAM → medium (CPU) + small."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA GTX 1050", vram_gb=3.5, ram_gb=16.0)
        tiers = detect_tiers(caps)
        assert "large" not in tiers
        assert "medium" in tiers
        assert tiers["medium"].device == "cpu"

    # ── P3: MPS into large tier (peer_leader_flexibility_plan) ──

    def test_mps_24gb_mac_gets_large_tier(self):
        """24GB Mac with 18GB effective VRAM (from P2 unified memory
        detection) should land in the large tier, not medium. This
        is the headline regression that P3 fixes: before this change,
        Apple Silicon was hard-excluded from the large tier and
        capped at mistral-7b regardless of memory."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=18.0, ram_gb=24.0)
        tiers = detect_tiers(caps)
        assert "large" in tiers
        assert "medium" not in tiers

    def test_mps_large_tier_uses_auto_device(self):
        """MPS large tier uses device='auto' so llama.cpp's Metal
        backend picks the right offload strategy. CUDA uses device='gpu'."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=18.0, ram_gb=24.0)
        tiers = detect_tiers(caps)
        assert tiers["large"].device == "auto"
        assert tiers["large"].n_gpu_layers == -1

    def test_mps_large_tier_requires_gpu_false(self):
        """MPS sets requires_gpu=False because that flag is
        historically used for CUDA-specific worker-pool gating.
        Setting it True on MPS would reserve a CUDA worker that
        doesn't exist and starve the lane."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=18.0, ram_gb=24.0)
        tiers = detect_tiers(caps)
        assert tiers["large"].requires_gpu is False

    def test_cuda_large_tier_unchanged(self):
        """Regression check: CUDA path still sets requires_gpu=True
        and device='gpu'. The P3 change only affects the MPS path."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA RTX 5080", vram_gb=16.0, ram_gb=64.0)
        tiers = detect_tiers(caps)
        assert tiers["large"].device == "gpu"
        assert tiers["large"].requires_gpu is True

    def test_mps_vram_below_4gb_falls_to_medium(self):
        """MPS with low effective VRAM (an 8GB Mac at 0.75 headroom =
        6GB effective — wait that's actually above 4. Test the edge
        case of a hypothetical tiny MPS system.)"""
        # 4GB Mac at 0.75 headroom = 3GB effective → below threshold
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=3.0, ram_gb=4.0)
        tiers = detect_tiers(caps)
        assert "large" not in tiers
        # Falls through to the medium path but 4GB RAM is below the
        # _MEDIUM_RAM_TIERS threshold (8GB minimum), so no medium either
        assert "medium" not in tiers
        # Only small remains
        assert list(tiers.keys()) == ["small"]

    def test_mps_intel_mac_falls_to_medium(self):
        """Intel Macs running legacy PyTorch MPS get vram_gb=0.0 from
        capabilities.py (the unified-memory assumption doesn't hold
        on Intel). They should land in the medium tier, not large."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=0.0, ram_gb=32.0)
        tiers = detect_tiers(caps)
        assert "large" not in tiers
        assert "medium" in tiers
        assert tiers["medium"].device == "auto"  # MPS path preserves auto

    def test_mps_large_tier_picks_qwen_14b_by_default(self):
        """18 GB effective VRAM on Mac (24 GB total at 0.75 headroom)
        should pick qwen2.5-14b-instruct from the 16 GB+ tier added
        in P4a. Earlier this test expected llama-2-13b-chat from the
        pre-P4a table."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=18.0, ram_gb=24.0)
        tiers = detect_tiers(caps)
        assert tiers["large"].model_profile == "qwen2.5-14b-instruct"

    def test_mps_medium_fallback_uses_auto_device(self):
        """When an MPS system falls through to the medium tier (Intel
        Mac with vram_gb=0.0, or Apple Silicon with effective VRAM
        below 4GB but enough RAM for a medium model), the medium
        lane should still use device='auto' and n_gpu_layers=-1 so
        llama.cpp can offload to Metal if possible."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=0.0, ram_gb=16.0)
        tiers = detect_tiers(caps)
        assert "medium" in tiers
        assert tiers["medium"].device == "auto"
        assert tiers["medium"].n_gpu_layers == -1

    def test_cpu_medium_fallback_uses_cpu_device(self):
        """Non-MPS, non-CUDA systems fall to the medium tier with
        device='cpu' and n_gpu_layers=0."""
        caps = RuntimeCapabilities(has_gpu=False, vram_gb=0.0, ram_gb=16.0)
        tiers = detect_tiers(caps)
        assert "medium" in tiers
        assert tiers["medium"].device == "cpu"
        assert tiers["medium"].n_gpu_layers == 0

    # ── P4a: qwen2.5-14b row at 16 GB threshold ──

    def test_p4a_boundary_vram_exactly_16gb_picks_qwen(self):
        """VRAM exactly 16 GB → qwen2.5-14b-instruct (inclusive boundary)."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA RTX 5080", vram_gb=16.0, ram_gb=64.0)
        tiers = detect_tiers(caps)
        assert tiers["large"].model_profile == "qwen2.5-14b-instruct"

    def test_p4a_vram_just_below_16gb_picks_llama13b(self):
        """15.9 GB → llama-2-13b-chat (below the new 16 GB threshold)."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA RTX 5080", vram_gb=15.9, ram_gb=64.0)
        tiers = detect_tiers(caps)
        assert tiers["large"].model_profile == "llama-2-13b-chat"

    def test_p4a_profile_available_filter_falls_through_qwen(self):
        """If qwen2.5-14b-instruct is not available (not downloaded)
        but llama-2-13b-chat is, the walk falls through to the next
        tier that satisfies the availability callback."""
        available = {"llama-2-13b-chat", "smollm-1.7b-instruct"}
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA A100", vram_gb=40.0, ram_gb=64.0)
        tiers = detect_tiers(caps, profile_available=lambda p: p in available)
        assert tiers["large"].model_profile == "llama-2-13b-chat"

    def test_p4a_mps_24gb_picks_qwen(self):
        """Apple Silicon 24 GB Mac (18 GB effective VRAM at default
        headroom) picks qwen2.5-14b-instruct from the 16 GB tier.

        This is the headline change the P4 wave enables on Mac
        hardware: previously capped at mistral-7b via the medium
        fallback, now gets the 14B model at the large tier.
        """
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=18.0, ram_gb=24.0)
        tiers = detect_tiers(caps)
        assert "large" in tiers
        assert tiers["large"].model_profile == "qwen2.5-14b-instruct"


# ─── apply_tier_config_overrides ──────────────────────────────────────────

from maxim.runtime.lane_models import apply_tier_config_overrides


class TestApplyTierConfigOverrides:
    def _base_tiers(self) -> dict[str, LaneConfig]:
        return {
            "large": LaneConfig(name="large", max_workers=1, model_profile="llama-2-13b-chat", device="gpu"),
            "small": LaneConfig(name="small", max_workers=2, model_profile="smollm-1.7b-instruct", device="cpu"),
        }

    def test_override_model_profile(self):
        tiers = apply_tier_config_overrides(
            self._base_tiers(),
            {
                "large": {"model_profile": "qwen2.5-14b"},
            },
        )
        assert tiers["large"].model_profile == "qwen2.5-14b"
        assert tiers["large"].device == "gpu"  # Unchanged

    def test_override_max_workers(self):
        tiers = apply_tier_config_overrides(
            self._base_tiers(),
            {
                "small": {"max_workers": 3},
            },
        )
        assert tiers["small"].max_workers == 3

    def test_add_new_tier(self):
        tiers = apply_tier_config_overrides(
            self._base_tiers(),
            {
                "medium": {"model_profile": "mistral-7b", "device": "cpu"},
            },
        )
        assert "medium" in tiers
        assert tiers["medium"].model_profile == "mistral-7b"

    def test_does_not_mutate_input(self):
        base = self._base_tiers()
        apply_tier_config_overrides(base, {"large": {"model_profile": "new"}})
        assert base["large"].model_profile == "llama-2-13b-chat"

    def test_empty_config_is_noop(self):
        base = self._base_tiers()
        result = apply_tier_config_overrides(base, {})
        assert result["large"].model_profile == "llama-2-13b-chat"


# ─── load_function_overrides ──────────────────────────────────────────────

from maxim.runtime.lane_models import load_function_overrides


class TestLoadFunctionOverrides:
    def test_basic_override(self):
        overrides = load_function_overrides(
            {
                "fear_review": {"tier": "medium", "fallback": ["large"]},
            }
        )
        assert "fear_review" in overrides
        assert overrides["fear_review"].tier == "medium"
        assert overrides["fear_review"].fallback == ("large",)

    def test_empty_fallback(self):
        overrides = load_function_overrides(
            {
                "custom_func": {"tier": "small"},
            }
        )
        assert overrides["custom_func"].fallback == ()

    def test_skips_entries_without_tier(self):
        overrides = load_function_overrides(
            {
                "bad_entry": {"description": "no tier set"},
            }
        )
        assert "bad_entry" not in overrides

    def test_skips_non_dict_entries(self):
        overrides = load_function_overrides(
            {
                "bad": "not a dict",
            }
        )
        assert "bad" not in overrides
