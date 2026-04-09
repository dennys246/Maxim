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
        """Dual GPU server: high VRAM → large with best profile."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA A100", vram_gb=40.0, ram_gb=64.0)
        tiers = detect_tiers(caps)
        assert "large" in tiers
        assert tiers["large"].model_profile == "llama-2-13b-chat"  # top of VRAM tiers

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
