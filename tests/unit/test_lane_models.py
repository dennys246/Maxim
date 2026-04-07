"""Tests for capability-driven per-lane LLM profile assignment."""

from __future__ import annotations

from maxim.runtime.capabilities import RuntimeCapabilities
from maxim.runtime.lane_models import (
    LaneModelConfig,
    apply_lane_env_overrides,
    build_lane_model_config,
)
from maxim.runtime.worker_pool import LaneConfig


def _caps(has_gpu: bool, vram_gb: float, ram_gb: float) -> RuntimeCapabilities:
    return RuntimeCapabilities(has_gpu=has_gpu, vram_gb=vram_gb, ram_gb=ram_gb)


class TestInferLaneAssignment:
    def test_large_gpu_gets_13b(self):
        # 16GB (RTX 5080 class) — lands in the >=14 tier
        cfg = build_lane_model_config(_caps(True, 15.9, 32.0))
        assert cfg["infer"].profile == "llama-2-13b-chat"
        assert cfg["infer"].device == "gpu"
        assert cfg["infer"].n_gpu_layers == -1

    def test_midrange_gpu_gets_8b(self):
        # 10GB card (3080 class)
        cfg = build_lane_model_config(_caps(True, 10.0, 16.0))
        assert cfg["infer"].profile == "llama-3-8b-instruct"
        assert cfg["infer"].device == "gpu"

    def test_small_gpu_gets_7b(self):
        # 6GB card
        cfg = build_lane_model_config(_caps(True, 6.0, 16.0))
        assert cfg["infer"].profile == "mistral-7b-instruct-v0.2"
        assert cfg["infer"].device == "gpu"

    def test_tiny_gpu_uses_smollm_on_gpu(self):
        # 2GB VRAM — below the 7B tier, still enough for SmolLM on GPU
        cfg = build_lane_model_config(_caps(True, 2.0, 16.0))
        assert cfg["infer"].profile == "smollm-1.7b-instruct"
        assert cfg["infer"].device == "gpu"
        assert cfg["infer"].n_gpu_layers == -1

    def test_no_gpu_uses_smollm_cpu(self):
        cfg = build_lane_model_config(_caps(False, 0.0, 16.0))
        assert cfg["infer"].profile == "smollm-1.7b-instruct"
        assert cfg["infer"].device == "cpu"
        assert cfg["infer"].n_gpu_layers == 0


class TestReviewLaneAssignment:
    def test_review_gets_own_cpu_model_when_ram_allows(self):
        cfg = build_lane_model_config(_caps(True, 15.9, 32.0))
        assert cfg["review"].profile == "smollm-1.7b-instruct"
        assert cfg["review"].device == "cpu"
        assert cfg["review"].n_gpu_layers == 0
        # And it's a distinct config from infer
        assert cfg["review"] is not cfg["infer"]

    def test_review_shares_infer_when_ram_tight(self):
        # 2GB RAM — can't afford a second model
        cfg = build_lane_model_config(_caps(True, 15.9, 2.0))
        assert cfg["review"] is cfg["infer"]

    def test_review_ram_threshold_exact(self):
        # At the threshold (4GB) the review lane still gets its own model
        cfg = build_lane_model_config(_caps(False, 0.0, 4.0))
        assert cfg["review"] is not cfg["infer"]
        assert cfg["review"].profile == "smollm-1.7b-instruct"


class TestStructure:
    def test_returns_both_lanes(self):
        cfg = build_lane_model_config(_caps(True, 15.9, 32.0))
        assert set(cfg.keys()) == {"infer", "review"}

    def test_record_lane_intentionally_omitted(self):
        # record lane is fire-and-forget persistence — no LLM assignment
        cfg = build_lane_model_config(_caps(True, 15.9, 32.0))
        assert "record" not in cfg

    def test_assignments_are_frozen(self):
        cfg = build_lane_model_config(_caps(True, 15.9, 32.0))
        assert isinstance(cfg["infer"], LaneModelConfig)
        import dataclasses

        assert dataclasses.is_dataclass(cfg["infer"])


class TestAvailabilityFilter:
    def test_skips_unavailable_tier_profiles(self):
        """When the VRAM-selected profile isn't available, fall through the tiers."""
        # User has 16GB VRAM (would pick llama-2-13b-chat) but only mistral-7b exists
        available = {"mistral-7b-instruct-v0.2", "smollm-1.7b-instruct"}
        cfg = build_lane_model_config(
            _caps(True, 15.9, 32.0),
            profile_available=lambda p: p in available,
        )
        assert cfg["infer"].profile == "mistral-7b-instruct-v0.2"
        assert cfg["infer"].device == "gpu"

    def test_falls_back_to_smollm_when_nothing_else_available(self):
        """If only smollm exists, even a 16GB GPU gets smollm."""
        available = {"smollm-1.7b-instruct"}
        cfg = build_lane_model_config(
            _caps(True, 15.9, 32.0),
            profile_available=lambda p: p in available,
        )
        assert cfg["infer"].profile == "smollm-1.7b-instruct"
        # Still GPU placement since has_gpu + vram fit
        assert cfg["infer"].device == "gpu"

    def test_review_lane_shares_infer_when_review_profile_unavailable(self):
        """If smollm is missing, review shares infer instead of dangling."""
        available = {"mistral-7b-instruct-v0.2"}
        cfg = build_lane_model_config(
            _caps(True, 8.0, 32.0),
            profile_available=lambda p: p in available,
        )
        assert cfg["review"] is cfg["infer"]

    def test_no_filter_falls_back_to_trust_all(self):
        """Backward compat: omitting the filter preserves old behavior."""
        cfg = build_lane_model_config(_caps(True, 15.9, 32.0))
        assert cfg["infer"].profile == "llama-2-13b-chat"  # original pick

    def test_all_unavailable_still_returns_smollm(self):
        """Edge case: when NOTHING is available, return smollm as final fallback."""
        cfg = build_lane_model_config(
            _caps(True, 15.9, 32.0),
            profile_available=lambda _: False,
        )
        # Infer gets smollm as last-resort fallback (even though it too is unavailable)
        assert cfg["infer"].profile == "smollm-1.7b-instruct"
        # Review shares infer because smollm didn't pass the availability check
        assert cfg["review"] is cfg["infer"]


class TestApplyLaneEnvOverrides:
    def _lanes(self) -> dict[str, LaneConfig]:
        return {
            "infer": LaneConfig(name="infer", max_workers=1, model_profile="mistral-7b-instruct-v0.2"),
            "review": LaneConfig(name="review", max_workers=1, model_profile="smollm-1.7b-instruct"),
        }

    def test_no_env_vars_returns_unchanged(self, monkeypatch):
        monkeypatch.delenv("MAXIM_LANE_INFER_REMOTE_URL", raising=False)
        monkeypatch.delenv("MAXIM_LANE_REVIEW_REMOTE_URL", raising=False)
        lanes = self._lanes()
        out = apply_lane_env_overrides(lanes)
        assert out["infer"].remote_url is None
        assert out["infer"].model_profile == "mistral-7b-instruct-v0.2"
        assert out["review"].remote_url is None

    def test_remote_url_populates_lane(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_INFER_REMOTE_URL", "http://127.0.0.1:8000/v1")
        monkeypatch.setenv("MAXIM_LANE_INFER_REMOTE_MODEL", "mistral-7b")
        monkeypatch.delenv("MAXIM_LANE_REVIEW_REMOTE_URL", raising=False)
        out = apply_lane_env_overrides(self._lanes())
        assert out["infer"].remote_url == "http://127.0.0.1:8000/v1"
        assert out["infer"].remote_model == "mistral-7b"
        # Other lanes untouched
        assert out["review"].remote_url is None

    def test_remote_api_key_optional(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_INFER_REMOTE_URL", "http://127.0.0.1:8000/v1")
        monkeypatch.delenv("MAXIM_LANE_INFER_REMOTE_API_KEY", raising=False)
        out = apply_lane_env_overrides(self._lanes())
        assert out["infer"].remote_api_key is None

    def test_remote_api_key_propagated(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_INFER_REMOTE_URL", "https://api.anthropic.com/v1")
        monkeypatch.setenv("MAXIM_LANE_INFER_REMOTE_API_KEY", "sk-ant-test")
        out = apply_lane_env_overrides(self._lanes())
        assert out["infer"].remote_api_key == "sk-ant-test"

    def test_does_not_mutate_input(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_INFER_REMOTE_URL", "http://127.0.0.1:8000/v1")
        lanes = self._lanes()
        apply_lane_env_overrides(lanes)
        # Original dict values are unchanged
        assert lanes["infer"].remote_url is None

    def test_empty_url_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_INFER_REMOTE_URL", "   ")
        out = apply_lane_env_overrides(self._lanes())
        assert out["infer"].remote_url is None
