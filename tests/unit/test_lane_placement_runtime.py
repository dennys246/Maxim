"""Phase 3c of lane_capability_placement_split.md — runtime producer (config.json
placement → LaneConfig.placement), multi-element tail-injection, and CLI
re-expression of --cloud-lane / --cloud-fallback as placement edits.
"""

from __future__ import annotations

from unittest.mock import patch

import maxim.runtime.config_loader as config_loader
from maxim.runtime.config_loader import (
    LanesConfigSection,
    LaneTierConfig,
    LaneTierPlacement,
    MaximConfig,
)
from maxim.runtime.lane_backends import (
    LaneBackendManager,
    _apply_config_placements,
    _config_placement_to_runtime,
    _placement_entry_to_provider,
)
from maxim.runtime.worker_pool import LaneConfig, Origin, ProviderPlacement


# ─── producer: config.json placement → LaneConfig.placement ─────────────────


class TestConfigPlacementProducer:
    def test_single_entry_resolve(self):
        e = LaneTierPlacement(origin="peer", url="http://10.0.0.9:8100/v1", model="leader")
        p = _config_placement_to_runtime(e, where="x")
        assert p.origin is Origin.PEER
        assert p.url == "http://10.0.0.9:8100/v1"
        assert p.model == "leader"

    def test_api_key_ref_resolved_from_file(self, tmp_path):
        key_file = tmp_path / "key"
        key_file.write_text("sk-secret\n")
        e = LaneTierPlacement(origin="cloud", model="claude-sonnet", api_key_ref=str(key_file))
        p = _config_placement_to_runtime(e, where="x")
        assert p.api_key == "sk-secret"  # resolved + stripped

    def test_producer_sets_placement_from_config(self, monkeypatch):
        cfg = MaximConfig(
            lanes=LanesConfigSection(
                large=LaneTierConfig(placement=(LaneTierPlacement(origin="local", model="mistral-7b"),))
            )
        )
        monkeypatch.setattr(config_loader, "load_config", lambda *a, **k: cfg)
        lane_configs = {"large": LaneConfig(name="large", max_workers=1, model_profile="smollm")}
        out = _apply_config_placements(lane_configs, None)
        assert len(out["large"].placement) == 1
        assert out["large"].placement[0].origin is Origin.LOCAL
        assert out["large"].placement[0].model == "mistral-7b"

    def test_producer_leaves_unconfigured_tiers_untouched(self, monkeypatch):
        cfg = MaximConfig(lanes=LanesConfigSection())  # no placement anywhere
        monkeypatch.setattr(config_loader, "load_config", lambda *a, **k: cfg)
        lane_configs = {"large": LaneConfig(name="large", max_workers=1, model_profile="smollm")}
        out = _apply_config_placements(lane_configs, None)
        assert out["large"].placement == ()  # unchanged

    def test_producer_creates_lane_for_config_only_tier(self, monkeypatch):
        # config places a tier the hardware detection didn't create (e.g. a
        # remote-only peer on a GPU-less box).
        cfg = MaximConfig(
            lanes=LanesConfigSection(
                large=LaneTierConfig(placement=(LaneTierPlacement(origin="peer", url="http://10.0.0.9:8100/v1"),))
            )
        )
        monkeypatch.setattr(config_loader, "load_config", lambda *a, **k: cfg)
        out = _apply_config_placements({}, None)
        assert "large" in out and out["large"].placement[0].origin is Origin.PEER


# ─── per-entry compile (_placement_entry_to_provider) ───────────────────────


class TestPlacementEntryToProvider:
    def test_peer_entry(self):
        prov, is_cloud = _placement_entry_to_provider(
            ProviderPlacement(origin=Origin.PEER, url="http://10.0.0.9:8100/v1", model="m"), "k"
        )
        assert prov["type"] == "maxim_peer" and prov["base_url"] == "http://10.0.0.9:8100/v1"
        assert prov["allow_local_endpoints"] is True and is_cloud is False

    def test_cloud_url_entry(self):
        prov, is_cloud = _placement_entry_to_provider(
            ProviderPlacement(origin=Origin.CLOUD, url="https://api.x.com/v1", model="m"), "k"
        )
        assert prov["type"] == "openai" and prov["base_url"] == "https://api.x.com/v1" and is_cloud is True

    def test_cloud_profile_entry_uses_profile_backend(self):
        # claude-sonnet is a cloud profile → its backend (anthropic), not openai.
        prov, is_cloud = _placement_entry_to_provider(
            ProviderPlacement(origin=Origin.CLOUD, model="claude-sonnet"), "k"
        )
        assert prov is not None and is_cloud is True
        assert prov["type"] in ("anthropic", "claude")
        assert prov["model"].startswith("claude-sonnet")

    def test_non_cloud_profile_cloud_entry_skipped(self):
        prov, is_cloud = _placement_entry_to_provider(
            ProviderPlacement(origin=Origin.CLOUD, model="definitely-not-a-profile-xyz"), "k"
        )
        assert prov is None


# ─── multi-element tail injection (end to end through get_backend) ──────────


class TestMultiElementTailInjection:
    def test_local_primary_cloud_fallback(self):
        from maxim.models.language.config import LLMConfig

        cfg = LaneConfig(
            name="large",
            max_workers=1,
            placement=(
                ProviderPlacement(origin=Origin.LOCAL, model="mistral-7b"),
                ProviderPlacement(origin=Origin.CLOUD, model="claude-sonnet"),
            ),
        )
        mgr = LaneBackendManager({"large": cfg}, max_backends=5)
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            mock_load.return_value = LLMConfig()
            mgr.get_backend("large")
            (built_cfg,), _ = mock_router.call_args
            priority = built_cfg.routing.get("provider_priority", [])
            # primary + one cloud fallback, fallback AFTER primary
            assert "large-placement-fallback-1" in priority
            assert priority.index("large-placement-fallback-1") == len(priority) - 1
            fb = built_cfg.providers["large-placement-fallback-1"]
            assert fb["type"] in ("anthropic", "claude") and fb["model"].startswith("claude-sonnet")
            assert built_cfg.cloud_enabled is True

    def test_single_element_placement_no_tail_injected(self):
        from maxim.models.language.config import LLMConfig

        cfg = LaneConfig(
            name="large",
            max_workers=1,
            placement=(ProviderPlacement(origin=Origin.LOCAL, model="mistral-7b"),),
        )
        mgr = LaneBackendManager({"large": cfg}, max_backends=5)
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            mock_load.return_value = LLMConfig()
            mgr.get_backend("large")
            (built_cfg,), _ = mock_router.call_args
            priority = built_cfg.routing.get("provider_priority", []) if built_cfg.routing else []
            assert not any("placement-fallback" in k for k in priority)


class TestCloudProfilePrimaryBuild:
    def test_cloud_profile_primary_builds_via_profile_path_with_cloud_enabled(self):
        # An explicit CLOUD primary with no url (a cloud profile) must build via
        # the profile-driven local path (correct anthropic backend), NOT the
        # openai remote path, and have cloud_enabled forced on.
        from maxim.models.language.config import LLMConfig

        cfg = LaneConfig(
            name="large",
            max_workers=1,
            placement=(ProviderPlacement(origin=Origin.CLOUD, model="claude-sonnet"),),
        )
        mgr = LaneBackendManager({"large": cfg}, max_backends=5, max_cloud_lanes=1)
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            mock_load.return_value = LLMConfig(backend="anthropic", model="claude-sonnet-4-6")
            mgr.get_backend("large")
            # profile-driven path: load_llm_config called with the cloud profile
            _, kwargs = mock_load.call_args
            assert kwargs.get("profile_override") == "claude-sonnet"
            # cloud_enabled forced on
            (built_cfg,), _ = mock_router.call_args
            assert built_cfg.cloud_enabled is True
            assert built_cfg.backend == "anthropic"


# ─── CLI re-expression as placement edits ───────────────────────────────────

from maxim.runtime.lane_backends import (  # noqa: E402
    _apply_cloud_cli_overrides,
    _apply_cloud_fallback_override,
    _apply_local_llm_override,
)

_CLOUD = "claude-sonnet-4-6"  # canonical cloud-profile key (cli_utils normalizes)


class TestCliReexpression:
    def test_cloud_lane_prepends_cloud_primary(self, monkeypatch):
        monkeypatch.setenv("MAXIM_CLOUD_LANE_LARGE_MODEL", _CLOUD)
        lane_configs = {"large": LaneConfig(name="large", max_workers=1, model_profile="mistral-7b")}
        out = _apply_cloud_cli_overrides(lane_configs, None)
        pl = out["large"].placement
        assert pl[0].origin is Origin.CLOUD and pl[0].model == _CLOUD
        assert out["large"].model_profile is None  # legacy field cleared

    def test_cloud_lane_prepends_on_config_base(self, monkeypatch):
        monkeypatch.setenv("MAXIM_CLOUD_LANE_LARGE_MODEL", _CLOUD)
        base = (ProviderPlacement(origin=Origin.PEER, url="http://leader/v1"),)
        lane_configs = {"large": LaneConfig(name="large", max_workers=1, placement=base)}
        out = _apply_cloud_cli_overrides(lane_configs, None)
        pl = out["large"].placement
        assert [p.origin for p in pl] == [Origin.CLOUD, Origin.PEER]  # cloud primary, config as fallback

    def test_cloud_fallback_appends_after_derived_primary(self, monkeypatch):
        monkeypatch.setenv("MAXIM_CLOUD_FALLBACK_MODEL", _CLOUD)
        lane_configs = {"large": LaneConfig(name="large", max_workers=1, model_profile="mistral-7b")}
        out = _apply_cloud_fallback_override(lane_configs, None, peer_owned=False)
        pl = out["large"].placement
        assert [p.origin for p in pl] == [Origin.LOCAL, Origin.CLOUD]  # derived local primary, cloud fallback
        assert pl[0].model == "mistral-7b" and pl[1].model == _CLOUD

    def test_cloud_fallback_ignored_when_no_primary(self, monkeypatch):
        monkeypatch.setenv("MAXIM_CLOUD_FALLBACK_MODEL", _CLOUD)
        lane_configs = {"large": LaneConfig(name="large", max_workers=1)}  # no profile, no url
        out = _apply_cloud_fallback_override(lane_configs, None, peer_owned=False)
        assert out["large"].placement == ()  # nothing to fall back from

    def test_llm_overrides_placement_primary_keeping_tail(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "mistral-7b")
        base = (
            ProviderPlacement(origin=Origin.PEER, url="http://leader/v1"),
            ProviderPlacement(origin=Origin.CLOUD, model=_CLOUD),
        )
        lane_configs = {"large": LaneConfig(name="large", max_workers=1, placement=base)}
        out = _apply_local_llm_override(lane_configs, None)
        pl = out["large"].placement
        assert pl[0].origin is Origin.LOCAL and pl[0].model == "mistral-7b"  # primary overridden
        assert pl[1].origin is Origin.CLOUD  # fallback tail kept


# ─── review-fold regression guards ──────────────────────────────────────────

import pytest  # noqa: E402

from maxim.runtime.lane_backends import BackendGateError  # noqa: E402


class TestCloudPlacementCap:
    def test_cloud_primary_placement_consumes_cap(self):
        # A CLOUD *primary* (placement_kind="cloud") consumes a MAX_CLOUD_LANES
        # slot — refused at cap=0 (the cap check fires before _build_backend).
        cfg = LaneConfig(
            name="large", max_workers=1, placement=(ProviderPlacement(origin=Origin.CLOUD, model="claude-sonnet"),)
        )
        mgr = LaneBackendManager({"large": cfg}, max_backends=5, max_cloud_lanes=0)
        with pytest.raises(BackendGateError, match="MAX_CLOUD_LANES"):
            mgr.get_backend("large")

    def test_config_cloud_placement_bumps_cap(self, monkeypatch):
        # Writing a CLOUD placement in config.json bumps MAX_CLOUD_LANES so the
        # canonical declarative cloud lane isn't rejected at cap=0.
        monkeypatch.delenv("MAXIM_MAX_CLOUD_LANES", raising=False)
        cfg = MaximConfig(
            lanes=LanesConfigSection(
                large=LaneTierConfig(placement=(LaneTierPlacement(origin="cloud", model="claude-sonnet"),))
            )
        )
        monkeypatch.setattr(config_loader, "get_config", lambda *a, **k: cfg)
        _apply_config_placements({"large": LaneConfig(name="large", max_workers=1)}, None)
        import os

        assert int(os.environ.get("MAXIM_MAX_CLOUD_LANES", "0")) >= 1


class TestFallbackKeyLaneQualified:
    def test_fallback_provider_key_includes_lane_name(self):
        from maxim.models.language.config import LLMConfig

        cfg = LaneConfig(
            name="medium",
            max_workers=1,
            placement=(
                ProviderPlacement(origin=Origin.LOCAL, model="mistral-7b"),
                ProviderPlacement(origin=Origin.CLOUD, model="claude-sonnet"),
            ),
        )
        mgr = LaneBackendManager({"medium": cfg}, max_backends=5)
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            mock_load.return_value = LLMConfig()
            mgr.get_backend("medium")
            (built_cfg,), _ = mock_router.call_args
            # lane-qualified so two tiers' fallbacks can't collide
            assert "medium-placement-fallback-1" in built_cfg.providers
