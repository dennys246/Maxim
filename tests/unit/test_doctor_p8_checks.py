"""Tests for the P8 routing-visibility checks added to `maxim doctor`.

Four new check functions in `doctor/checks.py`:

* `check_tier_effectiveness` — flags when tier detection landed on a
  smaller profile than the hardware allows because the bigger one isn't
  downloaded.
* `check_peer_vs_local_conflict` — informational notice when --llm
  + peer config will run locally per P1.
* `check_remote_reachability` — structured probe with outcome-specific
  fix hints (auth_rejected, dns_fail, etc.).
* `check_storage_footprint` — fail/warn/ok bands on free disk + the
  per-subdir breakdown from F0.5.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from maxim.doctor.checks import (
    check_peer_vs_local_conflict,
    check_remote_reachability,
    check_storage_footprint,
    check_tier_effectiveness,
)
from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
from maxim.runtime.capabilities import RuntimeCapabilities
from maxim.runtime.llm_server import ProbeResult
from maxim.runtime.worker_pool import LaneConfig
from maxim.utils.storage import StorageReport


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch):
    monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)
    monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)


# ─── check_tier_effectiveness ───────────────────────────────────────────────


class TestCheckTierEffectiveness:
    def _lane(self, name, profile):
        return LaneConfig(name=name, max_workers=1, model_profile=profile)

    def test_ok_when_actual_matches_ideal(self):
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="cuda", vram_gb=24, ram_gb=64)

        def fake_detect(_caps, *, profile_available=None):
            return {"large": self._lane("large", "qwen2.5-14b-instruct")}

        with patch("maxim.runtime.lane_models.detect_tiers", side_effect=fake_detect):
            result = check_tier_effectiveness(caps)
        assert result.status == "ok"
        assert "qwen2.5-14b-instruct" in result.message

    def test_warns_with_download_command_when_actual_lower_than_ideal(self):
        """Hardware could run qwen-14b but actual landed on mistral-7b
        because qwen isn't downloaded."""
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=24, ram_gb=24)
        actual_tiers = {"large": self._lane("large", "mistral-7b-instruct-v0.2")}
        ideal_tiers = {"large": self._lane("large", "qwen2.5-14b-instruct")}
        calls = {"n": 0}

        def fake_detect(_caps, *, profile_available=None):
            calls["n"] += 1
            return actual_tiers if calls["n"] == 1 else ideal_tiers

        with patch("maxim.runtime.lane_models.detect_tiers", side_effect=fake_detect):
            result = check_tier_effectiveness(caps)
        assert result.status == "warn"
        assert "mistral-7b-instruct-v0.2" in result.message
        assert "qwen2.5-14b-instruct" in result.message
        assert result.fix is not None
        assert "python -m maxim.models.download --llm qwen2.5-14b-instruct" in result.fix

    def test_info_when_no_large_or_medium(self):
        caps = RuntimeCapabilities(has_gpu=False, gpu_type=None, vram_gb=0, ram_gb=4)
        with patch("maxim.runtime.lane_models.detect_tiers", return_value={}):
            result = check_tier_effectiveness(caps)
        assert result.status == "info"


# ─── check_peer_vs_local_conflict ───────────────────────────────────────────


class TestCheckPeerVsLocalConflict:
    def test_ok_when_no_llm_override(self, monkeypatch):
        monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)
        result = check_peer_vs_local_conflict()
        assert result.status == "ok"

    def test_ok_when_no_peer_config(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "mistral-7b-instruct-v0.2")
        with patch("maxim.peer.config.read_peer_config", return_value=None):
            result = check_peer_vs_local_conflict()
        assert result.status == "ok"
        assert "no peer config" in result.message

    def test_ok_when_cloud_profile(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "claude-sonnet")
        from maxim.peer.config import PeerConfig

        fake_cfg = PeerConfig(url="https://leader/v1", api_key="x")
        with (
            patch("maxim.peer.config.read_peer_config", return_value=fake_cfg),
            patch.dict(
                "maxim.models.language.config._BUILTIN_PROFILES",
                {"claude-sonnet": {"cloud": True}},
                clear=False,
            ),
        ):
            result = check_peer_vs_local_conflict()
        assert result.status == "ok"
        assert "cloud" in result.message

    def test_info_when_local_overrides_peer(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "mistral-7b-instruct-v0.2")
        from maxim.peer.config import PeerConfig

        fake_cfg = PeerConfig(url="https://leader.example.com/v1", api_key="x")
        with patch("maxim.peer.config.read_peer_config", return_value=fake_cfg):
            result = check_peer_vs_local_conflict()
        assert result.status == "info"
        assert "LOCALLY" in result.message
        assert "leader.example.com" in result.message
        assert "maxim peer llm mistral-7b-instruct-v0.2" in result.message


# ─── check_remote_reachability ──────────────────────────────────────────────


class TestCheckRemoteReachability:
    URL = "https://leader.example.com/v1"

    def test_ok_returns_latency(self):
        result_obj = ProbeResult(self.URL, "ok", "HTTP 200", 42.0)
        with patch.object(_MaximPeerBackend, "health_check", return_value=result_obj):
            result = check_remote_reachability(self.URL, "k")
        assert result.status == "ok"
        assert "42" in result.message

    def test_auth_rejected_warns_with_key_rotation_hint(self):
        result_obj = ProbeResult(self.URL, "auth_rejected", "401 Unauthorized", 5.0)
        with patch.object(_MaximPeerBackend, "health_check", return_value=result_obj):
            result = check_remote_reachability(self.URL, "wrong-key")
        assert result.status == "warn"
        assert result.fix is not None
        # Post-config-unification UX fold (2026-06-03): the fix-hint
        # now routes through the canonical classifier in
        # peer/probe_classify.py and recommends `maxim peer connect`
        # for re-pairing the peer with the leader's current key.
        # The verbose hint also mentions `maxim tunnel key show` and
        # `maxim tunnel key rotate` as the leader-side commands.
        assert "maxim peer connect" in result.fix
        assert "maxim tunnel key" in result.fix

    def test_dns_fail_fails_with_hostname_hint(self):
        result_obj = ProbeResult(self.URL, "dns_fail", "name not known", None)
        with patch.object(_MaximPeerBackend, "health_check", return_value=result_obj):
            result = check_remote_reachability(self.URL, "k")
        assert result.status == "fail"
        assert "hostname" in (result.fix or "").lower()

    def test_connection_refused_suggests_starting_leader(self):
        result_obj = ProbeResult(self.URL, "connection_refused", "nope", None)
        with patch.object(_MaximPeerBackend, "health_check", return_value=result_obj):
            result = check_remote_reachability(self.URL, "k")
        assert result.status == "fail"
        assert "maxim" in (result.fix or "").lower() or "leader" in (result.fix or "").lower()

    def test_no_url_no_peer_config_returns_info(self):
        with patch("maxim.peer.config.read_peer_config", return_value=None):
            result = check_remote_reachability(None)
        assert result.status == "info"


# ─── check_storage_footprint ────────────────────────────────────────────────


class TestCheckStorageFootprint:
    def _report(self, free_gb, maxim_gb=5.0):
        from pathlib import Path

        return StorageReport(
            data_home=Path("/fake/.maxim"),
            fs_free_gb=free_gb,
            fs_total_gb=500.0,
            subdir_sizes_gb={"models": 4.0, "sessions": 1.0},
            total_maxim_gb=maxim_gb,
            walked_at=0.0,
        )

    def test_ok_when_plenty_free(self):
        with patch("maxim.utils.storage.report_storage", return_value=self._report(100.0)):
            result = check_storage_footprint()
        assert result.status == "ok"

    def test_warn_when_under_20gb(self):
        with patch("maxim.utils.storage.report_storage", return_value=self._report(15.0)):
            result = check_storage_footprint()
        assert result.status == "warn"

    def test_fail_when_under_10gb(self):
        with patch("maxim.utils.storage.report_storage", return_value=self._report(5.0)):
            result = check_storage_footprint()
        assert result.status == "fail"
        assert result.fix is not None
        assert "maxim --delete-model" in result.fix

    def test_info_when_fs_unknown(self):
        with patch(
            "maxim.utils.storage.report_storage",
            return_value=self._report(float("inf")),
        ):
            result = check_storage_footprint()
        assert result.status == "info"
