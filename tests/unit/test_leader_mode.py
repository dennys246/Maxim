"""Tests for leader-mode role detection (multi-LLM Phase 6b)."""

from __future__ import annotations

from unittest.mock import patch

from maxim.runtime.leader_mode import RoleDecision, detect_role


class TestExplicitEnvOverride:
    def test_leader(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        d = detect_role()
        assert d.role == "leader"
        assert d.is_leader
        assert d.bind_host == "0.0.0.0"

    def test_client(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "client")
        d = detect_role()
        assert d.role == "client"
        assert not d.is_leader
        assert d.bind_host == "127.0.0.1"

    def test_solo(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        d = detect_role()
        assert d.role == "solo"
        assert not d.is_leader
        assert d.bind_host == "127.0.0.1"

    def test_env_takes_priority_over_cloudflared(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "client")
        # Even if cloudflared exists, explicit client should win
        with patch("maxim.runtime.leader_mode._cloudflared_config_exists") as mock_cf:
            from pathlib import Path

            mock_cf.return_value = Path("/fake/cloudflared.yml")
            d = detect_role()
            assert d.role == "client"

    def test_invalid_value_ignored(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "bogus")
        monkeypatch.delenv("MAXIM_ROLE", raising=False)  # clean slate
        monkeypatch.setenv("MAXIM_ROLE", "bogus")
        with patch("maxim.runtime.leader_mode._cloudflared_config_exists") as mock_cf:
            mock_cf.return_value = None
            d = detect_role()
            # Invalid value → fall through to cloudflared check → default solo
            assert d.role == "solo"


class TestCloudflaredDetection:
    def test_cloudflared_config_present_implies_leader(self, monkeypatch):
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        with patch("maxim.runtime.leader_mode._cloudflared_config_exists") as mock_cf:
            from pathlib import Path

            mock_cf.return_value = Path("/home/u/.cloudflared/config.yml")
            d = detect_role()
            assert d.role == "leader"
            assert "cloudflared" in d.reason

    def test_no_cloudflared_no_env_defaults_to_solo(self, monkeypatch):
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        with patch("maxim.runtime.leader_mode._cloudflared_config_exists") as mock_cf:
            mock_cf.return_value = None
            d = detect_role()
            assert d.role == "solo"
            assert d.bind_host == "127.0.0.1"


class TestRoleDecision:
    def test_is_leader_property(self):
        assert RoleDecision(role="leader", bind_host="0.0.0.0", reason="x").is_leader is True
        assert RoleDecision(role="client", bind_host="127.0.0.1", reason="x").is_leader is False
        assert RoleDecision(role="solo", bind_host="127.0.0.1", reason="x").is_leader is False

    def test_frozen(self):
        d = RoleDecision(role="solo", bind_host="127.0.0.1", reason="x")
        import dataclasses

        assert dataclasses.is_dataclass(d)
