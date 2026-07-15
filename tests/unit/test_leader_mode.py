"""Tests for ``runtime/leader_mode.py`` — back-compat wrapper around
``runtime/role.py`` after C3 of config_unification.md.

**C3 unification:** ``leader_mode.detect_role`` is now a thin wrapper
that delegates to :func:`maxim.runtime.role.detect_role`. The
cloudflared check moved into ``role.py`` (rank 4) with the .yaml
extension widening. ``leader_mode`` keeps the legacy ``client``
return value for callers that haven't migrated to the ``peer``
vocabulary; new code should call ``role.py`` directly.

These tests cover the wrapper-specific contract (role/bind_host/reason
translation + back-compat ``client`` shape). The seven-rank detection
matrix is exercised in ``test_role_unification.py``.
"""

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

    def test_client_legacy_translation(self, monkeypatch):
        """``MAXIM_ROLE=client`` is the legacy term; role.py coerces
        to ``peer``; the wrapper translates back to ``client`` for
        back-compat with pre-C3 callers."""
        monkeypatch.setenv("MAXIM_ROLE", "client")
        d = detect_role()
        assert d.role == "client"  # legacy form preserved at wrapper
        assert not d.is_leader
        assert d.bind_host == "127.0.0.1"

    def test_peer(self, monkeypatch):
        """``MAXIM_ROLE=peer`` from new-vocabulary callers; the wrapper
        translates to ``client`` for back-compat."""
        monkeypatch.setenv("MAXIM_ROLE", "peer")
        d = detect_role()
        assert d.role == "client"  # wrapper still returns client through 1.x
        assert not d.is_leader
        assert d.bind_host == "127.0.0.1"

    def test_solo(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        d = detect_role()
        assert d.role == "solo"
        assert not d.is_leader
        assert d.bind_host == "127.0.0.1"

    def test_env_takes_priority_over_cloudflared(self, monkeypatch):
        """Rank 1 (env) wins over rank 4 (cloudflared). The cloudflared
        check now lives in role.py — patch there."""
        monkeypatch.setenv("MAXIM_ROLE", "client")
        with patch("maxim.runtime.role._cloudflared_config_exists") as mock_cf:
            from pathlib import Path

            mock_cf.return_value = Path("/fake/cloudflared.yml")
            d = detect_role()
            assert d.role == "client"  # env-set client outranks cloudflared-implies-leader

    def test_invalid_value_ignored(self, monkeypatch):
        """Invalid MAXIM_ROLE falls through to lower ranks. With no
        cloudflared / no peer.yml / no mesh.yml / no config.json, the
        unified detector defaults to ``leader`` at rank 7."""
        monkeypatch.setenv("MAXIM_ROLE", "bogus")
        with patch("maxim.runtime.role._cloudflared_config_exists") as mock_cf:
            mock_cf.return_value = None
            with patch("maxim.runtime.role._peer_yml_exists") as mock_peer:
                mock_peer.return_value = False
                with patch("maxim.runtime.role._mesh_yml_exists") as mock_mesh:
                    mock_mesh.return_value = False
                    d = detect_role()
                    # Post-C3 default is leader (rank 7), not solo as
                    # pre-C3 leader_mode used to return
                    assert d.role == "leader"


class TestCloudflaredDetection:
    def test_cloudflared_config_present_implies_leader(self, monkeypatch):
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        with patch("maxim.runtime.role._cloudflared_config_exists") as mock_cf:
            from pathlib import Path

            mock_cf.return_value = Path("/home/u/.cloudflared/config.yml")
            with patch("maxim.runtime.role._peer_yml_exists") as mock_peer:
                mock_peer.return_value = False
                with patch("maxim.runtime.role._mesh_yml_exists") as mock_mesh:
                    mock_mesh.return_value = False
                    d = detect_role()
                    assert d.role == "leader"
                    assert "cloudflared" in d.reason.lower()

    def test_no_signals_defaults_to_leader(self, monkeypatch):
        """Post-C3, the wrapper defaults to ``leader`` (rank 7) when
        nothing else fires — this is the unified-detector behavior.
        Pre-C3 ``leader_mode`` used to default to ``solo``, but that
        was the symptom of the two-detector divergence C3 fixes."""
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        with patch("maxim.runtime.role._cloudflared_config_exists") as mock_cf:
            mock_cf.return_value = None
            with patch("maxim.runtime.role._peer_yml_exists") as mock_peer:
                mock_peer.return_value = False
                with patch("maxim.runtime.role._mesh_yml_exists") as mock_mesh:
                    mock_mesh.return_value = False
                    d = detect_role()
                    assert d.role == "leader"
                    assert d.bind_host == "0.0.0.0"


class TestRoleDecision:
    def test_is_leader_property(self):
        assert RoleDecision(role="leader", bind_host="0.0.0.0", reason="x").is_leader is True
        assert RoleDecision(role="client", bind_host="127.0.0.1", reason="x").is_leader is False
        assert RoleDecision(role="peer", bind_host="127.0.0.1", reason="x").is_leader is False
        assert RoleDecision(role="solo", bind_host="127.0.0.1", reason="x").is_leader is False

    def test_frozen(self):
        d = RoleDecision(role="solo", bind_host="127.0.0.1", reason="x")
        import dataclasses

        assert dataclasses.is_dataclass(d)
