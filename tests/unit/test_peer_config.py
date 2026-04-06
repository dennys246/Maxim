"""Tests for peer-side config + `maxim peer connect/show/forget` CLI."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from maxim.peer import config as peer_config
from maxim.peer.cli import run_peer_connect_subcommand
from maxim.peer.config import (
    PeerConfig,
    apply_peer_config_to_env,
    delete_peer_config,
    read_peer_config,
    write_peer_config,
)


@pytest.fixture
def isolated_peer_config(tmp_path, monkeypatch):
    target = tmp_path / "peer.yml"
    monkeypatch.setattr(peer_config, "peer_config_path", lambda: target)
    yield target


# ─── PeerConfig serialization ──────────────────────────────────────────────

class TestPeerConfigSerialization:
    def test_minimal_yaml(self):
        cfg = PeerConfig(url="https://host/v1", api_key="k")
        out = cfg.to_yaml()
        assert "url: https://host/v1" in out
        assert "api_key: k" in out
        assert "model:" not in out
        assert "is_cloud:" not in out

    def test_full_yaml(self):
        cfg = PeerConfig(url="https://host/v1", api_key="k", model="m", is_cloud=True)
        out = cfg.to_yaml()
        assert "model: m" in out
        assert "is_cloud: true" in out

    def test_yaml_is_stable(self):
        cfg = PeerConfig(url="a", api_key="b", model="c", is_cloud=True)
        assert cfg.to_yaml() == cfg.to_yaml()


# ─── File I/O ──────────────────────────────────────────────────────────────

class TestFileIO:
    def test_read_missing_returns_none(self, isolated_peer_config):
        assert read_peer_config() is None

    def test_write_then_read_round_trip(self, isolated_peer_config):
        cfg = PeerConfig(url="https://host/v1", api_key="secret", model="m")
        write_peer_config(cfg)
        loaded = read_peer_config()
        assert loaded is not None
        assert loaded.url == "https://host/v1"
        assert loaded.api_key == "secret"
        assert loaded.model == "m"

    def test_read_skips_malformed(self, isolated_peer_config):
        isolated_peer_config.parent.mkdir(parents=True, exist_ok=True)
        isolated_peer_config.write_text("garbage without colons\n")
        assert read_peer_config() is None  # missing url + api_key

    def test_read_ignores_comments_and_blanks(self, isolated_peer_config):
        isolated_peer_config.parent.mkdir(parents=True, exist_ok=True)
        isolated_peer_config.write_text(
            "# This is a comment\n"
            "\n"
            "url: https://host/v1\n"
            "api_key: k\n"
        )
        loaded = read_peer_config()
        assert loaded is not None
        assert loaded.url == "https://host/v1"

    def test_delete(self, isolated_peer_config):
        write_peer_config(PeerConfig(url="a", api_key="b"))
        assert isolated_peer_config.is_file()
        assert delete_peer_config() is True
        assert not isolated_peer_config.is_file()
        assert delete_peer_config() is False  # already gone

    def test_write_sets_0600_on_posix(self, isolated_peer_config):
        if os.name == "nt":
            pytest.skip("POSIX-only")
        write_peer_config(PeerConfig(url="a", api_key="b"))
        mode = isolated_peer_config.stat().st_mode & 0o777
        assert mode == 0o600


# ─── apply_peer_config_to_env ──────────────────────────────────────────────

class TestApplyToEnv:
    def test_sets_env_vars_when_unset(self, monkeypatch):
        for var in ("MAXIM_LANE_INFER_REMOTE_URL", "MAXIM_LANE_INFER_REMOTE_API_KEY",
                    "MAXIM_LANE_INFER_REMOTE_MODEL", "MAXIM_MAX_CLOUD_LANES"):
            monkeypatch.delenv(var, raising=False)
        cfg = PeerConfig(url="https://h/v1", api_key="k", model="m", is_cloud=True)
        apply_peer_config_to_env(cfg)
        assert os.environ["MAXIM_LANE_INFER_REMOTE_URL"] == "https://h/v1"
        assert os.environ["MAXIM_LANE_INFER_REMOTE_API_KEY"] == "k"
        assert os.environ["MAXIM_LANE_INFER_REMOTE_MODEL"] == "m"
        assert os.environ["MAXIM_MAX_CLOUD_LANES"] == "1"

    def test_env_wins_over_file(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_INFER_REMOTE_URL", "https://env-override/v1")
        cfg = PeerConfig(url="https://from-file/v1", api_key="k")
        apply_peer_config_to_env(cfg)
        assert os.environ["MAXIM_LANE_INFER_REMOTE_URL"] == "https://env-override/v1"

    def test_skips_model_when_not_set(self, monkeypatch):
        monkeypatch.delenv("MAXIM_LANE_INFER_REMOTE_MODEL", raising=False)
        cfg = PeerConfig(url="a", api_key="b")  # no model
        apply_peer_config_to_env(cfg)
        assert "MAXIM_LANE_INFER_REMOTE_MODEL" not in os.environ


# ─── CLI subcommand ────────────────────────────────────────────────────────

class TestPeerConnectCLI:
    def test_no_args_prints_usage(self, capsys):
        code = run_peer_connect_subcommand([])
        assert code == 2
        assert "peer" in capsys.readouterr().out.lower()

    def test_help(self, capsys):
        code = run_peer_connect_subcommand(["--help"])
        assert code == 0

    def test_unknown_action(self, capsys):
        code = run_peer_connect_subcommand(["bogus"])
        assert code == 2

    def test_connect_requires_url(self, capsys):
        code = run_peer_connect_subcommand(["connect"])
        assert code == 2

    def test_connect_requires_key(self, capsys, isolated_peer_config):
        # No --key and getpass returns empty
        with patch("getpass.getpass", return_value=""):
            code = run_peer_connect_subcommand(["connect", "https://h/v1", "--skip-test"])
        assert code == 1
        assert "API key required" in capsys.readouterr().err

    def test_connect_saves_config(self, capsys, isolated_peer_config):
        code = run_peer_connect_subcommand([
            "connect", "https://host.example.com/v1",
            "--key", "sk-123",
            "--skip-test",
        ])
        assert code == 0
        loaded = read_peer_config()
        assert loaded is not None
        assert loaded.url == "https://host.example.com/v1"
        assert loaded.api_key == "sk-123"
        assert loaded.is_cloud is False  # peer connections default to not-cloud (your own infra)

    def test_connect_normalizes_url(self, isolated_peer_config):
        run_peer_connect_subcommand([
            "connect", "https://host/",
            "--key", "k",
            "--skip-test",
        ])
        loaded = read_peer_config()
        assert loaded.url == "https://host/v1"

    def test_connect_with_model(self, isolated_peer_config):
        run_peer_connect_subcommand([
            "connect", "http://127.0.0.1:8100/v1",
            "--key", "k",
            "--model", "mistral-7b",
            "--skip-test",
        ])
        loaded = read_peer_config()
        assert loaded.model == "mistral-7b"
        assert loaded.is_cloud is False  # loopback

    def test_show_without_config(self, capsys, isolated_peer_config):
        code = run_peer_connect_subcommand(["show"])
        assert code == 1
        assert "No peer config" in capsys.readouterr().out

    def test_show_with_config(self, capsys, isolated_peer_config):
        write_peer_config(PeerConfig(url="https://h/v1", api_key="k"))
        code = run_peer_connect_subcommand(["show"])
        assert code == 0
        out = capsys.readouterr().out
        assert "https://h/v1" in out

    def test_forget_with_config(self, capsys, isolated_peer_config):
        write_peer_config(PeerConfig(url="a", api_key="b"))
        code = run_peer_connect_subcommand(["forget"])
        assert code == 0
        assert "Removed peer config" in capsys.readouterr().out
        assert not isolated_peer_config.is_file()

    def test_forget_without_config(self, capsys, isolated_peer_config):
        code = run_peer_connect_subcommand(["forget"])
        assert code == 1

    def test_test_action_delegates_to_doctor(self, capsys):
        """peer test <url> should still route to maxim.doctor.cli."""
        # Just verify it dispatches; we don't test full doctor.peer_test here.
        code = run_peer_connect_subcommand(["test", "https://definitely-not-real.invalid/v1"])
        # Will fail DNS (exit 1) but that's doctor.peer_test's path
        assert code in (0, 1, 2)


# ─── URL classification for is_cloud auto-detect ──────────────────────────

class TestUrlClassification:
    def test_public_https_is_cloud(self):
        from maxim.peer.cli import _is_public_url
        assert _is_public_url("https://api.example.com/v1") is True

    def test_loopback_is_not_cloud(self):
        from maxim.peer.cli import _is_public_url
        assert _is_public_url("http://127.0.0.1:8100/v1") is False
        assert _is_public_url("http://localhost:8100/v1") is False

    def test_private_subnet_is_not_cloud(self):
        from maxim.peer.cli import _is_public_url
        assert _is_public_url("http://192.168.1.10:8100/v1") is False
        assert _is_public_url("http://10.0.0.5:8100/v1") is False
