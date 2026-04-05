"""Tests for cloudflared daemon auto-spawn (src/maxim/tunnel/daemon_spawner.py)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from maxim.tunnel import daemon_spawner
from maxim.tunnel.daemon_spawner import (
    TunnelDaemonSpawner,
    cloudflared_already_running,
    resolve_config_path,
)


# ─── cloudflared process detection ────────────────────────────────────────

class TestAlreadyRunning:
    def test_returns_true_when_pgrep_finds_process(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "12345 /usr/bin/cloudflared tunnel run maxim-llm\n"
            assert cloudflared_already_running() is True

    def test_returns_false_when_pgrep_empty(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            assert cloudflared_already_running() is False

    def test_returns_false_on_exception(self):
        with patch("subprocess.run", side_effect=OSError("pgrep not found")):
            assert cloudflared_already_running() is False


# ─── config path resolution ───────────────────────────────────────────────

class TestConfigPathResolution:
    def test_user_config_preferred_when_both_exist(self, tmp_path, monkeypatch):
        user_cfg = tmp_path / "user.yml"
        sys_cfg = tmp_path / "system.yml"
        user_cfg.write_text("user")
        sys_cfg.write_text("system")
        monkeypatch.setattr(daemon_spawner, "USER_CONFIG_PATH", user_cfg)
        monkeypatch.setattr(daemon_spawner, "SYSTEM_CONFIG_PATH", sys_cfg)
        assert resolve_config_path() == user_cfg

    def test_system_config_used_when_only_it_exists(self, tmp_path, monkeypatch):
        user_cfg = tmp_path / "missing-user.yml"
        sys_cfg = tmp_path / "system.yml"
        sys_cfg.write_text("system")
        monkeypatch.setattr(daemon_spawner, "USER_CONFIG_PATH", user_cfg)
        monkeypatch.setattr(daemon_spawner, "SYSTEM_CONFIG_PATH", sys_cfg)
        assert resolve_config_path() == sys_cfg

    def test_none_when_neither_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(daemon_spawner, "USER_CONFIG_PATH", tmp_path / "a.yml")
        monkeypatch.setattr(daemon_spawner, "SYSTEM_CONFIG_PATH", tmp_path / "b.yml")
        assert resolve_config_path() is None


# ─── spawner lifecycle ────────────────────────────────────────────────────

class TestSpawnerStart:
    def test_skips_start_when_cloudflared_missing(self, tmp_path, capsys):
        cfg = tmp_path / "config.yml"
        cfg.write_text("tunnel: x")
        with patch("maxim.tunnel.daemon_spawner.find_cloudflared", return_value=None):
            spawner = TunnelDaemonSpawner(config_path=cfg)
            assert spawner.start() is False
        err = capsys.readouterr().err
        assert "cloudflared not found" in err.lower()

    def test_skips_start_when_config_missing(self, capsys):
        with patch("maxim.tunnel.daemon_spawner.find_cloudflared", return_value="/usr/bin/cloudflared"):
            spawner = TunnelDaemonSpawner(config_path=Path("/tmp/does-not-exist-xyz.yml"))
            assert spawner.start() is False
        err = capsys.readouterr().err
        assert "no config" in err.lower()

    def test_builds_expected_cmd(self, tmp_path):
        cfg = tmp_path / "config.yml"
        cfg.write_text("tunnel: x")
        with patch("maxim.tunnel.daemon_spawner.find_cloudflared", return_value="/bin/cloudflared"), \
             patch("subprocess.Popen") as mock_popen, \
             patch("time.sleep"):
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None  # still running after sleep
            mock_popen.return_value = mock_proc
            spawner = TunnelDaemonSpawner(config_path=cfg)
            assert spawner.start() is True
            args = mock_popen.call_args[0][0]
        assert args[0] == "/bin/cloudflared"
        assert "--no-autoupdate" in args
        assert "--config" in args
        assert str(cfg) in args
        assert "tunnel" in args
        assert "run" in args

    def test_start_new_session_true(self, tmp_path):
        cfg = tmp_path / "config.yml"
        cfg.write_text("tunnel: x")
        with patch("maxim.tunnel.daemon_spawner.find_cloudflared", return_value="/bin/cloudflared"), \
             patch("subprocess.Popen") as mock_popen, \
             patch("time.sleep"):
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc
            TunnelDaemonSpawner(config_path=cfg).start()
            assert mock_popen.call_args.kwargs.get("start_new_session") is True

    def test_returns_false_when_process_dies_immediately(self, tmp_path):
        cfg = tmp_path / "config.yml"
        cfg.write_text("tunnel: x")
        with patch("maxim.tunnel.daemon_spawner.find_cloudflared", return_value="/bin/cloudflared"), \
             patch("subprocess.Popen") as mock_popen, \
             patch("time.sleep"):
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 1  # dead after sleep
            mock_popen.return_value = mock_proc
            assert TunnelDaemonSpawner(config_path=cfg).start() is False

    def test_subprocess_failure_returns_false(self, tmp_path, capsys):
        cfg = tmp_path / "config.yml"
        cfg.write_text("tunnel: x")
        with patch("maxim.tunnel.daemon_spawner.find_cloudflared", return_value="/bin/cloudflared"), \
             patch("subprocess.Popen", side_effect=OSError("nope")):
            assert TunnelDaemonSpawner(config_path=cfg).start() is False
        err = capsys.readouterr().err
        assert "failed to spawn" in err.lower()


class TestSpawnerStop:
    def test_stop_without_start_is_noop(self, tmp_path):
        spawner = TunnelDaemonSpawner(config_path=tmp_path / "nope.yml")
        spawner.stop()  # should not raise

    def test_stop_terminates_process(self, tmp_path):
        spawner = TunnelDaemonSpawner(config_path=tmp_path / "nope.yml")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        spawner._process = mock_proc
        spawner.stop()
        mock_proc.terminate.assert_called_once()
        assert spawner._process is None

    def test_stop_kills_on_timeout(self, tmp_path):
        spawner = TunnelDaemonSpawner(config_path=tmp_path / "nope.yml")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="", timeout=5.0), None]
        spawner._process = mock_proc
        spawner.stop()
        mock_proc.kill.assert_called_once()
