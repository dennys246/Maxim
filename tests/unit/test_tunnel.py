"""Tests for the `maxim tunnel` subcommand (multi-LLM Phase 6b)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from maxim.tunnel.cli import run_tunnel_subcommand
from maxim.tunnel.cloudflared import (
    CloudflaredNotInstalled,
    find_cloudflared,
    install_hint,
    require_cloudflared,
)
from maxim.tunnel.config import (
    TunnelConfig,
    read_config_summary,
    render_config_yml,
    write_config_yml,
)


# ─── cloudflared binary detection ──────────────────────────────────────────


class TestCloudflaredDetection:
    def test_find_cloudflared_returns_path_or_none(self):
        # Can't assert a value — depends on test host — just ensure it runs
        result = find_cloudflared()
        assert result is None or isinstance(result, str)

    def test_install_hint_returns_string(self):
        hint = install_hint()
        assert isinstance(hint, str)
        assert len(hint) > 0

    def test_require_raises_with_hint_when_missing(self):
        with patch("maxim.tunnel.cloudflared.find_cloudflared", return_value=None):
            with pytest.raises(CloudflaredNotInstalled) as exc_info:
                require_cloudflared()
        assert "cloudflared" in str(exc_info.value).lower()
        # The hint should be included in the error message
        assert "install" in str(exc_info.value).lower()

    def test_require_returns_path_when_found(self):
        with patch("maxim.tunnel.cloudflared.find_cloudflared", return_value="/usr/bin/cloudflared"):
            assert require_cloudflared() == "/usr/bin/cloudflared"


# ─── config.yml rendering ──────────────────────────────────────────────────


class TestConfigRendering:
    def test_render_contains_required_fields(self):
        cfg = TunnelConfig(
            tunnel_id="abc-123",
            credentials_file="/home/u/.cloudflared/abc-123.json",
            hostname="maxim.example.com",
            local_service="http://localhost:8100",
        )
        yaml = render_config_yml(cfg)
        assert "tunnel: abc-123" in yaml
        assert "credentials-file: /home/u/.cloudflared/abc-123.json" in yaml
        assert "hostname: maxim.example.com" in yaml
        assert "service: http://localhost:8100" in yaml
        assert "http_status:404" in yaml  # catch-all ingress rule

    def test_render_is_stable(self):
        cfg = TunnelConfig(
            tunnel_id="a",
            credentials_file="b",
            hostname="c",
            local_service="d",
        )
        assert render_config_yml(cfg) == render_config_yml(cfg)

    def test_write_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("maxim.tunnel.config.CONFIG_PATH", tmp_path / "config.yml")
        cfg = TunnelConfig(
            tunnel_id="t",
            credentials_file="/c.json",
            hostname="h.com",
            local_service="http://localhost:8100",
        )
        path = write_config_yml(cfg)
        assert path.is_file()
        content = path.read_text()
        assert "tunnel: t" in content

    def test_write_refuses_overwrite_by_default(self, tmp_path, monkeypatch):
        target = tmp_path / "config.yml"
        target.write_text("existing")
        monkeypatch.setattr("maxim.tunnel.config.CONFIG_PATH", target)
        cfg = TunnelConfig(
            tunnel_id="t",
            credentials_file="/c.json",
            hostname="h.com",
            local_service="http://localhost:8100",
        )
        with pytest.raises(FileExistsError):
            write_config_yml(cfg)
        # Original untouched
        assert target.read_text() == "existing"

    def test_write_overwrites_when_requested(self, tmp_path, monkeypatch):
        target = tmp_path / "config.yml"
        target.write_text("existing")
        monkeypatch.setattr("maxim.tunnel.config.CONFIG_PATH", target)
        cfg = TunnelConfig(
            tunnel_id="new",
            credentials_file="/c.json",
            hostname="h.com",
            local_service="http://localhost:8100",
        )
        write_config_yml(cfg, overwrite=True)
        assert "tunnel: new" in target.read_text()


# ─── config.yml parsing ────────────────────────────────────────────────────


class TestConfigReading:
    def test_read_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("maxim.tunnel.config.CONFIG_PATH", tmp_path / "nope.yml")
        assert read_config_summary() is None

    def test_read_extracts_fields(self, tmp_path, monkeypatch):
        target = tmp_path / "config.yml"
        target.write_text(
            "tunnel: uuid-123\n"
            "credentials-file: /tmp/creds.json\n"
            "ingress:\n"
            "  - hostname: maxim.example.com\n"
            "    service: http://localhost:8100\n"
            "  - service: http_status:404\n"
        )
        monkeypatch.setattr("maxim.tunnel.config.CONFIG_PATH", target)
        summary = read_config_summary()
        assert summary is not None
        assert summary["tunnel"] == "uuid-123"
        assert summary["credentials_file"] == "/tmp/creds.json"
        assert summary["hostname"] == "maxim.example.com"
        # service line shows http://localhost:8100 (no port split)
        assert "localhost" in summary["service"]


# ─── subcommand dispatch ───────────────────────────────────────────────────


class TestSubcommandDispatch:
    def test_no_args_prints_usage(self, capsys):
        code = run_tunnel_subcommand([])
        captured = capsys.readouterr()
        assert "Usage" in captured.out
        assert code == 2

    def test_help_flag(self, capsys):
        code = run_tunnel_subcommand(["--help"])
        assert code == 0

    def test_unknown_action_exits_2(self, capsys):
        code = run_tunnel_subcommand(["bogus"])
        assert code == 2

    def test_status_without_cloudflared(self, capsys):
        with patch("maxim.tunnel.cli.find_cloudflared", return_value=None):
            code = run_tunnel_subcommand(["status"])
        out = capsys.readouterr().out
        assert "not installed" in out
        assert code == 1

    def test_status_without_config(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setattr("maxim.tunnel.config.CONFIG_PATH", tmp_path / "none.yml")
        with (
            patch("maxim.tunnel.cli.find_cloudflared", return_value="/usr/bin/cloudflared"),
            patch("maxim.tunnel.cli.cloudflared_version", return_value="cloudflared 2024.1.0"),
        ):
            code = run_tunnel_subcommand(["status"])
        out = capsys.readouterr().out
        assert "not found" in out.lower()
        assert code == 1

    def test_status_with_full_config(self, capsys, tmp_path, monkeypatch):
        target = tmp_path / "config.yml"
        target.write_text(
            "tunnel: abc\n"
            "credentials-file: /tmp/c.json\n"
            "ingress:\n"
            "  - hostname: h.example.com\n"
            "    service: http://localhost:8100\n"
        )
        monkeypatch.setattr("maxim.tunnel.config.CONFIG_PATH", target)
        with (
            patch("maxim.tunnel.cli.find_cloudflared", return_value="/usr/bin/cloudflared"),
            patch("maxim.tunnel.cli.cloudflared_version", return_value="cloudflared 2024.1.0"),
            patch("maxim.tunnel.cli._is_daemon_running", return_value=True),
        ):
            code = run_tunnel_subcommand(["status"])
        out = capsys.readouterr().out
        assert "abc" in out
        assert "h.example.com" in out
        assert "running" in out
        assert code == 0

    def test_setup_exits_when_cloudflared_missing(self, capsys):
        with patch("maxim.tunnel.cli.find_cloudflared", return_value=None):
            code = run_tunnel_subcommand(["setup"])
        out = capsys.readouterr().out
        assert "not found" in out.lower()
        assert "install" in out.lower()
        assert code == 1

    def test_start_exits_when_no_config(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setattr("maxim.tunnel.config.CONFIG_PATH", tmp_path / "nope.yml")
        code = run_tunnel_subcommand(["start"])
        err = capsys.readouterr().err
        assert "No tunnel configured" in err
        assert code == 1

    def test_tail_help(self, capsys):
        code = run_tunnel_subcommand(["tail", "--help"])
        out = capsys.readouterr().out
        assert code == 0
        assert "tail" in out.lower() and "cloudflared" in out.lower()

    def test_tail_unknown_option(self, capsys):
        code = run_tunnel_subcommand(["tail", "--bogus"])
        assert code == 2

    def test_tail_invokes_journalctl(self, capsys):
        with patch("subprocess.call", return_value=0) as mock_call:
            code = run_tunnel_subcommand(["tail", "--since", "10m"])
        assert code == 0
        mock_call.assert_called_once()
        cmd = mock_call.call_args[0][0]
        assert cmd[0] == "journalctl"
        assert "-u" in cmd and "cloudflared" in cmd
        assert "--since" in cmd
        # Compact "10m" translates to journalctl's accepted relative phrase
        # (no leading dash — just "N UNIT ago")
        assert "10 minutes ago" in cmd
        assert "-f" in cmd

    def test_tail_since_compact_forms(self):
        from maxim.tunnel.cli import _to_journalctl_since

        # "N UNIT ago" form (no leading dash — that's what journalctl accepts)
        assert _to_journalctl_since("2m") == "2 minutes ago"
        assert _to_journalctl_since("5h") == "5 hours ago"
        assert _to_journalctl_since("1d") == "1 days ago"
        assert _to_journalctl_since("30s") == "30 seconds ago"

    def test_tail_since_passthrough_for_absolute(self):
        from maxim.tunnel.cli import _to_journalctl_since

        assert _to_journalctl_since("2024-01-02 15:04:05") == "2024-01-02 15:04:05"
        # "-2min" (no-space compact form) is also valid journalctl syntax
        assert _to_journalctl_since("-2min") == "-2min"
        assert _to_journalctl_since("-5h") == "-5h"
        # phrases with " ago" pass through
        assert _to_journalctl_since("5 hours ago") == "5 hours ago"
        assert _to_journalctl_since("yesterday") == "yesterday"  # let journalctl handle

    def test_tail_since_empty_defaults_to_2min(self):
        from maxim.tunnel.cli import _to_journalctl_since

        assert _to_journalctl_since("") == "2 minutes ago"

    def test_tail_since_invalid_falls_through(self):
        from maxim.tunnel.cli import _to_journalctl_since

        # Unknown forms pass through so journalctl can report its own error
        assert _to_journalctl_since("bogus") == "bogus"

    def test_tail_missing_journalctl(self, capsys):
        with patch("subprocess.call", side_effect=FileNotFoundError):
            code = run_tunnel_subcommand(["tail"])
        err = capsys.readouterr().err
        assert code == 1
        assert "journalctl" in err.lower()

    def test_tail_filter_option(self):
        with patch("subprocess.call", return_value=0) as mock_call:
            run_tunnel_subcommand(["tail", "--filter", "req=.*"])
        cmd = mock_call.call_args[0][0]
        assert "-g" in cmd
        assert "req=.*" in cmd


class TestStartDuplicateGuard:
    """`maxim tunnel start` should refuse when cloudflared is already running."""

    def _make_config(self, tmp_path, monkeypatch):
        target = tmp_path / "config.yml"
        target.write_text(
            "tunnel: abc-123\n"
            "credentials-file: /c.json\n"
            "ingress:\n  - hostname: h.example.com\n    service: http://localhost:8100\n"
        )
        monkeypatch.setattr("maxim.tunnel.config.CONFIG_PATH", target)

    def test_refuses_when_systemd_service_active(self, tmp_path, monkeypatch, capsys):
        self._make_config(tmp_path, monkeypatch)
        active = MagicMock(returncode=0, stdout="active\n")
        with (
            patch("maxim.tunnel.cli.require_cloudflared", return_value="/bin/cloudflared"),
            patch("subprocess.run", return_value=active),
            patch("subprocess.call") as mock_call,
        ):
            code = run_tunnel_subcommand(["start"])
        err = capsys.readouterr().err
        assert code == 1
        assert "already active" in err
        assert "--force" in err
        mock_call.assert_not_called()  # cloudflared never launched

    def test_refuses_when_foreground_process_running(self, tmp_path, monkeypatch, capsys):
        self._make_config(tmp_path, monkeypatch)

        # systemd inactive, but pgrep finds a running process
        def fake_run(cmd, **kw):
            if "systemctl" in cmd:
                return MagicMock(returncode=3, stdout="inactive\n")
            if "pgrep" in cmd:
                return MagicMock(returncode=0, stdout="12345 cloudflared tunnel run\n")
            return MagicMock(returncode=1, stdout="")

        with (
            patch("maxim.tunnel.cli.require_cloudflared", return_value="/bin/cloudflared"),
            patch("subprocess.run", side_effect=fake_run),
            patch("subprocess.call") as mock_call,
        ):
            code = run_tunnel_subcommand(["start"])
        err = capsys.readouterr().err
        assert code == 1
        assert "foreground process" in err
        assert "--force" in err
        mock_call.assert_not_called()

    def test_force_bypasses_guard(self, tmp_path, monkeypatch, capsys):
        self._make_config(tmp_path, monkeypatch)
        active = MagicMock(returncode=0, stdout="active\n")
        with (
            patch("maxim.tunnel.cli.require_cloudflared", return_value="/bin/cloudflared"),
            patch("subprocess.run", return_value=active),
            patch("subprocess.call", return_value=0) as mock_call,
        ):
            code = run_tunnel_subcommand(["start", "--force"])
        assert code == 0
        mock_call.assert_called_once()
        # --force should be consumed (not passed through to cloudflared)
        cmd = mock_call.call_args[0][0]
        assert "--force" not in cmd

    def test_proceeds_when_nothing_running(self, tmp_path, monkeypatch, capsys):
        self._make_config(tmp_path, monkeypatch)
        inactive = MagicMock(returncode=3, stdout="inactive\n")
        empty = MagicMock(returncode=1, stdout="")

        def fake_run(cmd, **kw):
            if "systemctl" in cmd:
                return inactive
            return empty  # pgrep

        with (
            patch("maxim.tunnel.cli.require_cloudflared", return_value="/bin/cloudflared"),
            patch("subprocess.run", side_effect=fake_run),
            patch("subprocess.call", return_value=0) as mock_call,
        ):
            code = run_tunnel_subcommand(["start"])
        assert code == 0
        mock_call.assert_called_once()


class TestServiceInstallHint:
    """`maxim tunnel start` should suggest installing as a service when one
    isn't already running, but stay quiet when systemd reports it active."""

    def _make_summary(self, tmp_path, monkeypatch):
        target = tmp_path / "config.yml"
        target.write_text(
            "tunnel: abc-123\n"
            "credentials-file: /c.json\n"
            "ingress:\n  - hostname: h.example.com\n    service: http://localhost:8100\n"
        )
        monkeypatch.setattr("maxim.tunnel.config.CONFIG_PATH", target)

    def test_hint_printed_when_no_service_active(self, tmp_path, monkeypatch, capsys):
        self._make_summary(tmp_path, monkeypatch)
        systemd_result = MagicMock(returncode=3, stdout="inactive\n")
        with (
            patch("maxim.tunnel.cli.require_cloudflared", return_value="/bin/cloudflared"),
            patch("subprocess.run", return_value=systemd_result),
            patch("subprocess.call", return_value=0),
        ):
            run_tunnel_subcommand(["start"])
        out = capsys.readouterr().out
        assert "service install" in out.lower()
        assert "persistent tunnel" in out.lower()

    def test_hint_skipped_when_service_active(self, tmp_path, monkeypatch, capsys):
        self._make_summary(tmp_path, monkeypatch)
        systemd_result = MagicMock(returncode=0, stdout="active\n")
        with (
            patch("maxim.tunnel.cli.require_cloudflared", return_value="/bin/cloudflared"),
            patch("subprocess.run", return_value=systemd_result),
            patch("subprocess.call", return_value=0),
        ):
            run_tunnel_subcommand(["start"])
        out = capsys.readouterr().out
        assert "service install" not in out.lower()

    def test_hint_linux_uses_systemctl(self, tmp_path, monkeypatch, capsys):
        self._make_summary(tmp_path, monkeypatch)
        systemd_result = MagicMock(returncode=3, stdout="inactive\n")
        with (
            patch("maxim.tunnel.cli.require_cloudflared", return_value="/bin/cloudflared"),
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", return_value=systemd_result),
            patch("subprocess.call", return_value=0),
        ):
            run_tunnel_subcommand(["start"])
        out = capsys.readouterr().out
        assert "systemctl enable" in out

    def test_hint_macos_uses_launchctl(self, tmp_path, monkeypatch, capsys):
        self._make_summary(tmp_path, monkeypatch)
        with (
            patch("maxim.tunnel.cli.require_cloudflared", return_value="/bin/cloudflared"),
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", side_effect=OSError("no systemctl")),
            patch("subprocess.call", return_value=0),
        ):
            run_tunnel_subcommand(["start"])
        out = capsys.readouterr().out
        assert "launchctl" in out

    def test_hint_windows_uses_sc(self, tmp_path, monkeypatch, capsys):
        self._make_summary(tmp_path, monkeypatch)
        with (
            patch("maxim.tunnel.cli.require_cloudflared", return_value="/bin/cloudflared"),
            patch("platform.system", return_value="Windows"),
            patch("subprocess.run", side_effect=OSError("no systemctl")),
            patch("subprocess.call", return_value=0),
        ):
            run_tunnel_subcommand(["start"])
        out = capsys.readouterr().out
        assert "Services.msc" in out or "sc start" in out
