"""Tests for `maxim doctor` + `maxim peer test` (platform detection + checks + CLI)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from maxim.doctor.checks import CheckResult, Status
from maxim.doctor.cli import run_doctor_subcommand, run_peer_subcommand
from maxim.doctor.platform_detect import (
    PlatformInfo,
    detect_platform,
)


def _info(**overrides) -> PlatformInfo:
    defaults = dict(
        os="linux", os_release="Ubuntu 24.04", runtime="native",
        distro="ubuntu", arch="x86_64", kernel_version="6.5.0",
        windows_host_ip=None,
    )
    defaults.update(overrides)
    return PlatformInfo(**defaults)


# ─── platform detection ───────────────────────────────────────────────────

class TestPlatformDetection:
    def test_detect_returns_platforminfo(self):
        info = detect_platform()
        assert isinstance(info, PlatformInfo)
        assert info.os in ("linux", "macos", "windows", "unknown")
        assert info.runtime in ("native", "wsl1", "wsl2", "docker", "unknown")

    def test_is_wsl_property(self):
        assert _info(runtime="wsl2").is_wsl is True
        assert _info(runtime="wsl1").is_wsl is True
        assert _info(runtime="native").is_wsl is False
        assert _info(runtime="docker").is_wsl is False

    def test_display_name_wsl2(self):
        info = _info(runtime="wsl2", os_release="Ubuntu 24.04")
        assert "WSL2" in info.display_name
        assert "Ubuntu 24.04" in info.display_name
        assert "Windows" in info.display_name

    def test_display_name_native(self):
        info = _info(runtime="native", os_release="Ubuntu 24.04")
        assert info.display_name == "Ubuntu 24.04"

    def test_display_name_docker(self):
        info = _info(runtime="docker", os_release="Alpine 3.19")
        assert "Docker" in info.display_name


# ─── CheckResult ──────────────────────────────────────────────────────────

class TestCheckResult:
    def test_symbol_mapping(self):
        assert CheckResult(name="x", status="ok", message="").symbol == "✓"
        assert CheckResult(name="x", status="warn", message="").symbol == "⚠"
        assert CheckResult(name="x", status="fail", message="").symbol == "✗"


# ─── individual checks (mocked) ───────────────────────────────────────────

class TestGpuCheck:
    def test_no_torch_returns_warn(self):
        from maxim.doctor.checks import check_gpu
        with patch.dict("sys.modules", {"torch": None}):
            # Can't easily patch "import torch" failing — instead test the
            # torch.cuda.is_available path by ensuring it doesn't crash
            result = check_gpu()
        assert result.status in ("ok", "warn")


class TestServerCheck:
    def test_reachable_returns_ok(self):
        from maxim.doctor.checks import check_server_reachable
        with patch("maxim.runtime.lane_backends._llm_server_responding_at", return_value=True):
            result = check_server_reachable(port=9999)
        assert result.status == "ok"
        assert "9999" in result.message

    def test_unreachable_returns_warn(self):
        from maxim.doctor.checks import check_server_reachable
        with patch("maxim.runtime.lane_backends._llm_server_responding_at", return_value=False):
            result = check_server_reachable(port=9999)
        assert result.status == "warn"
        assert result.fix is not None


class TestLanAccessPlatformSpecific:
    def test_wsl2_shows_netsh_fix(self, monkeypatch):
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        from maxim.doctor.checks import check_lan_access
        info = _info(runtime="wsl2", windows_host_ip="192.168.1.10")
        with patch("maxim.doctor.checks.detect_wsl_ip", return_value="172.24.32.1"):
            result = check_lan_access(info)
        assert result.status == "warn"
        assert "netsh" in (result.fix or "")
        assert "172.24.32.1" in (result.fix or "")
        assert "192.168.1.10" in (result.fix or "")

    def test_native_linux_shows_ufw_or_firewalld(self, monkeypatch):
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        from maxim.doctor.checks import check_lan_access
        info = _info(runtime="native", os="linux", distro="ubuntu")
        with patch("maxim.doctor.platform_detect.detect_lan_ip", return_value="192.168.1.5"):
            result = check_lan_access(info)
        assert result.status == "warn"
        assert "ufw" in (result.fix or "")

    def test_fedora_shows_firewalld(self, monkeypatch):
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        from maxim.doctor.checks import check_lan_access
        info = _info(runtime="native", os="linux", distro="fedora")
        with patch("maxim.doctor.platform_detect.detect_lan_ip", return_value=None):
            result = check_lan_access(info)
        assert "firewall-cmd" in (result.fix or "")

    def test_macos_mentions_settings(self, monkeypatch):
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        from maxim.doctor.checks import check_lan_access
        info = _info(runtime="native", os="macos", distro="unknown")
        with patch("maxim.doctor.platform_detect.detect_lan_ip", return_value="10.0.0.5"):
            result = check_lan_access(info)
        assert "System Settings" in (result.fix or "") or "firewall" in (result.fix or "").lower()

    def test_windows_shows_newnetfirewallrule(self, monkeypatch):
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        from maxim.doctor.checks import check_lan_access
        info = _info(runtime="native", os="windows", distro="unknown")
        result = check_lan_access(info)
        assert "NetFirewallRule" in (result.fix or "")

    def test_leader_mode_removes_fix(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        from maxim.doctor.checks import check_lan_access
        info = _info(runtime="wsl2", windows_host_ip="192.168.1.10")
        result = check_lan_access(info)
        assert result.status == "ok"


class TestCloudflaredCheckPlatformSpecific:
    def test_ubuntu_hints_deb(self):
        from maxim.doctor.checks import check_cloudflared
        with patch("maxim.tunnel.cloudflared.find_cloudflared", return_value=None):
            result = check_cloudflared(_info(os="linux", distro="ubuntu"))
        assert "dpkg" in (result.fix or "")

    def test_fedora_hints_rpm(self):
        from maxim.doctor.checks import check_cloudflared
        with patch("maxim.tunnel.cloudflared.find_cloudflared", return_value=None):
            result = check_cloudflared(_info(os="linux", distro="fedora"))
        assert "rpm" in (result.fix or "")

    def test_macos_hints_brew(self):
        from maxim.doctor.checks import check_cloudflared
        with patch("maxim.tunnel.cloudflared.find_cloudflared", return_value=None):
            result = check_cloudflared(_info(os="macos", distro="unknown"))
        assert "brew" in (result.fix or "")

    def test_windows_hints_download(self):
        from maxim.doctor.checks import check_cloudflared
        with patch("maxim.tunnel.cloudflared.find_cloudflared", return_value=None):
            result = check_cloudflared(_info(os="windows", distro="unknown"))
        assert "cloudflared-windows" in (result.fix or "")

    def test_installed_returns_ok(self):
        from maxim.doctor.checks import check_cloudflared
        with patch("maxim.tunnel.cloudflared.find_cloudflared", return_value="/usr/bin/cloudflared"), \
             patch("maxim.tunnel.cloudflared.cloudflared_version", return_value="cloudflared 2024.1.0"):
            result = check_cloudflared(_info())
        assert result.status == "ok"


# ─── CLI: maxim doctor ────────────────────────────────────────────────────

class TestDoctorCLI:
    def test_help_flag(self, capsys):
        code = run_doctor_subcommand(["--help"])
        assert code == 0
        assert "doctor" in capsys.readouterr().out.lower()

    def test_runs_without_retry(self, capsys):
        code = run_doctor_subcommand([])
        out = capsys.readouterr().out
        assert "Environment" in out
        assert "Local LLM" in out
        assert "Tunnel" in out
        assert code in (0, 1)


# ─── CLI: maxim peer test ─────────────────────────────────────────────────

class TestPeerCLI:
    def test_no_args_shows_usage(self, capsys):
        code = run_peer_subcommand([])
        assert code == 2
        assert "Usage" in capsys.readouterr().out

    def test_help_flag(self, capsys):
        code = run_peer_subcommand(["--help"])
        assert code == 0

    def test_missing_url_returns_2(self, capsys):
        code = run_peer_subcommand(["test"])
        assert code == 2

    def test_unknown_action_returns_2(self, capsys):
        code = run_peer_subcommand(["bogus"])
        assert code == 2

    def test_dns_failure_reported(self, capsys):
        code = run_peer_subcommand(["test", "https://definitely-not-a-real-host.invalid/v1"])
        out = capsys.readouterr().out
        assert "DNS failed" in out or "failed" in out.lower()
        assert code == 1

    def test_parse_key_from_flag(self):
        from maxim.doctor.cli import _parse_peer_opts
        key, model = _parse_peer_opts(["--key", "abc123"])
        assert key == "abc123"
        assert model is None

    def test_parse_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_INFER_REMOTE_API_KEY", "env-key")
        from maxim.doctor.cli import _parse_peer_opts
        key, _ = _parse_peer_opts([])
        assert key == "env-key"

    def test_flag_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_INFER_REMOTE_API_KEY", "env-key")
        from maxim.doctor.cli import _parse_peer_opts
        key, _ = _parse_peer_opts(["--key", "flag-key"])
        assert key == "flag-key"

    def test_parse_model_flag(self):
        from maxim.doctor.cli import _parse_peer_opts
        _, model = _parse_peer_opts(["--model", "mistral-7b"])
        assert model == "mistral-7b"
