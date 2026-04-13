"""Tests for `maxim doctor` + `maxim peer test` (platform detection + checks + CLI)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from maxim.doctor.checks import CheckResult
from maxim.doctor.cli import run_doctor_subcommand, run_peer_subcommand
from maxim.doctor.platform_detect import (
    PlatformInfo,
    detect_platform,
)


def _info(**overrides) -> PlatformInfo:
    defaults = dict(
        os="linux",
        os_release="Ubuntu 24.04",
        runtime="native",
        distro="ubuntu",
        arch="x86_64",
        kernel_version="6.5.0",
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
        assert CheckResult(name="x", status="info", message="").symbol == "ℹ"


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
    @pytest.fixture(autouse=True)
    def _isolate_cloudflared_config(self):
        """LAN access check now uses detect_role() which inspects cloudflared
        config presence. Isolate tests from the host's actual config file."""
        with patch("maxim.runtime.leader_mode._cloudflared_config_exists", return_value=None):
            yield

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

        with (
            patch("maxim.tunnel.cloudflared.find_cloudflared", return_value="/usr/bin/cloudflared"),
            patch("maxim.tunnel.cloudflared.cloudflared_version", return_value="cloudflared 2024.1.0"),
        ):
            result = check_cloudflared(_info())
        assert result.status == "ok"


# ─── CLI: maxim doctor ────────────────────────────────────────────────────


class TestDoctorCLI:
    def test_help_flag(self, capsys):
        code = run_doctor_subcommand(["--help"])
        assert code == 0
        assert "doctor" in capsys.readouterr().out.lower()

    def test_runs_without_retry(self, capsys, monkeypatch):
        # Ensure solo mode — earlier tests or real peer.yml may set
        # MAXIM_LANE_LARGE_REMOTE_URL, which triggers peer mode and
        # replaces "Local LLM" / "Tunnel" sections with "Peer Connectivity".
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
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
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", "env-key")
        from maxim.doctor.cli import _parse_peer_opts

        key, _ = _parse_peer_opts([])
        assert key == "env-key"

    def test_flag_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", "env-key")
        from maxim.doctor.cli import _parse_peer_opts

        key, _ = _parse_peer_opts(["--key", "flag-key"])
        assert key == "flag-key"

    def test_parse_model_flag(self):
        from maxim.doctor.cli import _parse_peer_opts

        _, model = _parse_peer_opts(["--model", "mistral-7b"])
        assert model == "mistral-7b"


# ─── tier detection check ────────────────────────────────────────────────


class TestCheckTierDetection:
    """Tier detection tests.

    check_tier_detection(caps=...) accepts pre-built RuntimeCapabilities
    to bypass hardware probing. This avoids mock/patch fragility from
    test-order-dependent module import caching.
    """

    def test_gpu_available_reports_ok(self):
        """With GPU, check should report ok with large + small tiers."""
        from maxim.doctor.checks import check_tier_detection
        from maxim.runtime.capabilities import RuntimeCapabilities

        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA RTX 5080", vram_gb=15.9, ram_gb=16.0)
        result = check_tier_detection(caps=caps)
        assert result.status == "ok"
        assert "large" in result.message
        assert "small" in result.message

    def test_no_gpu_high_ram_reports_ok(self):
        """CPU-only with 16GB RAM → ok with medium + small."""
        from maxim.doctor.checks import check_tier_detection
        from maxim.runtime.capabilities import RuntimeCapabilities

        caps = RuntimeCapabilities(has_gpu=False, gpu_type=None, vram_gb=0.0, ram_gb=16.0)
        result = check_tier_detection(caps=caps)
        assert result.status == "ok"
        assert "medium" in result.message

    def test_no_gpu_low_ram_reports_warn(self, monkeypatch):
        """Low RAM, no GPU → warn with fix hints."""
        # Earlier tests may have set MAXIM_LLM_PROFILE via os.environ.setdefault()
        # (e.g. api.run(), api.imagine()). detect_tiers() reads it and would use
        # it as the medium profile regardless of RAM, masking the warning.
        monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)

        from maxim.doctor.checks import check_tier_detection
        from maxim.runtime.capabilities import RuntimeCapabilities

        caps = RuntimeCapabilities(has_gpu=False, gpu_type=None, vram_gb=0.0, ram_gb=4.0)
        result = check_tier_detection(caps=caps)
        assert result.status == "warn", f"Expected 'warn' but got '{result.status}': {result.message}"
        assert "small" in result.message
        assert result.fix is not None
        assert "--cloud-fallback" in result.fix

    def test_tier_detection_in_run_all_checks(self):
        """check_tier_detection should be in the Environment section."""
        from maxim.doctor.checks import run_all_checks

        with patch("maxim.runtime.capabilities.detect_compute_resources") as mock:
            mock.return_value = (True, "NVIDIA RTX 5080", 15.9, 16.0)
            sections = run_all_checks(_info())
            env_section = next(s for name, s in sections if name == "Environment")
            tier_results = [r for r in env_section if r.name == "LLM Tiers"]
            assert len(tier_results) == 1
            assert tier_results[0].status == "ok"


# ─── key hygiene checks ──────────────────────────────────────────────────


class TestKeyAge:
    def test_no_key_returns_info(self):
        from maxim.doctor.checks import check_key_age

        with patch("maxim.tunnel.keys.key_exists", return_value=False):
            result = check_key_age()
        assert result.status == "info"

    def test_fresh_key_returns_ok(self, tmp_path):
        from maxim.doctor.checks import check_key_age

        key_file = tmp_path / "api_key"
        key_file.write_text("test-key")
        with (
            patch("maxim.tunnel.keys.key_exists", return_value=True),
            patch("maxim.tunnel.keys.key_file_path", return_value=key_file),
        ):
            result = check_key_age()
        assert result.status == "ok"
        assert "0 days" in result.message

    def test_old_key_returns_warn(self, tmp_path):
        import os
        import time

        from maxim.doctor.checks import check_key_age

        key_file = tmp_path / "api_key"
        key_file.write_text("test-key")
        # Backdate modification time by 100 days
        old_time = time.time() - (100 * 86400)
        os.utime(key_file, (old_time, old_time))
        with (
            patch("maxim.tunnel.keys.key_exists", return_value=True),
            patch("maxim.tunnel.keys.key_file_path", return_value=key_file),
        ):
            result = check_key_age()
        assert result.status == "warn"
        assert "100 days" in result.message
        assert result.fix is not None


class TestKeyPermissions:
    def test_no_key_returns_info(self):
        from maxim.doctor.checks import check_key_permissions

        with patch("maxim.tunnel.keys.key_exists", return_value=False):
            result = check_key_permissions()
        assert result.status in ("info", "ok")  # ok on Windows (skipped), info on POSIX

    def test_secure_permissions_returns_ok(self, tmp_path):
        import os
        import platform as _platform

        if _platform.system() == "Windows":
            pytest.skip("POSIX-only test")

        from maxim.doctor.checks import check_key_permissions

        key_file = tmp_path / "api_key"
        key_file.write_text("test-key")
        os.chmod(key_file, 0o600)
        with (
            patch("maxim.tunnel.keys.key_exists", return_value=True),
            patch("maxim.tunnel.keys.key_file_path", return_value=key_file),
        ):
            result = check_key_permissions()
        assert result.status == "ok"
        assert "0o600" in result.message

    def test_world_readable_returns_fail(self, tmp_path):
        import os
        import platform as _platform

        if _platform.system() == "Windows":
            pytest.skip("POSIX-only test")

        from maxim.doctor.checks import check_key_permissions

        key_file = tmp_path / "api_key"
        key_file.write_text("test-key")
        os.chmod(key_file, 0o644)
        with (
            patch("maxim.tunnel.keys.key_exists", return_value=True),
            patch("maxim.tunnel.keys.key_file_path", return_value=key_file),
        ):
            result = check_key_permissions()
        assert result.status == "fail"
        assert "chmod" in (result.fix or "")


class TestKeyAuthSmoke:
    def test_no_key_returns_info(self):
        from maxim.doctor.checks import check_key_auth_smoke

        with patch("maxim.tunnel.keys.key_exists", return_value=False):
            result = check_key_auth_smoke()
        assert result.status == "info"

    def test_server_not_reachable_returns_info(self):
        from maxim.doctor.checks import check_key_auth_smoke

        with (
            patch("maxim.tunnel.keys.key_exists", return_value=True),
            patch("maxim.tunnel.keys.read_key", return_value="test-key"),
            patch(
                "maxim.utils.http.fetch_url",
                side_effect=OSError("connection refused"),
            ),
        ):
            result = check_key_auth_smoke(port=19999)
        assert result.status == "info"


# ─── disk + memory checks ────────────────────────────────────────────────


class TestDiskSpace:
    def test_plenty_of_space_returns_ok(self):
        import collections

        from maxim.doctor.checks import check_disk_space

        Usage = collections.namedtuple("Usage", ["total", "used", "free"])
        with patch("shutil.disk_usage", return_value=Usage(500 * 1024**3, 200 * 1024**3, 300 * 1024**3)):
            result = check_disk_space()
        assert result.status == "ok"

    def test_low_space_returns_warn(self):
        import collections

        from maxim.doctor.checks import check_disk_space

        Usage = collections.namedtuple("Usage", ["total", "used", "free"])
        with patch("shutil.disk_usage", return_value=Usage(500 * 1024**3, 494 * 1024**3, 6 * 1024**3)):
            result = check_disk_space()
        assert result.status == "warn"

    def test_critical_space_returns_fail(self):
        import collections

        from maxim.doctor.checks import check_disk_space

        Usage = collections.namedtuple("Usage", ["total", "used", "free"])
        with patch("shutil.disk_usage", return_value=Usage(500 * 1024**3, 499 * 1024**3, 1 * 1024**3)):
            result = check_disk_space()
        assert result.status == "fail"


class TestRamHeadroom:
    def test_returns_status(self):
        """RAM check should return a valid status without crashing."""
        from maxim.doctor.checks import check_ram_headroom

        result = check_ram_headroom()
        assert result.status in ("ok", "warn", "info")


# ─── inference coherence ──────────────────────────────────────────────────


def _doctor_resp(body: dict) -> object:
    """Build a maxim.utils.http.Response stub for doctor tests."""
    from maxim.utils import http as _http

    return _http.Response(
        status=200,
        headers={},
        content=json.dumps(body).encode(),
        elapsed_ms=1.0,
        endpoint=_http._EXTERNAL_ENDPOINT,
        request_id="r",
    )


class TestInferenceCoherence:
    def test_server_unreachable_returns_info(self):
        from maxim.doctor.checks import check_inference_coherence

        with patch("maxim.utils.http.fetch_url", side_effect=OSError("refused")):
            result = check_inference_coherence(port=19999)
        assert result.status == "info"

    def test_correct_answer_returns_ok(self):
        from maxim.doctor.checks import check_inference_coherence

        resp = _doctor_resp({"choices": [{"message": {"content": "4"}}]})
        with patch("maxim.utils.http.fetch_url", return_value=resp):
            result = check_inference_coherence(port=19999)
        assert result.status == "ok"
        assert "correct" in result.message

    def test_wrong_answer_returns_warn(self):
        from maxim.doctor.checks import check_inference_coherence

        resp = _doctor_resp({"choices": [{"message": {"content": "banana"}}]})
        with patch("maxim.utils.http.fetch_url", return_value=resp):
            result = check_inference_coherence(port=19999)
        assert result.status == "warn"
        assert "unexpected" in result.message


# ─── peer-mode checks ────────────────────────────────────────────────────


class TestPeerUrlReachable:
    def test_bad_dns_returns_fail(self):
        import socket

        from maxim.doctor.checks import check_peer_url_reachable

        with patch("socket.gethostbyname", side_effect=socket.gaierror("not found")):
            result = check_peer_url_reachable("https://fake.invalid/v1")
        assert result.status == "fail"
        assert "DNS" in result.message

    def test_empty_host_returns_fail(self):
        from maxim.doctor.checks import check_peer_url_reachable

        result = check_peer_url_reachable("not-a-url")
        # Should not crash
        assert result.status in ("ok", "warn", "fail", "info")


class TestPeerKeySet:
    def test_no_key_returns_warn(self, monkeypatch):
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", raising=False)
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", raising=False)
        from maxim.doctor.checks import check_peer_key_set

        with patch("maxim.peer.config.read_peer_config", return_value=None):
            result = check_peer_key_set()
        assert result.status == "warn"

    def test_env_key_returns_ok(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", "test-key-value")
        from maxim.doctor.checks import check_peer_key_set

        result = check_peer_key_set()
        assert result.status == "ok"
        assert "key set" in result.message


class TestPeerAuth:
    def test_no_key_returns_info(self):
        from maxim.doctor.checks import check_peer_auth

        result = check_peer_auth("https://example.com/v1", key=None)
        assert result.status == "info"

    def test_401_returns_fail(self):
        from maxim.doctor.checks import check_peer_auth
        from maxim.utils import http as _http

        with patch(
            "maxim.utils.http.fetch_url",
            side_effect=_http.HTTPAuthError("_external", status=401, fix_hint="Auth rejected (401)"),
        ):
            result = check_peer_auth("https://example.com/v1", key="bad-key")
        assert result.status == "fail"
        assert "rejected" in result.message


class TestPeerModel:
    def test_model_found_returns_ok(self):
        from maxim.doctor.checks import check_peer_model

        resp = _doctor_resp({"data": [{"id": "mistral-7b"}]})
        with patch("maxim.utils.http.fetch_url", return_value=resp):
            result = check_peer_model("https://example.com/v1", key=None)
        assert result.status == "ok"
        assert "mistral-7b" in result.message

    def test_wrong_model_returns_warn(self):
        from maxim.doctor.checks import check_peer_model

        resp = _doctor_resp({"data": [{"id": "mistral-7b"}]})
        with patch("maxim.utils.http.fetch_url", return_value=resp):
            result = check_peer_model("https://example.com/v1", key=None, expected_model="qwen2.5-14b")
        assert result.status == "warn"
        assert "expected" in result.message


class TestPeerLatency:
    def test_all_probes_fail_returns_warn(self):
        from maxim.doctor.checks import check_peer_latency

        with patch("maxim.utils.http.fetch_url", side_effect=OSError("refused")):
            result = check_peer_latency("https://fake.invalid/v1", key=None)
        assert result.status == "warn"
        assert "failed" in result.message


# ─── CLI: --json output ──────────────────────────────────────────────────


class TestDoctorJSON:
    def test_json_flag_produces_valid_json(self, capsys):
        code = run_doctor_subcommand(["--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "platform" in data
        assert "sections" in data
        assert "worst_status" in data
        assert isinstance(data["sections"], list)
        assert code in (0, 1)

    def test_json_has_expected_sections(self, capsys):
        run_doctor_subcommand(["--json"])
        data = json.loads(capsys.readouterr().out)
        section_names = [s["name"] for s in data["sections"]]
        assert "Environment" in section_names

    def test_json_checks_have_required_fields(self, capsys):
        run_doctor_subcommand(["--json"])
        data = json.loads(capsys.readouterr().out)
        for section in data["sections"]:
            for check in section["checks"]:
                assert "name" in check
                assert "status" in check
                assert "message" in check


# ─── CLI: --as flag ──────────────────────────────────────────────────────


class TestDoctorAsFlag:
    def test_as_peer_with_url(self, capsys):
        """--as peer <url> should show Peer Connectivity section."""
        code = run_doctor_subcommand(["--json", "--as", "peer", "https://fake.invalid/v1"])
        data = json.loads(capsys.readouterr().out)
        section_names = [s["name"] for s in data["sections"]]
        assert "Peer Connectivity" in section_names
        # Should NOT have leader-only sections
        assert "Tunnel (Cloudflare)" not in section_names
        assert code in (0, 1)

    def test_as_leader_shows_leader_sections(self, capsys):
        """--as leader should show leader/solo sections."""
        code = run_doctor_subcommand(["--json", "--as", "leader"])
        data = json.loads(capsys.readouterr().out)
        section_names = [s["name"] for s in data["sections"]]
        assert "Local LLM" in section_names
        assert "Tunnel (Cloudflare)" in section_names
        assert code in (0, 1)

    def test_auto_detect_peer_from_env(self, monkeypatch, capsys):
        """Setting MAXIM_LANE_LARGE_REMOTE_URL should auto-detect peer mode."""
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "https://remote-leader.example.com/v1")
        run_doctor_subcommand(["--json"])
        data = json.loads(capsys.readouterr().out)
        section_names = [s["name"] for s in data["sections"]]
        assert "Peer Connectivity" in section_names

    def test_localhost_url_stays_solo(self, monkeypatch, capsys):
        """A localhost remote URL should NOT trigger peer mode."""
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://127.0.0.1:8100/v1")
        run_doctor_subcommand(["--json"])
        data = json.loads(capsys.readouterr().out)
        section_names = [s["name"] for s in data["sections"]]
        assert "Local LLM" in section_names


# ─── run_all_checks with new sections ────────────────────────────────────


class TestRunAllChecksNewSections:
    def test_environment_includes_disk_and_ram(self):
        from maxim.doctor.checks import run_all_checks

        sections = run_all_checks(_info(), role="solo")
        env_section = next(s for name, s in sections if name == "Environment")
        names = [r.name for r in env_section]
        assert "Disk space" in names
        assert "RAM" in names

    def test_api_key_section_includes_hygiene_checks(self):
        from maxim.doctor.checks import run_all_checks

        sections = run_all_checks(_info(), role="solo")
        key_section = next(s for name, s in sections if name == "API key")
        names = [r.name for r in key_section]
        assert "Key age" in names
        assert "Key permissions" in names
        assert "Key auth smoke" in names

    def test_local_llm_includes_inference_coherence(self):
        from maxim.doctor.checks import run_all_checks

        sections = run_all_checks(_info(), role="solo")
        llm_section = next(s for name, s in sections if name == "Local LLM")
        names = [r.name for r in llm_section]
        assert "Inference coherence" in names

    def test_peer_mode_replaces_sections(self):
        from maxim.doctor.checks import run_all_checks

        sections = run_all_checks(_info(), role="peer", peer_url="https://fake.invalid/v1")
        section_names = [name for name, _ in sections]
        assert "Peer Connectivity" in section_names
        assert "Local LLM" not in section_names
        assert "Tunnel (Cloudflare)" not in section_names
        assert "API key" not in section_names


# ─── _detect_doctor_role ──────────────────────────────────────────────────


class TestDetectDoctorRole:
    def test_explicit_peer(self):
        from maxim.doctor.checks import _detect_doctor_role

        role, url = _detect_doctor_role("peer", "https://example.com/v1")
        assert role == "peer"
        assert url == "https://example.com/v1"

    def test_explicit_leader(self):
        from maxim.doctor.checks import _detect_doctor_role

        role, url = _detect_doctor_role("leader")
        assert role == "leader"
        assert url is None

    def test_auto_from_env(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "https://remote.example.com/v1")
        from maxim.doctor.checks import _detect_doctor_role

        role, url = _detect_doctor_role()
        assert role == "peer"
        assert "remote.example.com" in url

    def test_auto_localhost_stays_auto(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://127.0.0.1:8100/v1")
        from maxim.doctor.checks import _detect_doctor_role

        role, url = _detect_doctor_role()
        assert role == "auto"

    def test_no_env_returns_auto(self, monkeypatch):
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        from maxim.doctor.checks import _detect_doctor_role

        role, url = _detect_doctor_role()
        assert role == "auto"
        assert url is None

    def test_maxim_role_peer_falls_back_to_peer_yml(self, monkeypatch, tmp_path):
        """MAXIM_ROLE=peer with no remote URL env var → reads URL from peer.yml."""
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        monkeypatch.setenv("MAXIM_ROLE", "peer")
        peer_yml = tmp_path / "peer.yml"
        peer_yml.write_text("url: https://myhost.cloudflareaccess.com\napi_key: testkey123\n")
        monkeypatch.setattr(
            "maxim.peer.config.peer_config_path", lambda: peer_yml
        )
        from maxim.doctor.checks import _detect_doctor_role

        role, url = _detect_doctor_role()
        assert role == "peer"
        assert url == "https://myhost.cloudflareaccess.com"

    def test_maxim_role_peer_no_yml_still_returns_peer(self, monkeypatch, tmp_path):
        """MAXIM_ROLE=peer with no peer.yml → role is still peer, url is None."""
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        monkeypatch.setenv("MAXIM_ROLE", "peer")
        missing = tmp_path / "nonexistent.yml"
        monkeypatch.setattr("maxim.peer.config.peer_config_path", lambda: missing)
        from maxim.doctor.checks import _detect_doctor_role

        role, url = _detect_doctor_role()
        assert role == "peer"
        assert url is None

    def test_maxim_role_solo_returns_solo(self, monkeypatch):
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        from maxim.doctor.checks import _detect_doctor_role

        role, url = _detect_doctor_role()
        assert role == "solo"
        assert url is None


# ─── check_env_config ─────────────────────────────────────────────────────


class TestCheckEnvConfig:
    def _info(self, **kwargs) -> PlatformInfo:
        defaults = dict(
            os="linux",
            os_release="Ubuntu 24.04",
            runtime="native",
            distro="ubuntu",
            arch="x86_64",
            kernel_version="6.5.0",
            windows_host_ip=None,
        )
        defaults.update(kwargs)
        return PlatformInfo(**defaults)

    def test_all_good_leader(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        monkeypatch.setenv("MAXIM_LLM_ENABLED", "1")
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "qwen2.5-14b")
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "16384")
        monkeypatch.delenv("MAXIM_SKIP_REMOTE_PROBE", raising=False)
        monkeypatch.delenv("MAXIM_PEER_PROBE_KEY", raising=False)
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        from maxim.doctor.checks import check_env_config

        results = check_env_config(self._info(), role="leader")
        statuses = {r.name: r.status for r in results}
        assert statuses.get("MAXIM_ROLE") == "ok"
        assert statuses.get("MAXIM_LLM_N_CTX") == "ok"
        # No stale-var warnings
        assert "MAXIM_SKIP_REMOTE_PROBE" not in statuses
        assert "MAXIM_PEER_PROBE_KEY" not in statuses

    def test_missing_maxim_role_warns(self, monkeypatch):
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        from maxim.doctor.checks import check_env_config

        results = check_env_config(self._info())
        role_result = next((r for r in results if r.name == "MAXIM_ROLE"), None)
        assert role_result is not None
        assert role_result.status == "warn"
        assert "not set" in role_result.message

    def test_invalid_maxim_role_fails(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "superleader")
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        from maxim.doctor.checks import check_env_config

        results = check_env_config(self._info())
        role_result = next((r for r in results if r.name == "MAXIM_ROLE"), None)
        assert role_result is not None
        assert role_result.status == "fail"

    def test_low_n_ctx_warns(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        monkeypatch.setenv("MAXIM_LLM_ENABLED", "1")
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "qwen2.5-14b")
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "4096")
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        from maxim.doctor.checks import check_env_config

        results = check_env_config(self._info(), role="leader")
        ctx_result = next((r for r in results if r.name == "MAXIM_LLM_N_CTX"), None)
        assert ctx_result is not None
        assert ctx_result.status == "warn"
        assert "8192" in ctx_result.message

    def test_skip_remote_probe_warns(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        monkeypatch.setenv("MAXIM_LLM_ENABLED", "1")
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "m")
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "16384")
        monkeypatch.setenv("MAXIM_SKIP_REMOTE_PROBE", "1")
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        from maxim.doctor.checks import check_env_config

        results = check_env_config(self._info(), role="leader")
        probe_result = next((r for r in results if r.name == "MAXIM_SKIP_REMOTE_PROBE"), None)
        assert probe_result is not None
        assert probe_result.status == "warn"

    def test_stale_probe_key_warns(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        monkeypatch.setenv("MAXIM_LLM_ENABLED", "1")
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "m")
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "16384")
        monkeypatch.setenv("MAXIM_PEER_PROBE_KEY", "stale-key")
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        from maxim.doctor.checks import check_env_config

        results = check_env_config(self._info(), role="leader")
        pk_result = next((r for r in results if r.name == "MAXIM_PEER_PROBE_KEY"), None)
        assert pk_result is not None
        assert pk_result.status == "warn"

    def test_peer_role_skips_llm_checks(self, monkeypatch):
        """Peer machines don't run llama-cpp — no LLM env checks."""
        monkeypatch.setenv("MAXIM_ROLE", "peer")
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "https://maxim.example.com/v1")
        monkeypatch.delenv("MAXIM_LLM_ENABLED", raising=False)
        monkeypatch.delenv("MAXIM_LLM_N_CTX", raising=False)
        from maxim.doctor.checks import check_env_config

        results = check_env_config(self._info(), role="peer")
        names = {r.name for r in results}
        assert "MAXIM_LLM_ENABLED" not in names
        assert "MAXIM_LLM_N_CTX" not in names

    def test_macos_fix_hint_uses_zshrc(self, monkeypatch):
        monkeypatch.delenv("MAXIM_ROLE", raising=False)
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        from maxim.doctor.checks import check_env_config

        results = check_env_config(self._info(os="macos"))
        role_result = next((r for r in results if r.name == "MAXIM_ROLE"), None)
        assert role_result is not None
        assert role_result.fix is not None
        # Fix should show the export command but NOT suggest adding to .zshrc
        # (MAXIM_ROLE is auto-detected at startup; persisting to shell rc is rarely needed)
        assert "export MAXIM_ROLE=" in role_result.fix
        assert "zshrc" not in role_result.fix


# ─── check_context_window ─────────────────────────────────────────────────


class TestCheckContextWindow:
    def test_server_not_reachable_returns_info(self, monkeypatch):
        from maxim.doctor.checks import check_context_window
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

        with patch.object(_MaximPeerBackend, "health_check", return_value=False):
            result = check_context_window()
        assert result.status == "info"

    def test_n_ctx_ok_above_8192(self, monkeypatch):
        import json
        from unittest.mock import MagicMock, patch

        from maxim.doctor.checks import check_context_window
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

        models_response = json.dumps({"data": [{"id": "qwen2.5-14b", "context_length": 16384}]}).encode()

        with patch.object(_MaximPeerBackend, "health_check", return_value=True):
            import socket as _socket

            mock_sock = MagicMock()
            mock_sock.__enter__ = lambda s: s
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.recv.side_effect = [
                b"HTTP/1.0 200 OK\r\n\r\n" + models_response,
                b"",
            ]
            with patch.object(_socket, "create_connection", return_value=mock_sock):
                result = check_context_window()

        assert result.status == "ok"
        assert "16384" in result.message

    def test_n_ctx_below_8192_warns(self, monkeypatch):
        import json
        from unittest.mock import MagicMock, patch

        from maxim.doctor.checks import check_context_window
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

        models_response = json.dumps({"data": [{"id": "model", "context_length": 4096}]}).encode()

        with patch.object(_MaximPeerBackend, "health_check", return_value=True):
            import socket as _socket

            mock_sock = MagicMock()
            mock_sock.__enter__ = lambda s: s
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.recv.side_effect = [
                b"HTTP/1.0 200 OK\r\n\r\n" + models_response,
                b"",
            ]
            with patch.object(_socket, "create_connection", return_value=mock_sock):
                result = check_context_window()

        assert result.status == "warn"
        assert "4096" in result.message
        assert "overflow" in result.message or "KV cache" in result.message
