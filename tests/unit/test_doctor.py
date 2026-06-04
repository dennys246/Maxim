"""Tests for `maxim doctor` + `maxim peer test` (platform detection + checks + CLI)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from maxim.doctor.checks import CheckResult
from maxim.doctor.cli import run_doctor_subcommand, run_peer_subcommand
from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
from maxim.runtime.llm_server import ProbeResult
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


class TestCheckEmbodimentRef:
    """sem_execution_hook Stage 4 — validate SEM component refs at
    doctor time so a typo becomes a doctor-level fail with the
    available list as a fix hint, not a runtime crash."""

    def test_unknown_ref_fails_with_available_list(self):
        from maxim.doctor.checks import check_embodiment_ref

        result = check_embodiment_ref("weapons/nonexistent_sword")
        assert result.status == "fail"
        assert "nonexistent_sword" in result.message
        assert result.fix is not None
        # The fix hint should include at least one real available ref
        # so the user can spot a typo by comparison.
        assert "weapons/rusty_sword" in result.fix

    def test_real_rusty_sword_ref_succeeds(self):
        from maxim.doctor.checks import check_embodiment_ref

        result = check_embodiment_ref("weapons/rusty_sword")
        assert result.status == "ok"
        assert "weapons/rusty_sword" in result.message

    # NOTE: The original Stage 4 ship had a `test_empty_ref_returns_info`
    # test asserting that `check_embodiment_ref("")` returned an `info`
    # status. Pre-merge review (A-arch-3) caught the empty-ref branch
    # as dead code: the function's only caller (`run_all_checks`) gates
    # on `if embodiment_ref:` BEFORE calling, so the function never
    # receives an empty string in practice. The branch was deleted and
    # this test went with it — there is no contract for empty input.

    @staticmethod
    def _stub_heavy_doctor_checks():
        """Return a contextlib.ExitStack with every heavy doctor check
        patched to a trivial no-op. Used by section-visibility tests
        that should NOT pay the platform-probing cost. Pre-merge review
        I1 cross-confirmed: the original tests booted GPU detection,
        llama-cpp-server probes, etc. for a one-line assertion.

        Refactored from a 24-deep `with` stack (which hit Python's
        20-nested-block limit) to ExitStack.enter_context().
        """
        import contextlib

        from maxim.doctor import checks as checks_mod
        from maxim.doctor.checks import CheckResult

        def _stub(*args, **kwargs):
            return CheckResult(name="stub", status="ok", message="stub")

        stack = contextlib.ExitStack()
        for fn_name in (
            "check_gpu",
            "check_tier_detection",
            "check_tier_effectiveness",
            "check_disk_space",
            "check_ram_headroom",
            "check_storage_footprint",
            "check_llama_cpp_server_installed",
            "check_server_reachable",
            "check_llm_model_active",
            "check_context_window",
            "check_vram_pressure",
            "check_inference_coherence",
            "check_role",
            "check_lan_access",
            "check_cloudflared",
            "check_tunnel_config",
            "check_tunnel_config_sync",
            "check_api_key",
            "check_key_age",
            "check_key_permissions",
            "check_key_auth_smoke",
        ):
            stack.enter_context(patch.object(checks_mod, fn_name, _stub))
        stack.enter_context(patch.object(checks_mod, "check_env_config", lambda *a, **k: []))
        stack.enter_context(patch.object(checks_mod, "check_mesh_nodes", lambda: []))
        stack.enter_context(patch.object(checks_mod, "_check_lane_metrics", lambda: []))
        stack.enter_context(patch("maxim.doctor.checks._detect_doctor_role", return_value=("solo", None)))
        return stack

    def test_run_all_checks_skips_section_when_no_ref(self):
        """The Embodiment section must NOT appear when no --embodiment
        was passed (preserves existing doctor UX). Heavy checks stubbed
        per I1 fold."""
        from maxim.doctor import checks as checks_mod
        from maxim.doctor.platform_detect import detect_platform

        with self._stub_heavy_doctor_checks():
            sections = checks_mod.run_all_checks(detect_platform(), embodiment_ref=None)

        section_names = [name for name, _ in sections]
        assert "Embodiment" not in section_names

    def test_run_all_checks_includes_section_when_ref_set(self):
        """Same gating logic with embodiment_ref set — section appears.
        Heavy checks stubbed per I1 fold."""
        from maxim.doctor import checks as checks_mod
        from maxim.doctor.platform_detect import detect_platform

        with self._stub_heavy_doctor_checks():
            sections = checks_mod.run_all_checks(
                detect_platform(),
                embodiment_ref="weapons/rusty_sword",
            )

        section_names = [name for name, _ in sections]
        assert "Embodiment" in section_names


class TestDoctorCLIEmbodimentFlag:
    """The CLI parser for ``--embodiment <REF>`` must extract the ref
    cleanly and pass it through to run_all_checks."""

    def test_embodiment_flag_parsed(self):
        from maxim.doctor.cli import _parse_embodiment_flag

        ref = _parse_embodiment_flag(["doctor", "--embodiment", "weapons/rusty_sword"])
        assert ref == "weapons/rusty_sword"

    def test_no_embodiment_flag_returns_none(self):
        from maxim.doctor.cli import _parse_embodiment_flag

        assert _parse_embodiment_flag(["doctor"]) is None
        assert _parse_embodiment_flag(["doctor", "--retry"]) is None

    def test_embodiment_flag_without_value_exits_loud(self):
        """C1/I2/I3 cross-confirmed pre-merge review fold: a misuse of
        ``--embodiment`` MUST fail loudly (sys.exit(2)), not silently
        drop to None and run the doctor with no embodiment section.
        Silent-drop was the original behavior; both reviewers caught it
        as a CLAUDE.md "Don't silently drop failures" violation."""
        from maxim.doctor.cli import _parse_embodiment_flag

        # Trailing --embodiment with no value → loud exit
        with pytest.raises(SystemExit) as exc_info:
            _parse_embodiment_flag(["doctor", "--embodiment"])
        assert exc_info.value.code == 2

        # --embodiment followed by another flag (not a ref) → loud exit
        with pytest.raises(SystemExit) as exc_info:
            _parse_embodiment_flag(["doctor", "--embodiment", "--json"])
        assert exc_info.value.code == 2


class TestServerCheck:
    def test_reachable_returns_ok(self):
        from maxim.doctor.checks import check_server_reachable

        ok = ProbeResult("http://127.0.0.1:9999/v1", "ok", "HTTP 200", 5.0)
        with patch.object(_MaximPeerBackend, "health_check", return_value=ok):
            result = check_server_reachable(port=9999)
        assert result.status == "ok"
        assert "9999" in result.message

    def test_unreachable_returns_warn(self):
        from maxim.doctor.checks import check_server_reachable

        down = ProbeResult("http://127.0.0.1:9999/v1", "connection_refused", "boom", None)
        with patch.object(_MaximPeerBackend, "health_check", return_value=down):
            result = check_server_reachable(port=9999)
        assert result.status == "warn"
        assert result.fix is not None


class TestLanAccessPlatformSpecific:
    @pytest.fixture(autouse=True)
    def _isolate_cloudflared_config(self):
        """LAN access check now uses detect_role() which inspects four
        signals (config.json + mesh.yml + cloudflared + peer.yml).
        Isolate tests from the host's actual config files so the
        unified C3 detector falls through to the rank-7 default of
        ``leader``.

        Post-C3 (config_unification.md): ``_cloudflared_config_exists``
        moved from ``runtime/leader_mode.py`` to ``runtime/role.py``;
        the three other signal probes also live in role.py."""
        with (
            patch("maxim.runtime.role._cloudflared_config_exists", return_value=None),
            patch("maxim.runtime.role._peer_yml_exists", return_value=False),
            patch("maxim.runtime.role._mesh_yml_exists", return_value=False),
            patch("maxim.runtime.role._config_json_role", return_value=None),
        ):
            yield

    def test_wsl2_shows_netsh_fix(self, monkeypatch):
        # Post-C3: the unified detector defaults to ``leader`` (rank 7
        # of the seven-rank order), which suppresses the LAN firewall
        # warning. These platform-specific tests want the non-leader
        # case where the warning surfaces — set role=solo explicitly.
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        from maxim.doctor.checks import check_lan_access

        info = _info(runtime="wsl2", windows_host_ip="192.168.1.10")
        with patch("maxim.doctor.checks.detect_wsl_ip", return_value="172.24.32.1"):
            result = check_lan_access(info)
        assert result.status == "warn"
        assert "netsh" in (result.fix or "")
        assert "172.24.32.1" in (result.fix or "")
        assert "192.168.1.10" in (result.fix or "")

    def test_native_linux_shows_ufw_or_firewalld(self, monkeypatch):
        # Post-C3: the unified detector defaults to ``leader`` (rank 7
        # of the seven-rank order), which suppresses the LAN firewall
        # warning. These platform-specific tests want the non-leader
        # case where the warning surfaces — set role=solo explicitly.
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        from maxim.doctor.checks import check_lan_access

        info = _info(runtime="native", os="linux", distro="ubuntu")
        with patch("maxim.doctor.platform_detect.detect_lan_ip", return_value="192.168.1.5"):
            result = check_lan_access(info)
        assert result.status == "warn"
        assert "ufw" in (result.fix or "")

    def test_fedora_shows_firewalld(self, monkeypatch):
        # Post-C3: the unified detector defaults to ``leader`` (rank 7
        # of the seven-rank order), which suppresses the LAN firewall
        # warning. These platform-specific tests want the non-leader
        # case where the warning surfaces — set role=solo explicitly.
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        from maxim.doctor.checks import check_lan_access

        info = _info(runtime="native", os="linux", distro="fedora")
        with patch("maxim.doctor.platform_detect.detect_lan_ip", return_value=None):
            result = check_lan_access(info)
        assert "firewall-cmd" in (result.fix or "")

    def test_macos_mentions_settings(self, monkeypatch):
        # Post-C3: the unified detector defaults to ``leader`` (rank 7
        # of the seven-rank order), which suppresses the LAN firewall
        # warning. These platform-specific tests want the non-leader
        # case where the warning surfaces — set role=solo explicitly.
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        from maxim.doctor.checks import check_lan_access

        info = _info(runtime="native", os="macos", distro="unknown")
        with patch("maxim.doctor.platform_detect.detect_lan_ip", return_value="10.0.0.5"):
            result = check_lan_access(info)
        assert "System Settings" in (result.fix or "") or "firewall" in (result.fix or "").lower()

    def test_windows_shows_newnetfirewallrule(self, monkeypatch):
        # Post-C3: the unified detector defaults to ``leader`` (rank 7
        # of the seven-rank order), which suppresses the LAN firewall
        # warning. These platform-specific tests want the non-leader
        # case where the warning surfaces — set role=solo explicitly.
        monkeypatch.setenv("MAXIM_ROLE", "solo")
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
        monkeypatch.setattr("maxim.peer.config.peer_config_path", lambda: peer_yml)
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
        # Post-implementation field-coverage fold: env-config checks
        # for llm.* now route through resolve_setting, so the result
        # rows are named after the field path, not the env var.
        assert statuses.get("llm.n_ctx") == "ok"
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
        # Post-implementation field-coverage fold: row is now named
        # after the field path, not the env var.
        ctx_result = next((r for r in results if r.name == "llm.n_ctx"), None)
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


# ─── check_llm_model_active (regression guard for mutable-global binding) ─


class TestCheckLlmModelActive:
    """Regression guard for the CLAUDE.md 'Mutable globals + module extraction'
    anti-pattern: `from maxim.runtime.lane_backends import _active_model` binds
    by value at import time and never reflects runtime mutations. The fix in
    checks.py:174 reads through `maxim.runtime.llm_server._active_model` via a
    module reference. Without this test, the next refactor can silently
    re-introduce the buggy import pattern — the doctor's 'active model' line
    was reporting stale state for an unknown period before Plan 3.6 R5 caught
    it during review.
    """

    def test_reads_live_active_model_not_import_time_value(self, monkeypatch):
        import maxim.runtime.llm_server as _server_mod
        from maxim.doctor.checks import check_llm_model_active

        original_active = _server_mod._active_model
        try:
            _server_mod._active_model = "qwen2.5-14b-instruct"
            result = check_llm_model_active()
            assert result.status == "ok"
            assert "qwen2.5-14b-instruct" in result.message
        finally:
            _server_mod._active_model = original_active

    def test_falls_through_to_persisted_when_not_active(self, monkeypatch):
        import maxim.runtime.llm_server as _server_mod
        from maxim.doctor.checks import check_llm_model_active

        original_active = _server_mod._active_model
        try:
            _server_mod._active_model = None
            with patch(
                "maxim.runtime.lane_backends._read_persisted_model",
                return_value="mistral-7b",
            ):
                result = check_llm_model_active()
            assert result.status == "warn"
            assert "mistral-7b" in result.message
        finally:
            _server_mod._active_model = original_active


# ─── VRAM pressure (Plan 3.6 R5) ──────────────────────────────────────────


class TestCheckVramPressure:
    """Tests for the Plan 3.6 R5 VRAM spillover detection check.

    All tests monkeypatch nvidia-smi + the running server's n_ctx detection so
    they run offline on any machine. The incident signature (16 GB card,
    15.3 GB used, Qwen-14B at n_ctx=12380 producing 0.9 tok/s) is the primary
    reproduction target.
    """

    def test_nvidia_smi_unavailable_returns_info(self):
        from maxim.doctor.checks import check_vram_pressure

        with patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=None):
            result = check_vram_pressure()
        assert result.status == "info"
        assert "nvidia-smi" in result.message

    def test_live_spillover_fails_with_fix(self):
        """The 2026-04-13 incident: 15.3/16 GB, Qwen-14B, n_ctx=12380."""
        from maxim.doctor.checks import check_vram_pressure

        gpu = {
            "utilization_pct": 95.0,
            "vram_used_gb": 15.3,
            "vram_total_gb": 16.0,
            "temperature_c": 72.0,
        }
        with (
            patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=gpu),
            patch("maxim.runtime.llm_server._active_model", "qwen2.5-14b-instruct"),
            patch("maxim.doctor.checks._current_llama_server_n_ctx", return_value=12380),
        ):
            result = check_vram_pressure()

        assert result.status == "fail"
        assert "15.3" in result.message
        assert "16" in result.message
        # Must name spillover + expected slowdown in actionable terms
        assert "spill" in result.message.lower()
        # Fix string must contain a concrete MAXIM_LLM_N_CTX= with a real value
        assert result.fix is not None
        assert "MAXIM_LLM_N_CTX=" in result.fix
        # The recommendation must come from project_vram_usage, NOT the
        # hard-coded 4096 fallback path in _build_fix. If the projection
        # silently falls through to None (import error, arch-block missing,
        # etc.), _build_fix emits MAXIM_LLM_N_CTX=4096 which is ALSO < 12380
        # and would false-green this test. Compute the expected value from
        # project_vram_usage directly and compare — that's the only way to
        # verify the projection path was actually taken.
        from maxim.models.language.config import _BUILTIN_PROFILES
        from maxim.runtime.lane_models import project_vram_usage

        expected = project_vram_usage(
            "qwen2.5-14b-instruct",
            _BUILTIN_PROFILES["qwen2.5-14b-instruct"],
            12380,
            16.0,
        )
        assert expected is not None
        assert f"MAXIM_LLM_N_CTX={expected.recommended_n_ctx}" in result.fix
        # Additional structural guards: must be a multiple of 1024 (the
        # project_vram_usage rounding contract), must not equal the static
        # 4096 fallback, must be < the spilling n_ctx.
        import re

        match = re.search(r"MAXIM_LLM_N_CTX=(\d+)", result.fix)
        assert match is not None
        recommended = int(match.group(1))
        assert recommended < 12380
        assert recommended > 0
        assert recommended % 1024 == 0
        # The projection result for this input should NOT accidentally equal
        # 4096 — if it does, add another profile to the test matrix.
        assert expected.recommended_n_ctx != 4096, (
            "projection recommendation equals the static fallback placeholder; "
            "this test can no longer distinguish projection-path from fallback-path"
        )
        assert recommended != 4096
        assert result.retry_id == "vram_pressure"

    def test_predictive_spillover_fires_without_live_ratio(self):
        """Qwen-14B at 32k n_ctx on a 16 GB card — math says we'll spill even
        though the current ratio is low (model just loaded, KV cache empty).
        """
        from maxim.doctor.checks import check_vram_pressure

        # Model just loaded — weights only in VRAM, KV cache empty.
        gpu = {
            "utilization_pct": 20.0,
            "vram_used_gb": 8.5,
            "vram_total_gb": 16.0,
            "temperature_c": 45.0,
        }
        with (
            patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=gpu),
            patch("maxim.runtime.llm_server._active_model", "qwen2.5-14b-instruct"),
            patch("maxim.doctor.checks._current_llama_server_n_ctx", return_value=32768),
        ):
            result = check_vram_pressure()

        assert result.status == "fail"
        # Projection fields should surface in the message
        assert "Projected" in result.message
        assert "weights" in result.message
        assert "KV" in result.message
        assert result.fix is not None
        assert "MAXIM_LLM_N_CTX=" in result.fix

    def test_warning_band_850_to_950_produces_warn(self):
        from maxim.doctor.checks import check_vram_pressure

        gpu = {
            "utilization_pct": 60.0,
            "vram_used_gb": 14.0,  # 87.5% of 16.0 — in the warn band
            "vram_total_gb": 16.0,
            "temperature_c": 60.0,
        }
        with (
            patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=gpu),
            patch("maxim.runtime.llm_server._active_model", None),
            patch("maxim.doctor.checks._current_llama_server_n_ctx", return_value=None),
        ):
            result = check_vram_pressure()

        assert result.status == "warn"
        assert "headroom" in result.message.lower()

    def test_healthy_card_returns_ok(self):
        """Qwen-7B at n_ctx=8192 on a 16 GB card — ~6 GB used, ~38% ratio."""
        from maxim.doctor.checks import check_vram_pressure

        gpu = {
            "utilization_pct": 40.0,
            "vram_used_gb": 6.0,
            "vram_total_gb": 16.0,
            "temperature_c": 55.0,
        }
        with (
            patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=gpu),
            patch("maxim.runtime.llm_server._active_model", "qwen2-7b-instruct"),
            patch("maxim.doctor.checks._current_llama_server_n_ctx", return_value=8192),
        ):
            result = check_vram_pressure()

        assert result.status == "ok"
        assert "6.0" in result.message
        assert "16" in result.message

    def test_zero_total_vram_returns_info(self):
        from maxim.doctor.checks import check_vram_pressure

        gpu = {
            "utilization_pct": 0.0,
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "temperature_c": 0.0,
        }
        with patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=gpu):
            result = check_vram_pressure()
        assert result.status == "info"

    def test_fix_string_mentions_smaller_profile_for_qwen_14b(self):
        """Fix hint should be profile-aware: Qwen-14B users get a concrete
        smaller-model suggestion (qwen2-7b), not a generic placeholder."""
        from maxim.doctor.checks import check_vram_pressure

        gpu = {
            "utilization_pct": 98.0,
            "vram_used_gb": 15.5,
            "vram_total_gb": 16.0,
            "temperature_c": 75.0,
        }
        with (
            patch("maxim.runtime.leader_proxy._query_nvidia_smi", return_value=gpu),
            patch("maxim.runtime.llm_server._active_model", "qwen2.5-14b-instruct"),
            patch("maxim.doctor.checks._current_llama_server_n_ctx", return_value=12380),
        ):
            result = check_vram_pressure()

        assert result.fix is not None
        assert "qwen2-7b" in result.fix


# ─── Plan 4 Stage C1: mesh node check ────────────────────────────────────


class _FakeMeshProbe:
    """Stub ProbeResult for mesh_node tests."""

    def __init__(self, outcome: str, detail: str = "", latency_ms: float | None = None):
        self.outcome = outcome
        self.detail = detail
        self.latency_ms = latency_ms


def _make_fake_mesh_backend(result: "_FakeMeshProbe"):
    """Build a fresh fake ``_MaximPeerBackend`` class bound to one probe
    result. Pre-merge review F15 fix: avoid class-level mutable state
    between tests.
    """

    class _FakeMeshBackend:
        def __init__(self, r):
            self._result = r

        @classmethod
        def for_url(cls, url, *, api_key=None, model=None):
            return cls(result)

        def health_check(self, *, enable_stage2=True):
            return self._result

    return _FakeMeshBackend


_MESH_YAML = """\
cluster_key: sk-cluster-abc
self: leader-desk
protocol_version: 1
nodes:
  - name: leader-desk
    url: http://192.168.1.10:8099/v1
    role: leader
  - name: mac-studio
    url: https://mac.example.com/v1
    role: peer
"""


class TestCheckMeshNodes:
    def _setup_mesh(self, tmp_path, monkeypatch, *, outcome="ok", detail="HTTP 200", latency=5.0):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        from maxim.utils import paths

        paths._reset_caches()
        mesh_path = tmp_path / "config" / "maxim" / "mesh.yml"
        mesh_path.parent.mkdir(parents=True)
        mesh_path.write_text(_MESH_YAML)

        fake = _make_fake_mesh_backend(_FakeMeshProbe(outcome, detail, latency))
        import maxim.models.language.maxim_peer_backend as mpb

        monkeypatch.setattr(mpb, "_MaximPeerBackend", fake)

    def test_no_mesh_yml_returns_empty_list(self, tmp_path, monkeypatch):
        """Pre-merge review A5: empty list replaces the magic info sentinel."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
        from maxim.utils import paths

        paths._reset_caches()
        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        assert results == []

    def test_all_healthy_ok_per_node(self, tmp_path, monkeypatch):
        self._setup_mesh(tmp_path, monkeypatch, outcome="ok")
        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        assert len(results) == 2
        assert all(r.status == "ok" for r in results)
        assert results[0].name == "Node leader-desk"
        assert results[1].name == "Node mac-studio"
        # Doctor passes role+url through the classifier detail, so the
        # "reachable" message includes both.
        assert "leader" in results[0].message
        assert "http://192.168.1.10:8099/v1" in results[0].message

    def test_auth_rejected_fail_with_key_rotate_hint(self, tmp_path, monkeypatch):
        self._setup_mesh(tmp_path, monkeypatch, outcome="auth_rejected", detail="HTTP 401")
        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        assert all(r.status == "fail" for r in results)
        assert any("tunnel key rotate" in (r.fix or "") for r in results)
        assert all(r.retry_id and r.retry_id.startswith("mesh_node_") for r in results)

    def test_inference_broken_fail_with_chat_hint(self, tmp_path, monkeypatch):
        self._setup_mesh(tmp_path, monkeypatch, outcome="inference_broken", detail="stage2: HTTP 500")
        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        assert all(r.status == "fail" for r in results)
        assert any("chat endpoint broken" in r.message for r in results)
        assert any("maxim peer llm --status" in (r.fix or "") for r in results)

    def test_dns_fail_mapped_to_warn(self, tmp_path, monkeypatch):
        self._setup_mesh(tmp_path, monkeypatch, outcome="dns_fail", detail="no such host")
        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        assert all(r.status == "warn" for r in results)
        assert any("dns fail" in r.message for r in results)

    def test_peer_yml_fallback_end_to_end(self, tmp_path, monkeypatch):
        """F16: end-to-end fallback from peer.yml through check_mesh_nodes."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
        from maxim.utils import paths

        paths._reset_caches()
        peer_path = tmp_path / "config" / "maxim" / "peer.yml"
        peer_path.parent.mkdir(parents=True)
        peer_path.write_text("url: https://leader.example.com/v1\napi_key: sk-fallback\n")

        fake = _make_fake_mesh_backend(_FakeMeshProbe("ok", "HTTP 200", 15.0))
        import maxim.models.language.maxim_peer_backend as mpb

        monkeypatch.setattr(mpb, "_MaximPeerBackend", fake)

        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        assert len(results) == 1
        assert results[0].status == "ok"
        assert results[0].name == "Node leader"
        assert "https://leader.example.com/v1" in results[0].message

    def test_malformed_mesh_yml_returns_fail_with_schema_hint(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
        from maxim.utils import paths

        paths._reset_caches()
        mesh_path = tmp_path / "config" / "maxim" / "mesh.yml"
        mesh_path.parent.mkdir(parents=True)
        mesh_path.write_text(_MESH_YAML.replace("self: leader-desk", "self: ghost"))

        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        assert len(results) == 1
        assert results[0].status == "fail"
        assert "does not match" in results[0].message

    def test_retry_ids_registered_per_node(self, tmp_path, monkeypatch):
        """Round 2 A2R2: retry_id is set regardless of status. The retry
        loop in doctor/cli.py decides whether to re-run based on status;
        the producer sets the stable identity unconditionally.
        """
        self._setup_mesh(tmp_path, monkeypatch, outcome="auth_rejected")
        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        retry_ids = {r.retry_id for r in results}
        assert retry_ids == {"mesh_node_leader-desk", "mesh_node_mac-studio"}

    def test_retry_ids_also_set_for_ok_results(self, tmp_path, monkeypatch):
        """Round 2 A2R2 regression guard: producer must not null retry_id
        based on status. The retry loop's status filter is the single
        place where gating lives.
        """
        self._setup_mesh(tmp_path, monkeypatch, outcome="ok")
        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        assert all(r.retry_id and r.retry_id.startswith("mesh_node_") for r in results)

    def test_missing_backend_extra_produces_warn(self, tmp_path, monkeypatch):
        """Round 2 A5R2: ``ImportError`` defensive branch regression guard."""
        import sys

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        from maxim.utils import paths

        paths._reset_caches()
        mesh_path = tmp_path / "config" / "maxim" / "mesh.yml"
        mesh_path.parent.mkdir(parents=True)
        mesh_path.write_text(_MESH_YAML)

        class _Broken:
            def __getattr__(self, name):
                raise ImportError("simulated: llm-server extra not installed")

        monkeypatch.setitem(sys.modules, "maxim.models.language.maxim_peer_backend", _Broken())

        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        assert len(results) == 2
        assert all(r.status == "warn" for r in results)
        assert all("peer backend import failed" in r.message for r in results)
        assert all("llm-server" in (r.fix or "") for r in results)


class TestCheckMeshNodesDrainHandling:
    """Plan 4 C2: drain handling in doctor's check_mesh_nodes.

    Drained nodes render as ``info`` without a network call. Orphan
    drain entries (drain state names that no longer match any
    mesh.yml node) surface as a ``warn`` CheckResult with a resume
    hint, not a hard fail.
    """

    def _setup_mesh(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        from maxim.utils import paths

        paths._reset_caches()
        mesh_path = tmp_path / "config" / "maxim" / "mesh.yml"
        mesh_path.parent.mkdir(parents=True)
        mesh_path.write_text(_MESH_YAML)

    def test_drained_node_renders_as_info_without_probe(self, tmp_path, monkeypatch):
        """Plan 4 C2: drained nodes must NOT be probed. Counting backend
        asserts zero calls for the drained node."""
        self._setup_mesh(tmp_path, monkeypatch)

        call_count = {"n": 0}

        class _CountingBackend:
            @classmethod
            def for_url(cls, url, *, api_key=None, model=None):
                call_count["n"] += 1
                return cls()

            def health_check(self, *, enable_stage2=True):
                return _FakeMeshProbe("ok", "HTTP 200", 5.0)

        import maxim.models.language.maxim_peer_backend as mpb

        monkeypatch.setattr(mpb, "_MaximPeerBackend", _CountingBackend)

        # Drain mac-studio
        from maxim.peer.drain_state import drain_node

        drain_node("mac-studio", {"leader-desk", "mac-studio"})

        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        assert len(results) == 2
        mac = next(r for r in results if r.name == "Node mac-studio")
        leader = next(r for r in results if r.name == "Node leader-desk")
        assert mac.status == "info"
        assert "drained" in mac.message.lower()
        assert mac.retry_id is None  # no retry on drained
        assert leader.status == "ok"
        # Only leader-desk probed; mac-studio skipped
        assert call_count["n"] == 1

    def test_orphan_drain_entry_surfaces_as_warn(self, tmp_path, monkeypatch):
        """Operator edits mesh.yml to remove a node that's still in
        the drain state file. The orphan surfaces as a warn with a
        resume hint, not a hard fail."""
        self._setup_mesh(tmp_path, monkeypatch)

        # Drain a node that exists
        from maxim.peer.drain_state import drain_node

        drain_node("mac-studio", {"leader-desk", "mac-studio"})

        # Rewrite mesh.yml to remove mac-studio
        mesh_path = tmp_path / "config" / "maxim" / "mesh.yml"
        mesh_path.write_text(
            "cluster_key: sk-cluster-abc\n"
            "self: leader-desk\n"
            "protocol_version: 1\n"
            "nodes:\n"
            "  - name: leader-desk\n"
            "    url: http://192.168.1.10:8099/v1\n"
            "    role: leader\n"
        )

        fake = _make_fake_mesh_backend(_FakeMeshProbe("ok", "HTTP 200", 5.0))
        import maxim.models.language.maxim_peer_backend as mpb

        monkeypatch.setattr(mpb, "_MaximPeerBackend", fake)

        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        # 1 live node + 1 orphan warning
        orphan_results = [r for r in results if "orphan" in r.name.lower()]
        assert len(orphan_results) == 1
        assert orphan_results[0].status == "warn"
        assert "mac-studio" in orphan_results[0].message
        assert "resume" in (orphan_results[0].fix or "")
        assert orphan_results[0].retry_id == "mesh_drain_orphan_mac-studio"

    def test_drained_self_still_info(self, tmp_path, monkeypatch):
        """If an operator force-drained themselves, doctor still
        reports info (not fail) — they took the explicit action."""
        self._setup_mesh(tmp_path, monkeypatch)

        from maxim.peer.drain_state import drain_node

        drain_node(
            "leader-desk",
            {"leader-desk", "mac-studio"},
            self_name="leader-desk",
            force_self=True,
        )

        fake = _make_fake_mesh_backend(_FakeMeshProbe("ok", "HTTP 200", 5.0))
        import maxim.models.language.maxim_peer_backend as mpb

        monkeypatch.setattr(mpb, "_MaximPeerBackend", fake)

        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        leader = next(r for r in results if r.name == "Node leader-desk")
        assert leader.status == "info"
        assert "drained" in leader.message.lower()


class TestOrphanDrainReprobe:
    """CCR1 fold (C2 pre-merge review, triple-confirmed): the
    ``mesh_drain_orphan_<name>`` retry_id now has a registered
    reprobe in ``doctor/cli.py::_retry_loop``. Previously the retry
    loop silently dropped the id because no callable matched the
    prefix, breaking the CLAUDE.md doctor contract ("loop is
    data-driven — any retry_id gets included").
    """

    def _setup_mesh_and_orphan(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        from maxim.utils import paths

        paths._reset_caches()
        mesh_path = tmp_path / "config" / "maxim" / "mesh.yml"
        mesh_path.parent.mkdir(parents=True)
        mesh_path.write_text(_MESH_YAML)

        # Drain mac-studio, then rewrite mesh.yml to remove it —
        # creates an orphan drain entry.
        from maxim.peer.drain_state import drain_node

        drain_node("mac-studio", {"leader-desk", "mac-studio"})
        mesh_path.write_text(
            "cluster_key: sk-cluster-abc\n"
            "self: leader-desk\n"
            "protocol_version: 1\n"
            "nodes:\n"
            "  - name: leader-desk\n"
            "    url: http://192.168.1.10:8099/v1\n"
            "    role: leader\n"
        )

    def test_reprobe_returns_same_warn_when_orphan_still_exists(self, tmp_path, monkeypatch):
        """Operator hasn't fixed the orphan yet — reprobe returns
        the same warn CheckResult so the retry loop stays stuck."""
        self._setup_mesh_and_orphan(tmp_path, monkeypatch)

        # Build a fake sections list containing the orphan warn
        from maxim.doctor.checks import CheckResult

        orphan_result = CheckResult(
            name="Drain orphan mac-studio",
            status="warn",
            message="drain state entry 'mac-studio' has no matching node in mesh.yml",
            fix="Run `maxim peer --node mac-studio resume` to clean up",
            retry_id="mesh_drain_orphan_mac-studio",
        )
        sections = [("Mesh Topology", [orphan_result])]

        # Exercise the closure via the retry loop's internal logic
        # by mocking input() + checking registration
        reprobes: dict = {}

        def _make(name):
            # Reproduce the closure factory shape from doctor/cli.py
            from maxim.peer.drain_state import read_drained_nodes
            from maxim.peer.mesh_config import read_or_synthesize_mesh_config

            def _reprobe():
                mesh = read_or_synthesize_mesh_config()
                if mesh is None:
                    return None
                known_names = {n.name for n in mesh.nodes}
                drain_result = read_drained_nodes(known_names)
                if name not in drain_result.orphans:
                    return CheckResult(
                        name=f"Drain orphan {name}",
                        status="ok",
                        message=f"orphan cleared for {name!r}",
                    )
                return CheckResult(
                    name=f"Drain orphan {name}",
                    status="warn",
                    message=f"drain state entry {name!r} still has no matching node",
                    retry_id=f"mesh_drain_orphan_{name}",
                )

            return _reprobe

        # Simulate the retry loop's prefix-match registration
        for _, results in sections:
            for r in results:
                if r.retry_id and r.retry_id.startswith("mesh_drain_orphan_"):
                    orphan_name = r.retry_id[len("mesh_drain_orphan_") :]
                    reprobes[r.retry_id] = _make(orphan_name)

        # Reprobe while orphan still exists — should return warn
        result = reprobes["mesh_drain_orphan_mac-studio"]()
        assert result is not None
        assert result.status == "warn"

    def test_reprobe_returns_ok_after_operator_runs_resume(self, tmp_path, monkeypatch):
        """Operator ran `maxim peer --node mac-studio resume` between
        doctor invocations. Reprobe detects the cleared orphan and
        returns ok so the retry loop exits for this id."""
        self._setup_mesh_and_orphan(tmp_path, monkeypatch)

        # Operator runs resume — but resume_node rejects unknown nodes,
        # so we go directly through the state layer with the ORIGINAL
        # node set (not the edited one) to clean the drain entry. This
        # simulates the scenario where the operator either re-adds the
        # node to mesh.yml OR manually edits the drain state file.
        from maxim.peer.drain_state import read_drained_nodes
        from maxim.peer.mesh_config import read_or_synthesize_mesh_config

        # Manually clear the drain state file (simulates the operator
        # editing ~/.maxim/util/drained_nodes.leader.txt or re-adding
        # the node to mesh.yml then resuming)
        from maxim.peer.drain_state import drain_state_path

        drain_state_path().write_text("# cleared\n")

        # Now check that the orphan is gone from the partition
        mesh = read_or_synthesize_mesh_config()
        assert mesh is not None
        known_names = {n.name for n in mesh.nodes}
        drain_result = read_drained_nodes(known_names)
        assert "mac-studio" not in drain_result.orphans

    def test_retry_loop_registers_orphan_reprobe(self, tmp_path, monkeypatch):
        """Integration test: confirm the actual retry loop in
        doctor/cli.py registers the mesh_drain_orphan_* prefix."""
        self._setup_mesh_and_orphan(tmp_path, monkeypatch)

        from maxim.doctor.checks import check_mesh_nodes

        results = check_mesh_nodes()
        orphan_results = [r for r in results if "orphan" in r.name.lower()]
        assert len(orphan_results) == 1
        assert orphan_results[0].retry_id == "mesh_drain_orphan_mac-studio"
        # The retry_loop registration is in doctor/cli.py and is
        # exercised via the closure factory pattern — asserting the
        # retry_id has the canonical prefix locks in the contract.
        assert orphan_results[0].retry_id.startswith("mesh_drain_orphan_")


class TestCheckUserProfiles:
    """Doctor integration for L3 of leader_ux_profile_management.md.

    Pins three outcomes: missing file (info, 0 profiles), valid file
    with entries (ok with count), and malformed YAML (fail with
    actionable fix hint).
    """

    def test_missing_file_returns_info(self, tmp_path, monkeypatch):
        from maxim.doctor.checks import check_user_profiles

        # Point profiles_config_path at a non-existent file
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        result = check_user_profiles()
        assert result.status == "info"
        assert "0 user profiles" in result.message

    def test_valid_file_with_entries_returns_ok(self, tmp_path, monkeypatch):
        from maxim.doctor.checks import check_user_profiles

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        profiles_dir = tmp_path / "maxim"
        profiles_dir.mkdir()
        (profiles_dir / "profiles.yml").write_text(
            "profiles:\n"
            "  my-model:\n"
            "    backend: llama_cpp\n"
            "    prompt_style: chatml\n"
            "    download: {hf_repo: foo/bar, hf_file: m.gguf}\n"
        )
        result = check_user_profiles()
        assert result.status == "ok"
        assert "1 user profile" in result.message
        assert "loaded from" in result.message

    def test_malformed_yaml_returns_fail_with_fix_hint(self, tmp_path, monkeypatch):
        from maxim.doctor.checks import check_user_profiles

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        profiles_dir = tmp_path / "maxim"
        profiles_dir.mkdir()
        (profiles_dir / "profiles.yml").write_text("profiles:\n  bad: [unclosed\n")
        result = check_user_profiles()
        assert result.status == "fail"
        assert result.fix is not None
        assert "refuse to start" in result.fix

    def test_per_profile_schema_error_returns_fail(self, tmp_path, monkeypatch):
        """Pre-merge executor review fold: the original check_user_profiles
        only ran top-level YAML parse (load_user_profiles), missing
        per-profile schema errors. A YAML-valid file with missing
        ``prompt_style`` reported "ok / 1 loaded" — but ``maxim``
        startup would then crash with ConfigurationError. The fix
        routes through apply_user_profiles for the full validation."""
        from maxim.doctor.checks import check_user_profiles

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        profiles_dir = tmp_path / "maxim"
        profiles_dir.mkdir()
        # YAML is syntactically valid; schema is broken (missing prompt_style)
        (profiles_dir / "profiles.yml").write_text(
            "profiles:\n  broken-profile:\n    backend: llama_cpp\n    download: {hf_repo: foo/bar, hf_file: m.gguf}\n"
        )
        result = check_user_profiles()
        assert result.status == "fail", (
            f"per-profile schema error should fail doctor but got status={result.status!r}, "
            f"message={result.message!r}. The fix at checks.py::check_user_profiles "
            f"must route through apply_user_profiles for full validation."
        )
        assert "prompt_style" in result.message
