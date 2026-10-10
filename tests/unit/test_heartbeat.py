"""Unit tests for system metrics collector and heartbeat monitor."""

from __future__ import annotations

import json
import logging
import socket
import time
from unittest.mock import MagicMock, patch

import pytest

from maxim.runtime.system_metrics import (
    collect_all,
    collect_cpu,
    collect_disk,
    collect_memory,
    collect_network_interfaces,
    collect_platform,
    recordable_ip,
    short_hostname,
)
from maxim.runtime.heartbeat import HeartbeatMonitor


class TestSystemMetrics:
    """System metrics collection (graceful on all platforms)."""

    def test_collect_all_returns_dict(self) -> None:
        data = collect_all()
        assert isinstance(data, dict)
        assert "timestamp" in data
        assert "cpu" in data
        assert "memory" in data
        assert "disk" in data
        assert "platform" in data

    def test_collect_cpu(self) -> None:
        cpu = collect_cpu()
        assert cpu is not None
        assert "cores" in cpu
        assert cpu["cores"] > 0
        assert "load_1m" in cpu
        assert isinstance(cpu["load_1m"], float)

    def test_collect_memory(self) -> None:
        mem = collect_memory()
        assert mem is not None
        assert mem["total_gb"] > 0
        assert mem["used_gb"] >= 0
        assert 0 <= mem["usage_pct"] <= 100

    def test_macos_memory_falls_back_when_sysctl_is_blocked(self) -> None:
        from maxim.runtime.system_metrics import _memory_macos

        denied = MagicMock(returncode=1, stdout="")
        vm_stat = MagicMock(
            returncode=0,
            stdout=(
                "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 100.\nPages inactive: 200.\n"
            ),
        )
        page_size = MagicMock(returncode=1, stdout="")

        with (
            patch("maxim.runtime.system_metrics.subprocess.run", side_effect=[denied, vm_stat, page_size]),
            patch("maxim.runtime.system_metrics.os.sysconf", side_effect=[1_000_000, 4096]),
        ):
            mem = _memory_macos()

        assert mem is not None
        assert mem["total_gb"] > 0
        assert mem["used_gb"] >= 0

    def test_collect_disk(self) -> None:
        disk = collect_disk()
        assert disk is not None
        assert disk["total_gb"] > 0
        assert disk["free_gb"] >= 0
        assert "path" in disk

    def test_collect_network(self) -> None:
        net = collect_network_interfaces()
        assert net is not None
        assert "hostname" in net
        assert isinstance(net["hostname"], str)

    def test_collect_platform(self) -> None:
        plat = collect_platform()
        assert plat["system"] in ("Linux", "Darwin", "Windows")
        assert "pid" in plat
        assert plat["pid"] > 0

    def test_gpu_returns_none_without_nvidia(self) -> None:
        """GPU returns None gracefully when nvidia-smi isn't available."""
        from maxim.runtime.system_metrics import collect_gpu

        # May return None or a dict depending on hardware — just shouldn't crash
        result = collect_gpu()
        assert result is None or isinstance(result, dict)

    def test_wifi_doesnt_crash(self) -> None:
        """WiFi collection may return None but should never raise."""
        from maxim.runtime.system_metrics import collect_wifi_signal

        result = collect_wifi_signal()
        assert result is None or isinstance(result, dict)


class TestHeartbeatMonitor:
    """HeartbeatMonitor lifecycle and sampling."""

    def test_start_stop(self) -> None:
        monitor = HeartbeatMonitor(interval_s=0.1, stall_threshold_s=1.0)
        monitor.start()
        assert monitor._thread is not None
        assert monitor._thread.is_alive()
        monitor.stop()
        assert monitor._thread is None

    def test_double_start_is_idempotent(self) -> None:
        monitor = HeartbeatMonitor(interval_s=0.1)
        monitor.start()
        thread1 = monitor._thread
        monitor.start()
        assert monitor._thread is thread1  # same thread
        monitor.stop()

    def test_snapshot_returns_data(self) -> None:
        monitor = HeartbeatMonitor(interval_s=60)  # don't auto-sample
        data = monitor.snapshot()
        assert isinstance(data, dict)
        assert "cpu" in data
        assert "memory" in data
        assert "disk" in data
        assert "lanes" in data

    def test_loop_state_hook(self) -> None:
        monitor = HeartbeatMonitor(interval_s=60)
        monitor.set_loop_state_hook(lambda: {"idle_s": 5.0, "state": "active"})
        data = monitor.snapshot()
        assert data["loop"] == {"idle_s": 5.0, "state": "active"}

    def test_loop_state_hook_none_by_default(self) -> None:
        monitor = HeartbeatMonitor(interval_s=60)
        data = monitor.snapshot()
        assert data["loop"] is None

    def test_stall_detection(self) -> None:
        """Stall detection warns when loop idle exceeds threshold."""
        monitor = HeartbeatMonitor(interval_s=0.1, stall_threshold_s=0.5)
        monitor.set_loop_state_hook(lambda: {"idle_s": 2.0, "state": "stalled"})

        # Record some calls so stall detection activates
        from maxim.models.language.lane_metrics import LaneMetrics

        large = LaneMetrics(lane_name="large")
        large.record_call(100, success=True)

        # Mock the registry to return our large tier metrics
        data = monitor._collect()
        data["lanes"] = {"large": large.snapshot()}

        # Check stall — should set _last_stall_warn
        monitor._check_stall(data)
        assert monitor._last_stall_warn > 0

    def test_heartbeat_emits_without_crash(self) -> None:
        """A full heartbeat cycle (collect + emit) should never crash."""
        monitor = HeartbeatMonitor(interval_s=0.1)
        monitor.start()
        time.sleep(0.3)  # let at least 2 heartbeats fire
        monitor.stop()
        # If we got here without exception, the heartbeat is stable


# ── #1166: the host identity the heartbeat records (synthetic names and TEST-NET / documentation addresses) ──


FQDN = "box.example-isp.net"


@pytest.mark.parametrize(
    "raw, short",
    [
        (FQDN, "box"),
        ("box.", "box"),
        ("BOX.Example-ISP.NET.", "BOX"),
        ("box", "box"),
        ("c-203-0-113-7.example-isp.net", "ip-encoded"),
        ("ip-198-51-100-9.example-isp.net", "ip-encoded"),
        ("pool-192-0-2-44.example-isp.net", "ip-encoded"),
        ("203-0-113-7", "ip-encoded"),
        ("203.0.113.7", "ip-encoded"),
        ("2001:db8::7", "ip-encoded"),
        ("mac-mini-2024-10-01.example-isp.net", "mac-mini-2024-10-01"),  # a date is not an address
        ("", "unknown"),
        # three decimal groups already name a /24
        ("box-203-0-113.example-isp.net", "ip-encoded"),
        ("box-203-0-113", "ip-encoded"),
        # an address encoded in hex: a run of 8+ hex digits, or separated hex tokens that concatenate to 8+
        ("p5b0c1d2e.example-isp.net", "ip-encoded"),  # 5b0c1d2e
        ("cb007107.example-isp.net", "ip-encoded"),
        ("cb-00-71-07.example-isp.net", "ip-encoded"),
        ("CB-00-71-07", "ip-encoded"),
        ("host-20010db8000000000000000000000007", "ip-encoded"),  # 32 hex digits: an IPv6 address
        ("2001-db8-0-0-7.example-isp.net", "ip-encoded"),
        ("12345678", "ip-encoded"),  # an integer-form address
        # ordinary names stay
        ("big-mac-mini", "big-mac-mini"),
        ("big-mac-mini.example-isp.net", "big-mac-mini"),
        ("raspberrypi", "raspberrypi"),
        ("dennys-mbp", "dennys-mbp"),
        ("12345", "12345"),  # a bare all-digit name is a label, not an address
        ("ip-encoded", "ip-encoded"),
        ("unknown", "unknown"),
    ],
)
def test_short_hostname_keeps_the_machine_not_the_network(raw: str, short: str) -> None:
    assert short_hostname(raw) == short
    assert short_hostname(short) == short  # idempotent (no check relies on it: o19_rerun states its own rule)


@pytest.mark.parametrize(
    "raw, recorded",
    [
        ("10.0.0.5", "10.0.0.5"),
        ("192.168.1.20", "192.168.1.20"),
        ("127.0.0.1", "127.0.0.1"),
        ("169.254.3.4", "169.254.3.4"),
        ("fd00::5", "fd00::5"),
        ("fe80::1", "fe80::1"),
        ("203.0.113.7", "non-private"),  # stands in for a WAN address
        ("2001:db8::7", "non-private"),  # a global IPv6 (its /32 would name the ISP)
        ("100.64.0.9", "non-private"),  # CGNAT is not a private range here
        ("::ffff:10.0.0.5", "non-private"),  # IPv4-mapped: not in a kept range, whatever the IPv4 inside
        ("fe80::1%en0", "fe80::1%en0"),  # a scoped link-local keeps its zone (an interface name)
        ("unknown", "unknown"),
        ("not-an-address", "unknown"),
    ],
)
def test_recordable_ip_keeps_only_private_addresses(raw: str, recorded: str) -> None:
    assert recordable_ip(raw) == recorded
    assert recordable_ip(recorded) == recorded


@pytest.fixture
def isp_host(monkeypatch):
    """The DHCP-suffixed hostname macOS reports on some networks, resolving to a public address."""
    resolved: list[str] = []

    def by_name(name: str) -> str:
        resolved.append(name)
        return "203.0.113.7"

    monkeypatch.setattr(socket, "gethostname", lambda: FQDN)
    monkeypatch.setattr(socket, "gethostbyname", by_name)
    return resolved


def test_collect_network_records_the_short_name_and_no_public_address(isp_host) -> None:
    net = collect_network_interfaces()
    assert net == {"hostname": "box", "local_ip": "non-private"}
    assert isp_host == [FQDN]  # the RAW name resolves; only the recorded value is reduced


def test_collect_network_keeps_a_private_address(isp_host, monkeypatch) -> None:
    monkeypatch.setattr(socket, "gethostbyname", lambda name: "10.0.0.5")
    assert collect_network_interfaces() == {"hostname": "box", "local_ip": "10.0.0.5"}


def test_the_logged_heartbeat_carries_no_full_hostname(isp_host, tmp_path) -> None:
    """The record as MAXIM_LOG_FILE writes it (the StructuredFormatter), not only the collector's return."""
    from maxim.utils.structured_logging import StructuredFormatter

    log = logging.getLogger("maxim.heartbeat")
    handler = logging.FileHandler(tmp_path / "run_log.jsonl", encoding="utf-8")
    handler.setFormatter(StructuredFormatter())
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        monitor = HeartbeatMonitor(interval_s=60)
        monitor._emit(monitor._collect())
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)
        handler.close()
    text = (tmp_path / "run_log.jsonl").read_text()
    assert "example-isp" not in text and "203.0.113.7" not in text
    beats = [json.loads(ln) for ln in text.splitlines() if json.loads(ln).get("e") == "heartbeat"]
    assert beats and all(b["network"] == {"hostname": "box", "local_ip": "non-private"} for b in beats)


IWCONFIG = 'wlan0     IEEE 802.11  ESSID:"Example-Home-Net"\n          Link Quality=60/70  Signal level=-50 dBm\n'
AIRPORT = "     agrCtlRSSI: -55\n     agrCtlNoise: -90\n          SSID: Example-Home-Net\n       channel: 36\n"


@pytest.mark.parametrize("system, out", [("Linux", IWCONFIG), ("Darwin", AIRPORT)])
def test_wifi_records_signal_but_never_the_ssid(monkeypatch, system: str, out: str) -> None:
    from maxim.runtime import system_metrics

    monkeypatch.setattr(system_metrics.platform, "system", lambda: system)
    monkeypatch.setattr(
        system_metrics.subprocess, "run", lambda *a, **k: MagicMock(returncode=0, stdout=out, stderr="")
    )
    wifi = system_metrics.collect_wifi_signal()
    assert wifi is not None and wifi["rssi_dbm"] in (-50.0, -55.0)
    assert "ssid" not in wifi and "Example-Home-Net" not in json.dumps(wifi)


def test_wifi_linux_without_an_association_records_nothing(monkeypatch) -> None:
    from maxim.runtime import system_metrics

    monkeypatch.setattr(system_metrics.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        system_metrics.subprocess, "run", lambda *a, **k: MagicMock(returncode=0, stdout="lo  no wireless", stderr="")
    )
    assert system_metrics.collect_wifi_signal() is None
