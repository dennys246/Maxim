"""Unit tests for system metrics collector and heartbeat monitor."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from maxim.runtime.system_metrics import (
    collect_all,
    collect_cpu,
    collect_disk,
    collect_memory,
    collect_network_interfaces,
    collect_platform,
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
