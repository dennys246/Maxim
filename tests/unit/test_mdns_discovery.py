"""Unit tests for mDNS discovery and ReachyMiniController connection logic."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest


from maxim.hardware.reachy import probe as reachy_probe


def _status_unreadable(*_a, **_k):
    raise reachy_probe.DaemonStatusUnreadable("unit test: no daemon")


@pytest.fixture(autouse=True)
def _no_daemon_status_io(monkeypatch):
    """connect()'s pre-check runs the one daemon probe (hardware/reachy/probe.py, #1806), whose status read is an
    HTTP GET — never let unit tests do real network I/O (3s blackhole timeout per call; on a dev box with anything
    on :8000 the test would consume a real HTTP response)."""
    monkeypatch.setattr(reachy_probe, "read_daemon_status", _status_unreadable)
    # Era gate reads the REAL installed reachy-mini metadata — pin it to the
    # WS era so tests don't depend on what happens to be installed. (The
    # era-gate tests override this per-test.)
    monkeypatch.setattr(reachy_probe, "installed_sdk_version", lambda: (1, 8))


def _port_open():
    return patch.object(reachy_probe, "tcp_connect_error", return_value=None)


class TestMdnsResolution:
    """The no-host name: ``<robot_name>.local`` (underscores dashed), resolved once by ``probe.resolve_ipv4``."""

    def _make_controller(self, robot_name: str = "reachy_mini") -> object:
        """Create a ReachyMiniController without connecting."""
        from maxim.hardware.reachy.controller import ReachyMiniController

        return ReachyMiniController(robot_name=robot_name)

    def test_resolve_success(self):
        """Successful mDNS resolution returns the IP the controller dials."""
        ctrl = self._make_controller("reachy_mini")

        with patch("socket.gethostbyname", return_value="192.168.50.150"), _port_open():
            host, _ = ctrl._precheck_daemon("network", 2.0)

        assert host == "192.168.50.150"

    def test_resolve_converts_underscores_to_hyphens(self):
        """Robot name underscores are converted to hyphens for mDNS."""
        ctrl = self._make_controller("reachy_mini")

        with patch("socket.gethostbyname", return_value="10.0.0.1") as mock_resolve, _port_open():
            ctrl._precheck_daemon("network", 2.0)

        mock_resolve.assert_called_once_with("reachy-mini.local")

    def test_resolve_appends_dot_local(self):
        """mDNS name gets .local suffix."""
        ctrl = self._make_controller("my-robot")

        with patch("socket.gethostbyname", return_value="10.0.0.2") as mock_resolve, _port_open():
            ctrl._precheck_daemon("network", 2.0)

        mock_resolve.assert_called_once_with("my-robot.local")

    def test_resolve_failure_returns_none(self):
        """Failed resolution: no host to dial, and the TCP stage never runs."""
        ctrl = self._make_controller("reachy_mini")

        with (
            patch("socket.gethostbyname", side_effect=socket.gaierror("not found")),
            patch.object(reachy_probe, "tcp_connect_error") as tcp,
        ):
            host, media_released = ctrl._precheck_daemon("network", 1.0)

        assert host is None and media_released is None
        tcp.assert_not_called()

    def test_resolve_restores_default_timeout(self):
        """Socket default timeout is restored after resolution."""
        original_timeout = socket.getdefaulttimeout()

        with patch("socket.gethostbyname", return_value="10.0.0.1"):
            assert reachy_probe.resolve_ipv4("reachy-mini.local", timeout=3.0) == "10.0.0.1"

        assert socket.getdefaulttimeout() == original_timeout

    def test_resolve_restores_timeout_on_failure(self):
        """Socket default timeout is restored even on failure."""
        original_timeout = socket.getdefaulttimeout()

        with patch("socket.gethostbyname", side_effect=socket.gaierror("fail")), pytest.raises(OSError):
            reachy_probe.resolve_ipv4("reachy-mini.local", timeout=3.0)

        assert socket.getdefaulttimeout() == original_timeout

    def test_resolve_no_underscores_in_name(self):
        """Robot name without underscores works correctly."""
        ctrl = self._make_controller("reachymini")

        with patch("socket.gethostbyname", return_value="10.0.0.3") as mock_resolve, _port_open():
            ctrl._precheck_daemon("network", 2.0)

        mock_resolve.assert_called_once_with("reachymini.local")

    def test_an_ipv4_literal_is_never_looked_up(self):
        with patch("socket.gethostbyname") as mock_resolve:
            assert reachy_probe.resolve_ipv4("10.6.0.63", timeout=1.0) == "10.6.0.63"
        mock_resolve.assert_not_called()


class TestControllerConnect:
    """Test ReachyMiniController.connect() integration with mDNS."""

    def _make_controller(self, robot_name: str = "reachy_mini") -> object:
        from maxim.hardware.reachy.controller import ReachyMiniController

        return ReachyMiniController(robot_name=robot_name)

    def _connect(self, timeout: float, **resolve) -> tuple[object, MagicMock, MagicMock, MagicMock]:
        import sys

        ctrl = self._make_controller()
        with (
            patch.dict("sys.modules", {"reachy_mini": MagicMock()}),
            patch("socket.gethostbyname", **resolve) as mock_mdns,
            patch.object(reachy_probe, "probe_daemon", wraps=reachy_probe.probe_daemon) as probe,
            _port_open(),
        ):
            mock_sdk_module = sys.modules["reachy_mini"]
            mock_sdk_module.ReachyMini.return_value = MagicMock()
            result = ctrl.connect(timeout=timeout)
        return result, mock_mdns, probe, mock_sdk_module

    def test_connect_calls_mdns_resolve(self):
        """connect() pre-resolves mDNS before SDK init."""
        result, mock_mdns, _, sdk = self._connect(5.0, return_value="192.168.50.150")
        mock_mdns.assert_called_once_with("reachy-mini.local")
        assert sdk.ReachyMini.call_args.kwargs["host"] == "192.168.50.150"

    def test_connect_caps_mdns_timeout_at_5s(self):
        """The probe's stage budget is capped at 5 s (+3 s for the status read) even with a larger connect timeout."""
        _, _, probe, _ = self._connect(30.0, return_value="192.168.50.150")
        assert probe.call_args.kwargs["deadline_s"] == 8.0

    def test_connect_uses_network_mode(self):
        """connect() passes connection_mode='network' to SDK."""
        _, _, _, sdk = self._connect(5.0, return_value="192.168.50.150")
        sdk.ReachyMini.assert_called_once()
        assert sdk.ReachyMini.call_args[1]["connection_mode"] == "network"

    def test_connect_skips_sdk_when_mdns_fails(self):
        """connect() returns False immediately when mDNS fails (no SDK attempt)."""
        result, _, _, sdk = self._connect(5.0, side_effect=socket.gaierror("not found"))
        assert result is False
        # SDK should NOT have been called since mDNS failed
        sdk.ReachyMini.assert_not_called()

    def test_connect_returns_false_on_import_error(self):
        """connect() returns False when reachy-mini SDK is not installed."""
        ctrl = self._make_controller()

        # Ensure reachy_mini is not importable
        with patch.dict("sys.modules", {"reachy_mini": None}):
            result = ctrl.connect(timeout=1.0)

        assert result is False


class TestConnectionConfigMigration:
    """Test that ConnectionConfig uses connection_mode correctly."""

    def test_default_connection_mode_is_network(self):
        """Default connection_mode is 'network'."""
        from maxim.embodied_runtime.connection import ConnectionConfig

        config = ConnectionConfig()
        assert config.connection_mode == "network"

    def test_custom_connection_mode(self):
        """Can set custom connection_mode."""
        from maxim.embodied_runtime.connection import ConnectionConfig

        config = ConnectionConfig(connection_mode="auto")
        assert config.connection_mode == "auto"

    def test_localhost_only_still_exists_for_compat(self):
        """localhost_only field still exists for backward compatibility."""
        from maxim.embodied_runtime.connection import ConnectionConfig

        config = ConnectionConfig()
        assert hasattr(config, "localhost_only")
