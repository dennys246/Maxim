"""Unit tests for mDNS discovery and ReachyMiniController connection logic."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest


class TestMdnsResolution:
    """Test _resolve_mdns method on ReachyMiniController."""

    def _make_controller(self, robot_name: str = "reachy_mini") -> object:
        """Create a ReachyMiniController without connecting."""
        from maxim.hardware.reachy.controller import ReachyMiniController

        return ReachyMiniController(robot_name=robot_name)

    def test_resolve_success(self):
        """Successful mDNS resolution returns IP."""
        ctrl = self._make_controller("reachy_mini")

        with patch("socket.gethostbyname", return_value="192.168.50.150"):
            ip = ctrl._resolve_mdns(timeout=2.0)

        assert ip == "192.168.50.150"

    def test_resolve_converts_underscores_to_hyphens(self):
        """Robot name underscores are converted to hyphens for mDNS."""
        ctrl = self._make_controller("reachy_mini")

        with patch("socket.gethostbyname") as mock_resolve:
            mock_resolve.return_value = "10.0.0.1"
            ctrl._resolve_mdns()

        mock_resolve.assert_called_once_with("reachy-mini.local")

    def test_resolve_appends_dot_local(self):
        """mDNS name gets .local suffix."""
        ctrl = self._make_controller("my-robot")

        with patch("socket.gethostbyname") as mock_resolve:
            mock_resolve.return_value = "10.0.0.2"
            ctrl._resolve_mdns()

        mock_resolve.assert_called_once_with("my-robot.local")

    def test_resolve_failure_returns_none(self):
        """Failed resolution returns None."""
        ctrl = self._make_controller("reachy_mini")

        with patch("socket.gethostbyname", side_effect=socket.gaierror("not found")):
            ip = ctrl._resolve_mdns(timeout=1.0)

        assert ip is None

    def test_resolve_restores_default_timeout(self):
        """Socket default timeout is restored after resolution."""
        ctrl = self._make_controller("reachy_mini")
        original_timeout = socket.getdefaulttimeout()

        with patch("socket.gethostbyname", return_value="10.0.0.1"):
            ctrl._resolve_mdns(timeout=3.0)

        assert socket.getdefaulttimeout() == original_timeout

    def test_resolve_restores_timeout_on_failure(self):
        """Socket default timeout is restored even on failure."""
        ctrl = self._make_controller("reachy_mini")
        original_timeout = socket.getdefaulttimeout()

        with patch("socket.gethostbyname", side_effect=socket.gaierror("fail")):
            ctrl._resolve_mdns(timeout=3.0)

        assert socket.getdefaulttimeout() == original_timeout

    def test_resolve_no_underscores_in_name(self):
        """Robot name without underscores works correctly."""
        ctrl = self._make_controller("reachymini")

        with patch("socket.gethostbyname") as mock_resolve:
            mock_resolve.return_value = "10.0.0.3"
            ctrl._resolve_mdns()

        mock_resolve.assert_called_once_with("reachymini.local")


class TestControllerConnect:
    """Test ReachyMiniController.connect() integration with mDNS."""

    def _make_controller(self, robot_name: str = "reachy_mini") -> object:
        from maxim.hardware.reachy.controller import ReachyMiniController

        return ReachyMiniController(robot_name=robot_name)

    @patch("maxim.hardware.reachy.controller.ReachyMiniController._resolve_mdns")
    def test_connect_calls_mdns_resolve(self, mock_mdns):
        """connect() pre-resolves mDNS before SDK init."""
        mock_mdns.return_value = "192.168.50.150"
        ctrl = self._make_controller()

        with patch.dict("sys.modules", {"reachy_mini": MagicMock()}):
            import sys

            mock_sdk_module = sys.modules["reachy_mini"]
            mock_mini = MagicMock()
            mock_sdk_module.ReachyMini.return_value = mock_mini

            ctrl.connect(timeout=5.0)

        mock_mdns.assert_called_once_with(timeout=5.0)

    @patch("maxim.hardware.reachy.controller.ReachyMiniController._resolve_mdns")
    def test_connect_caps_mdns_timeout_at_5s(self, mock_mdns):
        """mDNS timeout is capped at 5 seconds even with larger connect timeout."""
        mock_mdns.return_value = "192.168.50.150"
        ctrl = self._make_controller()

        with patch.dict("sys.modules", {"reachy_mini": MagicMock()}):
            import sys

            mock_sdk_module = sys.modules["reachy_mini"]
            mock_mini = MagicMock()
            mock_sdk_module.ReachyMini.return_value = mock_mini

            ctrl.connect(timeout=30.0)

        mock_mdns.assert_called_once_with(timeout=5.0)

    @patch("maxim.hardware.reachy.controller.ReachyMiniController._resolve_mdns")
    def test_connect_uses_network_mode(self, mock_mdns):
        """connect() passes connection_mode='network' to SDK."""
        mock_mdns.return_value = "192.168.50.150"
        ctrl = self._make_controller()

        with patch.dict("sys.modules", {"reachy_mini": MagicMock()}):
            import sys

            mock_sdk_module = sys.modules["reachy_mini"]
            mock_mini = MagicMock()
            mock_sdk_module.ReachyMini.return_value = mock_mini

            ctrl.connect(timeout=5.0)

            mock_sdk_module.ReachyMini.assert_called_once()
            call_kwargs = mock_sdk_module.ReachyMini.call_args[1]
            assert call_kwargs["connection_mode"] == "network"

    @patch("maxim.hardware.reachy.controller.ReachyMiniController._resolve_mdns")
    def test_connect_skips_sdk_when_mdns_fails(self, mock_mdns):
        """connect() returns False immediately when mDNS fails (no SDK attempt)."""
        mock_mdns.return_value = None
        ctrl = self._make_controller()

        with patch.dict("sys.modules", {"reachy_mini": MagicMock()}):
            import sys

            mock_sdk_module = sys.modules["reachy_mini"]
            mock_mini = MagicMock()
            mock_sdk_module.ReachyMini.return_value = mock_mini

            result = ctrl.connect(timeout=5.0)

        assert result is False
        # SDK should NOT have been called since mDNS failed
        mock_sdk_module.ReachyMini.assert_not_called()

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
        from maxim.conscience.connection import ConnectionConfig

        config = ConnectionConfig()
        assert config.connection_mode == "network"

    def test_custom_connection_mode(self):
        """Can set custom connection_mode."""
        from maxim.conscience.connection import ConnectionConfig

        config = ConnectionConfig(connection_mode="auto")
        assert config.connection_mode == "auto"

    def test_localhost_only_still_exists_for_compat(self):
        """localhost_only field still exists for backward compatibility."""
        from maxim.conscience.connection import ConnectionConfig

        config = ConnectionConfig()
        assert hasattr(config, "localhost_only")
