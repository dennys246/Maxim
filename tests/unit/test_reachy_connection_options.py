"""Unit tests for ReachyMiniController connection flexibility (host/tunnel/mode).

Regression guard for the robots.yaml-driven connection options folded into the
controller (docs/plans/reachy_orient_live.md): an explicit `host` bypasses the
mDNS hard-gate that fails on macOS/hotspot; `tunnel` forces the localhost_only
SSH-tunnel path; defaults preserve the legacy network+mDNS behavior.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from maxim.hardware.reachy.controller import ReachyMiniController


def _mock_sdk():
    """(context-manager, mock_module) with reachy_mini importable as a mock."""
    mod = MagicMock()
    return patch.dict(sys.modules, {"reachy_mini": mod}), mod


def test_new_kwargs_stored():
    c = ReachyMiniController(
        robot_id="r",
        connection_mode="localhost_only",
        host="10.42.0.1",
        tunnel=True,
        ssh_user="pollen",
        ssh_port=2222,
    )
    assert c._connection_mode == "localhost_only"
    assert c._host == "10.42.0.1"
    assert c._tunnel is True
    assert c._ssh_user == "pollen"
    assert c._ssh_port == 2222
    assert c._tunnel_proc is None


def test_defaults_preserve_legacy_behavior():
    c = ReachyMiniController()
    assert c._connection_mode == "network"
    assert c._host is None
    assert c._tunnel is False


def test_tunnel_without_host_fails_fast():
    c = ReachyMiniController(tunnel=True)  # no host
    ctx, mod = _mock_sdk()
    with ctx:
        assert c.connect(timeout=2.0) is False
    mod.ReachyMini.assert_not_called()


def test_host_bypasses_mdns_gate():
    c = ReachyMiniController(host="10.42.0.1", connection_mode="network")
    ctx, mod = _mock_sdk()
    with (
        ctx,
        patch.object(ReachyMiniController, "_resolve_mdns") as mdns,
        patch.object(ReachyMiniController, "_port_open", return_value=True),
    ):
        assert c.connect(timeout=2.0) is True
    mdns.assert_not_called()  # explicit host => mDNS gate skipped
    assert mod.ReachyMini.call_args.kwargs["connection_mode"] == "network"


def test_tunnel_forces_localhost_only():
    c = ReachyMiniController(host="10.42.0.1", tunnel=True)
    ctx, mod = _mock_sdk()
    with (
        ctx,
        patch.object(ReachyMiniController, "_start_ssh_tunnel", return_value=True) as tun,
        patch.object(ReachyMiniController, "_port_open", return_value=True),
    ):
        assert c.connect(timeout=2.0) is True
    tun.assert_called_once()
    assert mod.ReachyMini.call_args.kwargs["connection_mode"] == "localhost_only"


def test_localhost_only_probes_localhost_not_mdns():
    c = ReachyMiniController(connection_mode="localhost_only")
    ctx, mod = _mock_sdk()
    with (
        ctx,
        patch.object(ReachyMiniController, "_resolve_mdns") as mdns,
        patch.object(ReachyMiniController, "_port_open", return_value=True) as probe,
    ):
        assert c.connect(timeout=2.0) is True
    mdns.assert_not_called()
    assert probe.call_args.args[0] == "127.0.0.1"
    assert mod.ReachyMini.call_args.kwargs["connection_mode"] == "localhost_only"


def test_mdns_path_also_probes_7447():
    """Default network/mDNS path fails fast if :7447 is unreachable (name resolving
    is not enough — the daemon can be down / zenoh bound to localhost)."""
    c = ReachyMiniController()  # network, no host
    ctx, mod = _mock_sdk()
    with (
        ctx,
        patch.object(ReachyMiniController, "_resolve_mdns", return_value="10.0.0.9"),
        patch.object(ReachyMiniController, "_port_open", return_value=False),
    ):
        assert c.connect(timeout=2.0) is False
    mod.ReachyMini.assert_not_called()


def test_mdns_path_probes_the_resolved_ip():
    c = ReachyMiniController()
    ctx, mod = _mock_sdk()
    with (
        ctx,
        patch.object(ReachyMiniController, "_resolve_mdns", return_value="10.0.0.9"),
        patch.object(ReachyMiniController, "_port_open", return_value=True) as probe,
    ):
        assert c.connect(timeout=2.0) is True
    assert probe.call_args.args[0] == "10.0.0.9"  # probed the resolved IP, not the name
    assert mod.ReachyMini.call_args.kwargs["connection_mode"] == "network"


def test_disconnect_stops_tunnel():
    c = ReachyMiniController()
    proc = MagicMock()
    c._tunnel_proc = proc
    c.disconnect()
    proc.terminate.assert_called_once()
    assert c._tunnel_proc is None
