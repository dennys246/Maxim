"""Unit tests for ReachyMiniController connection flexibility (host/tunnel/mode).

Regression guard for the robots.yaml-driven connection options folded into the
controller (docs/plans/deferred/reachy_orient_live.md): an explicit `host` bypasses the
mDNS hard-gate that fails on macOS/hotspot; `tunnel` forces the localhost_only
SSH-tunnel path; defaults preserve the legacy network+mDNS behavior.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from maxim.hardware.reachy.controller import ReachyMiniController


import pytest


@pytest.fixture(autouse=True)
def _no_daemon_status_io(monkeypatch):
    """connect() now calls _daemon_status (urllib GET) — never let unit tests
    do real network I/O (3s blackhole timeouts per test; on a dev box with
    anything on :8000 the test would consume a real HTTP response)."""
    monkeypatch.setattr(ReachyMiniController, "_daemon_status", staticmethod(lambda *a, **k: None))
    # Era gate reads the REAL installed reachy-mini metadata — pin it to the
    # WS era so tests don't depend on what happens to be installed. (The
    # era-gate tests override this per-test.)
    monkeypatch.setattr(ReachyMiniController, "_installed_sdk_version", staticmethod(lambda: (1, 8)))
    monkeypatch.setattr(ReachyMiniController, "_get_state_full", _state_from_the_mock)


def _state_from_the_mock(self):
    """The daemon's REST state (#1499 R1), served from the mock SDK: the pose its last goto_target commanded."""
    import numpy as np

    state = {"control_mode": "enabled", "head_pose": {"m": np.eye(4).ravel().tolist()}, "body_yaw": 0.0}
    goto = getattr(self._mini, "goto_target", None)
    kwargs = getattr(getattr(goto, "call_args", None), "kwargs", None) or {}
    if kwargs.get("head") is not None:
        state["head_pose"] = {"m": np.asarray(kwargs["head"], dtype=float).ravel().tolist()}
    if kwargs.get("body_yaw") is not None:
        state["body_yaw"] = float(kwargs["body_yaw"])
    if kwargs.get("antennas") is not None:
        state["antennas_position"] = list(kwargs["antennas"])
    return state


def _mock_sdk():
    """(context-manager, mock_module) with reachy_mini importable as a mock.

    The SDK's interpolation module is the verbatim copy: the controller's path
    sampler (#1499 R5) runs it.
    """
    import types

    from tests.unit._reachy_fakes import linear_pose_interpolation

    mod = MagicMock()
    interpolation = types.ModuleType("reachy_mini.utils.interpolation")
    interpolation.linear_pose_interpolation = linear_pose_interpolation
    return patch.dict(sys.modules, {"reachy_mini": mod, "reachy_mini.utils.interpolation": interpolation}), mod


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


def test_mdns_path_probe_failure_skips_sdk_connect():
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


# ── WS-era (SDK >= 1.5) transport invariants — 2026-07-15 pivot fold ──────────
# The zenoh->WebSocket pivot (docs/embodiment/reachy_mini/README.md) changed the
# probe port (7447->8000), the tunnel target, and added host=/port= threading
# into the constructor. The pre-existing tests above all passed UNCHANGED
# across that rewrite (they mock one level too high to see ports/kwargs), so
# these pin the WS-era specifics that would otherwise regress silently.


def test_probe_targets_daemon_port_8000_not_7447():
    c = ReachyMiniController(host="10.6.0.63", connection_mode="network")
    ctx, mod = _mock_sdk()
    probed: list[tuple[str, int]] = []
    with (
        ctx,
        patch.object(
            ReachyMiniController,
            "_port_open",
            side_effect=lambda host, port, timeout=3.0: probed.append((host, port)) or True,
        ),
        patch.object(ReachyMiniController, "_daemon_status", return_value=None),
    ):
        c.connect(timeout=2.0)
    assert ("10.6.0.63", 8000) in probed
    assert all(port != 7447 for _, port in probed), "zenoh-era :7447 probe resurfaced"


def test_explicit_host_threaded_into_sdk_constructor():
    c = ReachyMiniController(host="10.6.0.63", connection_mode="network")
    ctx, mod = _mock_sdk()
    with (
        ctx,
        patch.object(ReachyMiniController, "_port_open", return_value=True),
        patch.object(ReachyMiniController, "_daemon_status", return_value=None),
    ):
        assert c.connect(timeout=2.0) is True
    kwargs = mod.ReachyMini.call_args.kwargs
    assert kwargs["host"] == "10.6.0.63"
    assert kwargs["port"] == 8000


def test_mdns_resolved_ip_passed_to_sdk_not_dot_local():
    """No-host path: the OS-resolved IP goes into ReachyMini(host=...) — python's
    own .local resolution is unreliable on macOS (the 2026-07-15 failure mode)."""
    c = ReachyMiniController(connection_mode="network")
    ctx, mod = _mock_sdk()
    with (
        ctx,
        patch.object(ReachyMiniController, "_resolve_mdns", return_value="10.6.0.63"),
        patch.object(ReachyMiniController, "_port_open", return_value=True),
        patch.object(ReachyMiniController, "_daemon_status", return_value=None),
    ):
        assert c.connect(timeout=2.0) is True
    assert mod.ReachyMini.call_args.kwargs["host"] == "10.6.0.63"


def test_tunnel_forwards_port_8000():
    c = ReachyMiniController(host="10.6.0.63", tunnel=True)
    captured: dict = {}

    def _fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return MagicMock()

    with (
        patch("subprocess.Popen", side_effect=_fake_popen),
        patch.object(ReachyMiniController, "_wait_local_port", return_value=True),
    ):
        assert c._start_ssh_tunnel(timeout=1.0) is True
    assert "8000:127.0.0.1:8000" in captured["cmd"]
    assert not any("7447" in str(part) for part in captured["cmd"]), "zenoh-era tunnel target resurfaced"


def test_localhost_only_connects_via_loopback_host():
    c = ReachyMiniController(host="10.6.0.63", connection_mode="localhost_only")
    ctx, mod = _mock_sdk()
    with (
        ctx,
        patch.object(ReachyMiniController, "_port_open", return_value=True),
        patch.object(ReachyMiniController, "_daemon_status", return_value=None),
    ):
        assert c.connect(timeout=2.0) is True
    assert mod.ReachyMini.call_args.kwargs["host"] == "127.0.0.1"


def test_pre_pivot_sdk_fails_loud_with_fix_hint(monkeypatch):
    """Era gate: a zenoh-era SDK (< 1.5) sneaking into the env must fail LOUD
    at connect() with the upgrade hint — not a cryptic TypeError on host=.
    (The 1.2.6 cap held through the entire zenoh->WS pivot; this pins the
    guard that makes the next stale-SDK situation diagnosable.)"""
    c = ReachyMiniController(host="10.6.0.63", connection_mode="network")
    ctx, mod = _mock_sdk()
    monkeypatch.setattr(ReachyMiniController, "_installed_sdk_version", staticmethod(lambda: (1, 2)))
    with ctx:
        assert c.connect(timeout=2.0) is False
    mod.ReachyMini.assert_not_called()
    assert "pre-v1.5.0" in (c.state.error_message or "")


def test_ws_era_sdk_passes_era_gate(monkeypatch):
    c = ReachyMiniController(host="10.6.0.63", connection_mode="network")
    ctx, mod = _mock_sdk()
    monkeypatch.setattr(ReachyMiniController, "_installed_sdk_version", staticmethod(lambda: (1, 8)))
    with (
        ctx,
        patch.object(ReachyMiniController, "_port_open", return_value=True),
    ):
        assert c.connect(timeout=2.0) is True


class TestAutomaticBodyYaw:
    """The frame thief (2026-08-03): the daemon's automatic_body_yaw
    (SDK default True) rotated the body −25° behind the runtime, putting
    every focus_on_sound aim off by up to that much with a PERFECT sensor
    — the exact 'daemon modulates body_yaw behind you' failure the
    head-frame invariant warns about. Maxim owns the yaw axis: default
    False at SDK construction AND re-asserted after wake."""

    def test_default_disables_daemon_body_follow_at_construction(self):
        c = ReachyMiniController(host="10.42.0.1", connection_mode="network")
        ctx, mod = _mock_sdk()
        with ctx, patch.object(ReachyMiniController, "_port_open", return_value=True):
            assert c.connect()
        assert mod.ReachyMini.call_args.kwargs["automatic_body_yaw"] is False

    def test_robots_yaml_opt_in_is_a_config_error(self):
        """Inverted (O6): ``automatic_body_yaw: true`` used to pass through to the SDK."""
        from maxim.exceptions import InvalidConfigError

        ctx, mod = _mock_sdk()
        with ctx, pytest.raises(InvalidConfigError):
            c = ReachyMiniController(host="10.42.0.1", connection_mode="network", automatic_body_yaw=True)
            c.connect()
        mod.ReachyMini.assert_not_called()

    def test_wake_reasserts_the_setting(self):
        c = ReachyMiniController(host="10.42.0.1", connection_mode="network")
        ctx, mod = _mock_sdk()
        with ctx, patch.object(ReachyMiniController, "_port_open", return_value=True):
            assert c.connect()
            mini = mod.ReachyMini.return_value
            assert c.wake_up()
        mini.set_automatic_body_yaw.assert_called_with(False)

    def test_wake_fails_loudly_when_motor_enable_fails(self):
        """Torque failure cannot be reported as an operational robot."""
        c = ReachyMiniController(host="10.42.0.1", connection_mode="network")
        ctx, mod = _mock_sdk()
        with ctx, patch.object(ReachyMiniController, "_port_open", return_value=True):
            assert c.connect()
            mini = mod.ReachyMini.return_value
            mini.enable_motors.side_effect = RuntimeError("torque unavailable")
            assert not c.wake_up()

        mini.wake_up.assert_not_called()
        mini.goto_sleep.assert_not_called()  # inverted (5.3, Executor S6): the failure branch commands no motion
        mini.goto_target.assert_not_called()
        mini.disable_motors.assert_called_once_with()
        assert c.state.is_awake is False

    def test_wake_refuses_on_an_sdk_without_the_setter(self):
        """Inverted (O6, delta D4): if ``set_automatic_body_yaw(False)`` cannot be asserted, the daemon may rotate the
        body behind us, so the controller refuses motion until an assertion succeeds. It used to warn and wake."""
        c = ReachyMiniController(host="10.42.0.1", connection_mode="network")
        ctx, mod = _mock_sdk()
        with ctx, patch.object(ReachyMiniController, "_port_open", return_value=True):
            assert c.connect()
            mini = mod.ReachyMini.return_value
            mini.set_automatic_body_yaw.side_effect = AttributeError("old SDK")
            assert not c.wake_up()
        mini.goto_target.assert_not_called()


class TestStopLatchAcrossReconnect:
    """#1499 owner decisions 2026-10-29: a sleep's latch survives a reconnect (only a successful wake ends it);
    a disconnect's own latch ends with the new connection."""

    def _connected(self):
        from maxim.hardware.controller import MotionTarget

        c = ReachyMiniController(host="10.42.0.1", connection_mode="network")
        ctx, mod = _mock_sdk()
        return c, ctx, MotionTarget

    def test_a_sleep_latch_survives_a_reconnect(self):
        c, ctx, MotionTarget = self._connected()
        with ctx, patch.object(ReachyMiniController, "_port_open", return_value=True):
            assert c.connect()
            assert c.goto_sleep()
            c.disconnect()
            assert c.connect()
            out = c.goto_target(MotionTarget(head_yaw=0.1))
            assert not out and out.kind == "cancelled"
            assert c.wake_up()
            assert c.goto_target(MotionTarget(head_yaw=0.1))

    def test_a_disconnect_latch_ends_with_the_new_connection(self):
        c, ctx, MotionTarget = self._connected()
        with ctx, patch.object(ReachyMiniController, "_port_open", return_value=True):
            assert c.connect()
            c.disconnect()
            assert c.connect()
            assert c.goto_target(MotionTarget(head_yaw=0.1))
