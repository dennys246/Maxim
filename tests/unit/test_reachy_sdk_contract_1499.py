"""Contract pins between #1499's motion-path model and the real reachy-mini SDK (skipped without the ``reachy`` extra).

The unit lane has no SDK, so the controller's path sampler runs against a verbatim
copy of ``linear_pose_interpolation`` (tests/unit/_reachy_fakes.py), the lifecycle
steps against copied poses, and the fresh-readback parser against a modelled REST
body. These pin each copy to the installed SDK, so an SDK upgrade that changes one
fails here instead of silently diverging. They skip wherever the SDK is absent,
which today is every CI lane (the extras matrix has no reachy cell yet; R2's
reachy cell, Session S's line, will run them); they run on any dev box or rig with
the ``reachy`` extra installed.
docs/plans/motion_path_model.md R1, R5, O8; gate 24.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

reachy_mini = pytest.importorskip("reachy_mini")


def test_the_fake_interpolation_is_the_sdks():
    from reachy_mini.utils.interpolation import linear_pose_interpolation as sdk_interp
    from scipy.spatial.transform import Rotation as R

    from tests.unit._reachy_fakes import linear_pose_interpolation as copy_interp

    rng = np.random.default_rng(1499)
    for _ in range(50):
        a, b = np.eye(4), np.eye(4)
        a[:3, :3] = R.random(random_state=int(rng.integers(1 << 30))).as_matrix()
        b[:3, :3] = R.random(random_state=int(rng.integers(1 << 30))).as_matrix()
        a[:3, 3], b[:3, 3] = rng.normal(size=3) * 0.02, rng.normal(size=3) * 0.02
        for s in (0.0, 0.25, 0.5, 0.9, 1.0):
            np.testing.assert_allclose(copy_interp(a, b, s), sdk_interp(a, b, s), atol=1e-12)


def test_the_controller_lifecycle_poses_are_the_sdks():
    """O8: the controller composes the SDK's wake/sleep steps itself, from copied poses."""
    from reachy_mini import reachy_mini as sdk

    from maxim.hardware.reachy import controller as ctl

    np.testing.assert_allclose(ctl.INIT_HEAD_POSE, sdk.INIT_HEAD_POSE)
    np.testing.assert_allclose(ctl.SLEEP_HEAD_POSE, sdk.SLEEP_HEAD_POSE)
    assert tuple(ctl.INIT_ANTENNAS_RAD) == tuple(sdk.INIT_ANTENNAS_JOINT_POSITIONS)
    assert tuple(ctl.SLEEP_ANTENNAS_RAD) == tuple(sdk.SLEEP_ANTENNAS_JOINT_POSITIONS)


def _full_state_json(head: np.ndarray, body: float, *, use_matrix: bool) -> dict:
    from reachy_mini.daemon.app.models import FullState, as_any_pose

    state = FullState(
        control_mode="enabled",
        head_pose=as_any_pose(head, use_matrix),
        body_yaw=body,
        antennas_position=[0.1, -0.2],
    )
    return json.loads(state.model_dump_json())


@pytest.mark.parametrize("use_matrix", [True, False])
def test_the_fresh_readback_parser_reads_the_daemons_own_model(use_matrix):
    """Gate 24 (the parser half): the daemon's FullState, serialised as the route serialises it, parses to the pose."""
    from maxim.hardware.reachy import motion_path as mp

    head = mp.compose_head(math.radians(10), math.radians(-15), math.radians(40), math.radians(30), [0.01, 0.0, -0.02])
    pose, control_mode = mp.parse_full_state(_full_state_json(head, math.radians(30), use_matrix=use_matrix))
    assert control_mode == "enabled"
    assert pose.body == pytest.approx(math.radians(30))
    assert pose.antennas == pytest.approx((0.1, -0.2))
    assert mp.geodesic(pose.head, head) < 1e-6
    np.testing.assert_allclose(np.asarray(pose.head)[:3, 3], head[:3, 3], atol=1e-9)


def test_the_full_state_query_names_real_route_parameters():
    """The controller's query string only uses parameters the daemon's GET /api/state/full declares."""
    import inspect

    from reachy_mini.daemon.app.routers.state import get_full_state

    from maxim.hardware.reachy.controller import _STATE_FULL_QUERY

    declared = set(inspect.signature(get_full_state).parameters)
    for part in _STATE_FULL_QUERY.split("&"):
        name, _, value = part.partition("=")
        assert name in declared, name
        assert value == "true"


class _StubBackend:
    """Just enough of the daemon backend for ``get_full_state``; it HAS a body target, which the route builds."""

    target_body_yaw = 0.3

    def get_motor_control_mode(self):
        from types import SimpleNamespace

        return SimpleNamespace(value="enabled")

    def get_present_head_pose(self):
        return np.eye(4)

    def get_present_body_yaw(self):
        return 0.1

    def get_present_antenna_joint_positions(self):
        return [0.0, 0.0]


def _served_keys(**flags) -> set[str]:
    """The keys the real route's response carries for *flags*, after its response model (what reaches the wire)."""
    import asyncio

    from reachy_mini.daemon.app.routers.state import get_full_state

    state = asyncio.run(get_full_state(backend=_StubBackend(), **flags))
    return {k for k, v in json.loads(state.model_dump_json()).items() if v is not None}


def test_the_fakes_serve_only_what_the_real_route_serves():
    """A fake that serves a field the daemon drops hides a refusal on the robot (#1563: target_body_yaw, PR 3 review)."""
    from tests.unit._reachy_fakes import FakeDaemon

    from maxim.hardware.reachy.controller import _STATE_FULL_QUERY

    flags = {name: value == "true" for name, _, value in (p.partition("=") for p in _STATE_FULL_QUERY.split("&"))}
    served = _served_keys(**flags)
    fakes = {"FakeDaemon": FakeDaemon().state_full(), **_other_state_fakes()}
    for name, state in fakes.items():
        faked = {k for k, v in state.items() if v is not None}
        assert faked <= served, f"{name} serves keys the daemon does not: {sorted(faked - served)}"


def _other_state_fakes() -> dict:
    """The two other ``_get_state_full`` stubs, each run once against a minimal SDK handle."""
    import types

    import pytest as _pytest

    from tests.unit import _reachy_fakes
    from tests.unit import test_reachy_connection_options as conn

    from maxim.hardware.reachy.controller import ReachyMiniController

    mini = types.SimpleNamespace(
        get_current_head_pose=lambda: np.eye(4),
        get_current_joint_positions=lambda: ([0.0] * 7, [0.0, 0.0]),
        control_mode="enabled",
    )
    host = types.SimpleNamespace(_mini=mini)
    mp = _pytest.MonkeyPatch()
    try:
        _reachy_fakes.serve_state_from_ws_fake(mp)
        ws = ReachyMiniController._get_state_full(host)
    finally:
        mp.undo()
    return {"serve_state_from_ws_fake": ws, "_state_from_the_mock": conn._state_from_the_mock(host)}


def test_the_daemon_does_not_serve_its_body_target():
    """SDK 1.8.3: the route builds target_body_yaw, then FullState drops it, and no other route serves it (#1563).

    When an SDK upgrade starts serving it, this fails: then #1563's check (refuse when the target and the present
    body disagree) becomes buildable, gated on that daemon version.
    """
    assert "target_body_yaw" not in _served_keys(with_body_yaw=True, with_target_body_yaw=True)
