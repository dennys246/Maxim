"""Regression guard for the Reachy head-frame invariant (CLAUDE.md, 2026-07-16).

THE BUG THIS PINS: the Reachy head pose is in the WORLD frame and sits ABOVE
``body_yaw`` in the kinematic chain. ``goto_target(head=None, body_yaw=X)`` does
NOT leave the head alone — the daemon re-solves IK against the RETAINED world
head target, COUNTER-ROTATING the head while the body turns under it. **The
camera and microphone array are IN the head**, so a body-only turn barely moves
them (measured: 0.32 rad of sensor rotation for a 0.9 rad body command).

Reading a nearly-stationary sensor while believing it moved is indistinguishable
from a broken sensor: it cost a full session and six falsified hypotheses, and
produced a "the DoA is a tracking estimator" finding that had to be retracted.

Pollen's prescription (AGENTS.md, verbatim): "to make the head follow the body,
ship a `head` matrix in the same call with the body delta added to the head yaw."

These tests are the REAL regression guard — the invariant previously cited
``scripts/`` (a research script nothing imports, plus a diagnostic that needs a
robot on the LAN and can never run in CI), which is not a guard under CLAUDE.md
Principle 5. Everything here runs offline against a fake SDK object.
"""

from __future__ import annotations

import math
import sys
import types

import pytest

from maxim.hardware.controller import MotionTarget
from maxim.hardware.reachy.controller import ReachyMiniController


@pytest.fixture(autouse=True)
def _stub_reachy_sdk(monkeypatch):
    """Supply `reachy_mini.utils.create_head_pose` WITHOUT the optional `reachy` extra.

    THIS FIXTURE IS THE POINT OF THE FILE. The invariant guarded here was
    previously "guarded" by scripts that cannot run in CI — which is not a guard
    (CLAUDE.md Principle 5). A guard that passes only where an optional dep happens
    to be installed is the same non-guard with extra steps: the first version of
    this file did exactly that — green on a laptop with the `reachy` extra, RED in
    CI, where `from reachy_mini.utils import create_head_pose` raises and
    `goto_target` swallows it and returns False.

    `create_head_pose` is pure math (euler -> 4x4), so a faithful stub is cheap and
    keeps the guard running on every push. Composition is Rz @ Ry @ Rx, matching the
    controller's own extraction (`atan2(m[1][0], m[0][0])`).
    """

    def create_head_pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, mm=False, degrees=True):
        if degrees:
            roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y],
            [-sp, cp * sr, cp * cr, z],
            [0.0, 0.0, 0.0, 1.0],
        ]

    pkg = sys.modules.get("reachy_mini") or types.ModuleType("reachy_mini")
    utils = types.ModuleType("reachy_mini.utils")
    utils.create_head_pose = create_head_pose
    monkeypatch.setattr(pkg, "utils", utils, raising=False)
    monkeypatch.setitem(sys.modules, "reachy_mini", pkg)
    monkeypatch.setitem(sys.modules, "reachy_mini.utils", utils)


def _yaw_of(pose_matrix) -> float:
    """World yaw from a 4x4 head pose matrix (same extraction the SDK/controller use)."""
    return math.atan2(pose_matrix[1][0], pose_matrix[0][0])


class _FakeMini:
    """Records goto_target kwargs; reports a body yaw via the SDK's joint[0] convention."""

    def __init__(self, body_yaw: float = 0.0, head_world_yaw: float = 0.0) -> None:
        self._body_yaw = body_yaw
        self._head_world_yaw = head_world_yaw
        self.calls: list[dict] = []

    def get_current_head_pose(self):
        c, s = math.cos(self._head_world_yaw), math.sin(self._head_world_yaw)
        return [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

    def get_current_joint_positions(self):
        # [body_yaw, *stewart_legs] — the SDK's own fk() reads index 0 as body_yaw.
        return [self._body_yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0]

    def goto_target(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture
def controller() -> tuple[ReachyMiniController, _FakeMini]:
    import threading

    ctl = ReachyMiniController.__new__(ReachyMiniController)
    fake = _FakeMini()
    ctl._mini = fake
    ctl._motion_lock = threading.RLock()  # normally set by __init__ (2026-08-07 safety fold)
    ctl.is_connected = lambda: True  # type: ignore[method-assign]
    ctl._update_state = lambda **kw: None  # type: ignore[method-assign]
    return ctl, fake


def test_body_yaw_alone_still_ships_a_head_matrix(controller):
    """THE invariant: a body-only command must NOT pass head=None."""
    ctl, fake = controller
    assert ctl.goto_target(MotionTarget(body_yaw=0.9)) is True
    (call,) = fake.calls
    assert call["head"] is not None, (
        "head=None with body_yaw set => the daemon counter-rotates the head and the "
        "head-mounted camera/mics do not turn. This is the bug that faked a sensor "
        "pathology for a full session. See CLAUDE.md's Reachy head-frame invariant."
    )


def test_body_yaw_alone_makes_the_head_ride_along(controller):
    """The head's WORLD yaw must track the commanded body yaw (delta added)."""
    ctl, fake = controller
    ctl.goto_target(MotionTarget(body_yaw=0.9))
    (call,) = fake.calls
    assert _yaw_of(call["head"]) == pytest.approx(0.9, abs=1e-6), (
        "the head must end up at world yaw == body yaw (relative angle preserved), "
        "so the sensors rotate exactly as commanded"
    )


def test_head_yaw_is_body_relative_and_composed_onto_body(controller):
    """head_yaw means 'relative to the body'; the world pose adds the body yaw."""
    ctl, fake = controller
    ctl.goto_target(MotionTarget(head_yaw=0.1, body_yaw=0.5))
    (call,) = fake.calls
    assert _yaw_of(call["head"]) == pytest.approx(0.6, abs=1e-6), (
        "head_yaw=0.1 with body_yaw=0.5 must be world 0.6 — composing it as a bare "
        "world 0.1 pins the head while the body turns underneath"
    )


def test_head_only_command_preserves_current_body_yaw(controller):
    """A head-only command must compose against the body's ACTUAL angle, not 0."""
    ctl, fake = controller
    fake._body_yaw = 0.4  # body is already turned
    ctl.goto_target(MotionTarget(head_yaw=0.1))
    (call,) = fake.calls
    assert _yaw_of(call["head"]) == pytest.approx(0.5, abs=1e-6), (
        "assuming body_yaw=0 is the same frame mistake one layer down"
    )


def test_method_default_maps_to_a_valid_sdk_enum(controller):
    """MotionTarget's default method ('minimum_jerk') must reach the SDK as a valid enum.

    The SDK's InterpolationTechnique accepts only {linear, minjerk, ease_in_out,
    cartoon} and REJECTS 'minimum_jerk' with a pydantic error — so passing it
    through crashed every controller-driven motion on SDK >= 1.5 until 2026-07-17.
    """
    ctl, fake = controller
    ctl.goto_target(MotionTarget(body_yaw=0.3))  # default method
    (call,) = fake.calls
    assert call["method"] in {"linear", "minjerk", "ease_in_out", "cartoon"}, (
        f"method {call['method']!r} is not a valid SDK InterpolationTechnique value — "
        "the daemon rejects it and the motion never happens"
    )


def test_default_motiontarget_method_is_what_we_map_from(controller):
    """Pin the default so a rename of MotionTarget.method can't silently break the map."""
    assert MotionTarget(body_yaw=0.0).method == "minimum_jerk"
    assert ReachyMiniController._sdk_method("minimum_jerk") == "minjerk"


def test_get_current_pose_exposes_body_yaw(controller):
    """body_yaw must be readable — the head-frame conversion depends on it."""
    ctl, fake = controller
    fake._body_yaw = 0.33
    assert ctl.get_current_pose().get("body_yaw") == pytest.approx(0.33), (
        "without body_yaw, 'head rides along' silently assumes the body sits at 0"
    )
