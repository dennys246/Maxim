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

import pytest

from maxim.hardware.reachy.controller import ReachyMiniController
from maxim.hardware.controller import MotionTarget


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
    ctl = ReachyMiniController.__new__(ReachyMiniController)
    fake = _FakeMini()
    ctl._mini = fake
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


def test_get_current_pose_exposes_body_yaw(controller):
    """body_yaw must be readable — the head-frame conversion depends on it."""
    ctl, fake = controller
    fake._body_yaw = 0.33
    assert ctl.get_current_pose().get("body_yaw") == pytest.approx(0.33), (
        "without body_yaw, 'head rides along' silently assumes the body sits at 0"
    )
