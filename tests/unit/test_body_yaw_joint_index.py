"""Body-yaw joint-index regression guard (2026-08-04 phantom-frame fix).

SDK >= 1.5 joint vector is ``[body_yaw, *6 stewart_legs]`` — its own
kinematics reads INDEX 0 as body_yaw (``analytical_kinematics.fk``:
``body_yaw = joint_angles[0]``; ``ik`` returns ``[body_yaw] + stewart``).
``MovementMixin.sync_head_position`` read index 6 (zenoh-era ordering,
never migrated across the v1.5 pivot), so ``maxim.yaw`` — the
body-relative head yaw that feeds the DoA capture frame, focus_on_sound's
aim math, and the bounds learner — had a STEWART LEG angle folded in.

Measured live 2026-08-04: ``maxim.yaw`` reported 26.0° while the daemon's
ground truth (``/api/state/full``) showed world yaw −0.4°, body −0.3°,
with a leg at ≈ −0.45 rad ≈ −26°. Eight consecutive focus_on_sound calls
aimed in that phantom frame and the head never moved.

These tests are offline (fake SDK object) and pin: body_yaw comes from
index 0; a leg angle at index 6 must NOT perturb the synced yaw (the
pre-fix code fails the regression case by ~26°).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from maxim.embodied_runtime.movement import MovementMixin


def _world_pose_matrix(yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    m = np.eye(4)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


class _FakeMini:
    def __init__(self, head_joints, world_yaw_rad):
        self._joints = head_joints
        self._pose = _world_pose_matrix(world_yaw_rad)

    def get_current_joint_positions(self):
        return list(self._joints), [0.0, 0.0]

    def get_current_head_pose(self):
        return self._pose


class _Host(MovementMixin):
    def __init__(self, mini):
        self.mini = mini
        self.log = logging.getLogger("test_body_yaw_joint_index")
        self.x = self.y = self.z = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self.body_yaw = 0.0

    def _get_workspace_limits(self):
        return {"yaw": 55.0, "pitch": 35.0, "roll": 35.0}


class TestBodyYawJointIndex:
    def test_body_yaw_read_from_index_zero(self):
        """World yaw 30°, body (index 0) 20° → body-relative head yaw 10°."""
        mini = _FakeMini(
            head_joints=[math.radians(20.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            world_yaw_rad=math.radians(30.0),
        )
        host = _Host(mini)
        assert host.sync_head_position() is True
        assert host.yaw == pytest.approx(10.0, abs=0.1)
        assert host.body_yaw == pytest.approx(20.0, abs=0.1)

    def test_stewart_leg_angle_does_not_corrupt_yaw(self):
        """THE live regression: body at 0, head world yaw 0, one leg bent
        at −0.45 rad (a pitch posture). Pre-fix: yaw = 0 − (−25.8°) = +25.8°
        — the phantom 26° frame. Post-fix: yaw ≈ 0."""
        mini = _FakeMini(
            head_joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.45],
            world_yaw_rad=0.0,
        )
        host = _Host(mini)
        assert host.sync_head_position() is True
        assert host.yaw == pytest.approx(0.0, abs=0.5), (
            "a stewart LEG angle (index 6) leaked into body_yaw — the zenoh-era joint ordering is back"
        )
        assert host.body_yaw == pytest.approx(0.0, abs=0.1)

    def test_turned_body_with_centered_head(self):
        """Body turned 40° with the head riding along (world yaw 40°):
        body-relative head yaw is 0 — the head is centered ON the body."""
        mini = _FakeMini(
            head_joints=[math.radians(40.0), 0.1, -0.2, 0.3, -0.1, 0.2, -0.3],
            world_yaw_rad=math.radians(40.0),
        )
        host = _Host(mini)
        assert host.sync_head_position() is True
        assert host.yaw == pytest.approx(0.0, abs=0.5)
        assert host.body_yaw == pytest.approx(40.0, abs=0.1)
