"""Regression guard for the F1 retained-axes ratchet (H1 campaign, 2026-08-08).

THE INCIDENT THIS PINS: ``goto_target`` filled UNSPECIFIED axes from the
CURRENT POSE READBACK ("don't recenter the others", pre-fix behavior).
Under any achieved-vs-commanded bias that is positive feedback: each
command re-targets the previous ACHIEVED value, so the bias integrates.
During H1, ~20 yaw-only probe commands ratcheted roll +~1.3°/command up
to +38° (pitch drifted too), invisibly contaminating the first neck
envelope measurements — which were RETRACTED as roll-contaminated
(docs/experiments/protocols/h1_healthy_hardware_doa_preregistration.md, F1).

The fix: retain the last COMMANDED value per axis (post-clamp, as
dispatched) in ``ReachyMiniController._last_commanded``; the readback
seeds an axis only when it has never been commanded (or after
out-of-band motion — look_at_pixel / wake_up / goto_sleep — cleared the
stash). A one-shot seed absorbs at most one bias step; it has no
feedback loop.

Everything here runs offline against a fake SDK (same pattern as
test_reachy_head_frame.py / test_reachy_workspace_safety.py) whose
ACHIEVED pose runs a constant bias above the commanded pose — the
minimal plant model of the H1 hardware. The ratchet tests were verified
to FAIL against the pre-fix controller.
"""

from __future__ import annotations

import math
import sys
import threading
import types

import pytest

from maxim.hardware.controller import MotionTarget, PixelTarget
from maxim.hardware.reachy.controller import ReachyMiniController


def _rot(roll: float, pitch: float, yaw: float):
    """Rz @ Ry @ Rx — matches the SDK's create_head_pose composition and the
    controller's own euler extraction."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
        [-sp, cp * sr, cp * cr, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


@pytest.fixture(autouse=True)
def _stub_reachy_sdk(monkeypatch):
    """Pure-math stub for `reachy_mini.utils.create_head_pose` (CI has no
    `reachy` extra installed)."""

    def create_head_pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, mm=False, degrees=True):
        if degrees:
            roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))
        return _rot(roll, pitch, yaw)

    pkg = sys.modules.get("reachy_mini") or types.ModuleType("reachy_mini")
    utils = types.ModuleType("reachy_mini.utils")
    utils.create_head_pose = create_head_pose
    monkeypatch.setattr(pkg, "utils", utils, raising=False)
    monkeypatch.setitem(sys.modules, "reachy_mini", pkg)
    monkeypatch.setitem(sys.modules, "reachy_mini.utils", utils)


def _eulers_of(m) -> tuple[float, float, float]:
    """(roll, pitch, world_yaw) — the controller's own extraction."""
    yaw = math.atan2(m[1][0], m[0][0])
    pitch = math.atan2(-m[2][0], math.hypot(m[2][1], m[2][2]))
    roll = math.atan2(m[2][1], m[2][2])
    return roll, pitch, yaw


class _BiasedFakeMini:
    """Fake SDK whose ACHIEVED pose runs a constant bias above commanded.

    The minimal plant model of the H1 hardware: dispatch a head matrix and
    the readback reports commanded + bias on each axis. Feeding that
    readback into the next command's unspecified axes is the ratchet.
    """

    def __init__(
        self,
        *,
        roll_bias: float = 0.0,
        pitch_bias: float = 0.0,
        yaw_bias: float = 0.0,
        body_bias: float = 0.0,
    ) -> None:
        self._bias = (roll_bias, pitch_bias, yaw_bias, body_bias)
        # Achieved state (world frame for the head, like the real fk()).
        self.roll = 0.0
        self.pitch = 0.0
        self.world_yaw = 0.0
        self.body_yaw = 0.0
        self.calls: list[dict] = []

    def get_current_head_pose(self):
        return _rot(self.roll, self.pitch, self.world_yaw)

    def get_current_joint_positions(self):
        return [self.body_yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0]

    def goto_target(self, **kwargs):
        self.calls.append(kwargs)
        rb, pb, yb, bb = self._bias
        head = kwargs.get("head")
        if head is not None:
            roll, pitch, yaw = _eulers_of(head)
            self.roll = roll + rb
            self.pitch = pitch + pb
            self.world_yaw = yaw + yb
        if kwargs.get("body_yaw") is not None:
            self.body_yaw = kwargs["body_yaw"] + bb

    def look_at_image(self, u, v, duration=1.0):
        # Moves the head out-of-band (not through goto_target's stash).
        self.pitch += 0.2
        self.world_yaw += 0.3

    def commanded_eulers(self, call_index: int = -1) -> tuple[float, float, float]:
        return _eulers_of(self.calls[call_index]["head"])


def _make_controller(**bias) -> tuple[ReachyMiniController, _BiasedFakeMini]:
    ctl = ReachyMiniController.__new__(ReachyMiniController)
    fake = _BiasedFakeMini(**bias)
    ctl._mini = fake
    ctl._motion_lock = threading.RLock()
    ctl._last_commanded = {}
    ctl.is_connected = lambda: True  # type: ignore[method-assign]
    ctl._update_state = lambda **kw: None  # type: ignore[method-assign]
    return ctl, fake


# ─────────────────────────────────────────────────────────────────────────────
# THE ratchet: repeated single-axis commands must not integrate the bias
# ─────────────────────────────────────────────────────────────────────────────


class TestRetainedAxesDoNotRatchet:
    def test_yaw_only_commands_do_not_ratchet_roll(self):
        """The H1 F1 reproduction: ~20 yaw-only commands under a constant
        achieved-roll bias. Pre-fix, each command's roll target was the
        previous ACHIEVED roll (previous target + bias) — the commanded
        roll ratcheted ~+1.3°/command to +38° on hardware. Post-fix the
        commanded roll stays pinned at the one-shot seed."""
        bias = math.radians(2.0)
        ctl, fake = _make_controller(roll_bias=bias, pitch_bias=bias)

        for _ in range(20):
            assert ctl.goto_target(MotionTarget(head_yaw=0.1)) is True

        roll_cmd, pitch_cmd, _ = fake.commanded_eulers()
        assert abs(roll_cmd) <= bias / 2, (
            f"20 yaw-only commands ratcheted the commanded roll to "
            f"{math.degrees(roll_cmd):.1f}° under a {math.degrees(bias):.1f}° "
            "achieved-vs-commanded bias — unspecified axes are being filled "
            "from the READBACK instead of the last COMMANDED value (H1 F1: "
            "+38° of roll from yaw-only probes)."
        )
        assert abs(pitch_cmd) <= bias / 2, (
            f"pitch ratcheted to {math.degrees(pitch_cmd):.1f}° — same readback "
            "positive-feedback loop on the pitch axis."
        )

    def test_body_only_commands_do_not_ratchet_relative_yaw(self):
        """Same loop, yaw variant: body-only commands with a head-yaw
        achieved bias. Pre-fix the RETAINED relative yaw was re-derived
        from the readback (achieved world − achieved body) each command,
        integrating any head/body bias mismatch into the neck angle —
        toward the ±65° capability limit the workspace clamps exist for."""
        ctl, fake = _make_controller(yaw_bias=math.radians(3.0))

        body = 0.0
        for _ in range(15):
            body += 0.05
            assert ctl.goto_target(MotionTarget(body_yaw=body)) is True

        _, _, world_yaw_cmd = fake.commanded_eulers()
        relative_cmd = world_yaw_cmd - fake.calls[-1]["body_yaw"]
        assert abs(relative_cmd) <= math.radians(3.0) / 2, (
            f"commanded head-relative yaw crept to {math.degrees(relative_cmd):.1f}° "
            "across body-only commands — the retained yaw is being re-derived "
            "from the biased readback instead of held at the commanded value."
        )

    def test_head_only_commands_use_last_commanded_body_for_composition(self):
        """After a body command, head-only follow-ups must compose the world
        head yaw against the COMMANDED body angle, not the biased achieved
        one — otherwise the body bias leaks into the head world target."""
        ctl, fake = _make_controller(body_bias=math.radians(4.0))
        assert ctl.goto_target(MotionTarget(body_yaw=0.5)) is True
        assert ctl.goto_target(MotionTarget(head_yaw=0.2)) is True

        _, _, world_yaw_cmd = fake.commanded_eulers()
        assert world_yaw_cmd == pytest.approx(0.5 + 0.2, abs=1e-9), (
            "head world yaw composed against the achieved body "
            f"({math.degrees(world_yaw_cmd):.1f}° instead of "
            f"{math.degrees(0.7):.1f}°) — body fill-in must come from the "
            "last commanded body value."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Seeding: the readback is consulted exactly once per never-commanded axis
# ─────────────────────────────────────────────────────────────────────────────


class TestReadbackSeeding:
    def test_first_command_seeds_unspecified_axes_from_readback(self):
        """No stash yet: a yaw-only first command must hold the CURRENT roll/
        pitch (don't recenter the others — the behavior the readback fill-in
        existed for, preserved for the seed)."""
        ctl, fake = _make_controller()
        fake.roll = 0.15
        fake.pitch = -0.1
        assert ctl.goto_target(MotionTarget(head_yaw=0.2)) is True
        roll_cmd, pitch_cmd, _ = fake.commanded_eulers()
        assert roll_cmd == pytest.approx(0.15, abs=1e-9)
        assert pitch_cmd == pytest.approx(-0.1, abs=1e-9)

    def test_stash_survives_unreadable_pose_after_first_command(self):
        """Once axes have been commanded, an unreadable pose must not block a
        body-only command — the stash knows what was last commanded (the
        refuse-to-guess rule narrows to the SEED case)."""
        ctl, fake = _make_controller()
        assert ctl.goto_target(MotionTarget(head_yaw=0.3, head_roll=0.1)) is True
        ctl.get_current_pose = lambda: {}  # type: ignore[method-assign]
        assert ctl.goto_target(MotionTarget(body_yaw=0.4)) is True
        roll_cmd, _, world_yaw_cmd = fake.commanded_eulers()
        assert roll_cmd == pytest.approx(0.1, abs=1e-9)
        assert world_yaw_cmd == pytest.approx(0.3 + 0.4, abs=1e-9)

    def test_first_command_with_unreadable_pose_still_refuses(self):
        """The executor-lens E2 rule is intact where it matters: no stash +
        no readback = refuse, don't guess."""
        ctl, fake = _make_controller()
        ctl.get_current_pose = lambda: {}  # type: ignore[method-assign]
        assert ctl.goto_target(MotionTarget(body_yaw=0.4)) is False
        assert fake.calls == []

    def test_clamped_command_stashes_the_clamped_value(self):
        """The stash holds what was DISPATCHED (post-clamp): a follow-up on
        another axis must re-command the clamped value, not the illegal one."""
        ctl, fake = _make_controller()
        assert ctl.goto_target(MotionTarget(head_roll=2.0)) is True  # clamps to 40°
        assert ctl.goto_target(MotionTarget(head_yaw=0.1)) is True
        roll_cmd, _, _ = fake.commanded_eulers()
        assert roll_cmd == pytest.approx(ReachyMiniController._MAX_HEAD_ROLL_RAD, abs=1e-9)
        assert "head_roll" not in ctl.last_clamped_axes, (
            "re-commanding the stashed post-clamp roll must not re-trigger the clamp WARNING every subsequent command"
        )

    def test_failed_dispatch_does_not_mutate_stash(self):
        """Fail-fast-before-mutation: a raised SDK dispatch must not leave the
        stash asserting values that never shipped."""
        ctl, fake = _make_controller()

        def boom(**kwargs):
            raise RuntimeError("wire fell out")

        fake.goto_target = boom
        assert ctl.goto_target(MotionTarget(head_yaw=0.3)) is False
        assert ctl._last_commanded == {}


# ─────────────────────────────────────────────────────────────────────────────
# Out-of-band motion invalidates the stash (re-seed from reality once)
# ─────────────────────────────────────────────────────────────────────────────


class TestOutOfBandInvalidation:
    def test_look_at_pixel_clears_head_axes(self):
        """look_at_image moves the head outside the stash's sight — the next
        fill-in must re-seed head axes from the readback, not re-command a
        pose from before the look-at."""
        ctl, fake = _make_controller()
        assert ctl.goto_target(MotionTarget(head_yaw=0.3, head_pitch=0.1)) is True
        assert ctl.look_at_pixel(PixelTarget(u=100, v=100)) is True
        assert ctl.goto_target(MotionTarget(head_roll=0.05)) is True
        _, pitch_cmd, world_yaw_cmd = fake.commanded_eulers()
        # Achieved after look_at: pitch 0.1+0.2, world yaw 0.3+0.3.
        assert pitch_cmd == pytest.approx(0.3, abs=1e-9)
        assert world_yaw_cmd == pytest.approx(0.6, abs=1e-9)

    def test_look_at_pixel_keeps_body_stash(self):
        """look_at_image does not move the body — its stashed commanded value
        stays authoritative."""
        ctl, fake = _make_controller(body_bias=math.radians(4.0))
        assert ctl.goto_target(MotionTarget(body_yaw=0.5)) is True
        assert ctl.look_at_pixel(PixelTarget(u=100, v=100)) is True
        assert ctl.goto_target(MotionTarget(head_yaw=0.0)) is True
        _, _, world_yaw_cmd = fake.commanded_eulers()
        assert world_yaw_cmd == pytest.approx(0.5, abs=1e-9)

    def test_forget_commanded_clears_everything_by_default(self):
        ctl, _fake = _make_controller()
        assert ctl.goto_target(MotionTarget(head_yaw=0.3, body_yaw=0.2)) is True
        assert ctl._last_commanded
        ctl._forget_commanded()
        assert ctl._last_commanded == {}
