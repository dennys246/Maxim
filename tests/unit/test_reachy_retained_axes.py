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
from tests.unit._reachy_fakes import prepare_controller, serve_state_from_ws_fake


@pytest.fixture(autouse=True)
def _serve_the_fresh_readback(monkeypatch):
    """The controller reads its reference from the daemon's REST state (#1499 R1); serve it from this file's fake."""
    serve_state_from_ws_fake(monkeypatch)


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
        self.antennas = [0.0, 0.0]
        self.calls: list[dict] = []

    def get_current_head_pose(self):
        return _rot(self.roll, self.pitch, self.world_yaw)

    def get_current_joint_positions(self):
        return [self.body_yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], list(self.antennas)

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
        if kwargs.get("antennas") is not None:
            self.antennas = list(kwargs["antennas"])

    def look_at_image(self, u, v, duration=1.0, perform_movement=True):
        # The SDK's look_at: the pose it would move to (here: pitch +0.2, yaw +0.3 from the present head).
        # The controller only ever asks for the pose (perform_movement=False, #1499 5.2).
        pose = _rot(self.roll, self.pitch + 0.2, self.world_yaw + 0.3)
        if perform_movement:
            self.pitch += 0.2
            self.world_yaw += 0.3
            self.body_yaw = 0.0  # F1: the SDK's own look_at sends the body to 0
        return pose

    def commanded_eulers(self, call_index: int = -1) -> tuple[float, float, float]:
        return _eulers_of(self.calls[call_index]["head"])


def _make_controller(**bias) -> tuple[ReachyMiniController, _BiasedFakeMini]:
    ctl = ReachyMiniController.__new__(ReachyMiniController)
    fake = _BiasedFakeMini(**bias)
    ctl._mini = fake
    ctl._motion_lock = threading.RLock()
    ctl._last_commanded = {}
    ctl._automatic_body_yaw = False
    ctl._divergence_last_warn = 0.0
    ctl._robot_name = "test_reachy"
    ctl.is_connected = lambda: True  # type: ignore[method-assign]
    ctl._update_state = lambda **kw: None  # type: ignore[method-assign]
    prepare_controller(ctl)
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
            assert ctl.goto_target(MotionTarget(head_yaw=0.1))

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
            assert ctl.goto_target(MotionTarget(body_yaw=body))

        _, _, world_yaw_cmd = fake.commanded_eulers()
        relative_cmd = world_yaw_cmd - fake.calls[-1]["body_yaw"]
        assert abs(relative_cmd) <= math.radians(3.0) / 2, (
            f"commanded head-relative yaw crept to {math.degrees(relative_cmd):.1f}° "
            "across body-only commands — the retained yaw is being re-derived "
            "from the biased readback instead of held at the commanded value."
        )

    def test_head_only_commands_compose_against_the_readback_body(self):
        """Inverted by #1499 (R1, F2). With ``body_yaw=None`` the daemon holds the body at its PRESENT position and
        solves the head against it, so composing against the commanded body put the world head target off by the
        body's lag. The readback body composes; a 4° lag is inside the divergence tolerance, so it is not refused.
        The head axes the caller leaves out still come from the stash (the H1 F1 rule, pinned above)."""
        ctl, fake = _make_controller(body_bias=math.radians(4.0))
        assert ctl.goto_target(MotionTarget(body_yaw=0.5))
        assert ctl.goto_target(MotionTarget(head_yaw=0.2))

        _, _, world_yaw_cmd = fake.commanded_eulers()
        assert world_yaw_cmd == pytest.approx(0.5 + math.radians(4.0) + 0.2, abs=1e-9)


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
        assert ctl.goto_target(MotionTarget(head_yaw=0.2))
        roll_cmd, pitch_cmd, _ = fake.commanded_eulers()
        assert roll_cmd == pytest.approx(0.15, abs=1e-9)
        assert pitch_cmd == pytest.approx(-0.1, abs=1e-9)

    def test_an_unreadable_pose_refuses_and_keeps_the_stash(self):
        """Inverted by #1499 (R2). Every rate and path check starts from a fresh readback, so with none the move
        is refused rather than composed from the stash. The stash survives the refusal untouched."""
        ctl, fake = _make_controller()
        assert ctl.goto_target(MotionTarget(head_yaw=0.3, head_roll=0.1))
        stash = dict(ctl._last_commanded)
        n_calls = len(fake.calls)
        ctl.get_current_pose = lambda: {}  # type: ignore[method-assign]
        fake.rest_fails = True  # the fresh-read seam (the build's fake migration honours it)
        assert not ctl.goto_target(MotionTarget(body_yaw=0.4))
        assert len(fake.calls) == n_calls
        assert ctl._last_commanded == stash

    def test_first_command_with_unreadable_pose_still_refuses(self):
        """The executor-lens E2 rule is intact where it matters: no stash +
        no readback = refuse, don't guess."""
        ctl, fake = _make_controller()
        fake.rest_fails = True  # an unreadable pose: the reference is the REST readback since #1499 (R1)
        assert not ctl.goto_target(MotionTarget(body_yaw=0.4))
        assert fake.calls == []

    def test_clamped_command_stashes_the_clamped_value(self):
        """The stash holds what was DISPATCHED (post-clamp): a follow-up on
        another axis must re-command the clamped value, not the illegal one."""
        ctl, fake = _make_controller()
        assert ctl.goto_target(MotionTarget(head_roll=2.0))  # clamps to 40°
        assert ctl.goto_target(MotionTarget(head_yaw=0.1))
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
        assert not ctl.goto_target(MotionTarget(head_yaw=0.3))
        assert ctl._last_commanded == {}


# ─────────────────────────────────────────────────────────────────────────────
# Out-of-band motion invalidates the stash (re-seed from reality once)
# ─────────────────────────────────────────────────────────────────────────────


class TestOutOfBandInvalidation:
    def test_look_at_pixel_dispatches_through_the_controller(self):
        """Inverted by #1499 (5.2, F1). look_at no longer moves the head out of band through the SDK (which also
        sends the body to 0); it is computed with ``perform_movement=False`` and dispatched as a head-only
        ``goto_target``, so the stash records its axes like any other controller command."""
        ctl, fake = _make_controller()
        assert ctl.goto_target(MotionTarget(head_yaw=0.3, head_pitch=0.1))
        n_calls = len(fake.calls)
        assert ctl.look_at_pixel(PixelTarget(u=100, v=100))
        assert len(fake.calls) == n_calls + 1
        assert fake.calls[-1].get("body_yaw") is None
        assert "head_rel_yaw" in ctl._last_commanded

    def test_look_at_pixel_never_commands_the_body(self):
        """Inverted by #1499 (5.2, F1). Every dispatch a look_at makes is head-only, so the body stays where it is."""
        ctl, fake = _make_controller(body_bias=math.radians(4.0))
        assert ctl.goto_target(MotionTarget(body_yaw=0.5))
        body_before = fake.body_yaw
        n_calls = len(fake.calls)
        assert ctl.look_at_pixel(PixelTarget(u=100, v=100))
        assert len(fake.calls) > n_calls
        assert all(c.get("body_yaw") is None for c in fake.calls[n_calls:])
        assert fake.body_yaw == body_before

    def test_forget_commanded_clears_everything_by_default(self):
        ctl, _fake = _make_controller()
        assert ctl.goto_target(MotionTarget(head_yaw=0.3, body_yaw=0.2))
        assert ctl._last_commanded
        ctl._forget_commanded()
        assert ctl._last_commanded == {}


# ─────────────────────────────────────────────────────────────────────────────
# Two-lens review folds (2026-08-09, both findings cross-confirmed)
# ─────────────────────────────────────────────────────────────────────────────


class TestBodySeedRefusal:
    """Executor-lens BLOCKING repro: a stashed head yaw must not defeat the
    refuse-to-guess rule for the BODY seed. Guessing body=0 on an unreadable
    pose ships a world head yaw whose ACTUAL neck angle can be off by the
    full body rotation — the relative-frame clamp runs in the believed
    frame and cannot catch it."""

    def test_head_partial_with_unreadable_pose_and_no_body_stash_refuses(self):
        ctl, fake = _make_controller()
        # Body physically turned; only head axes ever commanded (head-only
        # dispatches deliberately do NOT stash the body).
        fake.body_yaw = math.radians(120.0)
        fake.world_yaw = math.radians(120.0)
        assert ctl.goto_target(MotionTarget(head_yaw=0.1))
        assert "body_yaw" not in ctl._last_commanded
        fake.rest_fails = True  # an unreadable pose (the REST readback, #1499 R1)
        n_calls = len(fake.calls)
        assert not ctl.goto_target(MotionTarget(head_pitch=0.2)), (
            "head-partial command with an unreadable pose and a never-commanded "
            "body must REFUSE — composing against a guessed body=0 while the "
            "body sits at 120° demands a neck angle of −114° past the ±65° "
            "capability (the motor-destruction failure class)."
        )
        assert len(fake.calls) == n_calls, "refusal must not dispatch"

    def test_explicit_head_yaw_with_unreadable_body_also_refuses(self):
        """Tightened vs pre-fix (which silently guessed 0 here too): an
        explicit head_yaw still needs a body angle for world composition."""
        ctl, fake = _make_controller()
        fake.rest_fails = True  # an unreadable pose (the REST readback, #1499 R1)
        assert not ctl.goto_target(MotionTarget(head_yaw=0.1))
        assert fake.calls == []

    def test_a_body_stash_does_not_unblock_an_unreadable_pose(self):
        """Inverted by #1499 (R1, R2). The body stash only detects divergence now; it never substitutes for a
        readback, so an unreadable pose refuses even after a body command."""
        ctl, fake = _make_controller()
        assert ctl.goto_target(MotionTarget(body_yaw=0.5))
        n_calls = len(fake.calls)
        ctl.get_current_pose = lambda: {}  # type: ignore[method-assign]
        fake.rest_fails = True
        assert not ctl.goto_target(MotionTarget(head_yaw=0.1))
        assert len(fake.calls) == n_calls


class TestMoveHeadInvalidation:
    """Architecture+executor cross-confirmed BLOCKING: motion/movement.py::
    move_head (the other sanctioned motion primitive) moves the head raw.
    Without invalidation, the controller's next body-only turn snaps the
    head back to the stale stashed pose — permanent gaze-fighting in the
    live embodied runtime (DN gaze loops never write the stash)."""

    def test_move_head_with_a_controller_keeps_the_stash_on_the_gaze_pose(self):
        """Since #1499 (5.4) move_head delegates to the controller, so the gaze is a controller command: the stash
        records it, and the next body-only turn keeps the gaze pose instead of snapping back to the pre-gaze one."""
        from maxim.motion.movement import move_head

        ctl, fake = _make_controller()
        # Controller centers the head — stash pins roll/pitch/rel_yaw at 0.
        assert ctl.goto_target(MotionTarget(head_yaw=0.0, head_pitch=0.0))
        # DN gaze via the primitive, controller threaded (the Selfy.move seam).
        assert move_head(fake, 0, 0, 0, 0.0, math.degrees(0.4), math.degrees(0.5), 0.5, controller=ctl)
        assert ctl._last_commanded["head_pitch"] == pytest.approx(0.4, abs=1e-9)
        assert ctl._last_commanded["head_rel_yaw"] == pytest.approx(0.5, abs=1e-9)
        assert ctl.goto_target(MotionTarget(body_yaw=0.3))
        _, pitch_cmd, _ = fake.commanded_eulers()
        assert pitch_cmd == pytest.approx(0.4, abs=1e-6), (
            "body-only turn re-commanded the stale pre-gaze pitch instead of "
            "re-seeding from where the gaze left the head."
        )

    def test_standalone_move_head_needs_the_motion_lock(self):
        """Inverted by #1499 (5.4; round-5 S2). The standalone path (scripts with no controller) still dispatches,
        but only under a lock the caller passes; forgetting it is a TypeError, not a silent unlocked move."""
        from maxim.motion.movement import move_head

        fake = _BiasedFakeMini()
        with pytest.raises(TypeError):
            move_head(fake, 0, 0, 0, 0, 0, 10.0, 0.5)
        move_head(fake, 0, 0, 0, 0, 0, 10.0, 0.5, motion_lock=threading.RLock())
        assert fake.calls


class TestLifecycleClearWiring:
    """Executor-lens SHOULD-FIX: deleting a _forget_commanded call from
    wake_up/goto_sleep/disconnect previously passed the whole suite. Pin
    the wiring — wake is the highest-traffic invalidation (every session
    starts with a wake motion)."""

    @staticmethod
    def _prep(ctl, fake):
        assert ctl.goto_target(MotionTarget(head_yaw=0.3, body_yaw=0.2))
        assert ctl._last_commanded
        # Lifecycle SDK surface the fake doesn't have by default.
        fake.enable_motors = lambda: None
        fake.set_automatic_body_yaw = lambda v: None

        def _never(*_a, **_k):
            raise AssertionError("the SDK's canned wake_up/goto_sleep must never be called (#1499 O8)")

        fake.wake_up = _never
        fake.goto_sleep = _never

    def test_wake_up_leaves_the_stash_on_the_wake_pose(self):
        """Since #1499 (5.3, O8) wake is composed controller dispatches (re-centre, then INIT): the stash describes
        where the wake left the head and body, never the pre-wake command."""
        ctl, fake = _make_controller()
        self._prep(ctl, fake)
        assert ctl.wake_up()
        stash = ctl._last_commanded
        assert stash["body_yaw"] == pytest.approx(0.0, abs=1e-9)
        for axis in ("head_roll", "head_pitch", "head_rel_yaw"):
            assert stash[axis] == pytest.approx(0.0, abs=1e-6)

    def test_goto_sleep_leaves_the_stash_on_the_sleep_pose(self):
        ctl, fake = _make_controller()
        self._prep(ctl, fake)
        assert ctl.goto_sleep()
        stash = ctl._last_commanded
        assert stash["body_yaw"] == pytest.approx(0.0, abs=1e-9)
        assert stash["head_rel_yaw"] == pytest.approx(0.0, abs=0.01)
        assert stash["head_pitch"] == pytest.approx(math.asin(0.413), abs=0.01)  # SLEEP_HEAD_POSE's ~24° pitch

    def test_disconnect_clears_stash_even_when_stream_stop_raises(self):
        class _BoomStream:
            def stop(self):
                raise RuntimeError("stream teardown failed")

        ctl, fake = _make_controller()
        self._prep(ctl, fake)
        ctl._tunnel_proc = None
        ctl._video_stream = _BoomStream()
        ctl._audio_stream = None
        ctl._resolved_host = None
        ctl._robot_name = "test_reachy"
        ctl.disconnect()
        assert ctl._last_commanded == {}, (
            "a raising stream stop skipped the stash clear — stale commanded "
            "values would masquerade as the robot's pose on reconnect."
        )


class TestAutomaticBodyYawMode:
    """Two-lens SHOULD-FIX: under automatic_body_yaw the daemon rotates the
    body behind the runtime with no invalidation hook — a stashed body
    would go stale without bound. The body axis must stay readback-driven
    in that mode."""

    def test_auto_body_yaw_mode_refuses_a_body_command(self):
        """Inverted by #1499 (O6). With the daemon moving the body behind us, the path and rate checks no longer
        describe what the motors do, so the controller refuses motion in that mode (construction raises too)."""
        ctl, fake = _make_controller()
        ctl._automatic_body_yaw = True
        assert not ctl.goto_target(MotionTarget(body_yaw=0.5))
        assert fake.calls == []

    def test_auto_body_yaw_mode_refuses_a_head_only_command(self):
        """Inverted by #1499 (O6): head-only moves are refused in that mode too."""
        ctl, fake = _make_controller()
        ctl._automatic_body_yaw = True
        fake.body_yaw = 0.9
        assert not ctl.goto_target(MotionTarget(head_yaw=0.1))
        assert fake.calls == []


class TestDivergenceWarning:
    """Architecture-lens observability fold: the stash silences the signal
    that exposed F1 (visible bias accumulation). The controller warns —
    rate-limited, never feeding back into commands — when achieved diverges
    from last-commanded past 5° (progressive-degradation early warning)."""

    def test_large_divergence_warns(self, caplog):
        ctl, fake = _make_controller(roll_bias=math.radians(10.0))
        with caplog.at_level("WARNING", logger="maxim.hardware.reachy.controller"):
            # A 10° plant bias: the move lands 10° off (partial, #1499 final-target check), roll is stashed.
            assert not ctl.goto_target(MotionTarget(head_roll=0.0))
            out = ctl.goto_target(MotionTarget(head_yaw=0.1))  # sees the 10° divergence
        assert any("diverges from last commanded" in r.message for r in caplog.records)
        assert not out and out.kind == "reference", "past 5° the divergence is refused too (decision 3)"

    def test_small_bias_stays_quiet(self, caplog):
        ctl, fake = _make_controller(roll_bias=math.radians(2.0))
        with caplog.at_level("WARNING", logger="maxim.hardware.reachy.controller"):
            for _ in range(5):
                assert ctl.goto_target(MotionTarget(head_yaw=0.1))
        assert not any("diverges from last commanded" in r.message for r in caplog.records)

    def test_warning_is_throttled(self, caplog):
        ctl, fake = _make_controller(roll_bias=math.radians(10.0))
        with caplog.at_level("WARNING", logger="maxim.hardware.reachy.controller"):
            # Under a 10° bias these alternate partial / refused (#1499); the warning still fires once.
            ctl.goto_target(MotionTarget(head_roll=0.0))
            for _ in range(5):
                ctl.goto_target(MotionTarget(head_yaw=0.1))
        n = sum(1 for r in caplog.records if "diverges from last commanded" in r.message)
        assert n == 1, f"expected exactly one throttled divergence warning, got {n}"
