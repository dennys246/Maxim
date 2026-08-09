"""Regression guard for the orient workspace-safety fold (2026-08-07).

THE INCIDENT THIS PINS: motors in Stewart positions 2 and 3 were destroyed
because an earlier Maxim iteration commanded a pose beyond the platform's
physical capability — the motors glitched, the head snapped violently to the
opposite extreme, and the robot rotated itself off the table. The breakage
spanned essentially the entire 1.0+ era (roadmap_1_1_to_1_3.md hardware note).

Two paths bypassed ``ReachyMiniController.goto_target`` — and with it the
head-frame composition AND any workspace clamping:

1. ``MoveTool`` without a ``robot_id`` fell back to ``Selfy.move()``, which
   applied only the per-call max-step limiter (whose defaults are ALL 0.0 =
   disabled) and enqueued the raw ABSOLUTE pose to the SDK. An LLM-passed
   ``yaw=170`` went straight to the motors.
2. ``Selfy.turn_around()`` hand-rolled a 3-step sequence against the raw SDK.
   Its body-rotation step shipped ``head=None``, so the daemon held the head
   at its RETAINED world pose (centered at world 0) while the body rotated up
   to ±160° beneath it — commanding a head-relative angle far past the ±65°
   neck capability.

The fold: physical-limit clamps + a motion RLock live in
``ReachyMiniController.goto_target`` (the single canonical dispatch point);
``turn_around`` routes through the controller; ``Selfy.move()`` clamps via
``_clamp_to_workspace_6d``. Everything here runs offline against a fake SDK
(same pattern as test_reachy_head_frame.py).
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
import types

import pytest

from maxim.hardware.controller import MotionTarget
from maxim.hardware.reachy.controller import ReachyMiniController


@pytest.fixture(autouse=True)
def _stub_reachy_sdk(monkeypatch):
    """Faithful pure-math stub for `reachy_mini.utils.create_head_pose`.

    Same rationale as test_reachy_head_frame.py: the guard must run in CI
    where the optional `reachy` extra is absent. Composition is Rz @ Ry @ Rx,
    matching the controller's own extraction (`atan2(m[1][0], m[0][0])`).
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


def _world_yaw_of(pose_matrix) -> float:
    """World yaw from a 4x4 head pose matrix (same extraction the SDK uses)."""
    return math.atan2(pose_matrix[1][0], pose_matrix[0][0])


class _FakeMini:
    """Records goto_target kwargs; reports body yaw via the SDK joint[0] convention."""

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


def _make_controller(body_yaw: float = 0.0, head_world_yaw: float = 0.0):
    ctl = ReachyMiniController.__new__(ReachyMiniController)
    fake = _FakeMini(body_yaw=body_yaw, head_world_yaw=head_world_yaw)
    ctl._mini = fake
    ctl._motion_lock = threading.RLock()
    ctl._last_commanded = {}  # normally set by __init__ (F1 retained-axes fix, 2026-08-08)
    ctl._automatic_body_yaw = False
    ctl._divergence_last_warn = 0.0
    ctl.is_connected = lambda: True  # type: ignore[method-assign]
    ctl._update_state = lambda **kw: None  # type: ignore[method-assign]
    return ctl, fake


@pytest.fixture
def controller():
    return _make_controller()


# ─────────────────────────────────────────────────────────────────────────────
# Physical-limit clamps at the canonical dispatch point
# ─────────────────────────────────────────────────────────────────────────────


class TestGotoTargetClamps:
    def test_body_yaw_beyond_160_deg_is_clamped(self, controller):
        ctl, fake = controller
        assert ctl.goto_target(MotionTarget(body_yaw=math.radians(300.0))) is True
        (call,) = fake.calls
        assert call["body_yaw"] == pytest.approx(math.radians(160.0)), (
            "A body command past ±160° must be clamped at the controller — an "
            "unclamped extreme is the motor-destruction failure class."
        )

    def test_body_yaw_negative_extreme_clamped(self, controller):
        ctl, fake = controller
        assert ctl.goto_target(MotionTarget(body_yaw=-6.0)) is True
        (call,) = fake.calls
        assert call["body_yaw"] == pytest.approx(-math.radians(160.0))

    def test_head_relative_yaw_beyond_65_deg_is_clamped(self, controller):
        ctl, fake = controller
        # Body stays at 0, so the shipped world yaw equals the relative yaw.
        assert ctl.goto_target(MotionTarget(head_yaw=math.radians(120.0))) is True
        (call,) = fake.calls
        assert _world_yaw_of(call["head"]) == pytest.approx(math.radians(65.0), abs=1e-6), (
            "head_yaw is BODY-RELATIVE and the neck's physical capability is ±65°; commanding past it must clamp."
        )

    def test_head_roll_and_pitch_clamped_to_40_deg(self, controller):
        ctl, fake = controller
        assert ctl.goto_target(MotionTarget(head_roll=2.0, head_pitch=-2.0)) is True
        (call,) = fake.calls
        m = call["head"]
        pitch = math.atan2(-m[2][0], math.hypot(m[2][1], m[2][2]))
        roll = math.atan2(m[2][1], m[2][2])
        assert roll == pytest.approx(math.radians(40.0), abs=1e-6)
        assert pitch == pytest.approx(-math.radians(40.0), abs=1e-6)

    def test_clamp_is_body_relative_not_world(self):
        """A LEGAL relative head yaw on a turned body must NOT be clamped.

        With the body at 100° and head_yaw=+40° (legal, < 65°), the world-frame
        sum is 140°. A clamp applied to the world sum would wrongly limit this;
        the physical constraint lives in the body-relative frame.
        """
        body = math.radians(100.0)
        ctl, fake = _make_controller(body_yaw=body, head_world_yaw=body)
        assert ctl.goto_target(MotionTarget(head_yaw=math.radians(40.0), body_yaw=body)) is True
        (call,) = fake.calls
        assert _world_yaw_of(call["head"]) == pytest.approx(math.radians(140.0), abs=1e-6)

    def test_legal_commands_pass_through_unchanged(self, controller):
        ctl, fake = controller
        assert ctl.goto_target(MotionTarget(head_yaw=0.3, body_yaw=0.5)) is True
        (call,) = fake.calls
        assert call["body_yaw"] == pytest.approx(0.5)
        assert _world_yaw_of(call["head"]) == pytest.approx(0.8, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# TOCTOU: the motion lock spans read → compose → dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestGotoTargetMotionLock:
    def test_read_compose_dispatch_does_not_interleave(self, controller):
        """Two concurrent goto_target calls must not overlap their pose reads.

        Without the lock, both threads sit inside the slow pose read at once
        and `overlap` fires; with the lock the second caller cannot enter
        until the first has dispatched.
        """
        ctl, fake = controller
        inside = threading.Event()
        overlap = threading.Event()

        real_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "body_yaw": 0.0}

        def slow_pose():
            if inside.is_set():
                overlap.set()
            inside.set()
            time.sleep(0.15)
            inside.clear()
            return dict(real_pose)

        ctl.get_current_pose = slow_pose  # type: ignore[method-assign]

        threads = [
            threading.Thread(target=lambda: ctl.goto_target(MotionTarget(body_yaw=0.4))),
            threading.Thread(target=lambda: ctl.goto_target(MotionTarget(head_yaw=0.2))),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not overlap.is_set(), (
            "Two goto_target calls interleaved their get_current_pose() → compose → "
            "dispatch sections: each composes a head matrix against a body yaw the "
            "other is concurrently changing (TOCTOU on live kinematic state)."
        )
        assert len(fake.calls) == 2

    def test_motion_lock_is_reentrant(self, controller):
        """RLock, not Lock: a caller already holding the lock can dispatch."""
        ctl, fake = controller
        with ctl._motion_lock:
            assert ctl.goto_target(MotionTarget(body_yaw=0.1)) is True
        assert len(fake.calls) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Bypass path 2: turn_around must route through the controller
# ─────────────────────────────────────────────────────────────────────────────


from maxim.embodied_runtime.movement import MovementMixin  # noqa: E402


class _TurnHost(MovementMixin):
    """Minimal Selfy stand-in exposing exactly what turn_around touches."""

    def __init__(self, robot) -> None:
        self._robot = robot
        self.log = logging.getLogger("test_turn_host")
        self.body_yaw = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.y = 0.0
        self.z = 0.0

    def _clear_motor_queue(self) -> int:
        return 0

    def sync_head_position(self) -> bool:
        return True


class TestTurnAroundRoutesThroughController:
    @pytest.fixture(autouse=True)
    def _fast_sleep(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)

    def test_body_rotation_ships_a_head_matrix(self):
        """THE fix: turn_around's body step must not reach the SDK with head=None.

        The pre-fold code called `self.mini.goto_target(body_yaw=...)` raw —
        the daemon held the head at its retained world pose while the body
        rotated up to ±160° beneath it, commanding a head-relative angle far
        past the ±65° neck capability (the motor-destruction incident).
        """
        ctl, fake = _make_controller()
        host = _TurnHost(ctl)
        host.turn_around(90.0, duration=5.0)

        assert len(fake.calls) == 2, "expected STEP 1 (center head) + STEP 2 (rotate body)"
        step2 = fake.calls[1]
        assert step2.get("body_yaw") is not None
        assert step2.get("head") is not None, (
            "turn_around's body rotation reached the SDK with head=None — the "
            "counter-rotation trap (head-frame invariant) AND the neck-limit "
            "breach that destroyed motors 2+3."
        )
        # Head must ride along: world head yaw ≈ target body yaw.
        assert _world_yaw_of(step2["head"]) == pytest.approx(math.radians(90.0), abs=1e-3)

    def test_turn_beyond_body_limit_is_clamped(self):
        ctl, fake = _make_controller()
        host = _TurnHost(ctl)
        host.turn_around(300.0, duration=5.0)
        step2 = fake.calls[1]
        assert step2["body_yaw"] == pytest.approx(math.radians(160.0), abs=1e-6)

    def test_no_controller_skips_motion_instead_of_raw_sdk(self):
        host = _TurnHost(None)
        host.mini = _FakeMini()  # a raw SDK handle exists but must NOT be used
        host.turn_around(90.0, duration=5.0)
        assert host.mini.calls == [], "turn_around must never dispatch to the raw SDK"


# ─────────────────────────────────────────────────────────────────────────────
# Bypass path 1: Selfy.move() must clamp to the workspace
# ─────────────────────────────────────────────────────────────────────────────


class _MoveHost(MovementMixin):
    """Minimal Selfy stand-in for move(): records what would hit the motors."""

    def __init__(self) -> None:
        self.log = logging.getLogger("test_move_host")
        self.mini = _FakeMini()
        self.duration = 0.5
        self.x = self.y = self.z = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self._workspace_limit_override = None
        self._default_network = None
        self.enqueued: list[tuple] = []

    def _enqueue_motor(self, fn, *args, **kwargs):
        self.enqueued.append((fn, args, kwargs))


class TestSelfyMoveClamps:
    def test_llm_scale_yaw_is_clamped_to_workspace(self):
        """The MoveTool no-robot_id fallback: move(yaw=170) must not reach the
        motors at 170°. Pre-fold, only the max-step limiter ran — and its
        defaults are all 0.0 (disabled) — so the raw absolute pose shipped."""
        host = _MoveHost()
        host.move(yaw=170.0)
        assert host.enqueued, "move() should have enqueued a motor command"
        _, args, _ = host.enqueued[0]
        # move_head(mini, x, y, z, roll, pitch, yaw, duration)
        commanded_yaw = args[6]
        assert abs(commanded_yaw) <= host._SAFE_YAW_LIMIT + 1e-6, (
            f"move() enqueued yaw={commanded_yaw}° — beyond the ±{host._SAFE_YAW_LIMIT}° "
            "workspace. This is bypass path 1 of the motor-destruction incident."
        )
        assert host.yaw == commanded_yaw, "internal mirror must match the clamped command"

    def test_extreme_translation_is_clamped(self):
        host = _MoveHost()
        host.move(y=500.0, z=-500.0)
        _, args, _ = host.enqueued[0]
        assert abs(args[2]) <= host._SAFE_Y_LIMIT + 1e-6
        assert abs(args[3]) <= host._SAFE_Z_LIMIT + 1e-6

    def test_legal_move_unchanged(self):
        host = _MoveHost()
        host.move(yaw=20.0, pitch=10.0)
        _, args, _ = host.enqueued[0]
        assert args[6] == pytest.approx(20.0)
        assert args[5] == pytest.approx(10.0)


# ─────────────────────────────────────────────────────────────────────────────
# Review folds (two-lens round, 2026-08-07)
# ─────────────────────────────────────────────────────────────────────────────


from maxim.motion.movement import move_head  # noqa: E402


class TestMoveHeadFrameComposition:
    """BLOCKING review finding: move_head shipped BODY-RELATIVE yaw AS a
    WORLD-frame matrix with body_yaw=None. With the body turned (post SEM
    turn), a 'centered' gaze command demanded a head-relative angle up to
    ±160° against the ±65° neck — the destruction mechanism one layer up."""

    def test_relative_zero_on_turned_body_ships_world_body_yaw(self):
        body = math.radians(120.0)
        fake = _FakeMini(body_yaw=body)
        move_head(fake, 0, 0, 0, 0, 0, 0.0, 0.5)
        (call,) = fake.calls
        assert _world_yaw_of(call["head"]) == pytest.approx(body, abs=1e-6), (
            "move_head(yaw=0) on a body at 120° must center the head ON THE BODY "
            "(world 120°), not drag it to world 0 — a relative demand of −120°."
        )

    def test_relative_yaw_composes_onto_body(self):
        body = math.radians(90.0)
        fake = _FakeMini(body_yaw=body)
        move_head(fake, 0, 0, 0, 0, 0, 30.0, 0.5)
        (call,) = fake.calls
        assert _world_yaw_of(call["head"]) == pytest.approx(body + math.radians(30.0), abs=1e-6)

    def test_rotations_clamped_to_capability(self):
        fake = _FakeMini()
        move_head(fake, 0, 0, 0, 90.0, -90.0, 170.0, 0.5)
        (call,) = fake.calls
        m = call["head"]
        yaw = _world_yaw_of(m)
        pitch = math.atan2(-m[2][0], math.hypot(m[2][1], m[2][2]))
        roll = math.atan2(m[2][1], m[2][2])
        assert yaw == pytest.approx(math.radians(65.0), abs=1e-6)
        assert roll == pytest.approx(math.radians(40.0), abs=1e-6)
        assert pitch == pytest.approx(-math.radians(40.0), abs=1e-6)

    def test_body_read_failure_falls_back_to_zero(self):
        class _NoJointsMini(_FakeMini):
            def get_current_joint_positions(self):
                raise RuntimeError("joint read failed")

        fake = _NoJointsMini()
        move_head(fake, 0, 0, 0, 0, 0, 20.0, 0.5)
        (call,) = fake.calls
        assert _world_yaw_of(call["head"]) == pytest.approx(math.radians(20.0), abs=1e-6)

    def test_workers_recovery_recenter_is_body_relative(self):
        """workers.py's IK-failure recovery calls move_head(mini, 0,...,0) —
        with the frame fix this now means 'centered on the body', never a
        world-zero demand against a turned body during IK distress."""
        body = math.radians(-150.0)
        fake = _FakeMini(body_yaw=body)
        move_head(fake, 0, 0, 0, 0, 0, 0, 0.5)
        (call,) = fake.calls
        assert _world_yaw_of(call["head"]) == pytest.approx(body, abs=1e-6)


class _PoseHost(_MoveHost):
    def __init__(self, poses=None) -> None:
        super().__init__()
        self.poses = poses or {}


class TestGotoPoseRoutesThroughMove:
    def test_config_pose_beyond_workspace_is_clamped(self):
        """Review finding: goto_pose enqueued config-authored poses raw."""
        host = _PoseHost(poses={"wild": [0, 0, 0, 0, 0, 170.0]})
        host.goto_pose("wild", duration=0.1)
        assert host.enqueued, "goto_pose should dispatch via move()"
        _, args, _ = host.enqueued[0]
        assert abs(args[6]) <= host._SAFE_YAW_LIMIT + 1e-6

    def test_centered_pose_dispatches(self):
        host = _PoseHost(poses={"centered": [0, 0, 0, 0, 0, 10.0]})
        host.goto_pose("centered", duration=0.1)
        assert host.enqueued
        _, args, _ = host.enqueued[0]
        assert args[6] == pytest.approx(10.0)


class TestBoundsLearnerTightenOnly:
    def test_learned_bound_cannot_widen_safe_limit(self):
        """Review finding: _get_workspace_limits replaced the hardcoded safe
        limits with learner output wholesale — a config-inflated learner
        ceiling silently widened the safety clamp."""

        class _WideLearner:
            def get_bound(self, axis):
                return 999.0

        class _DN:
            _bounds_learner = _WideLearner()

        host = _MoveHost()
        host._default_network = _DN()
        limits = host._get_workspace_limits()
        assert limits["yaw"] <= host._SAFE_YAW_LIMIT
        assert limits["y"] <= host._SAFE_Y_LIMIT

    def test_learned_bound_can_tighten(self):
        class _TightLearner:
            def get_bound(self, axis):
                return 10.0

        class _DN:
            _bounds_learner = _TightLearner()

        host = _MoveHost()
        host._default_network = _DN()
        assert host._get_workspace_limits()["yaw"] == pytest.approx(10.0)


class TestSimulatedControllerParity:
    def test_default_limits_match_real_controller(self):
        """Review finding: sim defaults were unclamped while the real
        controller now clamps unconditionally — silent sim/hardware
        divergence in the incident's axis."""
        from maxim.hardware.simulation.controller import SimulatedController

        sim = SimulatedController()
        sim.connect()
        assert sim.goto_target(MotionTarget(head_yaw=math.radians(120.0))) is True
        pose = sim.get_current_pose()
        rel = pose["yaw"] - pose["body_yaw"]
        assert rel == pytest.approx(ReachyMiniController._MAX_HEAD_REL_YAW_RAD, abs=1e-6)

    def test_default_body_limit_matches_real_controller(self):
        from maxim.hardware.simulation.controller import SimulatedController

        sim = SimulatedController()
        sim.connect()
        assert sim.goto_target(MotionTarget(body_yaw=6.0)) is True
        assert sim.get_current_pose()["body_yaw"] == pytest.approx(ReachyMiniController._MAX_BODY_YAW_RAD, abs=1e-6)

    def test_explicit_tighter_limit_still_wins(self):
        from maxim.hardware.simulation.controller import SimulatedController

        sim = SimulatedController(head_yaw_limit_deg=22.0)
        sim.connect()
        sim.goto_target(MotionTarget(head_yaw=math.radians(60.0)))
        pose = sim.get_current_pose()
        assert pose["yaw"] - pose["body_yaw"] == pytest.approx(math.radians(22.0), abs=1e-6)


class TestControllerClampHonesty:
    def test_clamped_axes_recorded(self, controller):
        ctl, fake = controller
        ctl.goto_target(MotionTarget(head_yaw=math.radians(120.0), body_yaw=6.0))
        assert "body_yaw" in ctl.last_clamped_axes
        assert any("head_yaw" in a for a in ctl.last_clamped_axes)

    def test_clean_dispatch_clears_previous_record(self, controller):
        ctl, fake = controller
        ctl.goto_target(MotionTarget(body_yaw=6.0))
        assert ctl.last_clamped_axes
        ctl.goto_target(MotionTarget(body_yaw=0.2))
        assert ctl.last_clamped_axes == ()


class TestExecutorLensFolds:
    """Second review-round folds (executor lens)."""

    def test_selfy_move_on_turned_body_dispatches_body_relative(self):
        """THE cross-confirmed BLOCKING finding, end-to-end: after a body turn
        (mirror yaw correctly body-relative ≈ 0), a Selfy.move() must NOT drag
        the head to world 0 — the enqueued move_head composes world = rel +
        current body at dispatch time."""
        body = math.radians(90.0)
        host = _MoveHost()
        host.mini = _FakeMini(body_yaw=body)
        host.move(pitch=5.0)  # yaw axis filled from mirror (rel 0)
        assert host.enqueued
        fn, args, kwargs = host.enqueued[0]
        kwargs.pop("_commanded_6d", None)
        fn(*args, **kwargs)  # execute what the motor worker would
        (call,) = host.mini.calls
        assert _world_yaw_of(call["head"]) == pytest.approx(body, abs=1e-6), (
            "move(pitch=5) with the body at 90° shipped the head toward world 0 — "
            "a relative demand of −90° against the ±65° neck: the destruction "
            "mechanism, reincarnated through the motor queue."
        )

    def test_turn_around_step1_centers_relative_to_body_not_world(self):
        """STEP 1 must center the head ON THE BODY. At body=0 'world 0' and
        'relative 0' coincide, so the original tests could not detect a
        world-frame regression — this starts with the body turned."""
        body = math.radians(60.0)
        ctl, fake = _make_controller(body_yaw=body, head_world_yaw=body)
        host = _TurnHost(ctl)
        host.turn_around(45.0, duration=5.0)
        step1 = fake.calls[0]
        assert step1.get("head") is not None
        assert _world_yaw_of(step1["head"]) == pytest.approx(body, abs=1e-3), (
            "STEP 1 centered the head at WORLD 0 with the body at 60° — the "
            "counter-rotation trap turn_around was rewritten to close."
        )

    def test_controller_refuses_retained_yaw_on_unreadable_pose(self):
        """Body-only command + unreadable pose must REFUSE, not guess world 0
        (a legal-under-clamp but violent swing while state is unknown)."""
        ctl, fake = _make_controller()
        ctl.get_current_pose = lambda: {}  # type: ignore[method-assign]
        assert ctl.goto_target(MotionTarget(body_yaw=0.5)) is False
        assert fake.calls == []

    def test_turn_around_rejected_motion_does_not_update_mirror(self):
        """A rejected STEP 2 must not leave the mirror asserting a rotation
        that never happened."""

        class _RejectingController:
            def get_current_pose(self):
                return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "body_yaw": 0.0}

            def goto_target(self, target):
                return False

        host = _TurnHost(_RejectingController())
        host.body_yaw = 0.0
        host.yaw = 12.0
        host.turn_around(90.0, duration=5.0)
        assert host.body_yaw == pytest.approx(0.0), "mirror must not claim the failed turn"
        assert host.yaw == pytest.approx(12.0)

    def test_move_threads_motion_lock_to_move_head(self):
        """Selfy.move must pass the controller's _motion_lock through the
        motor queue so the body-read→compose→dispatch in move_head is
        serialized against controller-routed callers."""
        host = _MoveHost()
        sentinel = threading.RLock()

        class _R:
            _motion_lock = sentinel

        host._robot = _R()
        host.move(yaw=10.0)
        _, _, kwargs = host.enqueued[0]
        assert kwargs.get("motion_lock") is sentinel
