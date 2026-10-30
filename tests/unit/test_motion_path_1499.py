"""No too-fast, wrongly-referenced or mid-path-unsafe command reaches a Reachy motor (#1499, PR 1).

THE HAZARD: the SDK interpolates the head on the WORLD SO(3) geodesic (the short way round) and the body in joint
space, so whenever the body and the neck together ask for more than about 180° of world yaw, the neck passes through
about 180° mid-move. Both endpoints are legal, so every endpoint check passes. Live on main through center_vision from
a turned body, every SDK look_at (it also sends the body to 0), wake_up/goto_sleep from a turned body, and a large
turn_around. Mid-path the IK fails, the daemon holds the last joints, then they jump: the 2026-08-07 "head snapped to
the opposite extreme" failure (embodiment.md §1).

Design: docs/plans/motion_path_model.md (v8; owner decisions O1-O8, 2026-10-29). Each gate drives the real controller
against tests/unit/_reachy_fakes.py. They were committed as strict xfails before the fix (0e204fe9, 040b35b4).
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time

import numpy as np
import pytest

from scipy.spatial.transform import Rotation as R

from tests.unit._reachy_fakes import (
    INIT_ANTENNAS,
    INIT_HEAD_POSE,
    SLEEP_HEAD_POSE,
    FakeDaemon,
    controller,
    head_pose,
    install,
    neck_twist_deg,
    worst_path_twist_deg,
)

D = math.radians


@pytest.fixture
def daemon(monkeypatch):
    d = FakeDaemon()
    install(monkeypatch, d)
    return d


def _place(d: FakeDaemon, *, body_deg: float, rel_deg: float = 0.0, roll_deg: float = 0.0, pitch_deg: float = 0.0):
    d.body = D(body_deg)
    d.head = head_pose(roll=D(roll_deg), pitch=D(pitch_deg), yaw=D(body_deg + rel_deg))


def _target(**kw):
    from maxim.hardware.controller import MotionTarget

    return MotionTarget(**kw)


def _kind(result) -> str | None:
    from maxim.hardware.controller import as_outcome

    return as_outcome(result).kind


def as_outcome_reason(result) -> str:
    from maxim.hardware.controller import as_outcome

    return as_outcome(result).reason


def _chunk_paths(d: FakeDaemon):
    """(start_world, start_body, target_world, target_body) for every goto the fake daemon executed."""
    out = []
    for call in d.gotos():
        start_body = call["start_body"]
        target_body = start_body if call["body_yaw"] is None else float(call["body_yaw"])
        target_world = call["start_head"] if call["head"] is None else call["head"]
        out.append((call["start_head"], start_body, target_world, target_body))
    return out


def _within_r5_bound(path) -> bool:
    start_world, start_body, target_world, target_body = path
    worst = worst_path_twist_deg(start_world, start_body, target_world, target_body)
    endpoints = max(abs(neck_twist_deg(start_world, start_body)), abs(neck_twist_deg(target_world, target_body)))
    return worst <= min(endpoints + 10.0, 90.0)


# ─────────────────────────────────────────────────────────────────────────────
# Gate 1: the path rule (R5). Every dispatched chunk keeps the sampled neck twist in bound.
# ─────────────────────────────────────────────────────────────────────────────

_SWEEPS = {
    # center_vision from a turned body: body 160°, neck +30° → body 0, neck 0 (#1499's table: 178.5°).
    "center_vision_from_160": (
        dict(body_deg=160, rel_deg=30),
        dict(body_yaw=0.0, head_yaw=0.0, head_roll=0.0, head_pitch=0.0),
    ),
    # F6: sigma = 149.6° passes a sigma-only rule, but roll and pitch swing the geodesic the other way (179.7°).
    "f6_roll_pitch": (
        dict(body_deg=-55.7, rel_deg=10.2, roll_deg=-25.6, pitch_deg=-34.5),
        dict(body_yaw=D(73.9), head_yaw=D(30.0), head_roll=D(35.9), head_pitch=D(-29.3)),
    ),
    # A full-range body swing with the head riding along.
    "body_160_to_minus_160": (dict(body_deg=160, rel_deg=0), dict(body_yaw=D(-160), head_yaw=0.0)),
}


@pytest.mark.parametrize("case", sorted(_SWEEPS))
def test_every_dispatched_chunk_keeps_the_neck_in_bound(daemon, case):
    start, target = _SWEEPS[case]
    _place(daemon, **start)
    ctl = controller(daemon)
    assert ctl.goto_target(_target(duration=1.0, **target))
    assert daemon.gotos(), "the move must be dispatched (split), not dropped"
    for path in _chunk_paths(daemon):
        assert _within_r5_bound(path), "a dispatched chunk sweeps the neck"


def test_an_explicit_duration_over_the_path_line_is_refused_without_motion(daemon):
    _place(daemon, body_deg=160, rel_deg=30)
    ctl = controller(daemon)
    out = ctl.goto_target(_target(body_yaw=0.0, head_yaw=0.0, duration=1.0, duration_policy="exact"))
    assert not out and _kind(out) == "path"
    assert daemon.gotos() == []


def test_a_seeded_sweep_of_legal_moves_never_sweeps_the_neck(daemon):
    rng = np.random.default_rng(1499)
    for _ in range(40):
        b0, b1 = rng.uniform(-160, 160, size=2)
        r0, r1 = rng.uniform(-65, 65, size=2)
        ro0, ro1, p0, p1 = rng.uniform(-40, 40, size=4)
        _place(daemon, body_deg=b0, rel_deg=r0, roll_deg=ro0, pitch_deg=p0)
        daemon.calls.clear()
        ctl = controller(daemon)
        ctl.goto_target(_target(body_yaw=D(b1), head_yaw=D(r1), head_roll=D(ro1), head_pitch=D(p1), duration=1.0))
        for path in _chunk_paths(daemon):
            assert _within_r5_bound(path)


# ─────────────────────────────────────────────────────────────────────────────
# Gate 2: the rate cap (R4, O2)
# ─────────────────────────────────────────────────────────────────────────────


def test_a_derived_fast_body_turn_is_stretched_to_the_rate_cap(daemon):
    ctl = controller(daemon)
    assert ctl.goto_target(_target(body_yaw=D(160), duration=0.5))
    total = sum(float(c["duration"]) for c in daemon.gotos())
    assert total >= 160.0 / 180.0 - 1e-6


def test_ease_in_out_is_stretched_for_its_peak(daemon):
    ctl = controller(daemon)
    assert ctl.goto_target(_target(body_yaw=D(160), duration=0.5, method="ease_in_out"))
    total = sum(float(c["duration"]) for c in daemon.gotos())
    assert total >= 2.0 * 160.0 / 340.0 - 1e-6


def test_an_explicit_too_fast_duration_is_refused(daemon):
    ctl = controller(daemon)
    out = ctl.goto_target(_target(body_yaw=D(160), duration=0.5, duration_policy="exact"))
    assert not out and _kind(out) == "rate"
    assert daemon.gotos() == []


def test_a_small_look_within_the_cap_is_not_slowed(daemon):
    """Pin: 30° in 0.3 s is 100°/s, under every cap; it ships at the requested duration."""
    ctl = controller(daemon)
    assert ctl.goto_target(_target(head_yaw=D(30), duration=0.3))
    assert [round(float(c["duration"]), 3) for c in daemon.gotos()] == [0.3]


# ─────────────────────────────────────────────────────────────────────────────
# Gate 3: the reference is a fresh REST readback (R1, O3); a bad read moves nothing
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["status500", "timeout", "missing_body", "nan_body", "unknown"])
def test_a_bad_fresh_read_refuses_with_reference_and_no_motion(daemon, mode):
    daemon.rest_mode = mode
    ctl = controller(daemon)
    stash_before = dict(ctl._last_commanded)
    out = ctl.goto_target(_target(head_yaw=D(10), duration=1.0))
    assert not out and _kind(out) == "reference"
    assert daemon.motion_calls() == []
    assert ctl._last_commanded == stash_before  # gate 15: a refusal before chunk 1 leaves the stash identical


def test_torque_off_refuses_with_reference(daemon):
    daemon.control_mode = "disabled"
    ctl = controller(daemon)
    out = ctl.goto_target(_target(head_yaw=D(10), duration=1.0))
    assert not out and _kind(out) == "reference"
    assert daemon.motion_calls() == []


@pytest.mark.parametrize("mode", ["nested", "head_m"])
def test_the_known_rest_shape_variants_parse(daemon, mode):
    daemon.rest_mode = mode
    ctl = controller(daemon)
    assert ctl.goto_target(_target(head_yaw=D(10), duration=1.0))
    assert daemon.rest_calls >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Gate 4 and 5: divergence and a missing reference (R1, R2)
# ─────────────────────────────────────────────────────────────────────────────


def test_a_diverged_body_stash_is_refused_once_then_forgotten(daemon):
    ctl = controller(daemon)
    ctl._last_commanded = {"body_yaw": D(150)}  # the daemon's body is at 0
    out = ctl.goto_target(_target(head_pitch=0.1, duration=1.0))
    assert not out and _kind(out) == "reference"
    assert daemon.gotos() == []
    assert "body_yaw" not in ctl._last_commanded
    assert ctl.goto_target(_target(head_pitch=0.1, duration=1.0))
    assert len(daemon.gotos()) == 1


@pytest.mark.parametrize("body_deg, rel_deg", [(170, 0), (0, 75)])
def test_an_impossible_readback_refuses_with_reference(daemon, body_deg, rel_deg):
    """R2: a body beyond 160° + 5°, or a neck beyond 65° + 5°, cannot be where the robot is: refuse, move nothing."""
    _place(daemon, body_deg=body_deg, rel_deg=rel_deg)
    ctl = controller(daemon)
    out = ctl.goto_target(_target(head_pitch=0.1, duration=1.0))
    assert not out and _kind(out) == "reference"
    assert daemon.gotos() == []


def test_a_split_with_any_failing_chunk_is_refused_before_chunk_1_moves(daemon, monkeypatch):
    """R5: every chunk is sampled before anything is dispatched; one failing chunk refuses the whole move (path)."""
    import maxim.hardware.reachy.motion_path as mp

    real = mp.neck_path_ok
    calls = {"n": 0}

    def _second_chunk_fails(start, target):
        calls["n"] += 1
        ok, worst = real(start, target)
        return (False, 120.0) if calls["n"] == 3 else (ok, worst)  # the third sampled path (a later chunk) fails

    monkeypatch.setattr(mp, "neck_path_ok", _second_chunk_fails)
    _place(daemon, body_deg=160)
    ctl = controller(daemon)
    stash_before = dict(ctl._last_commanded)
    out = ctl.goto_target(_target(body_yaw=D(-160), duration=1.0))
    assert not out and _kind(out) == "path"
    assert daemon.gotos() == []
    assert ctl._last_commanded == stash_before


def test_a_world_yaw_past_180_is_not_dropped_as_impossible(daemon):
    """Pin (R2): world yaw legitimately reaches 225° (body 160° + neck 65°); only body and neck values are bounded."""
    _place(daemon, body_deg=160, rel_deg=40)  # world yaw 200°
    ctl = controller(daemon)
    assert ctl.goto_target(_target(head_pitch=0.1, duration=1.0))


def test_a_wrapped_world_yaw_composes_a_small_relative_command(daemon):
    """Pin (R3): world 170°, body 160°, a target relative yaw of 15° → world 175°, no refusal."""
    _place(daemon, body_deg=160, rel_deg=10)
    ctl = controller(daemon)
    assert ctl.goto_target(_target(head_yaw=D(15), duration=1.0))
    (call,) = daemon.gotos()
    assert neck_twist_deg(call["head"], daemon.body) == pytest.approx(15.0, abs=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Gate 7: look_at never moves the body (F1)
# ─────────────────────────────────────────────────────────────────────────────


def test_look_at_pixel_leaves_the_body_where_it_is(daemon):
    from maxim.hardware.controller import PixelTarget

    _place(daemon, body_deg=math.degrees(1.0))
    ctl = controller(daemon)
    assert ctl.look_at_pixel(PixelTarget(u=320, v=240, duration=1.0))
    assert daemon.body == pytest.approx(1.0)
    assert [c for c in daemon.calls if c["name"] == "look_at_image_move"] == []
    assert all(c["body_yaw"] is None for c in daemon.gotos())


# ─────────────────────────────────────────────────────────────────────────────
# Gate 8: lifecycle motion re-centres first and never moves when torque fails (5.3)
# ─────────────────────────────────────────────────────────────────────────────


def _lifecycle_names(d: FakeDaemon) -> list[str]:
    return [c["name"] for c in d.calls if c["name"] in {"wake_up", "goto_sleep"}]


def _rot_close(a, b, tol_deg: float = 0.5) -> bool:
    return math.degrees(R.from_matrix(np.asarray(a)[:3, :3].T @ np.asarray(b)[:3, :3]).magnitude()) < tol_deg


def test_wake_up_from_a_turned_body_recentres_before_the_lifecycle_steps(daemon):
    _place(daemon, body_deg=150, rel_deg=20)
    ctl = controller(daemon)
    assert ctl.wake_up()
    assert _lifecycle_names(daemon) == [], "O8: the SDK's canned wake is never called"
    gotos = daemon.gotos()
    # The first lifecycle step is the first goto that commands the antennas: the re-centre never does, and its last
    # chunk is itself at INIT's rotation, so the head alone cannot tell the two apart.
    first_init = next(i for i, c in enumerate(gotos) if c["antennas"] is not None)
    assert _rot_close(gotos[first_init]["head"], INIT_HEAD_POSE)
    assert first_init > 0, "a turned body re-centres before the INIT step"
    assert abs(gotos[first_init]["start_body"]) < D(2.0), "the lifecycle steps must start from a centred body"
    assert sum(float(c["duration"]) for c in gotos[:first_init]) >= 2.0
    for path in _chunk_paths(daemon):
        assert _within_r5_bound(path)


def test_the_lifecycle_recentre_holds_the_head_translation_in_the_body_frame(daemon):
    """5.3: the re-centre holds the readback's x/y/z, rotated with the body, so the head is never lifted or slid."""
    body = D(160)  # two body chunks: the first ends at 80°, where a world-held offset would have slid sideways
    t_body = np.array([-0.021, 0.0, -0.044])  # the sleep pose's offset, body frame
    c, s_ = math.cos(body), math.sin(body)
    world_t = np.array([c * t_body[0] - s_ * t_body[1], s_ * t_body[0] + c * t_body[1], t_body[2]])
    daemon.body = body
    daemon.head = head_pose(pitch=D(20), yaw=body, x=world_t[0], y=world_t[1], z=world_t[2])
    ctl = controller(daemon)
    assert ctl.wake_up()
    recentre = []
    for call in daemon.gotos():
        if call["antennas"] is not None:
            break
        recentre.append(call)
    assert len(recentre) >= 2, "a 160° body re-centres in more than one chunk"
    for call in recentre:
        b = float(call["body_yaw"])
        cb, sb = math.cos(-b), math.sin(-b)
        t = np.asarray(call["head"])[:3, 3]
        in_body = np.array([cb * t[0] - sb * t[1], sb * t[0] + cb * t[1], t[2]])
        np.testing.assert_allclose(in_body, t_body, atol=1e-9)


def test_wake_up_from_the_sleep_pose_at_body_zero_runs_only_the_lifecycle_steps(daemon):
    """5.3's 2° deadband: asleep at body 0 there is no re-centre, so the first dispatch is the INIT step (F11)."""
    daemon.head = SLEEP_HEAD_POSE.copy()
    ctl = controller(daemon)
    assert ctl.wake_up()
    assert _lifecycle_names(daemon) == []
    heads = [c["head"] for c in daemon.gotos()]
    assert len(heads) == 3, "INIT, the 20° roll, INIT"
    assert _rot_close(heads[0], INIT_HEAD_POSE) and _rot_close(heads[2], INIT_HEAD_POSE)
    assert _rot_close(heads[1], head_pose(roll=D(20)))
    assert tuple(daemon.gotos()[0]["antennas"]) == pytest.approx(INIT_ANTENNAS)
    assert all(c["body_yaw"] == pytest.approx(0.0) for c in daemon.gotos())
    assert daemon.media.sounds == ["wake_up.wav"]


def test_goto_sleep_clamps_the_sleep_antennas_and_never_calls_the_sdk_sleep(daemon):
    from maxim.hardware.reachy.controller import ANTENNA_LIMIT_RAD

    ctl = controller(daemon)  # the fake starts at INIT_HEAD_POSE, body 0
    assert ctl.goto_sleep()
    assert _lifecycle_names(daemon) == []
    gotos = daemon.gotos()
    assert len(gotos) == 1, "a head already at INIT skips the INIT step"
    (last,) = gotos
    assert _rot_close(last["head"], SLEEP_HEAD_POSE)
    assert tuple(last["antennas"]) == pytest.approx((-ANTENNA_LIMIT_RAD, ANTENNA_LIMIT_RAD))
    assert daemon.media.sounds == ["go_sleep.wav"]
    assert "disable_motors" not in [c["name"] for c in daemon.calls], "the SDK's sleep keeps torque on; so does ours"


def test_goto_sleep_from_a_raised_head_goes_through_init_first(daemon):
    from maxim.hardware.reachy.controller import ANTENNA_LIMIT_RAD

    _place(daemon, body_deg=0, pitch_deg=-30)
    ctl = controller(daemon)
    assert ctl.goto_sleep()
    gotos = daemon.gotos()
    assert len(gotos) == 2
    assert _rot_close(gotos[0]["head"], INIT_HEAD_POSE) and float(gotos[0]["duration"]) >= 1.0
    assert tuple(gotos[0]["antennas"]) == pytest.approx(INIT_ANTENNAS)
    assert _rot_close(gotos[1]["head"], SLEEP_HEAD_POSE)
    for c in gotos:
        assert all(abs(a) <= ANTENNA_LIMIT_RAD + 1e-9 for a in c["antennas"])


def test_a_failed_sound_fails_no_lifecycle_step(daemon, caplog):
    daemon.media.sound_raises = True
    ctl = controller(daemon)
    with caplog.at_level(logging.WARNING):
        assert ctl.goto_sleep()
    assert _lifecycle_names(daemon) == []
    assert _rot_close(daemon.gotos()[-1]["head"], SLEEP_HEAD_POSE)
    assert any("sound" in r.getMessage().lower() for r in caplog.records), "a failed sound is logged, not swallowed"


def test_a_failed_enable_motors_commands_no_motion(daemon):
    daemon.enable_raises = True
    ctl = controller(daemon)
    assert not ctl.wake_up()
    assert daemon.motion_calls() == []


def test_wake_up_waits_for_torque_then_wakes(daemon):
    daemon.control_mode = "disabled"
    daemon.torque_delay_s = 0.3
    ctl = controller(daemon)
    assert ctl.wake_up()
    assert daemon.rest_calls >= 1
    assert _lifecycle_names(daemon) == [] and daemon.gotos(), "woken by the composed steps (O8)"


def test_wake_up_with_torque_that_never_comes_on_moves_nothing(daemon):
    daemon.control_mode = "disabled"
    daemon.torque_delay_s = 30.0
    ctl = controller(daemon)
    assert not ctl.wake_up()
    assert daemon.motion_calls() == []


# ─────────────────────────────────────────────────────────────────────────────
# Gate 9: automatic_body_yaw true is a config error (O6)
# ─────────────────────────────────────────────────────────────────────────────


def test_automatic_body_yaw_true_is_a_config_error():
    from maxim.exceptions import InvalidConfigError
    from maxim.hardware.reachy.controller import ReachyMiniController

    with pytest.raises(InvalidConfigError):
        ReachyMiniController(robot_name="reachy_mini", automatic_body_yaw=True)


def test_a_robots_yaml_automatic_body_yaw_true_is_refused_not_filtered():
    """Gate 9 through the registry (delta D4): connect_robot filters config keys to the constructor's signature, so
    the parameter stays and refuses; a true value is never silently dropped at DEBUG."""
    from maxim.exceptions import InvalidConfigError
    from maxim.hardware.reachy.controller import ReachyMiniController
    from maxim.hardware.registry import RobotRegistry

    registry = RobotRegistry()
    registry.register_controller_type("reachy_mini", ReachyMiniController)
    with pytest.raises(InvalidConfigError):
        registry.connect_robot(
            robot_id="r1", robot_type="reachy_mini", config={"robot_name": "reachy_mini", "automatic_body_yaw": True}
        )
    assert registry.get_robot("r1") is None


# ─────────────────────────────────────────────────────────────────────────────
# Gate 12 (PR-1 half): a move that did not succeed never feeds the learners
# ─────────────────────────────────────────────────────────────────────────────


def test_a_falsy_motor_result_never_reaches_the_learners():
    from maxim.embodied_runtime.workers import motor_worker

    booked: list = []

    class _Attention:
        def record_gaze(self, pos, success):
            booked.append(("gaze", success))

    class _Bounds:
        def record_movement_outcome(self, **kw):
            booked.append(("bounds", kw))

    class _DN:
        _attention_network = _Attention()
        _bounds_learner = _Bounds()

    class _Selfy:
        log = logging.getLogger("test.motion_path_1499")
        _default_network = _DN()
        yaw = pitch = roll = x = y = z = 0.0

        def sync_head_position(self):
            return True

    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    q.put((lambda **kw: False, (), {"_position_info": (0.1, 0.2), "_commanded_6d": {"yaw": 0.1}}))

    def _stop_when_drained():
        while not q.empty():
            stop.wait(0.01)
        stop.wait(0.3)
        stop.set()

    threading.Thread(target=_stop_when_drained, daemon=True).start()
    motor_worker(q, stop, _Selfy())
    assert ("gaze", True) not in booked
    assert not any(kind == "bounds" for kind, _ in booked)


# ─────────────────────────────────────────────────────────────────────────────
# Gates 16, 17, 22: a split stops cleanly (abort, SDK timeout, a dropped chunk)
# ─────────────────────────────────────────────────────────────────────────────


def test_an_abort_between_chunks_stops_the_split(daemon):
    _place(daemon, body_deg=160)
    ctl = controller(daemon)
    real_goto = daemon.goto_target

    def _goto_then_abort(**kw):
        real_goto(**kw)
        ctl.request_abort()

    daemon.goto_target = _goto_then_abort  # type: ignore[method-assign]
    out = ctl.goto_target(_target(body_yaw=D(-160), duration=1.0))
    assert not out and _kind(out) == "cancelled"
    assert len(daemon.gotos()) == 1


def test_an_sdk_timeout_mid_split_is_partial_and_blocks_until_settled(daemon):
    _place(daemon, body_deg=160)
    daemon.timeout_on_goto = 1
    ctl = controller(daemon)
    out = ctl.goto_target(_target(body_yaw=D(-160), duration=1.0))
    assert not out and _kind(out) == "partial"
    daemon.calls.clear()
    # The daemon kept moving after the timeout: the neck is at +24° now, neither where the timed-out chunk was going
    # nor stable. Every head axis is explicit, so only the settle rule can refuse this.
    daemon.head = head_pose(yaw=daemon.body + D(24))
    out2 = ctl.goto_target(_target(head_yaw=D(20), head_pitch=0.1, head_roll=0.0, duration=1.0))
    assert not out2 and _kind(out2) == "reference"
    assert "settled" in as_outcome_reason(out2)
    assert daemon.gotos() == []


def test_a_dropped_chunk_stops_the_split_as_partial(daemon):
    _place(daemon, body_deg=160)
    daemon.drop_next = 1
    ctl = controller(daemon)
    out = ctl.goto_target(_target(body_yaw=D(-160), duration=1.0))
    assert not out and _kind(out) == "partial"
    assert len(daemon.gotos()) == 1, "chunk 2 must never run from an unexpected start"


# ─────────────────────────────────────────────────────────────────────────────
# Gate 18: move_head delegates to the controller; standalone needs the lock
# ─────────────────────────────────────────────────────────────────────────────


def test_move_head_with_a_controller_delegates_to_it(daemon):
    from maxim.motion.movement import move_head

    ctl = controller(daemon)
    seen: list = []
    ctl.goto_target = lambda target: seen.append(target) or True  # type: ignore[method-assign]
    move_head(daemon, 0, 0, 0, 0, 0, 10, 1.0, controller=ctl)
    assert len(seen) == 1
    assert daemon.gotos() == []


def test_standalone_move_head_requires_the_motion_lock(daemon):
    from maxim.motion.movement import move_head

    with pytest.raises(TypeError, match="requires motion_lock"):
        move_head(daemon, 0, 0, 0, 0, 0, 10, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Gate 20: plugins that return bool keep working (O7)
# ─────────────────────────────────────────────────────────────────────────────


def test_as_outcome_reads_a_plugin_bool_as_an_outcome():
    from maxim.hardware.controller import MotionOutcome, as_outcome

    assert as_outcome(True).ok and as_outcome(True).kind is None
    assert not as_outcome(False) and as_outcome(False).kind == "error"
    done = MotionOutcome(ok=True, kind=None, reason="", duration_s=1.0, chunks_planned=1, chunks_done=1)
    assert as_outcome(done) is done


# ─────────────────────────────────────────────────────────────────────────────
# Gate 25: a dispatch from a tool thread cannot wait past the tool timeout
# ─────────────────────────────────────────────────────────────────────────────


def test_a_tool_thread_dispatch_behind_a_held_lock_returns_busy(daemon, monkeypatch):
    import maxim.hardware.reachy.controller as reachy_controller

    monkeypatch.setattr(reachy_controller, "MOTION_LOCK_WAIT_S", 0.2, raising=False)
    ctl = controller(daemon)
    held = threading.Event()
    release = threading.Event()

    def _hold():
        with ctl._motion_lock:
            held.set()
            release.wait(5.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    held.wait(1.0)
    result: list = []
    caller = threading.Thread(
        target=lambda: result.append(ctl.goto_target(_target(head_yaw=0.1, duration=1.0))), daemon=True
    )
    caller.start()
    caller.join(1.5)
    finished = not caller.is_alive()
    release.set()
    holder.join(2.0)
    caller.join(2.0)
    assert finished, "the dispatch must give up within the bound"
    assert result and not result[0] and _kind(result[0]) == "cancelled"


# ─────────────────────────────────────────────────────────────────────────────
# The pre-merge review folds (2026-10-29): owner decisions on the stop latch, the lifecycle settle rule, torque-off
# sleep and the read retry; plus tests for each mechanism a review deletion probe found green.
# ─────────────────────────────────────────────────────────────────────────────


def _raise_on_goto(daemon: FakeDaemon, exc: BaseException, *, on_call: int = 0):
    real = daemon.goto_target
    count = {"n": 0}

    def _goto(**kw):
        n = count["n"]
        count["n"] += 1
        if n == on_call:
            daemon._record("goto_target_failed", **{k: v for k, v in kw.items() if k in {"body_yaw", "duration"}})
            raise exc
        return real(**kw)

    daemon.goto_target = _goto  # type: ignore[method-assign]


@pytest.mark.parametrize("call", ["wake_up", "goto_sleep"])
def test_an_sdk_error_in_a_lifecycle_move_is_a_refusal_not_a_raise(daemon, call):
    """S1 (three lenses): the old wake/sleep caught everything; a raise would skip is_awake and escape unguarded callers."""
    daemon.head = SLEEP_HEAD_POSE.copy() if call == "wake_up" else daemon.head
    _raise_on_goto(daemon, RuntimeError("Task failed: daemon error"))
    ctl = controller(daemon)
    out = getattr(ctl, call)()
    assert not out and _kind(out) == "error"
    assert "Task failed" in as_outcome_reason(out)


def test_a_stop_during_a_wake_ends_it_at_the_next_step(daemon):
    """Fence S2: one stop generation per lifecycle call, checked before every step, so a stop mid-wake ends the wake."""
    daemon.head = SLEEP_HEAD_POSE.copy()
    ctl = controller(daemon)
    real = daemon.goto_target

    def _goto_then_stop(**kw):
        real(**kw)
        ctl.request_abort()

    daemon.goto_target = _goto_then_stop  # type: ignore[method-assign]
    out = ctl.wake_up()
    assert not out and _kind(out) == "cancelled"
    assert len(daemon.gotos()) == 1, "no wake step runs after the stop"
    assert daemon.media.sounds == [], "not even the wake sound"


def test_after_sleep_a_new_move_is_refused_until_wake(daemon):
    """The stop latch (owner, 2026-10-29): a command that starts after a stop never moves the robot."""
    from maxim.hardware.controller import PixelTarget

    ctl = controller(daemon)
    assert ctl.goto_sleep()
    n = len(daemon.gotos())
    out = ctl.goto_target(_target(head_yaw=0.2, duration=1.0))
    assert not out and _kind(out) == "cancelled"
    assert not ctl.look_at_pixel(PixelTarget(u=320, v=240, duration=1.0))
    assert len(daemon.gotos()) == n
    assert ctl.wake_up()
    assert ctl.goto_target(_target(head_yaw=0.2, duration=1.0))


def test_a_latched_dispatch_refuses_at_once_without_waiting_for_the_lock(daemon):
    """The latch refuses before the lock wait: a tool thread never sits 25 s behind a sleep it cannot follow."""
    ctl = controller(daemon)
    ctl._stop_latched = True
    held, release = threading.Event(), threading.Event()

    def _hold():
        with ctl._motion_lock:
            held.set()
            release.wait(5.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    held.wait(1.0)
    started = time.monotonic()
    out = ctl.goto_target(_target(head_yaw=0.1, duration=1.0))
    elapsed = time.monotonic() - started
    release.set()
    holder.join(2.0)
    assert not out and _kind(out) == "cancelled" and "stopped" in as_outcome_reason(out)
    assert elapsed < 0.5


def test_a_dispatch_waiting_behind_a_stop_never_moves(daemon):
    """Fence P2: a dispatch that was waiting for the lock when the stop came ends cancelled, never moves."""
    _place(daemon, body_deg=0)
    ctl = controller(daemon)
    held, release = threading.Event(), threading.Event()

    def _hold():
        with ctl._motion_lock:
            held.set()
            release.wait(5.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    held.wait(1.0)
    result: list = []
    waiter = threading.Thread(
        target=lambda: result.append(ctl.goto_target(_target(body_yaw=D(60), duration=1.0))), daemon=True
    )
    waiter.start()
    time.sleep(0.1)  # the waiter is blocked on the lock
    ctl.request_abort()
    release.set()
    waiter.join(5.0)
    holder.join(2.0)
    assert result and not result[0] and _kind(result[0]) == "cancelled"
    assert daemon.gotos() == []


def _unsettle(ctl, daemon):
    """Leave the controller unsettled after an SDK timeout, with the robot stopped somewhere else."""
    import maxim.hardware.reachy.motion_path as mp

    ctl._unsettled = (mp.Pose(head_pose(yaw=D(100)), D(100)), None, None)


def test_after_a_timeout_the_wake_refuses_until_settled(daemon):
    """Owner, 2026-10-29: wake refuses (no wait) while the robot may still be moving."""
    daemon.head = SLEEP_HEAD_POSE.copy()
    ctl = controller(daemon)
    _unsettle(ctl, daemon)
    out = ctl.wake_up()
    assert not out and _kind(out) == "reference"
    assert daemon.gotos() == []


def test_after_a_timeout_the_sleep_waits_for_settling_then_sleeps(daemon):
    """Owner, 2026-10-29: sleep waits (≤ 2 s) for the robot to settle; a completed sleep clears the unsettled state."""
    ctl = controller(daemon)
    _unsettle(ctl, daemon)  # the robot is still (at INIT), so it settles by stability after 1 s
    started = time.monotonic()
    assert ctl.goto_sleep()
    assert 1.0 <= time.monotonic() - started < 3.0
    assert ctl._unsettled is None


def test_after_a_timeout_the_sleep_refuses_when_the_robot_does_not_settle_in_time(daemon, monkeypatch):
    import maxim.hardware.reachy.controller as reachy_controller

    monkeypatch.setattr(reachy_controller, "_SLEEP_SETTLE_WAIT_S", 0.3)
    ctl = controller(daemon)
    _unsettle(ctl, daemon)
    out = ctl.goto_sleep()
    assert not out and _kind(out) == "reference"
    assert daemon.gotos() == []


def test_sleep_with_torque_off_is_already_safe(daemon):
    """Owner, 2026-10-29: a limp robot is safe; sleep returns ok and moves nothing (api releases the connection)."""
    daemon.control_mode = "disabled"
    ctl = controller(daemon)
    out = ctl.goto_sleep()
    assert out and "torque off" in as_outcome_reason(out)
    assert daemon.motion_calls() == []


def test_settled_clears_when_the_robot_is_at_the_timed_out_target(daemon):
    import maxim.hardware.reachy.motion_path as mp

    ctl = controller(daemon)
    ctl._unsettled = (mp.Pose(daemon.head.copy(), daemon.body), None, None)  # it got there after all
    assert ctl.goto_target(_target(head_yaw=0.1, duration=1.0))
    assert ctl._unsettled is None


def test_settled_clears_after_a_second_of_stability(daemon):
    ctl = controller(daemon)
    _unsettle(ctl, daemon)
    first = ctl.goto_target(_target(head_yaw=0.1, duration=1.0))
    assert not first and _kind(first) == "reference"
    time.sleep(1.05)
    assert ctl.goto_target(_target(head_yaw=0.1, duration=1.0))
    assert ctl._unsettled is None


def test_a_slow_fresh_read_past_the_deadline_refuses(daemon, monkeypatch):
    import maxim.hardware.reachy.controller as reachy_controller

    monkeypatch.setattr(reachy_controller, "FRESH_READ_DEADLINE_S", 0.02)
    daemon.rest_delay_s = 0.05
    ctl = controller(daemon)
    out = ctl.goto_target(_target(head_yaw=0.1, duration=1.0))
    assert not out and _kind(out) == "reference" and "deadline" in as_outcome_reason(out)
    assert daemon.gotos() == []


@pytest.mark.parametrize("tunnel, mode", [(True, "network"), (False, "localhost_only")])
def test_a_tunnelled_read_gets_the_tunnel_deadline(daemon, monkeypatch, tunnel, mode):
    """An auto-tunnel or a manual one (localhost_only) gets the longer deadline (Executor NIT)."""
    import maxim.hardware.reachy.controller as reachy_controller

    monkeypatch.setattr(reachy_controller, "FRESH_READ_DEADLINE_S", 0.02)
    monkeypatch.setattr(reachy_controller, "FRESH_READ_DEADLINE_TUNNEL_S", 1.0)
    daemon.rest_delay_s = 0.05
    ctl = controller(daemon)
    ctl._tunnel, ctl._connection_mode = tunnel, mode
    assert ctl.goto_target(_target(head_yaw=0.1, duration=1.0))


def test_the_fresh_read_is_private_and_bounded(daemon):
    """R1: the read is pinned to a private address and to a per-phase timeout that sums to the deadline."""
    import maxim.hardware.reachy.controller as reachy_controller

    ctl = controller(daemon)
    assert ctl.goto_target(_target(head_yaw=0.1, duration=1.0))
    kw = daemon.last_fetch_kwargs
    assert kw["address_policy"] == "private"
    policy = kw["timeout"]
    phases = policy.connect_s + policy.read_s + policy.total_s
    assert phases == pytest.approx(reachy_controller.FRESH_READ_DEADLINE_S)


def test_a_failed_read_between_chunks_is_retried_once(daemon):
    """Owner, 2026-10-29: one retry for the reads between chunks and after the last; the move completes."""
    _place(daemon, body_deg=160)
    ctl = controller(daemon)
    daemon.fail_rest_calls = {2}  # 1: the start read; 2: the read before chunk 2
    out = ctl.goto_target(_target(body_yaw=D(-160), duration=1.0))
    assert out, as_outcome_reason(out)
    assert len(daemon.gotos()) == 3


def test_the_first_read_of_a_move_is_never_retried(daemon):
    ctl = controller(daemon)
    daemon.fail_rest_calls = {1}
    out = ctl.goto_target(_target(head_yaw=0.1, duration=1.0))
    assert not out and _kind(out) == "reference"
    assert daemon.gotos() == [] and daemon.rest_calls == 1


def test_the_path_is_re_checked_from_the_re_read_pose_before_each_chunk(daemon, monkeypatch):
    """R5 delta S1: before chunk k > 1 the path is re-sampled from where the robot actually is."""
    import maxim.hardware.reachy.motion_path as mp

    real = mp.neck_path_ok
    calls = {"n": 0}

    def _recheck_fails(start, target):
        calls["n"] += 1
        return (False, 120.0) if calls["n"] == 4 else real(start, target)  # 1-3: planning; 4: the re-check

    monkeypatch.setattr(mp, "neck_path_ok", _recheck_fails)
    _place(daemon, body_deg=160)
    ctl = controller(daemon)
    out = ctl.goto_target(_target(body_yaw=D(-160), duration=1.0))
    assert not out and _kind(out) == "partial"
    assert len(daemon.gotos()) == 1


def test_move_head_translation_reaches_the_controller_in_the_body_frame(daemon):
    """5.4: move_head's mm nudges ride the delegation as body-frame extras, rotated with the body."""
    from maxim.motion.movement import move_head

    _place(daemon, body_deg=90)
    ctl = controller(daemon)
    assert move_head(daemon, 10.0, 0.0, -5.0, 0, 0, 0, 1.0, controller=ctl)
    (call,) = daemon.gotos()
    np.testing.assert_allclose(np.asarray(call["head"])[:3, 3], [0.0, 0.010, -0.005], atol=1e-9)  # x forward → +y world


def _run_worker(fn_result, *, with_info=True):
    from maxim.embodied_runtime.workers import motor_worker

    booked: list = []
    syncs: list = []

    class _Attention:
        def record_gaze(self, pos, success):
            booked.append(("gaze", success))

    class _Bounds:
        def record_movement_outcome(self, **kw):
            booked.append(("bounds", kw["ik_failure"]))

    class _DN:
        _attention_network = _Attention()
        _bounds_learner = _Bounds()

    class _Selfy:
        log = logging.getLogger("test.motion_path_1499")
        _default_network = _DN()
        yaw = pitch = roll = x = y = z = 0.0

        def sync_head_position(self):
            syncs.append(1)
            return True

    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    kwargs = {"_position_info": (0.1, 0.2), "_commanded_6d": {"yaw": 0.1}, "duration": 0.1} if with_info else {}
    q.put((lambda **kw: fn_result, (), kwargs))

    def _stop_when_drained():
        while not q.empty():
            stop.wait(0.01)
        stop.wait(0.4)
        stop.set()

    threading.Thread(target=_stop_when_drained, daemon=True).start()
    motor_worker(q, stop, _Selfy())
    return booked, syncs


def test_a_completed_move_books_the_learners():
    from maxim.hardware.controller import MotionOutcome

    booked, _ = _run_worker(MotionOutcome(ok=True, chunks_planned=1, chunks_done=1))
    assert ("gaze", True) in booked and ("bounds", False) in booked


@pytest.mark.parametrize("kind", ["reference", "partial", "cancelled", "rate"])
def test_a_non_ok_outcome_books_nothing_and_re_syncs_the_mirror(kind):
    """O4, and Wire F2: a refused or partial move re-reads the real pose (Selfy wrote its mirror before enqueueing)."""
    from maxim.hardware.controller import MotionOutcome

    booked, syncs = _run_worker(MotionOutcome(ok=False, kind=kind, reason="x"))
    assert booked == []
    assert syncs, "the mirror must re-sync after a move that did not complete"


def test_a_callable_that_returns_nothing_books_nothing():
    booked, syncs = _run_worker(None)
    assert booked == [] and syncs == []


def test_the_registry_broadcasts_stay_dict_of_bool():
    """O7 / MP7: the registry's broadcasts keep dict[str, bool] whatever a controller returns."""
    from maxim.hardware.controller import MotionOutcome, PixelTarget, RobotController
    from maxim.hardware.registry import RobotRegistry, RobotRegistration

    class _OutcomeController(RobotController):
        robot_type = "outcome"

        def connect(self, timeout=30.0):
            return True

        def disconnect(self):
            return None

        def goto_target(self, target):
            return MotionOutcome(ok=False, kind="rate", reason="too fast")

        def look_at_pixel(self, target):
            return MotionOutcome(ok=True)

        def get_current_pose(self):
            return {}

        def wake_up(self):
            return MotionOutcome(ok=True)

        def goto_sleep(self):
            return MotionOutcome(ok=True)

        def start_recording(self):
            return True

        def stop_recording(self):
            return True

        def get_video_stream(self):
            return None

        def get_audio_stream(self):
            return None

    registry = RobotRegistry()
    ctl = _OutcomeController("r1")
    registry._robots["r1"] = RobotRegistration(robot_id="r1", robot_type="outcome", controller=ctl, config={})
    for results in (
        registry.move_all(_target(head_yaw=0.1)),
        registry.look_at_all(PixelTarget(u=1, v=1)),
        registry.center_all(),
        registry.wake_all(),
        registry.sleep_all(),
    ):
        assert all(type(v) is bool for v in results.values()), results
    assert registry.move_all(_target(head_yaw=0.1)) == {"r1": False}


# ─────────────────────────────────────────────────────────────────────────────
# The delta round's folds (2026-10-29): the wake/sleep latch race, the owner's latch decisions, and guards for the
# fold mechanisms the delta round's mutation runner showed surviving deletion.
# ─────────────────────────────────────────────────────────────────────────────


class _OrderedLock:
    """The motion RLock, but a thread named "sleep" acquires only after the "wake" thread has: an unfair-lock race."""

    def __init__(self, inner, wake_in: threading.Event):
        self._inner, self._wake_in = inner, wake_in

    def acquire(self, blocking=True, timeout=-1):
        if threading.current_thread().name == "sleep":
            self._wake_in.wait(5.0)
        ok = self._inner.acquire(blocking, timeout)
        if ok and threading.current_thread().name == "wake":
            self._wake_in.set()
        return ok

    def release(self):
        self._inner.release()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


def test_a_wake_that_wins_the_lock_from_a_waiting_sleep_never_leaves_it_unlatched(daemon):
    """Delta SHOULD-FIX 1: sleep latches, waits; a wake requested after it wins the unfair lock and runs; the sleep then
    runs too. The robot must end asleep AND latched (the wake is lost: fail closed, owner 2026-10-29), so the next
    command is refused. Before the fold, the wake's latch clear survived the sleep."""
    ctl = controller(daemon)
    states: list = []
    ctl._update_state = lambda **kw: states.append(kw)  # type: ignore[method-assign]
    wake_in = threading.Event()
    ctl._motion_lock = _OrderedLock(ctl._motion_lock, wake_in)
    results: dict = {}
    sleeper = threading.Thread(target=lambda: results.__setitem__("sleep", ctl.goto_sleep()), name="sleep")
    sleeper.start()
    deadline = time.monotonic() + 5.0
    while not ctl._stop_latched and time.monotonic() < deadline:  # the sleep has latched and waits for the lock
        time.sleep(0.01)
    assert ctl._stop_latched
    waker = threading.Thread(target=lambda: results.__setitem__("wake", ctl.wake_up()), name="wake")
    waker.start()
    waker.join(10.0)
    sleeper.join(10.0)
    assert results["sleep"], "the sleep ran last"
    assert ctl._stop_latched and ctl._stop_by_sleep, "a sleep's latch, which a reconnect may not clear"
    assert [kw for kw in states if "is_awake" in kw][-1] == {"is_awake": False}, "the sleeping robot reports not awake"
    n = len(daemon.gotos())
    out = ctl.goto_target(_target(head_yaw=0.2, duration=1.0))
    assert not out and _kind(out) == "cancelled"
    ctl._clear_latch_after_connect()  # a reconnect
    assert not ctl.goto_target(_target(head_yaw=0.2, duration=1.0))
    assert len(daemon.gotos()) == n


def test_a_latched_robot_reports_not_awake_even_when_the_sleep_fails(daemon):
    """Owner 2026-10-29: latched means not awake, so the next run wakes it (which clears the latch)."""
    states: list = []
    ctl = controller(daemon)
    ctl._update_state = lambda **kw: states.append(kw)  # type: ignore[method-assign]
    daemon.rest_mode = "status500"  # the sleep cannot read the robot, so it refuses
    assert not ctl.goto_sleep()
    assert ctl._stop_latched
    assert {"is_awake": False} in states


def test_a_failed_wake_keeps_the_latch(daemon):
    """Owner 2026-10-29: only a wake that succeeds clears a stop."""
    ctl = controller(daemon)
    assert ctl.goto_sleep()
    daemon.enable_raises = True
    assert not ctl.wake_up()
    assert ctl._stop_latched
    assert not ctl.goto_target(_target(head_yaw=0.2, duration=1.0))
    daemon.enable_raises = False
    assert ctl.wake_up()
    assert not ctl._stop_latched


def test_a_wake_waiting_behind_a_stop_never_moves(daemon):
    """Mutant M2: a wake that was waiting for the lock when a stop came ends cancelled, moves nothing."""
    daemon.head = SLEEP_HEAD_POSE.copy()
    ctl = controller(daemon)
    held, release = threading.Event(), threading.Event()

    def _hold():
        with ctl._motion_lock:
            held.set()
            release.wait(5.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    held.wait(1.0)
    result: list = []
    waker = threading.Thread(target=lambda: result.append(ctl.wake_up()), daemon=True)
    waker.start()
    time.sleep(0.1)
    ctl.request_abort()
    release.set()
    waker.join(5.0)
    holder.join(2.0)
    assert result and not result[0] and _kind(result[0]) == "cancelled"
    assert daemon.motion_calls() == []


def test_a_stop_during_the_first_read_cancels_chunk_one(daemon):
    """Mutant M5: the stop generation is checked before chunk 1 too, not only between chunks."""
    ctl = controller(daemon)
    real = ctl._fresh_readback

    def _read_then_stop(**kw):
        pose = real(**kw)
        ctl.request_abort()
        return pose

    ctl._fresh_readback = _read_then_stop  # type: ignore[method-assign]
    out = ctl.goto_target(_target(head_yaw=0.2, duration=1.0))
    assert not out and _kind(out) == "cancelled" and out.chunks_done == 0
    assert daemon.gotos() == []


def test_the_final_read_is_retried_once(daemon):
    """Mutant M1 (owner MP12): the read after the last chunk gets one retry."""
    ctl = controller(daemon)
    daemon.fail_rest_calls = {2}  # 1: the start read; 2: the final read
    out = ctl.goto_target(_target(head_yaw=0.2, duration=1.0))
    assert out, as_outcome_reason(out)
    assert daemon.rest_calls == 3


def test_a_stop_during_the_settle_wait_ends_the_sleep_at_once(daemon):
    """Delta NIT 1: the sleep's settle wait honours the stop generation; it does not sit out its 2 s."""
    ctl = controller(daemon)
    _unsettle(ctl, daemon)
    threading.Timer(0.2, ctl.request_abort).start()
    started = time.monotonic()
    out = ctl.goto_sleep()
    assert not out and _kind(out) == "cancelled"
    assert time.monotonic() - started < 0.8
    assert daemon.gotos() == []


def test_a_wake_reports_the_whole_lifecycle_duration(daemon):
    """Delta NIT 4: the outcome's duration is the sum of every lifecycle step, not the last one."""
    daemon.head = SLEEP_HEAD_POSE.copy()
    ctl = controller(daemon)
    out = ctl.wake_up()
    assert out
    assert out.duration_s == pytest.approx(sum(float(c["duration"]) for c in daemon.gotos()))


# ─────────────────────────────────────────────────────────────────────────────
# The second delta round's folds (2026-10-29): a sleep with no SDK handle latches and refuses (owner), the latch
# transitions are atomic with the stop generation (owner), and guards for the mechanisms that survived its mutants.
# ─────────────────────────────────────────────────────────────────────────────


def test_a_sleep_with_no_sdk_handle_latches_and_refuses(daemon):
    """Delta-2 SF1 (owner: refuse): mid-reconnect there is no handle; the sleep latches first, then refuses, and the
    following connect does not clear a sleep's latch."""
    ctl = controller(daemon)
    states: list = []
    ctl._update_state = lambda **kw: states.append(kw)  # type: ignore[method-assign]
    ctl._mini = None
    out = ctl.goto_sleep()
    assert not out and _kind(out) == "error"
    assert {"is_awake": False} in states
    assert ctl._stop_latched and ctl._stop_by_sleep
    ctl._mini = daemon
    ctl._clear_latch_after_connect()  # what connect() does once the new handle is up
    out = ctl.goto_target(_target(head_yaw=0.2, duration=1.0))
    assert not out and _kind(out) == "cancelled"
    assert daemon.gotos() == []


def test_a_stop_after_the_wake_steps_finish_keeps_the_latch(daemon):
    """Delta-2 C/L: the wake's steps all succeed, but a sleep's stop lands before the wake's success check: the
    robot reports not awake, stays latched, and the next command is refused."""
    ctl = controller(daemon)
    assert ctl.goto_sleep()
    states: list = []
    ctl._update_state = lambda **kw: states.append(kw)  # type: ignore[method-assign]
    real = ctl._run_lifecycle

    def _then_stop(*a, **kw):
        out = real(*a, **kw)
        ctl._stop(by_sleep=True)  # what goto_sleep does before it waits for the lock
        return out

    ctl._run_lifecycle = _then_stop  # type: ignore[method-assign]
    out = ctl.wake_up()
    assert not out and _kind(out) == "cancelled"
    assert ctl._stop_latched
    assert states[-1] == {"is_awake": False}
    n = len(daemon.gotos())
    assert not ctl.goto_target(_target(head_yaw=0.2, duration=1.0))
    assert len(daemon.gotos()) == n


def test_a_sleep_latches_before_it_waits_for_the_lock(daemon):
    """Delta-2 J: a command that arrives while the sleep waits for the lock is refused at once, never moves."""
    ctl = controller(daemon)
    held, release = threading.Event(), threading.Event()

    def _hold():
        with ctl._motion_lock:
            held.set()
            release.wait(5.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    held.wait(1.0)
    sleeper = threading.Thread(target=ctl.goto_sleep, daemon=True)
    sleeper.start()
    time.sleep(0.2)  # the sleep has started and is blocked on the motion lock
    started = time.monotonic()
    out = ctl.goto_target(_target(head_yaw=0.2, duration=1.0))  # the lock is still held
    elapsed = time.monotonic() - started
    release.set()
    sleeper.join(5.0)
    holder.join(2.0)
    assert not out and _kind(out) == "cancelled"
    assert elapsed < 0.5, "refused by the latch at once, not after waiting for the lock"


def test_a_successful_wake_ends_the_sleep_flag(daemon):
    """Delta-2 E: after a sleep then a wake, a later disconnect's latch ends with the next connection."""
    ctl = controller(daemon)
    assert ctl.goto_sleep()
    assert ctl.wake_up()
    assert not ctl._stop_by_sleep and not ctl._stop_latched


def test_a_wake_behind_a_stop_never_enables_torque(daemon):
    """Delta-2 N: the wake checks for a stop before enable_motors."""
    ctl = controller(daemon)
    held, release = threading.Event(), threading.Event()

    def _hold():
        with ctl._motion_lock:
            held.set()
            release.wait(5.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    held.wait(1.0)
    result: list = []
    waker = threading.Thread(target=lambda: result.append(ctl.wake_up()), daemon=True)
    waker.start()
    time.sleep(0.1)
    ctl.request_abort()
    release.set()
    waker.join(5.0)
    holder.join(2.0)
    assert result and not result[0]
    assert "enable_motors" not in [c["name"] for c in daemon.calls]


def test_a_sleep_refuses_when_auto_yaw_is_not_asserted(daemon):
    """Delta-2 NIT 1: goto_sleep's own auto-yaw check (it reads the state itself, so it repeats the readback's)."""
    ctl = controller(daemon)
    ctl._auto_yaw_asserted = False
    out = ctl.goto_sleep()
    assert not out and _kind(out) == "reference"
    assert daemon.motion_calls() == []
    assert daemon.rest_calls == 0, "refused before reading the robot"


class _CountingLock:
    def __init__(self):
        self._inner = threading.Lock()
        self.acquired = 0

    def __enter__(self):
        self._inner.acquire()
        self.acquired += 1
        return self

    def __exit__(self, *exc):
        self._inner.release()


@pytest.mark.parametrize(
    "transition",
    [
        lambda c: c._stop(by_sleep=True),
        lambda c: c._latch_for_sleep(),
        lambda c: c._clear_latch_after_connect(),
        lambda c: c._clear_latch_after_wake(c._abort_gen),
    ],
    ids=["stop", "relatch", "connect_clear", "wake_clear"],
)
def test_every_latch_transition_runs_under_the_abort_lock(daemon, transition):
    """Delta-2 SF3 (owner: fold now): each latch read-then-write runs under _abort_lock with the stop generation."""
    ctl = controller(daemon)
    lock = _CountingLock()
    ctl._abort_lock = lock
    transition(ctl)
    assert lock.acquired == 1


def test_a_disconnect_during_a_split_stops_it_at_the_next_chunk(daemon):
    """Delta-3 F: disconnect stops a split in progress (its _stop bumps the generation), so no chunk follows."""
    _place(daemon, body_deg=160)
    ctl = controller(daemon)
    ctl._tunnel_proc = None
    ctl._video_stream = ctl._audio_stream = None
    real = daemon.goto_target

    def _goto_then_disconnect(**kw):
        real(**kw)
        mini = ctl._mini
        ctl.disconnect()  # releases the handle; the split must not touch it again
        ctl._mini = mini  # keep the fake reachable, so a further chunk would be recorded, not crash

    daemon.goto_target = _goto_then_disconnect  # type: ignore[method-assign]
    out = ctl.goto_target(_target(body_yaw=D(-160), duration=1.0))
    assert not out and _kind(out) == "cancelled"
    assert len(daemon.gotos()) == 1
    assert ctl._stop_latched and not ctl._stop_by_sleep
