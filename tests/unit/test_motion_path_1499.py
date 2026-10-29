"""No too-fast, wrongly-referenced or mid-path-unsafe command reaches a Reachy motor (#1499, PR 1).

THE HAZARD: the SDK interpolates the head on the WORLD SO(3) geodesic (the short way round) and the body in joint
space, so whenever the body and the neck together ask for more than about 180° of world yaw, the neck passes through
about 180° mid-move. Both endpoints are legal, so every endpoint check passes. Live on main through center_vision from
a turned body, every SDK look_at (it also sends the body to 0), wake_up/goto_sleep from a turned body, and a large
turn_around. Mid-path the IK fails, the daemon holds the last joints, then they jump: the 2026-08-07 "head snapped to
the opposite extreme" failure (embodiment.md §1).

Design: docs/plans/motion_path_model.md (v8; owner decisions O1-O7, 2026-10-29). Each gate drives the real controller
against tests/unit/_reachy_fakes.py, and is a strict xfail until PR 1's fix lands.
"""

from __future__ import annotations

import logging
import math
import queue
import threading

import numpy as np
import pytest

from tests.unit._reachy_fakes import (
    SLEEP_HEAD_POSE,
    FakeDaemon,
    controller,
    head_pose,
    install,
    neck_twist_deg,
    worst_path_twist_deg,
)

D = math.radians
PR1 = "#1499 PR 1 (docs/plans/motion_path_model.md v8): "


def _gate(why: str):
    return pytest.mark.xfail(strict=True, reason=PR1 + why)


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


@_gate("the controller dispatches one move whose sampled neck twist passes ~180° (R5 split absent)")
@pytest.mark.parametrize("case", sorted(_SWEEPS))
def test_every_dispatched_chunk_keeps_the_neck_in_bound(daemon, case):
    start, target = _SWEEPS[case]
    _place(daemon, **start)
    ctl = controller(daemon)
    assert ctl.goto_target(_target(duration=1.0, **target))
    assert daemon.gotos(), "the move must be dispatched (split), not dropped"
    for path in _chunk_paths(daemon):
        assert _within_r5_bound(path), "a dispatched chunk sweeps the neck"


@_gate("MotionTarget has no duration_policy; an explicit duration over the path line is not refused")
def test_an_explicit_duration_over_the_path_line_is_refused_without_motion(daemon):
    _place(daemon, body_deg=160, rel_deg=30)
    ctl = controller(daemon)
    out = ctl.goto_target(_target(body_yaw=0.0, head_yaw=0.0, duration=1.0, duration_policy="exact"))
    assert not out and _kind(out) == "path"
    assert daemon.gotos() == []


@_gate("no path sampler; seeded legal moves are dispatched as single unchecked moves")
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


@_gate("durations are not stretched to the 180°/s average cap")
def test_a_derived_fast_body_turn_is_stretched_to_the_rate_cap(daemon):
    ctl = controller(daemon)
    assert ctl.goto_target(_target(body_yaw=D(160), duration=0.5))
    total = sum(float(c["duration"]) for c in daemon.gotos())
    assert total >= 160.0 / 180.0 - 1e-6


@_gate("ease_in_out's 2.0x peak is not capped at 340°/s")
def test_ease_in_out_is_stretched_for_its_peak(daemon):
    ctl = controller(daemon)
    assert ctl.goto_target(_target(body_yaw=D(160), duration=0.5, method="ease_in_out"))
    total = sum(float(c["duration"]) for c in daemon.gotos())
    assert total >= 2.0 * 160.0 / 340.0 - 1e-6


@_gate("an explicit too-fast duration is dispatched instead of refused")
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


@_gate("dispatch composes against the WS cache and never reads /api/state/full")
@pytest.mark.parametrize("mode", ["status500", "timeout", "missing_body", "nan_body", "unknown"])
def test_a_bad_fresh_read_refuses_with_reference_and_no_motion(daemon, mode):
    daemon.rest_mode = mode
    ctl = controller(daemon)
    stash_before = dict(ctl._last_commanded)
    out = ctl.goto_target(_target(head_yaw=D(10), duration=1.0))
    assert not out and _kind(out) == "reference"
    assert daemon.motion_calls() == []
    assert ctl._last_commanded == stash_before  # gate 15: a refusal before chunk 1 leaves the stash identical


@_gate("torque state is never checked before a dispatch")
def test_torque_off_refuses_with_reference(daemon):
    daemon.control_mode = "disabled"
    ctl = controller(daemon)
    out = ctl.goto_target(_target(head_yaw=D(10), duration=1.0))
    assert not out and _kind(out) == "reference"
    assert daemon.motion_calls() == []


@_gate("the REST read is not used, so its known shape variants are untested")
@pytest.mark.parametrize("mode", ["nested", "head_m"])
def test_the_known_rest_shape_variants_parse(daemon, mode):
    daemon.rest_mode = mode
    ctl = controller(daemon)
    assert ctl.goto_target(_target(head_yaw=D(10), duration=1.0))
    assert daemon.rest_calls >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Gate 4 and 5: divergence and a missing reference (R1, R2)
# ─────────────────────────────────────────────────────────────────────────────


@_gate("a head-only move composes against the stashed body, not the readback")
def test_a_diverged_body_stash_is_refused_once_then_forgotten(daemon):
    ctl = controller(daemon)
    ctl._last_commanded = {"body_yaw": D(150)}  # the daemon's body is at 0
    out = ctl.goto_target(_target(head_pitch=0.1, duration=1.0))
    assert not out and _kind(out) == "reference"
    assert daemon.gotos() == []
    assert "body_yaw" not in ctl._last_commanded
    assert ctl.goto_target(_target(head_pitch=0.1, duration=1.0))
    assert len(daemon.gotos()) == 1


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


@_gate("look_at_pixel calls the SDK's look_at_image, which sends the body to 0")
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


@_gate("wake_up runs the SDK's canned move (body → 0) from a turned body")
def test_wake_up_from_a_turned_body_recentres_before_the_canned_move(daemon):
    _place(daemon, body_deg=150, rel_deg=20)
    ctl = controller(daemon)
    assert ctl.wake_up()
    (wake,) = [c for c in daemon.calls if c["name"] == "wake_up"]
    assert abs(wake["start_body"]) < D(2.0), "the canned wake must start from a centred body"
    recentre = [c for c in daemon.gotos() if daemon.calls.index(c) < daemon.calls.index(wake)]
    assert sum(float(c["duration"]) for c in recentre) >= 2.0
    for path in _chunk_paths(daemon):
        assert _within_r5_bound(path)


def test_wake_up_from_the_sleep_pose_at_body_zero_does_not_recentre(daemon):
    """Pin (5.3's 2° deadband): asleep at body 0, nothing moves before the canned wake."""
    daemon.head = SLEEP_HEAD_POSE.copy()
    ctl = controller(daemon)
    ctl.wake_up()
    assert daemon.gotos() == []


@_gate("the enable_motors failure branch still commands a safety goto_sleep")
def test_a_failed_enable_motors_commands_no_motion(daemon):
    daemon.enable_raises = True
    ctl = controller(daemon)
    assert not ctl.wake_up()
    assert daemon.motion_calls() == []


@_gate("wake_up does not wait for the fire-and-forget torque-on")
def test_wake_up_waits_for_torque_then_wakes(daemon):
    daemon.control_mode = "disabled"
    daemon.torque_delay_s = 0.3
    ctl = controller(daemon)
    assert ctl.wake_up()
    assert daemon.rest_calls >= 1
    assert [c["name"] for c in daemon.calls].count("wake_up") == 1


@_gate("wake_up does not check that torque came on at all")
def test_wake_up_with_torque_that_never_comes_on_moves_nothing(daemon):
    daemon.control_mode = "disabled"
    daemon.torque_delay_s = 30.0
    ctl = controller(daemon)
    assert not ctl.wake_up()
    assert daemon.motion_calls() == []


# ─────────────────────────────────────────────────────────────────────────────
# Gate 9: automatic_body_yaw true is a config error (O6)
# ─────────────────────────────────────────────────────────────────────────────


@_gate("a controller configured with automatic_body_yaw=True is constructed")
def test_automatic_body_yaw_true_is_a_config_error():
    from maxim.exceptions import InvalidConfigError
    from maxim.hardware.reachy.controller import ReachyMiniController

    with pytest.raises(InvalidConfigError):
        ReachyMiniController(robot_name="reachy_mini", automatic_body_yaw=True)


# ─────────────────────────────────────────────────────────────────────────────
# Gate 12 (PR-1 half): a move that did not succeed never feeds the learners
# ─────────────────────────────────────────────────────────────────────────────


@_gate("motor_worker books any callable that returns, even False, as a successful gaze")
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


@_gate("there is no abort token, and a full-range move is one dispatch")
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


@_gate("an SDK TimeoutError mid-split is not a partial outcome, and nothing waits for the move to settle")
def test_an_sdk_timeout_mid_split_is_partial_and_blocks_until_settled(daemon):
    _place(daemon, body_deg=160)
    daemon.timeout_on_goto = 1
    ctl = controller(daemon)
    out = ctl.goto_target(_target(body_yaw=D(-160), duration=1.0))
    assert not out and _kind(out) == "partial"
    daemon.calls.clear()
    daemon.head = head_pose(yaw=D(77))  # not where the timed-out chunk was going, and not yet stable
    out2 = ctl.goto_target(_target(head_pitch=0.1, duration=1.0))
    assert not out2 and _kind(out2) == "reference"
    assert daemon.gotos() == []


@_gate("a silently dropped chunk is followed by an unchecked 300° body move")
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


@_gate("move_head with a controller still dispatches the raw SDK itself")
def test_move_head_with_a_controller_delegates_to_it(daemon):
    from maxim.motion.movement import move_head

    ctl = controller(daemon)
    seen: list = []
    ctl.goto_target = lambda target: seen.append(target) or True  # type: ignore[method-assign]
    move_head(daemon, 0, 0, 0, 0, 0, 10, 1.0, controller=ctl)
    assert len(seen) == 1
    assert daemon.gotos() == []


@_gate("standalone move_head accepts a call without motion_lock")
def test_standalone_move_head_requires_the_motion_lock(daemon):
    from maxim.motion.movement import move_head

    with pytest.raises(TypeError):
        move_head(daemon, 0, 0, 0, 0, 0, 10, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Gate 20: plugins that return bool keep working (O7)
# ─────────────────────────────────────────────────────────────────────────────


@_gate("hardware.controller has no MotionOutcome or as_outcome")
def test_as_outcome_reads_a_plugin_bool_as_an_outcome():
    from maxim.hardware.controller import MotionOutcome, as_outcome

    assert as_outcome(True).ok and as_outcome(True).kind is None
    assert not as_outcome(False) and as_outcome(False).kind == "error"
    done = MotionOutcome(ok=True, kind=None, reason="", duration_s=1.0, chunks_planned=1, chunks_done=1)
    assert as_outcome(done) is done


# ─────────────────────────────────────────────────────────────────────────────
# Gate 25: a dispatch from a tool thread cannot wait past the tool timeout
# ─────────────────────────────────────────────────────────────────────────────


@_gate("goto_target waits for _motion_lock without a bound")
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
