"""No non-finite, out-of-range-duration, or instant command ever reaches a Reachy motor (#1496).

THE INCIDENT CLASS: about 27 motion-path clamps of the form ``max(-L, min(L, x))`` turn NaN into a
finite +L, and several sit upstream of any check (LLM tool parameters via ``float("nan")``,
``json.load`` of ``default_poses.json``, a component YAML ``head_yaw``). A NaN ``head_yaw`` in a body
YAML commanded a 160 deg body swing in 1 s. Durations were unbounded (the SDK refuses only ``<= 0``,
so NaN passes), and antennas and the SDK ``look_at`` took an instant ``set_target`` on duration 0.
Motors 2+3 were destroyed by an unclamped pose (embodiment.md section 1).

Approach note: docs/plans/motion_finite_1496.md (design cleared 2026-10-27). Each gate drives the
real path against a fake SDK and is a strict xfail until its fix lands.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import types

import numpy as np
import pytest

NAN = float("nan")
INF = float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────


def _rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
            [-sp, cp * sr, cp * cr, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture(autouse=True)
def _stub_reachy_sdk(monkeypatch):
    """Pure-math stub for `reachy_mini.utils.create_head_pose` (CI has no `reachy` extra)."""

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


class _Mini:
    """Fake SDK handle: records every motion call; readbacks are configurable."""

    def __init__(self, *, body: float = 0.0, head=None, antennas=(0.0, 0.0), joints_raise: bool = False):
        self.body = body
        self.head = np.eye(4) if head is None else head
        self.antennas = list(antennas)
        self.joints_raise = joints_raise
        self.head_raise = False
        self.calls: list[tuple] = []

    def get_current_joint_positions(self):
        if self.joints_raise:
            raise RuntimeError("joint read failed")
        return [self.body] + [0.0] * 6, list(self.antennas)

    def get_current_head_pose(self):
        if self.head_raise:
            raise AssertionError("no head_pose message yet")
        return self.head

    def get_present_antenna_joint_positions(self):
        return tuple(self.antennas)

    def goto_target(self, **kw):
        self.calls.append(("goto_target", kw))

    def set_target(self, **kw):
        self.calls.append(("set_target", kw))

    def look_at_image(self, *a, **kw):
        self.calls.append(("look_at_image", a, kw))

    def motion_calls(self) -> list[tuple]:
        return [c for c in self.calls if c[0] in {"goto_target", "set_target", "look_at_image"}]


def _controller(mini: _Mini):
    from maxim.hardware.reachy.controller import ReachyMiniController

    ctl = ReachyMiniController.__new__(ReachyMiniController)
    ctl._mini = mini
    ctl._motion_lock = threading.RLock()
    ctl._last_commanded = {}
    ctl._automatic_body_yaw = False
    ctl._divergence_last_warn = 0.0
    ctl._robot_name = "test_reachy"
    ctl.is_connected = lambda: True  # type: ignore[method-assign]
    ctl._update_state = lambda **kw: None  # type: ignore[method-assign]
    return ctl


def _body(mini=None, robot=None):
    """A Selfy-shaped MovementMixin with the motor queue recorded instead of run."""
    from maxim.embodied_runtime.movement import MovementMixin

    class _Body(MovementMixin):
        def __init__(self):
            self.mini = mini
            self._robot = robot
            self.log = logging.getLogger("test.motion_finite_1496")
            self.duration = 0.5
            self.x = self.y = self.z = 0.0
            self.roll = self.pitch = self.yaw = 0.0
            self.body_yaw = 0.0
            self.poses = {"centered": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]}
            self._default_head_pose = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
            self._default_network = None
            self.enqueued: list[tuple] = []

        def _enqueue_motor(self, fn, *args, **kwargs):
            self.enqueued.append((fn, args, kwargs))

        def _clear_motor_queue(self) -> int:
            return 0

    return _Body()


class _RecordingRobot:
    """A RobotController-shaped fake that records every MotionTarget."""

    def __init__(self, *, body_yaw: float = 0.0, results=None):
        self.targets: list = []
        self._body_yaw = body_yaw
        self._results = list(results) if results else []
        self._motion_lock = threading.RLock()

    def is_connected(self) -> bool:
        return True

    def get_current_pose(self):
        return {"yaw": self._body_yaw, "pitch": 0.0, "roll": 0.0, "body_yaw": self._body_yaw}

    def goto_target(self, target) -> bool:
        self.targets.append(target)
        return self._results.pop(0) if self._results else True


@pytest.fixture
def no_sleep(monkeypatch):
    import time

    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    return slept


def _refused():
    from maxim.exceptions import MotionRefused

    return MotionRefused


# ─────────────────────────────────────────────────────────────────────────────
# Gate 1: a bad orient delta is refused at load; on the shipped path, bodiless plus one ERROR
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1496 red gate 1: orient deltas unvalidated")
@pytest.mark.parametrize("delta", [math.radians(120), NAN, INF, True])
def test_orient_backend_refuses_a_non_finite_bool_or_over_cap_delta(delta):
    from maxim.hardware.reachy.motor_backend import ReachyOrientMotorBackend

    with pytest.raises(_refused()):
        ReachyOrientMotorBackend(
            robot=None, deltas={"turn_left": delta}, modulator_name="orient", entity_name="reachy_mini"
        )


def _inject_head_yaw(monkeypatch, value):
    """Rewrite every orient self_effect head_yaw in the resolved spec, as a bad YAML would."""
    from maxim.embodiment.component_registry import ComponentRegistry

    real_get = ComponentRegistry.get

    def _walk(node):
        if isinstance(node, dict):
            se = node.get("self_effect")
            if isinstance(se, dict) and "head_yaw" in se:
                se["head_yaw"] = value
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    def _get(self, ref):
        spec = real_get(self, ref)
        _walk(spec)
        return spec

    monkeypatch.setattr(ComponentRegistry, "get", _get)


@pytest.mark.xfail(strict=True, reason="#1496 red gate 1: a bad YAML delta is not bodiless+ERROR")
@pytest.mark.parametrize("value", [2.5, NAN])
def test_a_bad_yaml_orient_delta_runs_bodiless_with_one_error(monkeypatch, caplog, value):
    import maxim.hardware.config as cfg_mod
    from maxim.embodied_runtime.agentic_runtime import AgenticRuntimeMixin
    from maxim.hardware.config import RobotsConfig

    declared = {"type": "reachy_mini", "primary": True, "config": {"body": "bodies/reachy_mini"}}
    cfg = RobotsConfig.from_dict({"robots": {"primary": declared}})
    monkeypatch.setattr(cfg_mod, "load_robots_config", lambda *a, **k: cfg)
    _inject_head_yaw(monkeypatch, value)

    class _Stub(AgenticRuntimeMixin):
        def __init__(self):
            self._robot_id = None
            self.name = "reachy_mini"
            self.log = logging.getLogger("test.resolve_body_wiring_1496")

    with caplog.at_level(logging.DEBUG):
        assert _Stub()._resolve_body_wiring(object(), object()) == (None, None)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "head_yaw" in errors[0].getMessage()


# ─────────────────────────────────────────────────────────────────────────────
# Gate 2: an LLM NaN never becomes motion, and a garbage angle is not a default 90 deg turn
# ─────────────────────────────────────────────────────────────────────────────


def _command_tool(maxim):
    from maxim.tools.reachy import MaximCommandTool

    tool = MaximCommandTool.__new__(MaximCommandTool)
    tool._maxim = maxim
    return tool


@pytest.mark.xfail(strict=True, reason="#1496 red gate 2: tool parameters unvalidated")
@pytest.mark.parametrize(
    "params",
    [{"angle": "nan"}, {"angle": "abc"}, {"angle": 45, "duration": "abc"}, {"angle": 45, "duration": 1e6}],
)
def test_maxim_command_turn_around_refuses_garbage_without_motion(params, no_sleep):
    robot = _RecordingRobot()
    body = _body(mini=_Mini(), robot=robot)
    result = _command_tool(body).execute(command="turn_around", params=params)
    assert result.success is False
    assert robot.targets == [] and body.enqueued == []


@pytest.mark.xfail(strict=True, reason="#1496 red gate 2: move() accepts NaN")
def test_maxim_command_move_with_nan_yaw_is_an_error_and_leaves_the_mirror():
    body = _body(mini=_Mini())
    before = (body.x, body.y, body.z, body.roll, body.pitch, body.yaw, body.duration)
    result = _command_tool(body).execute(command="move", params={"yaw": "nan"})
    assert result.success is False
    assert body.enqueued == []
    assert (body.x, body.y, body.z, body.roll, body.pitch, body.yaw, body.duration) == before


# ─────────────────────────────────────────────────────────────────────────────
# Gate 3: a NaN pose is refused without falling through to the default pose
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1496 red gate 3: goto_pose falls back to the default pose")
def test_goto_pose_with_a_nan_pose_is_refused_without_motion_or_sleep(no_sleep):
    body = _body(mini=_Mini())
    body.poses["bad"] = [0.0, 0.0, 0.0, 0.0, 0.0, NAN, 0.5]
    with pytest.raises(_refused()):
        body.goto_pose("bad")
    assert body.enqueued == [] and no_sleep == []


@pytest.mark.xfail(strict=True, reason="#1496 red gate 3: an unknown pose name moves to the default")
def test_goto_pose_with_an_unknown_name_is_an_error_without_motion(no_sleep):
    body = _body(mini=_Mini())
    result = _command_tool(body).execute(command="goto_pose", params={"name": "no_such_pose"})
    assert result.success is False and "centered" in (result.error or "")
    assert body.enqueued == []
    body2 = _body(mini=_Mini())
    assert _command_tool(body2).execute(command="goto_pose", params={"name": " centered "}).success is True


@pytest.mark.xfail(strict=True, reason="#1496 red gate 3: load_poses drops unparseable poses")
def test_an_unparseable_pose_is_kept_as_invalid_and_refused(tmp_path, no_sleep):
    from maxim.motion.movement import load_poses

    path = tmp_path / "poses.json"
    path.write_text('{"centered": [0,0,0,0,0,0,0.5], "weird": [0, null, 0, 0, 0, 0]}', encoding="utf-8")
    body = _body(mini=_Mini())
    body.poses = load_poses(path)
    assert "weird" in body.poses
    with pytest.raises(_refused()):
        body.goto_pose("weird")
    assert body.enqueued == []


# ─────────────────────────────────────────────────────────────────────────────
# Gate 4: MotionTarget / PixelTarget refuse non-finite values, bad durations, cartoon
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1496 red gate 4: MotionTarget unvalidated")
@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration": 0},
        {"duration": -1},
        {"duration": NAN},
        {"duration": 11},
        {"duration": True},
        {"body_yaw": NAN},
        {"head_yaw": INF},
        {"extras": {"antenna_left": NAN}},
        {"method": "Cartoon "},
        {"method": "bogus"},
    ],
)
def test_motion_target_refuses(kwargs):
    from maxim.hardware.controller import MotionTarget

    with pytest.raises(_refused()):
        MotionTarget(**kwargs)


@pytest.mark.xfail(strict=True, reason="#1496 red gate 4: PixelTarget unvalidated")
@pytest.mark.parametrize("kwargs", [{"duration": 0}, {"duration": NAN}, {"u": NAN}, {"v": 1.5}])
def test_pixel_target_refuses(kwargs):
    from maxim.hardware.controller import PixelTarget

    args = {"u": 320, "v": 240}
    args.update(kwargs)
    with pytest.raises(_refused()):
        PixelTarget(**args)


@pytest.mark.xfail(strict=True, reason="#1496 red gate 4: MotionTarget unvalidated")
def test_motion_target_accepts_the_inclusive_bounds_and_aliases():
    from maxim.hardware.controller import MotionTarget

    MotionTarget(duration=0.1)
    MotionTarget(duration=10.0)
    for m in ("minjerk", "min_jerk", "minimum_jerk", "linear", "ease", "ease_in_out", "Ease-In-Out"):
        MotionTarget(method=m)
    with pytest.raises(_refused()):
        MotionTarget(duration=0.0999)


# ─────────────────────────────────────────────────────────────────────────────
# Gate 5: a NaN readback refuses the move (controller and move_head)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1496 red gate 5: NaN body readback composes into the head target")
def test_controller_refuses_a_nan_body_readback():
    from maxim.hardware.controller import MotionTarget

    mini = _Mini(body=NAN)
    ctl = _controller(mini)
    assert ctl.goto_target(MotionTarget(head_yaw=0.1, head_pitch=0.0, head_roll=0.0, duration=0.5)) is False
    assert mini.motion_calls() == []


@pytest.mark.xfail(strict=True, reason="#1496 red gate 5: move_head composes NaN body")
@pytest.mark.parametrize("body", [NAN, INF])
def test_move_head_refuses_a_non_finite_body_readback(body):
    from maxim.motion.movement import move_head

    mini = _Mini(body=body)
    with pytest.raises(_refused()):
        move_head(mini, 0, 0, 0, 0, 0, 10, 0.5)
    assert mini.motion_calls() == []


@pytest.mark.xfail(strict=True, reason="#1496 red gate 5: NaN args clamp to +L")
def test_move_head_refuses_nan_arguments_instead_of_clamping():
    from maxim.motion.movement import move_head

    mini = _Mini()
    with pytest.raises(_refused()):
        move_head(mini, 0, 0, 0, 0, 0, NAN, 0.5)
    with pytest.raises(_refused()):
        move_head(mini, 0, 0, 0, 0, 0, 10, NAN)
    assert mini.motion_calls() == []


# ─────────────────────────────────────────────────────────────────────────────
# Gate 6: antennas always interpolate, bounded; look_at durations bounded
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1496 red gate 6: antenna set_target / unbounded")
def test_move_antenna_is_bounded_interpolated_and_keeps_the_body():
    from maxim.motion.movement import move_antenna

    mini = _Mini()
    with pytest.raises(_refused()):
        move_antenna(mini, right=10, duration=0)
    with pytest.raises(_refused()):
        move_antenna(mini, right=NAN)
    assert mini.motion_calls() == []

    move_antenna(mini, right=200, left=-200, method="min_jerk")
    move_antenna(mini, right=5, method="ease")
    assert [c[0] for c in mini.motion_calls()] == ["goto_target", "goto_target"]
    first, second = mini.motion_calls()[0][1], mini.motion_calls()[1][1]
    assert first["antennas"] == pytest.approx([math.radians(90), -math.radians(90)])
    assert first["method"] == "minjerk" and second["method"] == "ease_in_out"
    assert first["body_yaw"] is None and second["body_yaw"] is None


@pytest.mark.xfail(strict=True, reason="#1496 red gate 6: look_at takes the SDK's instant path")
def test_look_at_with_zero_duration_is_refused_on_both_paths():
    from maxim.hardware.controller import PixelTarget

    body = _body(mini=_Mini())
    with pytest.raises(_refused()):
        body.look_at_image(320, 240, duration=0)
    assert body.enqueued == []

    mini = _Mini()
    ctl = _controller(mini)
    target = PixelTarget(u=320, v=240, duration=0.5)
    target.duration = 0.0  # a post-construction write bypasses __post_init__
    assert ctl.look_at_pixel(target) is False
    assert mini.motion_calls() == []


# ─────────────────────────────────────────────────────────────────────────────
# Gate 7: turn_around validates first, floors derived steps, never runs step 2 after step 1 fails
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1496 red gate 7: derived step duration below the floor")
def test_turn_around_floors_its_derived_step_durations(no_sleep):
    robot = _RecordingRobot()
    _body(mini=_Mini(), robot=robot).turn_around(45, duration=0.2)
    assert len(robot.targets) == 2
    assert all(0.1 <= t.duration <= 10.0 for t in robot.targets)


@pytest.mark.xfail(strict=True, reason="#1496 red gate 7: NaN duration laundered into finite steps")
def test_turn_around_refuses_a_nan_duration_before_inhibiting(no_sleep):
    robot = _RecordingRobot()
    body = _body(mini=_Mini(), robot=robot)
    inhibited: list = []
    body._default_network = types.SimpleNamespace(inhibit=lambda **kw: inhibited.append(kw), release=lambda: None)
    with pytest.raises(_refused()):
        body.turn_around(45, duration=NAN)
    assert robot.targets == [] and inhibited == []


@pytest.mark.xfail(strict=True, reason="#1496 red gate 7: step 2 runs after step 1 fails")
def test_turn_around_skips_step_two_when_step_one_is_rejected(no_sleep):
    robot = _RecordingRobot(results=[False])
    body = _body(mini=_Mini(), robot=robot)
    body.yaw = 12.0
    body.turn_around(45, duration=5.0)
    assert len(robot.targets) == 1
    assert body.yaw == 12.0


# ─────────────────────────────────────────────────────────────────────────────
# Gate 8 + 11: a refused target leaves the stash identical; the last mile re-checks ranges
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1496 red gates 8+11: no last-mile range check")
def test_a_post_construction_bad_duration_is_refused_at_dispatch_and_keeps_the_stash():
    from maxim.hardware.controller import MotionTarget

    mini = _Mini()
    ctl = _controller(mini)
    assert ctl.goto_target(MotionTarget(head_yaw=0.2, head_pitch=0.0, head_roll=0.0, duration=0.5)) is True
    stash = dict(ctl._last_commanded)
    calls = len(mini.motion_calls())
    target = MotionTarget(head_yaw=0.3, duration=0.5)
    target.duration = 0.01
    assert ctl.goto_target(target) is False
    target2 = MotionTarget(head_yaw=0.3, duration=0.5)
    target2.method = "cartoon"
    assert ctl.goto_target(target2) is False
    assert ctl._last_commanded == stash
    assert len(mini.motion_calls()) == calls


# ─────────────────────────────────────────────────────────────────────────────
# Gate 9: the CI lint catches planted violations
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1496 red gate 9: no motion-safety lint")
def test_motion_safety_lint_catches_planted_violations(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "lint_motion_safety", Path(__file__).resolve().parents[2] / "scripts" / "lint_motion_safety.py"
    )
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)

    clamp = tmp_path / "src" / "maxim" / "motion" / "planted.py"
    clamp.parent.mkdir(parents=True)
    clamp.write_text(
        "def f(v):\n    a = max(-1.0, min(1.0, v))\n    b = min(2.0, max(-2.0, v))\n    return np.clip(v, -1, 1)\n"
    )
    raw = tmp_path / "src" / "maxim" / "tools" / "planted_raw.py"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        "def g(self):\n"
        "    fn = self.mini.set_target\n"
        "    getattr(self.mini, 'goto_target')(head=None)\n"
        "    self._mini.look_at_world(1, 2, 3)\n"
    )
    findings = lint.scan(tmp_path)
    kinds = sorted(f.kind for f in findings)
    assert kinds.count("clamp") == 3
    assert kinds.count("raw_sdk") == 3


# ─────────────────────────────────────────────────────────────────────────────
# Gate 10: a non-finite readback never reaches the mirror, never hangs, never counts as a disconnect
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1496 red gate 10: sync writes NaN into the mirror")
@pytest.mark.parametrize("body", [NAN, INF])
def test_sync_keeps_the_mirror_on_a_non_finite_readback(body):
    from maxim.embodied_runtime.movement import SyncResult

    selfy = _body(mini=_Mini(body=body))
    selfy.yaw, selfy.body_yaw = 7.0, 3.0
    done: list = []
    t = threading.Thread(target=lambda: done.append(selfy.sync_head_position()), daemon=True)
    t.start()
    t.join(timeout=2.0)
    assert done, "sync hung on a non-finite body readback"
    assert done[0] is SyncResult.NON_FINITE and not done[0]
    assert (selfy.yaw, selfy.body_yaw) == (7.0, 3.0)


@pytest.mark.xfail(strict=True, reason="#1496 red gate 10: a refusal counts as a connection failure")
def test_motor_worker_handles_a_refusal_without_a_connection_failure(caplog):
    import queue

    from maxim.embodied_runtime.workers import motor_worker
    from maxim.motion.movement import move_head

    mini = _Mini(body=NAN)
    selfy = _body(mini=mini)
    failures: list = []
    selfy._note_connection_failure = lambda *a: failures.append(a)
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    q.put((move_head, (mini, 0, 0, 0, 0, 0, 10, 0.5), {}))

    def _stop_when_drained():
        while not q.empty():
            stop.wait(0.01)
        stop.wait(0.3)
        stop.set()

    threading.Thread(target=_stop_when_drained, daemon=True).start()
    with caplog.at_level(logging.DEBUG):
        motor_worker(q, stop, selfy)
    assert failures == []
    assert mini.motion_calls() == []
    assert any(r.levelno == logging.ERROR for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# Gate 13: the named swallow-to-default sites
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1496 red gate 13: TrackTarget substitutes 0 -> 0.3")
def test_track_target_refuses_an_explicit_zero_duration():
    from maxim.tools.reachy import TrackTargetTool

    tool = TrackTargetTool(_body(mini=_Mini()))
    result = tool.execute(target="person", duration_s=0)
    assert result.success is False and "duration" in (result.error or "").lower()


@pytest.mark.xfail(strict=True, reason="#1496 red gate 13: a NaN centred pose seeds the mirror")
def test_a_nan_centred_pose_never_seeds_the_mirror():
    from maxim.embodied_runtime.selfy import centered_pose_from

    assert centered_pose_from({"centered": [0.0, 0.0, NAN, 0.0, 0.0, 0.0]}) is None
    assert centered_pose_from({"centered": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}) == {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "roll": 4.0,
        "pitch": 5.0,
        "yaw": 6.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gate 14: present-pose finiteness check (defence in depth; the WS transport drops non-finite poses)
# ─────────────────────────────────────────────────────────────────────────────


def _nan_head() -> np.ndarray:
    m = np.eye(4)
    m[0, 0] = NAN
    return m


def _inf_translation() -> np.ndarray:
    m = np.eye(4)
    m[2, 3] = INF
    return m


@pytest.mark.xfail(strict=True, reason="#1496 red gate 14: no present-pose check")
@pytest.mark.parametrize("head", [_nan_head(), _inf_translation()])
def test_a_non_finite_present_pose_refuses_every_motion_path(head):
    from maxim.hardware.controller import MotionTarget, PixelTarget
    from maxim.motion.movement import move_antenna, move_head

    mini = _Mini(head=head)
    with pytest.raises(_refused()):
        move_head(mini, 0, 0, 0, 0, 0, 5, 0.5)
    with pytest.raises(_refused()):
        move_antenna(mini, right=5)
    ctl = _controller(mini)
    assert ctl.goto_target(MotionTarget(body_yaw=0.1, duration=1.0)) is False
    assert ctl.look_at_pixel(PixelTarget(u=320, v=240, duration=0.5)) is False

    selfy = _body(mini=mini, robot=ctl)
    selfy._enqueue_sdk_look_at(320, 240, (320, 240), 0.5, perform_movement=True)
    fn, args, kwargs = selfy.enqueued[-1]
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
    with pytest.raises(_refused()):
        fn(*args, **kwargs)
    assert mini.motion_calls() == []

    mini.head = np.eye(4)
    move_head(mini, 0, 0, 0, 0, 0, 5, 0.5)
    assert len(mini.motion_calls()) == 1


@pytest.mark.xfail(strict=True, reason="#1496 red gate 14: present-pose helper missing")
def test_a_failed_pose_read_counts_as_missing_not_refused():
    from maxim.hardware.controller import require_present_pose_finite
    from maxim.motion.movement import move_head

    mini = _Mini()
    mini.head_raise = True
    require_present_pose_finite(mini)  # missing never refuses
    mini.head_raise = False
    mini.joints_raise = True
    require_present_pose_finite(mini)
    move_head(mini, 0, 0, 0, 0, 0, 5, 0.5)  # today's body-read fallback is unchanged (#1499 owns it)
    assert len(mini.motion_calls()) == 1
    bad = _Mini()
    bad.head = np.array([[1.0, None, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=object)
    with pytest.raises(_refused()):
        require_present_pose_finite(bad)


def test_the_ws_transport_drops_non_finite_poses(monkeypatch):
    """Documentation pin (#1496 decision 9). Over the reachy-mini WS transport a non-finite pose is
    serialised as null and fails client validation, so the client cache keeps the last finite pose and
    a readback finiteness check cannot fire on hardware. If this flips, the check became live."""
    # The autouse fixture stubs `reachy_mini`; this pin needs the real SDK (skipped where it is absent).
    for name in [m for m in sys.modules if m == "reachy_mini" or m.startswith("reachy_mini.")]:
        monkeypatch.delitem(sys.modules, name)
    protocol = pytest.importorskip("reachy_mini.io.protocol")
    from pydantic import ValidationError

    head = protocol.HeadPoseMsg(head_pose=[[NAN] * 4] * 4)
    with pytest.raises(ValidationError):
        protocol.HeadPoseMsg.model_validate_json(head.model_dump_json())
    fields = protocol.JointPositionsMsg.model_fields
    joint_kwargs = {
        name: ([NAN] + [0.0] * 6 if "head" in name else [0.0, 0.0])
        for name in fields
        if name != "type" and "joint" in name
    }
    joints = protocol.JointPositionsMsg(**joint_kwargs)
    with pytest.raises(ValidationError):
        protocol.JointPositionsMsg.model_validate_json(joints.model_dump_json())
