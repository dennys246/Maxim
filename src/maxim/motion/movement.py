from __future__ import annotations

import logging
import math
import json
import numpy as np
from pathlib import Path
from typing import Any

from maxim.hardware.controller import clamp_finite, floor_duration, require_duration, require_finite
from maxim.hardware.reachy.controller import ANTENNA_LIMIT_RAD, check_sdk_command
from maxim.utils.logging import log_swallowed_exception, warn

_logger = logging.getLogger(__name__)

# Bundled seed data — shipped inside the package (src/maxim/_data/motion/).
_DATA_DIR = Path(__file__).resolve().parent.parent / "_data" / "motion"
_DEFAULT_ACTIONS_PATH = _DATA_DIR / "default_actions.json"
_DEFAULT_POSES_PATH = _DATA_DIR / "default_poses.json"
_DEFAULT_THRESHOLDS_PATH = _DATA_DIR / "movement_thresholds.json"


def _to_rad(value: float, *, degrees: bool) -> float:
    value = require_finite("angle", value)  # refuses a bool, a string or NaN as MotionRefused (#1496)
    return math.radians(value) if degrees else value


def load_actions(path: Path | str = _DEFAULT_ACTIONS_PATH) -> dict[str, Any]:
    actions_path = Path(path)
    with actions_path.open("r", encoding="utf-8") as file:
        actions = json.load(file)

    if not isinstance(actions, dict):
        raise ValueError(f"Expected top-level JSON object in {actions_path}, got {type(actions).__name__}")

    return actions


def load_poses(path: Path | str = _DEFAULT_POSES_PATH) -> dict[str, list[float] | None]:
    """
    Load named head poses from JSON.

    Each pose can be either:
    - list: [x, y, z, roll, pitch, yaw] or [x, y, z, roll, pitch, yaw, duration]
    - dict: {"x":..,"y":..,"z":..,"roll":..,"pitch":..,"yaw":..,"duration":..}

    A pose that does not parse is kept as ``None`` (the invalid sentinel), so
    ``goto_pose`` refuses that name instead of falling back to the default
    pose (#1496); so does a scalar, a string, or a list shorter than six
    values. NaN and Infinity survive the load and are refused at ``goto_pose``.
    """
    poses_path = Path(path)
    default: dict[str, list[float] | None] = {"centered": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]}

    if not poses_path.exists():
        return default

    try:
        with poses_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
    except Exception as e:
        warn("Failed to load poses from '%s': %s", poses_path, e)
        return default

    if not isinstance(raw, dict):
        return default

    parsed: dict[str, list[float] | None] = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not name.strip():
            continue

        vec: list[float] | None = None
        if isinstance(spec, (list, tuple)) and len(spec) >= 6:
            try:
                vec = [float(spec[i]) for i in range(6)]
                if len(spec) >= 7 and spec[6] is not None:
                    vec.append(float(spec[6]))
            except (TypeError, ValueError, OverflowError):
                parsed[name.strip()] = None  # invalid sentinel: goto_pose refuses this name
                continue
        elif isinstance(spec, dict):
            try:
                vec = [
                    float(spec.get("x", 0.0) or 0.0),
                    float(spec.get("y", 0.0) or 0.0),
                    float(spec.get("z", 0.0) or 0.0),
                    float(spec.get("roll", 0.0) or 0.0),
                    float(spec.get("pitch", 0.0) or 0.0),
                    float(spec.get("yaw", 0.0) or 0.0),
                ]
                if spec.get("duration") is not None:
                    vec.append(float(spec.get("duration") or 0.0))
            except (TypeError, ValueError, OverflowError):
                parsed[name.strip()] = None  # invalid sentinel: goto_pose refuses this name
                continue
        else:
            parsed[name.strip()] = None  # a scalar, string or short list: invalid, refused by name
            continue

        if vec is not None:
            parsed[name.strip()] = vec

    if not parsed:
        return default

    parsed.setdefault("centered", default["centered"])
    return parsed


def centered_pose_from(poses: Any, *, log: logging.Logger | None = None) -> dict[str, float] | None:
    """The ``centered`` pose as finite floats, or None to keep the seed values (#1496).

    A second reader of ``default_poses.json`` (``goto_pose`` is the first). A
    pose ``load_poses`` could not parse (kept as ``None``), a short one, or one
    holding NaN/Inf is logged at ERROR and ignored: the seed mirror stays, and
    ``goto_pose("centered")`` refuses that pose for the run. Every component is
    checked before any is used, so the mirror is never half-assigned.
    """
    if not isinstance(poses, dict) or "centered" not in poses:
        return None
    centered = poses["centered"]
    axes = ("x", "y", "z", "roll", "pitch", "yaw")
    if isinstance(centered, (list, tuple)) and len(centered) >= 6:
        try:
            values = [float(centered[i]) for i in range(6)]
        except (TypeError, ValueError):
            values = []
        if len(values) == 6 and all(math.isfinite(v) for v in values):
            return dict(zip(axes, values))
    (log or _logger).error(
        "default_poses.json: the 'centered' pose %r is not six finite numbers; keeping the seed head pose", centered
    )
    return None


def load_movement_thresholds(path: Path | str = _DEFAULT_THRESHOLDS_PATH) -> dict[str, Any]:
    """
    Load movement thresholds (used to clamp per-call head movement steps).

    Units match Maxim's head pose interface:
    - x/y/z: millimeters
    - roll/pitch/yaw: degrees
    """
    thresholds_path = Path(path)
    default: dict[str, Any] = {
        "head": {
            "max_step": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        }
    }

    if not thresholds_path.exists():
        return default

    try:
        with thresholds_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
    except Exception as e:
        warn("Failed to load movement thresholds from '%s': %s", thresholds_path, e)
        return default

    if not isinstance(raw, dict):
        return default

    head = raw.get("head")
    if not isinstance(head, dict):
        return default

    max_step = head.get("max_step")
    if not isinstance(max_step, dict):
        max_step = {}

    def _f(key: str, fallback: float = 0.0) -> float:
        try:
            val = max_step.get(key, fallback)
            return float(val)
        except Exception:
            return float(fallback)

    return {
        "head": {
            "max_step": {
                "x": _f("x"),
                "y": _f("y"),
                "z": _f("z"),
                "roll": _f("roll"),
                "pitch": _f("pitch"),
                "yaw": _f("yaw"),
            }
        }
    }


# Physical rotation limits (degrees) — CAPABILITY ceilings, mirroring
# ReachyMiniController's clamps (2026-08-07 safety fold; motors 2+3 were
# destroyed by a pose commanded beyond capability). Vendor SDK docs:
# roll/pitch ±40°, head yaw RELATIVE TO BODY max 65°. NOTE the 65° figure
# has a provenance conflict with the ~22° measured in Exp 49 — but that
# measurement was taken on the degraded platform; H1 re-measures on
# healthy hardware (docs/experiments/protocols/h1_healthy_hardware_doa_preregistration.md).
_MAX_HEAD_ROLL_DEG = 40.0
_MAX_HEAD_PITCH_DEG = 40.0
_MAX_HEAD_REL_YAW_DEG = 65.0


def _current_body_yaw_deg(mini) -> float:
    """Body yaw (degrees) from the SDK joint vector, or ``MotionRefused("reference")`` when it cannot be read.

    Joint vector is ``[body_yaw, *stewart_legs]`` — the SDK's own fk() reads
    index 0 as body_yaw. A missing read is refused (#1499 R2): the historical
    0.0 fallback composed a world head yaw off by the full body angle.
    """
    from maxim.exceptions import MotionRefused

    body: Any = None
    try:
        head_joints, _ = mini.get_current_joint_positions()
        if head_joints is not None and len(head_joints) >= 1:
            body = head_joints[0]
    except Exception as e:  # missing: refused below
        log_swallowed_exception()
        raise MotionRefused("reference", f"the body yaw could not be read: {e}") from e
    if body is None:
        raise MotionRefused("reference", "the body yaw could not be read")
    return math.degrees(require_finite("readback body_yaw", body))


_REQUIRED: Any = object()


def move_head(mini, x, y, z, roll, pitch, yaw, duration, *, motion_lock=_REQUIRED, controller=None):
    """Move the head to a BODY-RELATIVE pose: ``roll``/``pitch``/``yaw`` in degrees, ``x``/``y``/``z`` in mm.

    With a ``controller`` (every Selfy and worker caller), this DELEGATES to its
    ``goto_target`` (#1499 5.4), so one copy of the fresh readback, the rate cap
    and the neck-path split governs every head move, and its ``MotionOutcome``
    is returned. The translation rides along as body-frame ``head_x/y/z`` extras.

    Standalone (scripts with no controller), ``motion_lock=`` is required, so
    forgetting it is a ``TypeError``; pass ``None`` only when nothing else can
    move the robot. The standalone path composes against the SDK's body read
    and applies the same rate rule (stretching) and path rule (refusing), since
    it cannot split. Returns ``None``; refusals raise ``MotionRefused``.

    FRAME: every caller works body-relative, and the SDK head pose matrix is
    WORLD-frame, so the body yaw is added at dispatch (the 2026-08-07 safety
    fold: shipping the relative yaw as world dragged the head to world 0 after a
    160° body turn, a −160° neck demand against a ±65° capability).
    """
    if controller is not None and hasattr(controller, "goto_target"):
        from maxim.hardware.controller import HEAD_TRANSLATION_EXTRAS, MotionTarget

        extras = {
            key: require_finite(key, value) / 1000.0
            for key, value in zip(HEAD_TRANSLATION_EXTRAS, (x, y, z), strict=True)
            if require_finite(key, value) != 0.0
        }
        return controller.goto_target(
            MotionTarget(
                head_roll=math.radians(require_finite("roll", roll)),
                head_pitch=math.radians(require_finite("pitch", pitch)),
                head_yaw=math.radians(require_finite("yaw", yaw)),
                extras=extras,
                duration=floor_duration(duration),
            )
        )
    if motion_lock is _REQUIRED:
        raise TypeError("move_head without a controller requires motion_lock= (#1499 5.4)")
    import contextlib

    lock = motion_lock if motion_lock is not None else contextlib.nullcontext()
    with lock:
        check_standalone_head_motion(mini, x, y, z, roll, pitch, yaw, duration)
    return None


def check_standalone_head_motion(mini, x, y, z, roll, pitch, yaw, duration) -> None:
    """The standalone head dispatch: compose against the SDK's body read, rate-stretch, refuse a sweep, dispatch."""
    from maxim.exceptions import MotionRefused
    from maxim.hardware.reachy import motion_path as mp

    body_yaw_deg = _current_body_yaw_deg(mini)
    roll = clamp_finite("roll", roll, -_MAX_HEAD_ROLL_DEG, _MAX_HEAD_ROLL_DEG)
    pitch = clamp_finite("pitch", pitch, -_MAX_HEAD_PITCH_DEG, _MAX_HEAD_PITCH_DEG)
    yaw = clamp_finite("yaw", yaw, -_MAX_HEAD_REL_YAW_DEG, _MAX_HEAD_REL_YAW_DEG)
    x, y, z = (require_finite(name, v) for name, v in (("x", x), ("y", y), ("z", z)))
    # Translation intent is body-frame (x forward, y left of the BODY);
    # rotate into the world frame so a turned body doesn't reinterpret a
    # forward nudge as sideways. mm-scale — not a destruction-class axis.
    b = math.radians(body_yaw_deg)
    x_w = x * math.cos(b) - y * math.sin(b)
    y_w = x * math.sin(b) + y * math.cos(b)
    pose = head_pose_matrix(x_w, y_w, z, roll, pitch, yaw + body_yaw_deg)
    duration = floor_duration(duration)
    try:
        current = np.asarray(mini.get_current_head_pose(), dtype=float)
    except Exception:  # missing: the rate and path checks need a start, so this refuses below
        log_swallowed_exception()
        current = None
    if current is None or current.shape != (4, 4) or not np.all(np.isfinite(current)):
        raise MotionRefused("reference", "standalone move_head: the present head pose is unreadable")
    start = mp.Pose(current, b)
    target = mp.Pose(pose, b)
    ok, worst = mp.neck_path_ok(start, target)
    if not ok:
        raise MotionRefused("path", f"standalone move_head would sweep the neck to {worst:.0f}° mid-path")
    duration = mp.resolve_duration(duration, mp.move_delta_deg(start, target), "minimum_jerk", "at_least")
    # Last mile (#1496): every scalar handed to the SDK, the duration's
    # range, and the present pose (defence in depth).
    duration, _ = check_sdk_command(mini, duration=duration, head=pose)
    mini.goto_target(head=pose, duration=duration, body_yaw=None)


def move_antenna(
    mini,
    right: float | None = None,
    left: float | None = None,
    *,
    duration: float | None = 0.5,
    method: str = "minjerk",
    degrees: bool = True,
    relative: bool = False,
) -> None:
    """Move the antennas, always interpolated and bounded (#1496).

    ``duration=None`` takes the default; any explicit duration must lie in
    [0.1 s, 10 s] (an explicit 0 used to take the SDK's instant ``set_target``).
    Targets clamp to ±``ANTENNA_LIMIT_RAD``; NaN is refused. ``body_yaw=None``
    keeps the body still (the SDK's default of 0.0 would turn it to 0).
    """
    if right is None and left is None:
        raise ValueError("At least one of right or left must be provided.")

    duration_s = floor_duration(0.5) if duration is None else require_duration(duration)

    current_right, current_left = mini.get_present_antenna_joint_positions()

    target_right = current_right
    target_left = current_left

    if right is not None:
        right = require_finite("right", _to_rad(right, degrees=degrees))
        target_right = current_right + right if relative else right

    if left is not None:
        left = require_finite("left", _to_rad(left, degrees=degrees))
        target_left = current_left + left if relative else left

    target = [
        clamp_finite("antenna right", target_right, -ANTENNA_LIMIT_RAD, ANTENNA_LIMIT_RAD),
        clamp_finite("antenna left", target_left, -ANTENNA_LIMIT_RAD, ANTENNA_LIMIT_RAD),
    ]
    duration_s, sdk_method = check_sdk_command(mini, duration=duration_s, antennas=target, method=method)
    mini.goto_target(antennas=target, duration=duration_s, method=sdk_method, body_yaw=None)


def head_pose_matrix(x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
    # Convert units
    x, y, z = x / 1000, y / 1000, z / 1000
    roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))

    # Rotation matrices
    Rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)],
        ]
    )

    Ry = np.array(
        [
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)],
        ]
    )

    Rz = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T
