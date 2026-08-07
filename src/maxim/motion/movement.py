from __future__ import annotations

import math
import json
import numpy as np
from pathlib import Path
from typing import Any

from maxim.utils.logging import warn

# Bundled seed data — shipped inside the package (src/maxim/_data/motion/).
_DATA_DIR = Path(__file__).resolve().parent.parent / "_data" / "motion"
_DEFAULT_ACTIONS_PATH = _DATA_DIR / "default_actions.json"
_DEFAULT_POSES_PATH = _DATA_DIR / "default_poses.json"
_DEFAULT_THRESHOLDS_PATH = _DATA_DIR / "movement_thresholds.json"


def _to_rad(value: float, *, degrees: bool) -> float:
    value = float(value)
    return math.radians(value) if degrees else value


def load_actions(path: Path | str = _DEFAULT_ACTIONS_PATH) -> dict[str, Any]:
    actions_path = Path(path)
    with actions_path.open("r", encoding="utf-8") as file:
        actions = json.load(file)

    if not isinstance(actions, dict):
        raise ValueError(f"Expected top-level JSON object in {actions_path}, got {type(actions).__name__}")

    return actions


def load_poses(path: Path | str = _DEFAULT_POSES_PATH) -> dict[str, list[float]]:
    """
    Load named head poses from JSON.

    Each pose can be either:
    - list: [x, y, z, roll, pitch, yaw] or [x, y, z, roll, pitch, yaw, duration]
    - dict: {"x":..,"y":..,"z":..,"roll":..,"pitch":..,"yaw":..,"duration":..}
    """
    poses_path = Path(path)
    default = {"centered": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]}

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

    parsed: dict[str, list[float]] = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not name.strip():
            continue

        vec: list[float] | None = None
        if isinstance(spec, (list, tuple)) and len(spec) >= 6:
            try:
                vec = [float(spec[i]) for i in range(6)]
                if len(spec) >= 7 and spec[6] is not None:
                    vec.append(float(spec[6]))
            except Exception:
                vec = None
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
            except Exception:
                vec = None

        if vec is not None:
            parsed[name.strip()] = vec

    if not parsed:
        return default

    parsed.setdefault("centered", default["centered"])
    return parsed


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
    """Best-effort body yaw (degrees) from the SDK joint vector; 0.0 on failure.

    Joint vector is ``[body_yaw, *stewart_legs]`` — the SDK's own fk() reads
    index 0 as body_yaw.
    """
    try:
        head_joints, _ = mini.get_current_joint_positions()
        if head_joints is not None and len(head_joints) >= 1:
            return math.degrees(float(head_joints[0]))
    except Exception:
        pass
    return 0.0


def move_head(mini, x, y, z, roll, pitch, yaw, duration, *, motion_lock=None):
    """Dispatch a head pose to the SDK. ``yaw`` is BODY-RELATIVE degrees.

    FRAME (2026-08-07 safety fold — this was the reincarnation of the
    head-frame trap one layer up): every Selfy-layer caller works in the
    body-relative frame (``sync_head_position`` computes ``yaw = world −
    body``), but the SDK head pose matrix is WORLD-frame. Pre-fold this
    function shipped the relative yaw AS world yaw with ``body_yaw=None``
    (retain) — correct only while the body sat at 0. After any SEM body
    turn (body at e.g. 160°), a gaze command of "relative ≈ 0" dragged the
    head to WORLD 0, demanding a head-relative angle of −160° against a
    ±65° neck capability: the motor-destruction failure class, and the
    same mechanism `turn_around`'s STEP 2 was just fixed for.

    Composition happens at DISPATCH time (this runs on the motor-queue
    thread), so the body yaw read is as fresh as it can be. Rotations are
    clamped to physical capability as belt-and-suspenders — Selfy.move()
    clamps the tighter style workspace upstream, but goto_pose and the
    workers.py IK-recovery recenter also land here.
    """
    import contextlib

    # When the caller can supply the controller's _motion_lock (Selfy.move
    # threads it through), the body-yaw read + compose + dispatch here is
    # serialized against controller-routed callers too — otherwise a
    # controller goto_target changing body yaw between our read and our
    # dispatch re-opens the TOCTOU one path over (executor-lens review
    # finding). Lock-less callers (workers.py recovery) degrade to a fresh
    # read with a small residual window.
    lock = motion_lock if motion_lock is not None else contextlib.nullcontext()
    with lock:
        body_yaw_deg = _current_body_yaw_deg(mini)
        roll = max(-_MAX_HEAD_ROLL_DEG, min(_MAX_HEAD_ROLL_DEG, float(roll)))
        pitch = max(-_MAX_HEAD_PITCH_DEG, min(_MAX_HEAD_PITCH_DEG, float(pitch)))
        yaw = max(-_MAX_HEAD_REL_YAW_DEG, min(_MAX_HEAD_REL_YAW_DEG, float(yaw)))
        # Translation intent is body-frame (x forward, y left of the BODY);
        # rotate into the world frame so a turned body doesn't reinterpret a
        # forward nudge as sideways. mm-scale — not a destruction-class axis.
        b = math.radians(body_yaw_deg)
        x_w = float(x) * math.cos(b) - float(y) * math.sin(b)
        y_w = float(x) * math.sin(b) + float(y) * math.cos(b)
        pose = head_pose_matrix(x_w, y_w, z, roll, pitch, yaw + body_yaw_deg)
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
    if right is None and left is None:
        raise ValueError("At least one of right or left must be provided.")

    if isinstance(method, str):
        method = method.strip().lower().replace("-", "_")
        method = {"min_jerk": "minjerk", "ease": "ease_in_out"}.get(method, method)

    current_right, current_left = mini.get_present_antenna_joint_positions()

    target_right = current_right
    target_left = current_left

    if right is not None:
        right = _to_rad(right, degrees=degrees)
        target_right = current_right + right if relative else right

    if left is not None:
        left = _to_rad(left, degrees=degrees)
        target_left = current_left + left if relative else left

    target = [target_right, target_left]

    if duration is None or float(duration) <= 0:
        mini.set_target(antennas=target, body_yaw=None)
    else:
        mini.goto_target(antennas=target, duration=float(duration), method=method, body_yaw=None)


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
