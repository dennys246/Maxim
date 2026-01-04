from __future__ import annotations

import math, json
import numpy as np
from pathlib import Path
from typing import Any

from maxim.utils.logging import warn

_DEFAULT_ACTIONS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "motion" / "default_actions.json"
)

_DEFAULT_POSES_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "motion" / "default_poses.json"
)

def _to_rad(value: float, *, degrees: bool) -> float:
    value = float(value)
    return math.radians(value) if degrees else value

def load_actions(path: Path | str = _DEFAULT_ACTIONS_PATH) -> dict[str, Any]:
    
    actions_path = Path(path)
    with actions_path.open("r", encoding="utf-8") as file:
        actions = json.load(file)

    if not isinstance(actions, dict):
        raise ValueError(
            f"Expected top-level JSON object in {actions_path}, got {type(actions).__name__}"
        )

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

def move_head(mini, x, y, z, roll, pitch, yaw, duration):
    pose = head_pose_matrix(x, y, z, roll, pitch, yaw)
    mini.goto_target(head=pose, duration=duration, body_yaw=None)

def move_antenna(
    mini,
    right: float | None = None,
    left: float | None = None,
    *,
    duration: float | None = 0.5,
    method: str = "minjerk",
    degrees: bool = True,
    relative: bool = False) -> None:
    
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
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)],
    ])

    Ry = np.array([
        [ math.cos(pitch), 0, math.sin(pitch)],
        [ 0,               1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)],
    ])

    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0,              0,             1],
    ])

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T
