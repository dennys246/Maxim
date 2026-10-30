"""The Reachy Mini motion-path model: rate, neck path and the split (#1499).

The SDK interpolates the head on the WORLD SO(3) geodesic (the short way round,
``reachy_mini.utils.interpolation.linear_pose_interpolation``) while the body
yaw and the antennas move along joint-space lines. Whenever the body and the
neck together ask for more than about 180° of world yaw, the neck passes through
about 180° mid-move, although both endpoints are legal. This module describes
that interpolation so the controller can refuse, stretch or split a move before
any of it reaches a motor. docs/plans/motion_path_model.md, R3 to R5.

Frames: ``world = Rz(body) · R_rel`` (the SDK's ``create_head_pose`` is extrinsic
"xyz", so a world head yaw is the body yaw plus the neck's). The neck measure is
the TWIST of ``R_rel`` about the body's z axis, never an Euler difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from maxim.exceptions import MotionRefused
from maxim.hardware.controller import DurationPolicy, floor_duration, require_duration

# R4 (owner decision O2): average speed ≤ 180°/s, and each method's peak, by
# its peak factor, ≤ 340°/s. Peak factors are the methods' max velocity over
# their average (F8).
MAX_RATE_AVG_DEG_S = 180.0
MAX_RATE_PEAK_DEG_S = 340.0
_PEAK_FACTOR = {"minimum_jerk": 1.875, "ease_in_out": 2.0, "linear": 1.0}

# R5. σ = |Δbody + Δrel_yaw| above this always fails; below it the sampler still
# runs (F6: σ = 149.6° with roll and pitch moving sweeps the neck to 179.7°).
PATH_SIGMA_PREFILTER_DEG = 150.0
# A path passes when its worst |twist| is at most the endpoints' worst + the
# slack, and never more than the ceiling (far above the 65° neck plus the
# measured overshoot, far below the 180° sweep). Fold-made choices (§12 item 1).
NECK_PATH_SLACK_DEG = 10.0
NECK_PATH_CEILING_DEG = 90.0
NECK_PATH_SAMPLES = 32
# Body chunks: ⌈|Δbody| / 150°⌉, with the neck held (Executor-measured worst
# twist about 65° per stage).
BODY_CHUNK_MAX_DEG = 150.0


def rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rz(yaw)·Ry(pitch)·Rx(roll), the SDK's ``create_head_pose`` convention (F7)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def euler_xyz(m: np.ndarray) -> tuple[float, float, float]:
    """(roll, pitch, yaw) of a rotation, the controller's long-standing extraction."""
    yaw = math.atan2(m[1][0], m[0][0])
    pitch = math.atan2(-m[2][0], math.hypot(m[2][1], m[2][2]))
    roll = math.atan2(m[2][1], m[2][2])
    return roll, pitch, yaw


def relative_rotation(world: np.ndarray, body: float) -> np.ndarray:
    """R_rel = Rz(−body)·R_world (R3: a left multiply, never an Euler subtraction)."""
    return rz(-body) @ np.asarray(world, dtype=float)[:3, :3]


def neck_twist(world: np.ndarray, body: float) -> float:
    """The neck's twist about the body's z axis, radians in (−π, π] (R5's measure)."""
    m = relative_rotation(world, body)
    # Quaternion (w, z) of R_rel, then the twist about z: 2·atan2(z, w).
    w = math.sqrt(max(0.0, 1.0 + m[0][0] + m[1][1] + m[2][2])) / 2.0
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0][0] - m[1][1] + m[2][2])) / 2.0, m[1][0] - m[0][1])
    twist = 2.0 * math.atan2(z, w)
    return math.atan2(math.sin(twist), math.cos(twist))


def geodesic(a: np.ndarray, b: np.ndarray) -> float:
    """The rotation angle between two rotations, radians in [0, π]."""
    r = np.asarray(a, dtype=float)[:3, :3].T @ np.asarray(b, dtype=float)[:3, :3]
    cos_angle = (float(np.trace(r)) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, cos_angle)))  # motion-lint: measurement (float noise past ±1)


@dataclass(frozen=True, eq=False)
class Pose:
    """A full robot pose the dispatch moves between: world head matrix, body yaw, antennas.

    Internal to one dispatch; never persisted or sent over a wire (CC3 scope).
    """

    head: np.ndarray  # 4×4, world frame
    body: float
    antennas: tuple[float, float] | None = None


def neck_path_ok(start: Pose, target: Pose) -> tuple[bool, float]:
    """(passes, worst |twist| in degrees) along the SDK's own path from ``start`` to ``target``.

    The head follows ``linear_pose_interpolation`` (imported from the SDK, not
    re-implemented) and the body its joint-space line, at ``NECK_PATH_SAMPLES``
    points.
    """
    from reachy_mini.utils.interpolation import linear_pose_interpolation

    start_head = np.asarray(start.head, dtype=float)
    target_head = np.asarray(target.head, dtype=float)
    worst = 0.0
    for s in np.linspace(0.0, 1.0, NECK_PATH_SAMPLES):
        world = linear_pose_interpolation(start_head, target_head, float(s))
        body = start.body + (target.body - start.body) * float(s)
        worst = max(worst, abs(math.degrees(neck_twist(world, body))))
    endpoints = max(
        abs(math.degrees(neck_twist(start_head, start.body))), abs(math.degrees(neck_twist(target_head, target.body)))
    )
    return worst <= min(endpoints + NECK_PATH_SLACK_DEG, NECK_PATH_CEILING_DEG), worst


def move_delta_deg(start: Pose, target: Pose) -> float:
    """The largest per-axis change of a move, degrees (R4's Δ): body, the head geodesic in R_rel, each antenna."""
    deltas = [
        abs(math.degrees(target.body - start.body)),
        math.degrees(geodesic(relative_rotation(start.head, start.body), relative_rotation(target.head, target.body))),
    ]
    if start.antennas is not None and target.antennas is not None:
        deltas.extend(abs(math.degrees(b - a)) for a, b in zip(start.antennas, target.antennas, strict=True))
    return max(deltas)


def required_duration(delta_deg: float, method: str) -> float:
    """The shortest duration a move of ``delta_deg`` may take under the rate caps (R4, O2)."""
    return delta_deg * max(1.0 / MAX_RATE_AVG_DEG_S, _PEAK_FACTOR[method] / MAX_RATE_PEAK_DEG_S)


def resolve_duration(requested: float, delta_deg: float, method: str, policy: DurationPolicy) -> float:
    """The duration one dispatch ships with, or ``MotionRefused`` (``rate`` / ``duration``).

    ``"exact"``: below the required duration is refused. ``"at_least"``: lifted
    to it. Over 10 s is refused, never clamped (``require_duration``).
    """
    required = required_duration(delta_deg, method)
    if policy == "exact" and requested < required - 1e-9:
        raise MotionRefused(
            "rate",
            f"{delta_deg:.0f}° in {requested:.2f} s is faster than the rate cap; it needs at least {required:.2f} s",
        )
    return require_duration(max(requested, required))


def sigma_deg(start: Pose, target: Pose) -> float:
    """σ = |Δbody + Δrel_yaw|, degrees: the world-yaw swing R5's pre-filter bounds."""
    rel0 = euler_xyz(relative_rotation(start.head, start.body))[2]
    rel1 = euler_xyz(relative_rotation(target.head, target.body))[2]
    d_rel = math.atan2(math.sin(rel1 - rel0), math.cos(rel1 - rel0))
    return abs(math.degrees((target.body - start.body) + d_rel))


def compose_head(roll: float, pitch: float, rel_yaw: float, body: float, translation: Any = None) -> np.ndarray:
    """The world 4×4 head matrix for a body-relative head pose at ``body`` (translation in metres, world)."""
    m = np.eye(4)
    m[:3, :3] = rz(body) @ rot_xyz(roll, pitch, rel_yaw)
    if translation is not None:
        m[:3, 3] = np.asarray(translation, dtype=float)
    return m


def plan_chunks(start: Pose, target: Pose, *, translation_body_frame: Any = None) -> list[Pose]:
    """The chunk targets for one move, all checked by the sampler before any moves (R5).

    One chunk when the move passes as one dispatch. Otherwise the split order:
    body chunks with the neck held at its start, one relative-yaw chunk, one
    roll-and-pitch chunk (empty stages skipped). Raises ``MotionRefused("path")``
    when any planned chunk fails, so a split is never started that cannot finish.

    ``translation_body_frame``: a head translation held in the BODY frame (the
    lifecycle re-centre, 5.3), rotated with each chunk's body. None keeps the
    target's own translation.
    """
    if sigma_deg(start, target) <= PATH_SIGMA_PREFILTER_DEG and neck_path_ok(start, target)[0]:
        return [target]
    roll0, pitch0, rel0 = euler_xyz(relative_rotation(start.head, start.body))
    roll1, pitch1, rel1 = euler_xyz(relative_rotation(target.head, target.body))

    def at(roll: float, pitch: float, rel: float, body: float) -> np.ndarray:
        if translation_body_frame is not None:
            moved = rz(body) @ np.asarray(translation_body_frame, dtype=float)
        else:
            moved = np.asarray(target.head, dtype=float)[:3, 3]
        return compose_head(roll, pitch, rel, body, moved)

    chunks: list[Pose] = []
    d_body = target.body - start.body
    n_body = math.ceil(abs(math.degrees(d_body)) / BODY_CHUNK_MAX_DEG - 1e-9) if abs(d_body) > 1e-9 else 0
    for i in range(1, n_body + 1):
        body = start.body + d_body * i / n_body
        chunks.append(Pose(at(roll0, pitch0, rel0, body), body))  # antennas held until the final chunk
    if abs(math.atan2(math.sin(rel1 - rel0), math.cos(rel1 - rel0))) > 1e-9:
        chunks.append(Pose(at(roll0, pitch0, rel1, target.body), target.body))
    final = Pose(np.asarray(target.head, dtype=float), target.body, target.antennas)
    if not chunks or not _same_pose(chunks[-1], final):
        chunks.append(final)
    else:
        chunks[-1] = final  # the roll-and-pitch stage is empty: the last chunk is the target itself
    previous = start
    for k, chunk in enumerate(chunks, start=1):
        ok, worst = neck_path_ok(previous, chunk)
        if not ok:
            raise MotionRefused("path", f"split chunk {k}/{len(chunks)} would sweep the neck to {worst:.0f}° mid-path")
        previous = chunk
    return chunks


def _same_pose(a: Pose, b: Pose) -> bool:
    return (
        abs(a.body - b.body) < 1e-6
        and geodesic(a.head, b.head) < 1e-6
        and np.allclose(np.asarray(a.head)[:3, 3], np.asarray(b.head)[:3, 3], atol=1e-6)
    )


def share_durations(requested: float, start: Pose, chunks: list[Pose]) -> list[float]:
    """The requested duration divided across chunks by each chunk's Δ, each share floored (R4 under a split)."""
    deltas = []
    previous = start
    for chunk in chunks:
        deltas.append(move_delta_deg(previous, chunk))
        previous = chunk
    total = sum(deltas)
    if total <= 0.0:
        return [floor_duration(requested)] * len(chunks)
    shares = [floor_duration(requested * d / total) for d in deltas]
    shortfall = requested - sum(shares)
    if shortfall > 0.0:
        shares[-1] = floor_duration(shares[-1] + shortfall)  # float rounding never lets a floor fall short
    return shares


# ── R1: the fresh daemon readback (GET /api/state/full) ───────────────────────


def _finite_or_refuse(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise MotionRefused("reference", f"fresh readback {name} is not a number: {value!r}") from e
    if not math.isfinite(number):
        raise MotionRefused("reference", f"fresh readback {name} is not finite")
    return number


def _head_matrix(raw: Any) -> np.ndarray:
    """The world head matrix from the shapes the daemon has served (F12; delta D5).

    ``{"m": [16 floats]}`` (``Matrix4x4Pose``, row-major), ``{"m": 4×4}``, a bare
    4×4, or an ``XYZRPYPose`` dict (``x y z roll pitch yaw``). Anything else is missing.
    """
    if isinstance(raw, dict) and "m" in raw:
        raw = raw["m"]
    elif isinstance(raw, dict) and {"roll", "pitch", "yaw"} <= raw.keys():
        angles = [_finite_or_refuse(f"head {k}", raw[k]) for k in ("roll", "pitch", "yaw")]
        xyz = [_finite_or_refuse(f"head {k}", raw.get(k, 0.0)) for k in ("x", "y", "z")]
        return compose_head(angles[0], angles[1], angles[2], 0.0, xyz)
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise MotionRefused("reference", f"fresh readback head_pose has an unknown shape: {e}") from e
    if arr.size != 16:
        raise MotionRefused("reference", f"fresh readback head_pose has {arr.size} entries, not 16")
    arr = arr.reshape(4, 4)
    if not np.all(np.isfinite(arr)):
        raise MotionRefused("reference", "fresh readback head_pose is not finite")
    return arr


def _find_body_yaw(data: dict[str, Any]) -> Any:
    """``body_yaw`` at the top level or one level down (the shape varies by version, F12)."""
    if "body_yaw" in data:
        return data["body_yaw"]
    for value in data.values():
        if isinstance(value, dict) and "body_yaw" in value:
            return value["body_yaw"]
    raise MotionRefused("reference", "fresh readback has no body_yaw")


def parse_full_state(data: Any) -> tuple[Pose, str]:
    """(pose, control_mode) from a ``GET /api/state/full`` body, or ``MotionRefused("reference")``."""
    if not isinstance(data, dict):
        raise MotionRefused("reference", f"fresh readback is not an object: {type(data).__name__}")
    if "head_pose" not in data or data["head_pose"] is None:
        raise MotionRefused("reference", "fresh readback has no head_pose")
    head = _head_matrix(data["head_pose"])
    body = _finite_or_refuse("body_yaw", _find_body_yaw(data))
    antennas = None
    raw_antennas = data.get("antennas_position")
    if isinstance(raw_antennas, (list, tuple)) and len(raw_antennas) == 2:
        antennas = (
            _finite_or_refuse("antenna", raw_antennas[0]),
            _finite_or_refuse("antenna", raw_antennas[1]),
        )
    return Pose(head=head, body=body, antennas=antennas), str(data.get("control_mode"))
