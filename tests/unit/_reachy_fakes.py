"""A shared fake Reachy Mini SDK handle and daemon for the motion tests (#1499; docs/plans/motion_path_model.md §7).

The fake models the daemon facts the motion-path design rests on (reachy-mini 1.8.3):

- F1: ``look_at_image(..., perform_movement=True)`` sends the body to 0 (``goto_target``'s ``body_yaw`` default).
- F2: a ``goto_target`` with ``body_yaw=None`` holds the body where it is.
- F3: a move can be dropped silently while the client still reports it complete (``drop_next``).
- F11: the canned ``wake_up``/``goto_sleep`` drive the body to 0.
- F12: the REST ``GET /api/state/full`` read, served from the modelled state, with its failure modes.

``linear_pose_interpolation`` is a verbatim copy of the SDK's (``reachy_mini/utils/interpolation.py``), because the
unit lane has no ``reachy`` extra. A contract test in the reachy extras cell pins the copy to the real function.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
import types
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

INIT_HEAD_POSE = np.eye(4)
SLEEP_HEAD_POSE = np.array(
    [
        [math.cos(math.radians(24)), 0.0, math.sin(math.radians(24)), -0.021],
        [0.0, 1.0, 0.0, 0.0],
        [-math.sin(math.radians(24)), 0.0, math.cos(math.radians(24)), -0.044],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def head_pose(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
    """Extrinsic "xyz" (Rz·Ry·Rx), the SDK's ``create_head_pose`` convention (F7), radians."""
    m = np.eye(4)
    m[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    m[:3, 3] = [x, y, z]
    return m


def linear_pose_interpolation(start_pose, target_pose, t: float):
    """Verbatim copy of reachy_mini 1.8.3 ``utils.interpolation.linear_pose_interpolation``."""
    rot_start = R.from_matrix(start_pose[:3, :3])
    rot_end = R.from_matrix(target_pose[:3, :3])
    q_rel = rot_start.inv() * rot_end
    rotvec_rel = q_rel.as_rotvec()
    rot_interp = (rot_start * R.from_rotvec(rotvec_rel * t)).as_matrix()
    pos_start = start_pose[:3, 3]
    pos_end = target_pose[:3, 3]
    pos_interp = pos_start + (pos_end - pos_start) * t
    interp_pose = np.eye(4)
    interp_pose[:3, :3] = rot_interp
    interp_pose[:3, 3] = pos_interp
    return interp_pose


def neck_twist_deg(world: np.ndarray, body: float) -> float:
    """The neck twist about the body's z axis (R5's measure), degrees, in (-180, 180]."""
    c, s = math.cos(-body), math.sin(-body)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    q = R.from_matrix(rz @ world[:3, :3]).as_quat()  # x, y, z, w
    tw = math.degrees(2.0 * math.atan2(q[2], q[3]))
    return (tw + 180.0) % 360.0 - 180.0


def worst_path_twist_deg(start_world, start_body, target_world, target_body, samples: int = 64) -> float:
    """The largest |neck twist| along the SDK's own path between two states (head geodesic, body joint line)."""
    worst = 0.0
    for s in np.linspace(0.0, 1.0, samples):
        w = linear_pose_interpolation(start_world, target_world, float(s))
        b = start_body + (target_body - start_body) * float(s)
        worst = max(worst, abs(neck_twist_deg(w, b)))
    return worst


class FakeDaemon:
    """The SDK handle and the daemon behind it, as one object. Every motion call is recorded with its start state."""

    def __init__(self, *, body: float = 0.0, head=None, antennas=(0.0, 0.0), control_mode: str = "enabled"):
        self.body = float(body)
        self.head = (head_pose(yaw=body) if head is None else np.array(head, dtype=float)).copy()
        self.antennas = list(antennas)
        self.control_mode = control_mode
        self.calls: list[dict[str, Any]] = []
        self.drop_next = 0  # F3: drop this many upcoming goto_target moves, reporting them complete
        self.timeout_on_goto: int | None = None  # raise TimeoutError on this goto_target index (0-based)
        self.enable_raises = False
        self.torque_delay_s = 0.0  # enable_motors takes effect this long after the call (fire-and-forget)
        self._torque_at: float | None = None
        self.rest_mode = "ok"  # ok | status500 | timeout | missing_body | nan_body | nested | head_m | unknown
        self.rest_calls = 0
        self.lock = threading.Lock()

    # ── recording ──────────────────────────────────────────────────────────
    def _record(self, name: str, **kw) -> None:
        self.calls.append({"name": name, "start_body": self.body, "start_head": self.head.copy(), **kw})

    def motion_calls(self) -> list[dict[str, Any]]:
        return [
            c
            for c in self.calls
            if c["name"] in {"goto_target", "set_target", "look_at_image_move", "wake_up", "goto_sleep"}
        ]

    def gotos(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["name"] == "goto_target"]

    # ── SDK motion ─────────────────────────────────────────────────────────
    def goto_target(self, head=None, antennas=None, duration=0.5, method="minimum_jerk", body_yaw=None, **_):
        index = len(self.gotos())
        self._record(
            "goto_target",
            head=None if head is None else np.array(head),
            body_yaw=body_yaw,
            antennas=antennas,
            duration=duration,
            method=method,
        )
        if self.timeout_on_goto is not None and index == self.timeout_on_goto:
            raise TimeoutError("task did not complete in time")
        if self.drop_next > 0:
            self.drop_next -= 1
            return  # F3: dropped, but the client reports it complete
        if body_yaw is not None:
            self.body = float(body_yaw)  # F2: None holds the body
        if head is not None:
            self.head = np.array(head, dtype=float)
        if antennas is not None:
            self.antennas = list(antennas)

    def set_target(self, **kw):
        self._record("set_target", **kw)

    def look_at_image(self, u, v, duration=1.0, perform_movement=True, **_):
        """The pose the head would take: the current head, turned by the pixel offset (centre = no change)."""
        yaw_off = -(float(u) - 320.0) / 640.0 * math.radians(60)
        pitch_off = (float(v) - 240.0) / 480.0 * math.radians(45)
        pose = self.head.copy()
        pose[:3, :3] = pose[:3, :3] @ R.from_euler("xyz", [0.0, pitch_off, yaw_off]).as_matrix()
        if perform_movement:
            self._record("look_at_image_move", u=u, v=v, duration=duration)
            self.body = 0.0  # F1
            self.head = pose
        return pose

    def wake_up(self):
        self._record("wake_up")
        self.body = 0.0  # F11
        self.head = INIT_HEAD_POSE.copy()

    def goto_sleep(self):
        self._record("goto_sleep")
        self.body = 0.0  # F11
        self.head = SLEEP_HEAD_POSE.copy()

    def enable_motors(self):
        self._record("enable_motors")
        if self.enable_raises:
            raise RuntimeError("enable_motors failed")
        if self.torque_delay_s <= 0:
            self.control_mode = "enabled"
        else:
            self._torque_at = time.monotonic() + self.torque_delay_s

    def disable_motors(self):
        self._record("disable_motors")
        self.control_mode = "disabled"

    def set_automatic_body_yaw(self, enabled):
        self._record("set_automatic_body_yaw", enabled=bool(enabled))

    # ── WS cache reads (observation; R2: no longer a dispatch reference) ─
    def get_current_head_pose(self):
        return self.head.copy()

    def get_current_joint_positions(self):
        return [self.body] + [0.0] * 6, list(self.antennas)

    def get_present_antenna_joint_positions(self):
        return tuple(self.antennas)

    # ── REST: GET /api/state/full (F12) ────────────────────────────────────
    def _control_mode_now(self) -> str:
        if self._torque_at is not None and time.monotonic() >= self._torque_at:
            self.control_mode = "enabled"
            self._torque_at = None
        return self.control_mode

    def state_full(self) -> dict[str, Any]:
        body: Any = self.body
        if self.rest_mode == "nan_body":
            body = float("nan")
        state: dict[str, Any] = {
            "control_mode": self._control_mode_now(),
            "head_pose": self.head.tolist(),
            "body_yaw": body,
            "antennas_position": list(self.antennas),
            "timestamp": "2026-10-29T00:00:00+00:00",
        }
        if self.rest_mode == "missing_body":
            state.pop("body_yaw")
        elif self.rest_mode == "nested":
            state["body"] = {"body_yaw": state.pop("body_yaw")}
        elif self.rest_mode == "head_m":
            state["head_pose"] = {"m": state["head_pose"]}
        elif self.rest_mode == "unknown":
            state = {"pose": "?"}
        return state

    def fetch_url(self, url: str, **_kw):
        from maxim.utils.http import Response

        if "/api/state/full" in url:
            self.rest_calls += 1
            if self.rest_mode == "timeout":
                raise TimeoutError("read timed out")
            if self.rest_mode == "status500":
                return Response(
                    status=500, headers={}, content=b"{}", elapsed_ms=1.0, endpoint="_external", request_id="t"
                )
            body = json.dumps(self.state_full()).encode()
            return Response(status=200, headers={}, content=body, elapsed_ms=1.0, endpoint="_external", request_id="t")
        if "/api/daemon/status" in url:
            return Response(
                status=200,
                headers={},
                content=b'{"state": "running"}',
                elapsed_ms=1.0,
                endpoint="_external",
                request_id="t",
            )
        raise AssertionError(f"unexpected URL in a motion test: {url}")


def install(monkeypatch, daemon: FakeDaemon) -> None:
    """Route the SDK utils and the daemon's REST read to *daemon* (CI has no ``reachy`` extra)."""
    import maxim.utils.http as maxim_http

    def create_head_pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, mm=False, degrees=True):
        if degrees:
            roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))
        return head_pose(roll=roll, pitch=pitch, yaw=yaw, x=x, y=y, z=z)

    pkg = sys.modules.get("reachy_mini") or types.ModuleType("reachy_mini")
    utils = types.ModuleType("reachy_mini.utils")
    utils.create_head_pose = create_head_pose
    interpolation = types.ModuleType("reachy_mini.utils.interpolation")
    interpolation.linear_pose_interpolation = linear_pose_interpolation
    utils.interpolation = interpolation
    monkeypatch.setattr(pkg, "utils", utils, raising=False)
    monkeypatch.setitem(sys.modules, "reachy_mini", pkg)
    monkeypatch.setitem(sys.modules, "reachy_mini.utils", utils)
    monkeypatch.setitem(sys.modules, "reachy_mini.utils.interpolation", interpolation)
    monkeypatch.setattr(maxim_http, "fetch_url", daemon.fetch_url)


def controller(daemon: FakeDaemon):
    """A connected ``ReachyMiniController`` over *daemon*, without the network connect."""
    from maxim.hardware.reachy.controller import ReachyMiniController

    ctl = ReachyMiniController.__new__(ReachyMiniController)
    ctl._mini = daemon
    ctl._motion_lock = threading.RLock()
    ctl._last_commanded = {}
    ctl._automatic_body_yaw = False
    ctl._divergence_last_warn = 0.0
    ctl._robot_name = "test_reachy"
    ctl._resolved_host = "127.0.0.1"
    ctl._connection_mode = "network"
    ctl.last_clamped_axes = ()
    ctl.is_connected = lambda: True  # type: ignore[method-assign]
    ctl._update_state = lambda **kw: None  # type: ignore[method-assign]
    return ctl
