"""Reachy Mini controller implementation.

Wraps the Reachy Mini SDK to provide a RobotController interface.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import socket
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from maxim.hardware.capabilities import (
    REACHY_MINI_CAPABILITIES,
    RobotConnectionState,
)
from maxim.exceptions import InvalidConfigError, MotionRefused
from maxim.hardware.controller import (
    HEAD_TRANSLATION_EXTRAS,
    MotionOutcome,
    MotionTarget,
    PixelTarget,
    RobotController,
    clamp_finite,
    normalize_method,
    require_duration,
    require_finite,
)
from maxim.hardware.reachy import motion_path as mp
from maxim.utils.logging import log_swallowed_exception
from maxim.hardware.reachy.streams import ReachyAudioStream, ReachyVideoStream
from maxim.hardware.streams import AudioStream, VideoStream

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ── Reachy last-mile checks (#1496) ───────────────────────────────────────────
# One copy, used by every SDK motion call site: the controller dispatch,
# look_at_pixel, motion/movement.py's primitives, Selfy's SDK look_at wrapper
# and the allow-listed live harness. docs/plans/motion_finite_1496.md.

# Provisional: neither the SDK nor the daemon declares an antenna limit. To be
# replaced by the hand-measured range (requested from pulse 2026-10-26).
ANTENNA_LIMIT_RAD = math.radians(90.0)

# ── The motion-path model (#1499; docs/plans/motion_path_model.md) ────────────
# The math (rate, neck path, split, the REST parser) is in motion_path.py; these
# are the controller's own bounds.

# goto_target and look_at_pixel wait at most this long for _motion_lock, then
# return "motor busy" (delta N5). It bounds the WAIT only: a tool whose move
# starts at 24 s still runs past per_tool_timeout (30 s), and its executor
# abandons the result while the robot finishes the move. Every caller is
# bounded, the motor thread included (it logs and drops the command). The
# lifecycle (wake, sleep) waits without a bound; the stop fence bounds it.
MOTION_LOCK_WAIT_S = 25.0
# The fresh readback's overall deadline (R1). Provisional; measured on the rig
# before release (§12 item 3). Tunnel: an auto-tunnel or a manual one
# (connection_mode localhost_only). The reads between chunks and after the last
# chunk get one retry (owner, 2026-10-29); the first read of a move never does.
FRESH_READ_DEADLINE_S = 0.1
FRESH_READ_DEADLINE_TUNNEL_S = 0.3
_STATE_FULL_QUERY = (
    "use_pose_matrix=true&with_head_pose=true&with_body_yaw=true&with_antenna_positions=true&with_control_mode=true"
)
# A retained axis whose readback is this far from the last command is refused
# and forgotten (decision 3), and the per-chunk re-check and the final-target
# check use the same tolerance. Provisional 5°: end-of-move lag plus a margin.
_STASH_DIVERGENCE_REFUSE_RAD = math.radians(5.0)
# A readback body or neck beyond its limit by more than this is impossible (R2).
_IMPOSSIBLE_MARGIN_RAD = math.radians(5.0)
# After an SDK timeout: settled when within 2° of the timed-out chunk, or when
# stable within 0.5° for a full second (the move ended somewhere else).
_SETTLED_NEAR_RAD = math.radians(2.0)
_SETTLED_STABLE_RAD = math.radians(0.5)
_SETTLED_STABLE_S = 1.0

# Lifecycle motion (5.3, O8). Maxim composes the SDK's wake_up / goto_sleep
# steps itself (reachy_mini.py 1.8.3), each one a clamped controller dispatch:
# the SDK's own sleep sends the antennas to ±3.05 rad, past any measured range.
INIT_HEAD_POSE = np.eye(4)
SLEEP_HEAD_POSE = np.array(
    [
        [0.911, 0.004, 0.413, -0.021],
        [-0.004, 1.0, -0.001, 0.001],
        [-0.413, -0.001, 0.911, -0.044],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
INIT_ANTENNAS_RAD = (-0.1745, 0.1745)
SLEEP_ANTENNAS_RAD = (-3.05, 3.05)  # clamped to ±ANTENNA_LIMIT_RAD before dispatch
_LIFECYCLE_DEADBAND_RAD = math.radians(2.0)
_LIFECYCLE_RECENTRE_FLOOR_S = 2.0
_SLEEP_VIA_INIT_RAD = 0.2  # pose-space stand-in for the SDK's 0.2 joint-space test
_TORQUE_ON_WAIT_S = 1.0
# After an SDK timeout, sleep waits this long for the robot to settle before it
# refuses; wake refuses at once (owner, 2026-10-29).
_SLEEP_SETTLE_WAIT_S = 2.0


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# What every dispatch returns while the stop latch is set (goto_sleep / disconnect until wake_up / connect).
_STOPPED = MotionOutcome(ok=False, kind="cancelled", reason="stopped: the robot is asleep or stopping; wake it first")

# A lifecycle step: (world head pose, antennas or None, duration, sound) — a move when the head is set, a sound when
# it is None.
_LifecycleStep = tuple[Any, "tuple[float, float] | None", float, "str | None"]


# Canonical interpolation name (hardware.controller.normalize_method) -> the
# SDK's InterpolationTechnique spelling. No default: an unknown name is refused,
# and ``cartoon`` is absent because it overshoots its target past any clamp.
_SDK_METHODS = {
    "minimum_jerk": "minjerk",
    "linear": "linear",
    "ease_in_out": "ease_in_out",
}


def sdk_method(method: Any) -> str:
    """The Reachy SDK spelling of an interpolation method, or ``MotionRefused(kind="method")``."""
    return _SDK_METHODS[normalize_method(method)]


def _numeric_or_missing(read: Any) -> np.ndarray | None:
    """A readback as an object array (entries checked one by one), or None when it is missing."""
    if read is None:
        return None
    try:
        arr = np.asarray(read, dtype=object)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return arr


def require_present_pose_finite(mini: Any) -> None:
    """Refuse when a RETURNED present pose holds a non-finite value (defence in depth).

    Over the reachy-mini WS transport this cannot fire: the daemon serialises a
    non-finite pose as ``null`` and the client drops the message, keeping the last
    finite pose. It guards fakes, sims and future transports; the hardware signal
    (a stale cache) is #1499's. A read that raises, returns None, has the wrong
    shape or will not convert counts as MISSING and never refuses; a ``None``
    inside a returned matrix or vector is refused.
    """
    reads: list[tuple[str, Any, tuple[int, ...] | None]] = []
    try:
        reads.append(("head pose", mini.get_current_head_pose(), (4, 4)))
    except Exception:  # missing, not refused (#1499 owns missing readbacks)
        log_swallowed_exception()
    try:
        head_joints, antenna_joints = mini.get_current_joint_positions()
        reads.append(("head joints", head_joints, None))
        reads.append(("antenna joints", antenna_joints, None))
    except Exception:  # missing, not refused
        log_swallowed_exception()
    for name, read, shape in reads:
        arr = _numeric_or_missing(read)
        if arr is None or (shape is not None and arr.shape != shape) or (shape is None and arr.ndim != 1):
            continue
        for entry in arr.ravel():
            if entry is None:
                raise MotionRefused("non_finite", f"present {name} contains None")
            try:
                number = float(entry)
            except (TypeError, ValueError):
                break  # a non-numeric object (a test double) is missing, not refused
            if not math.isfinite(number):
                raise MotionRefused("non_finite", f"present {name} is not finite")


def check_sdk_command(
    mini: Any,
    *,
    duration: Any,
    head: Any = None,
    body_yaw: Any = None,
    antennas: Any = None,
    method: Any = "minimum_jerk",
) -> tuple[float, str]:
    """The last mile before a Reachy SDK motion call: refuse, or return ``(duration, sdk_method)``.

    Every scalar handed to the SDK must be finite (the 16 head-matrix entries,
    body yaw, antennas), the duration must lie in [0.1 s, 10 s] (0 or NaN takes
    the SDK's instant set_target branch), the method must be known, and the
    present pose must not be non-finite (defence in depth). Raises ``MotionRefused``.
    """
    if body_yaw is not None:
        require_finite("body_yaw", body_yaw)
    for name, values in (("head pose entry", head), ("antenna", antennas)):
        if values is not None:
            try:
                entries = np.asarray(values, dtype=np.float64).ravel()
            except (TypeError, ValueError) as e:
                raise MotionRefused("invalid", f"{name} values are not numbers: {e}") from e
            for entry in entries:
                require_finite(name, entry)
    checked = require_duration(duration), sdk_method(method)
    require_present_pose_finite(mini)
    return checked


class ReachyMiniController(RobotController):
    """Controller for Reachy Mini robot.

    Wraps the reachy-mini SDK to provide hardware-agnostic
    robot control through the RobotController interface.
    """

    def __init__(
        self,
        robot_id: str | None = None,
        *,
        robot_name: str = "reachy_mini",
        media_backend: str = "default",
        connection_mode: str = "network",
        host: str | None = None,
        tunnel: bool = False,
        ssh_user: str = "pollen",
        ssh_port: int = 22,
        automatic_body_yaw: bool = False,
    ) -> None:
        """Initialize Reachy Mini controller.

        All keyword args below are populated from ``robots.yaml``'s per-robot
        ``config:`` block (the registry splats it as ``**config``).

        Args:
            robot_id: Unique identifier for this robot (defaults to robot_name).
            robot_name: mDNS name of the robot (default: "reachy_mini").
            media_backend: Media backend to use (default: "default").
            connection_mode: SDK connection mode (WS era, SDK >= 1.5) —
                ``"network"`` (direct WebSocket to ``ws://<host>:8000/ws/sdk``,
                default), ``"localhost_only"`` (localhost:8000 — onboard or
                through a tunnel), or ``"auto"`` (localhost first, then host).
            host: Robot IP or hostname, passed straight into
                ``ReachyMini(host=...)``. Prefer the IP: ``.local`` resolution
                from Python is unreliable on macOS (getaddrinfo/IPv6-first),
                even when curl resolves it. When unset, the controller
                resolves ``<robot-name>.local`` via IPv4-only
                ``socket.gethostbyname`` — which sidesteps the IPv6-first
                flake — and passes the RESOLVED IP to the SDK. (TCC-blocked
                mDNS still requires an explicit ``host:``.)
            tunnel: DEPRECATED for wireless robots — the ``--wireless-version``
                daemon binds 0.0.0.0:8000, no tunnel needed. Still useful for
                loopback-bound daemons (Lite / non-wireless, Pollen PR #1205):
                auto-starts ``ssh -N -L 8000:127.0.0.1:8000 <ssh_user>@<host>``
                and forces ``connection_mode="localhost_only"``. Needs
                key-based SSH (BatchMode); else start the tunnel manually and
                set ``tunnel: false, connection_mode: localhost_only``.
                (Zenoh-era :7447 tunnels are obsolete — see
                docs/embodiment/reachy_mini/README.md, transport pivot.)
            ssh_user: SSH user for the tunnel (default "pollen").
            ssh_port: SSH port for the tunnel (default 22).
            automatic_body_yaw: Must be ``False``; ``True`` raises
                ``InvalidConfigError`` (#1499 O6). With it on, the daemon
                rotates the body behind the runtime (observed 2026-08-03:
                −25° of uncommanded body rotation put every focus_on_sound
                aim off by up to that much with a perfect sensor), and its
                IK moves the body itself, so no readback-based check can hold.
                The parameter stays so that a robots.yaml
                ``automatic_body_yaw: true`` is refused rather than silently
                dropped by the registry's signature filter.

        Raises:
            InvalidConfigError: ``automatic_body_yaw`` is true.
        """
        if automatic_body_yaw:
            raise InvalidConfigError(
                "automatic_body_yaw: true is not supported: the daemon would rotate the body behind Maxim's "
                "motion checks (#1499). Remove it from robots.yaml."
            )
        super().__init__(robot_id or robot_name)
        self._robot_name = robot_name
        self._media_backend = media_backend
        self._connection_mode = connection_mode
        self._host = host
        self._tunnel = tunnel
        self._ssh_user = ssh_user
        self._ssh_port = ssh_port
        self._automatic_body_yaw = False
        self._tunnel_proc: subprocess.Popen | None = None
        self._mini: Any = None  # ReachyMini SDK instance
        self._video_stream: ReachyVideoStream | None = None
        self._audio_stream: ReachyAudioStream | None = None
        # The host connect() actually resolved + spoke to (mDNS-resolved IP,
        # explicit host, or 127.0.0.1 under tunnel/localhost_only). connect()
        # previously resolved this as a local and dropped it; the DoA REST
        # reader needs it to be self-sufficient (Stage 1b).
        self._resolved_host: str | None = None
        # Serializes goto_target's read→compose→dispatch. Two overlapping
        # callers (e.g. focus_on_sound + a SEM body turn) each read
        # get_current_pose() and compose a head matrix against it; without
        # the lock the second composes against a body yaw the first is
        # already changing (TOCTOU on live kinematic state).
        self._motion_lock = threading.RLock()
        # Honesty surface for the clamps: axes clamped by the MOST RECENT
        # goto_target dispatch. Callers that report pose outcomes to the
        # LLM (MoveTool) read this instead of echoing the commanded value —
        # a silently-clamped "success" is the faced_sound=True dishonesty
        # class PR #459 fixed.
        self.last_clamped_axes: tuple[str, ...] = ()
        # Last COMMANDED value per axis, as dispatched (post-clamp). Keys:
        # "head_roll" / "head_pitch" / "head_rel_yaw" (body-relative) /
        # "body_yaw". goto_target fills UNSPECIFIED head axes from here, never
        # from the live readback (the F1 ratchet, see _compose). The body
        # stash is only checked for divergence (#1499 R1). Mutated only
        # under _motion_lock; cleared on disconnect, and per axis when it
        # diverges, so the next fill-in re-seeds from the readback once.
        self._last_commanded: dict[str, float] = {}
        # Throttle for the achieved-vs-commanded divergence WARNING.
        self._divergence_last_warn: float = 0.0
        self._init_motion_state()

    def _init_motion_state(self) -> None:
        """The #1499 dispatch state (also called by test fakes that build the controller without connecting)."""
        # The abort generation (5.1 stop fence): _stop() (sleep, disconnect) and
        # request_abort() increment it, under _abort_lock.
        self._abort_lock = threading.Lock()
        self._abort_gen = 0
        # Set after an SDK timeout: (timed-out chunk, stability anchor, anchor time).
        self._unsettled: tuple[mp.Pose, mp.Pose | None, float | None] | None = None
        # True once set_automatic_body_yaw(False) has been asserted on this handle (O6).
        self._auto_yaw_asserted = False
        # The stop latch (owner, 2026-10-29): set by goto_sleep and disconnect.
        # While set, goto_target and look_at_pixel refuse, so a head or body
        # command that starts after a stop never moves the robot, and the robot
        # reports is_awake=False. Only a wake that SUCCEEDS clears it; connect
        # clears it only when disconnect, not a sleep, set it (_stop_by_sleep).
        self._stop_latched = False
        self._stop_by_sleep = False

    @property
    def robot_type(self) -> str:
        """Get the robot type identifier."""
        return "reachy_mini"

    @property
    def mini(self) -> Any | None:
        """Get the underlying ReachyMini SDK instance.

        Provided for backward compatibility during migration.
        Prefer using the abstracted methods.
        """
        return self._mini

    # ─────────────────────────────────────────────────────────────────────────
    # Connection Management
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_mdns(self, timeout: float = 5.0) -> str | None:
        """Pre-resolve the robot's mDNS hostname to an IP address.

        Args:
            timeout: DNS resolution timeout in seconds.

        Returns:
            Resolved IP address string, or None if resolution failed.
        """
        mdns_name = self._robot_name.replace("_", "-") + ".local"
        try:
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout)
            try:
                ip = socket.gethostbyname(mdns_name)
                logger.info("mDNS resolved %s -> %s", mdns_name, ip)
                return ip
            finally:
                socket.setdefaulttimeout(old_timeout)
        except socket.gaierror:
            logger.warning("mDNS resolution failed for %s", mdns_name)
            return None

    @staticmethod
    def _port_open(host: str, port: int, timeout: float = 3.0) -> bool:
        """Single TCP reachability probe (fast-fail replacement for the mDNS gate)."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    @staticmethod
    def _installed_sdk_version() -> tuple[int, int] | None:
        """(major, minor) of the installed reachy-mini SDK; None if unknown.

        Uses importlib.metadata — pre-1.5 SDKs have no ``__version__``
        attribute, so package metadata is the only era probe that works
        across the pivot.
        """
        try:
            from importlib.metadata import version

            parts = version("reachy-mini").split(".")
            return (int(parts[0]), int(parts[1]))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _daemon_status(host: str, port: int = 8000, timeout: float = 3.0) -> dict | None:
        """Best-effort ``GET /api/daemon/status`` (WS-era daemon). None on failure."""
        from maxim.utils import http as maxim_http

        try:
            resp = maxim_http.fetch_url(f"http://{host}:{port}/api/daemon/status", timeout=timeout)
            return resp.json() if hasattr(resp, "json") else None
        except Exception:  # noqa: BLE001
            return None

    def _wait_local_port(self, port: int, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._port_open("127.0.0.1", port, timeout=1.0):
                return True
            time.sleep(0.3)
        return False

    def _start_ssh_tunnel(self, timeout: float = 10.0) -> bool:
        """Auto-start ``ssh -N -L 8000:127.0.0.1:8000 user@host`` (key-based / BatchMode).

        WS-era (SDK >= 1.5): the daemon's single control port is 8000. Only
        needed for loopback-bound daemons; wireless daemons bind 0.0.0.0.
        """
        cmd = [
            "ssh",
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=15",
            "-p",
            str(self._ssh_port),
            "-L",
            "8000:127.0.0.1:8000",
            f"{self._ssh_user}@{self._host}",
        ]
        logger.info("Starting SSH tunnel: %s", " ".join(cmd))
        try:
            self._tunnel_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            logger.error("ssh not found — cannot auto-tunnel")
            return False
        if self._wait_local_port(8000, timeout=timeout):
            logger.info("SSH tunnel up: localhost:8000 -> %s:8000", self._host)
            return True
        logger.error(
            "SSH tunnel to %s did not open localhost:8000. Auto-tunnel needs key-based SSH "
            "(BatchMode) — run `ssh-copy-id %s@%s` — OR start it manually and set "
            "tunnel=false, connection_mode=localhost_only.",
            self._host,
            self._ssh_user,
            self._host,
        )
        self._stop_ssh_tunnel()
        return False

    def _stop_ssh_tunnel(self) -> None:
        if self._tunnel_proc is None:
            return
        try:
            self._tunnel_proc.terminate()
            try:
                self._tunnel_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._tunnel_proc.kill()
        except Exception as e:  # noqa: BLE001
            logger.debug("Error stopping SSH tunnel: %s", e)
        finally:
            self._tunnel_proc = None

    def connect(self, timeout: float = 30.0) -> bool:
        """Connect to the Reachy Mini.

        Args:
            timeout: Connection timeout in seconds.

        Returns:
            True if connection successful.
        """
        self._update_state(connection_state=RobotConnectionState.CONNECTING)

        try:
            from reachy_mini import ReachyMini

            # Era gate: client and daemon must both be post-v1.5.0 (zenoh was
            # removed there; this controller speaks the WS transport). A
            # pre-pivot SDK would fail with a cryptic TypeError on host= —
            # fail loud with the fix instead.
            sdk_version = self._installed_sdk_version()
            if sdk_version is not None and sdk_version < (1, 5):
                logger.error(
                    "reachy-mini SDK %s is pre-v1.5.0 (zenoh era) — this controller requires the "
                    "WebSocket transport. Fix: pip install 'reachy-mini[gstreamer]>=1.8.3,<2.0' "
                    "(and check you're running the right interpreter/venv).",
                    ".".join(str(x) for x in sdk_version),
                )
                self._update_state(
                    connection_state=RobotConnectionState.ERROR,
                    error_message="reachy-mini SDK is pre-v1.5.0 (zenoh era); upgrade to >=1.8.3",
                )
                return False

            effective_mode = self._connection_mode

            # Optional SSH tunnel: localhost:8000 -> robot:8000. Only needed for
            # loopback-bound daemons (Lite); wireless daemons bind 0.0.0.0:8000.
            if self._tunnel:
                if not self._host:
                    logger.error("tunnel=True requires host= (robot IP / SSH target)")
                    self._update_state(connection_state=RobotConnectionState.DISCONNECTED)
                    return False
                if not self._start_ssh_tunnel(timeout=min(timeout, 10.0)):
                    self._update_state(connection_state=RobotConnectionState.DISCONNECTED)
                    return False
                effective_mode = "localhost_only"

            # Reachability pre-check (fast-fail so we don't wait on the SDK).
            # WS era (SDK >= 1.5): the single control port is the daemon's
            # FastAPI :8000 (WebSocket /ws/sdk rides it). A resolvable name /
            # alive host is NOT enough — the daemon can be down. When host is
            # unset, resolve <robot-name>.local via the OS FIRST and hand the
            # RESOLVED IP to the SDK (python's own .local resolution is flaky
            # on macOS even when the OS resolver works).
            if effective_mode == "localhost_only":
                probe_host: str | None = "127.0.0.1"
            elif self._host is not None:
                probe_host = self._host
            else:  # no explicit host: resolve mDNS -> probe + connect via the IP
                probe_host = self._resolve_mdns(timeout=min(timeout, 5.0))
                if probe_host is None:
                    logger.warning(
                        "Could not resolve %s via mDNS — robot not reachable, skipping SDK connection "
                        "(set host: <ip> in robots.yaml; on macOS grant Local Network permission)",
                        self._robot_name,
                    )
                    self._update_state(connection_state=RobotConnectionState.DISCONNECTED)
                    return False
            if probe_host and not self._port_open(probe_host, 8000, timeout=min(timeout, 5.0)):
                logger.warning(
                    "daemon :8000 unreachable at %s — skipping SDK connection (daemon down, "
                    "wrong host, or on macOS grant Local Network permission)",
                    probe_host,
                )
                self._stop_ssh_tunnel()
                self._update_state(connection_state=RobotConnectionState.DISCONNECTED)
                return False
            # Best-effort readiness + version visibility (never fatal).
            status = self._daemon_status(probe_host) if probe_host else None
            if status is not None:
                logger.info(
                    "Reachy daemon at %s: version=%s state=%s",
                    probe_host,
                    status.get("version", "?"),
                    status.get("state", "?"),
                )

            logger.info("Connecting to Reachy Mini: %s (mode=%s)", self._robot_name, effective_mode)

            connect_host = "127.0.0.1" if effective_mode == "localhost_only" else probe_host
            self._resolved_host = connect_host
            self._mini = ReachyMini(
                robot_name=self._robot_name,
                host=connect_host,
                port=8000,
                connection_mode=effective_mode,
                media_backend=self._media_backend,
                timeout=timeout,
                # Maxim owns the yaw axis (head-frame invariant, O6): stop the
                # daemon from rotating the body behind the runtime.
                automatic_body_yaw=False,
            )
            self._assert_auto_yaw_off()
            self._probe_state_shape()
            self._clear_latch_after_connect()

            # Create stream wrappers — but only for devices that exist.
            # Under media_backend="no_media" the SDK's media manager has
            # camera=None / audio=None; wrapping it anyway made
            # get_video_stream()/get_audio_stream() return live-looking
            # objects for absent hardware, defeating the stream-surface
            # contract derive_media_capabilities relies on ("a connected
            # controller whose getter returns None has positively declared
            # the device absent"). Mirror its semantics: downgrade only on
            # positive evidence; an un-introspectable media shape keeps
            # the wrapper.
            _media = getattr(self._mini, "media", None)
            _has_cam = not (hasattr(_media, "camera") and _media.camera is None) if _media is not None else True
            _has_aud = not (hasattr(_media, "audio") and _media.audio is None) if _media is not None else True
            self._video_stream = (
                ReachyVideoStream(
                    self._mini,
                    resolution=(640, 480),  # Will be updated after recording starts
                )
                if _has_cam
                else None
            )
            self._audio_stream = ReachyAudioStream(self._mini) if _has_aud else None
            if not _has_cam or not _has_aud:
                logger.info(
                    "Media streams gated by backend '%s': video=%s audio=%s",
                    self._media_backend,
                    _has_cam,
                    _has_aud,
                )

            # Build capabilities
            self._capabilities = dataclasses.replace(
                REACHY_MINI_CAPABILITIES,
                robot_id=self._robot_id,
            )

            self._update_state(
                connection_state=RobotConnectionState.CONNECTED,
                last_heartbeat=time.time(),
            )

            logger.info("Connected to Reachy Mini: %s", self._robot_name)
            return True

        except ImportError:
            logger.error("reachy-mini SDK not installed")
            self._update_state(
                connection_state=RobotConnectionState.ERROR,
                error_message="reachy-mini SDK not installed",
            )
            return False

        except Exception as e:
            logger.error("Failed to connect to Reachy Mini: %s", e)
            self._stop_ssh_tunnel()
            self._update_state(
                connection_state=RobotConnectionState.ERROR,
                error_message=str(e),
            )
            return False

    def disconnect(self) -> None:
        """Disconnect from the Reachy Mini."""
        # Latch (nothing new starts; connect clears it unless a sleep set it) and the stop
        # fence (5.1: a split in progress ends at its next chunk), as one step.
        self._stop(by_sleep=False)
        self._unsettled = None  # a reconnect starts from a fresh read (§12 item 18)
        self._auto_yaw_asserted = False
        self._stop_ssh_tunnel()
        if self._mini is None:
            return

        try:
            # Stop streams
            if self._video_stream:
                self._video_stream.stop()
            if self._audio_stream:
                self._audio_stream.stop()

            # SDK doesn't have explicit disconnect, just release reference
            self._mini = None
            self._video_stream = None
            self._audio_stream = None
            self._resolved_host = None

        except Exception as e:
            logger.error("Error during disconnect: %s", e)

        finally:
            # A reconnect finds the robot wherever it is now — stale
            # commanded values from the previous connection must not
            # masquerade as its pose. In the finally block so a raising
            # stream stop cannot skip the clear (two-lens fold).
            self._forget_commanded()
            self._update_state(
                connection_state=RobotConnectionState.DISCONNECTED,
                is_awake=False,
                is_recording=False,
            )

        logger.info("Disconnected from Reachy Mini: %s", self._robot_name)

    # ─────────────────────────────────────────────────────────────────────────
    # Motion Control
    # ─────────────────────────────────────────────────────────────────────────

    # MotionTarget.method is the SDK-AGNOSTIC name (documented "minimum_jerk"). The
    # Reachy SDK's InterpolationTechnique enum spells it "minjerk" and REJECTS
    # anything else with a pydantic validation error — so passing target.method
    # straight through crashed EVERY controller-driven motion on SDK >= 1.5
    # (caught 2026-07-17 by the #397 hardware smoke, before which nothing exercised
    # this path on hardware). Translation is the controller's job — the abstract
    # MotionTarget must not know SDK spellings.
    #
    # Keyed by the CANONICAL name (hardware.controller.normalize_method). There is
    # no default: an unknown name is refused, never silently minjerk, and
    # ``cartoon`` is absent because it overshoots its target past any clamp (#1496).
    _METHOD_MAP = _SDK_METHODS

    @classmethod
    def _sdk_method(cls, method: str) -> str:
        """Map a MotionTarget.method name to the Reachy SDK's InterpolationTechnique value."""
        return sdk_method(method)

    # ── PHYSICAL workspace limits (radians) ──────────────────────────────
    # These are CAPABILITY limits, not style preferences — commanding past
    # them is what destroyed motors 2+3 (2026-08-07 hardware note: a pose
    # beyond physical capability made the motors glitch, the head snapped to
    # the opposite extreme, and the robot rotated itself off the table).
    # Sources: Reachy Mini SDK docs — roll/pitch ±40°, head yaw RELATIVE TO
    # BODY max 65°; body yaw ±160° matches the SEM motor backend's
    # _MAX_BODY_YAW_RAD. The clamp lives HERE, at the single dispatch point
    # every sanctioned caller routes through, so no caller can forget it.
    _MAX_HEAD_ROLL_RAD = 0.6981317007977318  # 40°
    _MAX_HEAD_PITCH_RAD = 0.6981317007977318  # 40°
    _MAX_HEAD_REL_YAW_RAD = 1.1344640137963142  # 65° head-relative-to-body
    _MAX_BODY_YAW_RAD = 2.792526803190927  # 160°

    def _clamp(self, value: float, limit: float, axis: str, clamped_axes: list[str]) -> float:
        """Clamp to ±limit; WARN + record the axis when the command exceeded it.

        A non-finite value is refused (#1496): ``abs(nan) > limit`` is False, so
        NaN would otherwise pass straight through to the SDK.
        """
        value = require_finite(axis, value)
        if abs(value) > limit:
            logger.warning(
                "goto_target: %s %.3f rad exceeds physical limit ±%.3f — clamping "
                "(a pose beyond capability is the motor-destruction failure class)",
                axis,
                value,
                limit,
            )
            clamped_axes.append(axis)
            return limit if value > 0 else -limit
        return value

    def _forget_commanded(self, *axes: str) -> None:
        """Drop retained commanded values (all of them when no axes given).

        Call after any motion the stash did not see (look_at_pixel, wake_up,
        goto_sleep) so the next goto_target fill-in re-seeds the dropped
        axes from the readback ONCE. A one-shot seed absorbs at most one
        step of achieved-vs-commanded bias; RE-USING the readback on every
        command is the F1 ratchet this stash exists to prevent.
        """
        with self._motion_lock:
            if not axes:
                self._last_commanded.clear()
            else:
                for axis in axes:
                    self._last_commanded.pop(axis, None)

    def request_abort(self) -> None:
        """Stop any move in progress at its next chunk boundary, without latching (#1499 5.1, the stop fence).

        Increments the abort generation. A dispatch records the generation
        before it waits for the lock and checks it before every chunk, so a
        dispatch waiting behind a stop, or a split that is mid-way, ends
        ``cancelled``. ``goto_sleep`` and ``disconnect`` do not call this: they
        use ``_stop``, which also sets the stop latch in the same locked step.
        This is the generation-only stop, for a caller that wants the current
        move to end but the next command to run.
        """
        with self._abort_lock:
            self._abort_gen += 1

    def goto_target(self, target: MotionTarget) -> MotionOutcome:
        """Move robot to target pose (#1499: fresh readback, rate cap, neck-path split).

        Args:
            target: Target pose specification.

        Returns:
            A ``MotionOutcome``, truthy iff the whole move completed. A refusal
            carries its ``kind`` and ``reason`` and moves nothing.
        """
        if self._mini is None or not self.is_connected():
            return MotionOutcome.refused("error", "the robot is not connected")
        if self._stop_latched:
            return _STOPPED
        gen = self._abort_gen
        # Bounded (delta N5): a dispatch behind a long move returns "motor busy"
        # instead of waiting without limit (the bound covers the wait only).
        if not self._motion_lock.acquire(timeout=MOTION_LOCK_WAIT_S):
            return MotionOutcome.refused("cancelled", "motor busy")
        try:
            return self._goto_target_locked(target, gen=gen)
        except Exception as e:
            logger.error("Motion command failed: %s", e)
            return MotionOutcome.refused("error", f"motion command failed: {e}")
        finally:
            self._motion_lock.release()

    def _goto_target_locked(self, target: MotionTarget, *, gen: int, start: mp.Pose | None = None) -> MotionOutcome:
        """Body of goto_target; runs under ``_motion_lock``. Phases (5.1): read, divergence, compose, then move.

        THE HEAD-FRAME RULE (CLAUDE.md invariant). The SDK's head pose is WORLD
        frame and sits above body_yaw in the kinematic chain: ``head=None``
        counter-rotates the head while the body turns, and a body-relative yaw
        sent as world pins the head at the wrong angle. So every dispatch ships a
        world head matrix composed as ``Rz(body) · R_rel``.
        """
        if self._abort_gen != gen:
            return MotionOutcome.refused("cancelled", "a stop was requested before the move started")
        if self._stop_latched:
            return _STOPPED
        try:
            if start is None:
                start = self._fresh_readback()
            self._require_settled(start)
            self._check_divergence(target, start)
            clamped: list[str] = []
            composed = self._compose(target, start, clamped)
        except MotionRefused as e:
            logger.warning("goto_target refused (%s): %s", e.kind, e.reason)
            return MotionOutcome.refused(e.kind, e.reason)
        if composed is None:
            return MotionOutcome(ok=True, reason="nothing to move", duration_s=0.0)
        pose, axes, body_commanded = composed
        return self._run_move(
            start,
            pose,
            requested=target.duration,
            method=target.method,
            policy=target.duration_policy,
            gen=gen,
            body_commanded=body_commanded,
            final_axes=axes,
            clamped=clamped,
            translation_body_frame=self._body_frame_translation(target),
        )

    # ── 5.1 phase 1: the fresh readback (R1, O3) ──────────────────────────────

    def _get_state_full(self) -> Any:
        """``GET /api/state/full`` from the daemon this controller connected to; JSON or ``MotionRefused``.

        Through ``maxim_http.fetch_url`` (the ``_daemon_status`` pattern) with a
        private address policy and a short per-phase timeout whose phases sum to
        the deadline. The deadline is checked again on return. One read, no
        retry: ``_fresh_readback(retry=True)`` owns the single retry.
        """
        from maxim.utils import http as maxim_http

        host = self._resolved_host
        if not host:
            raise MotionRefused("reference", "no daemon host to read the pose from")
        tunnelled = self._tunnel or self._connection_mode == "localhost_only"
        deadline = FRESH_READ_DEADLINE_TUNNEL_S if tunnelled else FRESH_READ_DEADLINE_S
        phase = deadline / 3.0
        started = time.monotonic()
        try:
            resp = maxim_http.fetch_url(
                f"http://{host}:8000/api/state/full?{_STATE_FULL_QUERY}",
                timeout=maxim_http.TimeoutPolicy(connect_s=phase, read_s=phase, total_s=phase),
                address_policy="private",
            )
        except Exception as e:  # noqa: BLE001 - any transport failure is a missing reference
            raise MotionRefused("reference", f"fresh readback failed: {e}") from e
        elapsed = time.monotonic() - started
        logger.debug("fresh readback: %.1f ms (deadline %.0f ms)", elapsed * 1000, deadline * 1000)
        if elapsed > deadline:
            raise MotionRefused(
                "reference", f"fresh readback took {elapsed * 1000:.0f} ms (deadline {deadline * 1000:.0f} ms)"
            )
        if resp.status != 200:
            raise MotionRefused("reference", f"fresh readback returned HTTP {resp.status}")
        try:
            return resp.json()
        except ValueError as e:
            raise MotionRefused("reference", f"fresh readback is not JSON: {e}") from e

    def _fresh_readback(self, *, retry: bool = False) -> mp.Pose:
        """The pose every check starts from (R1, R2), or ``MotionRefused("reference")``.

        ``retry``: one more read when ``_get_state_full`` fails (the transport, the
        deadline, a non-200 status, a body that is not JSON, a missing host) on
        the reads between chunks and after the last chunk. A read that parses but
        says torque-off or impossible, or has the wrong shape, is never retried.
        """
        if self._automatic_body_yaw or not self._auto_yaw_asserted:
            # O6: with the daemon moving the body behind us, no readback-based check describes what the motors do.
            raise MotionRefused(
                "reference", "automatic_body_yaw is not asserted off, so the daemon may rotate the body"
            )
        try:
            data = self._get_state_full()
        except MotionRefused as first:
            if not retry:
                raise
            logger.debug("fresh readback failed (%s); retrying once", first.reason)
            data = self._get_state_full()
        pose, control_mode = mp.parse_full_state(data)
        return self._checked_pose(pose, control_mode)

    def _checked_pose(self, pose: mp.Pose, control_mode: str) -> mp.Pose:
        """A parsed readback that moves may start from: torque on and nothing impossible (R1, R2)."""
        if control_mode != "enabled":
            raise MotionRefused("reference", f"motor torque is not enabled (control_mode={control_mode})")
        # R2: an impossible body or neck value is refused (world yaw is never
        # bounded: body 160° plus neck 65° legitimately reaches 225°).
        if abs(pose.body) > self._MAX_BODY_YAW_RAD + _IMPOSSIBLE_MARGIN_RAD:
            raise MotionRefused("reference", f"fresh readback body_yaw {math.degrees(pose.body):.0f}° is impossible")
        twist = mp.neck_twist(pose.head, pose.body)
        if abs(twist) > self._MAX_HEAD_REL_YAW_RAD + _IMPOSSIBLE_MARGIN_RAD:
            raise MotionRefused("reference", f"fresh readback neck twist {math.degrees(twist):.0f}° is impossible")
        return pose

    def _require_settled(self, start: mp.Pose) -> None:
        """After an SDK timeout the daemon may still be moving: refuse until it has settled (5.1, delta N4)."""
        unsettled = self._unsettled
        if unsettled is None:
            return
        target, anchor, anchor_t = unsettled
        if self._pose_distance(start, target) <= _SETTLED_NEAR_RAD:
            self._unsettled = None
            return
        now = time.monotonic()
        if anchor is not None and anchor_t is not None and self._pose_distance(start, anchor) <= _SETTLED_STABLE_RAD:
            if now - anchor_t >= _SETTLED_STABLE_S:
                self._unsettled = None  # the move ended somewhere else
                return
        else:
            self._unsettled = (target, start, now)
        raise MotionRefused("reference", "the last move timed out and the robot has not settled yet")

    @staticmethod
    def _pose_distance(a: mp.Pose, b: mp.Pose) -> float:
        """The largest per-axis difference between two poses, radians (body, world head geodesic, antennas)."""
        dist = max(abs(a.body - b.body), mp.geodesic(a.head, b.head))
        if a.antennas is not None and b.antennas is not None:
            dist = max(dist, *(abs(x - y) for x, y in zip(a.antennas, b.antennas, strict=True)))
        return dist

    # ── 5.1 phase 2: divergence (decision 3) ─────────────────────────────────

    def _check_divergence(self, target: MotionTarget, start: mp.Pose) -> None:
        """Refuse when a retained axis (or the body stash) disagrees with the readback; forget that axis.

        Retained axes are the head axes this command leaves unspecified, which
        the stash fills. Relative yaw is compared through R_rel (R3).
        """
        stash = self._last_commanded
        roll, pitch, rel = mp.euler_xyz(mp.relative_rotation(start.head, start.body))
        checks = []
        if target.head_roll is None and "head_roll" in stash:
            checks.append(("head_roll", stash["head_roll"], roll))
        if target.head_pitch is None and "head_pitch" in stash:
            checks.append(("head_pitch", stash["head_pitch"], pitch))
        if target.head_yaw is None and "head_rel_yaw" in stash:
            checks.append(("head_rel_yaw", stash["head_rel_yaw"], rel))
        if "body_yaw" in stash:
            checks.append(("body_yaw", stash["body_yaw"], start.body))
        self._warn_commanded_vs_achieved(checks)
        diverged = [
            f"{axis} {math.degrees(_wrap(commanded - achieved)):+.1f}°"
            for axis, commanded, achieved in checks
            if abs(_wrap(commanded - achieved)) > _STASH_DIVERGENCE_REFUSE_RAD
        ]
        if diverged:
            self._forget_commanded(*(d.split(" ", 1)[0] for d in diverged))
            raise MotionRefused(
                "reference",
                f"the readback diverges from the last command ({', '.join(diverged)}); "
                "the stale axis is forgotten, so the next command reads it back",
            )

    # ── 5.1 phase 3: compose against the readback body and the stash-filled head axes ──

    def _compose(
        self, target: MotionTarget, start: mp.Pose, clamped: list[str]
    ) -> tuple[mp.Pose, tuple[float, float, float], bool] | None:
        """(target pose, (roll, pitch, rel_yaw), body commanded?) — None when nothing is asked to move.

        Unspecified head axes fill from the last COMMANDED value, never the
        readback (the H1 F1 ratchet: filling from the readback integrates any
        achieved-vs-commanded bias). The readback seeds an axis only when the
        stash has none. The body composes against the READBACK, never the stash
        (R1): the stash is only checked for divergence.
        """
        head_requested = any(x is not None for x in (target.head_roll, target.head_pitch, target.head_yaw))
        if not head_requested and target.body_yaw is None:
            return None
        stash = self._last_commanded
        roll0, pitch0, rel0 = mp.euler_xyz(mp.relative_rotation(start.head, start.body))
        body_commanded = target.body_yaw is not None
        body = (
            self._clamp(target.body_yaw, self._MAX_BODY_YAW_RAD, "body_yaw", clamped)
            if target.body_yaw is not None
            else start.body
        )

        def pick(value: float | None, key: str, readback: float) -> float:
            if value is not None:
                return value
            return stash.get(key, readback)

        # Clamp in the BODY-RELATIVE frame, where the 65° neck limit lives.
        rel = self._clamp(
            pick(target.head_yaw, "head_rel_yaw", rel0), self._MAX_HEAD_REL_YAW_RAD, "head_yaw (body-relative)", clamped
        )
        roll = self._clamp(pick(target.head_roll, "head_roll", roll0), self._MAX_HEAD_ROLL_RAD, "head_roll", clamped)
        pitch = self._clamp(
            pick(target.head_pitch, "head_pitch", pitch0), self._MAX_HEAD_PITCH_RAD, "head_pitch", clamped
        )
        translation = self._body_frame_translation(target)
        world_t = None if translation is None else mp.rz(body) @ translation
        return mp.Pose(mp.compose_head(roll, pitch, rel, body, world_t), body), (roll, pitch, rel), body_commanded

    @staticmethod
    def _body_frame_translation(target: MotionTarget) -> np.ndarray | None:
        """The head translation a caller asked for, body frame, metres (``extras`` ``head_x/y/z``), or None.

        ``motion/movement.py::move_head`` carries its mm-scale nudges here when it
        delegates (5.4). Without them the head composes at zero translation, as
        it always has.
        """
        if not any(key in target.extras for key in HEAD_TRANSLATION_EXTRAS):
            return None
        return np.array([float(target.extras.get(key, 0.0)) for key in HEAD_TRANSLATION_EXTRAS])

    # ── 5.1 phases 4 and 5: plan every chunk, then move them one by one ───────

    def _plan(
        self,
        start: mp.Pose,
        pose: mp.Pose,
        *,
        requested: float,
        method: str,
        policy: str,
        translation_body_frame: Any = None,
    ) -> list[tuple[mp.Pose, float]]:
        """Every chunk and its duration, all checked before anything moves (R4, R5); or ``MotionRefused``."""
        if policy == "exact":
            # An explicit duration is never split: rate first (the whole move), then the path as one dispatch.
            duration = mp.resolve_duration(requested, mp.move_delta_deg(start, pose), method, "exact")
            sigma = mp.sigma_deg(start, pose)
            ok, worst = mp.neck_path_ok(start, pose)
            if sigma > mp.PATH_SIGMA_PREFILTER_DEG or not ok:
                raise MotionRefused(
                    "path",
                    f"this move would sweep the neck to {worst:.0f}° mid-path (world-yaw swing {sigma:.0f}°); "
                    "drop the explicit duration so it can be split, or move in smaller steps",
                )
            return [(pose, duration)]
        chunks = mp.plan_chunks(start, pose, translation_body_frame=translation_body_frame)
        shares = [requested] if len(chunks) == 1 else mp.share_durations(requested, start, chunks)
        planned = []
        previous = start
        for chunk, share in zip(chunks, shares, strict=True):
            planned.append((chunk, mp.resolve_duration(share, mp.move_delta_deg(previous, chunk), method, "at_least")))
            previous = chunk
        return planned

    def _run_move(
        self,
        start: mp.Pose,
        pose: mp.Pose,
        *,
        requested: float,
        method: str,
        policy: str,
        gen: int,
        body_commanded: bool,
        final_axes: tuple[float, float, float] | None = None,
        clamped: list[str] | None = None,
        translation_body_frame: Any = None,
    ) -> MotionOutcome:
        """Plan, then dispatch each chunk: abort check, re-check, last mile, SDK call, stash write (5.1)."""
        try:
            planned = self._plan(
                start,
                pose,
                requested=requested,
                method=method,
                policy=policy,
                translation_body_frame=translation_body_frame,
            )
        except MotionRefused as e:
            logger.warning("goto_target refused (%s): %s", e.kind, e.reason)
            return MotionOutcome.refused(e.kind, e.reason)
        n = len(planned)
        done = 0
        total = 0.0
        previous = start
        for k, (chunk, duration) in enumerate(planned, start=1):
            if self._abort_gen != gen:
                return MotionOutcome(
                    ok=False,
                    kind="cancelled",
                    reason="stopped before chunk %d/%d" % (k, n),
                    duration_s=total,
                    chunks_planned=n,
                    chunks_done=done,
                )
            try:
                if k > 1:
                    self._recheck_before_chunk(previous, chunk)
                body_yaw = chunk.body if body_commanded else None
                checked, sdk_m = check_sdk_command(
                    self._mini,
                    duration=duration,
                    head=chunk.head,
                    body_yaw=body_yaw,
                    antennas=chunk.antennas,
                    method=method,
                )
                if k == 1 and clamped is not None:
                    self.last_clamped_axes = tuple(clamped)  # after the last mile: a refusal leaves it as it was
                kwargs: dict[str, Any] = {
                    "head": chunk.head,
                    "body_yaw": body_yaw,
                    "duration": checked,
                    "method": sdk_m,
                }
                if chunk.antennas is not None:
                    kwargs["antennas"] = list(chunk.antennas)
                self._mini.goto_target(**kwargs)
            except MotionRefused as e:
                if k == 1:
                    logger.warning("goto_target refused (%s): %s", e.kind, e.reason)
                    return MotionOutcome.refused(e.kind, e.reason, chunks_planned=n)
                return self._partial(f"chunk {k}/{n}: {e.reason}", total, n, done)
            except TimeoutError as e:
                # F4: the daemon may still be moving. Refuse until it settles.
                self._unsettled = (chunk, None, None)
                return self._partial(f"chunk {k}/{n} timed out ({e}); waiting for the robot to settle", total, n, done)
            except Exception as e:
                if k == 1:
                    raise
                return self._partial(f"chunk {k}/{n} failed: {e}", total, n, done)
            # Stash AFTER the dispatch returns (fail-fast-before-mutation), describing this chunk.
            axes = (
                final_axes
                if k == n and final_axes is not None
                else mp.euler_xyz(mp.relative_rotation(chunk.head, chunk.body))
            )
            self._last_commanded.update({"head_roll": axes[0], "head_pitch": axes[1], "head_rel_yaw": axes[2]})
            if body_commanded:
                self._last_commanded["body_yaw"] = chunk.body
            done = k
            total += checked
            previous = chunk
        try:
            final = self._fresh_readback(retry=True)
        except MotionRefused as e:
            return self._partial(f"the final readback failed: {e.reason}", total, n, done)
        if self._pose_distance(final, previous) > _STASH_DIVERGENCE_REFUSE_RAD:
            return self._partial("the robot did not reach the target (final readback)", total, n, done)
        return MotionOutcome(ok=True, duration_s=total, chunks_planned=n, chunks_done=done)

    def _recheck_before_chunk(self, previous: mp.Pose, chunk: mp.Pose) -> None:
        """Before chunk k > 1: the robot must be where chunk k − 1 went, and chunk k's path from there must pass (R5).

        Catches a chunk the daemon dropped silently while the client reported
        it complete (F3): without this, the next chunk would run from the old
        start as one oversized move.
        """
        read = self._fresh_readback(retry=True)
        if self._pose_distance(read, previous) > _STASH_DIVERGENCE_REFUSE_RAD:
            raise MotionRefused("reference", "the previous chunk did not arrive (the daemon may have dropped it)")
        ok, worst = mp.neck_path_ok(read, chunk)
        if not ok:
            raise MotionRefused("path", f"from the re-read pose the next chunk would sweep the neck to {worst:.0f}°")

    @staticmethod
    def _partial(reason: str, total: float, planned: int, done: int) -> MotionOutcome:
        logger.warning("goto_target stopped after %d/%d chunks: %s", done, planned, reason)
        return MotionOutcome(
            ok=False, kind="partial", reason=reason, duration_s=total, chunks_planned=planned, chunks_done=done
        )

    # Achieved-vs-commanded divergence threshold + throttle for the
    # observability WARNING below. 5° is well above the XVF-era measured
    # per-command bias (~1.3°) so healthy hardware stays quiet.
    _DIVERGENCE_WARN_RAD = 0.0872664625997  # 5°
    _DIVERGENCE_WARN_INTERVAL_S = 30.0

    def _warn_commanded_vs_achieved(self, checks: list[tuple[str, float, float]]) -> None:
        """Rate-limited WARNING when the readback diverges from the stash (progressive-degradation early warning, H1 F1).

        Observational: compare what we last commanded against what the
        hardware reports, per retained axis, and warn (throttled) past 5°. The
        refusal itself is ``_check_divergence``'s.
        """
        deltas = [
            f"{axis} {math.degrees(_wrap(commanded - achieved)):+.1f}deg"
            for axis, commanded, achieved in checks
            if abs(_wrap(commanded - achieved)) > self._DIVERGENCE_WARN_RAD
        ]
        if not deltas:
            return
        now = time.time()
        if now - self._divergence_last_warn < self._DIVERGENCE_WARN_INTERVAL_S:
            return
        self._divergence_last_warn = now
        logger.warning(
            "goto_target: achieved pose diverges from last commanded (%s) — "
            "possible actuator degradation or external motion the controller "
            "did not see (progressive-degradation early warning, H1 F1)",
            ", ".join(deltas),
        )

    def note_external_head_motion(self) -> None:
        """Tell the controller the head moved outside its dispatch paths.

        For scripts that drive the raw SDK and hand control back. Drops the
        retained head axes so the next ``goto_target`` fills them from the
        readback once, instead of snapping the head back to a stale stashed
        pose. (The body is never composed from the stash, R1.)
        """
        self._forget_commanded("head_roll", "head_pitch", "head_rel_yaw")

    def look_at_pixel(self, target: PixelTarget) -> MotionOutcome:
        """Turn the head to look at a pixel, head only: the body never moves (#1499 5.2, F1).

        The SDK's ``look_at_image(perform_movement=True)`` sends the body to 0.
        Instead the SDK only computes the world pose (``perform_movement=False``),
        the controller takes it relative to the fresh readback's body and
        dispatches it through the same path as ``goto_target``.
        """
        if self._mini is None or not self.is_connected():
            return MotionOutcome.refused("error", "the robot is not connected")
        if self._stop_latched:
            return _STOPPED
        gen = self._abort_gen
        if not self._motion_lock.acquire(timeout=MOTION_LOCK_WAIT_S):
            return MotionOutcome.refused("cancelled", "motor busy")
        try:
            try:
                # Last mile first (#1496): PixelTarget is mutable, so a
                # post-construction 0 or NaN duration is refused here.
                require_finite("u", target.u)
                require_finite("v", target.v)
                require_duration(target.duration)
                start = self._fresh_readback()
            except MotionRefused as e:
                return MotionOutcome.refused(e.kind, e.reason)
            world = self._mini.look_at_image(target.u, target.v, duration=target.duration, perform_movement=False)
            roll, pitch, rel = mp.euler_xyz(mp.relative_rotation(np.asarray(world, dtype=float), start.body))
            head_only = MotionTarget(
                head_roll=roll,
                head_pitch=pitch,
                head_yaw=rel,
                duration=target.duration,
                duration_policy=target.duration_policy,
            )
            return self._goto_target_locked(head_only, gen=gen, start=start)
        except Exception as e:
            logger.error("Look-at command failed: %s", e)
            return MotionOutcome.refused("error", f"look-at failed: {e}")
        finally:
            self._motion_lock.release()

    def get_current_pose(self) -> dict[str, float]:
        """Get current joint positions.

        Returns:
            Dict mapping joint names to current angles. ``yaw``/``pitch``/``roll``
            are the head's **WORLD-frame** euler angles (the SDK's fk() folds
            body_yaw in), and ``body_yaw`` is the base angle — so the head's angle
            RELATIVE to the body is ``yaw - body_yaw``. Mixing those frames up is
            the head-frame trap (see CLAUDE.md).
        """
        if self._mini is None:
            return {}

        try:
            pose = {}

            # SDK >= 1.5: get_current_joint_positions() returns the 7 Stewart
            # MOTOR angles — not (roll, pitch, yaw). Euler angles come from
            # the 4x4 head pose matrix instead.
            import math

            m = self._mini.get_current_head_pose()
            if m is not None:
                pose["yaw"] = float(math.atan2(m[1][0], m[0][0]))
                pose["pitch"] = float(math.atan2(-m[2][0], math.hypot(m[2][1], m[2][2])))
                pose["roll"] = float(math.atan2(m[2][1], m[2][2]))

            try:
                head_joints, antenna_joints = self._mini.get_current_joint_positions()
                # The SDK's joint vector is [body_yaw, *stewart_legs] — its own
                # kinematics reads index 0 as body_yaw (analytical_kinematics.fk:
                # `body_yaw = joint_angles[0]`; ik returns `[body_yaw] + stewart`).
                # Load-bearing for goto_target's head-frame handling: the head pose
                # is WORLD-referenced, so converting a body-relative head yaw needs
                # the body's actual angle. Without this, "head rides along" silently
                # assumes the body sits at 0.
                if head_joints is not None and len(head_joints) >= 1:
                    pose["body_yaw"] = float(head_joints[0])
                if antenna_joints is not None and len(antenna_joints) >= 2:
                    pose["antenna_left"] = float(antenna_joints[0])
                    pose["antenna_right"] = float(antenna_joints[1])
            except Exception:  # noqa: BLE001 - joint read is best-effort
                pass

            self._update_state(current_pose=pose)
            return pose

        except Exception as e:
            logger.debug("Failed to get pose: %s", e)
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def _assert_auto_yaw_off(self) -> bool:
        """Assert ``set_automatic_body_yaw(False)`` on the handle (O6); motion is refused until it succeeds.

        The setting is daemon-global and has no getter, so another client (or an
        SDK default of True on a handle Maxim did not build) can turn it on.
        """
        try:
            self._mini.set_automatic_body_yaw(False)
        except Exception as e:  # noqa: BLE001 - an SDK without the setter cannot be trusted with the body
            logger.error(
                "set_automatic_body_yaw(False) failed (%s): refusing motion, since the daemon may rotate the body", e
            )
            self._auto_yaw_asserted = False
            return False
        self._auto_yaw_asserted = True
        return True

    def _probe_state_shape(self) -> None:
        """One readback at connect: an unknown ``/api/state/full`` shape is one ERROR now, not a refusal later."""
        try:
            mp.parse_full_state(self._get_state_full())
        except MotionRefused as e:
            logger.error(
                "The daemon's GET /api/state/full could not be read or did not parse (%s; reachy-mini SDK %s): "
                "every motion will be refused with 'reference' until it can",
                e.reason,
                ".".join(str(x) for x in (self._installed_sdk_version() or ("?",))),
            )

    def _wait_torque_on(self) -> bool:
        """Poll the REST ``control_mode`` until ``enabled`` (``enable_motors`` is fire-and-forget; delta S3)."""
        deadline = time.monotonic() + _TORQUE_ON_WAIT_S
        while True:
            try:
                _, control_mode = mp.parse_full_state(self._get_state_full())
                if control_mode == "enabled":
                    return True
            except MotionRefused as e:
                logger.debug("torque-on poll: %s", e.reason)
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def _play_sound(self, name: str) -> None:
        """A lifecycle sound, best-effort: a failed sound fails no step (O8)."""
        try:
            self._mini.media.play_sound(name)
        except Exception as e:  # noqa: BLE001 - audio is cosmetic here; the move is what matters
            logger.warning("lifecycle sound %s failed: %s", name, e)

    def _lifecycle_step(
        self, head: Any, antennas: tuple[float, float] | None, duration: float, *, gen: int
    ) -> MotionOutcome:
        """One composed lifecycle step: a world head pose at body 0, antennas clamped, through the dispatch (O8)."""
        start = self._fresh_readback()
        clamped = None
        if antennas is not None:
            clamped = (
                clamp_finite("antenna left", antennas[0], -ANTENNA_LIMIT_RAD, ANTENNA_LIMIT_RAD),
                clamp_finite("antenna right", antennas[1], -ANTENNA_LIMIT_RAD, ANTENNA_LIMIT_RAD),
            )
        target = mp.Pose(np.asarray(head, dtype=float), 0.0, clamped)
        return self._run_move(
            start,
            target,
            requested=duration,
            method="minimum_jerk",
            policy="at_least",
            gen=gen,
            body_commanded=True,
        )

    def _recentre_if_turned(self, *, gen: int) -> MotionOutcome | None:
        """Before a lifecycle move: bring a turned body and neck to 0 first (5.3), or None when already there.

        Holds the readback's roll, pitch and translation (in the body frame), so
        the head is never lifted out of the sleep pose, with a 2 s floor shared
        across the chunks.
        """
        start = self._fresh_readback()
        twist = mp.neck_twist(start.head, start.body)
        if abs(start.body) <= _LIFECYCLE_DEADBAND_RAD and abs(twist) <= _LIFECYCLE_DEADBAND_RAD:
            return None
        roll, pitch, _ = mp.euler_xyz(mp.relative_rotation(start.head, start.body))
        held = mp.rz(-start.body) @ np.asarray(start.head, dtype=float)[:3, 3]
        target = mp.Pose(mp.compose_head(roll, pitch, 0.0, 0.0, held), 0.0)
        return self._run_move(
            start,
            target,
            requested=_LIFECYCLE_RECENTRE_FLOOR_S,
            method="minimum_jerk",
            policy="at_least",
            gen=gen,
            body_commanded=True,
            translation_body_frame=held,
        )

    def _await_settled(self, wait_s: float, *, gen: int) -> None:
        """After an SDK timeout: settled now, or within ``wait_s`` (re-reading every 0.1 s), else ``MotionRefused``.

        A stop during the wait ends it at once (``cancelled``).
        """
        deadline = time.monotonic() + wait_s
        while True:
            if self._abort_gen != gen:
                raise MotionRefused("cancelled", "a stop was requested while waiting for the robot to settle")
            try:
                self._require_settled(self._fresh_readback())
                return
            except MotionRefused:
                if time.monotonic() >= deadline:
                    raise
            time.sleep(0.1)

    def _run_lifecycle(self, steps: list[_LifecycleStep], *, gen: int, settle_wait_s: float) -> MotionOutcome:
        """Re-centre if needed, then each step in order; the first failure, or a stop, ends it (5.3, O8).

        A step is ``(head, antennas, duration)`` for a move or ``(None, None, 0, sound)`` for a sound.
        ``gen`` is the stop generation captured once for the whole call, so a
        stop between steps ends the sequence, not just the current chunk.
        Every failure, an unexpected one included, returns a refused outcome.
        """
        try:
            if self._unsettled is not None:
                self._await_settled(settle_wait_s, gen=gen)
            recentred = self._recentre_if_turned(gen=gen)
            if recentred is not None and not recentred:
                return recentred
            outcome = MotionOutcome(ok=True, duration_s=recentred.duration_s if recentred else 0.0)
            total = outcome.duration_s or 0.0
            for head, antennas, duration, sound in steps:
                if self._abort_gen != gen:
                    return MotionOutcome.refused("cancelled", "a stop was requested between lifecycle steps")
                if head is None:
                    self._play_sound(sound or "")
                    continue
                outcome = self._lifecycle_step(head, antennas, duration, gen=gen)
                if not outcome:
                    return outcome
                total += outcome.duration_s or 0.0
            # (_await_settled has already cleared any unsettled state: a lifecycle that ran is settled.)
            return MotionOutcome(
                ok=True, duration_s=total, chunks_planned=outcome.chunks_planned, chunks_done=outcome.chunks_done
            )
        except MotionRefused as e:
            logger.error("lifecycle motion refused (%s): %s", e.kind, e.reason)
            return MotionOutcome.refused(e.kind, e.reason)
        except Exception as e:  # noqa: BLE001 - the lifecycle reports, never raises (callers do not guard it)
            logger.error("lifecycle motion failed: %s", e)
            return MotionOutcome.refused("error", f"lifecycle motion failed: {e}")

    def wake_up(self) -> MotionOutcome:
        """Enable torque, then wake the robot with the SDK's steps, composed and clamped here (#1499 5.3, O8).

        SDK >= 1.5: the daemon boots torque-off (``--no-wake-up-on-start``),
        and without ``enable_motors()`` every motion command is silently
        ignored while reads keep working (docs/embodiment/reachy_mini/
        troubleshooting.md). ``enable_motors`` is fire-and-forget, so the
        wake waits (≤ 1 s) for the REST ``control_mode`` to read ``enabled``.
        A turned body is re-centred first; then INIT with the INIT antennas
        (2 s), the wake sound, a 20° roll and back (0.2 s each). Only a wake that
        succeeds clears the stop latch; a stop requested after the wake was,
        while it waits or during it, ends it and leaves the latch set.

        Returns:
            A ``MotionOutcome``, truthy iff the robot is awake.
        """
        if self._mini is None or not self.is_connected():
            return MotionOutcome.refused("error", "the robot is not connected")
        requested_gen = self._abort_gen
        with self._motion_lock:  # the lifecycle waits without the tool bound (5.1)
            gen = requested_gen  # one read: a stop at any point after the request ends this wake
            if self._abort_gen != gen:
                return MotionOutcome.refused("cancelled", "a stop was requested while the wake waited")
            try:
                self._mini.enable_motors()
            except Exception as e:  # noqa: BLE001 - hardware boundary must fail closed
                # No motion here (Executor S6): torque state is unknown, so a move could fight it.
                logger.error("enable_motors() failed (%s) — refusing to claim the robot is awake", e)
                try:
                    self._mini.disable_motors()
                except Exception as off_error:  # noqa: BLE001 - best-effort, logged
                    logger.error("disable_motors() after the failed enable also failed: %s", off_error)
                self._update_state(is_awake=False)
                return MotionOutcome.refused("error", f"enable_motors failed: {e}")
            if not self._assert_auto_yaw_off():
                self._update_state(is_awake=False)
                return MotionOutcome.refused("reference", "automatic_body_yaw could not be asserted off")
            if not self._wait_torque_on():
                logger.error("motor torque did not come on within %.1f s — not waking", _TORQUE_ON_WAIT_S)
                self._update_state(is_awake=False)
                return MotionOutcome.refused("reference", "motor torque did not come on")
            roll_20 = mp.compose_head(math.radians(20.0), 0.0, 0.0, 0.0)
            outcome = self._run_lifecycle(
                [
                    (INIT_HEAD_POSE, INIT_ANTENNAS_RAD, 2.0, None),
                    (None, None, 0.0, "wake_up.wav"),
                    (roll_20, None, 0.2, None),
                    (INIT_HEAD_POSE, None, 0.2, None),
                ],
                gen=gen,
                settle_wait_s=0.0,  # wake refuses until the robot has settled (owner, 2026-10-29)
            )
            # Only a wake that SUCCEEDED, with no stop since it was requested, ends a stop (owner, 2026-10-29).
            if outcome and self._clear_latch_after_wake(gen):
                self._update_state(is_awake=True)
                logger.info("Reachy Mini awake: %s", self._robot_name)
                return outcome
            self._update_state(is_awake=False)
            return (
                outcome if not outcome else MotionOutcome.refused("cancelled", "a stop was requested during the wake")
            )

    def goto_sleep(self) -> MotionOutcome:
        """Put the robot to sleep with the SDK's steps, composed and clamped here (#1499 5.3, O8).

        Sets the stop latch (and is_awake=False) and calls the stop fence first:
        no new head or body move starts, a move in progress ends at its next
        chunk (or step), and sleep never waits behind it as "motor busy". The
        latch holds even when the sleep then fails, until a wake succeeds. With
        torque already off the robot is safe and nothing moves (ok). After an SDK timeout it waits up to 2 s for
        the robot to settle. A turned body is re-centred; the head goes through
        INIT when far from it, then the sleep sound and the sleep pose with the
        antennas clamped to ±``ANTENNA_LIMIT_RAD`` (the SDK's ±3.05 rad never
        reaches a motor). Torque stays on, as with the SDK's own sleep.

        Returns:
            A ``MotionOutcome``, truthy iff the robot is in the sleep pose or its torque is off.
        """
        # Latch and stop first, as one step, even with no SDK handle (mid-reconnect):
        # nothing new may move the robot, and connect will not clear a sleep's latch.
        self._stop(by_sleep=True)
        if self._mini is None:
            # Owner, 2026-10-29: nothing was made safe, so this refuses (api.py keeps treating it as unproven).
            return MotionOutcome.refused("error", "not connected: the robot was not put to sleep")
        with self._motion_lock:  # no timeout: the fence bounds the wait to the current chunk
            # Again under the lock: a wake that won the (unfair) lock while this
            # sleep waited must not leave the sleeping robot unlatched.
            self._latch_for_sleep()
            gen = self._abort_gen
            try:
                if self._automatic_body_yaw or not self._auto_yaw_asserted:
                    raise MotionRefused("reference", "automatic_body_yaw is not asserted off")
                pose, control_mode = mp.parse_full_state(self._get_state_full())  # one read for both questions
                if control_mode == "disabled":
                    logger.info("Reachy Mini asleep: %s (torque off, nothing to move)", self._robot_name)
                    return MotionOutcome(ok=True, reason="torque off: already safe")
                start = self._checked_pose(pose, control_mode)
            except MotionRefused as e:
                logger.error("goto_sleep refused (%s): %s", e.kind, e.reason)
                return MotionOutcome.refused(e.kind, e.reason)
            steps: list[_LifecycleStep] = []
            if mp.geodesic(mp.relative_rotation(start.head, start.body), INIT_HEAD_POSE) > _SLEEP_VIA_INIT_RAD:
                steps.append((INIT_HEAD_POSE, INIT_ANTENNAS_RAD, 1.0, None))
            steps.append((None, None, 0.0, "go_sleep.wav"))
            steps.append((SLEEP_HEAD_POSE, SLEEP_ANTENNAS_RAD, 2.0, None))
            outcome = self._run_lifecycle(steps, gen=gen, settle_wait_s=_SLEEP_SETTLE_WAIT_S)
            if outcome:
                logger.info("Reachy Mini asleep: %s", self._robot_name)
            return outcome

    # ── The stop latch (MP9). Every transition runs under _abort_lock together with
    # the stop generation, so no read-then-write of the latch can interleave with
    # another (connect, a wake's success, a sleep, a disconnect).

    def _stop(self, *, by_sleep: bool) -> None:
        """Latch and bump the stop generation as one step (goto_sleep, disconnect)."""
        with self._abort_lock:
            self._stop_latched = True
            if by_sleep:
                self._stop_by_sleep = True
            self._abort_gen += 1
        if by_sleep:
            self._update_state(is_awake=False)

    def _latch_for_sleep(self) -> None:
        """Re-assert a sleep's latch once it holds the motion lock (a wake may have won the lock first)."""
        with self._abort_lock:
            self._stop_latched = True
            self._stop_by_sleep = True
        self._update_state(is_awake=False)

    def _clear_latch_after_connect(self) -> None:
        """A new connection ends a disconnect's latch, never a sleep's."""
        with self._abort_lock:
            if not self._stop_by_sleep:
                self._stop_latched = False

    def _clear_latch_after_wake(self, gen: int) -> bool:
        """End the stop after a successful wake, only if no stop came since the wake was requested."""
        with self._abort_lock:
            if self._abort_gen != gen:
                return False
            self._stop_latched = False
            self._stop_by_sleep = False
            return True

    def start_recording(self) -> bool:
        """Start media recording/streaming.

        Returns:
            True if recording started.
        """
        if self._mini is None or not self.is_connected():
            return False

        try:
            self._mini.start_recording()

            # Start streams
            if self._video_stream:
                self._video_stream.start()
            if self._audio_stream:
                self._audio_stream.start()
                self._audio_stream.update_sample_rates()

            self._update_state(is_recording=True)
            logger.info("Recording started: %s", self._robot_name)
            return True

        except Exception as e:
            logger.error("Failed to start recording: %s", e)
            return False

    def stop_recording(self) -> bool:
        """Stop media recording/streaming.

        Returns:
            True if recording stopped.
        """
        if self._mini is None:
            return True

        try:
            # Stop streams
            if self._video_stream:
                self._video_stream.stop()
            if self._audio_stream:
                self._audio_stream.stop()

            self._mini.stop_recording()

            self._update_state(is_recording=False)
            logger.info("Recording stopped: %s", self._robot_name)
            return True

        except Exception as e:
            logger.error("Failed to stop recording: %s", e)
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Data Streams
    # ─────────────────────────────────────────────────────────────────────────

    def get_video_stream(self) -> VideoStream | None:
        """Get the video stream.

        Returns:
            ReachyVideoStream, or None if not available.
        """
        return self._video_stream

    def get_audio_stream(self) -> AudioStream | None:
        """Get the audio stream.

        Returns:
            ReachyAudioStream, or None if not available.
        """
        return self._audio_stream

    def get_doa_reader(self):
        """Get a DoA reader for the XVF3800 sound-localization chip.

        Transport choice mirrors ``build_reachy_audio_orienting_source``
        (live_audio_orient_wiring.md Stage 1b):

        * Running ON the robot (``localhost_only`` without a tunnel): the
          onboard local-USB path — ``mini.media.get_DoA()`` via
          ``make_reachy_doa_reader``. This call is local-USB in SDK >= 1.5
          and yields nothing off-robot, hence the split.
        * Networked / tunneled: the daemon serves DoA at
          ``GET /api/state/doa`` — ``make_reachy_rest_doa_reader`` against
          the host ``connect()`` resolved (127.0.0.1 under a tunnel, which
          the tunnel forwards).

        Returns ``None`` when not connected — absent capability, no dead
        config.
        """
        if self._mini is None or not self.is_connected():
            return None
        from maxim.embodiment.audio_localization import (
            make_reachy_doa_reader,
            make_reachy_rest_doa_reader,
        )

        if self._connection_mode == "localhost_only" and not self._tunnel:
            return make_reachy_doa_reader(self._mini)
        if self._resolved_host:
            return make_reachy_rest_doa_reader(self._resolved_host)
        return None
