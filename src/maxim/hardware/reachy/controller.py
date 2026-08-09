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

from maxim.hardware.capabilities import (
    REACHY_MINI_CAPABILITIES,
    RobotConnectionState,
)
from maxim.hardware.controller import MotionTarget, PixelTarget, RobotController
from maxim.hardware.reachy.streams import ReachyAudioStream, ReachyVideoStream
from maxim.hardware.streams import AudioStream, VideoStream

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
            automatic_body_yaw: Whether the DAEMON may auto-rotate the body
                to follow the head. Default ``False`` — Maxim's loops own
                the yaw axis, and per the CLAUDE.md head-frame invariant a
                daemon that modulates body_yaw behind the runtime rotates
                the frame every yaw computation lives in (observed
                2026-08-03: −25° of uncommanded body rotation put every
                focus_on_sound aim off by up to that much with a perfect
                sensor). The orient scripts have always disabled it
                (Step-2 calibration); this makes the runtime match.
                Re-enable via robots.yaml ``config: {automatic_body_yaw:
                true}`` if the daemon behavior is explicitly wanted.
        """
        super().__init__(robot_id or robot_name)
        self._robot_name = robot_name
        self._media_backend = media_backend
        self._connection_mode = connection_mode
        self._host = host
        self._tunnel = tunnel
        self._ssh_user = ssh_user
        self._ssh_port = ssh_port
        self._automatic_body_yaw = bool(automatic_body_yaw)
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
        # "body_yaw". goto_target fills UNSPECIFIED axes from here, never
        # from the live readback — see the F1 ratchet note in
        # _goto_target_locked. Mutated only under _motion_lock; cleared by
        # out-of-band motion (look_at_pixel / wake_up / goto_sleep) and
        # disconnect via _forget_commanded so the next fill-in re-seeds
        # from the readback once.
        self._last_commanded: dict[str, float] = {}
        # Throttle for the achieved-vs-commanded divergence WARNING.
        self._divergence_last_warn: float = 0.0

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
                # Maxim owns the yaw axis (head-frame invariant): stop the
                # daemon from rotating the body behind the runtime.
                automatic_body_yaw=self._automatic_body_yaw,
            )

            # Create stream wrappers
            self._video_stream = ReachyVideoStream(
                self._mini,
                resolution=(640, 480),  # Will be updated after recording starts
            )
            self._audio_stream = ReachyAudioStream(self._mini)

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
    _METHOD_MAP = {
        "minimum_jerk": "minjerk",
        "minjerk": "minjerk",
        "linear": "linear",
        "ease_in_out": "ease_in_out",
        "cartoon": "cartoon",
    }

    @classmethod
    def _sdk_method(cls, method: str) -> str:
        """Map a MotionTarget.method name to the Reachy SDK's InterpolationTechnique value."""
        return cls._METHOD_MAP.get(method, "minjerk")

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
        """Clamp to ±limit; WARN + record the axis when the command exceeded it."""
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

    def goto_target(self, target: MotionTarget) -> bool:
        """Move robot to target pose.

        Args:
            target: Target pose specification.

        Returns:
            True if motion command accepted.
        """
        if self._mini is None or not self.is_connected():
            return False

        try:
            with self._motion_lock:
                return self._goto_target_locked(target)
        except Exception as e:
            logger.error("Motion command failed: %s", e)
            return False

    def _goto_target_locked(self, target: MotionTarget) -> bool:
        """Body of goto_target; runs under ``_motion_lock``.

        The lock spans get_current_pose() → head-matrix composition → SDK
        dispatch so overlapping callers cannot compose against a body yaw
        that another call is concurrently changing (TOCTOU on live
        kinematic state).
        """
        # Build head target. SDK >= 1.5 goto_target(head=...) requires a
        # 4x4 pose MATRIX (the zenoh-era (roll, pitch, yaw) tuple would be
        # flatten()ed to 3 values where the daemon expects 16). Compose
        # the pose from euler targets via the SDK's own helper.
        #
        # THE HEAD-FRAME RULE (CLAUDE.md invariant, 2026-07-16). The head
        # pose is in the WORLD frame and sits ABOVE body_yaw in the
        # kinematic chain, so:
        #   * head=None does NOT mean "leave the head alone" — the daemon
        #     re-solves IK against the RETAINED world head target, which
        #     COUNTER-ROTATES the head while the body turns under it. The
        #     head-mounted camera and mic array then barely move (measured:
        #     0.32 rad of sensor rotation for a 0.9 rad body command).
        #   * a head yaw expressed as if body-relative is also wrong: the
        #     pose is world, so head_yaw=0.1 with body_yaw=0.5 pins the head
        #     at world 0.1 instead of 0.6.
        # Both were live here until 2026-07-16. Pollen's prescription:
        # "to make the head follow the body, ship a `head` matrix in the same
        # call with the body delta added to the head yaw."
        clamped_axes: list[str] = []
        body_yaw_target = target.body_yaw
        if body_yaw_target is not None:
            body_yaw_target = self._clamp(body_yaw_target, self._MAX_BODY_YAW_RAD, "body_yaw", clamped_axes)
        head_requested = any(x is not None for x in [target.head_roll, target.head_pitch, target.head_yaw])

        head_target = None
        roll = pitch = relative_yaw = 0.0
        if head_requested or body_yaw_target is not None:
            from reachy_mini.utils import create_head_pose

            # Fill UNSPECIFIED axes from the last COMMANDED value, never the
            # live readback (F1 ratchet fix, H1 2026-08-08). Filling from the
            # readback is positive feedback under any achieved-vs-commanded
            # bias: each command re-targets the previous ACHIEVED value, so
            # the bias integrates — ~20 yaw-only probe commands ratcheted
            # roll +~1.3°/command up to +38° during H1 (see the F1 finding in
            # docs/experiments/protocols/h1_healthy_hardware_doa_preregistration.md).
            # The readback SEEDS an axis only when it has never been
            # commanded through here (or after out-of-band motion cleared
            # the stash) — a one-shot read has no feedback loop.
            stash = self._last_commanded
            current = self.get_current_pose()
            # Refuse to guess (executor-lens review fold, mirroring
            # ReachyOrientMotorBackend's E2 rule): SEEDING the retained
            # relative yaw needs both the world head yaw and the body yaw
            # from the readback. Treating an unreadable one as 0 commands a
            # swing of up to the full body angle — legal under the clamp,
            # violent in practice. With a stashed commanded yaw the readback
            # is not needed for the yaw seed, so an unreadable pose no
            # longer blocks — but see the body-seed refusal just below.
            if (
                target.head_yaw is None
                and "head_rel_yaw" not in stash
                and ("yaw" not in current or "body_yaw" not in current)
            ):
                logger.error(
                    "goto_target: current pose unreadable (have keys %s) — refusing to guess "
                    "the retained head yaw; re-issue with an explicit head_yaw or retry",
                    sorted(current.keys()),
                )
                return False
            # Same rule for the BODY seed (two-lens fold, cross-confirmed):
            # the world-yaw composition needs a body angle. Guessing 0 when
            # the body was never commanded AND the readback is dark ships a
            # world head yaw that can demand a neck angle off by the full
            # body rotation — the relative-frame clamp cannot catch it
            # because it runs in the BELIEVED frame. (Head-only dispatches
            # deliberately do not stash the body, so this seed recurs until
            # a body command or a successful readback.)
            if body_yaw_target is None and "body_yaw" not in stash and "body_yaw" not in current:
                logger.error(
                    "goto_target: body yaw unreadable (have keys %s) and never commanded — "
                    "refusing to guess the body angle for head composition; re-issue with an "
                    "explicit body_yaw or retry",
                    sorted(current.keys()),
                )
                return False
            # get_current_pose() reports head yaw in the WORLD frame (the
            # SDK's fk() folds body_yaw in), so the head's angle RELATIVE to
            # the body is (world head yaw - body yaw). Callers naturally mean
            # "point the head there relative to the body", so re-add the
            # TARGET body yaw to keep the head riding along.
            # target.head_yaw is interpreted BODY-RELATIVE (what callers mean by
            # "look left"); current["yaw"] is WORLD. Convert, then re-add the
            # target body yaw so the head ends up riding along with the body.
            if target.head_yaw is not None:
                relative_yaw = target.head_yaw
            elif "head_rel_yaw" in stash:
                relative_yaw = stash["head_rel_yaw"]
            else:
                relative_yaw = current["yaw"] - current["body_yaw"]
            # Body fill-in for the world-yaw composition: last commanded
            # body when one exists. A never-commanded body reads back its
            # actual angle instead — head-only dispatches don't move the
            # body, so there is no command→achieve→re-command loop there.
            # (Under automatic_body_yaw the daemon moves the body behind
            # the runtime, so the body is never stashed — see the stash
            # write below — and this branch always reads back.)
            if body_yaw_target is not None:
                target_body = body_yaw_target
            elif "body_yaw" in stash:
                target_body = stash["body_yaw"]
            else:
                target_body = current.get("body_yaw", 0.0) or 0.0
            self._warn_commanded_vs_achieved(stash, current)
            # Clamp in the BODY-RELATIVE frame — that is where the physical
            # 65° neck constraint lives. Clamping the world-frame sum instead
            # would spuriously limit a legal head pose whenever the body is
            # turned, and miss an illegal one whenever body and head offsets
            # cancel.
            relative_yaw = self._clamp(
                relative_yaw, self._MAX_HEAD_REL_YAW_RAD, "head_yaw (body-relative)", clamped_axes
            )
            roll = self._clamp(
                (
                    target.head_roll
                    if target.head_roll is not None
                    else stash.get("head_roll", current.get("roll", 0.0))
                ),
                self._MAX_HEAD_ROLL_RAD,
                "head_roll",
                clamped_axes,
            )
            pitch = self._clamp(
                (
                    target.head_pitch
                    if target.head_pitch is not None
                    else stash.get("head_pitch", current.get("pitch", 0.0))
                ),
                self._MAX_HEAD_PITCH_RAD,
                "head_pitch",
                clamped_axes,
            )
            head_target = create_head_pose(
                roll=roll,
                pitch=pitch,
                yaw=relative_yaw + target_body,
                degrees=False,
            )

        self.last_clamped_axes = tuple(clamped_axes)
        self._mini.goto_target(
            head=head_target,
            body_yaw=body_yaw_target,
            duration=target.duration,
            method=self._sdk_method(target.method),
        )

        # Stash AFTER the dispatch returns (fail-fast-before-mutation): a
        # raised dispatch must not leave the stash asserting values that
        # never shipped. Post-clamp values — what was actually commanded.
        if head_target is not None:
            stash = self._last_commanded
            stash["head_roll"] = roll
            stash["head_pitch"] = pitch
            stash["head_rel_yaw"] = relative_yaw
        # Under automatic_body_yaw the daemon rotates the body behind the
        # runtime with no invalidation hook, so a stashed body value would
        # go stale without bound (two-lens fold): never stash the body in
        # that mode — the fill-in above then always reads the body back.
        if body_yaw_target is not None and not self._automatic_body_yaw:
            self._last_commanded["body_yaw"] = body_yaw_target

        return True

    # Achieved-vs-commanded divergence threshold + throttle for the
    # observability WARNING below. 5° is well above the XVF-era measured
    # per-command bias (~1.3°) so healthy hardware stays quiet.
    _DIVERGENCE_WARN_RAD = 0.0872664625997  # 5°
    _DIVERGENCE_WARN_INTERVAL_S = 30.0

    def _warn_commanded_vs_achieved(self, stash: dict[str, float], current: dict[str, float]) -> None:
        """Rate-limited WARNING when the readback diverges from the stash.

        The stash deliberately stops the readback from steering commands
        (the F1 ratchet), but that also silences the signal that exposed
        F1 in the first place: a persistently mis-achieving actuator. This
        is the early-warning channel for progressive motor degradation —
        compare what we last commanded against what the hardware reports,
        per axis, and warn (throttled) past 5°. Purely observational;
        never feeds back into any command.
        """
        deltas: list[str] = []
        for key, achieved in (
            ("head_roll", current.get("roll")),
            ("head_pitch", current.get("pitch")),
            ("body_yaw", current.get("body_yaw")),
        ):
            if key in stash and achieved is not None:
                d = stash[key] - achieved
                if abs(d) > self._DIVERGENCE_WARN_RAD:
                    deltas.append(f"{key} {math.degrees(d):+.1f}deg")
        if "head_rel_yaw" in stash and "yaw" in current and "body_yaw" in current:
            d = stash["head_rel_yaw"] - (current["yaw"] - current["body_yaw"])
            if abs(d) > self._DIVERGENCE_WARN_RAD:
                deltas.append(f"head_rel_yaw {math.degrees(d):+.1f}deg")
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

        Public invalidation hook for sanctioned raw-SDK head movers
        (``motion/movement.py::move_head``, Selfy's raw ``look_at_image``
        enqueue) and for scripts that drive the raw SDK and hand control
        back. Drops the retained head axes so the next ``goto_target``
        fill-in re-seeds them from the readback once — without this, the
        next controller command would snap the head back to the stale
        stashed pose. The body stash is kept: raw head movers do not
        command the body axis.
        """
        self._forget_commanded("head_roll", "head_pitch", "head_rel_yaw")

    def look_at_pixel(self, target: PixelTarget) -> bool:
        """Move head to look at a pixel coordinate.

        Args:
            target: Target pixel coordinates.

        Returns:
            True if motion command accepted.
        """
        if self._mini is None or not self.is_connected():
            return False

        try:
            # Same lock as goto_target: a look_at in flight changing the head
            # pose between another caller's pose read and its dispatch is the
            # same TOCTOU (clamp risk is low here — pixel→angle is
            # FOV-bounded — but the serialization claim must be true).
            with self._motion_lock:
                # look_at_image moves the head outside goto_target's stash —
                # the retained head axes will no longer describe where the
                # head was commanded. Forget BEFORE dispatching (two-lens
                # fold): invalidation must be conservative in the opposite
                # direction from population — a spurious clear costs one
                # readback seed, a missed clear (dispatch raising after
                # partial motion) leaves stale re-commands.
                # (Body is untouched by look_at_image; its stash stays.)
                self._forget_commanded("head_roll", "head_pitch", "head_rel_yaw")
                self._mini.look_at_image(
                    target.u,
                    target.v,
                    duration=target.duration,
                )
            return True

        except Exception as e:
            logger.error("Look-at command failed: %s", e)
            return False

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

    def wake_up(self) -> bool:
        """Enable motor torque, then wake the robot.

        SDK >= 1.5: ``wake_up()`` only moves + plays a sound — it does NOT
        enable torque, and the daemon boots torque-off
        (``--no-wake-up-on-start``). Without ``enable_motors()`` first, every
        motion command is silently ignored while reads keep working (see
        docs/embodiment/reachy_mini/troubleshooting.md).

        Returns:
            True if wake-up successful.
        """
        if self._mini is None or not self.is_connected():
            return False

        try:
            try:
                self._mini.enable_motors()
            except Exception as e:  # noqa: BLE001 - older SDKs may lack it
                logger.warning("enable_motors() failed (%s) — motion may be ignored", e)
            # The wake motion moves the robot outside goto_target's stash —
            # forget BEFORE dispatching so a raise after partial motion
            # cannot leave stale values (spurious clear costs one seed).
            self._forget_commanded()
            self._mini.wake_up()
            # Belt-and-suspenders re-assert after wake (mirrors the orient
            # scripts): construction already passed automatic_body_yaw, but
            # a daemon-side reset on wake must not silently re-enable the
            # frame-rotating behavior the runtime's yaw math assumes off.
            try:
                self._mini.set_automatic_body_yaw(self._automatic_body_yaw)
                logger.info(
                    "automatic_body_yaw=%s (daemon body-follow %s)",
                    self._automatic_body_yaw,
                    "enabled" if self._automatic_body_yaw else "disabled — Maxim owns the yaw axis",
                )
            except Exception as e:  # noqa: BLE001 - older SDKs may lack it
                logger.warning("set_automatic_body_yaw unavailable (%s) — daemon may auto-rotate the body", e)
            self._update_state(is_awake=True)
            logger.info("Reachy Mini awake: %s", self._robot_name)
            return True

        except Exception as e:
            logger.error("Wake-up failed: %s", e)
            return False

    def goto_sleep(self) -> bool:
        """Put the robot to sleep (disable motors).

        Returns:
            True if sleep command successful.
        """
        if self._mini is None:
            return True  # Already "asleep"

        try:
            # The sleep fold moves the robot outside goto_target's stash —
            # forget BEFORE dispatching (same rationale as wake_up).
            self._forget_commanded()
            self._mini.goto_sleep()
            self._update_state(is_awake=False)
            logger.info("Reachy Mini asleep: %s", self._robot_name)
            return True

        except Exception as e:
            logger.error("Sleep command failed: %s", e)
            return False

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
