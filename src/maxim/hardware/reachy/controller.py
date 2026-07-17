"""Reachy Mini controller implementation.

Wraps the Reachy Mini SDK to provide a RobotController interface.
"""

from __future__ import annotations

import dataclasses
import logging
import socket
import subprocess
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
        """
        super().__init__(robot_id or robot_name)
        self._robot_name = robot_name
        self._media_backend = media_backend
        self._connection_mode = connection_mode
        self._host = host
        self._tunnel = tunnel
        self._ssh_user = ssh_user
        self._ssh_port = ssh_port
        self._tunnel_proc: subprocess.Popen | None = None
        self._mini: Any = None  # ReachyMini SDK instance
        self._video_stream: ReachyVideoStream | None = None
        self._audio_stream: ReachyAudioStream | None = None

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
            self._mini = ReachyMini(
                robot_name=self._robot_name,
                host=connect_host,
                port=8000,
                connection_mode=effective_mode,
                media_backend=self._media_backend,
                timeout=timeout,
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

        except Exception as e:
            logger.error("Error during disconnect: %s", e)

        finally:
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
            body_yaw_target = target.body_yaw
            head_requested = any(x is not None for x in [target.head_roll, target.head_pitch, target.head_yaw])

            head_target = None
            if head_requested or body_yaw_target is not None:
                from reachy_mini.utils import create_head_pose

                # Fill unspecified axes from the current pose so single-axis
                # commands don't recenter the others.
                current = self.get_current_pose()
                current_body = current.get("body_yaw", 0.0) or 0.0
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
                else:
                    relative_yaw = current.get("yaw", 0.0) - current_body
                target_body = body_yaw_target if body_yaw_target is not None else current_body
                head_target = create_head_pose(
                    roll=(target.head_roll if target.head_roll is not None else current.get("roll", 0.0)),
                    pitch=(target.head_pitch if target.head_pitch is not None else current.get("pitch", 0.0)),
                    yaw=relative_yaw + target_body,
                    degrees=False,
                )

            self._mini.goto_target(
                head=head_target,
                body_yaw=body_yaw_target,
                duration=target.duration,
                method=self._sdk_method(target.method),
            )

            return True

        except Exception as e:
            logger.error("Motion command failed: %s", e)
            return False

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
            self._mini.wake_up()
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
