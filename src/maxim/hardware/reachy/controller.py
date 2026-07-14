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
            connection_mode: SDK connection mode — ``"network"`` (zenoh multicast,
                default, current behavior), ``"localhost_only"`` (tcp/localhost:7447 —
                use with a tunnel or when running onboard), or ``"auto"``.
            host: Robot IP. Set it to (a) BYPASS the mDNS pre-check (which hard-fails
                on setups where ``reachy-mini.local`` doesn't resolve — macOS/hotspot),
                using a direct ``host:7447`` reachability probe instead, and (b) be the
                SSH tunnel target. In ``"network"`` mode the SDK still discovers via
                multicast (it has no explicit-IP option); ``host`` only replaces the
                reachability pre-check there. Explicit-IP connect requires ``tunnel``.
            tunnel: If True (requires ``host``), auto-start
                ``ssh -N -L 7447:127.0.0.1:7447 <ssh_user>@<host>`` and force
                ``connection_mode="localhost_only"`` — bypasses macOS multicast/mDNS
                entirely. Needs key-based SSH (BatchMode); else start the tunnel
                manually and set ``tunnel: false, connection_mode: localhost_only``.
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

    def _wait_local_port(self, port: int, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._port_open("127.0.0.1", port, timeout=1.0):
                return True
            time.sleep(0.3)
        return False

    def _start_ssh_tunnel(self, timeout: float = 10.0) -> bool:
        """Auto-start ``ssh -N -L 7447:127.0.0.1:7447 user@host`` (key-based / BatchMode)."""
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
            "7447:127.0.0.1:7447",
            f"{self._ssh_user}@{self._host}",
        ]
        logger.info("Starting SSH tunnel: %s", " ".join(cmd))
        try:
            self._tunnel_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            logger.error("ssh not found — cannot auto-tunnel")
            return False
        if self._wait_local_port(7447, timeout=timeout):
            logger.info("SSH tunnel up: localhost:7447 -> %s:7447", self._host)
            return True
        logger.error(
            "SSH tunnel to %s did not open localhost:7447. Auto-tunnel needs key-based SSH "
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

            effective_mode = self._connection_mode

            # Optional SSH tunnel: localhost:7447 -> robot:7447. Lets us reach a robot
            # whose zenoh isn't network-exposed, or where macOS multicast/mDNS fails.
            if self._tunnel:
                if not self._host:
                    logger.error("tunnel=True requires host= (robot IP / SSH target)")
                    self._update_state(connection_state=RobotConnectionState.DISCONNECTED)
                    return False
                if not self._start_ssh_tunnel(timeout=min(timeout, 10.0)):
                    self._update_state(connection_state=RobotConnectionState.DISCONNECTED)
                    return False
                effective_mode = "localhost_only"

            # Reachability pre-check (fast-fail so we don't wait ~25s on the SDK).
            # Legacy default (network + no host + no tunnel) keeps the mDNS gate; an
            # explicit host / tunnel uses a direct TCP probe of :7447 instead (the mDNS
            # gate hard-fails where reachy-mini.local doesn't resolve — macOS/hotspot).
            if effective_mode == "network" and self._host is None:
                if self._resolve_mdns(timeout=min(timeout, 5.0)) is None:
                    logger.warning(
                        "Could not resolve %s via mDNS — robot not reachable, skipping SDK connection",
                        self._robot_name,
                    )
                    self._update_state(connection_state=RobotConnectionState.DISCONNECTED)
                    return False
            else:
                probe_host = "127.0.0.1" if effective_mode == "localhost_only" else self._host
                if probe_host and not self._port_open(probe_host, 7447, timeout=min(timeout, 5.0)):
                    logger.warning(
                        "zenoh not reachable at %s:7447 — skipping SDK connection "
                        "(check host/tunnel; on macOS grant Local Network permission)",
                        probe_host,
                    )
                    self._stop_ssh_tunnel()
                    self._update_state(connection_state=RobotConnectionState.DISCONNECTED)
                    return False

            logger.info("Connecting to Reachy Mini: %s (mode=%s)", self._robot_name, effective_mode)

            self._mini = ReachyMini(
                robot_name=self._robot_name,
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
            # Build head target (roll, pitch, yaw)
            head_target = None
            if any(x is not None for x in [target.head_roll, target.head_pitch, target.head_yaw]):
                # Get current pose to fill in None values
                current = self.get_current_pose()
                head_target = (
                    target.head_roll if target.head_roll is not None else current.get("roll", 0.0),
                    target.head_pitch if target.head_pitch is not None else current.get("pitch", 0.0),
                    target.head_yaw if target.head_yaw is not None else current.get("yaw", 0.0),
                )

            # Build body yaw target
            body_yaw_target = target.body_yaw

            self._mini.goto_target(
                head=head_target,
                body_yaw=body_yaw_target,
                duration=target.duration,
                method=target.method,
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
            Dict mapping joint names to current angles.
        """
        if self._mini is None:
            return {}

        try:
            head_joints, antenna_joints = self._mini.get_current_joint_positions()

            pose = {}

            # Head joints: (roll, pitch, yaw)
            if head_joints is not None and len(head_joints) >= 3:
                pose["roll"] = float(head_joints[0])
                pose["pitch"] = float(head_joints[1])
                pose["yaw"] = float(head_joints[2])

            # Antenna joints: (left, right)
            if antenna_joints is not None and len(antenna_joints) >= 2:
                pose["antenna_left"] = float(antenna_joints[0])
                pose["antenna_right"] = float(antenna_joints[1])

            self._update_state(current_pose=pose)
            return pose

        except Exception as e:
            logger.debug("Failed to get pose: %s", e)
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def wake_up(self) -> bool:
        """Wake up the robot (enable motors).

        Returns:
            True if wake-up successful.
        """
        if self._mini is None or not self.is_connected():
            return False

        try:
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
