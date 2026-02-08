"""Robot controller abstraction.

The RobotController ABC defines the interface for controlling
any robot hardware. Implementations wrap specific SDKs
(Reachy, simulation, future robots).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from maxim.hardware.capabilities import (
    RobotCapabilities,
    RobotConnectionState,
    RobotState,
)
from maxim.hardware.streams import AudioStream, VideoStream

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MotionTarget:
    """Target pose for robot motion.

    All angles are in radians. None values mean "don't change".
    """

    # Head pose (3-DOF)
    head_roll: float | None = None
    head_pitch: float | None = None
    head_yaw: float | None = None

    # Body rotation
    body_yaw: float | None = None

    # Antenna positions (Reachy-specific, left/right angle)
    antenna_left: float | None = None
    antenna_right: float | None = None

    # Motion duration in seconds
    duration: float = 1.0

    # Motion method (e.g., "linear", "minimum_jerk")
    method: str = "minimum_jerk"


@dataclass(slots=True)
class PixelTarget:
    """Target pixel coordinates for look_at operations."""

    u: int  # X coordinate (0 = left)
    v: int  # Y coordinate (0 = top)
    duration: float = 0.5


class RobotController(ABC):
    """Abstract base class for robot controllers.

    Implementations wrap specific robot SDKs and provide a
    unified interface for:
    - Connection management
    - Motion control
    - Lifecycle (wake/sleep)
    - Data stream access

    The controller is responsible for maintaining connection
    state and providing access to video/audio streams.

    Usage:
        controller = ReachyMiniController(robot_name="reachy_mini")
        if controller.connect(timeout=30.0):
            controller.wake_up()
            video = controller.get_video_stream()
            frame = video.get_frame()
            controller.goto_target(MotionTarget(head_pitch=0.2))
            controller.goto_sleep()
            controller.disconnect()
    """

    def __init__(self, robot_id: str) -> None:
        """Initialize controller.

        Args:
            robot_id: Unique identifier for this robot instance.
        """
        self._robot_id = robot_id
        self._state = RobotState()
        self._capabilities: RobotCapabilities | None = None

    @property
    def robot_id(self) -> str:
        """Get the robot's unique identifier."""
        return self._robot_id

    @property
    @abstractmethod
    def robot_type(self) -> str:
        """Get the robot type identifier (e.g., 'reachy_mini', 'simulated')."""
        ...

    @property
    def capabilities(self) -> RobotCapabilities | None:
        """Get the robot's capabilities (None if not connected)."""
        return self._capabilities

    @property
    def state(self) -> RobotState:
        """Get the robot's current state."""
        return self._state

    def _update_state(self, **kwargs: Any) -> None:
        """Update state fields."""
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)

    # ─────────────────────────────────────────────────────────────────────────
    # Connection Management
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self, timeout: float = 30.0) -> bool:
        """Connect to the robot.

        Args:
            timeout: Connection timeout in seconds.

        Returns:
            True if connection successful, False otherwise.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot."""
        ...

    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._state.is_connected

    def reconnect(self, timeout: float = 30.0, max_attempts: int = 3) -> bool:
        """Attempt to reconnect to the robot.

        Args:
            timeout: Connection timeout per attempt.
            max_attempts: Maximum number of reconnection attempts.

        Returns:
            True if reconnection successful.
        """
        self._update_state(connection_state=RobotConnectionState.RECONNECTING)

        for attempt in range(max_attempts):
            self._update_state(connection_attempts=attempt + 1)
            logger.info(
                "Reconnection attempt %d/%d for %s",
                attempt + 1,
                max_attempts,
                self._robot_id,
            )

            self.disconnect()
            if self.connect(timeout=timeout):
                return True

        self._update_state(
            connection_state=RobotConnectionState.ERROR,
            error_message=f"Failed to reconnect after {max_attempts} attempts",
        )
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Motion Control
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def goto_target(self, target: MotionTarget) -> bool:
        """Move robot to target pose.

        Args:
            target: Target pose specification.

        Returns:
            True if motion command accepted.
        """
        ...

    @abstractmethod
    def look_at_pixel(self, target: PixelTarget) -> bool:
        """Move head to look at a pixel coordinate.

        Uses the robot's camera model to convert pixel
        coordinates to head angles.

        Args:
            target: Target pixel coordinates.

        Returns:
            True if motion command accepted.
        """
        ...

    @abstractmethod
    def get_current_pose(self) -> dict[str, float]:
        """Get current joint positions.

        Returns:
            Dict mapping joint names to current angles (radians).
        """
        ...

    def center_vision(self, duration: float = 0.5) -> bool:
        """Center the head to look straight ahead.

        Returns:
            True if motion command accepted.
        """
        return self.goto_target(
            MotionTarget(
                head_roll=0.0,
                head_pitch=0.0,
                head_yaw=0.0,
                body_yaw=0.0,
                duration=duration,
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def wake_up(self) -> bool:
        """Wake up the robot (enable motors, start streaming).

        Returns:
            True if wake-up successful.
        """
        ...

    @abstractmethod
    def goto_sleep(self) -> bool:
        """Put the robot to sleep (disable motors, stop streaming).

        Returns:
            True if sleep command successful.
        """
        ...

    @abstractmethod
    def start_recording(self) -> bool:
        """Start media recording/streaming.

        Returns:
            True if recording started successfully.
        """
        ...

    @abstractmethod
    def stop_recording(self) -> bool:
        """Stop media recording/streaming.

        Returns:
            True if recording stopped successfully.
        """
        ...

    # ─────────────────────────────────────────────────────────────────────────
    # Data Streams
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_video_stream(self) -> VideoStream | None:
        """Get the video stream for this robot.

        Returns:
            VideoStream implementation, or None if not available.
        """
        ...

    @abstractmethod
    def get_audio_stream(self) -> AudioStream | None:
        """Get the audio stream for this robot.

        Returns:
            AudioStream implementation, or None if not available.
        """
        ...

    # ─────────────────────────────────────────────────────────────────────────
    # Context Manager
    # ─────────────────────────────────────────────────────────────────────────

    def __enter__(self) -> RobotController:
        """Context manager entry - connect and wake up."""
        if self.connect():
            self.wake_up()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - sleep and disconnect."""
        try:
            self.goto_sleep()
        finally:
            self.disconnect()

    # ─────────────────────────────────────────────────────────────────────────
    # String Representation
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"robot_id={self._robot_id!r}, "
            f"robot_type={self.robot_type!r}, "
            f"connected={self.is_connected()})"
        )
