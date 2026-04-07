"""Simulated robot controller for testing.

Provides a fully functional RobotController that simulates
robot behavior for testing without hardware.
"""

from __future__ import annotations

import dataclasses
import logging
import time

from maxim.hardware.capabilities import (
    SIMULATED_CAPABILITIES,
    RobotConnectionState,
)
from maxim.hardware.controller import MotionTarget, PixelTarget, RobotController
from maxim.hardware.simulation.streams import MockAudioStream, MockVideoStream
from maxim.hardware.streams import AudioStream, VideoStream

logger = logging.getLogger(__name__)


class SimulatedController(RobotController):
    """Simulated robot controller for testing.

    Provides a complete RobotController implementation that
    simulates robot behavior without requiring hardware.
    Useful for:
    - Unit testing
    - Development without hardware
    - CI/CD pipelines
    - Demonstration

    All operations succeed instantly (simulated).
    Motion commands update the internal pose state.
    Streams provide synthetic data.
    """

    def __init__(
        self,
        robot_id: str | None = None,
        *,
        video_resolution: tuple[int, int] = (640, 480),
        video_fps: float = 10.0,
        audio_sample_rate: int = 16000,
        simulate_delays: bool = False,
    ) -> None:
        """Initialize simulated controller.

        Args:
            robot_id: Unique identifier (defaults to "simulated").
            video_resolution: Resolution for mock video stream.
            video_fps: FPS for mock video stream.
            audio_sample_rate: Sample rate for mock audio stream.
            simulate_delays: If True, add realistic delays to operations.
        """
        super().__init__(robot_id or "simulated")
        self._video_resolution = video_resolution
        self._video_fps = video_fps
        self._audio_sample_rate = audio_sample_rate
        self._simulate_delays = simulate_delays

        self._video_stream: MockVideoStream | None = None
        self._audio_stream: MockAudioStream | None = None

        # Simulated pose state
        self._current_pose: dict[str, float] = {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "body_yaw": 0.0,
        }

    @property
    def robot_type(self) -> str:
        """Get the robot type identifier."""
        return "simulated"

    # ─────────────────────────────────────────────────────────────────────────
    # Connection Management
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self, timeout: float = 30.0) -> bool:
        """Connect to the simulated robot (always succeeds).

        Args:
            timeout: Ignored in simulation.

        Returns:
            True.
        """
        self._update_state(connection_state=RobotConnectionState.CONNECTING)

        if self._simulate_delays:
            time.sleep(0.1)

        # Create stream wrappers
        self._video_stream = MockVideoStream(
            resolution=self._video_resolution,
            fps=self._video_fps,
        )
        self._audio_stream = MockAudioStream(
            input_sample_rate=self._audio_sample_rate,
            output_sample_rate=self._audio_sample_rate,
        )

        # Build capabilities
        self._capabilities = dataclasses.replace(
            SIMULATED_CAPABILITIES,
            robot_id=self._robot_id,
            video_resolution=self._video_resolution,
            audio_input_rate=self._audio_sample_rate,
            audio_output_rate=self._audio_sample_rate,
        )

        self._update_state(
            connection_state=RobotConnectionState.CONNECTED,
            last_heartbeat=time.time(),
        )

        logger.info("Simulated robot connected: %s", self._robot_id)
        return True

    def disconnect(self) -> None:
        """Disconnect from the simulated robot."""
        if self._video_stream:
            self._video_stream.stop()
        if self._audio_stream:
            self._audio_stream.stop()

        self._video_stream = None
        self._audio_stream = None

        self._update_state(
            connection_state=RobotConnectionState.DISCONNECTED,
            is_awake=False,
            is_recording=False,
        )

        logger.info("Simulated robot disconnected: %s", self._robot_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Motion Control
    # ─────────────────────────────────────────────────────────────────────────

    def goto_target(self, target: MotionTarget) -> bool:
        """Move robot to target pose (simulated).

        Updates internal pose state to reflect the motion.

        Args:
            target: Target pose specification.

        Returns:
            True.
        """
        if not self.is_connected():
            return False

        # Update pose state
        if target.head_roll is not None:
            self._current_pose["roll"] = target.head_roll
        if target.head_pitch is not None:
            self._current_pose["pitch"] = target.head_pitch
        if target.head_yaw is not None:
            self._current_pose["yaw"] = target.head_yaw
        if target.body_yaw is not None:
            self._current_pose["body_yaw"] = target.body_yaw

        if self._simulate_delays:
            time.sleep(target.duration * 0.1)  # 10% of actual duration

        self._update_state(current_pose=self._current_pose.copy())

        logger.debug("Simulated motion to: %s", self._current_pose)
        return True

    def look_at_pixel(self, target: PixelTarget) -> bool:
        """Move head to look at a pixel coordinate (simulated).

        Converts pixel to approximate head angles.

        Args:
            target: Target pixel coordinates.

        Returns:
            True.
        """
        if not self.is_connected():
            return False

        # Simple conversion: center of frame = (0, 0)
        # Assume 640x480 resolution, 60 degree FOV
        width, height = self._video_resolution
        center_x, center_y = width / 2, height / 2

        # Convert to approximate angles (radians)
        fov_h = 1.0  # ~60 degrees horizontal
        fov_v = 0.75  # ~45 degrees vertical

        yaw = -((target.u - center_x) / center_x) * (fov_h / 2)
        pitch = ((target.v - center_y) / center_y) * (fov_v / 2)

        self._current_pose["yaw"] = yaw
        self._current_pose["pitch"] = pitch

        if self._simulate_delays:
            time.sleep(target.duration * 0.1)

        self._update_state(current_pose=self._current_pose.copy())

        logger.debug("Simulated look at pixel (%d, %d) -> pitch=%.2f, yaw=%.2f", target.u, target.v, pitch, yaw)
        return True

    def get_current_pose(self) -> dict[str, float]:
        """Get current joint positions.

        Returns:
            Dict with simulated pose values.
        """
        return self._current_pose.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def wake_up(self) -> bool:
        """Wake up the robot (simulated).

        Returns:
            True.
        """
        if not self.is_connected():
            return False

        if self._simulate_delays:
            time.sleep(0.1)

        self._update_state(is_awake=True)
        logger.info("Simulated robot awake: %s", self._robot_id)
        return True

    def goto_sleep(self) -> bool:
        """Put the robot to sleep (simulated).

        Returns:
            True.
        """
        if self._simulate_delays:
            time.sleep(0.1)

        self._update_state(is_awake=False)
        logger.info("Simulated robot asleep: %s", self._robot_id)
        return True

    def start_recording(self) -> bool:
        """Start media recording/streaming.

        Returns:
            True.
        """
        if not self.is_connected():
            return False

        if self._video_stream:
            self._video_stream.start()
        if self._audio_stream:
            self._audio_stream.start()

        self._update_state(is_recording=True)
        logger.info("Simulated recording started: %s", self._robot_id)
        return True

    def stop_recording(self) -> bool:
        """Stop media recording/streaming.

        Returns:
            True.
        """
        if self._video_stream:
            self._video_stream.stop()
        if self._audio_stream:
            self._audio_stream.stop()

        self._update_state(is_recording=False)
        logger.info("Simulated recording stopped: %s", self._robot_id)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Data Streams
    # ─────────────────────────────────────────────────────────────────────────

    def get_video_stream(self) -> VideoStream | None:
        """Get the mock video stream.

        Returns:
            MockVideoStream, or None if not connected.
        """
        return self._video_stream

    def get_audio_stream(self) -> AudioStream | None:
        """Get the mock audio stream.

        Returns:
            MockAudioStream, or None if not connected.
        """
        return self._audio_stream
