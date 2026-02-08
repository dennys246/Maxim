"""Unit tests for RobotController abstraction."""

from __future__ import annotations

import pytest

from maxim.hardware import (
    MotionTarget,
    PixelTarget,
    RobotCapabilities,
    RobotConnectionState,
    RobotState,
)
from maxim.hardware.simulation import SimulatedController


class TestMotionTarget:
    """Test MotionTarget dataclass."""

    def test_default_values(self):
        """Default values are None except duration and method."""
        target = MotionTarget()

        assert target.head_roll is None
        assert target.head_pitch is None
        assert target.head_yaw is None
        assert target.body_yaw is None
        assert target.duration == 1.0
        assert target.method == "minimum_jerk"

    def test_partial_specification(self):
        """Can specify only some values."""
        target = MotionTarget(head_pitch=0.3, duration=0.5)

        assert target.head_roll is None
        assert target.head_pitch == 0.3
        assert target.head_yaw is None
        assert target.duration == 0.5


class TestPixelTarget:
    """Test PixelTarget dataclass."""

    def test_creation(self):
        """Can create with u, v coordinates."""
        target = PixelTarget(u=320, v=240)

        assert target.u == 320
        assert target.v == 240
        assert target.duration == 0.5

    def test_with_duration(self):
        """Can specify custom duration."""
        target = PixelTarget(u=100, v=200, duration=1.0)

        assert target.duration == 1.0


class TestRobotState:
    """Test RobotState dataclass."""

    def test_default_state(self):
        """Default state is disconnected."""
        state = RobotState()

        assert state.connection_state == RobotConnectionState.DISCONNECTED
        assert not state.is_awake
        assert not state.is_recording
        assert not state.is_connected
        assert not state.is_operational

    def test_is_operational(self):
        """is_operational requires connected and awake."""
        state = RobotState()

        # Not connected
        assert not state.is_operational

        # Connected but not awake
        state.connection_state = RobotConnectionState.CONNECTED
        assert state.is_connected
        assert not state.is_operational

        # Connected and awake
        state.is_awake = True
        assert state.is_operational


class TestSimulatedController:
    """Test SimulatedController implementation."""

    def test_creation(self):
        """Can create SimulatedController."""
        controller = SimulatedController()

        assert controller.robot_id == "simulated"
        assert controller.robot_type == "simulated"
        assert not controller.is_connected()

    def test_custom_robot_id(self):
        """Can create with custom robot_id."""
        controller = SimulatedController(robot_id="test_robot")

        assert controller.robot_id == "test_robot"

    def test_connect_disconnect(self):
        """Can connect and disconnect."""
        controller = SimulatedController()

        assert controller.connect()
        assert controller.is_connected()
        assert controller.capabilities is not None
        assert controller.capabilities.robot_type == "simulated"

        controller.disconnect()
        assert not controller.is_connected()

    def test_wake_sleep_cycle(self):
        """Can wake up and go to sleep."""
        controller = SimulatedController()
        controller.connect()

        assert not controller.state.is_awake

        assert controller.wake_up()
        assert controller.state.is_awake
        assert controller.state.is_operational

        assert controller.goto_sleep()
        assert not controller.state.is_awake

        controller.disconnect()

    def test_motion_target(self):
        """Can execute motion targets."""
        controller = SimulatedController()
        controller.connect()
        controller.wake_up()

        target = MotionTarget(head_pitch=0.3, head_yaw=-0.2, duration=0.5)
        assert controller.goto_target(target)

        pose = controller.get_current_pose()
        assert pose["pitch"] == 0.3
        assert pose["yaw"] == -0.2

        controller.disconnect()

    def test_look_at_pixel(self):
        """Can look at pixel coordinates."""
        controller = SimulatedController()
        controller.connect()
        controller.wake_up()

        target = PixelTarget(u=320, v=240)  # Center
        assert controller.look_at_pixel(target)

        pose = controller.get_current_pose()
        # Center should result in approximately zero angles
        assert abs(pose["pitch"]) < 0.1
        assert abs(pose["yaw"]) < 0.1

        controller.disconnect()

    def test_center_vision(self):
        """Can center vision."""
        controller = SimulatedController()
        controller.connect()
        controller.wake_up()

        # First move to non-center
        controller.goto_target(MotionTarget(head_pitch=0.5, head_yaw=0.5))
        pose = controller.get_current_pose()
        assert pose["pitch"] == 0.5

        # Center vision
        assert controller.center_vision()
        pose = controller.get_current_pose()
        assert pose["pitch"] == 0.0
        assert pose["yaw"] == 0.0

        controller.disconnect()

    def test_video_stream(self):
        """Can get video stream and frames."""
        controller = SimulatedController()
        controller.connect()
        controller.start_recording()

        video = controller.get_video_stream()
        assert video is not None
        assert video.is_active

        frame = video.get_frame_nonblocking()
        assert frame is not None
        assert frame.shape == (480, 640, 3)  # Default resolution

        controller.stop_recording()
        assert not video.is_active
        controller.disconnect()

    def test_audio_stream(self):
        """Can get audio stream and samples."""
        controller = SimulatedController()
        controller.connect()
        controller.start_recording()

        audio = controller.get_audio_stream()
        assert audio is not None
        assert audio.is_active
        assert audio.input_sample_rate == 16000
        assert audio.output_sample_rate == 16000

        controller.stop_recording()
        controller.disconnect()

    def test_context_manager(self):
        """Can use as context manager."""
        with SimulatedController() as controller:
            assert controller.is_connected()
            assert controller.state.is_awake

        assert not controller.is_connected()

    def test_repr(self):
        """Has useful string representation."""
        controller = SimulatedController(robot_id="test")
        repr_str = repr(controller)

        assert "SimulatedController" in repr_str
        assert "test" in repr_str
        assert "simulated" in repr_str

    def test_not_connected_operations_fail(self):
        """Operations fail when not connected."""
        controller = SimulatedController()

        assert not controller.wake_up()
        assert not controller.goto_target(MotionTarget())
        assert not controller.look_at_pixel(PixelTarget(u=0, v=0))
        assert controller.get_video_stream() is None
        assert controller.get_audio_stream() is None


class TestRobotCapabilities:
    """Test RobotCapabilities dataclass."""

    def test_simulated_capabilities(self):
        """SimulatedController has expected capabilities."""
        controller = SimulatedController()
        controller.connect()

        caps = controller.capabilities
        assert caps is not None
        assert caps.robot_type == "simulated"
        assert caps.has_video()
        assert caps.has_audio()
        assert caps.video_resolution == (640, 480)

        controller.disconnect()
