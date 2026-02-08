"""Hardware abstraction layer for multi-robot support.

This package provides hardware-agnostic interfaces for controlling
robots and accessing their data streams.

Key Components:
    - RobotController: Abstract base class for robot control
    - RobotRegistry: Singleton for managing multiple robot connections
    - VideoStream/AudioStream: Abstract data stream interfaces
    - RobotCapabilities/RobotState: Robot metadata and state

Implementations:
    - ReachyMiniController: Reachy Mini robot support
    - SimulatedController: Mock robot for testing

Example Usage:
    from maxim.hardware import RobotRegistry
    from maxim.hardware.reachy import ReachyMiniController
    from maxim.hardware.simulation import SimulatedController

    # Setup registry
    registry = RobotRegistry()
    registry.register_controller_type("reachy_mini", ReachyMiniController)
    registry.register_controller_type("simulated", SimulatedController)

    # Connect robots
    registry.connect_robot(
        robot_id="primary",
        robot_type="reachy_mini",
        config={"robot_name": "reachy_mini"},
        set_primary=True,
    )

    # Use the primary robot
    robot = registry.primary
    robot.wake_up()
    video = robot.get_video_stream()
    frame = video.get_frame()

    # Cleanup
    registry.disconnect_all()
"""

from maxim.hardware.capabilities import (
    MotionCapability,
    RobotCapabilities,
    RobotConnectionState,
    RobotState,
    StreamCapability,
)
from maxim.hardware.controller import (
    MotionTarget,
    PixelTarget,
    RobotController,
)
from maxim.hardware.registry import RobotRegistry
from maxim.hardware.streams import AudioStream, SensorStream, VideoStream
from maxim.hardware.config import (
    RobotConfig,
    RobotsConfig,
    load_robots_config,
    connect_configured_robots,
    setup_robots_from_config,
)

__all__ = [
    # Controller
    "RobotController",
    "MotionTarget",
    "PixelTarget",
    # Registry
    "RobotRegistry",
    # Streams
    "VideoStream",
    "AudioStream",
    "SensorStream",
    # Capabilities
    "RobotCapabilities",
    "RobotState",
    "RobotConnectionState",
    "MotionCapability",
    "StreamCapability",
    # Configuration
    "RobotConfig",
    "RobotsConfig",
    "load_robots_config",
    "connect_configured_robots",
    "setup_robots_from_config",
]
