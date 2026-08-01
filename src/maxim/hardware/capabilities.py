"""Robot capabilities and state dataclasses.

These dataclasses define what a robot can do (capabilities)
and its current operational state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class RobotConnectionState(Enum):
    """Connection state of a robot."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()


class MotionCapability(Enum):
    """Generic motion categories common across humanoid + quadruped robots.

    Robot-specific capabilities (Reachy antennas, Spot legs, hand gestures)
    are advertised via :attr:`RobotCapabilities.custom` so this enum doesn't
    grow per-robot entries. The agent loop checks the generic categories;
    specific tools check ``custom`` strings.
    """

    HEAD_PAN = auto()  # Left/right head rotation
    HEAD_TILT = auto()  # Up/down head rotation
    HEAD_ROLL = auto()  # Head roll (ear to shoulder)
    BODY_YAW = auto()  # Body rotation
    ARMS = auto()  # Arm movement (humanoid)
    GRIPPER = auto()  # Hand/gripper open/close
    WHEELS = auto()  # Mobile base (wheeled)
    LEGS = auto()  # Legged locomotion (quadruped/biped)


class StreamCapability(Enum):
    """Types of data streams a robot can provide."""

    VIDEO = auto()  # Camera feed
    AUDIO_INPUT = auto()  # Microphone
    AUDIO_OUTPUT = auto()  # Speaker
    DEPTH = auto()  # Depth camera (future)
    LIDAR = auto()  # LIDAR sensor (future)


@dataclass(frozen=True, slots=True)
class RobotCapabilities:
    """Immutable description of what a robot can do.

    This is set at connection time and doesn't change during runtime.
    Use this to check if a robot supports specific features before
    attempting to use them.

    Robot-specific capabilities that don't fit the generic
    :class:`MotionCapability` / :class:`StreamCapability` enums (e.g.
    Reachy antennas, Atlas precision grip, Spot LIDAR-equipped legs)
    are advertised via :attr:`custom` as freeform strings. This keeps
    the canonical enums small and lets new robot plugins add features
    without modifying the core. Example::

        REACHY_CAPS = RobotCapabilities(
            robot_type="reachy_mini",
            ...,
            motion=frozenset({MotionCapability.HEAD_PAN, ...}),
            custom=frozenset({"antenna_left", "antenna_right"}),
        )
    """

    robot_type: str
    robot_id: str

    # Motion capabilities
    motion: frozenset[MotionCapability] = field(default_factory=frozenset)

    # Stream capabilities
    streams: frozenset[StreamCapability] = field(default_factory=frozenset)

    # Robot-specific extension capabilities (free-form strings).
    # Used for joints / sensors / behaviours that don't fit the generic
    # enums above. Tool implementations check these to gate features.
    custom: frozenset[str] = field(default_factory=frozenset)

    # Video resolution (width, height) if VIDEO capability present
    video_resolution: tuple[int, int] | None = None

    # Audio sample rates if AUDIO capabilities present
    audio_input_rate: int | None = None
    audio_output_rate: int | None = None

    # Motion limits (joint ranges, speeds, etc.)
    motion_limits: dict[str, Any] = field(default_factory=dict)

    # Firmware/SDK version info
    firmware_version: str | None = None
    sdk_version: str | None = None

    def has_motion(self, capability: MotionCapability) -> bool:
        """Check if robot supports a motion capability."""
        return capability in self.motion

    def has_stream(self, capability: StreamCapability) -> bool:
        """Check if robot supports a stream capability."""
        return capability in self.streams

    def has_custom(self, capability: str) -> bool:
        """Check if robot supports a custom (robot-specific) capability."""
        return capability in self.custom

    def has_video(self) -> bool:
        """Convenience check for video capability."""
        return StreamCapability.VIDEO in self.streams

    def has_audio(self) -> bool:
        """Convenience check for audio I/O capability."""
        return StreamCapability.AUDIO_INPUT in self.streams or StreamCapability.AUDIO_OUTPUT in self.streams


@dataclass(slots=True)
class RobotState:
    """Mutable runtime state of a robot.

    Updated as the robot operates. Use this to check current
    operational status before sending commands.
    """

    connection_state: RobotConnectionState = RobotConnectionState.DISCONNECTED

    # Is the robot awake and ready for commands?
    is_awake: bool = False

    # Is recording/streaming active?
    is_recording: bool = False

    # Current pose (joint positions, varies by robot type)
    current_pose: dict[str, float] = field(default_factory=dict)

    # Error message if connection_state is ERROR
    error_message: str | None = None

    # Last successful communication timestamp
    last_heartbeat: float | None = None

    # Connection attempt count (for reconnection logic)
    connection_attempts: int = 0

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.connection_state == RobotConnectionState.CONNECTED

    @property
    def is_operational(self) -> bool:
        """Check if robot is connected and awake."""
        return self.is_connected and self.is_awake


# Default capabilities for known robot types
REACHY_MINI_CAPABILITIES = RobotCapabilities(
    robot_type="reachy_mini",
    robot_id="",  # Set at connection time
    motion=frozenset(
        {
            MotionCapability.HEAD_PAN,
            MotionCapability.HEAD_TILT,
            MotionCapability.HEAD_ROLL,
            MotionCapability.BODY_YAW,
        }
    ),
    streams=frozenset(
        {
            StreamCapability.VIDEO,
            StreamCapability.AUDIO_INPUT,
            StreamCapability.AUDIO_OUTPUT,
        }
    ),
    # Reachy-specific joints + the XVF3800 DoA capability (Stage 1c —
    # "audio_localization" advertises that get_doa_reader() yields a reader;
    # the zero-schema-change channel, same as the antenna joints).
    custom=frozenset({"antenna_left", "antenna_right", "audio_localization"}),
    video_resolution=(640, 480),  # Default, may vary
    audio_input_rate=48000,
    audio_output_rate=48000,
)

SIMULATED_CAPABILITIES = RobotCapabilities(
    robot_type="simulated",
    robot_id="",
    motion=frozenset(
        {
            MotionCapability.HEAD_PAN,
            MotionCapability.HEAD_TILT,
            MotionCapability.HEAD_ROLL,
            MotionCapability.BODY_YAW,
        }
    ),
    streams=frozenset(
        {
            StreamCapability.VIDEO,
            StreamCapability.AUDIO_INPUT,
            StreamCapability.AUDIO_OUTPUT,
        }
    ),
    video_resolution=(640, 480),
    audio_input_rate=16000,
    audio_output_rate=16000,
)
