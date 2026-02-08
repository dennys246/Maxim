"""Data stream abstractions for robot sensors.

These ABCs define the interface for video and audio streams,
allowing hardware-agnostic access to sensor data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VideoStream(ABC):
    """Abstract base class for video streams.

    Implementations provide access to camera frames from
    different robot hardware or simulated sources.
    """

    @abstractmethod
    def get_frame(self, timeout: float = 1.0) -> NDArray[np.uint8] | None:
        """Get the next video frame (blocking).

        Args:
            timeout: Maximum time to wait for a frame in seconds.

        Returns:
            BGR image as numpy array (H, W, 3), or None if no frame available.
        """
        ...

    @abstractmethod
    def get_frame_nonblocking(self) -> NDArray[np.uint8] | None:
        """Get the latest frame without blocking.

        Returns:
            BGR image as numpy array (H, W, 3), or None if no frame available.
        """
        ...

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """Get the video resolution (width, height)."""
        ...

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Check if the stream is currently active."""
        ...

    def start(self) -> None:
        """Start the video stream (optional, may be auto-started)."""
        pass

    def stop(self) -> None:
        """Stop the video stream."""
        pass


class AudioStream(ABC):
    """Abstract base class for audio streams.

    Implementations provide access to microphone input and
    speaker output from different robot hardware or simulated sources.
    """

    @abstractmethod
    def get_sample(self, timeout: float = 0.5) -> NDArray[np.float32] | None:
        """Get the next audio sample (blocking).

        Args:
            timeout: Maximum time to wait for a sample in seconds.

        Returns:
            Audio samples as float32 numpy array, or None if not available.
        """
        ...

    @abstractmethod
    def push_sample(self, samples: NDArray[np.float32]) -> bool:
        """Push audio samples to the speaker.

        Args:
            samples: Audio samples as float32 numpy array.

        Returns:
            True if samples were successfully queued for playback.
        """
        ...

    @property
    @abstractmethod
    def input_sample_rate(self) -> int:
        """Get the input (microphone) sample rate in Hz."""
        ...

    @property
    @abstractmethod
    def output_sample_rate(self) -> int:
        """Get the output (speaker) sample rate in Hz."""
        ...

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Check if the stream is currently active."""
        ...

    def start(self) -> None:
        """Start the audio stream (optional, may be auto-started)."""
        pass

    def stop(self) -> None:
        """Stop the audio stream."""
        pass


class SensorStream(ABC):
    """Abstract base class for future sensor streams.

    Placeholder for depth cameras, LIDAR, IMU, etc.
    To be implemented when hardware support is added.
    """

    @property
    @abstractmethod
    def sensor_type(self) -> str:
        """Get the sensor type identifier."""
        ...

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Check if the stream is currently active."""
        ...

    @abstractmethod
    def get_data(self, timeout: float = 1.0) -> dict | None:
        """Get sensor data.

        Returns:
            Sensor-specific data dict, or None if not available.
        """
        ...

    def start(self) -> None:
        """Start the sensor stream."""
        pass

    def stop(self) -> None:
        """Stop the sensor stream."""
        pass
