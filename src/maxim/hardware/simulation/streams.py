"""Simulated data streams for testing.

Provides mock VideoStream and AudioStream implementations
that generate synthetic data for testing without hardware.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from maxim.hardware.streams import AudioStream, VideoStream

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MockVideoStream(VideoStream):
    """Mock video stream for testing.

    Generates colored frames with optional noise for
    testing vision pipelines without hardware.
    """

    def __init__(
        self,
        *,
        resolution: tuple[int, int] = (640, 480),
        fps: float = 10.0,
        color: tuple[int, int, int] = (128, 128, 128),
        add_noise: bool = True,
    ) -> None:
        """Initialize mock video stream.

        Args:
            resolution: Video resolution (width, height).
            fps: Target frames per second.
            color: Base BGR color for generated frames.
            add_noise: Whether to add random noise to frames.
        """
        self._resolution = resolution
        self._fps = fps
        self._color = color
        self._add_noise = add_noise
        self._active = False
        self._frame_count = 0
        self._last_frame_time = 0.0

    def get_frame(self, timeout: float = 1.0) -> NDArray[np.uint8] | None:
        """Get the next video frame (blocking).

        Generates a new frame at the target FPS rate.
        """
        if not self._active:
            return None

        # Rate limit to target FPS
        min_period = 1.0 / self._fps
        elapsed = time.time() - self._last_frame_time

        if elapsed < min_period:
            wait_time = min(min_period - elapsed, timeout)
            time.sleep(wait_time)

        return self._generate_frame()

    def get_frame_nonblocking(self) -> NDArray[np.uint8] | None:
        """Get the latest frame without blocking."""
        if not self._active:
            return None
        return self._generate_frame()

    def _generate_frame(self) -> NDArray[np.uint8]:
        """Generate a synthetic frame."""
        width, height = self._resolution

        # Create base color frame
        frame = np.full((height, width, 3), self._color, dtype=np.uint8)

        # Add noise if enabled
        if self._add_noise:
            noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add timestamp overlay (simple frame counter)
        self._frame_count += 1
        self._last_frame_time = time.time()

        return frame

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the video resolution (width, height)."""
        return self._resolution

    @property
    def is_active(self) -> bool:
        """Check if the stream is currently active."""
        return self._active

    def start(self) -> None:
        """Start the video stream."""
        self._active = True
        self._frame_count = 0
        logger.debug("MockVideoStream started")

    def stop(self) -> None:
        """Stop the video stream."""
        self._active = False
        logger.debug("MockVideoStream stopped (frames=%d)", self._frame_count)


class MockAudioStream(AudioStream):
    """Mock audio stream for testing.

    Generates silent audio samples for testing
    audio pipelines without hardware.
    """

    def __init__(
        self,
        *,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 16000,
        chunk_size: int = 1024,
    ) -> None:
        """Initialize mock audio stream.

        Args:
            input_sample_rate: Microphone sample rate.
            output_sample_rate: Speaker sample rate.
            chunk_size: Samples per chunk.
        """
        self._input_rate = input_sample_rate
        self._output_rate = output_sample_rate
        self._chunk_size = chunk_size
        self._active = False
        self._samples_generated = 0
        self._samples_pushed = 0

    def get_sample(self, timeout: float = 0.5) -> NDArray[np.float32] | None:
        """Get the next audio sample (blocking).

        Generates silent audio samples.
        """
        if not self._active:
            return None

        # Simulate sample timing
        time.sleep(self._chunk_size / self._input_rate)

        # Generate silent samples with very low noise
        samples = np.random.randn(self._chunk_size).astype(np.float32) * 0.001
        self._samples_generated += self._chunk_size

        return samples

    def push_sample(self, samples: NDArray[np.float32]) -> bool:
        """Push audio samples to the speaker (simulated).

        Args:
            samples: Audio samples as float32 numpy array.

        Returns:
            True (always succeeds in simulation).
        """
        if not self._active:
            return False

        self._samples_pushed += len(samples)
        return True

    @property
    def input_sample_rate(self) -> int:
        """Get the input (microphone) sample rate."""
        return self._input_rate

    @property
    def output_sample_rate(self) -> int:
        """Get the output (speaker) sample rate."""
        return self._output_rate

    @property
    def is_active(self) -> bool:
        """Check if the stream is currently active."""
        return self._active

    def start(self) -> None:
        """Start the audio stream."""
        self._active = True
        self._samples_generated = 0
        self._samples_pushed = 0
        logger.debug("MockAudioStream started")

    def stop(self) -> None:
        """Stop the audio stream."""
        self._active = False
        logger.debug(
            "MockAudioStream stopped (generated=%d, pushed=%d)",
            self._samples_generated,
            self._samples_pushed,
        )
