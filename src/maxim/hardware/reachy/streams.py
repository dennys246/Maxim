"""Reachy Mini data stream implementations.

Wraps the Reachy Mini SDK media interface to provide
VideoStream and AudioStream implementations.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from maxim.hardware.streams import AudioStream, VideoStream

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ReachyVideoStream(VideoStream):
    """Video stream from Reachy Mini camera.

    Wraps the Reachy Mini SDK media.get_frame() method
    with thread-safe access and caching.
    """

    def __init__(
        self,
        mini: Any,  # ReachyMini instance
        *,
        resolution: tuple[int, int] = (640, 480),
    ) -> None:
        """Initialize video stream.

        Args:
            mini: ReachyMini SDK instance.
            resolution: Expected video resolution (width, height).
        """
        self._mini = mini
        self._resolution = resolution
        self._lock = threading.Lock()
        self._active = False
        self._last_frame: NDArray[np.uint8] | None = None
        self._last_frame_time: float = 0.0

    def get_frame(self, timeout: float = 1.0) -> NDArray[np.uint8] | None:
        """Get the next video frame (blocking).

        Args:
            timeout: Maximum time to wait for a frame.

        Returns:
            BGR image as numpy array, or None if not available.
        """
        if not self._active:
            return None

        deadline = time.time() + timeout

        while time.time() < deadline:
            frame = self._get_frame_internal()
            if frame is not None:
                return frame
            time.sleep(0.01)

        return None

    def get_frame_nonblocking(self) -> NDArray[np.uint8] | None:
        """Get the latest frame without blocking."""
        if not self._active:
            return None
        return self._get_frame_internal()

    def _get_frame_internal(self) -> NDArray[np.uint8] | None:
        """Internal frame capture with locking."""
        with self._lock:
            try:
                frame = self._mini.media.get_frame()
                if frame is None or (hasattr(frame, "size") and frame.size == 0):
                    return None

                frame_array = np.asarray(frame, dtype=np.uint8)
                self._last_frame = frame_array
                self._last_frame_time = time.time()
                return frame_array

            except Exception as e:
                logger.debug("Frame capture failed: %s", e)
                return None

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the video resolution (width, height)."""
        return self._resolution

    @property
    def is_active(self) -> bool:
        """Check if the stream is currently active."""
        return self._active

    def start(self) -> None:
        """Mark stream as active."""
        self._active = True
        logger.debug("ReachyVideoStream started")

    def stop(self) -> None:
        """Mark stream as inactive."""
        self._active = False
        logger.debug("ReachyVideoStream stopped")


class ReachyAudioStream(AudioStream):
    """Audio stream from/to Reachy Mini.

    Wraps the Reachy Mini SDK media audio methods
    with thread-safe access.
    """

    def __init__(
        self,
        mini: Any,  # ReachyMini instance
        *,
        input_sample_rate: int = 48000,
        output_sample_rate: int = 48000,
    ) -> None:
        """Initialize audio stream.

        Args:
            mini: ReachyMini SDK instance.
            input_sample_rate: Microphone sample rate.
            output_sample_rate: Speaker sample rate.
        """
        self._mini = mini
        self._input_rate = input_sample_rate
        self._output_rate = output_sample_rate
        self._lock = threading.Lock()
        self._active = False

    def get_sample(self, timeout: float = 0.5) -> NDArray[np.float32] | None:
        """Get the next audio sample (blocking).

        Args:
            timeout: Maximum time to wait for a sample.

        Returns:
            Audio samples as float32 numpy array, or None.
        """
        if not self._active:
            return None

        deadline = time.time() + timeout

        while time.time() < deadline:
            sample = self._get_sample_internal()
            if sample is not None:
                return sample
            time.sleep(0.01)

        return None

    def _get_sample_internal(self) -> NDArray[np.float32] | None:
        """Internal sample capture with locking."""
        with self._lock:
            try:
                sample = self._mini.media.get_audio_sample()
                if sample is None or len(sample) == 0:
                    return None

                return np.asarray(sample, dtype=np.float32)

            except Exception as e:
                logger.debug("Audio capture failed: %s", e)
                return None

    def push_sample(self, samples: NDArray[np.float32]) -> bool:
        """Push audio samples to the speaker.

        Args:
            samples: Audio samples as float32 numpy array.

        Returns:
            True if samples were successfully queued.
        """
        if not self._active:
            return False

        with self._lock:
            try:
                self._mini.media.push_audio_sample(samples)
                return True
            except Exception as e:
                logger.debug("Audio push failed: %s", e)
                return False

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
        """Mark stream as active."""
        self._active = True
        logger.debug("ReachyAudioStream started")

    def stop(self) -> None:
        """Mark stream as inactive."""
        self._active = False
        logger.debug("ReachyAudioStream stopped")

    def update_sample_rates(self) -> None:
        """Update sample rates from the SDK (call after connection)."""
        try:
            self._input_rate = int(self._mini.media.get_input_audio_samplerate())
            self._output_rate = int(self._mini.media.get_output_audio_samplerate())
        except Exception as e:
            logger.debug("Failed to update sample rates: %s", e)
