from __future__ import annotations

import logging
import math
import threading
from typing import Optional

import numpy as np
from scipy.signal import resample, resample_poly

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Local Audio Playback
# ─────────────────────────────────────────────────────────────────────────────

# Global state for local audio playback
_local_audio_available: bool | None = None
_local_audio_lock = threading.Lock()


def _check_local_audio() -> bool:
    """Check if local audio playback is available."""
    global _local_audio_available
    if _local_audio_available is not None:
        return _local_audio_available

    try:
        import sounddevice as sd  # noqa: F401
        _local_audio_available = True
        logger.debug("Local audio playback available (sounddevice)")
    except ImportError:
        _local_audio_available = False
        logger.debug("Local audio not available (install sounddevice: pip install sounddevice)")

    return _local_audio_available


def play_audio_local(
    samples: np.ndarray,
    sample_rate: int = 16000,
    blocking: bool = False,
) -> bool:
    """Play audio through local speakers (Mac/PC).

    Args:
        samples: Audio samples as numpy array (int16 or float).
        sample_rate: Sample rate in Hz (default 16000).
        blocking: If True, wait for playback to complete.

    Returns:
        True if playback started successfully, False otherwise.
    """
    if not _check_local_audio():
        logger.warning("Local audio not available. Install with: pip install sounddevice")
        return False

    if samples is None or len(samples) == 0:
        return False

    try:
        import sounddevice as sd

        # Convert to float32 for sounddevice (expects -1.0 to 1.0)
        if samples.dtype == np.int16:
            audio_float = samples.astype(np.float32) / 32767.0
        elif np.issubdtype(samples.dtype, np.floating):
            audio_float = samples.astype(np.float32)
        else:
            audio_float = samples.astype(np.float32) / 32767.0

        # Play audio
        with _local_audio_lock:
            sd.play(audio_float, samplerate=sample_rate)
            if blocking:
                sd.wait()

        return True

    except Exception as e:
        logger.warning("Local audio playback failed: %s", e)
        return False


def stop_local_audio() -> None:
    """Stop any currently playing local audio."""
    if not _check_local_audio():
        return

    try:
        import sounddevice as sd
        sd.stop()
    except Exception:
        pass


class LocalSpeaker:
    """Speaker that plays audio through local computer speakers.

    Can be used as a drop-in replacement for Reachy speaker when
    running remotely or for testing.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        """Initialize local speaker.

        Args:
            sample_rate: Expected sample rate of audio.
        """
        self.sample_rate = sample_rate
        self._available = _check_local_audio()

    @property
    def is_available(self) -> bool:
        """Check if local audio is available."""
        return self._available

    def __call__(self, samples: np.ndarray) -> bool:
        """Play audio samples (callable interface for compatibility).

        Args:
            samples: Audio samples to play.

        Returns:
            True if playback started successfully.
        """
        return play_audio_local(samples, self.sample_rate, blocking=False)

    def play(self, samples: np.ndarray, blocking: bool = False) -> bool:
        """Play audio samples.

        Args:
            samples: Audio samples to play.
            blocking: If True, wait for playback to complete.

        Returns:
            True if playback started successfully.
        """
        return play_audio_local(samples, self.sample_rate, blocking=blocking)

    def stop(self) -> None:
        """Stop any currently playing audio."""
        stop_local_audio()


# ─────────────────────────────────────────────────────────────────────────────
# Audio Format Conversion
# ─────────────────────────────────────────────────────────────────────────────


def to_int16(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.int16:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        clipped = np.clip(arr, -1.0, 1.0)
        return np.ascontiguousarray((clipped * 32767.0).astype(np.int16))
    return np.ascontiguousarray(np.clip(arr, -32768, 32767).astype(np.int16))


def resample_audio(sample: np.ndarray, input_rate: Optional[int], output_rate: Optional[int]) -> np.ndarray:
    if not input_rate or not output_rate or int(input_rate) == int(output_rate):
        return sample

    try:
        gcd = math.gcd(int(input_rate), int(output_rate))
        up = int(output_rate) // gcd
        down = int(input_rate) // gcd
        return resample_poly(sample, up, down, axis=0)
    except Exception:
        num_sample = int(int(output_rate) * len(sample) / int(input_rate))
        return resample(sample, num_sample)
