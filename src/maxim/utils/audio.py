from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, Callable, Optional

import numpy as np
from scipy.signal import resample, resample_poly

logger = logging.getLogger(__name__)


def _to_device_float32(samples: np.ndarray) -> np.ndarray:
    """Normalize samples to float32 in [-1.0, 1.0] for a float32 audio sink.

    The single conversion used by BOTH the local playback path and the Reachy
    device path. Piper TTS returns ``int16`` while ``reachy_mini``'s
    ``push_audio_sample`` is typed ``NDArray[float32]`` and wraps ``.tobytes()``
    with no conversion — pushing int16 straight through reinterprets the bytes as
    float32 (noise at half duration). This is that missing conversion, in one place.
    """
    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32767.0
    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float32)
    return samples.astype(np.float32) / 32767.0


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
        audio_float = _to_device_float32(samples)

        # Play audio
        with _local_audio_lock:
            sd.play(audio_float, samplerate=sample_rate)
            if blocking:
                sd.wait()

        return True

    except Exception as e:
        logger.warning("Local audio playback failed: %s", e)
        return False


def push_audio_to_device(
    mini: Any,
    samples: np.ndarray,
    sample_rate: int = 16000,
    *,
    log: logging.Logger | None = None,
    on_failure: Callable[[Exception], None] | None = None,
) -> bool:
    """Play audio on a Reachy Mini's speaker (converting to float32), with local fallback.

    The single device-speak implementation, needing ONLY the SDK handle ``mini`` — no
    embodied runtime, no agent. Used by :meth:`MediaLoopMixin.speak` and by an embedder
    that holds a bare ``reachy_mini`` before any agent exists (e.g. the spoken-code
    pairing announcer, which must speak before the SetupWizard runs).

    Tries ``mini.media.push_audio_sample`` first, **converting to float32** — the SDK is
    typed ``NDArray[float32]`` and wraps ``.tobytes()`` with no conversion, so pushing
    Piper's int16 straight through was noise at half duration — and falls back to local
    speakers on a WebRTC "not implemented" error. Never raises; returns whether it played.

    Args:
        mini: SDK handle exposing ``media.push_audio_sample(NDArray[float32])``.
        samples: audio samples (int16 from Piper, or float).
        sample_rate: rate for the LOCAL fallback (the device pipeline owns its own rate).
        log: logger for diagnostics (defaults to this module's).
        on_failure: optional callback for a non-WebRTC device error (e.g. the mixin's
            connection-failure note); the exception is passed to it.
    """
    log = log or logger
    if samples is None or len(samples) == 0:
        return False

    prefer_local = os.environ.get("MAXIM_TTS_LOCAL", "").lower() in ("1", "true", "yes")

    if not prefer_local and mini is not None:
        try:
            mini.media.push_audio_sample(_to_device_float32(samples))
            return True
        except Exception as e:
            error_str = str(e).lower()
            if "not implemented" in error_str or "webrtc" in error_str:
                log.info("Reachy speaker not available (WebRTC), using local audio")
            else:
                log.warning("Failed to play audio on Reachy: %s", e)
                if on_failure is not None:
                    on_failure(e)

    try:
        success = play_audio_local(samples, sample_rate=sample_rate, blocking=False)
        if success:
            log.debug("Playing audio through local speakers")
        return success
    except Exception as e:
        log.warning("Local audio playback failed: %s", e)
        return False


def make_device_speak_sink(mini: Any) -> Callable[..., bool]:
    """Bind a Reachy Mini handle into a ``speak(samples, sample_rate=...) -> bool`` sink.

    Construction is inert — it only closes over ``mini`` and touches no audio device
    until called — so it is safe to build offline / at import time. Intended for
    ``make_pairing_announcer(tts=..., speak=make_device_speak_sink(reachy_mini))``: the
    embedder gets a device sink from the bare SDK handle, with no composition of its own
    and no dependency on the embodied runtime.
    """

    def speak(samples: np.ndarray, sample_rate: int = 16000) -> bool:
        return push_audio_to_device(mini, samples, sample_rate)

    return speak


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
