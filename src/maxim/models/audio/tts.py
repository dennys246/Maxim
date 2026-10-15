"""Text-to-speech synthesis using Piper TTS.

Piper is a lightweight, fast, offline TTS engine suitable for
systems with limited resources (e.g., M2 Mac with 24GB unified memory).

Voice provisioning (~63 MB per voice, pinned and sha256-verified in ``models/download.py::TTS_MODELS``):
    python -m maxim.models.download --tts            # the default voice, by hand
    maxim.models.download.ensure_tts_voice(...)      # programmatic, with consent + its own lock
    VoiceProvisioner(...)                            # a device: background retry until the voice LOADS

Intended device-bootstrap usage (the caller lives in the device's bootstrap, e.g. maxim-pulse's Reachy app; no
pymaxim module wires it yet): arm spoken-code pairing only once the ``VoiceProvisioner`` is ready::

    voice = VoiceProvisioner(); voice.start()
    announcer = make_pairing_announcer(voice.engine, make_device_speak_sink(mini))
    app = build_app(dist, pairing_announcer=announcer, pairing_ready=voice.is_ready)
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Phrase Cache for common TTS outputs
# ─────────────────────────────────────────────────────────────────────────────

# Cache size: ~50 phrases * ~50KB avg = ~2.5MB memory
_PHRASE_CACHE_SIZE = 50


def _cache_key(text: str, sample_rate: int) -> str:
    """Generate cache key for text + sample rate."""
    return hashlib.md5(f"{text}:{sample_rate}".encode()).hexdigest()


class _AudioCache:
    """LRU cache for synthesized audio to avoid re-synthesis of common phrases."""

    def __init__(self, maxsize: int = _PHRASE_CACHE_SIZE) -> None:
        self._cache: dict[str, np.ndarray] = {}
        self._access_order: list[str] = []
        self._maxsize = maxsize

    def get(self, key: str) -> np.ndarray | None:
        """Get cached audio, updating access order."""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, audio: np.ndarray) -> None:
        """Cache audio, evicting oldest if at capacity."""
        if key in self._cache:
            return  # Already cached

        # Evict oldest if at capacity
        while len(self._cache) >= self._maxsize and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[key] = audio
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()


class PiperTTS:
    """Text-to-speech synthesis using Piper.

    Piper is a fast, local neural text-to-speech system that runs
    efficiently on CPU. It uses ONNX models for inference.

    Features:
    - Lazy model loading (loads on first use)
    - Raw audio synthesis (avoids WAV encode/decode overhead)
    - LRU phrase caching for common phrases
    """

    DEFAULT_MODEL_NAME = "en_US-lessac-medium"

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        models_dir: str | Path | None = None,
        cache_phrases: bool = True,
    ) -> None:
        """Initialize Piper TTS engine.

        Args:
            model_path: Explicit path to .onnx model file. If provided,
                model_name and models_dir are ignored.
            model_name: Name of the model (without extension).
            models_dir: Directory containing model files.
            cache_phrases: If True, cache synthesized audio for repeated phrases.
        """
        self._piper: Any = None
        self._voice: Any = None
        self._has_raw_synthesis: bool | None = None  # Detected on first use
        self.sample_rate: int = 22050  # Piper default, will be updated on load

        # Phrase caching
        self._cache_enabled = cache_phrases
        self._cache = _AudioCache() if cache_phrases else None

        # Resolve model path
        if model_path:
            self.model_path = Path(model_path)
        else:
            if models_dir is None:
                from maxim.utils.paths import model_dir

                models_dir = str(model_dir() / "tts")
            self.model_path = Path(models_dir) / f"{model_name}.onnx"

        self.config_path = self.model_path.with_suffix(".onnx.json")

    def _load_model(self) -> None:
        """Lazy-load the Piper TTS model.

        Raises:
            ImportError: If piper-tts is not installed.
            FileNotFoundError: If model files are not found.
        """
        if self._voice is not None:
            return

        try:
            from piper import PiperVoice
        except ImportError as e:
            raise ImportError("Missing dependency `piper-tts`. Install it with: pip install piper-tts") from e

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"TTS model not found at {self.model_path}. Provision it with: python -m maxim.models.download --tts"
            )

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"TTS model config not found at {self.config_path}. "
                f"Download the .onnx.json file alongside the .onnx model."
            )

        logger.info("Loading Piper TTS model from %s", self.model_path)
        self._voice = PiperVoice.load(str(self.model_path), str(self.config_path))

        # Update sample rate from model config
        if hasattr(self._voice, "config") and hasattr(self._voice.config, "sample_rate"):
            self.sample_rate = self._voice.config.sample_rate

        logger.info("Piper TTS loaded (sample_rate=%d)", self.sample_rate)

    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to audio samples.

        Uses raw synthesis if available (faster), falls back to WAV-based
        synthesis otherwise. Caches common phrases to avoid re-synthesis.

        Args:
            text: The text to synthesize.

        Returns:
            Audio samples as int16 numpy array.
        """
        if not text or not text.strip():
            return np.array([], dtype=np.int16)

        text = text.strip()
        self._load_model()

        # Check cache first
        if self._cache is not None:
            cache_key = _cache_key(text, self.sample_rate)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("TTS cache hit for: %s", text[:30])
                return cached.copy()  # Return copy to prevent mutation

        # Synthesize audio
        audio = self._synthesize_raw(text)

        # Cache the result (only for shorter phrases to save memory)
        if self._cache is not None and len(text) < 200:
            self._cache.put(cache_key, audio.copy())

        return audio

    def _synthesize_raw(self, text: str) -> np.ndarray:
        """Synthesize audio using raw method if available, else WAV fallback.

        Args:
            text: The text to synthesize.

        Returns:
            Audio samples as int16 numpy array.
        """
        # Detect if raw synthesis is available (only check once)
        if self._has_raw_synthesis is None:
            self._has_raw_synthesis = hasattr(self._voice, "synthesize_stream_raw")
            if self._has_raw_synthesis:
                logger.debug("Using raw synthesis (faster)")
            else:
                logger.debug("Using WAV-based synthesis (fallback)")

        # Use raw synthesis if available (avoids WAV encode/decode overhead)
        if self._has_raw_synthesis:
            try:
                audio_chunks = []
                for audio_bytes in self._voice.synthesize_stream_raw(text):
                    chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_chunks.append(chunk)
                if audio_chunks:
                    return np.concatenate(audio_chunks)
                return np.array([], dtype=np.int16)
            except Exception as e:
                logger.warning("Raw synthesis failed, falling back to WAV: %s", e)
                self._has_raw_synthesis = False

        # Fallback: WAV-based synthesis
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            self._voice.synthesize(text, wav_file)

        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            audio_data = wav_file.readframes(wav_file.getnframes())
            audio = np.frombuffer(audio_data, dtype=np.int16)

        return audio

    def synthesize_to_file(self, text: str, output_path: Path | str) -> Path:
        """Synthesize text and save to WAV file.

        Args:
            text: The text to synthesize.
            output_path: Path to save the WAV file.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._load_model()

        with wave.open(str(output_path), "wb") as wav_file:
            self._voice.synthesize(text, wav_file)

        logger.debug("Saved TTS audio to %s", output_path)
        return output_path

    @property
    def is_available(self) -> bool:
        """Check if TTS is AVAILABLE: piper installed and the voice's files present. Not that they are verified or
        that the voice loads: :meth:`VoiceProvisioner.is_ready` answers that, for a device that must speak."""
        return _piper_installed() and self.model_path.exists() and self.config_path.exists()


class TTSEngine:
    """Abstract TTS engine interface for potential future backends.

    Currently wraps PiperTTS but allows for easy swapping of backends.
    """

    def __init__(
        self,
        backend: str = "piper",
        model_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize TTS engine.

        Args:
            backend: TTS backend to use ("piper" only for now).
            model_path: Path to model file.
            **kwargs: Additional backend-specific arguments.
        """
        if backend != "piper":
            raise ValueError(f"Unsupported TTS backend: {backend}. Only 'piper' is supported.")

        self._backend = PiperTTS(model_path=model_path, **kwargs)

    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to audio samples."""
        return self._backend.synthesize(text)

    def synthesize_to_file(self, text: str, output_path: Path | str) -> Path:
        """Synthesize and save to file."""
        return self._backend.synthesize_to_file(text, output_path)

    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self._backend.sample_rate

    @property
    def is_available(self) -> bool:
        """Check if TTS is available."""
        return self._backend.is_available

    def load(self) -> None:
        """Load the voice now (it otherwise loads on first use), raising if it cannot: so a caller learns at
        startup, not when it first speaks, that the voice is missing, corrupt or ``piper`` is not installed."""
        self._backend._load_model()


def _piper_installed() -> bool:
    from maxim.utils.optional_deps import optional_dependency_available

    return optional_dependency_available("piper")


class VoiceProvisioner:
    """Provision a Piper voice in the background and report when it can SPEAK (#1268 follow-up F1).

    For a device: start it, build the announcer on :attr:`engine`, and pass :meth:`is_ready` to
    ``build_app(pairing_ready=...)``. A daemon thread calls ``ensure_tts_voice(auto=True)`` (deploying the robot app is
    the consent), checks that ``piper`` is installed and LOADS the voice; on any failure (no network at boot, the extra
    missing, a corrupt file) it retries with backoff from ``retry_s`` up to ``max_retry_s`` until :meth:`stop`. Until
    then the sign-in page offers only the paste path, and the by-ear path appears without a restart (owner decision
    2026-10-14). A failure is logged by type, never with the voice's bytes or a pairing code.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        models_dir: str | Path | None = None,
        retry_s: float = 15.0,
        max_retry_s: float = 300.0,
    ) -> None:
        from maxim.models import download

        self.model_name = model_name or download.DEFAULT_TTS
        self._models_dir = models_dir
        kwargs: dict[str, Any] = {"model_name": self.model_name}
        if models_dir is not None:
            kwargs["models_dir"] = models_dir
        self.engine = TTSEngine(**kwargs)
        self._retry_s, self._max_retry_s = retry_s, max_retry_s
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="tts-voice-provisioner")
            self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)

    def _attempt(self) -> bool:
        from maxim.models import download

        if not download.ensure_tts_voice(self.model_name, auto=True, models_dir=self._models_dir):
            return False
        if not _piper_installed():
            logger.warning(
                "TTS voice %s is on disk but `piper` is not installed (pip install 'pymaxim[tts]')", self.model_name
            )
            return False
        try:
            self.engine.load()
        except Exception as e:  # a corrupt or unloadable voice: retry, and say what kind of failure it was
            logger.warning("TTS voice %s did not load (%s)", self.model_name, type(e).__name__)
            return False
        return True

    def _run(self) -> None:
        delay, failures = self._retry_s, 0
        while not self._stop.is_set():
            try:
                ok = self._attempt()
            except Exception as e:  # ANY failure (a lock, a stat, storage) is a retry, never a dead thread
                ok = False
                (logger.warning if failures == 0 else logger.debug)(
                    "TTS voice %s provisioning attempt raised %s; retrying", self.model_name, type(e).__name__
                )
            if ok and not self._stop.is_set():
                self._ready.set()
                logger.info("TTS voice %s is ready", self.model_name)
                return
            failures += 1
            self._stop.wait(delay)
            delay = min(delay * 2, self._max_retry_s)
