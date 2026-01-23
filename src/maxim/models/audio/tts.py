"""Text-to-speech synthesis using Piper TTS.

Piper is a lightweight, fast, offline TTS engine suitable for
systems with limited resources (e.g., M2 Mac with 24GB unified memory).

Model download (~100MB):
    mkdir -p data/models/tts
    wget -O data/models/tts/en_US-lessac-medium.onnx \\
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
    wget -O data/models/tts/en_US-lessac-medium.onnx.json \\
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
"""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PiperTTS:
    """Text-to-speech synthesis using Piper.

    Piper is a fast, local neural text-to-speech system that runs
    efficiently on CPU. It uses ONNX models for inference.
    """

    # Default model paths relative to repo root
    DEFAULT_MODEL_DIR = "data/models/tts"
    DEFAULT_MODEL_NAME = "en_US-lessac-medium"

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        models_dir: str | Path = DEFAULT_MODEL_DIR,
    ) -> None:
        """Initialize Piper TTS engine.

        Args:
            model_path: Explicit path to .onnx model file. If provided,
                model_name and models_dir are ignored.
            model_name: Name of the model (without extension).
            models_dir: Directory containing model files.
        """
        self._piper: Any = None
        self._voice: Any = None
        self.sample_rate: int = 22050  # Piper default, will be updated on load

        # Resolve model path
        if model_path:
            self.model_path = Path(model_path)
        else:
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
            raise ImportError(
                "Missing dependency `piper-tts`. Install it with: pip install piper-tts"
            ) from e

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"TTS model not found at {self.model_path}. "
                f"Download it from: https://huggingface.co/rhasspy/piper-voices"
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

        Args:
            text: The text to synthesize.

        Returns:
            Audio samples as int16 numpy array.
        """
        if not text or not text.strip():
            return np.array([], dtype=np.int16)

        self._load_model()

        # Synthesize to WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            self._voice.synthesize(text, wav_file)

        # Read back as numpy array
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
        """Check if TTS is available (model exists and piper installed)."""
        try:
            from piper import PiperVoice  # noqa: F401

            return self.model_path.exists() and self.config_path.exists()
        except ImportError:
            return False


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
