from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WhisperConfig:
    """Configuration for Whisper transcription."""
    enabled: bool = True
    model: str = "distil-large-v3"
    device: str = "auto"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1
    vad_filter: bool = True
    cleanup_chunks: bool = True
    # VAD parameters - lower threshold = more sensitive to speech
    vad_threshold: float = 0.25  # Default Silero is 0.5, lowered for better detection
    vad_min_speech_duration_ms: int = 100  # Minimum speech duration (default 250)
    vad_min_silence_duration_ms: int = 1500  # Silence before split (default 2000)
    vad_speech_pad_ms: int = 300  # Padding around speech (default 400)


def load_whisper_config() -> WhisperConfig:
    """Load Whisper configuration from file or environment.

    Config sources (in order of precedence):
    1. Environment variables (MAXIM_WHISPER_*)
    2. data/util/whisper.json
    3. Default values
    """
    default = WhisperConfig()

    # Find config file
    candidates = [
        os.getenv("MAXIM_WHISPER_CONFIG", ""),
        os.path.join(os.getcwd(), "data", "util", "whisper.json"),
        os.path.join(os.getcwd(), "whisper.json"),
    ]

    # Try to find repo root
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        candidates.append(os.path.join(repo_root, "data", "util", "whisper.json"))
    except Exception:
        pass

    raw: dict[str, Any] = {}
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    raw = loaded
                    break
            except Exception:
                pass

    # Helper to get config value with env override
    def get_str(key: str, default_val: str) -> str:
        env_key = f"MAXIM_WHISPER_{key.upper()}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            return env_val.strip()
        return str(raw.get(key, default_val)).strip()

    def get_int(key: str, default_val: int) -> int:
        env_key = f"MAXIM_WHISPER_{key.upper()}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            try:
                return int(env_val)
            except ValueError:
                pass
        try:
            return int(raw.get(key, default_val))
        except (ValueError, TypeError):
            return default_val

    def get_bool(key: str, default_val: bool) -> bool:
        env_key = f"MAXIM_WHISPER_{key.upper()}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        val = raw.get(key, default_val)
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("1", "true", "yes", "on")

    def get_float(key: str, default_val: float) -> float:
        env_key = f"MAXIM_WHISPER_{key.upper()}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            try:
                return float(env_val)
            except ValueError:
                pass
        try:
            return float(raw.get(key, default_val))
        except (ValueError, TypeError):
            return default_val

    return WhisperConfig(
        enabled=get_bool("enabled", default.enabled),
        model=get_str("model", default.model),
        device=get_str("device", default.device),
        compute_type=get_str("compute_type", default.compute_type),
        language=get_str("language", default.language),
        beam_size=get_int("beam_size", default.beam_size),
        vad_filter=get_bool("vad_filter", default.vad_filter),
        cleanup_chunks=get_bool("cleanup_chunks", default.cleanup_chunks),
        vad_threshold=get_float("vad_threshold", default.vad_threshold),
        vad_min_speech_duration_ms=get_int("vad_min_speech_duration_ms", default.vad_min_speech_duration_ms),
        vad_min_silence_duration_ms=get_int("vad_min_silence_duration_ms", default.vad_min_silence_duration_ms),
        vad_speech_pad_ms=get_int("vad_speech_pad_ms", default.vad_speech_pad_ms),
    )


def resolve_device(device: str) -> str:
    """Resolve 'auto' device to actual device."""
    if device != "auto":
        return device

    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # Check CTranslate2 CUDA support
    try:
        import ctranslate2
        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            return "cuda"
    except Exception:
        pass

    return "cpu"


class WhisperTranscriber:
    """
    Thin wrapper around `faster-whisper` so the rest of the codebase only needs a
    single, stable interface.
    """

    def __init__(
        self,
        *,
        model_size_or_path: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        import logging
        import os

        log = logging.getLogger("maxim.transcribe")

        try:
            log.debug("Importing faster_whisper.WhisperModel...")
            from faster_whisper import WhisperModel
            log.debug("faster_whisper imported successfully")
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Missing dependency `faster-whisper`. Install it (and its backend) to enable transcription."
            ) from e

        self.model_size_or_path = str(model_size_or_path or "tiny")
        self.device = str(device or "cpu")
        self.compute_type = str(compute_type or "int8")

        log.debug(f"Initializing WhisperModel: model={self.model_size_or_path}, device={self.device}, compute_type={self.compute_type}")
        log.debug(f"Environment: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
        self._model = WhisperModel(self.model_size_or_path, device=self.device, compute_type=self.compute_type)
        log.debug("WhisperModel created successfully")

    def transcribe(
        self,
        audio: Any,
        *,
        language: str = "en",
        beam_size: int = 1,
        vad_filter: bool = True,
        vad_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Build transcribe kwargs
        kwargs: dict[str, Any] = {
            "language": str(language or "en"),
            "beam_size": int(beam_size or 1),
            "vad_filter": bool(vad_filter),
        }

        # Add VAD parameters if VAD is enabled and parameters provided
        if vad_filter and vad_parameters:
            kwargs["vad_parameters"] = vad_parameters

        segments, info = self._model.transcribe(audio, **kwargs)

        seg_list: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for seg in segments:
            seg_list.append(
                {
                    "start": float(getattr(seg, "start", 0.0) or 0.0),
                    "end": float(getattr(seg, "end", 0.0) or 0.0),
                    "text": str(getattr(seg, "text", "")),
                }
            )
            text_parts.append(str(getattr(seg, "text", "")))

        language_out = None
        duration_out = None
        try:
            language_out = getattr(info, "language", None)
            duration_out = getattr(info, "duration", None)
        except Exception:
            language_out = None
            duration_out = None

        return {
            "text": "".join(text_parts).strip(),
            "segments": seg_list,
            "language": language_out,
            "duration": duration_out,
        }
