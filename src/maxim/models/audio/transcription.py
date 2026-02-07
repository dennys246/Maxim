"""Backward compatibility - re-exports from maxim.inference.transcribe_audio."""

from maxim.inference.transcribe_audio import (
    WhisperConfig,
    WhisperTranscriber,
    load_whisper_config,
    resolve_device,
)

__all__ = [
    "WhisperConfig",
    "WhisperTranscriber",
    "load_whisper_config",
    "resolve_device",
]
