"""Runtime inference/control modules.

This module consolidates inference-related code:
- segment_vision: Vision observation and display utilities
- transcribe_audio: Audio transcription using Whisper
"""

from maxim.inference.segment_vision import (
    DEFAULT_CLASS_WEIGHTS,
    NoveltyTracker,
    display_detections,
    get_default_novelty_tracker,
    passive_listening,
    passive_observation,
    score_detection_weighted,
)
from maxim.inference.transcribe_audio import (
    WhisperConfig,
    WhisperTranscriber,
    create_task_file,
    load_whisper_config,
    resolve_device,
    transcribe_audio,
    transcription_worker,
    watch_and_transcribe,
)

__all__ = [
    # Vision
    "DEFAULT_CLASS_WEIGHTS",
    "NoveltyTracker",
    "display_detections",
    "get_default_novelty_tracker",
    "passive_listening",
    "passive_observation",
    "score_detection_weighted",
    # Audio transcription
    "WhisperConfig",
    "WhisperTranscriber",
    "create_task_file",
    "load_whisper_config",
    "resolve_device",
    "transcribe_audio",
    "transcription_worker",
    "watch_and_transcribe",
]
