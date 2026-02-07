"""Backward compatibility - re-exports from maxim.inference.transcribe_audio."""

from maxim.inference.transcribe_audio import (
    create_task_file,
    watch_and_transcribe,
)

__all__ = [
    "create_task_file",
    "watch_and_transcribe",
]
