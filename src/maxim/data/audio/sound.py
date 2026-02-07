"""Audio processing utilities.

Re-exports transcription functions from maxim.inference.transcribe_audio
for backward compatibility.
"""

from __future__ import annotations

import os

# CRITICAL: Set CUDA environment BEFORE importing any packages that might use CUDA
# This runs at module import time, ensuring isolation for subprocess workers
# Check if we're in a subprocess that should hide CUDA (heuristic: check parent environ)
if os.environ.get("MAXIM_TRANSCRIPTION_WORKER_CPU_ONLY") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Re-export transcription functions from the consolidated inference module
from maxim.inference.transcribe_audio import (
    transcribe_audio,
    transcription_worker,
)

__all__ = [
    "transcribe_audio",
    "transcription_worker",
]
