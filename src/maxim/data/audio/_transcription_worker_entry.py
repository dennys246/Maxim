#!/usr/bin/env python
"""
Clean entry point for transcription worker subprocess.

This module MUST be imported before any CUDA libraries to ensure
CUDA_VISIBLE_DEVICES is set early enough to prevent conflicts.
"""

import os
import sys


def run_transcription_worker(
    task_queue,
    output_path: str,
    *,
    model_size_or_path: str = "tiny",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
    beam_size: int = 1,
    vad_filter: bool = True,
    vad_parameters: dict | None = None,
    cleanup_chunks: bool = True,
    verbosity: int = 0,
    log_file: str | None = None,
) -> None:
    """
    Wrapper that sets CUDA environment before importing anything else.

    CRITICAL: This must be the entry point for the subprocess to ensure
    CUDA_VISIBLE_DEVICES is set before any CUDA libraries load.
    """
    # Set CUDA_VISIBLE_DEVICES FIRST, before any other imports
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Also set these to ensure complete CUDA isolation
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    # NOW import the actual worker (which imports CTranslate2, etc.)
    from maxim.data.audio.sound import transcription_worker

    # Call the real worker
    transcription_worker(
        task_queue=task_queue,
        output_path=output_path,
        model_size_or_path=model_size_or_path,
        device=device,
        compute_type=compute_type,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        cleanup_chunks=cleanup_chunks,
        verbosity=verbosity,
        log_file=log_file,
    )
