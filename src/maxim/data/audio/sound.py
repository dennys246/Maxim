from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

# CRITICAL: Set CUDA environment BEFORE importing any packages that might use CUDA
# This runs at module import time, ensuring isolation for subprocess workers
# Check if we're in a subprocess that should hide CUDA (heuristic: check parent environ)
if os.environ.get("MAXIM_TRANSCRIPTION_WORKER_CPU_ONLY") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import numpy as np

from maxim.utils.logging import warn


def _as_mono_float32(audio: Any) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim == 0:
        raise ValueError("Audio sample is scalar.")
    if arr.ndim == 1:
        mono = arr
    elif arr.ndim == 2:
        mono = arr.mean(axis=1)
    else:
        raise ValueError(f"Expected 1D/2D audio array, got shape {arr.shape}.")

    if np.issubdtype(mono.dtype, np.integer):
        scale = float(np.iinfo(mono.dtype).max) or 32767.0
        mono = mono.astype(np.float32) / scale
    else:
        mono = mono.astype(np.float32)

    return np.ascontiguousarray(mono)


def transcribe_audio(
    transcriber,
    audio: Any,
    *,
    language: str = "en",
    beam_size: int = 1,
    vad_filter: bool = True,
) -> dict[str, Any]:
    """
    Transcribe a chunk of audio using the configured Whisper transcriber.

    `audio` can be a file path (preferred for efficiency) or a numpy-like array.
    """
    if transcriber is None:
        raise ValueError("Missing transcriber instance.")

    audio_input: Any = audio
    if isinstance(audio, (list, tuple, np.ndarray)):
        audio_input = _as_mono_float32(audio)

    result = transcriber.transcribe(
        audio_input,
        language=str(language or "en"),
        beam_size=int(beam_size or 1),
        vad_filter=bool(vad_filter),
    )
    if not isinstance(result, dict):
        return {"text": str(result)}
    return result


def transcription_worker(
    task_queue,
    output_path: str,
    *,
    model_size_or_path: str = "tiny",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
    beam_size: int = 1,
    vad_filter: bool = True,
    cleanup_chunks: bool = True,
    verbosity: int = 0,
    log_file: str | None = None,
) -> None:
    """
    Multiprocessing worker that consumes chunk WAV paths and appends transcripts to a JSONL file.

    Expected queue messages:
      - None (sentinel): stop worker
      - dict: {"chunk_path": str, "chunk_index": int, "sample_rate": int, ...}
    """
    # CRITICAL: Hide GPU from CTranslate2 when using CPU mode
    # CTranslate2 doesn't respect CUDA_VISIBLE_DEVICES set before process spawn
    # and will attempt CUDA initialization even with device="cpu" if GPUs are visible
    # This is especially problematic with RTX 5080/Blackwell (sm_120) GPUs
    # See: https://github.com/OpenNMT/CTranslate2/issues/1693
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        import logging

        from maxim.utils.logging import configure_logging

        configure_logging(int(verbosity or 0), log_file=log_file)
        log = logging.getLogger("maxim.transcribe")
    except Exception:
        log = None

    if log:
        log.debug(f"Transcription worker starting: device={device}, compute_type={compute_type}")
        log.debug(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")

    # Try GPU first, fallback to CPU if initialization fails (e.g., RTX 5080/Blackwell issues)
    # Note: CTranslate2 4.6.2+ disables INT8 for Blackwell GPUs (sm_120)
    transcriber = None
    try:
        if log:
            log.debug("Importing WhisperTranscriber...")
        from maxim.models.audio.transcription import WhisperTranscriber

        if log:
            log.debug(f"Creating WhisperTranscriber(model={model_size_or_path}, device={device}, compute_type={compute_type})...")
        # Try with requested device (usually "cpu", but could be "cuda")
        transcriber = WhisperTranscriber(
            model_size_or_path=model_size_or_path,
            device=device,
            compute_type=compute_type,
        )
        if log:
            log.info(f"✅ Whisper initialized on {device} with {compute_type}")
    except Exception as e:
        # If GPU was requested and failed, try CPU fallback
        if device != "cpu":
            if log:
                log.warning(f"Whisper GPU initialization failed, falling back to CPU: {e}")

            # CRITICAL: Hide GPU from CTranslate2 before CPU fallback attempt
            # See: https://github.com/OpenNMT/CTranslate2/issues/1693
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

            # Use float32 for CPU fallback (int8 disabled for Blackwell in CTranslate2 4.6.2+)
            cpu_compute_type = "float32" if "int8" in str(compute_type).lower() else compute_type

            try:
                transcriber = WhisperTranscriber(
                    model_size_or_path=model_size_or_path,
                    device="cpu",
                    compute_type=cpu_compute_type,
                )
                if log:
                    log.info(f"Whisper initialized on CPU with {cpu_compute_type} (fallback)")
            except Exception as cpu_err:
                warn("Whisper transcriber unavailable (CPU fallback also failed): %s", cpu_err, logger=log)
                return
        else:
            warn("Whisper transcriber unavailable: %s", e, logger=log)
            return

    if transcriber is None:
        warn("Whisper transcriber could not be initialized", logger=log)
        return

    if log:
        log.debug(f"Creating output directory for: {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if log:
        log.debug(f"Opening transcript file: {output_path}")
    try:
        fp = open(output_path, "a", encoding="utf-8")
    except Exception as e:
        warn("Failed to open transcript file '%s': %s", output_path, e, logger=log)
        return

    if log:
        log.debug("Entering transcription worker main loop")
        log.debug(f"Queue object type: {type(task_queue)}")
        log.debug("About to call task_queue.get() - this is where segfault likely occurs")

    with fp:
        iteration = 0
        while True:
            iteration += 1
            if log:
                log.debug(f"Loop iteration {iteration}: calling task_queue.get()...")

            try:
                task = task_queue.get()
                if log:
                    log.debug(f"Loop iteration {iteration}: received task: {type(task)}")
            except Exception as e:
                if log:
                    log.error(f"Exception during task_queue.get(): {e}")
                import traceback
                traceback.print_exc()
                break

            if task is None:
                if log:
                    log.debug("Received sentinel (None), exiting worker")
                break

            if not isinstance(task, dict):
                if log:
                    log.warning(f"Received non-dict task: {type(task)}")
                continue

            chunk_path = task.get("chunk_path")
            if not chunk_path:
                if log:
                    log.warning("Task missing chunk_path")
                continue

            started = time.time()
            try:
                result = transcribe_audio(
                    transcriber,
                    chunk_path,
                    language=language,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                )
                record: dict[str, Any] = {
                    "time": time.time(),
                    "chunk_index": task.get("chunk_index"),
                    "chunk_path": chunk_path,
                    "audio_sample_rate": task.get("sample_rate"),
                    "start_s": task.get("start_s"),
                    "end_s": task.get("end_s"),
                    "text": result.get("text", ""),
                    "segments": result.get("segments"),
                    "language": result.get("language", language),
                    "duration": result.get("duration"),
                    "elapsed_s": float(time.time() - started),
                }
                fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                fp.flush()
            except Exception as e:
                warn("Transcription failed for '%s': %s", chunk_path, e, logger=log)
            finally:
                if cleanup_chunks:
                    try:
                        os.remove(chunk_path)
                    except Exception:
                        pass
