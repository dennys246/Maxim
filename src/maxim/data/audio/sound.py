from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

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
    try:
        import logging

        from maxim.utils.logging import configure_logging

        configure_logging(int(verbosity or 0), log_file=log_file)
        log = logging.getLogger("maxim.transcribe")
    except Exception:
        log = None

    try:
        from maxim.models.audio.whisper import WhisperTranscriber

        transcriber = WhisperTranscriber(
            model_size_or_path=model_size_or_path,
            device=device,
            compute_type=compute_type,
        )
    except Exception as e:
        warn("Whisper transcriber unavailable: %s", e, logger=log)
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        fp = open(output_path, "a", encoding="utf-8")
    except Exception as e:
        warn("Failed to open transcript file '%s': %s", output_path, e, logger=log)
        return

    with fp:
        while True:
            task = task_queue.get()
            if task is None:
                break

            if not isinstance(task, dict):
                continue

            chunk_path = task.get("chunk_path")
            if not chunk_path:
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
