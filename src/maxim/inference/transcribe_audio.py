"""Audio transcription utilities using Whisper.

Provides configuration, model loading, and transcription functions:
- WhisperConfig: Configuration dataclass for Whisper settings
- WhisperTranscriber: Wrapper around faster-whisper for transcription
- transcribe_audio: High-level transcription function
- transcription_worker: Multiprocessing worker for batch transcription
- File-based IPC for avoiding CUDA conflicts in subprocesses
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from maxim.utils.logging import configure_logging, warn


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

    def warmup(self) -> None:
        """Warm up the Whisper model with a dummy inference.

        This runs transcription on a short silence to:
        1. Complete any lazy initialization in faster-whisper
        2. Warm up GPU memory allocation (if using CUDA)
        3. Load VAD model if vad_filter is used

        Should be called during startup for ~1-3s latency reduction on first real transcription.
        """
        # Create 0.5 seconds of silence at 16kHz (Whisper's expected sample rate)
        dummy_audio = np.zeros(8000, dtype=np.float32)

        try:
            # Run with VAD to warm up the VAD model too
            segments, _ = self._model.transcribe(
                dummy_audio,
                language="en",
                beam_size=1,
                vad_filter=True,
            )
            # Consume the generator to ensure full warmup
            list(segments)
        except Exception:
            # Warmup failure is not critical - just means first real inference will be slower
            pass


def _as_mono_float32(audio: Any) -> np.ndarray:
    """Convert audio to mono float32 format."""
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
    transcriber: WhisperTranscriber,
    audio: Any,
    *,
    language: str = "en",
    beam_size: int = 1,
    vad_filter: bool = True,
    vad_parameters: dict[str, Any] | None = None,
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
        vad_parameters=vad_parameters,
    )
    if not isinstance(result, dict):
        return {"text": str(result)}
    return result


# -----------------------------------------------------------------------------
# File-based IPC for transcription worker
# -----------------------------------------------------------------------------

def create_task_file(
    chunk_dir: str,
    chunk_path: str,
    chunk_index: int,
    sample_rate: int,
) -> Optional[str]:
    """
    Create a .task file for the transcription worker to process.

    Args:
        chunk_dir: Directory to write task files
        chunk_path: Path to the audio chunk WAV file
        chunk_index: Sequential index of this chunk
        sample_rate: Audio sample rate

    Returns:
        Path to the created task file, or None if failed
    """
    try:
        task_data = {
            "chunk_path": chunk_path,
            "chunk_index": chunk_index,
            "sample_rate": sample_rate,
            "timestamp": time.time(),
        }

        # Use chunk index as filename to maintain order
        task_file = os.path.join(chunk_dir, f"task_{chunk_index:06d}.json")

        # Write atomically using temp file + rename
        temp_file = f"{task_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(task_data, f)

        os.rename(temp_file, task_file)
        return task_file

    except Exception:
        return None


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
    vad_parameters: dict[str, Any] | None = None,
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
            log.debug("Creating WhisperTranscriber...")
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
                # Final sync before exit
                try:
                    fp.flush()
                    os.fsync(fp.fileno())
                except Exception:
                    pass
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
                    vad_parameters=vad_parameters,
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
                os.fsync(fp.fileno())  # Force OS to write to disk immediately
            except Exception as e:
                warn("Transcription failed for '%s': %s", chunk_path, e, logger=log)
            finally:
                if cleanup_chunks:
                    try:
                        os.remove(chunk_path)
                    except Exception:
                        pass

    # Final filesystem sync to ensure all data is on disk before worker exits
    try:
        os.sync()
    except Exception:
        pass


def watch_and_transcribe(
    chunk_dir: str,
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
    shutdown_file: str | None = None,
    cuda_devices: str | None = None,
) -> None:
    """
    Watch chunk_dir for .task files and transcribe them.

    This function runs in a subprocess. If cuda_devices is provided and device="cuda",
    it will restore GPU visibility before importing CTranslate2.

    Args:
        chunk_dir: Directory to watch for task files
        output_path: JSONL file to append transcripts
        model_size_or_path: Whisper model name or path
        device: "cpu" or "cuda"
        compute_type: "int8", "float16", "float32", etc.
        language: Language code for transcription
        beam_size: Beam search size
        vad_filter: Whether to use VAD filtering
        cleanup_chunks: Whether to delete processed audio chunks
        verbosity: Logging level
        log_file: Optional log file path
        shutdown_file: Optional file path to signal shutdown
        cuda_devices: Original CUDA_VISIBLE_DEVICES to restore for GPU mode
    """
    # CRITICAL: Restore GPU visibility BEFORE any CUDA imports if using GPU
    # The parent may have hidden GPUs for TensorFlow/GStreamer compatibility,
    # but Whisper (CTranslate2) can use GPU independently
    if device == "cuda":
        # Restore GPU visibility - use provided value or default to "0" (first GPU)
        gpu_devices = cuda_devices if cuda_devices else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        # Also clear the CPU-only flag in case it was inherited
        os.environ.pop("MAXIM_TRANSCRIPTION_WORKER_CPU_ONLY", None)
    elif device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    configure_logging(int(verbosity or 0), log_file=log_file)
    log = logging.getLogger("maxim.transcribe")

    log.info("File-based transcription worker starting")
    log.info(f"Watching directory: {chunk_dir}")
    log.info(f"Requested device: {device}, Compute type: {compute_type}")
    log.info(f"cuda_devices param received: {cuda_devices!r}")
    log.info(f"CUDA_VISIBLE_DEVICES now set to: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')!r}")

    # Initialize Whisper model with fallback to CPU
    transcriber = None

    try:
        transcriber = WhisperTranscriber(
            model_size_or_path=model_size_or_path,
            device=device,
            compute_type=compute_type,
        )
        log.info(f"✅ Whisper initialized on {device} with {compute_type}")
    except Exception as e:
        if device == "cuda":
            # CUDA failed - fall back to CPU with float32
            log.warning(f"CUDA initialization failed: {e}")
            log.info("Falling back to CPU with float32...")
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                transcriber = WhisperTranscriber(
                    model_size_or_path=model_size_or_path,
                    device="cpu",
                    compute_type="float32",
                )
                log.info("✅ Whisper initialized on CPU with float32 (fallback)")
            except Exception as e2:
                warn(f"Failed to initialize Whisper on CPU fallback: {e2}", logger=log)
                return
        else:
            warn(f"Failed to initialize Whisper: {e}", logger=log)
            return

    # Create output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Track processed files
    processed = set()

    log.info("Entering file watch loop...")

    # Handle signals gracefully to ensure file is synced before exit
    shutdown_requested = False
    def _signal_handler(signum, frame):
        nonlocal shutdown_requested
        log.info(f"Received signal {signum}, requesting shutdown")
        shutdown_requested = True

    try:
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
    except Exception:
        pass  # Signal handling may not work in all contexts

    def _should_shutdown() -> bool:
        """Check if shutdown has been requested via signal or file."""
        return shutdown_requested or (shutdown_file and os.path.exists(shutdown_file))

    with open(output_path, "a", encoding="utf-8") as fp:
        while True:
            # Check for shutdown signal (file or signal)
            if _should_shutdown():
                log.info("Shutdown signal detected")
                if shutdown_file:
                    try:
                        os.remove(shutdown_file)
                    except Exception:
                        pass
                # Final sync before exit
                try:
                    fp.flush()
                    os.fsync(fp.fileno())
                except Exception:
                    pass
                break

            # Find task files
            try:
                task_files = sorted(Path(chunk_dir).glob("task_*.json"))
            except Exception as e:
                log.warning(f"Error listing task files: {e}")
                time.sleep(0.1)
                continue

            # Process new tasks
            found_work = False
            for task_file in task_files:
                task_path = str(task_file)

                if task_path in processed:
                    continue

                found_work = True
                log.debug(f"Processing task file: {task_file.name}")

                try:
                    # Read task
                    with open(task_path, "r", encoding="utf-8") as tf:
                        task = json.load(tf)

                    chunk_path = task.get("chunk_path")
                    if not chunk_path or not os.path.exists(chunk_path):
                        log.warning(f"Chunk file missing: {chunk_path}")
                        processed.add(task_path)
                        continue

                    # Transcribe
                    started = time.time()
                    result = transcribe_audio(
                        transcriber,
                        chunk_path,
                        language=language,
                        beam_size=beam_size,
                        vad_filter=vad_filter,
                        vad_parameters=vad_parameters,
                    )
                    elapsed = time.time() - started

                    # Write result
                    result["chunk_index"] = task.get("chunk_index", -1)
                    result["chunk_path"] = chunk_path
                    result["elapsed_time"] = elapsed

                    fp.write(json.dumps(result) + "\n")
                    fp.flush()
                    os.fsync(fp.fileno())  # Force OS to write to disk immediately

                    log.debug(f"Transcribed chunk {task.get('chunk_index', -1)} in {elapsed:.2f}s: \"{result.get('text', '')[:50]}...\"")

                    # Check for shutdown after each transcription (don't wait for next loop iteration)
                    if _should_shutdown():
                        log.info("Shutdown requested during processing, breaking out")
                        break

                    # Cleanup
                    if cleanup_chunks:
                        try:
                            os.remove(chunk_path)
                        except Exception:
                            pass

                    try:
                        os.remove(task_path)
                    except Exception:
                        pass

                    processed.add(task_path)

                except Exception as e:
                    log.error(f"Error processing task {task_path}: {e}")
                    processed.add(task_path)  # Don't retry failed tasks

            # Check if we broke out of the for loop due to shutdown
            if _should_shutdown():
                break

            # Sleep if no work found
            if not found_work:
                time.sleep(0.1)

    # Final filesystem sync to ensure all data is on disk
    try:
        os.sync()
    except Exception:
        pass

    log.info("Transcription worker shutting down")


__all__ = [
    "WhisperConfig",
    "WhisperTranscriber",
    "create_task_file",
    "load_whisper_config",
    "resolve_device",
    "transcribe_audio",
    "transcription_worker",
    "watch_and_transcribe",
]
