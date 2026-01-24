"""
File-based IPC for transcription worker.

This module implements a file-watching system to avoid multiprocessing Queue
conflicts with CUDA. The parent process writes audio chunks to a directory,
and the transcription worker watches for new files.

This completely eliminates shared memory between parent (TensorFlow+CUDA) and
child (CTranslate2+CPU), preventing segfaults on Blackwell GPUs.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional


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
    cleanup_chunks: bool = True,
    verbosity: int = 0,
    log_file: str | None = None,
    shutdown_file: str | None = None,
) -> None:
    """
    Watch chunk_dir for .task files and transcribe them.

    This function runs in a subprocess with CUDA_VISIBLE_DEVICES="" set,
    completely isolated from the parent's TensorFlow+CUDA state.

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
    """
    import logging

    from maxim.data.audio.sound import transcribe_audio
    from maxim.models.audio.transcription import WhisperTranscriber
    from maxim.utils.logging import configure_logging, warn

    configure_logging(int(verbosity or 0), log_file=log_file)
    log = logging.getLogger("maxim.transcribe")

    log.info(f"File-based transcription worker starting")
    log.info(f"Watching directory: {chunk_dir}")
    log.info(f"Device: {device}, Compute type: {compute_type}")
    log.debug(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")

    # Initialize Whisper model
    try:
        transcriber = WhisperTranscriber(
            model_size_or_path=model_size_or_path,
            device=device,
            compute_type=compute_type,
        )
        log.info(f"✅ Whisper initialized on {device} with {compute_type}")
    except Exception as e:
        warn(f"Failed to initialize Whisper: {e}", logger=log)
        return

    # Create output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Track processed files
    processed = set()

    log.info("Entering file watch loop...")

    with open(output_path, "a", encoding="utf-8") as fp:
        while True:
            # Check for shutdown signal
            if shutdown_file and os.path.exists(shutdown_file):
                log.info(f"Shutdown signal detected: {shutdown_file}")
                try:
                    os.remove(shutdown_file)
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
                    )
                    elapsed = time.time() - started

                    # Write result
                    result["chunk_index"] = task.get("chunk_index", -1)
                    result["chunk_path"] = chunk_path
                    result["elapsed_time"] = elapsed

                    fp.write(json.dumps(result) + "\n")
                    fp.flush()

                    log.debug(f"Transcribed chunk {task.get('chunk_index', -1)} in {elapsed:.2f}s: \"{result.get('text', '')[:50]}...\"")

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

            # Sleep if no work found
            if not found_work:
                time.sleep(0.1)

    log.info("Transcription worker shutting down")
