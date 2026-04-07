"""Live-loop worker functions extracted from ``Maxim.live()``.

Each function was originally a closure nested inside ``live()``.  They have
been promoted to module-level so they can be imported, tested and maintained
independently.  The ``self`` reference that each closure captured is replaced
by an explicit *maxim* parameter of type ``Any`` (to avoid circular imports).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import wave
from typing import Any

import cv2
import numpy as np

from maxim.utils.audio import resample_audio, to_int16
from maxim.utils.logging import warn
from maxim.utils.queueing import put_latest


# ---------------------------------------------------------------------------
# IK warning handler
# ---------------------------------------------------------------------------


class IKWarningHandler(logging.Handler):
    """Temporary handler to capture IK warnings during motor commands."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.ik_failure_detected = False
        self.failure_message = None

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage().lower()
        if "ik error" in msg or "pose not achievable" in msg or "collision detected" in msg:
            self.ik_failure_detected = True
            self.failure_message = record.getMessage()


# ---------------------------------------------------------------------------
# Motor worker
# ---------------------------------------------------------------------------

IK_FAILURE_THRESHOLD = 3  # Reset to center after this many consecutive failures


def motor_worker(
    motor_queue: queue.Queue,
    stop_event: threading.Event,
    maxim: Any,
) -> None:
    """Drain *motor_queue* and execute motor commands until *stop_event* is set."""

    # Local mutable state (was closure-captured lists in original code)
    ik_failure_count = 0
    last_ik_reset_time = 0.0

    # Get reachy loggers to monitor for IK warnings (daemon logs to specific path)
    reachy_loggers = [
        logging.getLogger("reachy_mini"),
        logging.getLogger("reachy_mini.daemon"),
        logging.getLogger("reachy_mini.daemon.backend"),
        logging.getLogger("reachy_mini.daemon.backend.robot.backend.throttled"),
    ]

    while not stop_event.is_set():
        try:
            fn, args, kwargs = motor_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # Create handler to detect IK failures
        ik_handler = IKWarningHandler()
        for rl in reachy_loggers:
            rl.addHandler(ik_handler)

        position_info = kwargs.pop("_position_info", None)  # Extract position if passed
        commanded_6d = kwargs.pop("_commanded_6d", None)  # Extract 6D pose for bounds learning
        movement_duration = kwargs.get("duration", 0.5)  # Get duration for sync timing

        try:
            fn(*args, **kwargs)

            # Check if IK failure was detected via warnings
            if ik_handler.ik_failure_detected:
                maxim.log.warning("IK failure detected: %s", ik_handler.failure_message)
                ik_failure_count += 1

                # Record failure in AttentionNetwork for dynamic bounds learning
                if position_info is not None:
                    default_network = getattr(maxim, "_default_network", None)
                    if default_network is not None:
                        attention = getattr(default_network, "_attention_network", None)
                        if attention is not None:
                            try:
                                attention.record_gaze(position_info, success=False)
                                maxim.log.debug("Recorded IK failure at position %s", position_info)
                            except Exception as rec_err:
                                maxim.log.debug("Failed to record IK failure: %s", rec_err)

                # Record IK failure for workspace bounds learning
                if commanded_6d is not None:
                    default_network = getattr(maxim, "_default_network", None)
                    if default_network is not None:
                        bounds_learner = getattr(default_network, "_bounds_learner", None)
                        if bounds_learner is not None:
                            try:
                                maxim.sync_head_position()
                                actual_6d = {
                                    "yaw": float(getattr(maxim, "yaw", 0.0) or 0.0),
                                    "pitch": float(getattr(maxim, "pitch", 0.0) or 0.0),
                                    "y": float(getattr(maxim, "y", 0.0) or 0.0),
                                    "z": float(getattr(maxim, "z", 0.0) or 0.0),
                                    "roll": float(getattr(maxim, "roll", 0.0) or 0.0),
                                    "x": float(getattr(maxim, "x", 0.0) or 0.0),
                                }
                                bounds_learner.record_movement_outcome(
                                    commanded=commanded_6d,
                                    actual=actual_6d,
                                    ik_failure=True,
                                )
                                maxim.log.debug("Recorded IK failure bounds at %s", commanded_6d)
                            except Exception as bounds_err:
                                maxim.log.debug("Failed to record bounds failure: %s", bounds_err)

                # If too many consecutive failures, sync with hardware and optionally reset
                now = time.time()
                if ik_failure_count >= IK_FAILURE_THRESHOLD and (now - last_ik_reset_time) > 2.0:
                    maxim.log.warning("Too many IK failures (%d), syncing with hardware position", ik_failure_count)
                    try:
                        # First try to sync with actual hardware position
                        # This corrects any drift between our tracking and reality
                        synced = maxim.sync_head_position()

                        if synced:
                            maxim.log.info(
                                "Synced position from hardware: yaw=%.1f°, pitch=%.1f°", maxim.yaw, maxim.pitch
                            )
                            # If we're already near center, just reset counter
                            if abs(maxim.yaw) < 5.0 and abs(maxim.pitch) < 5.0:
                                maxim.log.info("Already near center, clearing failure count")
                                ik_failure_count = 0
                            else:
                                # Move toward center from current synced position
                                maxim.log.info("Moving to center from synced position")
                                from maxim.motion.movement import move_head

                                move_head(maxim.mini, 0, 0, 0, 0, 0, 0, 0.5)
                                maxim.yaw = 0.0
                                maxim.pitch = 0.0
                                ik_failure_count = 0
                        else:
                            # Sync failed, fall back to blind reset
                            maxim.log.warning("Hardware sync failed, blind reset to center")
                            maxim.yaw = 0.0
                            maxim.pitch = 0.0
                            from maxim.motion.movement import move_head

                            move_head(maxim.mini, 0, 0, 0, 0, 0, 0, 0.5)
                            ik_failure_count = 0

                        last_ik_reset_time = now
                    except Exception as reset_err:
                        maxim.log.warning("Failed to reset head: %s", reset_err)
            else:
                # Movement succeeded - reset failure counter
                ik_failure_count = 0

                # Record success for reachability learning
                if position_info is not None:
                    default_network = getattr(maxim, "_default_network", None)
                    if default_network is not None:
                        attention = getattr(default_network, "_attention_network", None)
                        if attention is not None:
                            try:
                                attention.record_gaze(position_info, success=True)
                            except Exception:
                                pass

                # Record outcome for workspace bounds learning
                if commanded_6d is not None:
                    default_network = getattr(maxim, "_default_network", None)
                    if default_network is not None:
                        bounds_learner = getattr(default_network, "_bounds_learner", None)
                        if bounds_learner is not None:
                            try:
                                # Wait briefly for movement to settle, then sync
                                time.sleep(min(0.1, float(movement_duration) * 0.3))
                                maxim.sync_head_position()
                                actual_6d = {
                                    "yaw": float(getattr(maxim, "yaw", 0.0) or 0.0),
                                    "pitch": float(getattr(maxim, "pitch", 0.0) or 0.0),
                                    "y": float(getattr(maxim, "y", 0.0) or 0.0),
                                    "z": float(getattr(maxim, "z", 0.0) or 0.0),
                                    "roll": float(getattr(maxim, "roll", 0.0) or 0.0),
                                    "x": float(getattr(maxim, "x", 0.0) or 0.0),
                                }
                                bounds_learner.record_movement_outcome(
                                    commanded=commanded_6d,
                                    actual=actual_6d,
                                    ik_failure=False,
                                )
                            except Exception as bounds_err:
                                maxim.log.debug("Failed to record bounds outcome: %s", bounds_err)

        except Exception as e:
            warn("Motor command failed: %s", e, logger=maxim.log)
            maxim._note_connection_failure("motor", e)

            # Also record as failure if we have position info
            if position_info is not None:
                default_network = getattr(maxim, "_default_network", None)
                if default_network is not None:
                    attention = getattr(default_network, "_attention_network", None)
                    if attention is not None:
                        try:
                            attention.record_gaze(position_info, success=False)
                        except Exception:
                            pass
        finally:
            for rl in reachy_loggers:
                rl.removeHandler(ik_handler)


# ---------------------------------------------------------------------------
# Frame capture worker
# ---------------------------------------------------------------------------


def frame_capture_worker(
    stop_event: threading.Event,
    media_lock: threading.Lock,
    maxim: Any,
    frame_save_queue: queue.Queue,
    frame_obs_queue: Any,
) -> None:
    """Capture video frames from hardware and enqueue them for saving / observation."""

    min_period = 1.0 / float(getattr(maxim, "video_fps", 20.0) or 20.0)
    last_ts = 0.0
    if maxim.mini is None:
        return  # No robot — nothing to capture
    while not stop_event.is_set():
        frame = None
        try:
            with media_lock:
                frame = maxim.mini.media.get_frame()
        except Exception as e:
            warn("Failed to capture frame: %s", e, logger=maxim.log)
            maxim._note_connection_failure("video", e)
            time.sleep(0.01)
            continue

        is_empty = frame is None
        if not is_empty and hasattr(frame, "size"):
            is_empty = frame.size == 0
        if is_empty:
            time.sleep(0.005)
            continue

        now = time.time()
        try:
            frame_save_queue.put((now, frame), timeout=0.5)
        except queue.Full:
            frame_save_queue.put((now, frame))

        put_latest(frame_obs_queue, (now, frame))

        sleep_for = min_period - (now - last_ts)
        if sleep_for > 0:
            time.sleep(min(sleep_for, 0.05))
        last_ts = now


# ---------------------------------------------------------------------------
# Audio capture worker
# ---------------------------------------------------------------------------


def audio_capture_worker(
    stop_event: threading.Event,
    media_lock: threading.Lock,
    maxim: Any,
    audio_save_queue: queue.Queue | None,
    audio_input_rate: int | None,
    audio_output_rate: int | None,
) -> None:
    """Capture audio samples from hardware, resample, and enqueue for saving."""

    if not maxim.audio or audio_save_queue is None or maxim.mini is None:
        return  # No robot or no audio — nothing to capture

    while not stop_event.is_set():
        sample = None
        try:
            with media_lock:
                sample = maxim.mini.media.get_audio_sample()
        except Exception as e:
            warn("Failed to capture audio sample: %s", e, logger=maxim.log)
            maxim._note_connection_failure("audio", e)
            time.sleep(0.01)
            continue

        if sample is None or len(sample) == 0:
            time.sleep(0.005)
            continue

        try:
            sample_arr = np.asarray(sample)
            sample_arr = resample_audio(sample_arr, audio_input_rate, audio_output_rate)
            sample_i16 = to_int16(sample_arr)
        except Exception as e:
            warn("Failed to process audio sample: %s", e, logger=maxim.log)
            time.sleep(0.01)
            continue

        now = time.time()
        try:
            audio_save_queue.put((now, sample_i16, audio_output_rate or audio_input_rate), timeout=0.5)
        except queue.Full:
            audio_save_queue.put((now, sample_i16, audio_output_rate or audio_input_rate))


# ---------------------------------------------------------------------------
# Video writer worker
# ---------------------------------------------------------------------------


def video_writer_worker(
    stop_event: threading.Event,
    maxim: Any,
    frame_save_queue: queue.Queue,
    video_path: str,
) -> None:
    """Drain *frame_save_queue* and write frames to a video file on disk."""

    writer = None
    opened = False
    disabled = False
    width = None
    height = None
    frames_written = 0
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)

    while not stop_event.is_set() or not frame_save_queue.empty():
        try:
            _, frame = frame_save_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            frame_arr = np.asarray(frame)
            if frame_arr.ndim != 3 or frame_arr.shape[2] < 3:
                frame_save_queue.task_done()
                continue
            if frame_arr.dtype != np.uint8:
                frame_arr = np.clip(frame_arr, 0, 255).astype(np.uint8)
        except Exception:
            frame_save_queue.task_done()
            continue

        if writer is None and not disabled:
            try:
                height = int(frame_arr.shape[0])
                width = int(frame_arr.shape[1])
                fps = float(getattr(maxim, "video_fps", 20.0) or 20.0)
                for codec in ("mp4v", "avc1"):
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                    if writer is not None and writer.isOpened():
                        opened = True
                        break
                    try:
                        if writer is not None:
                            writer.release()
                    except Exception:
                        pass
                    writer = None
                if not opened:
                    warn("Failed to open video writer for '%s'.", video_path, logger=maxim.log)
                    disabled = True
            except Exception as e:
                warn("Failed to initialize video writer: %s", e, logger=maxim.log)
                writer = None
                disabled = True

        if opened and writer is not None:
            try:
                writer.write(frame_arr)
                frames_written += 1
            except Exception as e:
                warn("Failed to write video frame: %s", e, logger=maxim.log)

        frame_save_queue.task_done()

    try:
        if writer is not None:
            writer.release()
    except Exception:
        pass

    if frames_written == 0:
        file_size = None
        try:
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
        except Exception:
            file_size = None
        if file_size is not None:
            warn(
                "No video frames were written to '%s' (size=%d bytes). The file may be empty/unplayable.",
                video_path,
                int(file_size),
                logger=maxim.log,
            )
        else:
            warn(
                "No video frames were written to '%s'. The file may be empty/unplayable.",
                video_path,
                logger=maxim.log,
            )


# ---------------------------------------------------------------------------
# Audio writer worker
# ---------------------------------------------------------------------------


def audio_writer_worker(
    stop_event: threading.Event,
    maxim: Any,
    audio_save_queue: queue.Queue | None,
    audio_path: str,
    chunk_dir: str,
    audio_input_rate: int | None,
    audio_output_rate: int | None,
    transcribe_process: Any | None,
) -> None:
    """Drain *audio_save_queue*, write samples to a WAV file, and chunk for transcription."""

    if not maxim.audio or audio_save_queue is None:
        return

    os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)
    os.makedirs(chunk_dir, exist_ok=True)

    sample_rate = int(audio_output_rate or audio_input_rate or 16000)
    chunk_frames = None
    if transcribe_process is not None:
        chunk_frames = int(float(getattr(maxim, "audio_len", 5.0) or 5.0) * float(sample_rate))
        chunk_frames = max(chunk_frames, sample_rate)  # at least 1s

    wf = wave.open(audio_path, "wb")
    channels = None
    pending_tasks: list[dict] = []
    buffer: list[np.ndarray] = [] if chunk_frames is not None else []
    buffered_frames = 0
    total_frames = 0
    chunk_index = 0

    def _flush_pending() -> None:
        from maxim.data.audio._file_based_transcription import create_task_file

        nonlocal pending_tasks
        if transcribe_process is None:
            return
        while pending_tasks:
            try:
                task = pending_tasks[0]
                task_file = create_task_file(
                    chunk_dir=chunk_dir,
                    chunk_path=task["chunk_path"],
                    chunk_index=task["chunk_index"],
                    sample_rate=task["sample_rate"],
                )
                if task_file:
                    pending_tasks.pop(0)
                else:
                    break  # Failed to create task file, retry later
            except Exception:
                break

    def _write_chunk(chunk_arr: np.ndarray, start_frame: int) -> None:
        nonlocal chunk_index
        if transcribe_process is None:
            return
        chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:06d}.wav")
        wf_chunk = wave.open(chunk_path, "wb")
        try:
            wf_chunk.setnchannels(int(channels or 1))
            wf_chunk.setsampwidth(2)
            wf_chunk.setframerate(sample_rate)
            wf_chunk.writeframes(np.ascontiguousarray(chunk_arr).tobytes())
        finally:
            wf_chunk.close()

        task = {
            "chunk_path": chunk_path,
            "chunk_index": int(chunk_index),
            "sample_rate": int(sample_rate),
            "start_s": float(start_frame) / float(sample_rate),
            "end_s": float(start_frame + int(chunk_arr.shape[0])) / float(sample_rate),
        }
        pending_tasks.append(task)
        _flush_pending()
        chunk_index += 1

    try:
        while not stop_event.is_set() or not audio_save_queue.empty():
            _flush_pending()
            try:
                _, sample_i16, sr = audio_save_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                sample_arr = np.asarray(sample_i16, dtype=np.int16)
            except Exception:
                audio_save_queue.task_done()
                continue

            if channels is None:
                channels = 1 if sample_arr.ndim == 1 else int(sample_arr.shape[1])
                wf.setnchannels(int(channels))
                wf.setsampwidth(2)
                wf.setframerate(int(sample_rate))

            try:
                wf.writeframes(np.ascontiguousarray(sample_arr).tobytes())
            except Exception as e:
                warn("Failed to write audio frames: %s", e, logger=maxim.log)

            frames = int(sample_arr.shape[0])
            if chunk_frames is not None:
                buffer.append(sample_arr)
                buffered_frames += frames

            while chunk_frames is not None and buffered_frames >= chunk_frames:
                remaining = chunk_frames
                parts: list[np.ndarray] = []
                while remaining > 0 and buffer:
                    head = buffer[0]
                    if int(head.shape[0]) <= remaining:
                        parts.append(head)
                        remaining -= int(head.shape[0])
                        buffer.pop(0)
                    else:
                        parts.append(head[:remaining])
                        buffer[0] = head[remaining:]
                        remaining = 0

                if remaining > 0:
                    break

                chunk_arr = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
                _write_chunk(chunk_arr, start_frame=total_frames)
                total_frames += int(chunk_arr.shape[0])
                buffered_frames -= chunk_frames

            audio_save_queue.task_done()
    finally:
        try:
            wf.close()
        except Exception:
            pass

        if transcribe_process is not None:
            # Flush all pending transcription tasks
            _flush_pending()
            deadline = time.time() + 10.0
            while pending_tasks and time.time() < deadline:
                from maxim.data.audio._file_based_transcription import create_task_file

                try:
                    task = pending_tasks[0]
                    task_file = create_task_file(
                        chunk_dir=chunk_dir,
                        chunk_path=task["chunk_path"],
                        chunk_index=task["chunk_index"],
                        sample_rate=task["sample_rate"],
                    )
                    if task_file:
                        pending_tasks.pop(0)
                except Exception:
                    continue
