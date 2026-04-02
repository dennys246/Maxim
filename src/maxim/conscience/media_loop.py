from __future__ import annotations
import multiprocessing as mp
import os
import queue
import threading
import time
import wave
from typing import Optional
import cv2
import numpy as np
from maxim.utils.audio import resample_audio, to_int16
from maxim.utils.data_management import CLIInputLogger, TrainingSampleLogger, build_home
from maxim.utils.logging import configure_logging, log_swallowed_exception, warn
from maxim.data.camera.display import prepare_display, show_photo
from maxim.inference.observation import display_detections, passive_observation
from maxim.utils.gpu_compat import detect_blackwell

_gpu_state = detect_blackwell()
_original_cuda_devices = _gpu_state.original_cuda_devices


class MediaLoopMixin:
    """Mixin providing the live media capture/playback loop and related helpers."""

    def live(
        self,
        home_dir: Optional[str] = None,
        *,
        epochs: int | None = None,
        parallel: bool = True,
        vision: bool = True,
        motor: bool = True,
        wake_up: bool = True,
        run_id: str | None = None,
    ):
        if not run_id:
            run_id = time.strftime("%Y-%m-%d_%H%M%S")

        if home_dir is not None:
            self.home_dir = home_dir
        if epochs is not None:
            self._set_epochs(epochs)

        build_home(self.home_dir)

        log_path = os.path.join(self.home_dir, "logs", f"reachy_log_{run_id}.log")
        configure_logging(self.verbosity, log_file=log_path)

        video_path = os.path.join(self.home_dir, "videos", f"reachy_video_{run_id}.mp4")
        audio_path = os.path.join(self.home_dir, "audio", f"reachy_audio_{run_id}.wav")
        transcript_path = os.path.join(self.home_dir, "transcript", f"reachy_transcript_{run_id}.jsonl")
        chunk_dir = os.path.join(self.home_dir, "audio", "chunks")
        cli_path = os.path.join(self.home_dir, "cli", f"cli_input_{run_id}.jsonl")
        vision_events_path = os.path.join(self.home_dir, "vision", f"vision_events_{run_id}.jsonl")

        self.run_id = run_id
        self.run_start_ts = time.time()
        self.log_path = log_path
        self.video_path = video_path
        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.cli_path = cli_path
        self.vision_events_path = vision_events_path

        # Create transcript directory and empty file early so readers can open it immediately
        try:
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
            # Touch the file to create it if it doesn't exist
            with open(transcript_path, "a", encoding="utf-8"):
                pass
        except OSError as e:
            log_swallowed_exception(e, operation="create_transcript_dir", context={"path": transcript_path})

        try:
            prev_logger = getattr(self, "_training_logger", None)
            if prev_logger is not None:
                prev_logger.stop(timeout=0.5)
        except Exception as e:
            log_swallowed_exception(e, operation="stop_training_logger")

        try:
            training_dir = os.path.join(self.home_dir, "training")
            self._training_logger = TrainingSampleLogger(training_dir)
            self._training_logger.start()
        except Exception as e:
            self._training_logger = None
            warn("Failed to start training sample logger: %s", e, logger=self.log)

        try:
            prev_cli_logger = getattr(self, "_cli_logger", None)
            if prev_cli_logger is not None:
                prev_cli_logger.stop(timeout=0.5)
        except Exception as e:
            log_swallowed_exception(e, operation="stop_cli_logger")
        self._cli_logger = None

        try:
            prev_vision_logger = getattr(self, "_vision_event_logger", None)
            if prev_vision_logger is not None:
                self._stop_vision_event_stream(timeout=0.5)
        except Exception as e:
            log_swallowed_exception(e, operation="stop_vision_event_logger")
        self._vision_event_logger = None

        if bool(getattr(self, "interactive", True)):
            try:
                self._cli_logger = CLIInputLogger(cli_path)
                self._cli_logger.start()
                if int(getattr(self, "verbosity", 0) or 0) >= 1:
                    self.log.info("CLI input recording enabled: %s", cli_path)
            except Exception as e:
                self._cli_logger = None
                warn("Failed to start CLI input logger: %s", e, logger=self.log)

        epochs_label = "unlimited" if self.epochs is None else str(int(self.epochs))
        self.log.info(
            "Starting live loop (home_dir=%s, epochs=%s, observation_period=%s, mode=%s, audio=%s, audio_len=%.1fs)",
            self.home_dir,
            epochs_label,
            str(getattr(self, "observation_period", None)),
            str(getattr(self, "mode", "reflection")),
            str(bool(getattr(self, "audio", True))),
            float(getattr(self, "audio_len", 0.0) or 0.0),
        )
        if vision:
            self.log.info("Recording video: %s", video_path)
        if self.audio:
            self.log.info("Recording audio: %s", audio_path)
            self.log.info("Transcripts: %s", transcript_path)

        effective_wake_up = bool(wake_up)
        mode = str(getattr(self, "mode", "") or "").strip().lower()
        if mode == "sleep":
            effective_wake_up = False
        if str(getattr(self, "requested_mode", "") or "").strip().lower() == "sleep":
            effective_wake_up = False
        if self._robot is not None:
            self.awaken(vision=bool(vision), motor=bool(motor), audio=bool(self.audio), wake_up=effective_wake_up)
        if vision and self.verbose:
            # Keep OpenCV GUI calls on a dedicated process main thread (safer on Linux/WSL).
            prepare_display()

        # Create media lock BEFORE starting agentic runtime (CaptureManager needs it)
        media_lock = threading.Lock()
        self._media_lock = media_lock
        stop_event = threading.Event()
        self._live_stop_event = stop_event

        # Phase 2: Start agentic runtime automatically when not in sleep mode
        # The agentic system handles all decision-making; live() just does I/O
        # Note: Must happen AFTER _media_lock is created for CaptureManager to use it
        if mode != "sleep":
            try:
                self._start_agentic_runtime()
            except Exception as e:
                warn("Failed to start agentic runtime: %s", e, logger=self.log)

        frame_obs_queue: queue.Queue = queue.Queue(maxsize=1)
        frame_save_queue: queue.Queue = queue.Queue(maxsize=512)
        audio_save_queue: queue.Queue = queue.Queue(maxsize=512) if self.audio else None

        motor_queue: queue.Queue = queue.Queue(maxsize=4)
        self._motor_queue = motor_queue if parallel and motor else None

        audio_input_rate = None
        audio_output_rate = None
        if self.audio and self.mini is not None:
            try:
                audio_input_rate = int(self.mini.media.get_input_audio_samplerate())
                audio_output_rate = int(self.mini.media.get_output_audio_samplerate())
            except Exception as e:
                warn("Failed to read audio sample rates: %s", e, logger=self.log)

        transcribe_process = None
        transcribe_shutdown_file = None
        if self.audio and parallel and self._robot is not None:
            os.makedirs(chunk_dir, exist_ok=True)
            try:
                from maxim.data.audio._file_based_transcription import watch_and_transcribe
                from maxim.models.audio.transcription import load_whisper_config

                # Load whisper config from data/util/whisper.json
                whisper_cfg = load_whisper_config()
                self.log.info("Whisper config: model=%s, device=%s, compute_type=%s",
                              whisper_cfg.model, whisper_cfg.device, whisper_cfg.compute_type)

                # Use file-based IPC instead of multiprocessing Queues
                # Queues use shared memory which conflicts with TensorFlow+CUDA in parent
                # File watching completely isolates parent and child processes
                # See: https://github.com/tensorflow/tensorflow/issues/8220
                #      https://github.com/OpenNMT/CTranslate2/issues/1693
                ctx = mp.get_context("spawn")
                if self.log:
                    self.log.debug("Using file-based IPC (no shared memory)")

                # Get config values (can be overridden by env vars)
                vad_filter = whisper_cfg.vad_filter
                compute_type = whisper_cfg.compute_type
                whisper_model = whisper_cfg.model
                whisper_device = whisper_cfg.device

                # Auto-detect Blackwell GPUs (RTX 50 series) and adjust compute type
                # CTranslate2 has compatibility issues with sm_120 (Blackwell) architecture:
                # - int8 compute types fail with CUBLAS_STATUS_NOT_SUPPORTED
                # - float16 works fine on Blackwell CUDA
                # See: https://github.com/OpenNMT/CTranslate2/issues/1865
                #      https://github.com/SYSTRAN/faster-whisper/issues/1260
                blackwell_detected = False

                # Resolve "auto" device
                if whisper_device == "auto":
                    whisper_device = "cuda"  # Default to CUDA

                # Use nvidia-smi for detection (works even when CUDA_VISIBLE_DEVICES="")
                # This allows detection after parent has hidden GPUs from TensorFlow
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        gpu_names = result.stdout.strip().lower()
                        if 'rtx 50' in gpu_names or '5080' in gpu_names or '5090' in gpu_names:
                            blackwell_detected = True
                            self.log.warning("⚠️  Detected Blackwell GPU (nvidia-smi)")
                except Exception as e:
                    self.log.debug(f"nvidia-smi check failed: {e}")

                # For Blackwell GPUs: int8 fails, but float16 works on CUDA
                # Use float16 instead of falling back to CPU (faster and uses less VRAM than float32)
                if blackwell_detected and "int8" in compute_type.lower():
                    compute_type = "float16"
                    self.log.info("   Compute type auto-changed: int8 → float16 (int8 incompatible with Blackwell)")
                    self.log.info("   Keeping CUDA device (float16 works on Blackwell)")

                # Build VAD parameters dict for fine-tuned voice detection
                # Lower threshold = more sensitive to speech (default Silero is 0.5)
                vad_parameters = {
                    "threshold": whisper_cfg.vad_threshold,
                    "min_speech_duration_ms": whisper_cfg.vad_min_speech_duration_ms,
                    "min_silence_duration_ms": whisper_cfg.vad_min_silence_duration_ms,
                    "speech_pad_ms": whisper_cfg.vad_speech_pad_ms,
                } if vad_filter else None

                self.log.info("Whisper model: %s", whisper_model)
                self.log.info("Transcription VAD filter: %s", "enabled" if vad_filter else "disabled")
                if vad_filter:
                    self.log.info("VAD threshold: %.2f (lower = more sensitive)", whisper_cfg.vad_threshold)
                self.log.info("Whisper compute type: %s", compute_type)
                self.log.info("Whisper device: %s (will fallback to CPU if unavailable)", whisper_device)

                # Get original CUDA devices for subprocess (stored before we hid GPUs)
                cuda_devices_for_subprocess = None
                if whisper_device == "cuda":
                    cuda_devices_for_subprocess = _original_cuda_devices
                    if cuda_devices_for_subprocess:
                        self.log.info("Whisper subprocess will use GPU: CUDA_VISIBLE_DEVICES=%s", cuda_devices_for_subprocess)

                # Set environment flag for CPU-only mode BEFORE spawning subprocess
                # This ensures CUDA_VISIBLE_DEVICES is set at module import time
                if whisper_device == "cpu":
                    os.environ["MAXIM_TRANSCRIPTION_WORKER_CPU_ONLY"] = "1"
                    self.log.debug("Set MAXIM_TRANSCRIPTION_WORKER_CPU_ONLY=1 for subprocess")

                # Create shutdown signal file path
                transcribe_shutdown_file = os.path.join(chunk_dir, ".shutdown")

                transcribe_process = ctx.Process(
                    target=watch_and_transcribe,
                    args=(chunk_dir, transcript_path),
                    kwargs={
                        "model_size_or_path": whisper_model,
                        "device": whisper_device,
                        "compute_type": compute_type,
                        "language": whisper_cfg.language,
                        "beam_size": whisper_cfg.beam_size,
                        "vad_filter": vad_filter,
                        "vad_parameters": vad_parameters,
                        "cleanup_chunks": whisper_cfg.cleanup_chunks,
                        "verbosity": int(self.verbosity or 0),
                        "log_file": log_path,
                        "shutdown_file": transcribe_shutdown_file,
                        "cuda_devices": cuda_devices_for_subprocess,
                    },
                    daemon=True,
                )
                transcribe_process.start()
                self.log.debug("Transcription process started, waiting for initialization...")
                time.sleep(0.1)
                if not transcribe_process.is_alive():
                    warn(
                        "Transcription worker exited immediately (is `faster-whisper` installed and model available?).",
                        logger=self.log,
                    )
                    transcribe_process = None
                    transcribe_shutdown_file = None
                else:
                    self.log.debug("Transcription process alive and running")
            except Exception as e:
                transcribe_process = None
                transcribe_shutdown_file = None
                warn("Failed to start transcription worker: %s", e, logger=self.log)

        self.log.debug("Continuing with main loop setup after transcription worker...")

        # Import worker functions from workers module
        from maxim.conscience.workers import (
            motor_worker,
            frame_capture_worker,
            audio_capture_worker,
            video_writer_worker,
            audio_writer_worker,
        )

        threads: list[threading.Thread] = []
        cli_thread = self._start_cli_listener(stop_event)
        if cli_thread is not None:
            threads.append(cli_thread)
            cli_thread.start()

        key_thread = self._start_key_listener(stop_event)
        if key_thread is not None:
            threads.append(key_thread)
            key_thread.start()

        transcript_thread = self._start_transcript_listener(stop_event)
        if transcript_thread is not None:
            threads.append(transcript_thread)
            transcript_thread.start()

        has_robot = self._robot is not None
        if parallel:
            if vision and has_robot:
                threads.append(threading.Thread(target=frame_capture_worker, args=(stop_event, media_lock, self, frame_save_queue, frame_obs_queue), name="maxim.capture.video", daemon=True))
                threads.append(threading.Thread(target=video_writer_worker, args=(stop_event, self, frame_save_queue, video_path), name="maxim.write.video", daemon=True))
            if motor and has_robot:
                threads.append(threading.Thread(target=motor_worker, args=(motor_queue, stop_event, self), name="maxim.motor", daemon=True))
            if self.audio and has_robot:
                threads.append(threading.Thread(target=audio_capture_worker, args=(stop_event, media_lock, self, audio_save_queue, audio_input_rate, audio_output_rate), name="maxim.capture.audio", daemon=True))
                threads.append(threading.Thread(target=audio_writer_worker, args=(stop_event, self, audio_save_queue, audio_path, chunk_dir, audio_input_rate, audio_output_rate, transcribe_process), name="maxim.write.audio", daemon=True))

            for t in threads:
                if t is key_thread or t is transcript_thread or t is cli_thread:
                    continue
                t.start()

        try:
            if not vision or not has_robot:
                if has_robot:
                    self.log.info("Audio-only mode: recording until Ctrl+C.")
                else:
                    self.log.info("Headless mode: agentic loop active, no media capture.")
                self._run_headless_loop(stop_event)
            else:
                while True:
                    if stop_event.is_set():
                        break
                    if self.epochs is not None and int(self.current_epoch) >= int(self.epochs):
                        self.log.info("Reached epochs limit (%d). Stopping.", int(self.epochs))
                        break

                    if parallel:
                        try:
                            frame_ts, photo = frame_obs_queue.get(timeout=2.0)
                        except queue.Empty:
                            if stop_event.is_set():
                                break
                            if self.verbosity >= 2:
                                self.log.debug("Waiting for camera frame...")
                            continue
                    else:
                        frame_ts = time.time()
                        photo = self.look(show=False)

                    if photo is None:
                        if self.verbosity >= 2:
                            self.log.debug("No frame captured.")
                        continue

                    try:
                        self._last_frame_ts = float(frame_ts)
                    except Exception:
                        self._last_frame_ts = None
                    try:
                        self._last_frame = photo
                    except Exception:
                        pass

                    self.current_epoch += 1

                    # Phase 3: Display-only observation loop
                    # Use CaptureManager frames when available (already segmented)
                    # Movement decisions are handled by the agentic runtime
                    if self.observation_period and self.current_epoch % self.observation_period == 0:
                        try:
                            with self._observation_lock:
                                # Check if CaptureManager has a recent frame with detections
                                capture_manager = getattr(self, "_capture_manager", None)
                                if capture_manager is not None:
                                    captured = capture_manager.get_latest_frame()
                                    if captured is not None and captured.segmented:
                                        # Use pre-segmented frame from CaptureManager
                                        if self.verbosity >= 2:
                                            self.log.debug(
                                                "Display frame from CaptureManager (epoch=%d, detections=%d)",
                                                self.current_epoch,
                                                len(captured.detections or []),
                                            )
                                        target_info = display_detections(
                                            captured.frame,
                                            captured.detections,
                                            segmenter=None,  # Already segmented
                                            window_name="Maxim Observation",
                                            wait_ms=1,
                                            show_pose=True,
                                        ) if self.verbose else None
                                    else:
                                        # Fall back to passive_observation if no segmented frame
                                        if self.verbosity >= 2:
                                            self.log.debug(
                                                "Display fallback to passive_observation (epoch=%d, captured=%s)",
                                                self.current_epoch,
                                                captured is not None,
                                            )
                                        target_info = passive_observation(self, photo, show=self.verbose)
                                else:
                                    # No CaptureManager, use legacy behavior
                                    if self.verbosity >= 2:
                                        self.log.debug(
                                            "Display using legacy passive_observation (epoch=%d)",
                                            self.current_epoch,
                                        )
                                    target_info = passive_observation(self, photo, show=self.verbose)

                                # Store target info for agentic system to act on
                                if target_info is not None:
                                    try:
                                        self._last_detection_target = target_info
                                    except Exception:
                                        pass
                        except Exception as e:
                            if self.verbosity >= 2:
                                self.log.exception(
                                    "Observation step failed (mode=%s)",
                                    getattr(self, "mode", "reflection"),
                                )
                            else:
                                self.log.error(
                                    "Observation step failed (mode=%s): %s",
                                    getattr(self, "mode", "reflection"),
                                    e,
                                )
        finally:
            stop_event.set()
            try:
                mini = getattr(self, "mini", None)
                if mini is not None:
                    try:
                        mini.stop_recording()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                with media_lock:
                    self._release_media()
            except Exception:
                pass
            for t in threads:
                t.join(timeout=2.0)

            # Signal transcription worker to shut down via file
            if transcribe_shutdown_file is not None:
                try:
                    # Create shutdown signal file
                    with open(transcribe_shutdown_file, "w") as f:
                        f.write("shutdown\n")
                except Exception:
                    pass

            if transcribe_process is not None:
                # Phase 1: Wait briefly for graceful shutdown via file signal
                try:
                    transcribe_process.join(timeout=2.0)
                except Exception:
                    pass

                # Phase 2: Send SIGTERM if still alive
                if transcribe_process.is_alive():
                    try:
                        transcribe_process.terminate()
                    except Exception:
                        pass
                    # Also send SIGTERM directly via os.kill for reliability
                    try:
                        import signal
                        os.kill(transcribe_process.pid, signal.SIGTERM)
                    except Exception:
                        pass
                    try:
                        transcribe_process.join(timeout=1.0)
                    except Exception:
                        pass

                # Phase 3: Force kill if still alive
                if transcribe_process.is_alive():
                    self.log.warning("Transcription process did not respond to SIGTERM, sending SIGKILL")
                    try:
                        transcribe_process.kill()
                    except Exception:
                        pass
                    # Also send SIGKILL directly for reliability
                    try:
                        import signal
                        os.kill(transcribe_process.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    try:
                        transcribe_process.join(timeout=1.0)
                    except Exception:
                        pass

                # Final check
                if transcribe_process.is_alive():
                    self.log.error("Transcription process could not be terminated (pid=%d)", transcribe_process.pid)

            # Cleanup shutdown file
            if transcribe_shutdown_file is not None:
                try:
                    if os.path.exists(transcribe_shutdown_file):
                        os.remove(transcribe_shutdown_file)
                except Exception:
                    pass

            self._motor_queue = None
            self._media_lock = None
            self._live_stop_event = None
            self.shutdown()

    def sleep(
        self,
        home_dir: Optional[str] = None,
        *,
        parallel: bool = True,
        run_id: str | None = None,
    ):
        """
        Audio-only loop: streams audio continuously (and transcribes when enabled),
        without waking the robot motors. Runs until interrupted.

        Movement behavior:
            - If self.sleeping is True, Reachy is already in sleep pose; no movement.
            - If self.sleeping is False, move Reachy to sleep pose first.

        Args:
            home_dir: Home directory for artifacts.
            parallel: Run in parallel mode.
            run_id: Run identifier for logging.
        """
        self.audio = True
        self.mode = "sleep"
        if int(getattr(self, "verbosity", 0) or 0) >= 2:
            try:
                self.log.debug("Entering sleep: reuse live loop (vision=False, motor=False).")
            except Exception:
                pass

        # Only move to sleep pose if not already sleeping
        if not self.sleeping:
            if self._robot is not None:
                self._robot.goto_sleep()
            self.sleeping = True

        return self.live(
            home_dir=home_dir,
            parallel=parallel,
            vision=False,
            motor=False,
            wake_up=False,
            run_id=run_id,
        )

    def _run_headless_loop(self, stop_event: threading.Event) -> None:
        """Event-driven loop for headless mode (no media capture).

        Replaces the tight frame-capture loop with a slow poll that:
        - Processes gateway events (incoming SMS, webhooks) if comms enabled
        - Keeps the main thread alive while the agentic thread runs
        - Uses 0.5s sleep (not 30Hz polling) to minimize CPU usage
        """
        gateway = getattr(self, "_gateway", None)
        while not stop_event.is_set():
            # Poll gateway for incoming messages (SMS, webhook, etc.)
            if gateway is not None:
                try:
                    gateway.poll()
                except Exception as e:
                    log_swallowed_exception(e, operation="gateway_poll")
            # Sleep until next poll (event-driven, not frame-rate-driven)
            stop_event.wait(timeout=0.5)

    def _enqueue_motor(self, fn, *args, **kwargs):
        q = getattr(self, "_motor_queue", None)
        if q is None:
            return fn(*args, **kwargs)

        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait((fn, args, kwargs))
        except queue.Full:
            pass
        return None

    def _clear_motor_queue(self) -> int:
        """Clear all pending motor commands from the queue.

        Returns:
            Number of commands that were cleared.
        """
        q = getattr(self, "_motor_queue", None)
        if q is None:
            return 0

        cleared = 0
        try:
            while True:
                q.get_nowait()
                cleared += 1
        except queue.Empty:
            pass

        if cleared > 0:
            self.log.debug("Cleared %d pending motor commands from queue", cleared)
        return cleared

    def act(self, action):
        for movement in self.actions[action]["movements"]:
            self.move(
                movement[0],
                movement[1],
                movement[2],
                movement[3],
                movement[4],
                movement[5],
                movement[6]
            )
            time.sleep(movement[6])

    def speak(self, samples, sample_rate: int = 16000):
        """Push audio samples to Reachy Mini speaker, with local fallback.

        Tries to play through Reachy's speaker first. If that fails (e.g.,
        when using WebRTC remote connection), falls back to playing through
        local computer speakers.

        Args:
            samples: Audio samples as int16 numpy array.
            sample_rate: Sample rate of the audio (default 16000).

        Returns:
            True if audio was played successfully, False otherwise.
        """
        if samples is None or len(samples) == 0:
            return False

        # Check if local audio is preferred (set via environment)
        prefer_local = os.environ.get("MAXIM_TTS_LOCAL", "").lower() in ("1", "true", "yes")

        # Try Reachy speaker first (unless local is preferred)
        if not prefer_local:
            try:
                self.mini.media.push_audio_sample(samples)
                return True
            except Exception as e:
                # Check if this is a "not implemented" error (WebRTC limitation)
                error_str = str(e).lower()
                if "not implemented" in error_str or "webrtc" in error_str:
                    self.log.info("Reachy speaker not available (WebRTC), using local audio")
                else:
                    warn("Failed to play audio on Reachy: %s", e, logger=self.log)
                    self._note_connection_failure("audio", e)

        # Fall back to local audio playback
        try:
            from maxim.utils.audio import play_audio_local
            success = play_audio_local(samples, sample_rate=sample_rate, blocking=False)
            if success:
                self.log.debug("Playing audio through local speakers")
            return success
        except Exception as e:
            warn("Local audio playback failed: %s", e, logger=self.log)
            return False

    def look(self, save_file = None, show = True, release = False):
        # Grab frame from reachy mini camera
        if self.mini is None:
            return None
        frame = None
        try:
            try:
                frame = self.mini.media.get_frame()
            except Exception as e:
                warn("Failed to capture frame: %s", e, logger=self.log)
                return None

            is_empty = frame is None
            if not is_empty and hasattr(frame, "size"):
                is_empty = frame.size == 0
            if is_empty:
                warn("Empty frame received.", logger=self.log)
                return None

            # Show frame if requested
            if show:
                try:
                    show_photo(frame)
                except Exception as e:
                    warn("Failed to display frame: %s", e, logger=self.log)
                finally:
                    self._release_cv2()

            # Save frame to file if specified
            if save_file is not None:
                os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
                try:
                    ok = cv2.imwrite(save_file, frame)
                    if not ok:
                        warn("Failed to write image to '%s'.", save_file, logger=self.log)
                except Exception as e:
                    warn("Failed to write image to '%s': %s", save_file, e, logger=self.log)

            return frame
        finally:
            if release:
                self._release_media()

    def listen(self, save_file: Optional[str] = None):
        # Grab audio samples from Reachy Mini microphone.
        if self.mini is None:
            return None
        try:
            sample = self.mini.media.get_audio_sample()
        except Exception as e:
            warn("Failed to capture audio sample: %s", e, logger=self.log)
            return None

        if sample is None or len(sample) == 0:
            warn("Empty audio sample received.", logger=self.log)
            return None

        # Resample to local rate.
        input_rate = None
        output_rate = None
        try:
            input_rate = int(self.mini.media.get_input_audio_samplerate())
            output_rate = int(self.mini.media.get_output_audio_samplerate())
        except Exception:
            input_rate = None
            output_rate = None

        sample_arr = np.asarray(sample)
        sample_arr = resample_audio(sample_arr, input_rate, output_rate)

        if save_file:
            os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
            try:
                wav_rate = int(output_rate or input_rate or 16000)
                wf = wave.open(save_file, "wb")
                try:
                    channels = 1 if sample_arr.ndim == 1 else int(sample_arr.shape[1])
                    wf.setnchannels(channels)
                    wf.setsampwidth(2)
                    wf.setframerate(wav_rate)
                    sample_i16 = to_int16(sample_arr)
                    wf.writeframes(np.ascontiguousarray(sample_i16).tobytes())
                finally:
                    wf.close()
            except Exception as e:
                warn("Failed to write audio to '%s': %s", save_file, e, logger=self.log)

        return sample_arr
