import os
import queue
import threading

#os.environ["REACHY_MEDIA_BACKEND"] = "zenoh"
#os.environ["REACHY_DISABLE_WEBRTC"] = "1"
#os.environ["GST_DISABLE_REGISTRY_FORK"] = "1"

import json, random, uuid
import time, atexit, cv2
import logging
import multiprocessing as mp
import wave
from typing import Optional

import numpy as np

from reachy_mini import ReachyMini

from maxim.motion.movement import load_actions, load_poses, move_antenna, move_head
from maxim.utils.audio import resample_audio, to_int16
from maxim.utils.data_management import TrainingSampleLogger, build_home
from maxim.utils.logging import configure_logging, warn
from maxim.utils.queueing import put_latest

from maxim.data.camera.display import show_photo
from maxim.inference.observation import (
    face_observation,
    motor_cortex_control,
    passive_observation,
    passive_listening,
)
from maxim.models.vision.segmentation import YOLO8

os.environ["PYOPENGL_PLATFORM"] = "egl"

class Maxim:
    """
    A class for orchestracting models and agents with Reachy-Mini's.
    """

    def __init__(
        self,
        robot_name: str = "reachy_mini",
        timeout: float = 30.0,
        media_backend: str = "default",  # avoid WebRTC/GStreamer if signalling is down
        home_dir: str = "data/",
        epochs: int = 1000,
        *,
        verbosity: int = 0,
        verbose: bool = False,
        mode: str = "passive-interaction",
        train: bool | None = None,
        audio: bool = True,
        audio_len: float = 5.0):
        
        #
        self.verbosity = int(verbosity or 0)
        if verbose and self.verbosity <= 0:
            self.verbosity = 1
        self.verbose = self.verbosity > 0

        if self.verbose:
            configure_logging(self.verbosity)

        self.log = logging.getLogger("maxim.Maxim")

        self.alive = True
        self._closed = False
        self._woke_up = False

        self.name = robot_name or os.getenv("MAXIM_ROBOT_NAME", "reachy_mini")
        self.log.info("Connecting to Reachy Mini '%s'...", self.name)
        self.start = time.time()
        self.duration = 1.0
        self.home_dir = home_dir

        self.current_epoch = 0
        self.epochs = epochs
        mode = str(mode or "passive-interaction").strip().lower()
        if train is not None:
            mode = "train" if bool(train) else "live"
        self.mode = mode
        self.train = self.mode == "train"

        self.observation_period = 1
        self.audio = bool(audio)
        try:
            self.audio_len = float(audio_len)
        except Exception:
            self.audio_len = 5.0
        if self.audio_len <= 0:
            self.audio_len = 5.0

        self.video_fps = 20.0

        self.interests = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        self.actions = load_actions()
        self.poses = load_poses()

        # robot_name must match the daemon namespace (default: reachy_mini).
        # localhost_only=False enables zenoh peer discovery across the LAN.
        self.mini = ReachyMini(
            robot_name=self.name,
            localhost_only=False,
            spawn_daemon=False,
            use_sim=False,
            timeout=timeout,
            media_backend=media_backend,
        )
        self.log.info("Connected. Starting recording...")
        try:
            self.mini.start_recording()
        except Exception as e:
            self.log.warning("Failed to start recording: %s", e)

        self.x = 0.01
        self.y = 0.01
        self.z = 0.01

        self.roll = 0.01
        self.pitch = 0.01
        self.yaw = 0.01

        centered = None
        try:
            centered = getattr(self, "poses", {}).get("centered")
        except Exception:
            centered = None
        if isinstance(centered, (list, tuple)) and len(centered) >= 6:
            try:
                self.x = float(centered[0])
                self.y = float(centered[1])
                self.z = float(centered[2])
                self.roll = float(centered[3])
                self.pitch = float(centered[4])
                self.yaw = float(centered[5])
            except Exception:
                pass

        self._default_head_pose = {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "roll": float(self.roll),
            "pitch": float(self.pitch),
            "yaw": float(self.yaw),
        }

        self._training_paused = threading.Event()
        self._observation_lock = threading.Lock()

        self.key_responses = self._load_key_responses()

        self.movement_model = None
        self.motor_history: list[dict] = []

        atexit.register(self.shutdown)

    def _load_key_responses(self) -> dict[str, dict]:
        default = {
            "c": {"call": "center_vision", "pause_training": True},
            "u": {"call": "mark_trainable_moment"},
        }

        candidates: list[str] = []
        env_path = str(os.getenv("MAXIM_KEY_RESPONSES", "")).strip()
        if env_path:
            candidates.append(env_path)
        candidates.append(os.path.join(os.getcwd(), "data", "util", "key_responses.json"))
        candidates.append(os.path.join(os.getcwd(), "key_responses.json"))
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            candidates.append(os.path.join(repo_root, "data", "util", "key_responses.json"))
            candidates.append(os.path.join(repo_root, "key_responses.json"))
        except Exception:
            pass

        raw = None
        for path in candidates:
            if path and os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as fp:
                        raw = json.load(fp)
                    break
                except Exception as e:
                    warn("Failed to load key responses from '%s': %s", path, e, logger=self.log)
                    return default

        if not isinstance(raw, dict):
            return default

        parsed: dict[str, dict] = {}
        for key, spec in raw.items():
            if not isinstance(key, str) or not key:
                continue

            if isinstance(spec, str):
                parsed[key] = {"call": spec}
                continue

            if not isinstance(spec, dict):
                continue

            call = spec.get("call") or spec.get("method")
            if not isinstance(call, str) or not call:
                continue

            parsed[key] = {
                "call": call,
                "args": spec.get("args") if isinstance(spec.get("args"), list) else [],
                "kwargs": spec.get("kwargs") if isinstance(spec.get("kwargs"), dict) else {},
                "pause_training": bool(spec.get("pause_training", False)),
            }

        return parsed or default

    def _start_key_listener(self, stop_event: threading.Event) -> threading.Thread | None:
        if not isinstance(getattr(self, "key_responses", None), dict) or not self.key_responses:
            return None

        def _worker() -> None:
            try:
                import select
                import sys
                import termios
                import tty
            except Exception as e:
                warn("Keyboard listener unavailable: %s", e, logger=self.log)
                return

            stdin = sys.stdin
            if stdin is None or not hasattr(stdin, "isatty") or not stdin.isatty():
                return

            try:
                fd = stdin.fileno()
                old = termios.tcgetattr(fd)
            except Exception:
                return

            try:
                tty.setcbreak(fd)
                try:
                    new = termios.tcgetattr(fd)
                    new[3] &= ~termios.ECHO
                    termios.tcsetattr(fd, termios.TCSADRAIN, new)
                except Exception:
                    pass

                while not stop_event.is_set():
                    try:
                        ready, _, _ = select.select([stdin], [], [], 0.1)
                    except Exception:
                        continue
                    if not ready:
                        continue
                    try:
                        ch = stdin.read(1)
                    except Exception:
                        continue
                    if not ch or ch in ("\n", "\r"):
                        continue
                    self._handle_keypress(ch)
            finally:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                except Exception:
                    pass

        return threading.Thread(target=_worker, name="maxim.keyboard", daemon=True)

    def _handle_keypress(self, key: str) -> None:
        try:
            spec = getattr(self, "key_responses", {}).get(key)
        except Exception:
            spec = None
        if not isinstance(spec, dict):
            return

        call = spec.get("call")
        if not isinstance(call, str) or not call:
            return

        pause_training = bool(spec.get("pause_training", False)) and bool(getattr(self, "train", False))
        if pause_training:
            self._training_paused.set()

        try:
            with self._observation_lock:
                fn = getattr(self, call, None)
                if not callable(fn):
                    warn("Unknown key response for '%s': %s", key, call, logger=self.log)
                    return
                args = spec.get("args") if isinstance(spec.get("args"), list) else []
                kwargs = spec.get("kwargs") if isinstance(spec.get("kwargs"), dict) else {}
                fn(*args, **kwargs)
        except Exception as e:
            warn("Key '%s' action failed: %s", key, e, logger=self.log)
        finally:
            if pause_training:
                self._training_paused.clear()

    def _release_cv2(self) -> None:
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass

    def _release_media(self) -> None:
        mini = getattr(self, "mini", None)
        if mini is None:
            return

        try:
            mini.media.close()
        except Exception as e:
            warn("Failed to close media: %s", e, logger=getattr(self, "log", None))

        self._release_cv2()
    
    def live(
        self,
        home_dir: Optional[str] = None,
        *,
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

        build_home(self.home_dir)

        log_path = os.path.join(self.home_dir, "logs", f"reachy_log_{run_id}.log")
        configure_logging(self.verbosity, log_file=log_path)

        video_path = os.path.join(self.home_dir, "videos", f"reachy_video_{run_id}.mp4")
        audio_path = os.path.join(self.home_dir, "audio", f"reachy_audio_{run_id}.wav")
        transcript_path = os.path.join(self.home_dir, "transcript", f"reachy_transcript_{run_id}.jsonl")
        chunk_dir = os.path.join(self.home_dir, "audio", "chunks")

        self.run_id = run_id
        self.run_start_ts = time.time()
        self.log_path = log_path
        self.video_path = video_path
        self.audio_path = audio_path
        self.transcript_path = transcript_path

        try:
            prev_logger = getattr(self, "_training_logger", None)
            if prev_logger is not None:
                prev_logger.stop(timeout=0.5)
        except Exception:
            pass

        try:
            training_dir = os.path.join(self.home_dir, "training")
            self._training_logger = TrainingSampleLogger(training_dir)
            self._training_logger.start()
        except Exception as e:
            self._training_logger = None
            warn("Failed to start training sample logger: %s", e, logger=self.log)

        self.log.info(
            "Starting live loop (home_dir=%s, epochs=%d, observation_period=%s, mode=%s, audio=%s, audio_len=%.1fs)",
            self.home_dir,
            int(self.epochs),
            str(getattr(self, "observation_period", None)),
            str(getattr(self, "mode", "passive-interaction")),
            str(bool(getattr(self, "audio", True))),
            float(getattr(self, "audio_len", 0.0) or 0.0),
        )
        if vision:
            self.log.info("Recording video: %s", video_path)
        if self.audio:
            self.log.info("Recording audio: %s", audio_path)
            self.log.info("Transcripts: %s", transcript_path)

        self.awaken(vision=bool(vision), motor=bool(motor), audio=bool(self.audio), wake_up=bool(wake_up))

        media_lock = threading.Lock()
        stop_event = threading.Event()

        frame_obs_queue: queue.Queue = queue.Queue(maxsize=1)
        frame_save_queue: queue.Queue = queue.Queue(maxsize=512)
        audio_save_queue: queue.Queue = queue.Queue(maxsize=512) if self.audio else None

        motor_queue: queue.Queue = queue.Queue(maxsize=1)
        self._motor_queue = motor_queue if parallel and motor else None

        audio_input_rate = None
        audio_output_rate = None
        if self.audio:
            try:
                audio_input_rate = int(self.mini.media.get_input_audio_samplerate())
                audio_output_rate = int(self.mini.media.get_output_audio_samplerate())
            except Exception as e:
                warn("Failed to read audio sample rates: %s", e, logger=self.log)

        transcribe_queue = None
        transcribe_process = None
        if self.audio and parallel:
            os.makedirs(chunk_dir, exist_ok=True)
            try:
                from maxim.data.audio.sound import transcription_worker

                ctx = mp.get_context("spawn")
                transcribe_queue = ctx.Queue(maxsize=64)
                transcribe_process = ctx.Process(
                    target=transcription_worker,
                    args=(transcribe_queue, transcript_path),
                    kwargs={
                        "model_size_or_path": "tiny",
                        "device": "cpu",
                        "compute_type": "int8",
                        "language": "en",
                        "beam_size": 1,
                        "vad_filter": True,
                        "cleanup_chunks": True,
                        "verbosity": int(self.verbosity or 0),
                        "log_file": log_path,
                    },
                    daemon=True,
                )
                transcribe_process.start()
                time.sleep(0.1)
                if not transcribe_process.is_alive():
                    warn(
                        "Transcription worker exited immediately (is `faster-whisper` installed and model available?).",
                        logger=self.log,
                    )
                    transcribe_queue = None
                    transcribe_process = None
            except Exception as e:
                transcribe_queue = None
                transcribe_process = None
                warn("Failed to start transcription worker: %s", e, logger=self.log)

        def _motor_worker() -> None:
            while not stop_event.is_set():
                try:
                    fn, args, kwargs = motor_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    fn(*args, **kwargs)
                except Exception as e:
                    warn("Motor command failed: %s", e, logger=self.log)

        def _frame_capture_worker() -> None:
            min_period = 1.0 / float(getattr(self, "video_fps", 20.0) or 20.0)
            last_ts = 0.0
            while not stop_event.is_set():
                frame = None
                try:
                    with media_lock:
                        frame = self.mini.media.get_frame()
                except Exception as e:
                    warn("Failed to capture frame: %s", e, logger=self.log)
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

        def _audio_capture_worker() -> None:
            if not self.audio or audio_save_queue is None:
                return

            while not stop_event.is_set():
                sample = None
                try:
                    with media_lock:
                        sample = self.mini.media.get_audio_sample()
                except Exception as e:
                    warn("Failed to capture audio sample: %s", e, logger=self.log)
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
                    warn("Failed to process audio sample: %s", e, logger=self.log)
                    time.sleep(0.01)
                    continue

                now = time.time()
                try:
                    audio_save_queue.put((now, sample_i16, audio_output_rate or audio_input_rate), timeout=0.5)
                except queue.Full:
                    audio_save_queue.put((now, sample_i16, audio_output_rate or audio_input_rate))

        def _video_writer_worker() -> None:
            writer = None
            opened = False
            disabled = False
            width = None
            height = None
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
                        fps = float(getattr(self, "video_fps", 20.0) or 20.0)
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
                            warn("Failed to open video writer for '%s'.", video_path, logger=self.log)
                            disabled = True
                    except Exception as e:
                        warn("Failed to initialize video writer: %s", e, logger=self.log)
                        writer = None
                        disabled = True

                if opened and writer is not None:
                    try:
                        writer.write(frame_arr)
                    except Exception as e:
                        warn("Failed to write video frame: %s", e, logger=self.log)

                frame_save_queue.task_done()

            try:
                if writer is not None:
                    writer.release()
            except Exception:
                pass

        def _audio_writer_worker() -> None:
            if not self.audio or audio_save_queue is None:
                return

            os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)
            os.makedirs(chunk_dir, exist_ok=True)

            sample_rate = int(audio_output_rate or audio_input_rate or 16000)
            chunk_frames = None
            if transcribe_queue is not None:
                chunk_frames = int(float(getattr(self, "audio_len", 5.0) or 5.0) * float(sample_rate))
                chunk_frames = max(chunk_frames, sample_rate)  # at least 1s

            wf = wave.open(audio_path, "wb")
            channels = None
            pending_tasks: list[dict] = []
            buffer: list[np.ndarray] = [] if chunk_frames is not None else []
            buffered_frames = 0
            total_frames = 0
            chunk_index = 0

            def _flush_pending() -> None:
                nonlocal pending_tasks
                if transcribe_queue is None:
                    return
                while pending_tasks:
                    try:
                        transcribe_queue.put_nowait(pending_tasks[0])
                        pending_tasks.pop(0)
                    except Exception:
                        break

            def _write_chunk(chunk_arr: np.ndarray, start_frame: int) -> None:
                nonlocal chunk_index
                if transcribe_queue is None:
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
                        warn("Failed to write audio frames: %s", e, logger=self.log)

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

                if transcribe_queue is not None:
                    _flush_pending()
                    deadline = time.time() + 10.0
                    while pending_tasks and time.time() < deadline:
                        try:
                            transcribe_queue.put(pending_tasks.pop(0), timeout=1.0)
                        except Exception:
                            continue

        threads: list[threading.Thread] = []
        key_thread = self._start_key_listener(stop_event)
        if key_thread is not None:
            threads.append(key_thread)
            key_thread.start()

        if parallel:
            if vision:
                threads.append(threading.Thread(target=_frame_capture_worker, name="maxim.capture.video", daemon=True))
                threads.append(threading.Thread(target=_video_writer_worker, name="maxim.write.video", daemon=True))
            if motor:
                threads.append(threading.Thread(target=_motor_worker, name="maxim.motor", daemon=True))
            if self.audio:
                threads.append(threading.Thread(target=_audio_capture_worker, name="maxim.capture.audio", daemon=True))
                threads.append(threading.Thread(target=_audio_writer_worker, name="maxim.write.audio", daemon=True))

            for t in threads:
                if t is not key_thread:
                    t.start()

        try:
            if not vision:
                self.log.info("Audio-only mode: recording until Ctrl+C.")
                while True:
                    time.sleep(0.25)
            else:
                while True:
                    if int(self.current_epoch) >= int(self.epochs):
                        self.log.info("Reached epochs limit (%d). Stopping.", int(self.epochs))
                        break

                    if parallel:
                        try:
                            frame_ts, photo = frame_obs_queue.get(timeout=2.0)
                        except queue.Empty:
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

                    self.current_epoch += 1

                    if self.observation_period and self.current_epoch % self.observation_period == 0:
                        try:
                            with self._observation_lock:
                                if getattr(self, "mode", "passive-interaction") == "passive-interaction":
                                    passive_observation(self, photo, show=self.verbose)
                                else:
                                    train_enabled = bool(getattr(self, "train", False)) and not self._training_paused.is_set()
                                    motor_cortex_control(
                                        self,
                                        self.movement_model,
                                        photo,
                                        train=train_enabled,
                                        show=self.verbose,
                                    )
                        except Exception as e:
                            if self.verbosity >= 2:
                                self.log.exception(
                                    "Observation step failed (mode=%s)",
                                    getattr(self, "mode", "passive-interaction"),
                                )
                            else:
                                self.log.error(
                                    "Observation step failed (mode=%s): %s",
                                    getattr(self, "mode", "passive-interaction"),
                                    e,
                                )
        finally:
            stop_event.set()
            for t in threads:
                t.join(timeout=2.0)

            if transcribe_queue is not None:
                try:
                    transcribe_queue.put(None)
                except Exception:
                    pass

            if transcribe_process is not None:
                try:
                    transcribe_process.join(timeout=5.0)
                except Exception:
                    pass

            self._motor_queue = None
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
        """
        self.audio = True
        return self.live(
            home_dir=home_dir,
            parallel=parallel,
            vision=False,
            motor=False,
            wake_up=False,
            run_id=run_id,
        )

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

    def center_vision(self, *, duration: Optional[float] = None) -> None:
        return self.goto_pose("centered", duration=duration)

    def mark_trainable_moment(self) -> None:
        sample = getattr(self, "_last_motor_sample", None)
        training_logger = getattr(self, "_training_logger", None)
        if training_logger is None:
            warn("Training sample logger is not running.", logger=self.log)
            return
        if not isinstance(sample, dict) or not sample:
            warn("No recent motor sample to mark yet.", logger=self.log)
            return

        record = dict(sample)
        record["user_marked"] = True
        record["mark_time"] = time.time()
        record["mark_id"] = uuid.uuid4().hex
        record["marked_from_sample_id"] = record.get("sample_id")

        try:
            training_logger.log_motor_sample(record, flush=True)
        except Exception as e:
            warn("Failed to mark trainable moment: %s", e, logger=self.log)

    def goto_pose(self, name: str = "centered", *, duration: Optional[float] = None) -> None:
        pose = None
        try:
            pose = getattr(self, "poses", {}).get(name)
        except Exception:
            pose = None

        if isinstance(pose, (list, tuple)) and len(pose) >= 6:
            try:
                self.x = float(pose[0])
                self.y = float(pose[1])
                self.z = float(pose[2])
                self.roll = float(pose[3])
                self.pitch = float(pose[4])
                self.yaw = float(pose[5])
                if duration is None and len(pose) >= 7:
                    duration = float(pose[6])
            except Exception:
                pose = None

        if pose is None:
            fallback = getattr(self, "_default_head_pose", None)
            if not isinstance(fallback, dict):
                fallback = {}
            self.x = float(fallback.get("x", 0.0) or 0.0)
            self.y = float(fallback.get("y", 0.0) or 0.0)
            self.z = float(fallback.get("z", 0.0) or 0.0)
            self.roll = float(fallback.get("roll", 0.0) or 0.0)
            self.pitch = float(fallback.get("pitch", 0.0) or 0.0)
            self.yaw = float(fallback.get("yaw", 0.0) or 0.0)

        if duration is None:
            duration = float(getattr(self, "duration", 0.5) or 0.5)

        try:
            self._enqueue_motor(
                move_head,
                self.mini,
                self.x,
                self.y,
                self.z,
                self.roll,
                self.pitch,
                self.yaw,
                float(duration),
            )
        except Exception as e:
            warn("Failed to center vision: %s", e, logger=self.log)

        try:
            time.sleep(float(duration))
        except Exception:
            pass

    def look_at_image(
        self,
        u: int,
        v: int,
        *,
        duration: Optional[float] = None,
        perform_movement: bool = True,
    ) -> None:
        if duration is None:
            duration = getattr(self, "duration", 0.5)
        self._enqueue_motor(
            self.mini.look_at_image,
            int(u),
            int(v),
            duration=float(duration),
            perform_movement=bool(perform_movement),
        )

    def move(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        duration: Optional[float] = None) -> None:

        """
        Docstring for move
        
        :param self: Description
        :param x: Description
        :type x: Optional[float]
        :param y: Description
        :type y: Optional[float]
        :param z: Description
        :type z: Optional[float]
        :param roll: Description
        :type roll: Optional[float]
        :param pitch: Description
        :type pitch: Optional[float]
        :param yaw: Description
        :type yaw: Optional[float]
        :param duration: Description
        :type duration: Optional[float]
        """ 
        
        # Update duration if specified
        if duration is not None:
            self.duration = duration


        # Update only specified parameters
        updates = {
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }
        for attr, val in updates.items():
            if val is not None:
                setattr(self, attr, val)

        # Execute head movement
        self._enqueue_motor(
            move_head,
            self.mini,
            self.x,
            self.y,
            self.z,
            self.roll,
            self.pitch,
            self.yaw,
            self.duration,
        )

    def move_antenna(
        self,
        right: Optional[float] = None,
        left: Optional[float] = None,
        angle: Optional[float] = None,
        duration: Optional[float] = None,
        method: str = "minjerk",
        degrees: bool = True,
        relative: bool = False,
    ) -> None:
        if angle is not None:
            right = angle
            left = angle
        if duration is None:
            duration = self.duration

        self._enqueue_motor(
            move_antenna,
            self.mini,
            right=right,
            left=left,
            duration=duration,
            method=method,
            degrees=degrees,
            relative=relative,
        )

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
    
    def speak(self, samples):
        # Push audio samples to reachy mini speaker
        self.mini.media.push_audio_sample(samples)
        return

    def look(self, save_file = None, show = True, release = False):
        # Grab frame from reachy mini camera
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

    def learn(self):
        return
    
    def journal(self):
        entry = {
            "date": time.time(),
            "epoch": self.current_epoch,
        }
        return entry
    
    def awaken(self, vision: bool = True, motor: bool = True, audio: bool = True, wake_up: bool = True):
        # Load models
        if vision:
            self.log.info("Loading vision models (YOLOv8 seg+pose)...")
            self.segmenter = YOLO8(pose_model=True) # Visual segmentation + pose model

        if motor and self.movement_model is None:
            try:
                from maxim.models.movement.motor_cortex import LayerScale, MotorCortex
                from maxim.utils import config as motor_config

                self.log.info("Initializing motor cortex...")
                cfg = motor_config.build(motor_config.DEFAULT_SAVE_ROOT)
                self.movement_model = MotorCortex(cfg)

                checkpoint_path = getattr(cfg, "checkpoint_path", None)
                legacy_checkpoint_path = None
                if checkpoint_path and not os.path.exists(checkpoint_path):
                    try:
                        legacy_checkpoint_path = (
                            motor_config.LEGACY_SAVE_ROOT / motor_config.DEFAULT_CHECKPOINT_FILENAME
                        ).as_posix()
                    except Exception:
                        legacy_checkpoint_path = None

                load_path = None
                for candidate in (checkpoint_path, legacy_checkpoint_path):
                    if candidate and os.path.exists(candidate):
                        load_path = candidate
                        break

                if load_path:
                    try:
                        import keras

                        if load_path != checkpoint_path:
                            self.log.info("Loading legacy motor checkpoint: %s", load_path)
                        else:
                            self.log.info("Loading motor checkpoint: %s", load_path)
                        loaded = keras.models.load_model(
                            load_path,
                            custom_objects={
                                "LayerScale": LayerScale,
                                "MotorCortex": MotorCortex,
                                "motor_cortex": MotorCortex,
                            },
                        )
                        self.movement_model.model = loaded
                    except Exception as e:
                        self.log.warning("Failed to load motor checkpoint '%s': %s", load_path, e)
                else:
                    self.log.info("No motor checkpoint found; starting fresh.")
            except Exception as e:
                self.movement_model = None
                self.log.warning("Motor cortex unavailable: %s", e)

        if wake_up:
            # Wake up Reachy
            self.log.info("Waking up Reachy...")
            self.mini.wake_up()
            self._woke_up = True

        return

    def shutdown(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True

        try:
            training_logger = getattr(self, "_training_logger", None)
            if training_logger is not None:
                training_logger.stop(timeout=2.0)
        except Exception:
            pass
        self._training_logger = None

        # Persist the motor cortex state (best-effort; never blocks shutdown).
        try:
            movement_model = getattr(self, "movement_model", None)
            if movement_model is not None:
                cfg = getattr(movement_model, "config", None)
                checkpoint_path = getattr(cfg, "checkpoint_path", None) if cfg is not None else None
                save_dir = getattr(cfg, "save_dir", None) if cfg is not None else None

                if not checkpoint_path:
                    try:
                        from maxim.utils import config as motor_config

                        checkpoint_path = (
                            motor_config.DEFAULT_SAVE_ROOT / motor_config.DEFAULT_CHECKPOINT_FILENAME
                        ).as_posix()
                        save_dir = save_dir or motor_config.DEFAULT_SAVE_ROOT.as_posix()
                    except Exception:
                        checkpoint_path = os.path.join(self.home_dir, "models", "motor_cortex.keras")

                os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                to_save = getattr(movement_model, "model", movement_model)
                if hasattr(to_save, "save"):
                    try:
                        to_save.save(checkpoint_path)
                        self.log.info("Saved motor model: %s", checkpoint_path)
                    except Exception as e:
                        self.log.warning("Failed to save motor model to '%s': %s", checkpoint_path, e)

                history = getattr(self, "motor_history", None)
                if history is not None:
                    if save_dir:
                        history_path = os.path.join(str(save_dir).rstrip("/"), "motor_cortex_history.json")
                    else:
                        history_path = os.path.join(os.path.dirname(checkpoint_path) or ".", "motor_cortex_history.json")

                    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
                    tmp_path = f"{history_path}.tmp"
                    payload = {
                        "time": time.time(),
                        "checkpoint_path": checkpoint_path,
                        "train_step": int(getattr(movement_model, "_train_step", 0) or 0),
                        "records": history,
                    }
                    with open(tmp_path, "w", encoding="utf-8") as fp:
                        json.dump(payload, fp, indent=2, default=str)
                    os.replace(tmp_path, history_path)
                    try:
                        num_records = len(history)
                    except Exception:
                        num_records = 0
                    self.log.info("Saved motor history: %s (%d records)", history_path, num_records)

                    try:
                        from maxim.utils.plotting import update_motor_cortex_loss_plot
                        from maxim.utils.plotting import update_motor_cortex_pixel_error_plot

                        update_motor_cortex_loss_plot(history, save_dir=save_dir)
                        update_motor_cortex_pixel_error_plot(history, save_dir=save_dir)
                    except Exception as e:
                        self.log.warning("Failed to write motor plots: %s", e)
        except Exception as e:
            self.log.warning("Failed to save motor artifacts: %s", e)

        if getattr(self, "_woke_up", False):
            # Send Reachy to sleep
            try:
                self.mini.goto_sleep()
            except Exception as e:
                warn("Failed to send Reachy to sleep: %s", e, logger=getattr(self, "log", None))

        # Stop recording data
        try:
            self.mini.stop_recording()
        except Exception as e:
            warn("Failed to stop recording: %s", e, logger=getattr(self, "log", None))

        # Release the camera + any OpenCV resources
        self._release_media()


        return

if __name__ == "__main__":
    conscience = Maxim()
    
