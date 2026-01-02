import os
import queue
import threading

#os.environ["REACHY_MEDIA_BACKEND"] = "zenoh"
#os.environ["REACHY_DISABLE_WEBRTC"] = "1"
#os.environ["GST_DISABLE_REGISTRY_FORK"] = "1"

import json, random
import time, atexit, cv2
import logging
import math
import multiprocessing as mp
import wave
from typing import Optional

import numpy as np
from scipy.signal import resample, resample_poly

from reachy_mini import ReachyMini

from src.motion.movement import move_antenna, move_head, load_actions
from src.utils.data_management import build_home
from src.utils.logging import configure_logging, warn

from src.data.camera.display import show_photo
from src.inference.observation import (
    face_observation,
    motor_cortex_control,
    passive_observation,
    passive_listening,
)
from src.models.vision.segmentation import YOLO8

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
        home_dir: str = "experiments/maxim/",
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

        self.movement_model = None
        self.motor_history: list[dict] = []

        atexit.register(self.shutdown)

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
        transcript_path = os.path.join(self.home_dir, "text", f"reachy_transcript_{run_id}.jsonl")
        chunk_dir = os.path.join(self.home_dir, "audio", "chunks")

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

        def _put_latest(q: queue.Queue, item) -> None:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

        def _to_int16(arr: np.ndarray) -> np.ndarray:
            if arr.dtype == np.int16:
                return np.ascontiguousarray(arr)
            if np.issubdtype(arr.dtype, np.floating):
                clipped = np.clip(arr, -1.0, 1.0)
                return np.ascontiguousarray((clipped * 32767.0).astype(np.int16))
            return np.ascontiguousarray(np.clip(arr, -32768, 32767).astype(np.int16))

        def _resample_audio(sample: np.ndarray, input_rate: Optional[int], output_rate: Optional[int]) -> np.ndarray:
            if not input_rate or not output_rate or int(input_rate) == int(output_rate):
                return sample

            try:
                gcd = math.gcd(int(input_rate), int(output_rate))
                up = int(output_rate) // gcd
                down = int(input_rate) // gcd
                return resample_poly(sample, up, down, axis=0)
            except Exception:
                num_sample = int(int(output_rate) * len(sample) / int(input_rate))
                return resample(sample, num_sample)

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
                from src.data.audio.sound import transcription_worker

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

                _put_latest(frame_obs_queue, (now, frame))

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
                    sample_arr = _resample_audio(sample_arr, audio_input_rate, audio_output_rate)
                    sample_i16 = _to_int16(sample_arr)
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
                            _, photo = frame_obs_queue.get(timeout=2.0)
                        except queue.Empty:
                            if self.verbosity >= 2:
                                self.log.debug("Waiting for camera frame...")
                            continue
                    else:
                        photo = self.look(show=False)

                    if photo is None:
                        if self.verbosity >= 2:
                            self.log.debug("No frame captured.")
                        continue

                    self.current_epoch += 1

                    if self.observation_period and self.current_epoch % self.observation_period == 0:
                        try:
                            if getattr(self, "mode", "passive-interaction") == "passive-interaction":
                                passive_observation(self, photo, show=self.verbose)
                            else:
                                motor_cortex_control(
                                    self,
                                    self.movement_model,
                                    photo,
                                    train=bool(getattr(self, "train", False)),
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
        if input_rate and output_rate and input_rate != output_rate:
            try:
                gcd = math.gcd(int(input_rate), int(output_rate))
                up = int(output_rate) // gcd
                down = int(input_rate) // gcd
                sample_arr = resample_poly(sample_arr, up, down, axis=0)
            except Exception:
                num_sample = int(int(output_rate) * len(sample_arr) / int(input_rate))
                sample_arr = resample(sample_arr, num_sample)

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
                    if sample_arr.dtype != np.int16:
                        if np.issubdtype(sample_arr.dtype, np.floating):
                            clipped = np.clip(sample_arr, -1.0, 1.0)
                            sample_i16 = (clipped * 32767.0).astype(np.int16)
                        else:
                            sample_i16 = np.clip(sample_arr, -32768, 32767).astype(np.int16)
                    else:
                        sample_i16 = sample_arr
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
                from src.models.movement.motor_cortex import LayerScale, MotorCortex
                from src.utils import config as motor_config

                self.log.info("Initializing motor cortex...")
                cfg = motor_config.build(os.path.join("experiments", "models", "MotorCortex"))
                self.movement_model = MotorCortex(cfg)

                checkpoint_path = getattr(cfg, "checkpoint_path", None)
                if checkpoint_path and os.path.exists(checkpoint_path):
                    try:
                        import keras

                        self.log.info("Loading motor checkpoint: %s", checkpoint_path)
                        loaded = keras.models.load_model(
                            checkpoint_path,
                            custom_objects={
                                "LayerScale": LayerScale,
                                "MotorCortex": MotorCortex,
                                "motor_cortex": MotorCortex,
                            },
                        )
                        self.movement_model.model = loaded
                    except Exception as e:
                        self.log.warning("Failed to load motor checkpoint '%s': %s", checkpoint_path, e)
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

        # Persist the motor cortex state (best-effort; never blocks shutdown).
        try:
            movement_model = getattr(self, "movement_model", None)
            if movement_model is not None:
                cfg = getattr(movement_model, "config", None)
                checkpoint_path = getattr(cfg, "checkpoint_path", None) if cfg is not None else None
                save_dir = getattr(cfg, "save_dir", None) if cfg is not None else None

                if not checkpoint_path:
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
                        from src.utils.plotting import update_motor_cortex_loss_plot
                        from src.utils.plotting import update_motor_cortex_pixel_error_plot

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
    
