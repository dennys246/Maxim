import os
import queue
import threading

# CRITICAL: Set multiprocessing start method to 'spawn' BEFORE any other imports.
# On Linux, the default 'fork' method causes GStreamer segfaults because:
# - Thread pools become invalid when accessed from different processes
# - Mutexes become corrupted across process boundaries
# - Memory allocators experience concurrent access corruption
# See: https://www.blog.neudeep.com/howto/fixing-gstreamer-segmentation-faults-in-python-multiprocessing/
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set, ignore

# Detect Blackwell GPU and disable GStreamer/WebRTC BEFORE any other imports.
from maxim.utils.gpu_compat import detect_blackwell, is_connection_error, is_gpu_available
from maxim.utils.thread_manager import ThreadRegistry
from maxim.utils.response_config import (
    load_key_responses,
    load_phrase_responses,
    normalize_trigger_text,
    normalize_transcript_text,
)
from maxim.modes.state_manager import StateManager

_gpu_state = detect_blackwell()
_blackwell_detected = _gpu_state.blackwell_detected
_original_cuda_devices = _gpu_state.original_cuda_devices

import json, uuid
import re
import time, atexit, cv2
import logging
import wave
from typing import Any, Optional

import numpy as np

from maxim.motion.movement import load_actions, load_movement_thresholds, load_poses, move_antenna, move_head
from maxim.utils.audio import resample_audio, to_int16
from maxim.utils.data_management import CLIInputLogger, TrainingSampleLogger, VisionEventLogger, build_home
from maxim.utils.logging import configure_logging, log_swallowed_exception, warn
from maxim.utils.plotting import preflight_matplotlib_fonts, preload_matplotlib_fonts
from maxim.utils.queueing import put_latest

from maxim.data.camera.display import prepare_display, show_photo
from maxim.inference.observation import (
    DEFAULT_CLASS_WEIGHTS,
    NoveltyTracker,
    display_detections,
    passive_observation,
)
from maxim.models.vision.registry import build_segmentation_model

# Hardware abstraction layer for multi-robot support
from maxim.hardware import RobotRegistry
from maxim.hardware.reachy import ReachyMiniController
from maxim.hardware.simulation import SimulatedController

os.environ["PYOPENGL_PLATFORM"] = "egl"

# Initialize global robot registry with controller types
_robot_registry = RobotRegistry()
_robot_registry.register_controller_type("reachy_mini", ReachyMiniController)
_robot_registry.register_controller_type("simulated", SimulatedController)


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
        epochs: int | None = None,
        *,
        verbosity: int = 0,
        verbose: bool = False,
        mode: str = "exploration",
        train: bool | None = None,
        audio: bool = True,
        audio_len: float = 5.0,
        interactive: bool = True,
        novelty_tracker: "NoveltyTracker | None" = None,
        simulation: bool = False,
        robot_id: str | None = None):
        
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
        self.sleeping = False  # Track if Reachy is already in sleep pose

        # Central thread registry for coordinated shutdown
        # All threads created by Maxim should be registered here
        self._thread_registry = ThreadRegistry()

        # Hardware abstraction layer
        self._simulation = bool(simulation)
        self._robot_id = robot_id
        self._robot: ReachyMiniController | SimulatedController | None = None

        self.name = robot_name or os.getenv("MAXIM_ROBOT_NAME", "reachy_mini")
        if _blackwell_detected:
            self.log.info("Blackwell GPU detected - GStreamer hardware acceleration disabled")
        self.log.info("Connecting to Reachy Mini '%s'...", self.name)
        self._connect_kwargs = {
            "robot_name": self.name,
            "localhost_only": False,
            "spawn_daemon": False,
            "use_sim": False,
            "timeout": float(timeout),
            "media_backend": media_backend,
        }
        self._media_lock: threading.Lock | None = None
        self._reconnect_lock = threading.Lock()
        self._last_reconnect_ts = 0.0
        self._reconnect_cooldown_s = 20.0
        self._reconnect_window_s = 5.0
        self._reconnect_thresholds = {"motor": 3, "video": 5, "audio": 5}
        self._connection_failures = {
            "motor": {"count": 0, "last_ts": 0.0},
            "video": {"count": 0, "last_ts": 0.0},
            "audio": {"count": 0, "last_ts": 0.0},
        }
        self.start = time.time()
        self.duration = 1.0
        self.home_dir = home_dir

        # Load Matplotlib before Reachy/GStreamer so ft2font binds to stable libs.
        preload_matplotlib_fonts(
            cache_dir=os.path.join(self.home_dir, "matplotlib"),
            logger=self.log,
        )

        self.current_epoch = 0
        self._set_epochs(epochs)
        mode = str(mode or "exploration").strip().lower()
        if train is not None:
            mode = "train" if bool(train) else "live"
        self.mode = mode
        self.train = self.mode == "train"

        self.observation_period = 1
        self.audio = bool(audio)
        try:
            self.audio_len = float(audio_len)
        except (TypeError, ValueError) as e:
            log_swallowed_exception(e, operation="parse_audio_len", context={"audio_len": audio_len})
            self.audio_len = 5.0
        if self.audio_len <= 0:
            self.audio_len = 5.0

        self.video_fps = 20.0

        self.interactive = bool(interactive)

        # Salience boost classes: these get 2x priority in perception scoring.
        # All 80 COCO classes are always detected; this controls attention priority.
        self.interests = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        # Novelty tracking for attention prioritization
        # Novel objects get higher priority, familiar objects decay over time
        # Can be shared with DefaultNetwork for unified tracking
        if novelty_tracker is not None:
            self.novelty_tracker = novelty_tracker
        else:
            self.novelty_tracker = NoveltyTracker(
                focus_decay_seconds=10.0,  # Time for novelty to decay while focused
                recovery_seconds=20.0,     # Time for novelty to recover when not focused
                max_novelty=2.0,           # Novelty boost for new objects
                min_novelty=0.5,           # Minimum novelty for familiar objects
            )
        # Class weights for attention (default gives slight preference to people)
        self.class_weights = dict(DEFAULT_CLASS_WEIGHTS)

        self.actions = load_actions()
        self.poses = load_poses()
        self.movement_thresholds = load_movement_thresholds()
        self._head_max_step = {}
        try:
            head_cfg = self.movement_thresholds.get("head") if isinstance(self.movement_thresholds, dict) else None
            if isinstance(head_cfg, dict) and isinstance(head_cfg.get("max_step"), dict):
                self._head_max_step = dict(head_cfg.get("max_step") or {})
        except (KeyError, TypeError, AttributeError) as e:
            log_swallowed_exception(e, operation="load_head_max_step")
            self._head_max_step = {}

        # Connect to robot using hardware abstraction layer.
        # Supports both real Reachy Mini hardware and simulation mode.
        robot_type = "simulated" if self._simulation else "reachy_mini"
        effective_robot_id = self._robot_id or self.name

        self.log.info("Connecting to robot '%s' (type=%s)...", effective_robot_id, robot_type)

        # Use the global registry to connect
        self._robot = _robot_registry.connect_robot(
            robot_id=effective_robot_id,
            robot_type=robot_type,
            config={
                "robot_name": self.name,
                "media_backend": media_backend,
            } if not self._simulation else {
                "video_resolution": (640, 480),
                "simulate_delays": False,
            },
            timeout=timeout,
            set_primary=True,
        )

        if self._robot is None:
            raise RuntimeError(f"Failed to connect to robot: {effective_robot_id}")

        # On Blackwell GPUs, don't start recording - it will crash in GStreamer
        self._recording_started = False
        if not _blackwell_detected and not self._simulation:
            self.log.info("Connected. Starting recording...")
            try:
                self._robot.start_recording()
                self._recording_started = True
            except Exception as e:
                self.log.warning("Failed to start recording: %s", e)
        elif self._simulation:
            # Simulation always starts recording
            self._robot.start_recording()
            self._recording_started = True
            self.log.info("Connected to simulated robot.")
        else:
            self.log.info("Connected. (Recording disabled for Blackwell GPU compatibility)")

        self.x = 0.01
        self.y = 0.01
        self.z = 0.01

        self.roll = 0.01
        self.pitch = 0.01
        self.yaw = 0.01
        self.body_yaw = 0.0  # Body rotation (separate from head yaw)
        self._last_turn_around_time = 0.0  # Cooldown tracking for turn_around

        centered = None
        try:
            centered = getattr(self, "poses", {}).get("centered")
        except (AttributeError, TypeError) as e:
            log_swallowed_exception(e, operation="get_centered_pose")
            centered = None
        if isinstance(centered, (list, tuple)) and len(centered) >= 6:
            try:
                self.x = float(centered[0])
                self.y = float(centered[1])
                self.z = float(centered[2])
                self.roll = float(centered[3])
                self.pitch = float(centered[4])
                self.yaw = float(centered[5])
            except (TypeError, ValueError, IndexError) as e:
                log_swallowed_exception(e, operation="parse_centered_pose", context={"centered": centered})

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

        self.key_responses = load_key_responses(self.log)
        self.phrase_responses = load_phrase_responses(self.log)
        self._voice_agentic_enabled = False
        self._phrase_last_trigger_ts: dict[str, float] = {}
        self._outcome_code = 0
        self._last_action_event_id: str | None = None
        self._last_transcript_event: dict | None = None
        self.requested_mode: str | None = None
        # New architecture state tracking via StateManager
        self._state_manager = StateManager()
        self._agentic_stop_event: threading.Event | None = None
        self._agentic_thread: threading.Thread | None = None
        self._agentic_agent = None
        self._agentic_state = None
        self._cli_logger: CLIInputLogger | None = None
        self._vision_event_logger: VisionEventLogger | None = None
        self._vision_event_thread: threading.Thread | None = None
        self._vision_event_stop_event: threading.Event | None = None
        self._vision_event_last_frame_ts: float | None = None
        self.vision_events_path: str | None = None

        self.movement_model = None
        self.segmenter = None
        self._segmenter_model: str | None = None
        self.motor_history: list[dict] = []

        atexit.register(self.shutdown)

    # ─────────────────────────────────────────────────────────────────────────
    # Hardware Properties (backward compatibility + abstraction)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def mini(self) -> Any:
        """Get the underlying ReachyMini SDK instance.

        DEPRECATED: Use self._robot (RobotController) for new code.
        This property is provided for backward compatibility during migration.

        Returns:
            ReachyMini SDK instance if using real hardware, None otherwise.
        """
        if self._robot is None:
            return None
        # ReachyMiniController exposes the SDK instance via .mini property
        return getattr(self._robot, "mini", None)

    @property
    def robot(self) -> "ReachyMiniController | SimulatedController | None":
        """Get the robot controller (new abstraction layer).

        Use this for new code that should work with any robot type.
        """
        return self._robot

    def get_frame(self) -> np.ndarray | None:
        """Get the current video frame from the robot.

        Works with both real hardware and simulation.

        Returns:
            BGR image as numpy array, or None if not available.
        """
        if self._robot is None:
            return None

        video_stream = self._robot.get_video_stream()
        if video_stream is not None:
            return video_stream.get_frame_nonblocking()

        # Fallback to direct SDK access for backward compatibility
        mini = self.mini
        if mini is not None:
            try:
                return mini.media.get_frame()
            except Exception:
                return None

        return None

    def get_audio_sample(self) -> np.ndarray | None:
        """Get an audio sample from the robot.

        Works with both real hardware and simulation.

        Returns:
            Audio samples as numpy array, or None if not available.
        """
        if self._robot is None:
            return None

        audio_stream = self._robot.get_audio_stream()
        if audio_stream is not None:
            return audio_stream.get_sample(timeout=0.1)

        # Fallback to direct SDK access for backward compatibility
        mini = self.mini
        if mini is not None:
            try:
                return mini.media.get_audio_sample()
            except Exception:
                return None

        return None

    def push_audio_sample(self, samples: np.ndarray) -> bool:
        """Push audio samples to the robot speaker.

        Works with both real hardware and simulation.

        Args:
            samples: Audio samples as numpy array.

        Returns:
            True if samples were pushed successfully.
        """
        if self._robot is None:
            return False

        audio_stream = self._robot.get_audio_stream()
        if audio_stream is not None:
            return audio_stream.push_sample(samples)

        # Fallback to direct SDK access for backward compatibility
        mini = self.mini
        if mini is not None:
            try:
                mini.media.push_audio_sample(samples)
                return True
            except Exception:
                return False

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # State Properties (delegate to StateManager)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def operational_mode(self) -> str:
        """Current operational mode (passive/active/singularity)."""
        return self._state_manager.operational_mode

    @property
    def processing_state(self) -> str:
        """Current processing state (awake/sleep)."""
        return self._state_manager.processing_state

    @property
    def current_strategy(self) -> str:
        """Current strategy (observe/explore/research/assist/reflect/learn)."""
        return self._state_manager.strategy

    def _start_key_listener(self, stop_event: threading.Event) -> threading.Thread | None:
        if bool(getattr(self, "interactive", True)):
            return None
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

    def _start_cli_listener(self, stop_event: threading.Event) -> threading.Thread | None:
        if not bool(getattr(self, "interactive", True)):
            return None

        def _worker() -> None:
            try:
                import select
                import sys
            except Exception as e:
                warn("CLI listener unavailable: %s", e, logger=self.log)
                return

            stdin = sys.stdin
            stdout = sys.stdout
            if stdin is None or not hasattr(stdin, "isatty") or not stdin.isatty():
                return

            prompt = "maxim> "
            while not stop_event.is_set():
                try:
                    stdout.write(prompt)
                    stdout.flush()
                except Exception:
                    pass

                line = None
                while not stop_event.is_set():
                    try:
                        ready, _, _ = select.select([stdin], [], [], 0.1)
                    except Exception:
                        ready = []
                    if not ready:
                        continue
                    try:
                        line = stdin.readline()
                    except Exception:
                        line = None
                    break

                if stop_event.is_set():
                    break
                if not line:
                    time.sleep(0.05)
                    continue
                self._handle_cli_text(line)

        return threading.Thread(target=_worker, name="maxim.cli", daemon=True)

    def _start_transcript_listener(self, stop_event: threading.Event) -> threading.Thread | None:
        if not bool(getattr(self, "audio", False)):
            return None
        if not isinstance(getattr(self, "phrase_responses", None), dict) or not self.phrase_responses:
            return None

        transcript_path = getattr(self, "transcript_path", None)
        if not isinstance(transcript_path, str) or not transcript_path.strip():
            return None
        transcript_path = transcript_path.strip()

        def _worker() -> None:
            fp = None
            try:
                while not stop_event.is_set():
                    if fp is None:
                        try:
                            fp = open(transcript_path, "r", encoding="utf-8")
                        except FileNotFoundError:
                            time.sleep(0.25)
                            continue
                        except Exception as e:
                            warn("Transcript listener unavailable: %s", e, logger=self.log)
                            return

                    line = fp.readline()
                    if not line:
                        time.sleep(0.05)
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(record, dict):
                        self._handle_transcript_record(record)
            finally:
                if fp is not None:
                    try:
                        fp.close()
                    except Exception:
                        pass

        return threading.Thread(target=_worker, name="maxim.transcript", daemon=True)

    def _handle_phrase_text(
        self,
        text: str,
        *,
        source: str,
        transcript: dict | None = None,
    ) -> None:
        raw_text = str(text or "").strip()
        if not raw_text:
            return
        source_label = str(source or "").strip().lower()
        is_cli = source_label == "cli"
        normalized_text = normalize_transcript_text(raw_text)
        if not normalized_text:
            return

        def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
            if not needle:
                return True
            if not haystack:
                return False
            hi = 0
            for token in needle:
                while hi < len(haystack) and haystack[hi] != token:
                    hi += 1
                if hi >= len(haystack):
                    return False
                hi += 1
            return True

        transcript_tokens = normalized_text.split()
        has_maxim = "maxim" in transcript_tokens

        wake_tokens: set[str] = set()
        for _, spec in getattr(self, "phrase_responses", {}).items():
            if not isinstance(spec, dict) or not bool(spec.get("wake_word", False)):
                continue
            norm = spec.get("_normalized")
            if isinstance(norm, str) and norm:
                wake_tokens.update(norm.split())

        command_tokens = [t for t in transcript_tokens if t not in wake_tokens]
        command_token_set = set(command_tokens)

        now = time.time()
        matches: list[tuple[str, dict]] = []
        for phrase, spec in getattr(self, "phrase_responses", {}).items():
            if not isinstance(phrase, str) or not phrase:
                continue
            if not isinstance(spec, dict):
                continue

            pattern = spec.get("_pattern")
            matched = False
            try:
                if pattern is not None:
                    matched = bool(pattern.search(raw_text))
                else:
                    matched = phrase.lower() in raw_text.lower()
            except Exception:
                matched = False
            if not matched:
                normalized_phrase = spec.get("_normalized")
                if isinstance(normalized_phrase, str) and normalized_phrase:
                    haystack = f" {normalized_text} "
                    needle = f" {normalized_phrase} "
                    matched = needle in haystack
            if not matched:
                continue

            if not is_cli and bool(spec.get("requires_agentic", False)) and not bool(
                getattr(self, "_voice_agentic_enabled", False)
            ):
                continue

            if not is_cli:
                cooldown_s = float(spec.get("cooldown_s", 0.0) or 0.0)
                last_ts = float(getattr(self, "_phrase_last_trigger_ts", {}).get(phrase, 0.0) or 0.0)
                if cooldown_s > 0 and (now - last_ts) < cooldown_s:
                    continue

            matches.append((phrase, spec))

        if not matches:
            return

        command_matches = [(phrase, spec) for phrase, spec in matches if not bool(spec.get("wake_word", False))]
        wake_matches = [(phrase, spec) for phrase, spec in matches if bool(spec.get("wake_word", False))]

        def _pick_best(candidates: list[tuple[str, dict]]) -> tuple[str, dict] | None:
            best = None
            best_score: tuple[int, int] = (-1, -1)
            for phrase, spec in candidates:
                normalized_phrase = spec.get("_normalized")
                if not isinstance(normalized_phrase, str) or not normalized_phrase:
                    normalized_phrase = normalize_trigger_text(phrase)
                score = (len(normalized_phrase.split()), len(normalized_phrase))
                if score > best_score:
                    best = (phrase, spec)
                    best_score = score
            return best

        best = _pick_best(command_matches)
        if best is None and has_maxim:
            inferred: list[tuple[str, dict, tuple[int, int, int]]] = []
            for phrase, spec in getattr(self, "phrase_responses", {}).items():
                if not isinstance(phrase, str) or not phrase:
                    continue
                if not isinstance(spec, dict) or bool(spec.get("wake_word", False)):
                    continue
                if not is_cli and bool(spec.get("requires_agentic", False)) and not bool(
                    getattr(self, "_voice_agentic_enabled", False)
                ):
                    continue

                if not is_cli:
                    cooldown_s = float(spec.get("cooldown_s", 0.0) or 0.0)
                    last_ts = float(getattr(self, "_phrase_last_trigger_ts", {}).get(phrase, 0.0) or 0.0)
                    if cooldown_s > 0 and (now - last_ts) < cooldown_s:
                        continue

                normalized_phrase = spec.get("_normalized")
                if not isinstance(normalized_phrase, str) or not normalized_phrase:
                    normalized_phrase = normalize_trigger_text(phrase)
                phrase_tokens = normalized_phrase.split()
                required_tokens = [t for t in phrase_tokens if t not in wake_tokens]
                if not required_tokens:
                    continue
                if not set(required_tokens) <= command_token_set:
                    continue
                if not _is_subsequence(required_tokens, command_tokens):
                    continue
                full_subseq = int(_is_subsequence(phrase_tokens, transcript_tokens))
                score = (len(required_tokens), full_subseq, len(phrase_tokens))
                inferred.append((phrase, spec, score))

            if inferred:
                inferred.sort(key=lambda item: item[2], reverse=True)
                best = inferred[0][0], inferred[0][1]

        if best is None:
            best = _pick_best(wake_matches)
            if best is None:
                return
            if bool(getattr(self, "_voice_agentic_enabled", False)):
                return

        phrase, spec = best
        if not is_cli:
            try:
                self._phrase_last_trigger_ts[phrase] = now
            except Exception:
                pass
        self._run_action_spec(source=source, trigger=phrase, spec=spec, transcript=transcript)

    def _handle_transcript_record(self, record: dict) -> None:
        text = str(record.get("text", "") or "").strip()
        if not text:
            return
        try:
            self._last_transcript_event = record
        except Exception:
            pass

        # Forward voice transcript to agentic state so LLM worker can see it
        # Only forward if transcript explicitly contains wake word (maxim, maximum, or reachy)
        text_lower = text.lower()
        has_wake_word = "maxim" in text_lower or "reachy" in text_lower  # "maxim" matches "maximum" too
        if has_wake_word:
            agentic_state = getattr(self, "_agentic_state", None)
            if agentic_state is not None and hasattr(agentic_state, "data"):
                agentic_state.data["pending_voice_input"] = text
                agentic_state.data["pending_voice_transcript"] = record
                self.log.info("Voice transcript forwarded to agentic state: %s", text[:50])
            else:
                self.log.warning("Agentic state not available for voice forwarding (state=%s)", agentic_state)

        self._handle_phrase_text(text, source="voice", transcript=record)

    def _handle_cli_text(self, text: str) -> None:
        raw = str(text or "").strip()
        if not raw:
            return
        self._log_cli_input(raw)

        # Forward CLI input to agentic state so LLM worker can see it
        agentic_state = getattr(self, "_agentic_state", None)
        if agentic_state is not None and hasattr(agentic_state, "data"):
            agentic_state.data["pending_cli_input"] = raw
            self.log.warning("CLI input forwarded to agentic state: %s", raw[:50])
        else:
            self.log.warning("Agentic state not available for CLI forwarding (state=%s)", agentic_state)

        if len(raw) == 1:
            spec = getattr(self, "key_responses", {}).get(raw)
            if isinstance(spec, dict):
                self._run_action_spec(source="cli", trigger=raw, spec=spec)
                return
        self._handle_phrase_text(raw, source="cli", transcript={"text": raw})

    def _log_cli_input(self, text: str) -> None:
        logger = getattr(self, "_cli_logger", None)
        if logger is None:
            return
        record = {
            "kind": "cli_input",
            "time": float(time.time()),
            "run_id": getattr(self, "run_id", None),
            "mode": getattr(self, "mode", None),
            "text": str(text),
        }
        try:
            logger.log_input(record)
        except Exception:
            return

    def _log_event(self, record: dict, *, flush: bool = False) -> None:
        training_logger = getattr(self, "_training_logger", None)
        if training_logger is None:
            return
        try:
            training_logger.log_event(record, flush=flush)
        except Exception:
            return

    def _run_action_spec(
        self,
        *,
        source: str,
        trigger: str,
        spec: dict,
        transcript: dict | None = None,
    ) -> None:
        call = spec.get("call")
        if not isinstance(call, str) or not call:
            return

        args = spec.get("args") if isinstance(spec.get("args"), list) else []
        kwargs = spec.get("kwargs") if isinstance(spec.get("kwargs"), dict) else {}

        if call == "label_outcome":
            fn = getattr(self, call, None)
            if callable(fn):
                try:
                    kw = dict(kwargs)
                    kw.setdefault("source", source)
                    kw.setdefault("trigger", trigger)
                    fn(*args, **kw)
                except Exception as e:
                    warn("Outcome label failed: %s", e, logger=self.log)
            return

        pause_training = bool(spec.get("pause_training", False)) and bool(getattr(self, "train", False))
        if pause_training:
            self._training_paused.set()

        event_id = uuid.uuid4().hex
        now = time.time()

        last_motor_sample_id = None
        sample = getattr(self, "_last_motor_sample", None)
        if isinstance(sample, dict):
            last_motor_sample_id = sample.get("sample_id")

        event: dict = {
            "kind": "action_event",
            "event_id": event_id,
            "time": float(now),
            "source": str(source),
            "trigger": str(trigger),
            "call": str(call),
            "args": list(args),
            "kwargs": dict(kwargs),
            "pause_training": bool(pause_training),
            "voice_agentic_enabled": bool(getattr(self, "_voice_agentic_enabled", False)),
            "outcome_code": int(getattr(self, "_outcome_code", 0) or 0),
            "run_id": getattr(self, "run_id", None),
            "mode": getattr(self, "mode", None),
            "epoch": int(getattr(self, "current_epoch", 0) or 0),
            "video_path": getattr(self, "video_path", None),
            "audio_path": getattr(self, "audio_path", None),
            "transcript_path": getattr(self, "transcript_path", None),
            "last_motor_sample_id": last_motor_sample_id,
            "parent_event_id": getattr(self, "_last_action_event_id", None),
        }
        if isinstance(transcript, dict):
            event["transcript"] = {
                "chunk_index": transcript.get("chunk_index"),
                "start_s": transcript.get("start_s"),
                "end_s": transcript.get("end_s"),
                "text": str(transcript.get("text", "") or "")[:280],
            }

        try:
            with self._observation_lock:
                fn = getattr(self, call, None)
                if not callable(fn):
                    warn("Unknown %s action for '%s': %s", source, trigger, call, logger=self.log)
                    event["success"] = False
                    event["error"] = f"Unknown action: {call}"
                else:
                    fn(*args, **kwargs)
                    event["success"] = True
        except Exception as e:
            warn("%s '%s' action failed: %s", source, trigger, e, logger=self.log)
            event["success"] = False
            event["error"] = str(e)
        finally:
            self._log_event(event)
            try:
                self._last_action_event_id = str(event_id)
            except Exception:
                pass
            if pause_training:
                self._training_paused.clear()

    def _handle_keypress(self, key: str) -> None:
        try:
            spec = getattr(self, "key_responses", {}).get(key)
        except Exception:
            spec = None
        if not isinstance(spec, dict):
            return
        self._run_action_spec(source="keyboard", trigger=key, spec=spec)

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

    def _reset_connection_failures(self) -> None:
        for state in self._connection_failures.values():
            try:
                state["count"] = 0
                state["last_ts"] = 0.0
            except Exception:
                continue

    def _note_connection_failure(self, kind: str, error: object) -> None:
        if not is_connection_error(error):
            if int(getattr(self, "verbosity", 0) or 0) >= 2:
                try:
                    self.log.debug("Ignoring non-connection error (%s): %s", kind, error)
                except Exception:
                    pass
            return
        stop_event = getattr(self, "_live_stop_event", None)
        if stop_event is not None and stop_event.is_set():
            if int(getattr(self, "verbosity", 0) or 0) >= 2:
                try:
                    self.log.debug("Ignoring connection failure during shutdown (%s): %s", kind, error)
                except Exception:
                    pass
            return
        requested_mode = str(getattr(self, "requested_mode", "") or "").strip().lower()
        if requested_mode:
            if int(getattr(self, "verbosity", 0) or 0) >= 2:
                try:
                    self.log.debug(
                        "Ignoring connection failure during mode switch (%s -> %s): %s",
                        str(getattr(self, "mode", "") or "").strip().lower(),
                        requested_mode,
                        error,
                    )
                except Exception:
                    pass
            return

        state = self._connection_failures.get(kind)
        if not isinstance(state, dict):
            state = {"count": 0, "last_ts": 0.0}
            self._connection_failures[kind] = state

        now = time.time()
        last_ts = float(state.get("last_ts") or 0.0)
        if now - last_ts > float(getattr(self, "_reconnect_window_s", 5.0) or 5.0):
            state["count"] = 0

        state["count"] = int(state.get("count", 0) or 0) + 1
        state["last_ts"] = now

        threshold = int(self._reconnect_thresholds.get(kind, 3) or 3)
        if int(getattr(self, "verbosity", 0) or 0) >= 2:
            try:
                self.log.debug(
                    "Connection failure (%s): count=%d threshold=%d window_s=%.1f error=%s",
                    kind,
                    int(state["count"]),
                    threshold,
                    float(getattr(self, "_reconnect_window_s", 5.0) or 5.0),
                    error,
                )
            except Exception:
                pass
        if state["count"] >= threshold:
            self._soft_reconnect(reason=f"{kind}_connection_failed", error=error)

    def _soft_reconnect(self, *, reason: str, error: object | None = None) -> bool:
        if getattr(self, "_closed", False):
            return False
        stop_event = getattr(self, "_live_stop_event", None)
        if stop_event is not None and stop_event.is_set():
            return False

        now = time.time()
        last_ts = float(getattr(self, "_last_reconnect_ts", 0.0) or 0.0)
        cooldown_s = float(getattr(self, "_reconnect_cooldown_s", 20.0) or 20.0)
        if now - last_ts < cooldown_s:
            if int(getattr(self, "verbosity", 0) or 0) >= 2:
                try:
                    remaining = max(0.0, cooldown_s - (now - last_ts))
                    self.log.debug(
                        "Soft reconnect suppressed (cooldown %.1fs remaining): %s",
                        remaining,
                        reason,
                    )
                except Exception:
                    pass
            return False
        if not self._reconnect_lock.acquire(blocking=False):
            if int(getattr(self, "verbosity", 0) or 0) >= 2:
                try:
                    self.log.debug("Soft reconnect suppressed (lock busy): %s", reason)
                except Exception:
                    pass
            return False

        self._last_reconnect_ts = now
        try:
            if int(getattr(self, "verbosity", 0) or 0) >= 2:
                try:
                    self.log.debug("Soft reconnect begin: %s", reason)
                except Exception:
                    pass
            warn(
                "Soft reconnect requested (%s): %s",
                reason,
                error if error is not None else "unknown error",
                logger=self.log,
            )

            old_mini = getattr(self, "mini", None)
            if old_mini is not None:
                try:
                    old_mini.stop_recording()
                except Exception:
                    pass

                try:
                    media = getattr(old_mini, "media", None)
                    if media is not None:
                        media_lock = getattr(self, "_media_lock", None)
                        if media_lock is not None:
                            with media_lock:
                                media.close()
                        else:
                            media.close()
                except Exception as e:
                    warn("Failed to close media during reconnect: %s", e, logger=self.log)

                for attr in ("disconnect", "close", "shutdown"):
                    fn = getattr(old_mini, attr, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass

            # Use the RobotController's reconnect method
            if self._robot is None:
                warn("Soft reconnect failed (no robot controller)", logger=self.log)
                return False

            # For simulation mode, just reset state
            if self._simulation:
                self.log.info("Simulation mode - skipping reconnect")
                return True

            # Attempt reconnection via the controller
            if not self._robot.reconnect(timeout=30.0, max_attempts=1):
                warn("Soft reconnect failed (controller reconnect failed)", logger=self.log)
                return False

            try:
                self._robot.start_recording()
            except Exception as e:
                warn("Failed to start recording after reconnect: %s", e, logger=self.log)

            if getattr(self, "_woke_up", False):
                mode = str(getattr(self, "mode", "") or "").strip().lower()
                if mode != "sleep":
                    try:
                        self._robot.wake_up()
                    except Exception as e:
                        warn("Failed to wake Reachy after reconnect: %s", e, logger=self.log)

            motor_queue = getattr(self, "_motor_queue", None)
            if motor_queue is not None:
                try:
                    while True:
                        motor_queue.get_nowait()
                except queue.Empty:
                    pass

            self._reset_connection_failures()
            self.log.info("Soft reconnect complete.")
            return True
        finally:
            self._reconnect_lock.release()

    def _repo_root(self) -> str:
        try:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        except Exception:
            return os.getcwd()

    def _ensure_vision_logger(self) -> VisionEventLogger | None:
        if self._vision_event_logger is not None:
            return self._vision_event_logger

        run_id = getattr(self, "run_id", None) or time.strftime("%Y-%m-%d_%H%M%S")
        if not self.vision_events_path:
            self.vision_events_path = os.path.join(
                str(getattr(self, "home_dir", "data") or "data"),
                "vision",
                f"vision_events_{run_id}.jsonl",
            )

        try:
            logger = VisionEventLogger(self.vision_events_path)
            logger.start()
            self._vision_event_logger = logger
            return logger
        except Exception as e:
            warn("Failed to start vision event logger: %s", e, logger=self.log)
            self._vision_event_logger = None
            return None

    def _start_vision_event_stream(self) -> None:
        existing = getattr(self, "_vision_event_thread", None)
        if existing is not None and getattr(existing, "is_alive", lambda: False)():
            return

        logger = self._ensure_vision_logger()
        if logger is None:
            return

        stop_event = threading.Event()
        self._vision_event_stop_event = stop_event
        self._vision_event_last_frame_ts = None

        def _worker() -> None:
            try:
                hz = float(os.getenv("MAXIM_VISION_EVENT_HZ", "2.0") or 2.0)
            except Exception:
                hz = 2.0
            if hz <= 0:
                hz = 2.0
            sleep_s = 1.0 / hz

            while not stop_event.is_set():
                frame = getattr(self, "_last_frame", None)
                if frame is None or not isinstance(frame, np.ndarray):
                    time.sleep(sleep_s)
                    continue

                frame_ts = getattr(self, "_last_frame_ts", None)
                ts_val = None
                try:
                    ts_val = float(frame_ts) if frame_ts is not None else None
                except Exception:
                    ts_val = None

                if ts_val is not None and self._vision_event_last_frame_ts is not None:
                    if ts_val <= float(self._vision_event_last_frame_ts):
                        time.sleep(sleep_s)
                        continue

                segmenter = getattr(self, "segmenter", None)
                if segmenter is None:
                    try:
                        self._ensure_segmenter()
                    except Exception:
                        segmenter = None
                    segmenter = getattr(self, "segmenter", None)
                if segmenter is None:
                    time.sleep(sleep_s)
                    continue

                lock = getattr(self, "_observation_lock", None)
                acquired = False
                if lock is not None:
                    try:
                        acquired = lock.acquire(blocking=False)
                    except Exception:
                        acquired = False
                if lock is not None and not acquired:
                    time.sleep(sleep_s)
                    continue

                try:
                    observations = segmenter.segment_photos(frame)
                except Exception as e:
                    warn("Vision event segmentation failed: %s", e, logger=self.log)
                    observations = []
                finally:
                    if lock is not None and acquired:
                        try:
                            lock.release()
                        except Exception:
                            pass

                dets: list[dict[str, Any]] = []
                names = getattr(getattr(segmenter, "model", None), "names", None)
                for obs in observations or []:
                    if not isinstance(obs, (list, tuple)) or len(obs) < 8:
                        continue
                    try:
                        track_id = int(obs[0]) if obs[0] is not None else None
                    except Exception:
                        track_id = None
                    try:
                        cls_id = int(obs[7]) if obs[7] is not None else None
                    except Exception:
                        cls_id = None
                    try:
                        conf = float(obs[6]) if obs[6] is not None else 0.0
                    except Exception:
                        conf = 0.0

                    label = None
                    try:
                        if isinstance(names, dict) and cls_id in names:
                            label = names.get(cls_id)
                        elif isinstance(names, (list, tuple)) and cls_id is not None:
                            if 0 <= cls_id < len(names):
                                label = names[cls_id]
                    except Exception:
                        label = None

                    dets.append(
                        {
                            "track_id": track_id,
                            "class_id": cls_id,
                            "label": label,
                            "conf": conf,
                            "bbox_xyxy": [float(obs[2]), float(obs[3]), float(obs[4]), float(obs[5])],
                        }
                    )

                record = {
                    "kind": "vision_event",
                    "time": float(time.time()),
                    "run_id": getattr(self, "run_id", None),
                    "frame_ts": ts_val,
                    "model": getattr(self, "_segmenter_model", None),
                    "interests": list(getattr(self, "interests", []) or []),
                    "detections": dets,
                }
                try:
                    shape = getattr(frame, "shape", None)
                    if isinstance(shape, tuple) and len(shape) >= 2:
                        record["frame_shape"] = [int(shape[0]), int(shape[1])]
                except Exception:
                    pass

                logger.log_event(record)
                if ts_val is not None:
                    self._vision_event_last_frame_ts = float(ts_val)

                time.sleep(sleep_s)

        t = threading.Thread(target=_worker, name="maxim.vision.events", daemon=True)
        self._vision_event_thread = t
        self.register_thread("maxim.vision.events", t)
        t.start()

    def _stop_vision_event_stream(self, *, timeout: float = 2.0) -> None:
        try:
            ev = getattr(self, "_vision_event_stop_event", None)
            if ev is not None:
                ev.set()
        except Exception:
            pass
        t = getattr(self, "_vision_event_thread", None)
        if t is not None:
            try:
                t.join(timeout=float(timeout))
            except Exception:
                pass
        self._vision_event_thread = None
        self._vision_event_stop_event = None

        logger = getattr(self, "_vision_event_logger", None)
        if logger is not None:
            try:
                logger.stop(timeout=float(timeout))
            except Exception:
                pass
        self._vision_event_logger = None

    def _start_agentic_runtime(self, *, use_capture_manager: bool = True) -> None:
        """Start the agentic runtime.

        Args:
            use_capture_manager: If True, use CaptureManager for direct frame access (Phase 3).
                                If False, fall back to JSONL-based vision event stream.
        """
        existing = getattr(self, "_agentic_thread", None)
        if existing is not None and getattr(existing, "is_alive", lambda: False)():
            return

        # Allow CPU-only operation (e.g., when CUDA is hidden for Blackwell GPUs)
        # Note: llama_cpp backend has native Metal support on macOS, so skip this fallback
        # when using llama_cpp - it will use Metal GPU acceleration automatically
        if not is_gpu_available():
            # Check if we're using llama_cpp backend (has native Metal support)
            llm_config_path = os.path.join(str(getattr(self, "home_dir", "data") or "data"), "util", "llm.json")
            using_llama_cpp = False
            try:
                if os.path.exists(llm_config_path):
                    with open(llm_config_path) as f:
                        llm_cfg = json.load(f)
                    profile_name = llm_cfg.get("profile", "")
                    profiles = llm_cfg.get("profiles", {})
                    if profile_name in profiles:
                        using_llama_cpp = profiles[profile_name].get("backend") == "llama_cpp"
            except Exception:
                pass

            if using_llama_cpp:
                # llama.cpp has native Metal support on macOS - no fallback needed
                self.log.info("Using llama.cpp backend with native Metal GPU support")
            else:
                cuda_hidden = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
                if cuda_hidden:
                    self.log.info("GPU hidden for compatibility - agentic runtime will use CPU (slower)")
                else:
                    self.log.warning("No GPU available - agentic runtime will use CPU (slower)")

                # Configure CPU-friendly model defaults (only for non-llama_cpp backends)
                os.environ.setdefault("MAXIM_LLM_PROFILE", "smollm-1.7b-instruct")
                os.environ.setdefault("MAXIM_LLM_N_GPU_LAYERS", "0")
                self.log.info("Using CPU-friendly LLM: smollm-1.7b-instruct")

        try:
            from maxim.agents import MaximAgent
            from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SupervisionPolicy
            from maxim.agents.llm_worker import LLMWorker
            from maxim.environment import ReachyEnv
            from maxim.runtime import (
                CaptureManager,
                build_decision_engine,
                build_evaluators,
                build_executor,
                build_memory,
                build_state,
                build_tool_registry,
            )
            from maxim.runtime.bootstrap import build_default_network
            from maxim.runtime.agent_loop import run_agentic_loop
        except Exception as e:
            warn("Agentic runtime unavailable: %s", e, logger=self.log)
            return

        stop_event = threading.Event()
        self._agentic_stop_event = stop_event

        # Phase 3: Initialize CaptureManager for direct frame access
        capture_manager = None
        if use_capture_manager:
            try:
                capture_manager = CaptureManager(
                    maxim=self,
                    target_fps=float(getattr(self, "video_fps", 10.0) or 10.0),
                    enable_segmentation=True,
                )
                self._capture_manager = capture_manager
            except Exception as e:
                warn("Failed to create CaptureManager: %s (falling back to JSONL)", e, logger=self.log)
                capture_manager = None

        agent = MaximAgent(
            interests=list(getattr(self, "interests", []) or []),
            data_folder=str(getattr(self, "home_dir", "data") or "data"),
            capture_manager=capture_manager,
        )
        env = ReachyEnv(repo_root=os.getcwd(), data_dir=str(getattr(self, "home_dir", "data") or "data"))
        state = build_state(max_steps=1_000_000)
        try:
            state.data["maxim_runtime"] = {
                "mode": getattr(self, "mode", None),
                "interests": list(getattr(self, "interests", []) or []),
            }
        except Exception:
            pass
        self._agentic_agent = agent
        self._agentic_state = state
        self._state_manager.set_agent(agent)

        # Propagate exploration mode context if set by CLI
        if bool(getattr(self, "_exploration_mode", False)):
            state.data["exploration_mode"] = True
            state.data["exploration_focus"] = str(getattr(self, "_exploration_focus", "") or "")
            state.data["exploration_session_id"] = str(getattr(self, "_exploration_session_id", "") or "")
            state.data["exploration_policy"] = getattr(self, "_exploration_policy", {}) or {}
            state.data["mode"] = "exploration"  # Override mode for MemoryAgent
            # Also update maxim_runtime for consistency
            if isinstance(state.data.get("maxim_runtime"), dict):
                state.data["maxim_runtime"]["mode"] = "exploration"

        memory = build_memory()
        decision_engine = build_decision_engine()

        # Set up ResponseOutput for LLM responses
        from pathlib import Path
        from maxim.utils.response_output import ResponseOutput

        sandbox_path = Path(self.home_dir) / "sandbox"
        tts_engine = None
        speaker_fn = None

        # Check if TTS is enabled (set via environment or config)
        # Environment variables:
        #   MAXIM_TTS_ENABLED=1     - Enable TTS
        #   MAXIM_TTS_MODEL=name    - TTS voice model (default: en_US-lessac-medium)
        #   MAXIM_TTS_LOCAL=1       - Force local audio (skip Reachy speaker)
        if os.environ.get("MAXIM_TTS_ENABLED", "").lower() in ("1", "true", "yes"):
            try:
                from maxim.models.audio.tts import TTSEngine
                from maxim.utils.audio import _check_local_audio

                tts_model = os.environ.get("MAXIM_TTS_MODEL", "en_US-lessac-medium")
                tts_engine = TTSEngine(model_name=tts_model)
                if tts_engine.is_available:
                    speaker_fn = self.speak  # Uses Reachy speaker with local fallback
                    self.log.info("TTS enabled with model: %s", tts_model)

                    # Check local audio availability for fallback
                    prefer_local = os.environ.get("MAXIM_TTS_LOCAL", "").lower() in ("1", "true", "yes")
                    if prefer_local:
                        self.log.info("TTS using local audio (MAXIM_TTS_LOCAL=1)")
                    elif _check_local_audio():
                        self.log.info("Local audio available as fallback for remote connections")
                else:
                    self.log.warning("TTS model not found, TTS disabled")
                    tts_engine = None
            except Exception as e:
                self.log.warning("Failed to initialize TTS: %s", e)
                tts_engine = None

        response_output = ResponseOutput(
            sandbox_path=sandbox_path,
            logger=self.log,
            tts_engine=tts_engine,
            speaker_fn=speaker_fn,
        )

        # Check if internet access is allowed (from exploration policy or default to True)
        exploration_policy_dict = getattr(self, "_exploration_policy", {}) or {}
        allow_internet = exploration_policy_dict.get("allow_internet", True)

        # Create internet policy getter for tool registry
        def get_internet_policy():
            from maxim.utils.internet_access import InternetAccessPolicy
            return InternetAccessPolicy(enabled=allow_internet)

        # Only pass policy getter if internet is allowed
        internet_policy_getter = get_internet_policy if allow_internet else None

        registry = build_tool_registry(
            maxim=self,
            response_output=response_output,
            internet_policy_getter=internet_policy_getter,
        )
        executor = build_executor(registry)
        evaluators = build_evaluators()

        run_id = getattr(self, "run_id", None) or time.strftime("%Y-%m-%d_%H%M%S")

        # Build allowed tools set based on policy (uses allow_internet from above)
        allowed_tools = {
            "read_file",
            "focus_interests",
            "track_target",
            "maxim_command",
            "mode_switch",
            "speak",
            "respond",
            "list_directory",  # Allow directory listing
        }

        # Add internet tools if allowed
        if allow_internet:
            allowed_tools.add("internet_search")
            allowed_tools.add("http_fetch")

        # Set up autonomy controller with sensible defaults for live mode
        supervision_policy = SupervisionPolicy(
            allowed_tools=allowed_tools,
            forbidden_tools={"execute_file", "delete_file"},
            min_confidence_autonomous=0.7,
            requires_confirmation={"write_file"},
        )
        autonomy_controller = AutonomyController(
            initial_level=AutonomyLevel.SUPERVISED,
            supervision_policy=supervision_policy,
        )

        # Create LLM worker for handling user questions
        llm_worker = None
        try:
            from maxim.models.language.router import LLMRouter, load_llm_config

            llm_config = load_llm_config()
            if llm_config.enabled:
                llm_router = LLMRouter(llm_config)
                # Start warming up the LLM in background (reduces first-request latency)
                llm_router.warmup()
                llm_worker = LLMWorker(
                    llm=llm_router,
                    stale_threshold_s=5.0,
                    n_ctx=llm_router.n_ctx,
                    token_counter=llm_router.get_token_counter(),
                )
                llm_worker.start()
                self.log.info("LLM worker started for user responses")
            else:
                self.log.debug("LLM disabled in config, responses will use fallback")
        except Exception as e:
            warn("Failed to create LLM worker: %s", e, logger=self.log)
            llm_worker = None

        self._llm_worker = llm_worker

        # Extract AgentBus from MaximAgent for Default Network
        agent_bus = None
        try:
            agent_bus = getattr(agent, "_bus", None)
            if agent_bus is not None:
                self.log.debug("Extracted AgentBus from MaximAgent for DN")
        except Exception:
            pass

        # Create FearAgent for safety gating
        fear_agent = None
        try:
            from maxim.agents.fear_agent import FearAgent
            fear_agent = FearAgent(llm=None)  # No LLM needed for basic safety checks
            self._fear_agent = fear_agent
            self.log.debug("FearAgent created for DN safety gating")
        except Exception as e:
            warn("Failed to create FearAgent: %s", e, logger=self.log)

        # Create NAc for causal learning (used by pain detection and decision making)
        nac = None
        try:
            from maxim.decisions.nac import NAc
            nac = NAc()
            self._nac = nac
            self.log.debug("NAc created for causal learning")
        except Exception as e:
            warn("Failed to create NAc: %s", e, logger=self.log)

        # Build Default Network for reactive behaviors
        default_network = None
        try:
            default_network = build_default_network(
                maxim=self,
                bus=agent_bus,
                fear_agent=fear_agent,
                nac=nac,
                frame_size=(640, 480),
            )
            if default_network is not None:
                self._default_network = default_network
                self.log.info(
                    "DefaultNetwork built (bus=%s, fear_agent=%s)",
                    "connected" if agent_bus else "none",
                    "enabled" if fear_agent else "none",
                )
        except Exception as e:
            warn("Failed to build DefaultNetwork: %s", e, logger=self.log)
            default_network = None

        # Start capture manager or fall back to vision event stream
        if capture_manager is not None:
            try:
                capture_manager.start()
                self.log.info("CaptureManager started for direct frame access")
            except Exception as e:
                warn("Failed to start CaptureManager: %s", e, logger=self.log)
                capture_manager = None

        # Fall back to JSONL-based stream if no capture manager
        if capture_manager is None:
            self._start_vision_event_stream()

        def _on_step(ctx: dict) -> None:
            tool_result = ctx.get("tool_result")
            action = ctx.get("action") if isinstance(ctx.get("action"), dict) else None
            goal = ctx.get("goal")
            decision = ctx.get("decision") if isinstance(ctx.get("decision"), dict) else None

            output_preview = None
            output_size = None
            try:
                if tool_result is not None:
                    out = getattr(tool_result, "output", None)
                    if isinstance(out, str):
                        output_size = len(out)
                        output_preview = out[:160]
                    elif isinstance(out, dict):
                        output_preview = {k: out[k] for k in list(out)[:6]}
            except Exception:
                output_preview = None

            record = {
                "kind": "agentic_action",
                "event_id": uuid.uuid4().hex,
                "time": float(time.time()),
                "run_id": run_id,
                "agent_name": getattr(agent, "agent_name", getattr(agent, "name", None)),
                "goal": goal,
                "action": action,
                "score": decision.get("score") if isinstance(decision, dict) else None,
                "success": getattr(tool_result, "success", None) if tool_result is not None else None,
                "error": getattr(tool_result, "error", None) if tool_result is not None else None,
                "output_size": output_size,
                "output_preview": output_preview,
                "outcome_code": int(getattr(self, "_outcome_code", 0) or 0),
                "voice_agentic_enabled": bool(getattr(self, "_voice_agentic_enabled", False)),
            }
            self._log_event(record)

        def _worker() -> None:
            try:
                run_agentic_loop(
                    agent,
                    env,
                    state,
                    memory,
                    decision_engine,
                    executor,
                    autonomy_controller=autonomy_controller,
                    llm_worker=llm_worker,
                    default_network=default_network,
                    evaluators=evaluators,
                    max_steps=0,  # Unlimited
                    run_id=run_id,
                    stop_event=stop_event,
                    on_step=_on_step,
                    idle_sleep_s=0.1,
                    target_hz=10.0,  # 10 Hz for responsive CLI handling
                )
            except Exception as e:
                warn("Agentic runtime loop failed: %s", e, logger=self.log)
            finally:
                # Clean up LLM worker
                if llm_worker is not None:
                    try:
                        llm_worker.stop()
                    except Exception:
                        pass

        t = threading.Thread(target=_worker, name="maxim.agentic", daemon=True)
        self._agentic_thread = t
        t.start()

    def _stop_agentic_runtime(self, *, timeout: float = 10.0) -> None:
        """Stop the agentic runtime with graceful shutdown and force-kill fallback.

        Shutdown sequence:
        1. Signal all threads to stop via events
        2. Wait for graceful shutdown with timeout
        3. Force-terminate any threads that didn't respond

        Args:
            timeout: Total timeout for the entire shutdown sequence (default 10s).
                     Individual components get proportional timeouts.
        """
        import ctypes

        # Track threads that need to be stopped
        threads_to_stop: list[tuple[str, threading.Thread | None]] = []

        # Phase 1: Signal all stop events first (non-blocking)
        try:
            ev = getattr(self, "_agentic_stop_event", None)
            if ev is not None:
                ev.set()
        except Exception:
            pass

        capture_manager = getattr(self, "_capture_manager", None)
        if capture_manager is not None:
            try:
                # Just set the stop event, don't wait yet
                stop_ev = getattr(capture_manager, "_stop_event", None)
                if stop_ev is not None:
                    stop_ev.set()
            except Exception:
                pass

        default_network = getattr(self, "_default_network", None)
        if default_network is not None:
            try:
                # Signal DN to stop
                default_network._running = False
                bg_tasks = getattr(default_network, "_background_tasks", None)
                if bg_tasks is not None:
                    bg_tasks._running = False
                    stop_ev = getattr(bg_tasks, "_stop_event", None)
                    if stop_ev is not None:
                        stop_ev.set()
            except Exception:
                pass

        # Collect all threads
        t = getattr(self, "_agentic_thread", None)
        if t is not None:
            threads_to_stop.append(("agentic", t))

        if capture_manager is not None:
            for attr, name in [("_frame_thread", "capture.frame"),
                               ("_segmentation_thread", "capture.segmentation"),
                               ("_audio_thread", "capture.audio")]:
                thread = getattr(capture_manager, attr, None)
                if thread is not None:
                    threads_to_stop.append((name, thread))
            # Send poison pills to unblock queues
            try:
                seg_queue = getattr(capture_manager, "_segmentation_queue", None)
                if seg_queue is not None:
                    seg_queue.put_nowait(None)
            except Exception:
                pass

        if default_network is not None:
            dn_thread = getattr(default_network, "_thread", None)
            if dn_thread is not None:
                threads_to_stop.append(("default_network", dn_thread))
            bg_tasks = getattr(default_network, "_background_tasks", None)
            if bg_tasks is not None:
                bg_thread = getattr(bg_tasks, "_thread", None)
                if bg_thread is not None:
                    threads_to_stop.append(("background_tasks", bg_thread))

        # Phase 2: Wait for threads to stop gracefully (with per-thread timeout)
        per_thread_timeout = timeout / max(len(threads_to_stop), 1)
        still_alive: list[tuple[str, threading.Thread]] = []

        for name, thread in threads_to_stop:
            if thread is not None and thread.is_alive():
                try:
                    thread.join(timeout=per_thread_timeout)
                    if thread.is_alive():
                        still_alive.append((name, thread))
                        self.log.warning("Thread '%s' did not stop gracefully", name)
                except Exception:
                    still_alive.append((name, thread))

        # Phase 3: Force-terminate threads that didn't respond
        # This uses ctypes to raise SystemExit in the thread
        for name, thread in still_alive:
            try:
                if thread.is_alive():
                    thread_id = thread.ident
                    if thread_id is not None:
                        # Raise SystemExit in the thread
                        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_ulong(thread_id),
                            ctypes.py_object(SystemExit)
                        )
                        if res == 0:
                            self.log.warning("Invalid thread id for '%s'", name)
                        elif res > 1:
                            # Reset if more than one thread was affected
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                ctypes.c_ulong(thread_id), None
                            )
                        else:
                            self.log.info("Force-terminated thread '%s'", name)
                        # Give it a moment to terminate
                        thread.join(timeout=0.5)
            except Exception as e:
                self.log.warning("Failed to force-terminate thread '%s': %s", name, e)

        # Phase 4: Cleanup references
        self._agentic_thread = None
        self._agentic_stop_event = None
        self._agentic_agent = None
        self._agentic_state = None
        self._state_manager.set_agent(None)

        # Stop capture manager (will be fast since threads already stopped/terminated)
        if capture_manager is not None:
            try:
                capture_manager.stop(timeout=1.0)
            except Exception:
                pass

        # Stop Default Network (will be fast since threads already stopped/terminated)
        if default_network is not None:
            try:
                default_network.stop()
            except Exception:
                pass
        self._default_network = None
        self._capture_manager = None

        # Clear FearAgent reference
        self._fear_agent = None

        self._stop_vision_event_stream(timeout=2.0)

    def _set_epochs(self, epochs: int | None) -> None:
        try:
            value = int(epochs) if epochs is not None else 0
        except Exception:
            value = 0
        self.epochs = value if value > 0 else None
    
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
        if self.audio:
            try:
                audio_input_rate = int(self.mini.media.get_input_audio_samplerate())
                audio_output_rate = int(self.mini.media.get_output_audio_samplerate())
            except Exception as e:
                warn("Failed to read audio sample rates: %s", e, logger=self.log)

        transcribe_process = None
        transcribe_shutdown_file = None
        if self.audio and parallel:
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

        # IK failure detection handler for motor commands
        class _IKWarningHandler(logging.Handler):
            """Temporary handler to capture IK warnings during motor commands."""
            def __init__(self):
                super().__init__(logging.WARNING)
                self.ik_failure_detected = False
                self.failure_message = None

            def emit(self, record):
                msg = record.getMessage().lower()
                if "ik error" in msg or "pose not achievable" in msg or "collision detected" in msg:
                    self.ik_failure_detected = True
                    self.failure_message = record.getMessage()

        # Track consecutive IK failures to trigger recovery
        ik_failure_count = [0]  # Use list to allow modification in nested function
        IK_FAILURE_THRESHOLD = 3  # Reset to center after this many consecutive failures
        last_ik_reset_time = [0.0]

        def _motor_worker() -> None:
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
                ik_handler = _IKWarningHandler()
                for rl in reachy_loggers:
                    rl.addHandler(ik_handler)

                position_info = kwargs.pop("_position_info", None)  # Extract position if passed
                commanded_6d = kwargs.pop("_commanded_6d", None)  # Extract 6D pose for bounds learning
                movement_duration = kwargs.get("duration", 0.5)  # Get duration for sync timing

                try:
                    fn(*args, **kwargs)

                    # Check if IK failure was detected via warnings
                    if ik_handler.ik_failure_detected:
                        self.log.warning("IK failure detected: %s", ik_handler.failure_message)
                        ik_failure_count[0] += 1

                        # Record failure in AttentionNetwork for dynamic bounds learning
                        if position_info is not None:
                            default_network = getattr(self, "_default_network", None)
                            if default_network is not None:
                                attention = getattr(default_network, "_attention_network", None)
                                if attention is not None:
                                    try:
                                        attention.record_gaze(position_info, success=False)
                                        self.log.debug("Recorded IK failure at position %s", position_info)
                                    except Exception as rec_err:
                                        self.log.debug("Failed to record IK failure: %s", rec_err)

                        # Record IK failure for workspace bounds learning
                        if commanded_6d is not None:
                            default_network = getattr(self, "_default_network", None)
                            if default_network is not None:
                                bounds_learner = getattr(default_network, "_bounds_learner", None)
                                if bounds_learner is not None:
                                    try:
                                        self.sync_head_position()
                                        actual_6d = {
                                            "yaw": float(getattr(self, "yaw", 0.0) or 0.0),
                                            "pitch": float(getattr(self, "pitch", 0.0) or 0.0),
                                            "y": float(getattr(self, "y", 0.0) or 0.0),
                                            "z": float(getattr(self, "z", 0.0) or 0.0),
                                            "roll": float(getattr(self, "roll", 0.0) or 0.0),
                                            "x": float(getattr(self, "x", 0.0) or 0.0),
                                        }
                                        bounds_learner.record_movement_outcome(
                                            commanded=commanded_6d,
                                            actual=actual_6d,
                                            ik_failure=True,
                                        )
                                        self.log.debug("Recorded IK failure bounds at %s", commanded_6d)
                                    except Exception as bounds_err:
                                        self.log.debug("Failed to record bounds failure: %s", bounds_err)

                        # If too many consecutive failures, sync with hardware and optionally reset
                        now = time.time()
                        if ik_failure_count[0] >= IK_FAILURE_THRESHOLD and (now - last_ik_reset_time[0]) > 2.0:
                            self.log.warning(
                                "Too many IK failures (%d), syncing with hardware position",
                                ik_failure_count[0]
                            )
                            try:
                                # First try to sync with actual hardware position
                                # This corrects any drift between our tracking and reality
                                synced = self.sync_head_position()

                                if synced:
                                    self.log.info(
                                        "Synced position from hardware: yaw=%.1f°, pitch=%.1f°",
                                        self.yaw, self.pitch
                                    )
                                    # If we're already near center, just reset counter
                                    if abs(self.yaw) < 5.0 and abs(self.pitch) < 5.0:
                                        self.log.info("Already near center, clearing failure count")
                                        ik_failure_count[0] = 0
                                    else:
                                        # Move toward center from current synced position
                                        self.log.info("Moving to center from synced position")
                                        from maxim.motion.movement import move_head
                                        move_head(self.mini, 0, 0, 0, 0, 0, 0, 0.5)
                                        self.yaw = 0.0
                                        self.pitch = 0.0
                                        ik_failure_count[0] = 0
                                else:
                                    # Sync failed, fall back to blind reset
                                    self.log.warning("Hardware sync failed, blind reset to center")
                                    self.yaw = 0.0
                                    self.pitch = 0.0
                                    from maxim.motion.movement import move_head
                                    move_head(self.mini, 0, 0, 0, 0, 0, 0, 0.5)
                                    ik_failure_count[0] = 0

                                last_ik_reset_time[0] = now
                            except Exception as reset_err:
                                self.log.warning("Failed to reset head: %s", reset_err)
                    else:
                        # Movement succeeded - reset failure counter
                        ik_failure_count[0] = 0

                        # Record success for reachability learning
                        if position_info is not None:
                            default_network = getattr(self, "_default_network", None)
                            if default_network is not None:
                                attention = getattr(default_network, "_attention_network", None)
                                if attention is not None:
                                    try:
                                        attention.record_gaze(position_info, success=True)
                                    except Exception:
                                        pass

                        # Record outcome for workspace bounds learning
                        if commanded_6d is not None:
                            default_network = getattr(self, "_default_network", None)
                            if default_network is not None:
                                bounds_learner = getattr(default_network, "_bounds_learner", None)
                                if bounds_learner is not None:
                                    try:
                                        # Wait briefly for movement to settle, then sync
                                        time.sleep(min(0.1, float(movement_duration) * 0.3))
                                        self.sync_head_position()
                                        actual_6d = {
                                            "yaw": float(getattr(self, "yaw", 0.0) or 0.0),
                                            "pitch": float(getattr(self, "pitch", 0.0) or 0.0),
                                            "y": float(getattr(self, "y", 0.0) or 0.0),
                                            "z": float(getattr(self, "z", 0.0) or 0.0),
                                            "roll": float(getattr(self, "roll", 0.0) or 0.0),
                                            "x": float(getattr(self, "x", 0.0) or 0.0),
                                        }
                                        bounds_learner.record_movement_outcome(
                                            commanded=commanded_6d,
                                            actual=actual_6d,
                                            ik_failure=False,
                                        )
                                    except Exception as bounds_err:
                                        self.log.debug("Failed to record bounds outcome: %s", bounds_err)

                except Exception as e:
                    warn("Motor command failed: %s", e, logger=self.log)
                    self._note_connection_failure("motor", e)

                    # Also record as failure if we have position info
                    if position_info is not None:
                        default_network = getattr(self, "_default_network", None)
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
                    self._note_connection_failure("video", e)
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
                    self._note_connection_failure("audio", e)
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
                        frames_written += 1
                    except Exception as e:
                        warn("Failed to write video frame: %s", e, logger=self.log)

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
                        logger=self.log,
                    )
                else:
                    warn(
                        "No video frames were written to '%s'. The file may be empty/unplayable.",
                        video_path,
                        logger=self.log,
                    )

        def _audio_writer_worker() -> None:
            if not self.audio or audio_save_queue is None:
                return

            os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)
            os.makedirs(chunk_dir, exist_ok=True)

            sample_rate = int(audio_output_rate or audio_input_rate or 16000)
            chunk_frames = None
            if transcribe_process is not None:
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
                if t is key_thread or t is transcript_thread or t is cli_thread:
                    continue
                t.start()

        try:
            if not vision:
                self.log.info("Audio-only mode: recording until Ctrl+C.")
                while not stop_event.is_set():
                    time.sleep(0.25)
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

    def _request_mode(self, mode: str) -> None:
        requested = str(mode or "").strip().lower()
        if not requested:
            return

        current = str(getattr(self, "mode", "") or "").strip().lower()
        if requested != "shutdown" and requested == current:
            return

        try:
            self.log.info("Mode switch requested (%s -> %s).", current or None, requested)
        except Exception:
            pass

        self.requested_mode = requested

        # Only stop the live loop for shutdown mode
        # Other mode changes should just update the mode without stopping
        if requested == "shutdown":
            ev = getattr(self, "_live_stop_event", None)
            if ev is not None:
                try:
                    if int(getattr(self, "verbosity", 0) or 0) >= 2:
                        self.log.debug("Stopping live loop for shutdown")
                    ev.set()
                except Exception:
                    pass

    def request_shutdown(self) -> None:
        self._request_mode("shutdown")

    def request_sleep(self) -> None:
        """Enter sleep processing state (minimal processing, keyword monitoring)."""
        self._state_manager.request_sleep()

    def request_wake(self) -> None:
        """Wake from sleep processing state."""
        self._state_manager.request_wake()

    def request_observe(self) -> None:
        """Legacy: switch to observe strategy."""
        self._state_manager.request_strategy_observe()

    # Strategy switch methods (called by phrase responses)
    def request_strategy_observe(self) -> None:
        self._state_manager.request_strategy_observe()

    def request_strategy_explore(self) -> None:
        self._state_manager.request_strategy_explore()

    def request_strategy_research(self) -> None:
        self._state_manager.request_strategy_research()

    def request_strategy_assist(self) -> None:
        self._state_manager.request_strategy_assist()

    def request_strategy_reflect(self) -> None:
        self._state_manager.request_strategy_reflect()

    def request_strategy_learn(self) -> None:
        self._state_manager.request_strategy_learn()

    # Operational mode switch methods (called by phrase responses)
    def request_mode_passive(self) -> None:
        self._state_manager.request_mode_passive()

    def request_mode_active(self) -> None:
        self._state_manager.request_mode_active()

    def request_mode_singularity(self) -> None:
        self._state_manager.request_mode_singularity()

    def update_interests(
        self,
        add: list[int] | None = None,
        remove: list[int] | None = None,
    ) -> None:
        """Update interest class IDs for salience boosting.

        Interest classes receive higher salience scores in the perception pipeline,
        making them more likely to be noticed and acted upon. All COCO classes are
        always detected; this controls prioritization, not visibility.
        """
        updated = set(int(v) for v in (getattr(self, "interests", []) or []) if v is not None)
        if add:
            updated.update(int(v) for v in add if v is not None)
        if remove:
            updated.difference_update(int(v) for v in remove if v is not None)

        self.interests = sorted(updated)

        agent = getattr(self, "_agentic_agent", None)
        if agent is not None and hasattr(agent, "update_interests"):
            try:
                agent.update_interests(add=add, remove=remove)
            except Exception as e:
                warn("Failed to update agent interests: %s", e, logger=self.log)

        state = getattr(self, "_agentic_state", None)
        if state is not None:
            try:
                data = getattr(state, "data", None)
                if isinstance(data, dict):
                    runtime = data.get("maxim_runtime")
                    if isinstance(runtime, dict):
                        runtime["interests"] = list(self.interests)
            except Exception:
                pass

        try:
            self.log.info("Updated interests: %s", self.interests)
        except Exception:
            pass

    def wake_up_agentic(self) -> None:
        """Wake up Reachy and transition to exploration mode.

        Called when the wake word "maxim" is detected. This:
        1. Wakes up the robot motors
        2. Switches mode from sleep to exploration
        3. Starts the agentic runtime if available
        """
        try:
            self._voice_agentic_enabled = True
        except Exception:
            pass

        # Wake up the robot
        try:
            mini = getattr(self, "mini", None)
            if mini is not None:
                self._enqueue_motor(mini.wake_up)
                self._woke_up = True
                self.sleeping = False  # Mark as no longer sleeping
        except Exception as e:
            warn("Failed to wake up Reachy: %s", e, logger=self.log)

        # Switch to exploration mode if currently in sleep mode
        # This triggers a live loop restart with vision=True
        current_mode = str(getattr(self, "mode", "") or "").strip().lower()
        if current_mode == "sleep":
            try:
                self.log.info("Waking up from sleep -> requesting exploration mode")
                self._request_mode("exploration")
            except Exception:
                pass

        self._start_agentic_runtime()
        agentic_thread = getattr(self, "_agentic_thread", None)
        if agentic_thread is None or not getattr(agentic_thread, "is_alive", lambda: False)():
            try:
                self._voice_agentic_enabled = False
            except Exception:
                pass

    def label_outcome(
        self,
        code: int,
        *,
        source: str | None = None,
        trigger: str | None = None,
        note: str | None = None,
    ) -> None:
        try:
            code_int = int(code)
        except Exception:
            return
        if code_int < 0:
            code_int = 0
        if code_int > 9:
            code_int = 9
        self._outcome_code = int(code_int)

        last_motor_sample_id = None
        sample = getattr(self, "_last_motor_sample", None)
        if isinstance(sample, dict):
            last_motor_sample_id = sample.get("sample_id")

        target_action_event_id = getattr(self, "_last_action_event_id", None)

        transcript = getattr(self, "_last_transcript_event", None)
        transcript_ref = None
        if isinstance(transcript, dict):
            transcript_ref = {
                "chunk_index": transcript.get("chunk_index"),
                "start_s": transcript.get("start_s"),
                "end_s": transcript.get("end_s"),
                "text": str(transcript.get("text", "") or "")[:280],
            }

        record: dict = {
            "kind": "outcome_label",
            "event_id": uuid.uuid4().hex,
            "time": float(time.time()),
            "source": str(source) if source is not None else None,
            "trigger": str(trigger) if trigger is not None else None,
            "code": int(code_int),
            "note": str(note) if note is not None else None,
            "run_id": getattr(self, "run_id", None),
            "mode": getattr(self, "mode", None),
            "epoch": int(getattr(self, "current_epoch", 0) or 0),
            "video_path": getattr(self, "video_path", None),
            "audio_path": getattr(self, "audio_path", None),
            "transcript_path": getattr(self, "transcript_path", None),
            "target_action_event_id": target_action_event_id,
            "last_motor_sample_id": last_motor_sample_id,
            "transcript": transcript_ref,
        }

        self._log_event(record, flush=True)

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

    def sync_head_position(self) -> bool:
        """Sync internal position tracking with actual hardware position.

        Reads the current head pose from the reachy_mini SDK and updates
        the internal yaw/pitch tracking to match. This helps correct drift
        between software tracking and actual hardware position.

        Returns:
            True if sync was successful, False otherwise.
        """
        try:
            import math

            # Get current joint positions - this gives us body_yaw directly
            # Head joints: 6 Stewart platform joints + 1 body_yaw (index 6)
            # Antenna joints: 2 antenna positions
            head_joints, antenna_joints = self.mini.get_current_joint_positions()

            # Body yaw is the 7th joint (index 6) - in radians
            if len(head_joints) >= 7:
                body_yaw_rad = head_joints[6]
                body_yaw_deg = math.degrees(body_yaw_rad)
                self.body_yaw = body_yaw_deg
            else:
                body_yaw_deg = float(getattr(self, "body_yaw", 0.0) or 0.0)

            # Get current pose matrix from SDK
            pose_matrix = self.mini.get_current_head_pose()

            # Extract rotation matrix (upper-left 3x3)
            R = pose_matrix[:3, :3]

            # Extract Euler angles from rotation matrix
            # Using ZYX convention (yaw, pitch, roll)
            # Check for gimbal lock
            if abs(R[2, 0]) < 0.9999:
                pitch = -math.asin(R[2, 0])
                yaw_world = math.atan2(R[1, 0], R[0, 0])
                roll = math.atan2(R[2, 1], R[2, 2])
            else:
                # Gimbal lock - pitch is ±90°
                pitch = math.copysign(math.pi / 2, -R[2, 0])
                yaw_world = math.atan2(-R[0, 1], R[1, 1])
                roll = 0.0

            # Convert to degrees
            yaw_world_deg = math.degrees(yaw_world)
            pitch_deg = math.degrees(pitch)
            roll_deg = math.degrees(roll)

            # The pose matrix from get_current_head_pose() is in WORLD frame.
            # We need to convert to HEAD-relative yaw by accounting for body rotation.
            #
            # SDK sign convention:
            # - Negative body_yaw = body has turned LEFT
            # - Positive body_yaw = body has turned RIGHT
            #
            # Formula: head_yaw = world_yaw - body_yaw
            #
            # Example: body turned left 77° (body_yaw=-77), head at world_yaw=-65
            # - head_yaw = -65 - (-77) = -65 + 77 = +12° (head is 12° right of body forward)
            yaw_deg = yaw_world_deg - body_yaw_deg  # SUBTRACT to get head-relative

            # Normalize to [-180, 180]
            while yaw_deg > 180:
                yaw_deg -= 360
            while yaw_deg < -180:
                yaw_deg += 360

            # DEBUG: Log body_yaw and world_yaw to diagnose frame issues
            if abs(yaw_deg) > 55.0:
                self.log.warning(
                    "SYNC_DEBUG: yaw=%.1f EXCEEDS physical limit | world_yaw=%.1f body_yaw=%.1f",
                    yaw_deg, yaw_world_deg, body_yaw_deg
                )

            # SANITY CHECK: The head physically cannot rotate more than ±55° relative
            # to the body. If calculated yaw exceeds this, clamp it to prevent bad behavior.
            # Previous ±90° threshold was too permissive and allowed invalid values through.
            HEAD_YAW_PHYSICAL_LIMIT = 55.0
            if abs(yaw_deg) > HEAD_YAW_PHYSICAL_LIMIT:
                old_yaw = float(getattr(self, "yaw", 0.0) or 0.0)
                # Clamp to physical limit (preserving direction)
                clamped_yaw = max(-HEAD_YAW_PHYSICAL_LIMIT, min(HEAD_YAW_PHYSICAL_LIMIT, yaw_deg))
                self.log.warning(
                    "Sync: clamping invalid yaw=%.1f -> %.1f (world=%.1f - body=%.1f, old=%.1f)",
                    yaw_deg, clamped_yaw, yaw_world_deg, body_yaw_deg, old_yaw
                )
                yaw_deg = clamped_yaw

            # SANITY CHECK: The head physically cannot pitch more than ±35° relative
            # to the body. Clamp to prevent invalid values from causing issues.
            HEAD_PITCH_PHYSICAL_LIMIT = 35.0
            if abs(pitch_deg) > HEAD_PITCH_PHYSICAL_LIMIT:
                old_pitch = float(getattr(self, "pitch", 0.0) or 0.0)
                clamped_pitch = max(-HEAD_PITCH_PHYSICAL_LIMIT, min(HEAD_PITCH_PHYSICAL_LIMIT, pitch_deg))
                self.log.warning(
                    "Sync: clamping invalid pitch=%.1f -> %.1f (old=%.1f)",
                    pitch_deg, clamped_pitch, old_pitch
                )
                pitch_deg = clamped_pitch

            # SANITY CHECK: Roll also limited to ±35° physical limit
            HEAD_ROLL_PHYSICAL_LIMIT = 35.0
            if abs(roll_deg) > HEAD_ROLL_PHYSICAL_LIMIT:
                old_roll = float(getattr(self, "roll", 0.0) or 0.0)
                clamped_roll = max(-HEAD_ROLL_PHYSICAL_LIMIT, min(HEAD_ROLL_PHYSICAL_LIMIT, roll_deg))
                self.log.warning(
                    "Sync: clamping invalid roll=%.1f -> %.1f (old=%.1f)",
                    roll_deg, clamped_roll, old_roll
                )
                roll_deg = clamped_roll

            # Extract translation (convert meters to mm)
            x_mm = pose_matrix[0, 3] * 1000
            y_mm = pose_matrix[1, 3] * 1000
            z_mm = pose_matrix[2, 3] * 1000

            # Update internal tracking
            old_yaw = float(getattr(self, "yaw", 0.0) or 0.0)
            old_pitch = float(getattr(self, "pitch", 0.0) or 0.0)
            old_y = float(getattr(self, "y", 0.0) or 0.0)
            old_z = float(getattr(self, "z", 0.0) or 0.0)

            self.x = x_mm
            self.y = y_mm
            self.z = z_mm
            self.roll = roll_deg
            self.pitch = pitch_deg
            self.yaw = yaw_deg

            # Only log if there's a significant difference (>2° or >2mm)
            yaw_diff = abs(old_yaw - yaw_deg)
            pitch_diff = abs(old_pitch - pitch_deg)
            y_diff = abs(old_y - y_mm)
            z_diff = abs(old_z - z_mm)

            if yaw_diff > 2.0 or pitch_diff > 2.0 or y_diff > 2.0 or z_diff > 2.0:
                self.log.debug(
                    "Sync: yaw=%.1f (body=%.1f), pitch=%.1f, y=%.1f, z=%.1f",
                    yaw_deg, body_yaw_deg, pitch_deg, y_mm, z_mm
                )

            # Record position for pain detection (if enabled)
            pain_bridge = getattr(
                getattr(self, "_default_network", None), "_pain_bridge", None
            )
            if pain_bridge is not None:
                try:
                    pain_bridge.detector.record_position(
                        yaw=yaw_deg,
                        pitch=pitch_deg,
                        x=x_mm,
                        y=y_mm,
                        z=z_mm,
                        roll=roll_deg,
                    )
                except Exception as pain_e:
                    self.log.debug("Pain recording failed: %s", pain_e)

            return True

        except Exception as e:
            self.log.warning("Failed to sync head position: %s", e)
            return False

    # Safe pixel bounds for look_at_image to prevent IK failures.
    # With translation-first approach, we can use more of the frame.
    _LOOK_AT_PIXEL_BOUNDS = {
        "u_min": 64,    # ~10% from left edge
        "u_max": 576,   # ~10% from right edge
        "v_min": 48,    # ~10% from top edge
        "v_max": 432,   # ~10% from bottom edge
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # BLENDED 6-DOF WORKSPACE LIMITS
    # ═══════════════════════════════════════════════════════════════════════════
    # We use BOTH translation AND rotation in a balanced way:
    # - Translation provides mechanical stability for the Stewart platform
    # - Rotation provides natural head-movement feel
    #
    # Coordinate system (from SDK docs):
    #   X = forward (positive = toward what you're looking at)
    #   Y = left (positive = left, negative = right)
    #   Z = upward (positive = up, negative = down)
    #
    # SDK hint: "1 degree ≈ 1 mm" as rough equivalence
    # ═══════════════════════════════════════════════════════════════════════════

    # Translation limits (mm) - moderate range for stability
    _SAFE_X_LIMIT = 15.0    # Forward/backward (mm) - rarely used
    _SAFE_Y_LIMIT = 18.0    # Left/right (mm)
    _SAFE_Z_LIMIT = 15.0    # Up/down (mm)

    # Rotation limits (degrees) - based on Reachy Mini SDK documentation
    # SDK limits: Roll/Pitch [-40°, +40°], Yaw delta max 65° (head relative to body)
    # Using slightly conservative values for safety margin
    _SAFE_ROLL_LIMIT = 35.0   # Head tilt (degrees) - SDK allows ±40°
    _SAFE_PITCH_LIMIT = 35.0  # Look up/down (degrees) - SDK allows ±40°
    _SAFE_YAW_LIMIT = 55.0    # Look left/right (degrees) - SDK allows ±65° (relative to body)

    # Turn-around trigger settings
    # When head is at horizontal limit and target is beyond, rotate body instead
    # Note: Threshold is now LEARNED via pain detection - no hardcoded _TURN_AROUND_THRESHOLD
    _TURN_AROUND_MIN_PIXEL_OFFSET = 60  # Min pixel offset to trigger (avoid small targets)
    _TURN_AROUND_COOLDOWN = 12.0  # Seconds between turn_around triggers
    _TURN_AROUND_ANGLE = 45.0  # Degrees to rotate body (matches yaw limit for smooth transition)
    _TURN_AROUND_DURATION = 6.0  # Seconds for body rotation (slow and deliberate)
    _TURN_AROUND_PAIN_THRESHOLD = 0.3  # Pain risk threshold to trigger turn-around
    _TURN_AROUND_BOUNDS_THRESHOLD = 0.75  # Fallback: bounds usage threshold if pain not available

    # Camera parameters for converting pixels to movement
    # Internal coordinate system (matches behavior/detection pipeline)
    _IMAGE_CENTER_U = 320.0
    _IMAGE_CENTER_V = 240.0
    _IMAGE_WIDTH = 640.0
    _IMAGE_HEIGHT = 480.0

    # SDK's expected resolution (camera native resolution)
    # The SDK's look_at_image expects coordinates in the camera's native resolution
    _SDK_IMAGE_WIDTH = 1920.0
    _SDK_IMAGE_HEIGHT = 1080.0
    _SDK_IMAGE_CENTER_U = 960.0   # 1920 / 2
    _SDK_IMAGE_CENTER_V = 540.0   # 1080 / 2

    # Camera FOV (approximate)
    _HORIZONTAL_FOV_DEG = 70.0  # degrees
    _VERTICAL_FOV_DEG = 50.0    # degrees

    # ═══════════════════════════════════════════════════════════════════════════
    # BLENDED MOVEMENT PER PIXEL
    # ═══════════════════════════════════════════════════════════════════════════
    # Both translation and rotation contribute to movement.
    # Rotation is now more prominent for natural head movement,
    # while translation provides the mechanical stability base.
    # For a 100px offset: ~5mm translation + ~10° rotation
    _MM_PER_PIXEL_H = 0.05    # mm translation per pixel horizontally
    _MM_PER_PIXEL_V = 0.04    # mm translation per pixel vertically
    _DEG_PER_PIXEL_H = 0.10   # degrees rotation per pixel horizontally
    _DEG_PER_PIXEL_V = 0.08   # degrees rotation per pixel vertically

    def _get_workspace_limits(self) -> dict[str, float]:
        """Get workspace limits, preferring learned bounds over hardcoded.

        Returns:
            Dict with x, y, z, roll, pitch, yaw limits (all positive values).
        """
        # Try to get learned bounds from bounds_learner
        bounds_learner = None
        default_network = getattr(self, "_default_network", None)
        if default_network is not None:
            bounds_learner = getattr(default_network, "_bounds_learner", None)

        if bounds_learner is not None:
            return {
                "x": bounds_learner.get_bound("x"),
                "y": bounds_learner.get_bound("y"),
                "z": bounds_learner.get_bound("z"),
                "roll": bounds_learner.get_bound("roll"),
                "pitch": bounds_learner.get_bound("pitch"),
                "yaw": bounds_learner.get_bound("yaw"),
            }

        # Fallback to hardcoded limits
        return {
            "x": self._SAFE_X_LIMIT,
            "y": self._SAFE_Y_LIMIT,
            "z": self._SAFE_Z_LIMIT,
            "roll": self._SAFE_ROLL_LIMIT,
            "pitch": self._SAFE_PITCH_LIMIT,
            "yaw": self._SAFE_YAW_LIMIT,
        }

    def _clamp_to_workspace_6d(
        self,
        x: float, y: float, z: float,
        roll: float, pitch: float, yaw: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Clamp 6-DOF pose to the reachable Stewart platform workspace.

        The Stewart platform has a complex 6D workspace. We use a simplified
        model with separate ellipsoids for translation and rotation, plus
        a combined constraint to prevent extreme combinations.

        Translation (x, y, z) is the PRIMARY movement mechanism.
        Rotation (roll, pitch, yaw) is SECONDARY for personality/fine-tuning.

        Uses learned workspace bounds when available, falling back to hardcoded limits.

        Args:
            x, y, z: Translation in mm.
            roll, pitch, yaw: Rotation in degrees.

        Returns:
            Clamped (x, y, z, roll, pitch, yaw) tuple.
        """
        import math

        # Get limits (learned or hardcoded)
        limits = self._get_workspace_limits()
        x_limit = limits["x"]
        y_limit = limits["y"]
        z_limit = limits["z"]
        roll_limit = limits["roll"]
        pitch_limit = limits["pitch"]
        yaw_limit = limits["yaw"]

        # 1. Clamp translation to rectangular bounds first
        x = max(-x_limit, min(x_limit, x))
        y = max(-y_limit, min(y_limit, y))
        z = max(-z_limit, min(z_limit, z))

        # 2. Clamp rotation to rectangular bounds
        roll = max(-roll_limit, min(roll_limit, roll))
        pitch = max(-pitch_limit, min(pitch_limit, pitch))
        yaw = max(-yaw_limit, min(yaw_limit, yaw))

        # 3. Check translation ellipsoid: (x/max_x)² + (y/max_y)² + (z/max_z)² <= 1
        if x_limit > 0 and y_limit > 0 and z_limit > 0:
            norm_x = x / x_limit
            norm_y = y / y_limit
            norm_z = z / z_limit
            trans_dist = norm_x**2 + norm_y**2 + norm_z**2

            if trans_dist > 1.0:
                scale = 1.0 / math.sqrt(trans_dist)
                x, y, z = x * scale, y * scale, z * scale
                self.log.debug("Clamped translation to ellipsoid")

        # 4. Check rotation ellipsoid: (roll/max)² + (pitch/max)² + (yaw/max)² <= 1
        if roll_limit > 0 and pitch_limit > 0 and yaw_limit > 0:
            norm_roll = roll / roll_limit
            norm_pitch = pitch / pitch_limit
            norm_yaw = yaw / yaw_limit
            rot_dist = norm_roll**2 + norm_pitch**2 + norm_yaw**2

            if rot_dist > 1.0:
                scale = 1.0 / math.sqrt(rot_dist)
                roll, pitch, yaw = roll * scale, pitch * scale, yaw * scale
                self.log.debug("Clamped rotation to ellipsoid")

        # 5. Combined constraint: only reduce rotation when translation is VERY high
        # The blended approach allows both translation and rotation to work together,
        # but we still need to prevent extreme combined positions that cause IK failure.
        trans_usage = math.sqrt((x/x_limit)**2 + (y/y_limit)**2 + (z/z_limit)**2) if x_limit > 0 else 0
        rot_usage = math.sqrt((roll/roll_limit)**2 + (pitch/pitch_limit)**2 + (yaw/yaw_limit)**2) if yaw_limit > 0 else 0

        # Only apply combined constraint when BOTH are high
        combined = trans_usage * 0.5 + rot_usage * 0.5  # Equal weighting
        if combined > 0.85:  # If combined usage exceeds 85%
            # Scale both down proportionally to stay within workspace
            scale = 0.85 / combined
            x, y, z = x * scale, y * scale, z * scale
            roll, pitch, yaw = roll * scale, pitch * scale, yaw * scale
            self.log.debug("Scaled both trans/rot due to combined limit: %.2f -> %.2f", combined, 0.85)

        return (x, y, z, roll, pitch, yaw)

    def _clamp_to_workspace(
        self, yaw: float, pitch: float
    ) -> tuple[float, float]:
        """Legacy 2D clamp for backward compatibility.

        Delegates to 6D clamping with zero translation.
        """
        _, _, _, _, pitch_out, yaw_out = self._clamp_to_workspace_6d(
            0, 0, 0, 0, pitch, yaw
        )
        return (yaw_out, pitch_out)

    def _calculate_movement_for_pixel(
        self, du: float, dv: float
    ) -> tuple[float, float, float, float, float, float]:
        """Calculate 6-DOF movement needed to center a pixel offset.

        Uses TRANSLATION as the primary movement mechanism:
        - Y translation for horizontal movement (left/right)
        - Z translation for vertical movement (up/down)

        Uses ROTATION as secondary for personality and efficiency:
        - Yaw for horizontal fine-tuning
        - Pitch for vertical fine-tuning

        Args:
            du: Horizontal pixel offset from center (positive = right).
            dv: Vertical pixel offset from center (positive = down).

        Returns:
            (dx, dy, dz, droll, dpitch, dyaw) deltas to apply.
        """
        # Translation deltas (primary movement)
        # Note: positive du (right of center) requires negative Y (move head right)
        # Note: positive dv (below center) requires negative Z (move head down)
        dy = -du * self._MM_PER_PIXEL_H  # Y = left, so negative for right
        dz = -dv * self._MM_PER_PIXEL_V  # Z = up, so negative for down
        dx = 0.0  # X (forward) not typically needed for looking at pixels

        # Rotation deltas (secondary/supplementary)
        # Same sign convention as translation
        dyaw = -du * self._DEG_PER_PIXEL_H   # Turn right for right-of-center
        dpitch = dv * self._DEG_PER_PIXEL_V  # Look down for below-center
        droll = 0.0  # Roll not used for pixel tracking

        return (dx, dy, dz, droll, dpitch, dyaw)

    def look_at_image(
        self,
        u: int,
        v: int,
        *,
        duration: Optional[float] = None,
        perform_movement: bool = True,
        min_reachability: float = 0.2,
        clamp_to_bounds: bool = True,
    ) -> None:
        """Look at a position in image coordinates.

        Args:
            u: Horizontal pixel coordinate.
            v: Vertical pixel coordinate.
            duration: Movement duration in seconds.
            perform_movement: Whether to actually move (vs just calculate).
            min_reachability: Minimum reachability score to attempt movement (0-1).
                              Set to 0 to disable reachability gating.
            clamp_to_bounds: Whether to clamp coordinates to safe pixel bounds.
                             This prevents the robot from drifting outside servo limits.
        """
        if duration is None:
            duration = getattr(self, "duration", 0.5)

        # Clamp pixel coordinates to safe bounds to prevent drift outside servo limits
        if clamp_to_bounds:
            bounds = self._LOOK_AT_PIXEL_BOUNDS
            u_clamped = max(bounds["u_min"], min(bounds["u_max"], int(u)))
            v_clamped = max(bounds["v_min"], min(bounds["v_max"], int(v)))
            if u_clamped != int(u) or v_clamped != int(v):
                self.log.debug(
                    "Clamped look_at_image coordinates: (%d, %d) -> (%d, %d)",
                    u, v, u_clamped, v_clamped
                )
            u, v = u_clamped, v_clamped

        position = (int(u), int(v))

        # Check reachability before attempting movement (dynamic bounds learning)
        if min_reachability > 0:
            default_network = getattr(self, "_default_network", None)
            if default_network is not None:
                attention = getattr(default_network, "_attention_network", None)
                if attention is not None:
                    try:
                        reachability = attention.get_reachability(position)
                        if reachability < min_reachability:
                            self.log.debug(
                                "Skipping unreachable position (%d, %d) - reachability %.2f < %.2f",
                                u, v, reachability, min_reachability
                            )
                            return
                    except Exception:
                        pass  # Continue if reachability check fails

        # ═══════════════════════════════════════════════════════════════════════════
        # HYBRID APPROACH: SDK look_at_image + position-aware clamping
        # ═══════════════════════════════════════════════════════════════════════════
        # The SDK's look_at_image WORKS for centering - it knows the camera model.
        # The problem was IK failures at extreme positions.
        # Solution: clamp pixels based on current position to prevent going further
        # into the limits.
        # ═══════════════════════════════════════════════════════════════════════════
        if perform_movement:
            # Sync to get accurate current position
            self.sync_head_position()

            cur_yaw = float(getattr(self, "yaw", 0.0) or 0.0)
            cur_pitch = float(getattr(self, "pitch", 0.0) or 0.0)
            cur_y = float(getattr(self, "y", 0.0) or 0.0)
            cur_z = float(getattr(self, "z", 0.0) or 0.0)

            # Calculate pixel offset
            du = float(u) - self._IMAGE_CENTER_U
            dv = float(v) - self._IMAGE_CENTER_V

            # ADAPTIVE MOVEMENT GAIN: Use FocusLearner for adaptive dampening
            # Instead of a fixed dampening factor, we learn the optimal gain
            # from focus feedback - how far the target ends up from center
            # after each movement.
            default_network = getattr(self, "_default_network", None)
            focus_learner = getattr(default_network, "_focus_learner", None) if default_network else None

            # Debug: log why FocusLearner might not be available
            if focus_learner is None:
                if default_network is None:
                    self.log.debug("FocusLearner unavailable: _default_network is None")
                else:
                    self.log.debug("FocusLearner unavailable: _focus_learner is None in DefaultNetwork")

            if focus_learner is not None:
                gain_h, gain_v = focus_learner.get_gain(du, dv)
                # Record intent for later result tracking
                focus_learner.record_intent(du, dv, gain_h, gain_v)
                self.log.info(
                    "FocusLearner: raw_du=%.1f raw_dv=%.1f gain_h=%.3f gain_v=%.3f",
                    du, dv, gain_h, gain_v,
                )
                du = du * gain_h
                dv = dv * gain_v
            else:
                # Fallback to fixed dampening if FocusLearner not available
                DAMPENING_FACTOR = 0.5
                self.log.warning("Using fallback DAMPENING_FACTOR=0.5 (FocusLearner unavailable)")
                du = du * DAMPENING_FACTOR
                dv = dv * DAMPENING_FACTOR

            # POSITION-AWARE CLAMPING: If we're already near a limit,
            # don't allow pixels that would push us further that direction
            # This prevents accumulating into IK failure territory
            #
            # We check BOTH rotation (yaw, pitch) AND translation (y, z)
            # and use the MORE restrictive of the two.

            # === HORIZONTAL AXIS (left/right) ===
            # Rotation: yaw (positive = looking left)
            # Translation: Y (positive = platform shifted left in mm)
            # Use learned bounds when available
            limits = self._get_workspace_limits()
            yaw_limit = limits["yaw"]
            y_limit = limits["y"]
            yaw_usage = abs(cur_yaw) / yaw_limit if yaw_limit > 0 else 0
            y_usage = abs(cur_y) / y_limit if y_limit > 0 else 0
            h_usage = max(yaw_usage, y_usage)  # Use the more restrictive

            # Check direction: positive yaw/y = left, negative = right
            looking_left = cur_yaw > 0 or cur_y > 0
            looking_right = cur_yaw < 0 or cur_y < 0

            # Get pain bridge for learned movement control
            pain_bridge = getattr(
                getattr(self, "_default_network", None), "_pain_bridge", None
            )

            # === TURN AROUND TRIGGER (LEARNED) ===
            # If movement in target direction would cause pain, rotate body instead
            import time as time_module
            now = time_module.time()
            can_turn = (now - self._last_turn_around_time) > self._TURN_AROUND_COOLDOWN

            # Check if pixel is significantly in the blocked direction
            target_is_beyond = (
                (looking_left and du < -self._TURN_AROUND_MIN_PIXEL_OFFSET) or
                (looking_right and du > self._TURN_AROUND_MIN_PIXEL_OFFSET)
            )

            if can_turn and target_is_beyond:
                # Check pain prediction for the proposed movement
                should_turn = False
                turn_reason = ""

                if pain_bridge is not None:
                    # Create action signature for this movement
                    dyaw = abs(du * self._DEG_PER_PIXEL_H)
                    action_sig = f"look_at:dy={dyaw:.0f}:dp=0"
                    pain_risk = pain_bridge.get_pain_risk(action_sig)

                    if pain_risk >= self._TURN_AROUND_PAIN_THRESHOLD:
                        should_turn = True
                        turn_reason = f"pain_risk={pain_risk:.2f}"
                else:
                    # Fallback: use bounds usage threshold if pain prediction unavailable
                    if h_usage > self._TURN_AROUND_BOUNDS_THRESHOLD:
                        should_turn = True
                        turn_reason = f"bounds_usage={h_usage:.2f}"

                if should_turn:
                    # Determine turn direction
                    # Looking left + pixel further left = turn left (positive angle)
                    # Looking right + pixel further right = turn right (negative angle)
                    turn_angle = self._TURN_AROUND_ANGLE if looking_left else -self._TURN_AROUND_ANGLE

                    self.log.info(
                        "look_at_image triggering turn_around: %s, du=%.1f, turning %.0f°",
                        turn_reason, du, turn_angle
                    )
                    self._last_turn_around_time = now

                    # Record turn_around as action start for positive learning
                    if pain_bridge is not None:
                        try:
                            pain_bridge.record_action_start(
                                "turn_around",
                                context={"reason": turn_reason, "angle": turn_angle},
                            )
                        except Exception:
                            pass

                    self.turn_around(turn_angle, duration=self._TURN_AROUND_DURATION, recenter_head=True)

                    # Record successful turn_around (no pain) as positive outcome
                    if pain_bridge is not None:
                        try:
                            pain_bridge.record_action_complete(success=True)
                        except Exception:
                            pass

                    return  # Exit early - turn_around handles the movement

            # === LEARNED POSITION CLAMPING (HORIZONTAL) ===
            # Use pain prediction to determine how much to restrict movement
            h_restrict = 0.0
            if pain_bridge is not None:
                # Calculate pain risk for horizontal movement
                dyaw_h = abs(du * self._DEG_PER_PIXEL_H)
                action_sig_h = f"look_at:dy={dyaw_h:.0f}:dp=0"
                h_pain_risk = pain_bridge.get_pain_risk(action_sig_h)
                # Use pain risk directly as restriction factor
                h_restrict = min(1.0, h_pain_risk * 2.0)  # Scale up for faster response
            elif h_usage > 0.5:
                # Fallback to bounds-based restriction
                h_restrict = min(1.0, (h_usage - 0.5) * 2)

            if h_restrict > 0.1:
                if looking_left and du < 0:  # At left limit, pixel is further left
                    du = du * (1.0 - h_restrict * 0.8)  # Reduce by up to 80%
                elif looking_right and du > 0:  # At right limit, pixel is further right
                    du = du * (1.0 - h_restrict * 0.8)

            # === VERTICAL AXIS (up/down) ===
            # Rotation: pitch (positive = looking down in this system)
            # Translation: Z (positive = platform raised up = camera higher = looking down)
            # Use learned bounds (already fetched in horizontal axis section)
            pitch_limit = limits["pitch"]
            z_limit = limits["z"]
            pitch_usage = abs(cur_pitch) / pitch_limit if pitch_limit > 0 else 0
            z_usage = abs(cur_z) / z_limit if z_limit > 0 else 0
            v_usage = max(pitch_usage, z_usage)  # Use the more restrictive

            # === LEARNED POSITION CLAMPING (VERTICAL) ===
            v_restrict = 0.0
            if pain_bridge is not None:
                # Calculate pain risk for vertical movement
                dpitch_v = abs(dv * self._DEG_PER_PIXEL_V)
                action_sig_v = f"look_at:dy=0:dp={dpitch_v:.0f}"
                v_pain_risk = pain_bridge.get_pain_risk(action_sig_v)
                v_restrict = min(1.0, v_pain_risk * 2.0)
            elif v_usage > 0.5:
                # Fallback to bounds-based restriction
                v_restrict = min(1.0, (v_usage - 0.5) * 2)

            if v_restrict > 0.1:
                # Image v: positive = below center (looking down)
                # Pitch positive = looking down, Z positive = raised up = looking down
                looking_down = cur_pitch > 0 or cur_z > 0
                looking_up = cur_pitch < 0 or cur_z < 0
                if looking_down and dv > 0:  # At down limit, pixel is further down
                    dv = dv * (1.0 - v_restrict * 0.8)
                elif looking_up and dv < 0:  # At up limit, pixel is further up
                    dv = dv * (1.0 - v_restrict * 0.8)

            # Calculate final pixel coordinates
            final_u = int(self._IMAGE_CENTER_U + du)
            final_v = int(self._IMAGE_CENTER_V + dv)

            # Apply standard bounds as final safety
            px_bounds = self._LOOK_AT_PIXEL_BOUNDS
            final_u = max(px_bounds["u_min"], min(px_bounds["u_max"], final_u))
            final_v = max(px_bounds["v_min"], min(px_bounds["v_max"], final_v))

            # ENHANCED DEBUG: Trace pixel-to-motor direction
            # du > 0 means target is RIGHT of center (u=320), so yaw should DECREASE (turn right)
            # du < 0 means target is LEFT of center, so yaw should INCREASE (turn left)
            du_direction = "RIGHT" if du > 0 else "LEFT" if du < 0 else "CENTER"
            expected_yaw_change = "decrease" if du > 0 else "increase" if du < 0 else "none"
            self.log.warning(
                "LOOK_AT_DEBUG: input_pixel=(%d,%d) center=(320,240) du=%.1f (%s) dv=%.1f "
                "-> expect_yaw_to_%s | cur_yaw=%.1f cur_pitch=%.1f | final_pixel=(%d,%d)",
                u, v, du, du_direction, dv, expected_yaw_change,
                cur_yaw, cur_pitch, final_u, final_v
            )

            # Calculate estimated commanded 6D pose for bounds learning
            # This is an approximation since the SDK handles the actual conversion
            delta = self._calculate_movement_for_pixel(du, dv)
            commanded_6d = {
                "yaw": cur_yaw + delta[5],  # dyaw
                "pitch": cur_pitch + delta[4],  # dpitch
                "y": cur_y + delta[1],  # dy
                "z": cur_z + delta[2],  # dz
                "roll": float(getattr(self, "roll", 0.0) or 0.0) + delta[3],  # droll
                "x": float(getattr(self, "x", 0.0) or 0.0) + delta[0],  # dx
            }

            # Record action start for pain detection (before movement)
            # Includes target position for movement failure detection
            pain_bridge = getattr(
                getattr(self, "_default_network", None), "_pain_bridge", None
            )
            if pain_bridge is not None:
                try:
                    # Create action signature from movement magnitude
                    dyaw = abs(delta[5])
                    dpitch = abs(delta[4])
                    action_sig = f"look_at:dy={dyaw:.0f}:dp={dpitch:.0f}"
                    pain_bridge.record_action_start(
                        action_sig,
                        context={"position": position, "commanded_6d": commanded_6d},
                        # Pass target position for movement failure detection
                        target_yaw=commanded_6d.get("yaw"),
                        target_pitch=commanded_6d.get("pitch"),
                        target_y=commanded_6d.get("y"),
                        target_z=commanded_6d.get("z"),
                    )
                except Exception as e:
                    self.log.debug("Pain action start recording failed: %s", e)

            # SCALE COORDINATES: Convert from our 640x480 to SDK's 1920x1080
            # The SDK expects coordinates in the camera's native resolution
            sdk_scale_x = self._SDK_IMAGE_WIDTH / self._IMAGE_WIDTH
            sdk_scale_y = self._SDK_IMAGE_HEIGHT / self._IMAGE_HEIGHT
            sdk_u = int(final_u * sdk_scale_x)
            sdk_v = int(final_v * sdk_scale_y)

            self.log.warning(
                "LOOK_AT_SDK: internal_pixel=(%d,%d) -> sdk_pixel=(%d,%d) scale=(%.2f,%.2f)",
                final_u, final_v, sdk_u, sdk_v, sdk_scale_x, sdk_scale_y
            )

            # Use the SDK's look_at_image - it knows how to center on a pixel
            self._enqueue_motor(
                self.mini.look_at_image,
                sdk_u,
                sdk_v,
                duration=float(duration),
                perform_movement=True,
                _position_info=position,
                _commanded_6d=commanded_6d,
            )
        else:
            # SCALE COORDINATES for non-movement case too
            sdk_scale_x = self._SDK_IMAGE_WIDTH / self._IMAGE_WIDTH
            sdk_scale_y = self._SDK_IMAGE_HEIGHT / self._IMAGE_HEIGHT
            sdk_u = int(u * sdk_scale_x)
            sdk_v = int(v * sdk_scale_y)

            # If not performing movement, just use SDK's look_at_image for calculation
            self._enqueue_motor(
                self.mini.look_at_image,
                sdk_u,
                sdk_v,
                duration=float(duration),
                perform_movement=False,
                _position_info=position,
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

        # Execute head movement
        cur_x = float(getattr(self, "x", 0.0) or 0.0)
        cur_y = float(getattr(self, "y", 0.0) or 0.0)
        cur_z = float(getattr(self, "z", 0.0) or 0.0)
        cur_roll = float(getattr(self, "roll", 0.0) or 0.0)
        cur_pitch = float(getattr(self, "pitch", 0.0) or 0.0)
        cur_yaw = float(getattr(self, "yaw", 0.0) or 0.0)

        next_x = cur_x if x is None else float(x)
        next_y = cur_y if y is None else float(y)
        next_z = cur_z if z is None else float(z)
        next_roll = cur_roll if roll is None else float(roll)
        next_pitch = cur_pitch if pitch is None else float(pitch)
        next_yaw = cur_yaw if yaw is None else float(yaw)

        max_step = getattr(self, "_head_max_step", None)
        if isinstance(max_step, dict) and max_step:
            try:
                step = float(max_step.get("x", 0.0) or 0.0)
            except Exception:
                step = 0.0
            if step > 0:
                dx = next_x - cur_x
                if abs(dx) > step:
                    next_x = cur_x + (step if dx > 0 else -step)

            try:
                step = float(max_step.get("y", 0.0) or 0.0)
            except Exception:
                step = 0.0
            if step > 0:
                dy = next_y - cur_y
                if abs(dy) > step:
                    next_y = cur_y + (step if dy > 0 else -step)

            try:
                step = float(max_step.get("z", 0.0) or 0.0)
            except Exception:
                step = 0.0
            if step > 0:
                dz = next_z - cur_z
                if abs(dz) > step:
                    next_z = cur_z + (step if dz > 0 else -step)

            try:
                step = float(max_step.get("roll", 0.0) or 0.0)
            except Exception:
                step = 0.0
            if step > 0:
                droll = next_roll - cur_roll
                if abs(droll) > step:
                    next_roll = cur_roll + (step if droll > 0 else -step)

            try:
                step = float(max_step.get("pitch", 0.0) or 0.0)
            except Exception:
                step = 0.0
            if step > 0:
                dpitch = next_pitch - cur_pitch
                if abs(dpitch) > step:
                    next_pitch = cur_pitch + (step if dpitch > 0 else -step)

            try:
                step = float(max_step.get("yaw", 0.0) or 0.0)
            except Exception:
                step = 0.0
            if step > 0:
                dyaw = next_yaw - cur_yaw
                if abs(dyaw) > step:
                    next_yaw = cur_yaw + (step if dyaw > 0 else -step)

        if (
            next_x == cur_x
            and next_y == cur_y
            and next_z == cur_z
            and next_roll == cur_roll
            and next_pitch == cur_pitch
            and next_yaw == cur_yaw
        ):
            return

        self.x = float(next_x)
        self.y = float(next_y)
        self.z = float(next_z)
        self.roll = float(next_roll)
        self.pitch = float(next_pitch)
        self.yaw = float(next_yaw)

        # Track commanded 6D pose for bounds learning
        commanded_6d = {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "y": self.y,
            "z": self.z,
            "roll": self.roll,
            "x": self.x,
        }

        self._enqueue_motor(
            move_head, self.mini,
            self.x, self.y, self.z, self.roll, self.pitch, self.yaw, self.duration,
            _commanded_6d=commanded_6d,
        )

    def move_relative(
        self,
        delta: tuple[float, float],
        duration: Optional[float] = None,
    ) -> None:
        """Move the head by a relative amount (used by DefaultNetwork scan actions).

        This uses TRANSLATION-FIRST approach: the delta primarily affects
        translation (Y for horizontal, Z for vertical), with a smaller
        component affecting rotation for personality.

        Args:
            delta: (dx, dy) tuple where dx affects horizontal and dy affects vertical.
                   Values are in "scaled units" - divide by 10 to get actual movement.
            duration: Movement duration in seconds.
        """
        if duration is None:
            duration = getattr(self, "duration", 0.5)

        dx, dy = delta

        # Convert from scaled units to actual movement amounts
        # Split between translation (primary) and rotation (secondary)
        scale = 1.0 / 10.0

        # Translation deltas (primary) - mm
        delta_y = float(dx) * scale * 2.0   # Horizontal: Y translation (2mm per unit)
        delta_z = float(dy) * scale * 2.0   # Vertical: Z translation

        # Rotation deltas (secondary) - degrees (smaller contribution)
        delta_yaw = float(dx) * scale * 0.5   # Horizontal: yaw rotation (0.5° per unit)
        delta_pitch = float(dy) * scale * 0.5  # Vertical: pitch rotation

        # Get current 6D position
        cur_x = float(getattr(self, "x", 0.0) or 0.0)
        cur_y = float(getattr(self, "y", 0.0) or 0.0)
        cur_z = float(getattr(self, "z", 0.0) or 0.0)
        cur_roll = float(getattr(self, "roll", 0.0) or 0.0)
        cur_pitch = float(getattr(self, "pitch", 0.0) or 0.0)
        cur_yaw = float(getattr(self, "yaw", 0.0) or 0.0)

        # Calculate new 6D position
        new_x = cur_x  # X (forward) unchanged
        new_y = cur_y + delta_y
        new_z = cur_z + delta_z
        new_roll = cur_roll  # Roll unchanged
        new_pitch = cur_pitch + delta_pitch
        new_yaw = cur_yaw + delta_yaw

        # Clamp to 6D workspace
        new_x, new_y, new_z, new_roll, new_pitch, new_yaw = self._clamp_to_workspace_6d(
            new_x, new_y, new_z, new_roll, new_pitch, new_yaw
        )

        self.log.debug(
            "move_relative 6D: delta=(%.1f, %.1f) -> "
            "trans(y=%.1f,z=%.1f) rot(yaw=%.1f,pitch=%.1f)",
            dx, dy, new_y, new_z, new_yaw, new_pitch
        )

        # Execute the 6D movement
        self.move(
            x=new_x, y=new_y, z=new_z,
            roll=new_roll, pitch=new_pitch, yaw=new_yaw,
            duration=duration
        )

    def turn_around(
        self,
        angle: float,
        *,
        duration: float = 5.0,
        recenter_head: bool = True,
    ) -> None:
        """Rotate the body to bring a new area into view.

        This is used when the head is at its yaw limit and there's something
        interesting beyond what the head can see. The body rotates to bring
        that area into the camera's view.

        Args:
            angle: Degrees to rotate body. Positive = counterclockwise (left),
                   negative = clockwise (right).
            duration: Time for the rotation in seconds (default 5.0).
            recenter_head: If True, return head toward center during rotation.
        """
        import math
        import numpy as np

        self.log.info(
            "turn_around: rotating body %.0f° over %.1fs (recenter_head=%s)",
            angle, duration, recenter_head
        )

        # Clear any pending movements from the queue
        # This prevents old look_at_image commands from executing after the turn
        cleared = self._clear_motor_queue()
        if cleared > 0:
            self.log.debug("Cleared %d queued movements before turn_around", cleared)

        # Inhibit DefaultNetwork to prevent new tracking commands during turn
        default_network = getattr(self, "_default_network", None)
        if default_network is not None:
            try:
                # Inhibit for the duration of the turn + buffer
                default_network.inhibit(duration=duration + 2.0)
                self.log.debug("Inhibited DefaultNetwork for %.1fs during turn_around", duration + 2.0)
            except Exception as inh_e:
                self.log.debug("Failed to inhibit DefaultNetwork: %s", inh_e)

        import time as time_mod
        from maxim.motion.movement import head_pose_matrix

        try:
            # Get ACTUAL current body yaw from SDK
            try:
                head_joints, _ = self.mini.get_current_joint_positions()
                if len(head_joints) >= 7:
                    current_body_yaw = math.degrees(head_joints[6])
                else:
                    current_body_yaw = float(getattr(self, "body_yaw", 0.0) or 0.0)
            except Exception:
                current_body_yaw = float(getattr(self, "body_yaw", 0.0) or 0.0)

            target_body_yaw = current_body_yaw + angle
            target_body_yaw = max(-160.0, min(160.0, target_body_yaw))

            self.log.info(
                "turn_around: 3-step sequence starting (body %.1f -> %.1f)",
                current_body_yaw, target_body_yaw
            )

            # ═══════════════════════════════════════════════════════════════════
            # STEP 1: Return head to center first (prevents IK issues during turn)
            # ═══════════════════════════════════════════════════════════════════
            step1_duration = min(2.0, duration * 0.3)
            center_pose = head_pose_matrix(0, 0, 0, 0, 0, 0)
            
            self.log.info("turn_around STEP 1: centering head (%.1fs)", step1_duration)
            try:
                self.mini.goto_target(
                    head=center_pose,
                    duration=step1_duration,
                    method="minjerk",
                )
                time_mod.sleep(step1_duration + 0.3)  # Wait for completion + buffer
            except Exception as e1:
                self.log.warning("turn_around STEP 1 failed: %s", e1)

            # ═══════════════════════════════════════════════════════════════════
            # STEP 2: Rotate body (head stays centered relative to body)
            # ═══════════════════════════════════════════════════════════════════
            step2_duration = max(3.0, duration * 0.5)
            body_yaw_rad = np.deg2rad(target_body_yaw)
            
            self.log.info("turn_around STEP 2: rotating body to %.1f° (%.1fs)", target_body_yaw, step2_duration)
            try:
                self.mini.goto_target(
                    body_yaw=body_yaw_rad,
                    duration=step2_duration,
                    method="minjerk",
                )
                time_mod.sleep(step2_duration + 0.3)  # Wait for completion + buffer
            except Exception as e2:
                self.log.warning("turn_around STEP 2 failed: %s", e2)

            # ═══════════════════════════════════════════════════════════════════
            # STEP 3: Settle and sync position
            # ═══════════════════════════════════════════════════════════════════
            self.log.info("turn_around STEP 3: settling and syncing")
            time_mod.sleep(0.5)  # Let everything settle
            
            # Sync with hardware to get accurate position
            self.sync_head_position()

            # Update internal tracking
            self.body_yaw = target_body_yaw
            if recenter_head:
                self.yaw = 0.0
                self.pitch = 0.0
                self.y = 0.0
                self.z = 0.0

            self.log.info(
                "turn_around complete: body_yaw %.1f -> %.1f (total %.1fs)",
                current_body_yaw, target_body_yaw, step1_duration + step2_duration + 0.8
            )

        except Exception as e:
            self.log.warning("turn_around failed: %s", e)
            import traceback
            self.log.debug("turn_around traceback: %s", traceback.format_exc())

        finally:
            # Release DefaultNetwork inhibition
            if default_network is not None:
                try:
                    default_network.release()
                    self.log.debug("Released DefaultNetwork inhibition after turn_around")
                except Exception:
                    pass

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

    def _ensure_segmenter(self, *, force: bool = False, model_name: str | None = None) -> None:
        if not force and getattr(self, "segmenter", None) is not None:
            return

        seg_model = str(
            model_name or os.getenv("MAXIM_SEGMENTATION_MODEL", "rtm") or "rtm"
        ).strip() or "rtm"
        self.log.info("Loading vision models (%s seg+pose)...", seg_model)
        # Preflight matplotlib font cache in a subprocess to avoid hard crashes on Linux/WSL.
        preflight_ok = preflight_matplotlib_fonts(
            cache_dir=os.path.join(self.home_dir, "matplotlib"),
            logger=self.log,
        )
        if not preflight_ok:
            raise RuntimeError("Matplotlib font preflight failed; see README troubleshooting.")
        try:
            self.segmenter = build_segmentation_model(seg_model, pose_model=True)  # Visual segmentation + pose model
            self._segmenter_model = seg_model
        except Exception as e:
            warn("Failed to load segmentation model '%s': %s (falling back to rtm)", seg_model, e, logger=self.log)
            self.segmenter = build_segmentation_model("rtm", pose_model=True)
            self._segmenter_model = "rtm"
    
    def awaken(self, vision: bool = True, motor: bool = True, audio: bool = True, wake_up: bool = True):
        if wake_up:
            # Wake up Reachy before model init to avoid loading while asleep.
            self.log.info("Waking up Reachy...")

            # On Blackwell GPUs, skip wake_up() entirely due to GStreamer/GLib crash
            # The crash happens in native GLib.MainLoop.run() code in the WebRTC thread
            # Bug report: https://github.com/pollen-robotics/reachy_mini/issues
            if _blackwell_detected and not self._simulation:
                self.log.warning(
                    "Blackwell GPU detected - skipping wake_up() due to GStreamer incompatibility. "
                    "Robot should already be in usable state. See: https://github.com/pollen-robotics/reachy_mini/issues"
                )
                # Don't call wake_up() - the robot is already connected and motors are accessible
            elif self._robot is not None:
                self._robot.wake_up()
            self.sleeping = False  # Mark as awake so next sleep() will move to sleep pose
            self._woke_up = True

            # On Blackwell GPUs, start recording AFTER wake_up() to avoid
            # GStreamer/WebRTC threading issues that cause segfaults
            if _blackwell_detected and not self._simulation and not getattr(self, '_recording_started', False):
                time.sleep(0.3)  # Brief pause to let motor commands complete
                try:
                    if self._robot is not None:
                        self._robot.start_recording()
                    self._recording_started = True
                    self.log.info("Recording started (delayed for Blackwell GPU)")
                except Exception as e:
                    self.log.warning("Failed to start recording: %s", e)

        # Load models
        if vision:
            self._ensure_segmenter()

        if motor and self.movement_model is None:
            try:
                from maxim.models.movement.motor_cortex import LayerScale, MotorCortex
                from maxim.utils import config as motor_config
                import tensorflow as tf

                self.log.info("Initializing motor cortex...")

                cfg = motor_config.build(motor_config.DEFAULT_SAVE_ROOT)

                # Try GPU first, fallback to CPU if JIT compilation fails (e.g., RTX 5080/Blackwell)
                try:
                    self.movement_model = MotorCortex(cfg)
                    self.log.info("Motor cortex initialized on GPU")
                except (RuntimeError, tf.errors.InternalError) as gpu_err:
                    self.log.warning(f"GPU initialization failed ({type(gpu_err).__name__}), falling back to CPU mode")
                    self.log.warning("This is expected on RTX 5080/Blackwell GPUs with current TensorFlow")

                    # Force CPU mode
                    with tf.device('/CPU:0'):
                        self.movement_model = MotorCortex(cfg)
                    self.log.info("Motor cortex initialized on CPU")

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

        return

    # ─────────────────────────────────────────────────────────────────────────
    # Thread Registry for Coordinated Shutdown
    # ─────────────────────────────────────────────────────────────────────────

    def register_thread(self, name: str, thread: threading.Thread) -> None:
        """Register a thread for coordinated shutdown tracking."""
        self._thread_registry.register(name, thread)

    def unregister_thread(self, name: str) -> None:
        """Remove a thread from the registry."""
        self._thread_registry.unregister(name)

    def stop_all_threads(self, timeout: float = 10.0) -> list[str]:
        """Stop all registered threads with force-kill fallback."""
        return self._thread_registry.stop_all(timeout=timeout)

    def shutdown(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True

        # Stop all registered threads first
        failed = self.stop_all_threads(timeout=5.0)
        if failed:
            self.log.warning("Some threads did not stop: %s", failed)

        self._stop_agentic_runtime(timeout=10.0)

        try:
            training_logger = getattr(self, "_training_logger", None)
            if training_logger is not None:
                training_logger.stop(timeout=2.0)
        except Exception:
            pass
        self._training_logger = None
        try:
            cli_logger = getattr(self, "_cli_logger", None)
            if cli_logger is not None:
                cli_logger.stop(timeout=2.0)
        except Exception:
            pass
        self._cli_logger = None

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
            requested = str(getattr(self, "requested_mode", "") or "").strip().lower()
            if requested not in ("reflection", "exploration", "live", "train", "agentic"):
                try:
                    if self._robot is not None:
                        self._robot.goto_sleep()
                except Exception as e:
                    warn("Failed to send Reachy to sleep: %s", e, logger=getattr(self, "log", None))

        # Stop recording data
        try:
            if self._robot is not None:
                self._robot.stop_recording()
        except Exception as e:
            warn("Failed to stop recording: %s", e, logger=getattr(self, "log", None))

        # Release the camera + any OpenCV resources
        self._release_media()

        # Disconnect from robot via registry (handles cleanup)
        try:
            if self._robot is not None:
                robot_id = getattr(self._robot, "robot_id", None)
                if robot_id:
                    _robot_registry.disconnect_robot(robot_id)
                self._robot = None
        except Exception as e:
            warn("Failed to disconnect robot: %s", e, logger=getattr(self, "log", None))


        return

if __name__ == "__main__":
    conscience = Maxim()
    
