import os
import queue
import threading

#os.environ["REACHY_MEDIA_BACKEND"] = "zenoh"
#os.environ["REACHY_DISABLE_WEBRTC"] = "1"
#os.environ["GST_DISABLE_REGISTRY_FORK"] = "1"

import json, random, uuid
import re
import time, atexit, cv2
import logging
import multiprocessing as mp
import wave
from typing import Any, Optional

import numpy as np

from maxim.motion.movement import load_actions, load_movement_thresholds, load_poses, move_antenna, move_head
from maxim.utils.audio import resample_audio, to_int16
from maxim.utils.data_management import CLIInputLogger, TrainingSampleLogger, VisionEventLogger, build_home
from maxim.utils.logging import configure_logging, warn
from maxim.utils.plotting import preflight_matplotlib_fonts, preload_matplotlib_fonts
from maxim.utils.queueing import put_latest

from maxim.data.camera.display import prepare_display, show_photo
from maxim.inference.observation import (
    display_detections,
    face_observation,
    passive_observation,
    passive_listening,
)
from maxim.models.vision.registry import build_segmentation_model

os.environ["PYOPENGL_PLATFORM"] = "egl"

def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    return bool(default)


def _gpu_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        if torch.cuda.is_available():
            return True
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and getattr(mps, "is_available", None):
            return bool(mps.is_available())
    except Exception:
        return False
    return False

def _is_connection_error(error: object) -> bool:
    if error is None:
        return False
    message = str(error).strip().lower()
    if not message:
        return False
    if "lost connection" in message:
        return True
    if "disconnected" in message:
        return True
    if "timeout" in message or "timed out" in message:
        return True
    if "connection" in message and any(term in message for term in ("refused", "reset", "broken", "closed")):
        return True
    # Dynamixel/serial communication errors (rustypot panics)
    if "channel closed" in message:
        return True
    if "panicexception" in message:
        return True
    if "assertion failed" in message and "buffer" in message:
        return True
    if "flush serial" in message:
        return True
    return False

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
        interactive: bool = True):
        
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

        self.name = robot_name or os.getenv("MAXIM_ROBOT_NAME", "reachy_mini")
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
        except Exception:
            self.audio_len = 5.0
        if self.audio_len <= 0:
            self.audio_len = 5.0

        self.video_fps = 20.0

        self.interactive = bool(interactive)

        self.interests = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        self.actions = load_actions()
        self.poses = load_poses()
        self.movement_thresholds = load_movement_thresholds()
        self._head_max_step = {}
        try:
            head_cfg = self.movement_thresholds.get("head") if isinstance(self.movement_thresholds, dict) else None
            if isinstance(head_cfg, dict) and isinstance(head_cfg.get("max_step"), dict):
                self._head_max_step = dict(head_cfg.get("max_step") or {})
        except Exception:
            self._head_max_step = {}

        # robot_name must match the daemon namespace (default: reachy_mini).
        # localhost_only=False enables zenoh peer discovery across the LAN.
        # Import ReachyMini after Matplotlib preload to avoid native lib conflicts.
        from reachy_mini import ReachyMini

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
        self.phrase_responses = self._load_phrase_responses()
        self._voice_agentic_enabled = False
        self._phrase_last_trigger_ts: dict[str, float] = {}
        self._outcome_code = 0
        self._last_action_event_id: str | None = None
        self._last_transcript_event: dict | None = None
        self.requested_mode: str | None = None
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

    def _load_key_responses(self) -> dict[str, dict]:
        default = {
            "c": {"call": "center_vision", "pause_training": True},
            "u": {"call": "mark_trainable_moment"},
            **{str(i): {"call": "label_outcome", "args": [i]} for i in range(10)},
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

        merged = dict(default)
        merged.update(parsed)
        return merged

    def _load_phrase_responses(self) -> dict[str, dict]:
        default = {
            # Shutdown commands
            "maxim shutdown": {"call": "request_shutdown", "requires_agentic": False, "cooldown_s": 2.0},
            "shutdown maxim": {"call": "request_shutdown", "requires_agentic": False, "cooldown_s": 2.0},
            # Sleep mode commands
            "maxim sleep": {"call": "request_sleep", "requires_agentic": False, "cooldown_s": 2.0},
            "sleep maxim": {"call": "request_sleep", "requires_agentic": False, "cooldown_s": 2.0},
            "maxim nap": {"call": "request_sleep", "requires_agentic": False, "cooldown_s": 2.0},
            "maxim rest": {"call": "request_sleep", "requires_agentic": False, "cooldown_s": 2.0},
            # Passive-interaction mode commands (wake from sleep)
            "maxim observe": {"call": "request_observe", "requires_agentic": False, "cooldown_s": 2.0},
            "observe maxim": {"call": "request_observe", "requires_agentic": False, "cooldown_s": 2.0},
            "maxim watch": {"call": "request_observe", "requires_agentic": False, "cooldown_s": 2.0},
            "maxim wake": {"call": "request_observe", "requires_agentic": False, "cooldown_s": 2.0},
            "wake maxim": {"call": "request_observe", "requires_agentic": False, "cooldown_s": 2.0},
            "maxim wake up": {"call": "request_observe", "requires_agentic": False, "cooldown_s": 2.0},
            "wake up maxim": {"call": "request_observe", "requires_agentic": False, "cooldown_s": 2.0},
            "maxim passive": {"call": "request_observe", "requires_agentic": False, "cooldown_s": 2.0},
            "maxim reflection": {"call": "request_observe", "requires_agentic": False, "cooldown_s": 2.0},
            # Wake words (enable agentic mode)
            "maxim": {"call": "wake_up_agentic", "wake_word": True, "cooldown_s": 2.0},
            "reachy": {"call": "wake_up_agentic", "wake_word": True, "cooldown_s": 2.0},
            # Other commands
            "center": {"call": "center_vision", "pause_training": True, "requires_agentic": True, "cooldown_s": 2.0},
        }

        candidates: list[str] = []
        env_path = str(os.getenv("MAXIM_PHRASE_RESPONSES", "")).strip()
        if env_path:
            candidates.append(env_path)
        candidates.append(os.path.join(os.getcwd(), "data", "util", "phrase_responses.json"))
        candidates.append(os.path.join(os.getcwd(), "phrase_responses.json"))
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            candidates.append(os.path.join(repo_root, "data", "util", "phrase_responses.json"))
            candidates.append(os.path.join(repo_root, "phrase_responses.json"))
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
                    warn("Failed to load phrase responses from '%s': %s", path, e, logger=self.log)
                    return default

        if not isinstance(raw, dict):
            return default

        parsed: dict[str, dict] = {}
        for phrase, spec in raw.items():
            if not isinstance(phrase, str) or not phrase.strip():
                continue
            phrase = phrase.strip()

            if isinstance(spec, str):
                spec = {"call": spec}
            if not isinstance(spec, dict):
                continue

            call = spec.get("call") or spec.get("method")
            if not isinstance(call, str) or not call:
                continue

            wake_word = bool(spec.get("wake_word", False))
            requires_agentic = bool(spec.get("requires_agentic", not wake_word))
            cooldown_s = spec.get("cooldown_s")
            try:
                cooldown_s = float(cooldown_s) if cooldown_s is not None else 2.0
            except Exception:
                cooldown_s = 2.0
            if float(cooldown_s) <= 0:
                cooldown_s = 2.0

            parsed[phrase] = {
                "call": call,
                "args": spec.get("args") if isinstance(spec.get("args"), list) else [],
                "kwargs": spec.get("kwargs") if isinstance(spec.get("kwargs"), dict) else {},
                "pause_training": bool(spec.get("pause_training", False)),
                "wake_word": wake_word,
                "requires_agentic": requires_agentic,
                "cooldown_s": float(cooldown_s),
                "_pattern": self._compile_phrase_pattern(phrase),
                "_normalized": self._normalize_trigger_text(phrase),
            }

        merged = dict(default)
        merged.update(parsed)
        # Compile patterns for any defaults that weren't overridden.
        for phrase, spec in merged.items():
            if isinstance(spec, dict) and "_pattern" not in spec:
                spec["_pattern"] = self._compile_phrase_pattern(phrase)
            if isinstance(spec, dict) and "_normalized" not in spec:
                spec["_normalized"] = self._normalize_trigger_text(phrase)
        return merged

    def _compile_phrase_pattern(self, phrase: str):
        raw = str(phrase or "").strip()
        if not raw:
            return None
        escaped = re.escape(raw)
        pattern = escaped
        try:
            if re.match(r"^\w", raw) and re.search(r"\w$", raw):
                pattern = rf"\b{escaped}\b"
            return re.compile(pattern, flags=re.IGNORECASE)
        except Exception:
            return None

    def _normalize_trigger_text(self, text: str) -> str:
        raw = str(text or "").strip().lower()
        if not raw:
            return ""
        cleaned = re.sub(r"[^\w\s]", " ", raw, flags=re.UNICODE)
        return " ".join(cleaned.split())

    def _normalize_transcript_text(self, text: str) -> str:
        normalized = self._normalize_trigger_text(text)
        if not normalized:
            return ""
        raw_tokens = normalized.split()
        tokens = [t for t in raw_tokens if t and t != "s"]
        aliases = {
            "maximum": "maxim",
            "maximums": "maxim",
            "maxims": "maxim",
        }
        changed = tokens != raw_tokens
        for idx, token in enumerate(tokens):
            replacement = aliases.get(token)
            if replacement:
                tokens[idx] = replacement
                changed = True
        return " ".join(tokens) if changed else normalized

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
        normalized_text = self._normalize_transcript_text(raw_text)
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
                    normalized_phrase = self._normalize_trigger_text(phrase)
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
                    normalized_phrase = self._normalize_trigger_text(phrase)
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
        if not _is_connection_error(error):
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

            try:
                from reachy_mini import ReachyMini
            except Exception as e:
                warn("Soft reconnect failed (reachy_mini unavailable): %s", e, logger=self.log)
                return False

            try:
                new_mini = ReachyMini(**dict(getattr(self, "_connect_kwargs", {}) or {}))
            except Exception as e:
                warn("Soft reconnect failed (connect): %s", e, logger=self.log)
                return False

            self.mini = new_mini

            try:
                new_mini.start_recording()
            except Exception as e:
                warn("Failed to start recording after reconnect: %s", e, logger=self.log)

            if getattr(self, "_woke_up", False):
                mode = str(getattr(self, "mode", "") or "").strip().lower()
                if mode != "sleep":
                    try:
                        new_mini.wake_up()
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
                    observations = segmenter.segment_photos(
                        frame,
                        interests=list(getattr(self, "interests", []) or []),
                    )
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
        if not _gpu_available():
            import os
            cuda_hidden = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            if cuda_hidden:
                self.log.info("GPU hidden for compatibility - agentic runtime will use CPU (slower)")
            else:
                self.log.warning("No GPU available - agentic runtime will use CPU (slower)")

            # Configure CPU-friendly model defaults
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
        if os.environ.get("MAXIM_TTS_ENABLED", "").lower() in ("1", "true", "yes"):
            try:
                from maxim.models.audio.tts import TTSEngine

                tts_model = os.environ.get("MAXIM_TTS_MODEL", "en_US-lessac-medium")
                tts_engine = TTSEngine(model_name=tts_model)
                if tts_engine.is_available:
                    speaker_fn = self.speak  # Use Maxim's speak method
                    self.log.info("TTS enabled with model: %s", tts_model)
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

        registry = build_tool_registry(maxim=self, response_output=response_output)
        executor = build_executor(registry)
        evaluators = build_evaluators()

        run_id = getattr(self, "run_id", None) or time.strftime("%Y-%m-%d_%H%M%S")

        # Set up autonomy controller with sensible defaults for live mode
        supervision_policy = SupervisionPolicy(
            allowed_tools={
                "read_file",
                "focus_interests",
                "track_target",
                "maxim_command",
                "mode_switch",
                "speak",
                "respond",
            },
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
                llm_worker = LLMWorker(llm=llm_router, stale_threshold_s=5.0)
                llm_worker.start()
                self.log.info("LLM worker started for user responses")
            else:
                self.log.debug("LLM disabled in config, responses will use fallback")
        except Exception as e:
            warn("Failed to create LLM worker: %s", e, logger=self.log)
            llm_worker = None

        self._llm_worker = llm_worker

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

    def _stop_agentic_runtime(self, *, timeout: float = 2.0) -> None:
        try:
            ev = getattr(self, "_agentic_stop_event", None)
            if ev is not None:
                ev.set()
        except Exception:
            pass
        t = getattr(self, "_agentic_thread", None)
        if t is not None:
            try:
                t.join(timeout=float(timeout))
            except Exception:
                pass
        self._agentic_thread = None
        self._agentic_stop_event = None
        self._agentic_agent = None
        self._agentic_state = None

        # Stop capture manager (Phase 3)
        capture_manager = getattr(self, "_capture_manager", None)
        if capture_manager is not None:
            try:
                capture_manager.stop(timeout=timeout)
            except Exception:
                pass
            self._capture_manager = None

        self._stop_vision_event_stream(timeout=timeout)

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

        try:
            prev_cli_logger = getattr(self, "_cli_logger", None)
            if prev_cli_logger is not None:
                prev_cli_logger.stop(timeout=0.5)
        except Exception:
            pass
        self._cli_logger = None

        try:
            prev_vision_logger = getattr(self, "_vision_event_logger", None)
            if prev_vision_logger is not None:
                self._stop_vision_event_stream(timeout=0.5)
        except Exception:
            pass
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

        transcribe_process = None
        transcribe_shutdown_file = None
        if self.audio and parallel:
            os.makedirs(chunk_dir, exist_ok=True)
            try:
                from maxim.data.audio._file_based_transcription import watch_and_transcribe

                # Use file-based IPC instead of multiprocessing Queues
                # Queues use shared memory which conflicts with TensorFlow+CUDA in parent
                # File watching completely isolates parent and child processes
                # See: https://github.com/tensorflow/tensorflow/issues/8220
                #      https://github.com/OpenNMT/CTranslate2/issues/1693
                ctx = mp.get_context("spawn")
                if self.log:
                    self.log.debug("Using file-based IPC (no shared memory)")

                vad_filter = _env_flag("MAXIM_VAD_FILTER", True)
                compute_type = str(os.getenv("MAXIM_WHISPER_COMPUTE_TYPE", "int8") or "int8").strip()
                if not compute_type:
                    compute_type = "int8"

                # Auto-detect Blackwell GPUs (RTX 50 series) and force CPU + float32 for Whisper
                # CTranslate2 has critical compatibility issues with sm_120 (Blackwell) architecture:
                # - All int8 compute types fail with CUBLAS_STATUS_NOT_SUPPORTED (even in CPU mode)
                # - CUDA libraries are loaded regardless of device="cpu" setting
                # See: https://github.com/OpenNMT/CTranslate2/issues/1865
                #      https://github.com/SYSTRAN/faster-whisper/issues/1260
                default_whisper_device = "cuda"
                blackwell_detected = False

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
                            default_whisper_device = "cpu"
                            blackwell_detected = True
                            self.log.warning("⚠️  Detected Blackwell GPU (nvidia-smi)")
                            self.log.warning("   CTranslate2 int8 incompatible with RTX 50 series - forcing CPU + float32")
                except Exception as e:
                    self.log.debug(f"nvidia-smi check failed: {e}")

                whisper_device = str(os.getenv("MAXIM_WHISPER_DEVICE", default_whisper_device) or default_whisper_device).strip()

                # Force float32 for Blackwell GPUs (int8 causes segfaults even in CPU mode)
                if blackwell_detected and "int8" in compute_type.lower():
                    compute_type = "float32"
                    self.log.info("   Compute type auto-changed: int8 → float32 (required for Blackwell)")

                self.log.info("Transcription VAD filter: %s", "enabled" if vad_filter else "disabled")
                self.log.info("Whisper compute type: %s", compute_type)
                self.log.info("Whisper device: %s (will fallback to CPU if unavailable)", whisper_device)

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
                        "model_size_or_path": "tiny",
                        "device": whisper_device,
                        "compute_type": compute_type,
                        "language": "en",
                        "beam_size": 1,
                        "vad_filter": vad_filter,
                        "cleanup_chunks": True,
                        "verbosity": int(self.verbosity or 0),
                        "log_file": log_path,
                        "shutdown_file": transcribe_shutdown_file,
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
                    self._note_connection_failure("motor", e)

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
                try:
                    transcribe_process.join(timeout=5.0)
                except Exception:
                    pass
                try:
                    if transcribe_process.is_alive():
                        transcribe_process.terminate()
                        transcribe_process.join(timeout=2.0)
                except Exception:
                    pass
                try:
                    if transcribe_process.is_alive() and hasattr(transcribe_process, "kill"):
                        transcribe_process.kill()
                        transcribe_process.join(timeout=2.0)
                except Exception:
                    pass

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
            self.mini.goto_sleep()
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
        ev = getattr(self, "_live_stop_event", None)
        if ev is not None:
            try:
                if int(getattr(self, "verbosity", 0) or 0) >= 2:
                    self.log.debug("Stopping live loop for mode switch -> %s", requested)
                ev.set()
            except Exception:
                pass

    def request_shutdown(self) -> None:
        self._request_mode("shutdown")

    def request_sleep(self) -> None:
        self._request_mode("sleep")

    def request_observe(self) -> None:
        self._request_mode("reflection")

    def update_interests(
        self,
        add: list[int] | None = None,
        remove: list[int] | None = None,
    ) -> None:
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

        self._enqueue_motor(move_head, self.mini, self.x, self.y, self.z, self.roll, self.pitch, self.yaw, self.duration)

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

    def _ensure_segmenter(self, *, force: bool = False, model_name: str | None = None) -> None:
        if not force and getattr(self, "segmenter", None) is not None:
            return

        seg_model = str(
            model_name or os.getenv("MAXIM_SEGMENTATION_MODEL", "YOLO8") or "YOLO8"
        ).strip() or "YOLO8"
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
            warn("Failed to load segmentation model '%s': %s (falling back to YOLO8)", seg_model, e, logger=self.log)
            self.segmenter = build_segmentation_model("YOLO8", pose_model=True)
            self._segmenter_model = "YOLO8"
    
    def awaken(self, vision: bool = True, motor: bool = True, audio: bool = True, wake_up: bool = True):
        if wake_up:
            # Wake up Reachy before model init to avoid loading while asleep.
            self.log.info("Waking up Reachy...")
            self.mini.wake_up()
            self.sleeping = False  # Mark as awake so next sleep() will move to sleep pose
            self._woke_up = True

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

    def shutdown(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._stop_agentic_runtime(timeout=2.0)

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

        # Best-effort: close any lingering connections.
        try:
            mini = getattr(self, "mini", None)
            if mini is not None:
                for attr in ("disconnect", "close", "shutdown"):
                    fn = getattr(mini, attr, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass
        except Exception:
            pass


        return

if __name__ == "__main__":
    conscience = Maxim()
    
