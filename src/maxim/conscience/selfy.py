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
from typing import Optional

import numpy as np

from maxim.motion.movement import load_actions, load_movement_thresholds, load_poses, move_antenna, move_head
from maxim.utils.audio import resample_audio, to_int16
from maxim.utils.data_management import TrainingSampleLogger, build_home
from maxim.utils.logging import configure_logging, warn
from maxim.utils.plotting import preflight_matplotlib_fonts, preload_matplotlib_fonts
from maxim.utils.queueing import put_latest

from maxim.data.camera.display import prepare_display, show_photo
from maxim.inference.observation import (
    face_observation,
    motor_cortex_control,
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

        # Load Matplotlib before Reachy/GStreamer so ft2font binds to stable libs.
        preload_matplotlib_fonts(
            cache_dir=os.path.join(self.home_dir, "matplotlib"),
            logger=self.log,
        )

        self.current_epoch = 0
        self._set_epochs(epochs)
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
            "maxim shutdown": {"call": "request_shutdown", "requires_agentic": False, "cooldown_s": 2.0},
            "maxim sleep": {"call": "request_sleep", "requires_agentic": False, "cooldown_s": 2.0},
            "maxim observe": {"call": "request_observe", "requires_agentic": False, "cooldown_s": 2.0},
            "sleep maxim": {"call": "request_sleep", "requires_agentic": False, "cooldown_s": 2.0},
            "observe maxim": {"call": "request_observe", "requires_agentic": False, "cooldown_s": 2.0},
            "maxim": {"call": "wake_up_agentic", "wake_word": True, "cooldown_s": 2.0},
            "reachy": {"call": "wake_up_agentic", "wake_word": True, "cooldown_s": 2.0},
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

    def _handle_transcript_record(self, record: dict) -> None:
        text = str(record.get("text", "") or "").strip()
        if not text:
            return
        normalized_text = self._normalize_transcript_text(text)
        if not normalized_text:
            return
        try:
            self._last_transcript_event = record
        except Exception:
            pass

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
                    matched = bool(pattern.search(text))
                else:
                    matched = phrase.lower() in text.lower()
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

            if bool(spec.get("requires_agentic", False)) and not bool(getattr(self, "_voice_agentic_enabled", False)):
                continue

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
                if bool(spec.get("requires_agentic", False)) and not bool(getattr(self, "_voice_agentic_enabled", False)):
                    continue

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
        try:
            self._phrase_last_trigger_ts[phrase] = now
        except Exception:
            pass
        self._run_action_spec(source="voice", trigger=phrase, spec=spec, transcript=record)

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

    def _repo_root(self) -> str:
        try:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        except Exception:
            return os.getcwd()

    def _start_agentic_runtime(self) -> None:
        existing = getattr(self, "_agentic_thread", None)
        if existing is not None and getattr(existing, "is_alive", lambda: False)():
            return

        try:
            from maxim.agents import ReachyAgent
            from maxim.environment import ReachyEnv
            from maxim.runtime import (
                build_decision_engine,
                build_evaluators,
                build_executor,
                build_memory,
                build_state,
                build_tool_registry,
                run_agent_loop,
            )
        except Exception as e:
            warn("Agentic runtime unavailable: %s", e, logger=self.log)
            return

        stop_event = threading.Event()
        self._agentic_stop_event = stop_event

        agent = ReachyAgent()
        env = ReachyEnv(repo_root=os.getcwd(), data_dir=str(getattr(self, "home_dir", "data") or "data"))
        state = build_state(max_steps=1_000_000)
        try:
            state.data["maxim_runtime"] = {
                "mode": getattr(self, "mode", None),
                "interests": list(getattr(self, "interests", []) or []),
            }
        except Exception:
            pass
        memory = build_memory()
        decision_engine = build_decision_engine()
        registry = build_tool_registry(maxim=self)
        executor = build_executor(registry)
        evaluators = build_evaluators()

        run_id = getattr(self, "run_id", None) or time.strftime("%Y-%m-%d_%H%M%S")

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
                run_agent_loop(
                    agent,
                    env,
                    state,
                    memory,
                    decision_engine,
                    executor,
                    evaluators=evaluators,
                    max_steps=1_000_000,
                    run_id=run_id,
                    stop_event=stop_event,
                    on_step=_on_step,
                    break_on_no_intent=False,
                    idle_sleep_s=0.25,
                )
            except Exception as e:
                warn("Agentic runtime loop failed: %s", e, logger=self.log)

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

        epochs_label = "unlimited" if self.epochs is None else str(int(self.epochs))
        self.log.info(
            "Starting live loop (home_dir=%s, epochs=%s, observation_period=%s, mode=%s, audio=%s, audio_len=%.1fs)",
            self.home_dir,
            epochs_label,
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
        if vision and self.verbose:
            # Keep OpenCV GUI calls on a dedicated process main thread (safer on Linux/WSL).
            prepare_display()

        media_lock = threading.Lock()
        stop_event = threading.Event()
        self._live_stop_event = stop_event

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
                vad_filter = _env_flag("MAXIM_VAD_FILTER", True)
                compute_type = str(os.getenv("MAXIM_WHISPER_COMPUTE_TYPE", "int8") or "int8").strip()
                if not compute_type:
                    compute_type = "int8"
                self.log.info("Transcription VAD filter: %s", "enabled" if vad_filter else "disabled")
                self.log.info("Whisper compute type: %s", compute_type)
                transcribe_queue = ctx.Queue(maxsize=64)
                transcribe_process = ctx.Process(
                    target=transcription_worker,
                    args=(transcribe_queue, transcript_path),
                    kwargs={
                        "model_size_or_path": "tiny",
                        "device": "cpu",
                        "compute_type": compute_type,
                        "language": "en",
                        "beam_size": 1,
                        "vad_filter": vad_filter,
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
                if t is key_thread or t is transcript_thread:
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

            if transcribe_queue is not None:
                try:
                    transcribe_queue.put_nowait(None)
                except Exception:
                    try:
                        transcribe_queue.put(None, timeout=0.5)
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

            if transcribe_queue is not None:
                try:
                    transcribe_queue.close()
                except Exception:
                    pass
                try:
                    transcribe_queue.join_thread()
                except Exception:
                    pass

            self._motor_queue = None
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
        """
        self.audio = True
        self.mini.goto_sleep()
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
                ev.set()
            except Exception:
                pass

    def request_shutdown(self) -> None:
        self._request_mode("shutdown")

    def request_sleep(self) -> None:
        self._request_mode("sleep")

    def request_observe(self) -> None:
        self._request_mode("passive-interaction")

    def wake_up_agentic(self) -> None:
        try:
            self._voice_agentic_enabled = True
        except Exception:
            pass

        try:
            mini = getattr(self, "mini", None)
            if mini is not None:
                self._enqueue_motor(mini.wake_up)
                self._woke_up = True
        except Exception as e:
            warn("Failed to wake up Reachy: %s", e, logger=self.log)

        self._start_agentic_runtime()

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
            self._woke_up = True

        # Load models
        if vision:
            self._ensure_segmenter()

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
            if requested not in ("passive-interaction", "live", "train", "agentic"):
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
    
