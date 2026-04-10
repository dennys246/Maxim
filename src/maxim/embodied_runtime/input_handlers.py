"""Input handling mixin for the Maxim conscience loop.

Provides keyboard, CLI, and voice-transcript listener threads plus the
action-spec dispatch and outcome-labelling helpers.  Mixed into the main
Maxim class so that ``self.`` references resolve against the full instance.
"""

from __future__ import annotations

import json
import threading
import time
import uuid

from maxim.utils.response_config import normalize_trigger_text, normalize_transcript_text
from maxim.utils.logging import warn


class InputHandlerMixin:
    """Methods extracted from selfy.py that handle external input streams."""

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

            if (
                not is_cli
                and bool(spec.get("requires_agentic", False))
                and not bool(getattr(self, "_voice_agentic_enabled", False))
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
                if (
                    not is_cli
                    and bool(spec.get("requires_agentic", False))
                    and not bool(getattr(self, "_voice_agentic_enabled", False))
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

    # --- Thread-safe phrase registration (R5) ---

    def _register_phrase_response(self, phrase: str, spec: dict) -> None:
        """Thread-safe runtime addition of a phrase response.

        Uses copy-on-write: creates a new dict and atomically swaps the
        reference (GIL-safe). The transcript listener thread's in-progress
        iteration continues on the old dict; next iteration picks up the new one.
        """
        from maxim.utils.response_config import _compile_phrase_pattern, normalize_trigger_text

        # Pre-compile pattern and normalized form
        if "_pattern" not in spec:
            spec["_pattern"] = _compile_phrase_pattern(phrase)
        if "_normalized" not in spec:
            spec["_normalized"] = normalize_trigger_text(phrase)

        # Atomic copy-on-write swap
        new = dict(self.phrase_responses)
        new[phrase] = spec
        self.phrase_responses = new  # atomic reference swap under GIL
