from __future__ import annotations

import os
import re
import time
from typing import Any

from maxim.agents.base import Agent
from maxim.models.language import LLMRouter


class ReachyAgent(Agent):
    """
    Default agent for working with Reachy Mini artifacts under `data/`.

    This agent is intentionally simple: it proposes high-level goals and relies on
    the planner + decision engine + tools to execute.
    """

    agent_name = "reachy_mini"

    def __init__(self, *, name: str | None = None, enabled: bool = True) -> None:
        super().__init__(name=name, enabled=enabled)
        self._last_transcript_path: str | None = None
        self._last_transcript_mtime: float | None = None
        self._did_read_readme = False
        self._last_focus_ts = 0.0
        self._last_llm_chunk_index: int | None = None
        self._llm: LLMRouter | None = None

    @staticmethod
    def _normalize_transcript_text(text: str) -> str:
        raw = str(text or "").strip().lower()
        if not raw:
            return ""
        cleaned = re.sub(r"[^\w\s]", " ", raw, flags=re.UNICODE)
        normalized = " ".join(cleaned.split())
        if not normalized:
            return ""
        tokens = [t for t in normalized.split() if t and t != "s"]
        aliases = {"maximum": "maxim", "maximums": "maxim", "maxims": "maxim"}
        for idx, token in enumerate(tokens):
            repl = aliases.get(token)
            if repl:
                tokens[idx] = repl
        return " ".join(tokens)

    def _llm_router(self) -> LLMRouter | None:
        if self._llm is not None:
            return self._llm if self._llm.enabled() else None
        try:
            self._llm = LLMRouter()
        except Exception:
            self._llm = None
        return self._llm if self._llm is not None and self._llm.enabled() else None

    def propose_intent(self, state: Any, memory: Any, **kwargs: Any) -> dict[str, Any] | None:
        try:
            data = getattr(state, "data", None)
            maxim_runtime = data.get("maxim_runtime") if isinstance(data, dict) else None
            record = data.get("latest_transcript_record") if isinstance(data, dict) else None
            if isinstance(record, dict) and record.get("text") is not None:
                chunk_index = record.get("chunk_index")
                try:
                    chunk_index_int = int(chunk_index) if chunk_index is not None else None
                except Exception:
                    chunk_index_int = None

                if chunk_index_int is not None and chunk_index_int != self._last_llm_chunk_index:
                    self._last_llm_chunk_index = int(chunk_index_int)
                    text = str(record.get("text", "") or "").strip()
                    normalized = self._normalize_transcript_text(text)
                    tokens = normalized.split()

                    if "maxim" in tokens:
                        # Hard overrides: treat explicit mode switches as keyword-only (no LLM).
                        if "sleep" in tokens:
                            return None
                        if "observe" in tokens:
                            return None
                        if "shutdown" in tokens or ("shut" in tokens and "down" in tokens):
                            return None

                        remainder = [t for t in tokens if t != "maxim"]
                        if remainder:
                            llm = self._llm_router()
                            if llm is not None:
                                allowed_tools = {"read_file", "write_file", "execute_file"}
                                allowed_commands = set()
                                if maxim_runtime:
                                    allowed_tools |= {"focus_interests", "maxim_command"}
                                    allowed_commands = {
                                        "center_vision",
                                        "mark_trainable_moment",
                                        "label_outcome",
                                        "request_sleep",
                                        "request_observe",
                                        "request_shutdown",
                                    }

                                action = llm.route(
                                    text,
                                    allowed_tools=allowed_tools,
                                    allowed_commands=allowed_commands,
                                )
                                if isinstance(action, dict) and action.get("tool_name"):
                                    return {"goal": action, "confidence": 1.0}

            if maxim_runtime:
                mode = None
                if isinstance(maxim_runtime, dict):
                    mode = maxim_runtime.get("mode")
                if isinstance(mode, str) and mode.strip().lower() == "sleep":
                    return None

                now = time.time()
                if (now - float(self._last_focus_ts)) >= 0.5:
                    self._last_focus_ts = float(now)
                    return {"goal": {"tool_name": "focus_interests", "params": {}}, "confidence": 1.0}
                return None
        except Exception:
            pass

        latest_transcript = None
        repo_root = None
        try:
            data = getattr(state, "data", None)
            if isinstance(data, dict):
                latest_transcript = data.get("latest_transcript")
                repo_root = data.get("repo_root")
        except Exception:
            latest_transcript = None

        if isinstance(latest_transcript, str) and latest_transcript.strip():
            path = latest_transcript.strip()
            abs_path = path
            try:
                if not os.path.isabs(abs_path) and isinstance(repo_root, str) and repo_root.strip():
                    abs_path = os.path.join(repo_root.strip(), abs_path)
            except Exception:
                abs_path = path

            mtime = None
            try:
                mtime = float(os.path.getmtime(abs_path))
            except Exception:
                mtime = None

            last_path = self._last_transcript_path
            last_mtime = self._last_transcript_mtime
            if last_path != path or (mtime is not None and (last_mtime is None or mtime > float(last_mtime))):
                self._last_transcript_path = path
                self._last_transcript_mtime = mtime
                return {"goal": "read_latest_transcript", "confidence": 1.0}

        if not self._did_read_readme:
            self._did_read_readme = True
            return {"goal": "read_readme", "confidence": 0.5}
        return None
