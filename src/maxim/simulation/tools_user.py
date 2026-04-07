"""ask_user tool — human-in-the-loop interaction for generative campaigns.

Provides:
- ``AskUserTool`` — pauses simulation for human input via stdin
- JSONL audit writer for recording all interactions
- Replay reader for deterministic re-runs from recorded sessions
- Timeout escalation tracking (consecutive timeouts)

Modes:
- Interactive (default): prompt via stdin with randomized timeout
- Non-interactive: return default immediately (for CI/automated runs)
- Replay: read recorded responses from user_interactions.jsonl
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maxim.tools.base import Tool

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interaction record
# ---------------------------------------------------------------------------


@dataclass
class UserInteraction:
    """Record of one user interaction."""

    turn: int
    question: str
    options: list[str] | None = None
    response: str | None = None
    was_default: bool = False
    timed_out: bool = False
    elapsed_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "question": self.question,
            "options": self.options,
            "response": self.response,
            "was_default": self.was_default,
            "timed_out": self.timed_out,
            "elapsed_s": round(self.elapsed_s, 2),
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# JSONL audit writer
# ---------------------------------------------------------------------------


class InteractionAuditWriter:
    """Writes user interactions to JSONL for replay."""

    def __init__(self, session_dir: str | Path) -> None:
        self._path = Path(session_dir) / "user_interactions.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, interaction: UserInteraction) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps(interaction.to_dict()) + "\n")

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# Replay reader
# ---------------------------------------------------------------------------


class InteractionReplayReader:
    """Reads recorded interactions for deterministic replay."""

    def __init__(self, session_dir: str | Path) -> None:
        path = Path(session_dir) / "user_interactions.jsonl"
        self._interactions: list[dict[str, Any]] = []
        self._index = 0

        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._interactions.append(json.loads(line))

    def next_response(self) -> dict[str, Any] | None:
        """Return the next recorded interaction, or None if exhausted."""
        if self._index >= len(self._interactions):
            return None
        interaction = self._interactions[self._index]
        self._index += 1
        return interaction

    @property
    def remaining(self) -> int:
        return len(self._interactions) - self._index


# ---------------------------------------------------------------------------
# Timeout escalation tracker
# ---------------------------------------------------------------------------


class TimeoutEscalation:
    """Tracks consecutive timeouts for narrator escalation behavior."""

    def __init__(self) -> None:
        self._consecutive: int = 0
        self._total: int = 0

    def record_timeout(self) -> int:
        """Record a timeout. Returns new consecutive count."""
        self._consecutive += 1
        self._total += 1
        return self._consecutive

    def record_response(self) -> None:
        """Record a successful response — resets consecutive counter."""
        self._consecutive = 0

    @property
    def consecutive(self) -> int:
        return self._consecutive

    @property
    def total(self) -> int:
        return self._total

    @property
    def escalation_level(self) -> str:
        """Current escalation level for narrator guidance."""
        if self._consecutive == 0:
            return "none"
        elif self._consecutive == 1:
            return "gentle_nudge"
        elif self._consecutive <= 3:
            return "world_reacts"
        else:
            return "passive_protagonist"

    def to_narrator_hint(self) -> str:
        """Generate a hint for the narrator based on timeout history."""
        if self._consecutive == 0:
            return ""
        if self._consecutive == 1:
            return (
                "[TIMEOUT: 1st — gentle nudge. An NPC shifts impatiently, "
                "the wind picks up. Re-present the choice naturally.]"
            )
        if self._consecutive <= 3:
            return (
                f"[TIMEOUT: {self._consecutive}th — the world reacts. NPCs make "
                "their own decisions, opportunities close. Present a NEW path "
                "forward that doesn't require the missed choice.]"
            )
        return (
            f"[TIMEOUT: {self._consecutive}th — passive protagonist. The story "
            "happens TO them. Switch to observation-heavy scenes. Reduce "
            "interaction frequency.]"
        )


# ---------------------------------------------------------------------------
# AskUserTool
# ---------------------------------------------------------------------------


class AskUserTool(Tool):
    """Pause simulation for human input.

    Parameters
    ----------
    session_dir : str or Path, optional
        Directory for JSONL audit log.
    replay_from : str or Path, optional
        Session directory to replay responses from.
    non_interactive : bool
        If True, return default immediately (for CI/automated runs).
    timeout_base : float
        Base timeout in seconds (randomized +-2s around this).
    """

    name = "ask_user"
    description = (
        "Pause the simulation and ask the human player a question. "
        "Use this when you want the player to make a choice, provide input, "
        "or roll dice. Returns their response or a timeout indicator."
    )
    input_schema = {
        "question": str,
        "options": (list, None),
        "default": (str, ""),
        "timeout_sec": (int, 10),
    }

    def __init__(
        self,
        *,
        session_dir: str | Path | None = None,
        replay_from: str | Path | None = None,
        non_interactive: bool = False,
        timeout_base: float = 10.0,
        turn_counter: Any = None,
    ) -> None:
        super().__init__()
        self._non_interactive = non_interactive
        self._timeout_base = timeout_base
        self._turn_counter = turn_counter
        self._escalation = TimeoutEscalation()

        self._audit: InteractionAuditWriter | None = None
        if session_dir:
            self._audit = InteractionAuditWriter(session_dir)

        self._replay: InteractionReplayReader | None = None
        if replay_from:
            self._replay = InteractionReplayReader(replay_from)

    @property
    def escalation(self) -> TimeoutEscalation:
        return self._escalation

    def execute(
        self,
        question: str = "",
        options: list[str] | None = None,
        default: str = "",
        timeout_sec: int = 10,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Ask the user a question and return their response.

        Returns
        -------
        dict
            - response: str — the user's answer (or default if timed out)
            - was_default: bool — whether the default was used
            - timed_out: bool — whether the interaction timed out
            - escalation_level: str — current timeout escalation level
        """
        turn = self._turn_counter() if callable(self._turn_counter) else 0
        start = time.time()

        # Mode: replay
        if self._replay is not None:
            return self._execute_replay(question, options, default, turn, start)

        # Mode: non-interactive
        if self._non_interactive:
            return self._execute_non_interactive(question, options, default, turn, start)

        # Mode: interactive (stdin)
        return self._execute_interactive(question, options, default, timeout_sec, turn, start)

    def _execute_replay(
        self,
        question: str,
        options: list[str] | None,
        default: str,
        turn: int,
        start: float,
    ) -> dict[str, Any]:
        """Replay a recorded interaction (instant, no timeout)."""
        recorded = self._replay.next_response() if self._replay else None

        if recorded is not None:
            response = recorded.get("response") or default
            timed_out = recorded.get("timed_out", False)
        else:
            response = default
            timed_out = True

        if timed_out:
            self._escalation.record_timeout()
        else:
            self._escalation.record_response()

        interaction = UserInteraction(
            turn=turn,
            question=question,
            options=options,
            response=response,
            was_default=timed_out,
            timed_out=timed_out,
            elapsed_s=0.0,
        )
        if self._audit:
            self._audit.record(interaction)

        return {
            "response": response,
            "was_default": timed_out,
            "timed_out": timed_out,
            "escalation_level": self._escalation.escalation_level,
        }

    def _execute_non_interactive(
        self,
        question: str,
        options: list[str] | None,
        default: str,
        turn: int,
        start: float,
    ) -> dict[str, Any]:
        """Non-interactive mode — return default immediately."""
        interaction = UserInteraction(
            turn=turn,
            question=question,
            options=options,
            response=default,
            was_default=True,
            timed_out=False,
            elapsed_s=0.0,
        )
        if self._audit:
            self._audit.record(interaction)

        return {
            "response": default,
            "was_default": True,
            "timed_out": False,
            "escalation_level": self._escalation.escalation_level,
        }

    def _execute_interactive(
        self,
        question: str,
        options: list[str] | None,
        default: str,
        timeout_sec: int,
        turn: int,
        start: float,
    ) -> dict[str, Any]:
        """Interactive mode — prompt via stdin with timeout."""
        # Randomize timeout (+-2s) to feel natural
        actual_timeout = max(
            3.0,
            self._timeout_base + random.uniform(-2.0, 2.0),
        )

        # Display prompt
        print(f"\n  {question}")
        if options:
            for i, opt in enumerate(options, 1):
                print(f"    {i}. {opt}")
        if default:
            print(f"  [default: {default}, timeout: {actual_timeout:.0f}s]")
        print("  > ", end="", flush=True)

        # Read with timeout using select (Unix) or simple input
        response = None
        timed_out = False
        try:
            import select

            ready, _, _ = select.select([sys.stdin], [], [], actual_timeout)
            if ready:
                response = sys.stdin.readline().strip()
            else:
                timed_out = True
                print("(timed out)")
        except (ImportError, OSError):
            # select not available (Windows) — fall back to blocking input
            try:
                response = input()
            except EOFError:
                timed_out = True

        elapsed = time.time() - start

        if not response or timed_out:
            response = default
            if timed_out:
                self._escalation.record_timeout()
            else:
                self._escalation.record_response()
        else:
            self._escalation.record_response()
            # Map numbered choice to option text
            if options and response.isdigit():
                idx = int(response) - 1
                if 0 <= idx < len(options):
                    response = options[idx]

        was_default = timed_out or (response == default and not response)

        interaction = UserInteraction(
            turn=turn,
            question=question,
            options=options,
            response=response,
            was_default=was_default,
            timed_out=timed_out,
            elapsed_s=elapsed,
        )
        if self._audit:
            self._audit.record(interaction)

        return {
            "response": response,
            "was_default": was_default,
            "timed_out": timed_out,
            "escalation_level": self._escalation.escalation_level,
        }
