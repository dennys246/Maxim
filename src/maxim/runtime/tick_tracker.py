"""TickTypeTracker: per-tick classification for observability (Stage 5).

Tracks whether each agent loop tick produced thinking, action, speech,
or noop output.  Used by the heartbeat monitor to distinguish
deliberation ticks from action ticks and detect wedged deliberation.

Thread-safe via internal lock.  The tracker is a lightweight accumulator —
it does NOT drive behavior, only observability.

Env vars:
  MAXIM_THINKING_STALL_S — seconds of consecutive thinking without
      action before warning (default 45s).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("maxim.heartbeat")

DEFAULT_THINKING_STALL_S = 45.0


class TickType(str, Enum):
    """Classification of an agent loop tick."""

    THINKING = "thinking"
    ACTION = "action"
    SPEECH = "speech"
    NOOP = "noop"


@dataclass(frozen=True, slots=True)
class TickRecord:
    """Single tick observation."""

    tick_type: TickType
    timestamp: float
    detail: str = ""


class TickTypeTracker:
    """Accumulates tick-type observations for heartbeat + cost attribution.

    Thread-safe.  All methods are O(1) amortized.
    """

    def __init__(
        self,
        *,
        thinking_stall_s: float | None = None,
        window_size: int = 200,
    ) -> None:
        if thinking_stall_s is None:
            thinking_stall_s = float(os.environ.get("MAXIM_THINKING_STALL_S", str(DEFAULT_THINKING_STALL_S)))
        self._thinking_stall_s = thinking_stall_s
        self._history: deque[TickRecord] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._last_action_time = time.time()
        self._last_stall_warn = 0.0

        # Counters for cost breakdown
        self._counts = {t: 0 for t in TickType}

    def record(self, tick_type: TickType, detail: str = "") -> None:
        """Record a tick observation."""
        now = time.time()
        with self._lock:
            self._history.append(TickRecord(tick_type=tick_type, timestamp=now, detail=detail))
            self._counts[tick_type] += 1
            if tick_type in (TickType.ACTION, TickType.SPEECH):
                self._last_action_time = now

    def check_thinking_stall(self) -> bool:
        """Return True and log WARNING if thinking without action for too long.

        Called by HeartbeatMonitor on each sample.  Warns at most once
        per stall threshold window.
        """
        with self._lock:
            elapsed = time.time() - self._last_action_time
            if elapsed <= self._thinking_stall_s:
                return False

            # Check that recent ticks are actually thinking (not just idle)
            recent = list(self._history)
        if not recent:
            return False

        recent_types = {r.tick_type for r in recent[-10:]}
        if TickType.THINKING not in recent_types:
            return False  # idle, not wedged

        now = time.time()
        if now - self._last_stall_warn < self._thinking_stall_s:
            return True  # already warned recently

        self._last_stall_warn = now
        logger.warning(
            "[heartbeat] THINKING STALL — deliberating for %.0fs without "
            "producing an action or speech. Recent tick types: %s",
            elapsed,
            [r.tick_type.value for r in recent[-5:]],
        )
        return True

    def snapshot(self) -> dict[str, int]:
        """Return tick counts for display / cost breakdown."""
        with self._lock:
            return dict(self._counts)

    def recent(self, limit: int = 10) -> list[TickRecord]:
        """Last N tick records."""
        with self._lock:
            items = list(self._history)
        return items[-limit:]
