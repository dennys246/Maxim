"""KeyedRateLimiter — per-key rate limiting, burst detection, and escalating gating.

Cherry-picked from ``src/maxim/mesh/admission.py`` during R0 of the LLM path
refinement (see ``docs/plans/llm_path_foundation.md``). The mesh admission
control module was otherwise dead code (zero production imports), but this
logic is solid and will serve Plan 4's admin API rate limiting plus the
per-agent rate limiter.

Adapted from the original by:
- Renaming ``MeshAdmissionControl`` → ``KeyedRateLimiter`` (key-agnostic)
- Renaming ``peer_id`` → ``key``, ``trust_level`` → ``key_class``
- Removing mesh-specific docstrings + trust-level vocabulary

The sliding-window + burst-detection + escalating-gate algorithm is unchanged.
Plan 4 (``llm_path_operator_visibility.md``) will use this for admin API
rate limiting and per-agent rate limiting. Until Plan 4 ships, this module
is dormant — nothing imports it yet.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Default per-class rate limits (messages per window). Override at instantiation
# via the ``class_limits`` parameter.
DEFAULT_CLASS_LIMITS: dict[str, int] = {
    "verified": 120,
    "discovered": 60,
    "remote": 60,
    "unknown": 20,
}


@dataclass
class RateLimitState:
    """Tracks a single key's behavior for admission decisions."""

    key: str
    key_class: str  # e.g. "verified" | "discovered" | "remote" | "unknown" | caller-defined
    messages_received: int = 0
    messages_this_window: int = 0
    window_start: float = 0.0  # monotonic
    burst_timestamps: list[float] = field(default_factory=list)
    violations: int = 0
    gated_until: float = 0.0  # monotonic time when gate lifts (0 = not gated)
    gate_reason: str = ""

    @property
    def is_gated(self) -> bool:
        return time.monotonic() < self.gated_until


class KeyedRateLimiter:
    """Sliding-window rate limiter with burst detection and escalating gates.

    Keys that exceed rate limits accumulate violations; repeated violations
    trigger escalating gate durations. Misbehaving keys can also be gated
    manually via :meth:`gate`.

    Originally written for mesh admission control; generalized for reuse
    by Plan 4's admin API rate limiter and per-agent rate limiter.
    """

    # 1-minute sliding window for per-window rate limit
    WINDOW_SECONDS: float = 60.0

    # Escalating gate durations (indexed by violation count, capped at last)
    GATE_DURATIONS: list[int] = [30, 120, 600, 3600]  # 30s, 2min, 10min, 1hr

    # Burst detection: messages in a short window trigger immediate gate
    BURST_THRESHOLD: int = 20
    BURST_WINDOW: float = 5.0

    def __init__(self, class_limits: dict[str, int] | None = None) -> None:
        self._states: dict[str, RateLimitState] = {}
        self._lock = threading.Lock()
        self._class_limits = dict(class_limits) if class_limits else dict(DEFAULT_CLASS_LIMITS)

    def check(self, key: str, key_class: str = "unknown") -> tuple[bool, str]:
        """Check if a request from this key should be admitted.

        Returns ``(admitted, reason)``. Called on every incoming request
        before dispatching.
        """
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = RateLimitState(key=key, key_class=key_class)
                self._states[key] = state

            now = time.monotonic()

            # Check if key is currently gated
            if state.is_gated:
                remaining = state.gated_until - now
                return False, f"gated for {remaining:.0f}s ({state.gate_reason})"

            # Reset window if expired
            if now - state.window_start > self.WINDOW_SECONDS:
                state.messages_this_window = 0
                state.window_start = now

            state.messages_received += 1
            state.messages_this_window += 1

            # Burst detection — sliding window of recent timestamps
            state.burst_timestamps.append(now)
            burst_cutoff = now - self.BURST_WINDOW
            state.burst_timestamps = [t for t in state.burst_timestamps if t > burst_cutoff]
            if len(state.burst_timestamps) > self.BURST_THRESHOLD:
                return self._apply_gate(
                    state,
                    now,
                    f"burst detected ({len(state.burst_timestamps)} msgs in {self.BURST_WINDOW}s)",
                )

            # Per-window rate limit (class-aware)
            rate_limit = self._class_limits.get(key_class, 20)
            if state.messages_this_window > rate_limit:
                return self._apply_gate(
                    state,
                    now,
                    f"rate limit exceeded ({state.messages_this_window}/{rate_limit} in {self.WINDOW_SECONDS}s)",
                )

            return True, "ok"

    def _apply_gate(
        self,
        state: RateLimitState,
        now: float,
        reason: str,
    ) -> tuple[bool, str]:
        """Apply an escalating gate to a key. Returns ``(False, reason)``."""
        state.violations += 1
        idx = min(state.violations - 1, len(self.GATE_DURATIONS) - 1)
        duration = self.GATE_DURATIONS[idx]
        state.gated_until = now + duration
        state.gate_reason = reason
        logger.warning("rate_limit: gating key %s for %ds — %s", state.key, duration, reason)
        return False, reason

    def gate(self, key: str, duration_s: float, reason: str) -> None:
        """Manually gate a key (e.g., from an admin command or anomaly detector)."""
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = RateLimitState(key=key, key_class="unknown")
                self._states[key] = state
            state.gated_until = time.monotonic() + duration_s
            state.gate_reason = reason
            state.violations += 1
            logger.warning("rate_limit: manually gating key %s for %ds — %s", key, duration_s, reason)

    def is_gated(self, key: str) -> bool:
        """Quick check for whether a key is currently gated."""
        with self._lock:
            state = self._states.get(key)
            return state.is_gated if state else False

    def ungate(self, key: str) -> None:
        """Manually lift a gate (e.g., after the key is rehabilitated)."""
        with self._lock:
            state = self._states.get(key)
            if state:
                state.gated_until = 0.0
                state.gate_reason = ""

    def get_status(self) -> list[dict[str, Any]]:
        """Return admission state for all known keys (for admin endpoints)."""
        with self._lock:
            return [
                {
                    "key": s.key,
                    "key_class": s.key_class,
                    "messages_received": s.messages_received,
                    "violations": s.violations,
                    "is_gated": s.is_gated,
                    "gate_reason": s.gate_reason,
                }
                for s in self._states.values()
            ]

    def reset(self) -> None:
        """Clear all state (for testing)."""
        with self._lock:
            self._states.clear()
