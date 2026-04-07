"""MeshAdmissionControl — per-peer rate limiting, burst detection, and gating.

Sits between the /v1/mesh/message endpoint and PeerChannel.receive().
Peers that exceed rate limits accumulate violations; repeated violations
trigger escalating gate durations. Misbehaving peers can also be gated
manually.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Trust-level rate limits (messages per window)
_TRUST_RATE_LIMITS: dict[str, int] = {
    "verified": 120,
    "discovered": 60,
    "remote": 60,
    "unknown": 20,
}


@dataclass
class PeerAdmissionState:
    """Tracks a single peer's behavior for admission decisions."""

    peer_id: str
    trust_level: str  # "verified" | "discovered" | "remote" | "unknown"
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


class MeshAdmissionControl:
    """Rate limiter and circuit breaker for incoming mesh messages.

    Peers that exceed rate limits accumulate violations; repeated
    violations trigger escalating gate durations.
    """

    # 1-minute sliding window for per-window rate limit
    WINDOW_SECONDS: float = 60.0

    # Escalating gate durations (indexed by violation count, capped at last)
    GATE_DURATIONS: list[int] = [30, 120, 600, 3600]  # 30s, 2min, 10min, 1hr

    # Burst detection: messages in a short window trigger immediate gate
    BURST_THRESHOLD: int = 20
    BURST_WINDOW: float = 5.0

    def __init__(self) -> None:
        self._peers: dict[str, PeerAdmissionState] = {}
        self._lock = threading.Lock()

    def check(self, peer_id: str, trust_level: str = "unknown") -> tuple[bool, str]:
        """Check if a message from this peer should be admitted.

        Returns (admitted: bool, reason: str).
        Called on every incoming mesh message before dispatching.
        """
        with self._lock:
            state = self._peers.get(peer_id)
            if state is None:
                state = PeerAdmissionState(peer_id=peer_id, trust_level=trust_level)
                self._peers[peer_id] = state

            now = time.monotonic()

            # Check if peer is currently gated
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

            # Per-window rate limit (trust-level aware)
            rate_limit = _TRUST_RATE_LIMITS.get(trust_level, 20)
            if state.messages_this_window > rate_limit:
                return self._apply_gate(
                    state,
                    now,
                    f"rate limit exceeded ({state.messages_this_window}/{rate_limit} in {self.WINDOW_SECONDS}s)",
                )

            return True, "ok"

    def _apply_gate(
        self,
        state: PeerAdmissionState,
        now: float,
        reason: str,
    ) -> tuple[bool, str]:
        """Apply an escalating gate to a peer. Returns (False, reason)."""
        state.violations += 1
        idx = min(state.violations - 1, len(self.GATE_DURATIONS) - 1)
        duration = self.GATE_DURATIONS[idx]
        state.gated_until = now + duration
        state.gate_reason = reason
        logger.warning("mesh: gating peer %s for %ds — %s", state.peer_id, duration, reason)
        return False, reason

    def gate_peer(self, peer_id: str, duration_s: float, reason: str) -> None:
        """Manually gate a peer (e.g., from an admin command or anomaly detector)."""
        with self._lock:
            state = self._peers.get(peer_id)
            if state is None:
                state = PeerAdmissionState(peer_id=peer_id, trust_level="unknown")
                self._peers[peer_id] = state
            state.gated_until = time.monotonic() + duration_s
            state.gate_reason = reason
            state.violations += 1
            logger.warning("mesh: manually gating peer %s for %ds — %s", peer_id, duration_s, reason)

    def is_peer_gated(self, peer_id: str) -> bool:
        """Quick check used by distributed planning to skip gated peers."""
        with self._lock:
            state = self._peers.get(peer_id)
            return state.is_gated if state else False

    def ungate_peer(self, peer_id: str) -> None:
        """Manually lift a gate (e.g., after the peer is fixed)."""
        with self._lock:
            state = self._peers.get(peer_id)
            if state:
                state.gated_until = 0.0
                state.gate_reason = ""

    def get_status(self) -> list[dict[str, Any]]:
        """Return admission state for all known peers (for /v1/mesh/status)."""
        with self._lock:
            return [
                {
                    "peer_id": s.peer_id,
                    "trust_level": s.trust_level,
                    "messages_received": s.messages_received,
                    "violations": s.violations,
                    "is_gated": s.is_gated,
                    "gate_reason": s.gate_reason,
                }
                for s in self._peers.values()
            ]

    def reset(self) -> None:
        """Clear all admission state (for testing)."""
        with self._lock:
            self._peers.clear()
