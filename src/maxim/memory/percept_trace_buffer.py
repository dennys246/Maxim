"""Shared tick-driven ring buffer for recent percept activations.

Multiple consumers (NAc reward crediting, ReactionProducers, replay
schedulers) read from this buffer.  No agent-layer imports.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


@dataclass
class TraceEntry:
    """Single percept activation record."""

    agent_id: str
    percept_id: str
    tick: int
    activation_strength: float  # starts at 1.0, decays with τ
    registered_at: float  # wall-clock timestamp


class PerceptTraceBuffer:
    """Ring buffer of recent percept activations with exponential τ-decay."""

    def __init__(
        self,
        max_entries: int = 500,
        tau: float = 10.0,
        tick_rate: float = 1.0,
        min_activation: float = 0.01,
    ) -> None:
        self._max_entries = max_entries
        self._tau = tau
        self._tick_rate = tick_rate
        self._min_activation = min_activation
        self._entries: list[TraceEntry] = []
        self._tick_counter = 0
        self._lock = threading.Lock()

    def record(self, agent_id: str, percept_id: str, activation: float = 1.0) -> None:
        """Record a new percept activation."""
        entry = TraceEntry(
            agent_id=agent_id,
            percept_id=percept_id,
            tick=self._tick_counter,
            activation_strength=activation,
            registered_at=time.monotonic(),
        )
        with self._lock:
            self._entries.append(entry)
            # Evict oldest if over capacity
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]

    def tick(self) -> None:
        """Advance tick counter, apply decay, evict dead entries."""
        decay = math.exp(-1.0 / self._tau)
        with self._lock:
            self._tick_counter += 1
            for e in self._entries:
                e.activation_strength *= decay
            self._entries = [e for e in self._entries if e.activation_strength >= self._min_activation]

    def snapshot(self, agent_id: str | None = None, min_activation: float = 0.01) -> list[TraceEntry]:
        """Return active entries sorted by activation descending."""
        with self._lock:
            entries = [
                e
                for e in self._entries
                if e.activation_strength >= min_activation and (agent_id is None or e.agent_id == agent_id)
            ]
        return sorted(entries, key=lambda e: e.activation_strength, reverse=True)

    def recent(self, agent_id: str | None = None, k: int = 10) -> list[TraceEntry]:
        """Return the k most recently recorded entries."""
        with self._lock:
            if agent_id is None:
                entries = list(self._entries)
            else:
                entries = [e for e in self._entries if e.agent_id == agent_id]
        # Most recent = highest tick, then latest in insertion order
        return entries[-k:][::-1] if len(entries) > k else entries[::-1]

    def reset(self, agent_id: str | None = None) -> None:
        """Clear all entries, or just entries for a specific agent."""
        with self._lock:
            if agent_id is None:
                self._entries.clear()
            else:
                self._entries = [e for e in self._entries if e.agent_id != agent_id]

    @property
    def current_tick(self) -> int:
        return self._tick_counter

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
