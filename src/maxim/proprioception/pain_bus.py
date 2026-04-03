"""Central publish/subscribe for pain signals from any source.

Extracted from PainDetector's internal callback mechanism to allow
pain signals from motor, tool, simulation, energy, and cognitive
sources to reach all consumers through a single channel.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Callable

from maxim.proprioception.pain import PainSignal, PainType

logger = logging.getLogger(__name__)


class PainBus:
    """Central publish/subscribe for pain signals.

    Any system can publish a PainSignal. All subscribers are notified
    synchronously (consistent with existing PainDetector behavior).

    Example:
        bus = PainBus()
        bus.subscribe(lambda sig: print(f"Pain: {sig.pain_type}"))
        bus.publish(PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.7,
            timestamp=time.time(),
        ))
    """

    def __init__(self, history_size: int = 200) -> None:
        self._subscribers: list[Callable[[PainSignal], None]] = []
        self._lock = threading.Lock()
        self._history: deque[PainSignal] = deque(maxlen=history_size)
        self._total_published: int = 0

    def subscribe(self, callback: Callable[[PainSignal], None]) -> None:
        """Register a pain signal consumer."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[PainSignal], None]) -> None:
        """Remove a previously registered consumer."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def publish(self, signal: PainSignal) -> None:
        """Publish a pain signal to all subscribers.

        Callbacks are invoked outside the lock to prevent deadlocks
        with subscribers that query the bus.
        """
        with self._lock:
            self._history.append(signal)
            self._total_published += 1
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(signal)
            except Exception as e:
                logger.warning("PainBus subscriber error: %s", e)

    @property
    def recent(self) -> list[PainSignal]:
        """Recent pain signals (newest last)."""
        with self._lock:
            return list(self._history)

    def recent_by_type(self, pain_type: PainType) -> list[PainSignal]:
        """Recent signals filtered by type."""
        with self._lock:
            return [s for s in self._history if s.pain_type == pain_type]

    def get_stats(self) -> dict[str, int]:
        """Bus-level statistics."""
        with self._lock:
            return {
                "total_published": self._total_published,
                "subscriber_count": len(self._subscribers),
                "history_size": len(self._history),
            }
