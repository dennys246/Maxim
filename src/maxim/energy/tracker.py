"""Base energy tracker - Abstract interface for energy tracking.

Provides the base class and protocol for domain-specific energy trackers.
Each tracker handles a specific type of energy and provides consistent
recording and querying interfaces.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

from maxim.energy.signal import EnergySignal, EnergyType

logger = logging.getLogger(__name__)


@dataclass
class EnergyConfig:
    """Base configuration for energy trackers.

    Attributes:
        window_seconds: Time window for rate calculations.
        max_history: Maximum signals to retain in history.
        enable_logging: Whether to log energy events.
        log_threshold: Minimum energy amount to log.
    """

    window_seconds: float = 60.0        # 1-minute sliding window
    max_history: int = 1000             # Keep last 1000 signals
    enable_logging: bool = True
    log_threshold: float = 1.0          # Log signals >= 1.0 energy


class EnergyTracker(ABC):
    """Abstract base class for energy trackers.

    Each tracker monitors a specific domain of energy expenditure
    (e.g., LLM tokens, motor commands) and provides:
    - Recording of energy signals
    - Statistics over time windows
    - Integration with the energy registry

    Subclasses must implement:
    - name: Unique identifier for this tracker
    - energy_types: Set of EnergyType values this tracker handles
    - record(): Domain-specific recording method
    """

    name: str = "base"
    energy_types: set[EnergyType] = set()

    def __init__(self, config: EnergyConfig | None = None) -> None:
        """Initialize the tracker.

        Args:
            config: Configuration settings. Uses defaults if None.
        """
        self.config = config or EnergyConfig()
        self._history: deque[EnergySignal] = deque(maxlen=self.config.max_history)
        self._lock = threading.RLock()

        # Running totals
        self._total_energy = 0.0
        self._total_count = 0

    def _record_signal(self, signal: EnergySignal) -> None:
        """Internal method to record a signal.

        Args:
            signal: The energy signal to record.
        """
        with self._lock:
            self._history.append(signal)
            self._total_energy += signal.amount
            self._total_count += 1

        # Log significant signals
        if self.config.enable_logging and signal.amount >= self.config.log_threshold:
            logger.info(
                "Energy: %s %.2f units from %s (total=%.1f)",
                signal.energy_type.value,
                signal.amount,
                signal.source or self.name,
                self._total_energy,
            )

    @abstractmethod
    def record(self, **kwargs: Any) -> EnergySignal:
        """Record an energy expenditure.

        Subclasses implement this with domain-specific parameters.

        Returns:
            The recorded EnergySignal.
        """
        ...

    def get_window_stats(self, window_seconds: float | None = None) -> dict[str, Any]:
        """Get statistics for a time window.

        Args:
            window_seconds: Window size. Uses config default if None.

        Returns:
            Dict with count, total_energy, rate_per_second, etc.
        """
        window = window_seconds or self.config.window_seconds
        cutoff = time.time() - window

        with self._lock:
            window_signals = [s for s in self._history if s.timestamp >= cutoff]

        if not window_signals:
            return {
                "count": 0,
                "total_energy": 0.0,
                "rate_per_second": 0.0,
                "average_energy": 0.0,
                "window_seconds": window,
            }

        total = sum(s.amount for s in window_signals)
        duration = time.time() - window_signals[0].timestamp if window_signals else window

        return {
            "count": len(window_signals),
            "total_energy": round(total, 4),
            "rate_per_second": round(total / max(duration, 0.001), 4),
            "average_energy": round(total / len(window_signals), 4),
            "window_seconds": window,
            "oldest_timestamp": window_signals[0].timestamp if window_signals else None,
        }

    def get_total_stats(self) -> dict[str, Any]:
        """Get all-time statistics.

        Returns:
            Dict with total_count, total_energy, average_energy.
        """
        with self._lock:
            return {
                "total_count": self._total_count,
                "total_energy": round(self._total_energy, 4),
                "average_energy": round(
                    self._total_energy / max(self._total_count, 1), 4
                ),
                "history_size": len(self._history),
            }

    def get_recent_signals(self, count: int = 10) -> list[EnergySignal]:
        """Get most recent signals.

        Args:
            count: Number of signals to return.

        Returns:
            List of recent EnergySignal objects (newest first).
        """
        with self._lock:
            signals = list(self._history)
        return signals[-count:][::-1]

    def clear(self) -> None:
        """Clear all history and reset counters."""
        with self._lock:
            self._history.clear()
            self._total_energy = 0.0
            self._total_count = 0

    def handles(self, energy_type: EnergyType) -> bool:
        """Check if this tracker handles the given energy type.

        Args:
            energy_type: The type to check.

        Returns:
            True if this tracker handles the type.
        """
        return energy_type in self.energy_types


__all__ = [
    "EnergyConfig",
    "EnergyTracker",
]
