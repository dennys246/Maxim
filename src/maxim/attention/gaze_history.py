"""Gaze history tracking with spatial salience and inhibition-of-return.

Implements a spatial attention mechanism that:
1. Tracks recent gaze positions with timestamps
2. Applies temporal decay to visited positions (inhibition-of-return)
3. Computes salience scores for potential targets
4. Prevents excessive movements by requiring sufficient salience to move

This creates more natural, purposeful head movements by making Reachy
"remember" where it has looked recently and prefer novel positions.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GazeHistoryConfig:
    """Configuration for gaze history tracking.

    Attributes:
        decay_seconds: Time for salience to recover from 0 to ~63% (1 - 1/e).
        min_salience_to_move: Minimum salience required to execute a movement.
        spatial_sigma: Distance (in pixels) for Gaussian spatial similarity.
        max_history_size: Maximum number of gaze positions to track.
        dwell_time_seconds: Minimum time to stay at a position before moving.
    """

    decay_seconds: float = 2.0  # Salience recovers over ~2 seconds
    min_salience_to_move: float = 0.3  # Need 30% salience to move
    spatial_sigma: float = 80.0  # Pixels - positions within this are "similar"
    max_history_size: int = 50  # Track last 50 gaze positions
    dwell_time_seconds: float = 0.5  # Stay at position for at least 0.5s


@dataclass
class GazePosition:
    """A recorded gaze position with timestamp."""

    u: float  # Image x coordinate
    v: float  # Image y coordinate
    timestamp: float = field(default_factory=time.time)


class GazeHistory:
    """Tracks gaze history and computes spatial salience for targets.

    Uses inhibition-of-return (IOR) to prevent rapid, repetitive movements.
    Positions that were recently gazed at have low salience, which recovers
    over time according to exponential decay.

    Example:
        history = GazeHistory()

        # Check if we should move to a target
        target = (320, 240)
        salience = history.get_salience(target)
        if salience >= history.config.min_salience_to_move:
            # Execute movement
            history.record_gaze(target)
    """

    def __init__(self, config: GazeHistoryConfig | None = None) -> None:
        """Initialize gaze history tracker.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or GazeHistoryConfig()
        self._history: deque[GazePosition] = deque(maxlen=self.config.max_history_size)
        self._lock = threading.Lock()
        self._last_move_time: float = 0.0
        self._current_position: Tuple[float, float] | None = None

    def record_gaze(self, position: Tuple[float, float]) -> None:
        """Record a gaze position.

        Call this after executing a movement to the given position.

        Args:
            position: (u, v) image coordinates of gaze target.
        """
        u, v = position
        now = time.time()
        with self._lock:
            self._history.append(GazePosition(u=u, v=v, timestamp=now))
            self._last_move_time = now
            self._current_position = position

    def get_salience(self, target: Tuple[float, float]) -> float:
        """Compute salience score for a potential target position.

        Returns a value in [0, 1] where:
        - 1.0 = highly salient (never visited or visited long ago)
        - 0.0 = not salient (recently visited)

        The salience is computed as the minimum salience contribution from
        all recent gaze positions, weighted by spatial distance and temporal decay.

        Args:
            target: (u, v) image coordinates of potential target.

        Returns:
            Salience score in [0, 1].
        """
        target_u, target_v = target
        now = time.time()

        with self._lock:
            if not self._history:
                return 1.0  # No history = fully salient

            # Compute salience contribution from each historical gaze
            # Lower contribution = more inhibition
            min_salience = 1.0

            for gaze in self._history:
                # Spatial similarity (Gaussian falloff)
                dist_sq = (target_u - gaze.u) ** 2 + (target_v - gaze.v) ** 2
                spatial_weight = math.exp(-dist_sq / (2 * self.config.spatial_sigma**2))

                # Temporal decay (exponential recovery)
                time_since = now - gaze.timestamp
                temporal_recovery = 1.0 - math.exp(-time_since / self.config.decay_seconds)

                # Salience contribution: high spatial similarity + recent = low salience
                # As time passes, temporal_recovery → 1, so salience → 1
                contribution = 1.0 - spatial_weight * (1.0 - temporal_recovery)

                min_salience = min(min_salience, contribution)

            return min_salience

    def should_move(self, target: Tuple[float, float]) -> Tuple[bool, float, str]:
        """Check if a movement to target should be executed.

        Considers both salience and dwell time requirements.

        Args:
            target: (u, v) image coordinates of potential target.

        Returns:
            Tuple of (should_move, salience, reason).
        """
        now = time.time()
        salience = self.get_salience(target)

        with self._lock:
            time_since_last_move = now - self._last_move_time

        # Check dwell time
        if time_since_last_move < self.config.dwell_time_seconds:
            remaining = self.config.dwell_time_seconds - time_since_last_move
            return False, salience, f"dwell_time ({remaining:.2f}s remaining)"

        # Check salience threshold
        if salience < self.config.min_salience_to_move:
            return False, salience, f"low_salience ({salience:.2f} < {self.config.min_salience_to_move})"

        return True, salience, "ok"

    def get_current_position(self) -> Tuple[float, float] | None:
        """Get the current gaze position if known."""
        with self._lock:
            return self._current_position

    def time_since_last_move(self) -> float:
        """Get time since the last recorded movement."""
        with self._lock:
            if self._last_move_time == 0.0:
                return float("inf")
            return time.time() - self._last_move_time

    def clear(self) -> None:
        """Clear all gaze history."""
        with self._lock:
            self._history.clear()
            self._last_move_time = 0.0
            self._current_position = None

    @property
    def history_size(self) -> int:
        """Number of positions in history."""
        with self._lock:
            return len(self._history)


__all__ = [
    "GazeHistory",
    "GazeHistoryConfig",
    "GazePosition",
]
