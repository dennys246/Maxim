"""MovementTracker - Tracks position history and computes movement metrics.

Provides velocity and acceleration calculations for proprioceptive awareness,
enabling pain detection when movement becomes excessive.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class MovementSample:
    """A single position sample with timestamp."""

    timestamp: float
    yaw: float  # degrees
    pitch: float  # degrees
    x: float  # mm
    y: float  # mm
    z: float  # mm
    roll: float  # degrees


@dataclass
class MovementMetrics:
    """Computed movement metrics from recent samples."""

    angular_velocity: float  # deg/sec (combined yaw+pitch magnitude)
    translation_velocity: float  # mm/sec (combined x+y+z magnitude)
    angular_acceleration: float  # deg/sec²
    translation_acceleration: float  # mm/sec²
    direction_reversals: int  # Number of direction changes in window
    sample_count: int  # Number of samples used
    window_duration: float  # Actual time span of samples

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "angular_velocity": round(self.angular_velocity, 2),
            "translation_velocity": round(self.translation_velocity, 2),
            "angular_acceleration": round(self.angular_acceleration, 2),
            "translation_acceleration": round(self.translation_acceleration, 2),
            "direction_reversals": self.direction_reversals,
            "sample_count": self.sample_count,
            "window_duration": round(self.window_duration, 3),
        }


class MovementTracker:
    """Tracks position history and computes velocity/acceleration.

    Records position samples over time and computes movement metrics
    that can be used for pain detection (excessive velocity, thrashing, etc.).

    Example:
        tracker = MovementTracker(window_seconds=2.0)

        # Record positions as they come in
        metrics = tracker.record_position(
            yaw=10.0, pitch=5.0,
            x=0.0, y=2.0, z=1.0, roll=0.0,
        )

        if metrics and metrics.angular_velocity > 100.0:
            print("Moving too fast!")
    """

    def __init__(
        self,
        window_seconds: float = 2.0,
        sample_limit: int = 100,
        min_samples_for_metrics: int = 3,
    ) -> None:
        """Initialize the movement tracker.

        Args:
            window_seconds: Time window to consider for metrics (seconds).
            sample_limit: Maximum number of samples to retain.
            min_samples_for_metrics: Minimum samples needed to compute metrics.
        """
        self._window = window_seconds
        self._min_samples = min_samples_for_metrics
        self._samples: deque[MovementSample] = deque(maxlen=sample_limit)
        self._lock = threading.RLock()

        # Cache for velocity calculations (for acceleration)
        self._prev_angular_velocity: float | None = None
        self._prev_translation_velocity: float | None = None
        self._prev_velocity_time: float | None = None

    def record_position(
        self,
        yaw: float,
        pitch: float,
        x: float,
        y: float,
        z: float,
        roll: float,
    ) -> MovementMetrics | None:
        """Record current position and return metrics if enough samples.

        Args:
            yaw: Current yaw angle (degrees)
            pitch: Current pitch angle (degrees)
            x: Current x translation (mm)
            y: Current y translation (mm)
            z: Current z translation (mm)
            roll: Current roll angle (degrees)

        Returns:
            MovementMetrics if enough samples, None otherwise.
        """
        now = time.time()

        sample = MovementSample(
            timestamp=now,
            yaw=yaw,
            pitch=pitch,
            x=x,
            y=y,
            z=z,
            roll=roll,
        )

        with self._lock:
            self._samples.append(sample)
            return self._compute_metrics(now)

    def get_metrics(self) -> MovementMetrics | None:
        """Get current movement metrics without recording new position."""
        with self._lock:
            return self._compute_metrics(time.time())

    def _compute_metrics(self, now: float) -> MovementMetrics | None:
        """Compute metrics from samples within the time window."""
        if len(self._samples) < self._min_samples:
            return None

        # Get samples within window
        cutoff = now - self._window
        recent_samples = [s for s in self._samples if s.timestamp >= cutoff]

        if len(recent_samples) < self._min_samples:
            return None

        # Sort by timestamp
        recent_samples.sort(key=lambda s: s.timestamp)

        # Compute velocities
        angular_velocity = self._compute_angular_velocity(recent_samples)
        translation_velocity = self._compute_translation_velocity(recent_samples)

        # Compute accelerations
        angular_accel = 0.0
        translation_accel = 0.0

        if (
            self._prev_angular_velocity is not None
            and self._prev_velocity_time is not None
        ):
            dt = now - self._prev_velocity_time
            if dt > 0.01:  # Avoid division by very small numbers
                angular_accel = (angular_velocity - self._prev_angular_velocity) / dt
                if self._prev_translation_velocity is not None:
                    translation_accel = (
                        translation_velocity - self._prev_translation_velocity
                    ) / dt

        # Update velocity cache
        self._prev_angular_velocity = angular_velocity
        self._prev_translation_velocity = translation_velocity
        self._prev_velocity_time = now

        # Count direction reversals
        direction_reversals = self._count_direction_reversals(recent_samples)

        # Calculate actual window duration
        window_duration = recent_samples[-1].timestamp - recent_samples[0].timestamp

        return MovementMetrics(
            angular_velocity=angular_velocity,
            translation_velocity=translation_velocity,
            angular_acceleration=angular_accel,
            translation_acceleration=translation_accel,
            direction_reversals=direction_reversals,
            sample_count=len(recent_samples),
            window_duration=window_duration,
        )

    def _compute_angular_velocity(self, samples: list[MovementSample]) -> float:
        """Compute angular velocity from recent samples.

        Uses the magnitude of combined yaw+pitch velocity.
        """
        if len(samples) < 2:
            return 0.0

        # Use first and last samples for overall velocity
        first = samples[0]
        last = samples[-1]

        dt = last.timestamp - first.timestamp
        if dt <= 0:
            return 0.0

        # Calculate angular displacement
        yaw_delta = last.yaw - first.yaw
        pitch_delta = last.pitch - first.pitch
        roll_delta = last.roll - first.roll

        # Combined angular magnitude
        angular_distance = math.sqrt(yaw_delta**2 + pitch_delta**2 + roll_delta**2)

        return angular_distance / dt

    def _compute_translation_velocity(self, samples: list[MovementSample]) -> float:
        """Compute translation velocity from recent samples."""
        if len(samples) < 2:
            return 0.0

        first = samples[0]
        last = samples[-1]

        dt = last.timestamp - first.timestamp
        if dt <= 0:
            return 0.0

        # Calculate translation displacement
        x_delta = last.x - first.x
        y_delta = last.y - first.y
        z_delta = last.z - first.z

        # Euclidean distance
        translation_distance = math.sqrt(x_delta**2 + y_delta**2 + z_delta**2)

        return translation_distance / dt

    def _count_direction_reversals(self, samples: list[MovementSample]) -> int:
        """Count sign changes in velocity direction (yaw).

        A reversal indicates the robot changed direction, which is
        indicative of "thrashing" behavior when frequent.
        """
        if len(samples) < 3:
            return 0

        reversals = 0
        prev_direction: int | None = None

        for i in range(1, len(samples)):
            yaw_delta = samples[i].yaw - samples[i - 1].yaw

            # Determine direction (ignore small movements)
            if abs(yaw_delta) < 0.5:  # Ignore noise
                continue

            direction = 1 if yaw_delta > 0 else -1

            if prev_direction is not None and direction != prev_direction:
                reversals += 1

            prev_direction = direction

        return reversals

    def clear(self) -> None:
        """Clear all samples and reset state."""
        with self._lock:
            self._samples.clear()
            self._prev_angular_velocity = None
            self._prev_translation_velocity = None
            self._prev_velocity_time = None

    def get_sample_count(self) -> int:
        """Get current number of samples."""
        with self._lock:
            return len(self._samples)


__all__ = ["MovementTracker", "MovementSample", "MovementMetrics"]
