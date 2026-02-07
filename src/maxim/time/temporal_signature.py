"""Temporal signature for memory fingerprinting.

Captures phase across multiple biological rhythms for temporal similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


def circular_distance(a: float, b: float) -> float:
    """Compute circular distance between two phase values.

    Handles wrap-around: 0.95 is close to 0.05.

    Args:
        a: First phase value (0.0-1.0)
        b: Second phase value (0.0-1.0)

    Returns:
        Distance (0.0-0.5)
    """
    diff = abs(a - b)
    return min(diff, 1.0 - diff)


@dataclass(frozen=True, slots=True)
class TemporalSignature:
    """Temporal fingerprint for a memory.

    All phases are normalized to [0.0, 1.0) representing position
    within each cycle. This enables efficient binning and similarity
    calculations across different time scales.

    Attributes:
        timestamp: Unix timestamp (absolute reference)
        circadian_phase: 0.0-1.0 (midnight=0, noon=0.5)
        weekly_phase: 0.0-1.0 (Monday 00:00=0, Sunday 23:59≈1.0)
        monthly_phase: 0.0-1.0 (1st=0, ~15th=0.5, 28-31st≈1.0)
        annual_phase: 0.0-1.0 (Jan 1=0, July 1≈0.5, Dec 31≈1.0)
    """

    timestamp: float
    circadian_phase: float
    weekly_phase: float
    monthly_phase: float
    annual_phase: float

    @classmethod
    def from_timestamp(cls, ts: float) -> TemporalSignature:
        """Create temporal signature from Unix timestamp."""
        dt = datetime.fromtimestamp(ts)

        # Circadian: fraction of day (0 = midnight, 0.5 = noon)
        seconds_in_day = dt.hour * 3600 + dt.minute * 60 + dt.second
        circadian = seconds_in_day / 86400.0

        # Weekly: fraction of week (0 = Monday 00:00)
        seconds_in_week = dt.weekday() * 86400 + seconds_in_day
        weekly = seconds_in_week / (7 * 86400)

        # Monthly: fraction of month (approximated as day/31)
        monthly = (dt.day - 1) / 31.0

        # Annual: day of year / 365.25
        day_of_year = dt.timetuple().tm_yday
        annual = (day_of_year - 1) / 365.25

        return cls(
            timestamp=ts,
            circadian_phase=circadian,
            weekly_phase=weekly,
            monthly_phase=monthly,
            annual_phase=annual,
        )

    @classmethod
    def now(cls) -> TemporalSignature:
        """Create temporal signature for current time."""
        import time

        return cls.from_timestamp(time.time())

    def similarity(
        self,
        other: TemporalSignature,
        weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> float:
        """Calculate temporal similarity (0.0-1.0) with phase-aware distance.

        Uses circular distance for proper phase comparison.

        Args:
            other: Another temporal signature
            weights: (circadian, weekly, monthly, annual) weights

        Returns:
            Similarity score (0.0-1.0, higher = more similar)
        """
        distances = [
            circular_distance(self.circadian_phase, other.circadian_phase),
            circular_distance(self.weekly_phase, other.weekly_phase),
            circular_distance(self.monthly_phase, other.monthly_phase),
            circular_distance(self.annual_phase, other.annual_phase),
        ]

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_dist = sum(d * w for d, w in zip(distances, weights)) / total_weight
        # Max distance is 0.5 (opposite phase), scale to 0-1
        return 1.0 - (weighted_dist * 2)

    def to_bins(self) -> tuple[int, int, int, int]:
        """Convert to bin indices for SCN indexing.

        Returns:
            (hour_bin, day_bin, week_bin, month_bin)
        """
        return (
            int(self.circadian_phase * 24) % 24,  # Hour bin (0-23)
            int(self.weekly_phase * 7) % 7,  # Day bin (0-6)
            int(self.monthly_phase * 4) % 4,  # Week-of-month bin (0-3)
            int(self.annual_phase * 12) % 12,  # Month bin (0-11)
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp,
            "circadian_phase": self.circadian_phase,
            "weekly_phase": self.weekly_phase,
            "monthly_phase": self.monthly_phase,
            "annual_phase": self.annual_phase,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TemporalSignature:
        """Deserialize from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            circadian_phase=data["circadian_phase"],
            weekly_phase=data["weekly_phase"],
            monthly_phase=data["monthly_phase"],
            annual_phase=data["annual_phase"],
        )


__all__ = ["TemporalSignature", "circular_distance"]