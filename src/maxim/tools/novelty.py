"""Novelty tracking data structures.

Dataclasses used by :class:`NoveltyTrackTool` (and available for other agents)
to represent per-detection novelty state and scoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class NoveltyRecord:
    """Record of a detection's novelty state."""

    track_id: int
    class_id: int
    class_name: str
    first_seen: float
    last_seen: float
    seen_count: int = 1
    total_frames: int = 1
    bbox_history: list[tuple[float, float, float, float]] = field(default_factory=list)
    confidence_history: list[float] = field(default_factory=list)

    @property
    def novelty_score(self) -> float:
        """Compute novelty: decays as object is seen more often."""
        # Novelty decays with exposure: 1.0 for new, approaches 0 with repeated sightings
        decay_factor = 0.85  # How quickly novelty decays per sighting
        return decay_factor ** (self.seen_count - 1)

    @property
    def persistence(self) -> float:
        """How consistently this object appears (0-1)."""
        if self.total_frames == 0:
            return 0.0
        return self.seen_count / self.total_frames

    @property
    def age_seconds(self) -> float:
        """Time since first seen."""
        return time.time() - self.first_seen


@dataclass
class NoveltyInfo:
    """Novelty information for a single detection."""

    track_id: int
    class_id: int
    class_name: str
    novelty_score: float  # 0.0 (familiar) to 1.0 (novel)
    age_seconds: float
    seen_count: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    confidence: float
    is_new: bool  # True if this is the first time seeing this track_id
