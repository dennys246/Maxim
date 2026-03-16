"""Scene Context Detector - Detect significant scene changes.

When the scene significantly changes (robot turned around, entered new room,
many objects appeared/disappeared), the gaze system should trigger a rapid
scene scan to understand the new environment.

This module tracks scene state and detects when major changes occur.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SceneContextConfig:
    """Configuration for scene context detection.

    Attributes:
        change_threshold: Fraction of scene change to trigger (0-1).
        min_detections_for_comparison: Minimum detections for valid comparison.
        scene_stability_seconds: Time before scene is considered stable.
        histogram_memory: Number of class histograms to remember.
        position_change_threshold: Head position change to trigger scan.
    """
    change_threshold: float = 0.4  # 40% change triggers scene scan
    min_detections_for_comparison: int = 2
    scene_stability_seconds: float = 2.0
    histogram_memory: int = 5
    position_change_threshold: float = 30.0  # degrees yaw change


@dataclass
class SceneSnapshot:
    """Snapshot of scene state at a point in time."""
    timestamp: float
    class_histogram: dict[int, int]  # class_id -> count
    total_detections: int
    track_ids: set[int]
    head_yaw: float | None = None
    head_pitch: float | None = None


class SceneContextDetector:
    """Detect significant scene changes to trigger exploration.

    Monitors:
    1. Class histogram changes (what types of objects are visible)
    2. Track ID turnover (how many objects appeared/disappeared)
    3. Head position changes (robot turned around)

    Example:
        detector = SceneContextDetector()

        # Each frame
        changed = detector.update(
            detections=detections,
            head_yaw=current_yaw,
        )

        if changed:
            trigger_scene_scan()
    """

    def __init__(self, config: SceneContextConfig | None = None) -> None:
        """Initialize scene context detector.

        Args:
            config: Detector configuration.
        """
        self.config = config or SceneContextConfig()

        # Scene history
        self._snapshots: list[SceneSnapshot] = []
        self._last_stable_snapshot: SceneSnapshot | None = None

        # State
        self._scene_changed_at: float = 0.0
        self._is_scanning: bool = False
        self._last_head_yaw: float | None = None

    def update(
        self,
        detections: list[dict[str, Any]],
        head_yaw: float | None = None,
        head_pitch: float | None = None,
    ) -> bool:
        """Check if scene has significantly changed.

        Args:
            detections: Current YOLO detections.
            head_yaw: Current head yaw angle (degrees).
            head_pitch: Current head pitch angle (degrees).

        Returns:
            True if scene changed enough to warrant rapid scan.
        """
        now = time.time()

        # Build current snapshot
        current = self._build_snapshot(detections, now, head_yaw, head_pitch)

        # Check for head position change (rapid turn)
        if self._check_head_change(head_yaw):
            logger.info("SceneContext: head position changed significantly")
            self._scene_changed_at = now
            self._last_head_yaw = head_yaw
            self._last_stable_snapshot = current
            return True

        # Not enough detections for comparison
        if current.total_detections < self.config.min_detections_for_comparison:
            self._snapshots.append(current)
            self._trim_snapshots()
            return False

        # Compare to stable snapshot
        if self._last_stable_snapshot is not None:
            change_ratio = self._compute_change(self._last_stable_snapshot, current)

            if change_ratio >= self.config.change_threshold:
                logger.info(
                    "SceneContext: scene changed %.0f%% (threshold %.0f%%)",
                    change_ratio * 100,
                    self.config.change_threshold * 100,
                )
                self._scene_changed_at = now
                self._last_stable_snapshot = current
                self._is_scanning = True
                return True

        # Check if scene has stabilized
        if self._is_scanning:
            time_since_change = now - self._scene_changed_at
            if time_since_change > self.config.scene_stability_seconds:
                self._is_scanning = False
                logger.debug("SceneContext: scene stabilized after %.1fs", time_since_change)

        # Update stable snapshot if first or scene is stable
        if self._last_stable_snapshot is None:
            self._last_stable_snapshot = current

        self._snapshots.append(current)
        self._trim_snapshots()
        self._last_head_yaw = head_yaw

        return False

    def _build_snapshot(
        self,
        detections: list[dict[str, Any]],
        timestamp: float,
        head_yaw: float | None,
        head_pitch: float | None,
    ) -> SceneSnapshot:
        """Build a scene snapshot from detections."""
        histogram: dict[int, int] = {}
        track_ids: set[int] = set()

        for det in detections:
            class_id = det.get("class_id", -1)
            histogram[class_id] = histogram.get(class_id, 0) + 1

            track_id = det.get("track_id")
            if track_id is not None:
                track_ids.add(track_id)

        return SceneSnapshot(
            timestamp=timestamp,
            class_histogram=histogram,
            total_detections=len(detections),
            track_ids=track_ids,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
        )

    def _compute_change(self, old: SceneSnapshot, new: SceneSnapshot) -> float:
        """Compute change ratio between two snapshots.

        Returns value in [0, 1] where 0 = identical, 1 = completely different.
        """
        # Histogram difference
        all_classes = set(old.class_histogram) | set(new.class_histogram)
        if not all_classes:
            return 0.0

        hist_diff = sum(
            abs(old.class_histogram.get(c, 0) - new.class_histogram.get(c, 0))
            for c in all_classes
        )
        total_count = old.total_detections + new.total_detections
        hist_change = hist_diff / max(total_count, 1)

        # Track ID turnover (how many objects are new)
        if old.track_ids and new.track_ids:
            common = len(old.track_ids & new.track_ids)
            total = len(old.track_ids | new.track_ids)
            track_change = 1.0 - (common / max(total, 1))
        else:
            track_change = 0.5  # Unknown

        # Combined score (weight histogram more)
        return hist_change * 0.6 + track_change * 0.4

    def _check_head_change(self, current_yaw: float | None) -> bool:
        """Check if head has moved significantly."""
        if current_yaw is None or self._last_head_yaw is None:
            return False

        delta = abs(current_yaw - self._last_head_yaw)

        # Handle wrap-around (e.g., -170 to 170)
        if delta > 180:
            delta = 360 - delta

        return delta >= self.config.position_change_threshold

    def _trim_snapshots(self) -> None:
        """Keep only recent snapshots."""
        max_keep = self.config.histogram_memory
        if len(self._snapshots) > max_keep:
            self._snapshots = self._snapshots[-max_keep:]

    def get_scene_age(self) -> float:
        """Get time since last scene change.

        Returns:
            Seconds since scene changed, or inf if never.
        """
        if self._scene_changed_at == 0:
            return float('inf')
        return time.time() - self._scene_changed_at

    def is_scanning(self) -> bool:
        """Whether currently in post-change scan mode."""
        return self._is_scanning

    def force_scene_change(self) -> None:
        """Force a scene change (e.g., after turn_around completes)."""
        self._scene_changed_at = time.time()
        self._is_scanning = True
        self._last_stable_snapshot = None
        logger.info("SceneContext: forced scene change (scan mode)")

    def reset(self) -> None:
        """Reset detector state."""
        self._snapshots.clear()
        self._last_stable_snapshot = None
        self._scene_changed_at = 0.0
        self._is_scanning = False
        self._last_head_yaw = None

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics."""
        return {
            "is_scanning": self._is_scanning,
            "scene_age_seconds": self.get_scene_age(),
            "snapshots_count": len(self._snapshots),
            "last_head_yaw": self._last_head_yaw,
            "classes_in_scene": list(self._last_stable_snapshot.class_histogram.keys())
                if self._last_stable_snapshot else [],
        }


__all__ = [
    "SceneContextConfig",
    "SceneSnapshot",
    "SceneContextDetector",
]
