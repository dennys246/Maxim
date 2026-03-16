"""Movement Detector - Track object motion for salience boosting.

Humans reflexively attend to moving objects in their visual field,
especially in the periphery. This module tracks object motion and
provides movement scores that can boost object salience.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MovementConfig:
    """Configuration for movement detection.

    Attributes:
        velocity_normalization: Pixels per frame for max velocity (1.0 score).
        decay_seconds: How fast movement score decays when object stops.
        min_movement_threshold: Minimum pixels to register as movement.
        max_entries: Maximum tracked objects before cleanup.
        peripheral_boost: Extra boost for movement in peripheral vision.
        peripheral_threshold: Fraction from center to be considered peripheral.
    """
    velocity_normalization: float = 50.0  # pixels/frame for max score
    decay_seconds: float = 0.5  # Fast decay when stopped
    min_movement_threshold: float = 3.0  # Ignore tiny movements
    max_entries: int = 200
    peripheral_boost: float = 0.3  # Extra salience for peripheral motion
    peripheral_threshold: float = 0.6  # 60% from center = peripheral


@dataclass
class ObjectMotion:
    """Motion state for a tracked object."""
    last_position: tuple[float, float]
    last_velocity: float  # pixels/frame
    movement_score: float  # 0-1 smoothed score
    last_update: float
    frames_stationary: int = 0


class MovementDetector:
    """Detect and score object movement for attention.

    Tracks object positions across frames and computes movement scores
    that can be used to boost salience of moving objects.

    Example:
        detector = MovementDetector()

        # Each frame, update with detections
        scores = detector.update(detections, frame_center=(320, 240))

        # scores is dict mapping track_id -> movement_score (0-1)
        # Higher scores indicate more/faster movement
    """

    def __init__(self, config: MovementConfig | None = None) -> None:
        """Initialize movement detector.

        Args:
            config: Movement detection configuration.
        """
        self.config = config or MovementConfig()
        self._objects: dict[int, ObjectMotion] = {}
        self._frame_size: tuple[float, float] = (640.0, 480.0)

    def set_frame_size(self, width: float, height: float) -> None:
        """Set frame dimensions for peripheral calculations.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self._frame_size = (width, height)

    def update(
        self,
        detections: list[dict[str, Any]],
        frame_center: tuple[float, float] | None = None,
    ) -> dict[int, float]:
        """Update motion tracking and return movement scores.

        Args:
            detections: List of detection dicts with track_id and bbox_xyxy.
            frame_center: Optional (x, y) center for peripheral boost.

        Returns:
            Dict mapping track_id to movement score (0-1).
            Higher scores = more/faster movement.
        """
        now = time.time()
        scores: dict[int, float] = {}

        if frame_center is None:
            frame_center = (self._frame_size[0] / 2, self._frame_size[1] / 2)

        seen_ids: set[int] = set()

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            seen_ids.add(track_id)

            # Get bounding box center
            bbox = det.get("bbox_xyxy") or det.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue

            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            current_pos = (cx, cy)

            if track_id in self._objects:
                # Existing object - compute movement
                motion = self._objects[track_id]
                dx = cx - motion.last_position[0]
                dy = cy - motion.last_position[1]
                distance = math.sqrt(dx * dx + dy * dy)

                if distance < self.config.min_movement_threshold:
                    # Stationary - decay score
                    motion.frames_stationary += 1
                    dt = now - motion.last_update
                    decay = math.exp(-dt / self.config.decay_seconds)
                    motion.movement_score *= decay
                    motion.last_velocity = 0.0
                else:
                    # Moving - update score
                    motion.frames_stationary = 0
                    velocity = distance  # pixels per frame
                    normalized_velocity = min(velocity / self.config.velocity_normalization, 1.0)

                    # Smooth the score (exponential moving average)
                    alpha = 0.3  # Responsiveness
                    motion.movement_score = (
                        alpha * normalized_velocity +
                        (1 - alpha) * motion.movement_score
                    )
                    motion.last_velocity = velocity

                motion.last_position = current_pos
                motion.last_update = now

            else:
                # New object - initialize with moderate movement score
                # New things appearing are somewhat salient
                self._objects[track_id] = ObjectMotion(
                    last_position=current_pos,
                    last_velocity=0.0,
                    movement_score=0.3,  # Initial score for new objects
                    last_update=now,
                )

            # Get score with peripheral boost
            motion = self._objects[track_id]
            score = motion.movement_score

            # Apply peripheral boost
            if score > 0.1:  # Only boost if there's meaningful movement
                peripheral_factor = self._get_peripheral_factor(current_pos, frame_center)
                score = min(1.0, score + peripheral_factor * self.config.peripheral_boost)

            scores[track_id] = score

        # Decay scores for objects not seen this frame
        for track_id, motion in list(self._objects.items()):
            if track_id not in seen_ids:
                dt = now - motion.last_update
                decay = math.exp(-dt / self.config.decay_seconds)
                motion.movement_score *= decay
                motion.last_update = now

                # Still report decaying scores
                if motion.movement_score > 0.05:
                    scores[track_id] = motion.movement_score

        # Cleanup old entries
        if len(self._objects) > self.config.max_entries:
            self._cleanup(now)

        return scores

    def _get_peripheral_factor(
        self,
        position: tuple[float, float],
        center: tuple[float, float],
    ) -> float:
        """Compute how peripheral a position is (0 = center, 1 = edge).

        Args:
            position: (x, y) position to check.
            center: (x, y) center of frame.

        Returns:
            Peripheral factor from 0 (center) to 1 (edge/corner).
        """
        dx = abs(position[0] - center[0]) / (self._frame_size[0] / 2)
        dy = abs(position[1] - center[1]) / (self._frame_size[1] / 2)
        # Normalized distance from center (0-1.414 for corners)
        dist = math.sqrt(dx * dx + dy * dy)

        # Apply threshold - only boost beyond peripheral_threshold
        if dist < self.config.peripheral_threshold:
            return 0.0

        # Scale to 0-1 range beyond threshold
        return min(1.0, (dist - self.config.peripheral_threshold) /
                   (1.0 - self.config.peripheral_threshold))

    def get_movement_score(self, track_id: int) -> float:
        """Get current movement score for a track ID.

        Args:
            track_id: The track ID to look up.

        Returns:
            Movement score (0-1), or 0 if not tracked.
        """
        motion = self._objects.get(track_id)
        if motion is None:
            return 0.0
        return motion.movement_score

    def get_top_movers(self, n: int = 5) -> list[tuple[int, float]]:
        """Get the N objects with highest movement scores.

        Args:
            n: Number of objects to return.

        Returns:
            List of (track_id, score) tuples, sorted by score descending.
        """
        items = [
            (tid, motion.movement_score)
            for tid, motion in self._objects.items()
            if motion.movement_score > 0.1
        ]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]

    def _cleanup(self, now: float) -> None:
        """Remove stale entries to prevent memory growth.

        Args:
            now: Current timestamp.
        """
        # Remove objects not seen for 10+ seconds or with very low scores
        cutoff = now - 10.0
        to_remove = [
            tid for tid, motion in self._objects.items()
            if motion.last_update < cutoff or motion.movement_score < 0.01
        ]
        for tid in to_remove:
            del self._objects[tid]

        logger.debug("MovementDetector cleanup: removed %d entries", len(to_remove))

    def reset(self) -> None:
        """Reset all tracking state."""
        self._objects.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics.

        Returns:
            Dict with tracking statistics.
        """
        if not self._objects:
            return {
                "tracked_count": 0,
                "avg_movement_score": 0.0,
                "max_movement_score": 0.0,
            }

        scores = [m.movement_score for m in self._objects.values()]
        return {
            "tracked_count": len(self._objects),
            "avg_movement_score": sum(scores) / len(scores),
            "max_movement_score": max(scores),
            "moving_count": sum(1 for s in scores if s > 0.3),
        }


__all__ = [
    "MovementConfig",
    "MovementDetector",
    "ObjectMotion",
]
