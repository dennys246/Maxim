"""Motion Tracking behavior - Follow moving objects.

This behavior tracks objects that are moving across the visual field,
enabling smooth pursuit of dynamic targets.
"""

from __future__ import annotations

import time
from collections import defaultdict

from maxim.default_network.behaviors.base import Behavior, BehaviorState
from maxim.default_network.messages import ActionProposal


class MotionTracking(Behavior):
    """Track objects that are moving in the visual field.

    This behavior:
    1. Maintains position history for each track_id
    2. Calculates velocity from recent positions
    3. Proposes tracking objects with velocity above threshold

    Useful for following objects in motion like a ball or walking person.

    Attributes:
        velocity_threshold: Minimum pixels/second to trigger (default 50).
        history_seconds: How long to keep position history (default 0.5s).
        prediction_seconds: How far ahead to predict position (default 0.1s).
    """

    name = "motion"
    base_priority = 0.7
    cooldown_seconds = 0.1  # Fast updates for smooth tracking

    def __init__(
        self,
        velocity_threshold: float = 50.0,
        history_seconds: float = 0.5,
        prediction_seconds: float = 0.1,
    ) -> None:
        """Initialize motion tracking.

        Args:
            velocity_threshold: Minimum velocity (px/s) to track.
            history_seconds: Duration to keep position history.
            prediction_seconds: How far ahead to predict for leading.
        """
        super().__init__()
        self.velocity_threshold = velocity_threshold
        self.history_seconds = history_seconds
        self.prediction_seconds = prediction_seconds

        # track_id -> list of (timestamp, x, y)
        self._position_history: dict[int, list[tuple[float, float, float]]] = defaultdict(list)

    def evaluate(
        self,
        detections: list[dict],
        state: BehaviorState,
    ) -> ActionProposal | None:
        """Evaluate detections and propose tracking fastest moving object."""
        if not self.can_activate():
            return None

        now = time.time()
        cutoff = now - self.history_seconds

        # Update position history
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            bbox = det.get("bbox_xyxy") or det.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue

            # Center position
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            # Add to history
            history = self._position_history[track_id]
            history.append((now, cx, cy))

            # Prune old entries
            self._position_history[track_id] = [(t, x, y) for t, x, y in history if t > cutoff]

        # Find fastest moving object
        best_target = None
        best_velocity = self.velocity_threshold
        best_prediction = None

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            history = self._position_history.get(track_id, [])
            if len(history) < 2:
                continue

            # Calculate velocity from oldest to newest
            t0, x0, y0 = history[0]
            t1, x1, y1 = history[-1]
            dt = t1 - t0

            if dt < 0.05:  # Need at least 50ms of history
                continue

            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
            velocity = (vx**2 + vy**2) ** 0.5

            if velocity > best_velocity:
                best_velocity = velocity
                best_target = det
                # Predict future position
                best_prediction = (
                    x1 + vx * self.prediction_seconds,
                    y1 + vy * self.prediction_seconds,
                )

        if best_target is None or best_prediction is None:
            return None

        # Scale priority with velocity (faster = higher priority)
        priority_scale = min(best_velocity / 200.0, 1.0)

        return self._create_proposal(
            action_type="track",
            target=best_prediction,
            priority_scale=priority_scale,
            confidence=best_target.get("conf", 0.5),
            track_id=best_target.get("track_id"),
            class_id=best_target.get("class_id"),
            velocity=best_velocity,
            velocity_vector=(vx, vy),
        )

    def reset(self) -> None:
        """Reset motion history."""
        super().reset()
        self._position_history.clear()

    def cleanup_stale_tracks(self, active_track_ids: set[int]) -> None:
        """Remove history for tracks no longer visible.

        Args:
            active_track_ids: Set of currently visible track IDs.
        """
        stale = [tid for tid in self._position_history if tid not in active_track_ids]
        for tid in stale:
            del self._position_history[tid]


__all__ = ["MotionTracking"]
