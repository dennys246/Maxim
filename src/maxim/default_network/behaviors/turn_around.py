"""TurnAround behavior - Rotate body when head reaches yaw limits.

When the head is at its horizontal (yaw) limit and there's something
interesting in the direction it can't turn further, this behavior
proposes rotating the body to bring that area into view.

This only applies to horizontal movement - pitch limits don't trigger
this since body rotation is around the vertical axis.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING

from maxim.default_network.behaviors.base import Behavior, BehaviorState
from maxim.default_network.messages import ActionProposal

if TYPE_CHECKING:
    from maxim.salience import ThreadSafeNoveltyTracker

logger = logging.getLogger(__name__)


class TurnAround(Behavior):
    """Rotate body when head is at yaw limits with interesting content beyond.

    This behavior:
    1. Monitors head yaw position
    2. Detects when near the yaw limit (left or right)
    3. Checks for interesting detections at the edge of view in that direction
    4. Proposes body rotation to bring that area into view

    The turn is slow and deliberate (5 ± 1 seconds) to appear natural.

    Attributes:
        yaw_threshold: Fraction of max yaw at which to consider turning (default 0.85).
        edge_threshold: How close to frame edge a detection must be (default 0.15 = 15% from edge).
        turn_angle: How far to rotate the body in degrees (default 90).
        base_duration: Base duration for the turn in seconds (default 5.0).
        duration_jitter: Random variation in duration (default 1.0).
    """

    name = "turn_around"
    base_priority = 0.3  # Lower than tracking, but can override idle behaviors
    cooldown_seconds = 10.0  # Don't turn too frequently

    def __init__(
        self,
        novelty_tracker: "ThreadSafeNoveltyTracker | None" = None,
        yaw_threshold: float = 0.85,
        edge_threshold: float = 0.15,
        turn_angle: float = 90.0,
        base_duration: float = 5.0,
        duration_jitter: float = 1.0,
        max_yaw: float = 55.0,
        image_width: int = 640,
    ) -> None:
        """Initialize the turn around behavior.

        Args:
            novelty_tracker: Shared novelty tracker for interest scoring.
            yaw_threshold: Fraction of max yaw to trigger (0.85 = 85% of limit).
            edge_threshold: Detection must be within this fraction of frame edge.
            turn_angle: Degrees to rotate body (will match direction of interest).
            base_duration: Base seconds for the turn movement.
            duration_jitter: Random ± seconds added to duration.
            max_yaw: Maximum yaw angle in degrees (should match workspace limits).
            image_width: Image width for edge detection calculations.
        """
        super().__init__()
        self._novelty_tracker = novelty_tracker
        self.yaw_threshold = yaw_threshold
        self.edge_threshold = edge_threshold
        self.turn_angle = turn_angle
        self.base_duration = base_duration
        self.duration_jitter = duration_jitter
        self.max_yaw = max_yaw
        self.image_width = image_width

        # Will be set by DefaultNetwork with current head position
        self._head_yaw: float | None = None

        # Interest classes that can trigger turn around
        self._interests: frozenset[int] = frozenset()

    def set_novelty_tracker(self, tracker: "ThreadSafeNoveltyTracker") -> None:
        """Set or update the novelty tracker."""
        self._novelty_tracker = tracker

    def set_head_yaw(self, yaw: float) -> None:
        """Update current head yaw position (called by DefaultNetwork)."""
        self._head_yaw = yaw

    def set_interests(self, interests: frozenset[int]) -> None:
        """Update the set of interesting class IDs."""
        self._interests = interests

    def evaluate(
        self,
        detections: list[dict],
        state: BehaviorState,
    ) -> ActionProposal | None:
        """Evaluate if we should turn around to see something interesting."""
        if not self.can_activate():
            return None

        if self._head_yaw is None:
            return None

        # SANITY CHECK: If yaw is outside physical limits (±90°), the reading
        # is likely corrupted (e.g., world-frame yaw). Don't trigger turn based
        # on invalid data. The head physically cannot rotate more than ±65°.
        if abs(self._head_yaw) > 90.0:
            logger.debug("TurnAround: ignoring invalid yaw=%.1f (outside ±90°)", self._head_yaw)
            return None

        # Check if head is near horizontal limit
        yaw_fraction = abs(self._head_yaw) / self.max_yaw if self.max_yaw > 0 else 0
        if yaw_fraction < self.yaw_threshold:
            return None  # Head has room to move, no need to turn body

        # Determine which direction we're limited in
        # Positive yaw = head turned left, negative = head turned right
        looking_left = self._head_yaw > 0
        looking_right = self._head_yaw < 0

        # Find interesting detections near the edge in the limited direction
        edge_min_left = 0
        edge_max_left = self.image_width * self.edge_threshold
        edge_min_right = self.image_width * (1 - self.edge_threshold)
        edge_max_right = self.image_width

        interesting_on_edge = False
        best_detection = None
        best_score = 0.0

        for det in detections:
            # Check if detection is interesting
            class_id = det.get("class_id")
            is_interesting = class_id is not None and class_id in state.interests

            # Also consider high novelty as interesting
            track_id = det.get("track_id")
            novelty = 0.0
            if self._novelty_tracker and track_id is not None:
                novelty = self._novelty_tracker.get_novelty(track_id, class_id=class_id)
                if novelty > 1.5:
                    is_interesting = True

            if not is_interesting:
                continue

            # Get detection position
            bbox = det.get("bbox_xyxy") or det.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            center_u = (bbox[0] + bbox[2]) / 2

            # Check if it's on the edge in the direction we can't look
            on_left_edge = edge_min_left <= center_u <= edge_max_left
            on_right_edge = edge_min_right <= center_u <= edge_max_right

            # We want to turn if:
            # - Looking left (positive yaw at limit) AND something on left edge
            # - Looking right (negative yaw at limit) AND something on right edge
            if (looking_left and on_left_edge) or (looking_right and on_right_edge):
                score = det.get("conf", 0.5) + novelty * 0.3
                if score > best_score:
                    best_score = score
                    best_detection = det
                    interesting_on_edge = True

        if not interesting_on_edge or best_detection is None:
            return None

        # Calculate turn direction and duration
        # Turn in the direction we're looking (body rotates to bring that area to center)
        turn_direction = 1 if looking_left else -1  # Positive = counterclockwise
        turn_amount = self.turn_angle * turn_direction

        # Add random jitter to duration for natural feel
        duration = self.base_duration + random.uniform(-self.duration_jitter, self.duration_jitter)
        duration = max(3.0, duration)  # At least 3 seconds

        logger.info(
            "TurnAround triggered: yaw=%.1f (%.0f%% of limit), "
            "interesting detection at edge, turning %.0f° over %.1fs",
            self._head_yaw, yaw_fraction * 100, turn_amount, duration
        )

        return self._create_proposal(
            action_type="turn_around",
            target=None,  # No pixel target for body rotation
            priority_scale=1.0,
            confidence=min(0.9, best_score),
            turn_angle=turn_amount,
            duration=duration,
            reason="interesting_at_edge",
            detection_class=best_detection.get("class_id"),
        )


__all__ = ["TurnAround"]