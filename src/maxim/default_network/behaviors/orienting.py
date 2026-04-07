"""Orienting Response behavior - Look at novel or moving objects.

This behavior directs attention toward novel objects that appear
in the visual field. Novelty decays over time, so familiar objects
are deprioritized. Moving objects also attract attention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from maxim.default_network.behaviors.base import Behavior, BehaviorState
from maxim.default_network.messages import ActionProposal

if TYPE_CHECKING:
    from maxim.salience import ThreadSafeNoveltyTracker

logger = logging.getLogger(__name__)


class OrientingResponse(Behavior):
    """Look at novel or moving objects in the visual field.

    This behavior:
    1. Scans detections for objects above the salience threshold
    2. Combines novelty and movement scores for total salience
    3. Selects the most salient object
    4. Proposes looking at it

    The priority scales with salience - very novel/moving objects get higher priority.

    Attributes:
        novelty_threshold: Minimum novelty score to trigger (default 1.2).
        min_confidence: Minimum detection confidence (default 0.4).
        movement_weight: How much movement contributes to salience (default 0.5).
    """

    name = "orienting"
    base_priority = 0.8
    cooldown_seconds = 0.5  # Don't orient more than 2x/sec

    def __init__(
        self,
        novelty_tracker: "ThreadSafeNoveltyTracker | None" = None,
        novelty_threshold: float = 1.2,
        min_confidence: float = 0.4,
        movement_weight: float = 0.5,
    ) -> None:
        """Initialize the orienting response.

        Args:
            novelty_tracker: Shared novelty tracker. If None, novelty is ignored.
            novelty_threshold: Minimum novelty to trigger orienting.
            min_confidence: Minimum detection confidence to consider.
            movement_weight: Weight for movement contribution to salience (0-1).
        """
        super().__init__()
        self._novelty_tracker = novelty_tracker
        self.novelty_threshold = novelty_threshold
        self.min_confidence = min_confidence
        self.movement_weight = movement_weight

    def set_novelty_tracker(self, tracker: "ThreadSafeNoveltyTracker") -> None:
        """Set or update the novelty tracker."""
        self._novelty_tracker = tracker

    def evaluate(
        self,
        detections: list[dict],
        state: BehaviorState,
    ) -> ActionProposal | None:
        """Evaluate detections and propose looking at most salient object."""
        if not self.can_activate():
            return None

        if not detections:
            return None

        # Find most salient detection above threshold
        # Salience = novelty + (movement_score * movement_weight)
        best_target = None
        best_salience = self.novelty_threshold
        best_novelty = 0.0
        best_movement = 0.0

        for det in detections:
            # Check confidence
            conf = det.get("conf", 0)
            if conf < self.min_confidence:
                continue

            # Get novelty score (with class-level modulation)
            track_id = det.get("track_id")
            if self._novelty_tracker and track_id is not None:
                novelty = self._novelty_tracker.get_novelty(track_id, class_id=det.get("class_id"))
            else:
                # Without tracker, use confidence as proxy
                novelty = 1.0 + conf

            # Get movement score (injected by DefaultNetwork)
            movement = det.get("movement_score", 0.0)

            # Combine into total salience
            # Movement can boost an object above threshold even if novelty is low
            salience = novelty + (movement * self.movement_weight)

            if salience > best_salience:
                best_salience = salience
                best_novelty = novelty
                best_movement = movement
                best_target = det

        if best_target is None:
            return None

        # Calculate target center from bounding box
        bbox = best_target.get("bbox_xyxy") or best_target.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            target_u = (bbox[0] + bbox[2]) / 2
            target_v = (bbox[1] + bbox[3]) / 2
        else:
            return None

        # Scale priority with salience (capped at 1.0)
        priority_scale = min(best_salience / 2.0, 1.0)

        # Log when movement contributes significantly
        if best_movement > 0.3:
            logger.debug(
                "OrientingResponse: motion-boosted target track_id=%s (novelty=%.2f, movement=%.2f, salience=%.2f)",
                best_target.get("track_id"),
                best_novelty,
                best_movement,
                best_salience,
            )

        return self._create_proposal(
            action_type="look_at",
            target=(target_u, target_v),
            priority_scale=priority_scale,
            confidence=best_target.get("conf", 0.5),
            track_id=best_target.get("track_id"),
            class_id=best_target.get("class_id"),
            novelty=best_novelty,
            movement=best_movement,
            salience=best_salience,
        )


__all__ = ["OrientingResponse"]
