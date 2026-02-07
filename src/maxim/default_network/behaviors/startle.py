"""Startle Response behavior - Quick reaction to sudden appearances.

This behavior triggers a rapid orienting response when something
suddenly appears in the peripheral vision, mimicking the biological
startle reflex.
"""

from __future__ import annotations

import time

from maxim.default_network.behaviors.base import Behavior, BehaviorState
from maxim.default_network.messages import ActionProposal


class StartleResponse(Behavior):
    """Quick look at objects that suddenly appear in peripheral vision.

    This behavior:
    1. Tracks which track_ids have been seen before
    2. Detects new objects appearing in peripheral regions
    3. Triggers high-priority rapid look toward the new object

    Has very high priority because startle responses are reflexive.

    Attributes:
        peripheral_threshold: Fraction of frame edge to consider peripheral.
        appearance_window: Seconds for an appearance to be "sudden".
        min_confidence: Minimum detection confidence.
    """

    name = "startle"
    base_priority = 0.95  # Very high - reflexive response
    cooldown_seconds = 2.0  # Long cooldown to prevent constant startling

    def __init__(
        self,
        peripheral_threshold: float = 0.7,
        appearance_window: float = 0.3,
        min_confidence: float = 0.5,
        frame_size: tuple[int, int] = (640, 480),
    ) -> None:
        """Initialize startle response.

        Args:
            peripheral_threshold: Fraction from center to be peripheral.
            appearance_window: Max age (seconds) for sudden appearance.
            min_confidence: Minimum detection confidence.
            frame_size: (width, height) of video frame.
        """
        super().__init__()
        self.peripheral_threshold = peripheral_threshold
        self.appearance_window = appearance_window
        self.min_confidence = min_confidence
        self.frame_size = frame_size

        # track_id -> first_seen_timestamp
        self._first_seen: dict[int, float] = {}
        # track_id -> first_seen_position
        self._first_position: dict[int, tuple[float, float]] = {}

    def evaluate(
        self,
        detections: list[dict],
        state: BehaviorState,
    ) -> ActionProposal | None:
        """Detect sudden appearances and propose quick look."""
        if not self.can_activate():
            return None

        now = time.time()
        frame_w, frame_h = self.frame_size
        center_x, center_y = frame_w / 2, frame_h / 2

        # Thresholds for peripheral region
        periph_x = center_x * self.peripheral_threshold
        periph_y = center_y * self.peripheral_threshold

        startle_target = None
        startle_urgency = 0.0

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            conf = det.get("conf", 0)
            if conf < self.min_confidence:
                continue

            bbox = det.get("bbox_xyxy") or det.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue

            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            # Check if this is a new track
            if track_id not in self._first_seen:
                self._first_seen[track_id] = now
                self._first_position[track_id] = (cx, cy)

                # Check if it appeared in peripheral region
                dist_from_center_x = abs(cx - center_x)
                dist_from_center_y = abs(cy - center_y)

                is_peripheral = (
                    dist_from_center_x > periph_x or
                    dist_from_center_y > periph_y
                )

                if is_peripheral:
                    # Calculate urgency based on how peripheral
                    urgency = max(
                        dist_from_center_x / center_x,
                        dist_from_center_y / center_y,
                    )

                    if urgency > startle_urgency:
                        startle_urgency = urgency
                        startle_target = det

            else:
                # Check if it's still within the sudden window
                age = now - self._first_seen[track_id]
                if age > self.appearance_window:
                    continue

                # Check if it appeared in peripheral and is still new
                first_x, first_y = self._first_position[track_id]
                dist_from_center_x = abs(first_x - center_x)
                dist_from_center_y = abs(first_y - center_y)

                is_peripheral = (
                    dist_from_center_x > periph_x or
                    dist_from_center_y > periph_y
                )

                if is_peripheral:
                    urgency = max(
                        dist_from_center_x / center_x,
                        dist_from_center_y / center_y,
                    )
                    # Decay urgency with age
                    urgency *= (1.0 - age / self.appearance_window)

                    if urgency > startle_urgency:
                        startle_urgency = urgency
                        startle_target = det

        # Clean up old entries periodically
        if len(self._first_seen) > 100:
            self._cleanup(now)

        if startle_target is None:
            return None

        bbox = startle_target.get("bbox_xyxy") or startle_target.get("bbox")
        target_u = (bbox[0] + bbox[2]) / 2
        target_v = (bbox[1] + bbox[3]) / 2

        return self._create_proposal(
            action_type="look_at",
            target=(target_u, target_v),
            priority_scale=min(startle_urgency, 1.0),
            confidence=startle_target.get("conf", 0.5),
            track_id=startle_target.get("track_id"),
            class_id=startle_target.get("class_id"),
            startle_urgency=startle_urgency,
            peripheral=True,
        )

    def _cleanup(self, now: float) -> None:
        """Remove old entries from tracking dicts."""
        max_age = 30.0  # Keep entries for 30 seconds
        cutoff = now - max_age

        old_ids = [
            tid for tid, t in self._first_seen.items()
            if t < cutoff
        ]
        for tid in old_ids:
            del self._first_seen[tid]
            self._first_position.pop(tid, None)

    def reset(self) -> None:
        """Reset tracking state."""
        super().reset()
        self._first_seen.clear()
        self._first_position.clear()

    def set_frame_size(self, width: int, height: int) -> None:
        """Update frame size for peripheral calculations.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        self.frame_size = (width, height)


__all__ = ["StartleResponse"]
