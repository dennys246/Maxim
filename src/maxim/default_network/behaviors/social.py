"""Social Attention behavior - Preferentially attend to people.

This behavior gives high priority to tracking people, especially
their faces. It maintains tracking of a single person for stability.

Includes habituation: after looking at someone for a few seconds,
attention naturally decreases, creating a "glance, look away, glance back"
pattern similar to human social attention.
"""

from __future__ import annotations

import logging
import time

from maxim.default_network.behaviors.base import Behavior, BehaviorState
from maxim.default_network.messages import ActionProposal

logger = logging.getLogger(__name__)

# COCO class ID for person
PERSON_CLASS_ID = 0


class SocialAttention(Behavior):
    """Preferentially attend to people, especially faces.

    This behavior:
    1. Finds people (class_id=0) in detections
    2. Maintains tracking of a single person for stability
    3. Targets the face/head region if possible
    4. Habituates over time - priority decreases after sustained attention

    Has high priority because social interaction is important, but
    habituation prevents unnatural staring.

    Attributes:
        prefer_faces: If True, target upper portion of person bbox.
        tracking_hysteresis: Bonus confidence for currently tracked person.
        max_continuous_attention: Seconds before habituation fully kicks in.
        habituation_strength: How much priority drops at max habituation (0-1).
        recovery_rate: How fast habituation recovers when looking away (per second).
    """

    name = "social"
    base_priority = 0.9  # Higher than orienting
    # Cooldown should match or exceed typical track movement duration (0.15-0.3s)
    # to prevent new commands from replacing in-flight movements
    cooldown_seconds = 0.35

    def __init__(
        self,
        prefer_faces: bool = True,
        tracking_hysteresis: float = 0.1,
        max_continuous_attention: float = 3.0,
        habituation_strength: float = 0.5,
        recovery_rate: float = 0.3,
    ) -> None:
        """Initialize social attention.

        Args:
            prefer_faces: Target face region instead of body center.
            tracking_hysteresis: Confidence bonus for tracked person.
            max_continuous_attention: Seconds of attention before full habituation.
            habituation_strength: Priority reduction at max habituation (0-1).
            recovery_rate: Habituation recovery per second when not looking.
        """
        super().__init__()
        self.prefer_faces = prefer_faces
        self.tracking_hysteresis = tracking_hysteresis
        self.max_continuous_attention = max_continuous_attention
        self.habituation_strength = habituation_strength
        self.recovery_rate = recovery_rate

        self._tracked_person: int | None = None

        # Habituation state per person
        # Maps track_id -> cumulative dwell time (seconds)
        self._person_dwell: dict[int, float] = {}
        # Maps track_id -> last time we looked at them
        self._person_last_attended: dict[int, float] = {}
        # Last update timestamp for delta calculation
        self._last_update_time: float = time.time()

    def _get_habituation_factor(self, track_id: int) -> float:
        """Get habituation factor for a person.

        Returns a value between (1 - habituation_strength) and 1.0.
        1.0 means fresh attention, lower means habituated.

        Args:
            track_id: The person's track ID.

        Returns:
            Habituation factor to multiply priority by.
        """
        dwell = self._person_dwell.get(track_id, 0.0)
        # Normalize to 0-1 range (0 = fresh, 1 = fully habituated)
        habituation_level = min(dwell / self.max_continuous_attention, 1.0)
        # Convert to factor (1.0 = full priority, lower = reduced)
        factor = 1.0 - (habituation_level * self.habituation_strength)
        return factor

    def _update_habituation(self, current_target_id: int | None) -> None:
        """Update habituation state for all tracked people.

        - Person being looked at: dwell time increases
        - People not being looked at: dwell time decreases (recovery)

        Args:
            current_target_id: Track ID of person currently being attended to, or None.
        """
        now = time.time()
        dt = now - self._last_update_time
        self._last_update_time = now

        # Clamp dt to prevent huge jumps after pause
        dt = min(dt, 1.0)

        # Update dwell for current target (increases)
        if current_target_id is not None:
            current_dwell = self._person_dwell.get(current_target_id, 0.0)
            self._person_dwell[current_target_id] = current_dwell + dt
            self._person_last_attended[current_target_id] = now

        # Recovery for people not being attended
        recovery_amount = dt * self.recovery_rate * self.max_continuous_attention
        to_remove = []
        for track_id in list(self._person_dwell.keys()):
            if track_id != current_target_id:
                self._person_dwell[track_id] -= recovery_amount
                if self._person_dwell[track_id] <= 0:
                    to_remove.append(track_id)

        # Clean up fully recovered entries
        for track_id in to_remove:
            del self._person_dwell[track_id]
            self._person_last_attended.pop(track_id, None)

        # Cleanup stale entries (people not seen for 30+ seconds)
        cutoff = now - 30.0
        stale = [tid for tid, last_time in self._person_last_attended.items() if last_time < cutoff]
        for track_id in stale:
            self._person_dwell.pop(track_id, None)
            self._person_last_attended.pop(track_id, None)

    def evaluate(
        self,
        detections: list[dict],
        state: BehaviorState,
    ) -> ActionProposal | None:
        """Evaluate detections and propose tracking a person."""
        if not self.can_activate():
            logger.debug(
                "SocialAttention on cooldown (%.2fs remaining)", self.cooldown_seconds - self.time_since_activation
            )
            return None

        # Find people in detections
        people = [d for d in detections if d.get("class_id") == PERSON_CLASS_ID]

        if not people:
            if detections:
                # Log what class_ids we ARE seeing
                class_ids = {d.get("class_id") for d in detections}
                logger.debug("SocialAttention: no people found (class_ids seen: %s)", class_ids)
            # Update habituation (no one attended = recovery for all)
            self._update_habituation(None)
            self._tracked_person = None
            return None

        logger.debug("SocialAttention: found %d person(s)", len(people))

        # Prefer currently tracked person for stability, but consider habituation
        target = None
        best_score = -1.0

        for person in people:
            track_id = person.get("track_id")
            conf = person.get("conf", 0.5)
            habituation_factor = self._get_habituation_factor(track_id) if track_id else 1.0

            # Score combines confidence and habituation
            # Currently tracked person gets hysteresis bonus
            score = conf * habituation_factor
            if track_id == self._tracked_person:
                score += self.tracking_hysteresis

            # Spatial hysteresis: centralized bonus for detections near last tracked position
            # This handles track_id fragmentation when head moves
            bbox = person.get("bbox_xyxy") or person.get("bbox", [])
            if len(bbox) >= 4:
                det_cx = (bbox[0] + bbox[2]) / 2
                det_cy = (bbox[1] + bbox[3]) / 2
                score += state.get_tracking_bonus((det_cx, det_cy))

            if score > best_score:
                best_score = score
                target = person

        if target is None:
            self._update_habituation(None)
            return None

        target_track_id = target.get("track_id")

        # Switch tracked person if a different one is now best
        if target_track_id != self._tracked_person:
            if self._tracked_person is not None:
                old_dwell = self._person_dwell.get(self._tracked_person, 0.0)
                logger.debug(
                    "SocialAttention: switching from track_id=%s (dwell=%.1fs) to %s",
                    self._tracked_person,
                    old_dwell,
                    target_track_id,
                )
            self._tracked_person = target_track_id

        # Update habituation (current target accumulates, others recover)
        self._update_habituation(target_track_id)

        # Calculate target position
        bbox = target.get("bbox_xyxy") or target.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            return None

        if self.prefer_faces:
            # Check for pose keypoints (nose, eyes)
            pose = target.get("pose", {})
            keypoints = target.get("keypoints", {})

            if isinstance(pose, dict) and "nose" in pose:
                target_u, target_v = pose["nose"]
            elif isinstance(keypoints, dict) and "nose" in keypoints:
                target_u, target_v = keypoints["nose"]
            else:
                # Upper center of bbox (approximate head position)
                target_u = (bbox[0] + bbox[2]) / 2
                # 15% down from top of bbox
                target_v = bbox[1] + (bbox[3] - bbox[1]) * 0.15
        else:
            # Center of person bbox
            target_u = (bbox[0] + bbox[2]) / 2
            target_v = (bbox[1] + bbox[3]) / 2

        # Boost confidence if tracking same person
        confidence = target.get("conf", 0.5)
        if target_track_id == self._tracked_person:
            confidence = min(1.0, confidence + self.tracking_hysteresis)

        # Apply habituation to priority
        habituation_factor = self._get_habituation_factor(target_track_id) if target_track_id else 1.0
        priority_scale = habituation_factor

        # Log habituation state periodically
        dwell = self._person_dwell.get(target_track_id, 0.0)
        if dwell > 0.5:  # Only log when there's meaningful dwell time
            logger.debug(
                "SocialAttention: track_id=%s dwell=%.1fs habituation=%.2f priority_scale=%.2f",
                target_track_id,
                dwell,
                1.0 - habituation_factor,
                priority_scale,
            )

        # Record tracking target in centralized salience map for spatial hysteresis
        state.record_tracking_target((target_u, target_v))

        # Report focus result to FocusLearner for adaptive movement learning
        # This closes the control loop: look_at_image records intent,
        # then we report where the target actually ended up
        self._report_focus_result(target_u, target_v, state)

        logger.debug(
            "SocialAttention: proposing track to (%.1f, %.1f) for person track_id=%s, conf=%.2f, priority=%.2f",
            target_u,
            target_v,
            target_track_id,
            confidence,
            self.base_priority * priority_scale,
        )

        return self._create_proposal(
            action_type="track",  # Use track for smoother following
            target=(target_u, target_v),
            priority_scale=priority_scale,
            confidence=confidence,
            track_id=target_track_id,
            is_tracked=target_track_id == self._tracked_person,
            num_people=len(people),
            habituation=1.0 - habituation_factor,  # 0 = fresh, 1 = fully habituated
            dwell_time=dwell,
        )

    def _report_focus_result(
        self,
        target_u: float,
        target_v: float,
        state: BehaviorState,
    ) -> None:
        """Report where the tracked target is to the FocusLearner.

        This provides feedback for adaptive movement learning by telling
        the FocusLearner where the target ended up after a movement.

        Args:
            target_u: Current horizontal position of target in frame.
            target_v: Current vertical position of target in frame.
            state: Current behavior state (contains focus_learner reference).
        """
        if state.focus_learner is None:
            return

        try:
            error = state.focus_learner.record_result(target_u, target_v)
            if error is not None and (
                abs(error.overshoot_ratio_u - 1.0) > 0.2 or abs(error.overshoot_ratio_v - 1.0) > 0.2
            ):
                logger.debug(
                    "SocialAttention: focus error reported (overshoot_u=%.2f, overshoot_v=%.2f)",
                    error.overshoot_ratio_u,
                    error.overshoot_ratio_v,
                )
        except Exception as e:
            logger.debug("Focus result reporting failed: %s", e)

    def reset(self) -> None:
        """Reset tracking state."""
        super().reset()
        self._tracked_person = None
        self._person_dwell.clear()
        self._person_last_attended.clear()
        self._last_update_time = time.time()


__all__ = ["SocialAttention"]
