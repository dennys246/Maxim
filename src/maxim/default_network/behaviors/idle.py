"""Idle behaviors - Scanning and micro-movements when nothing is happening.

These low-priority behaviors activate when there are no other stimuli,
providing naturalistic movement even during quiet periods.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import TYPE_CHECKING

from maxim.default_network.behaviors.base import Behavior, BehaviorState
from maxim.default_network.messages import ActionProposal

if TYPE_CHECKING:
    from maxim.spatial import WorkspaceBoundsLearner

logger = logging.getLogger(__name__)


class IdleScan(Behavior):
    """Dynamically explore the environment when nothing interesting is happening.

    Uses zone-based exploration to cover the visual field:
    1. Divides frame into 9 zones (3x3 grid)
    2. Cycles through zones, avoiding recently visited
    3. Adds random jitter for natural movement

    Attributes:
        idle_timeout: Seconds of no activity before exploring (default 5.0).
        waypoint_dwell: Seconds to look at each waypoint (default 1.5-3.0).
        saccade_probability: Chance of quick saccade vs smooth move (default 0.3).
    """

    name = "idle_scan"
    base_priority = 0.2  # Low priority - only when nothing else happening
    cooldown_seconds = 0.4

    # Exploration zones (normalized 0-1 coordinates)
    _ZONES = [
        (0.2, 0.3),   # Upper left
        (0.5, 0.2),   # Upper center
        (0.8, 0.3),   # Upper right
        (0.15, 0.5),  # Mid left
        (0.5, 0.5),   # Center
        (0.85, 0.5),  # Mid right
        (0.2, 0.7),   # Lower left
        (0.5, 0.8),   # Lower center
        (0.8, 0.7),   # Lower right
    ]

    def __init__(
        self,
        idle_timeout: float = 5.0,
        waypoint_dwell_min: float = 1.5,
        waypoint_dwell_max: float = 3.0,
        saccade_probability: float = 0.3,
    ) -> None:
        """Initialize dynamic exploration.

        Args:
            idle_timeout: Seconds before starting to explore.
            waypoint_dwell_min: Minimum time at each waypoint.
            waypoint_dwell_max: Maximum time at each waypoint.
            saccade_probability: Chance of quick saccade (0-1).
        """
        super().__init__()
        self.idle_timeout = idle_timeout
        self.waypoint_dwell_min = waypoint_dwell_min
        self.waypoint_dwell_max = waypoint_dwell_max
        self.saccade_probability = saccade_probability

        self._last_stimulus_time: float = time.time()
        self._frame_size: tuple[float, float] = (640.0, 480.0)
        self._frame_center: tuple[float, float] = (320.0, 240.0)

        # Waypoint state
        self._current_waypoint: tuple[float, float] | None = None
        self._waypoint_start_time: float = 0.0
        self._waypoint_dwell_time: float = 2.0
        self._visited_zones: list[int] = []  # Track recently visited zones
        self._interpolation_progress: float = 1.0  # 0-1, 1 = arrived
        self._previous_target: tuple[float, float] | None = None

    def _pick_new_waypoint(self) -> tuple[float, float]:
        """Pick a new exploration waypoint from zones."""
        return self._pick_zone_waypoint()

    def _pick_zone_waypoint(self) -> tuple[float, float]:
        """Fallback: Pick waypoint from predefined zones."""
        # Find zones not recently visited
        all_zones = list(range(len(self._ZONES)))
        unvisited = [i for i in all_zones if i not in self._visited_zones[-4:]]

        if not unvisited:
            # All zones visited recently, reset and pick randomly
            self._visited_zones = []
            unvisited = all_zones

        # Pick from unvisited zones with some randomness
        zone_idx = random.choice(unvisited)
        self._visited_zones.append(zone_idx)

        # Get base zone position
        zx, zy = self._ZONES[zone_idx]

        # Add random jitter within the zone (±15% of frame)
        jitter_x = random.uniform(-0.15, 0.15)
        jitter_y = random.uniform(-0.10, 0.10)

        # Convert to pixel coordinates
        w, h = self._frame_size
        px = (zx + jitter_x) * w
        py = (zy + jitter_y) * h

        # Clamp to safe bounds (10%-90% of frame)
        px = max(w * 0.1, min(w * 0.9, px))
        py = max(h * 0.1, min(h * 0.9, py))

        return (px, py)

    def evaluate(
        self,
        detections: list[dict],
        state: BehaviorState,
    ) -> ActionProposal | None:
        """Propose exploration movement if idle for long enough."""
        now = time.time()

        # Reset idle timer if there are interesting detections
        if detections:
            self._last_stimulus_time = now
            self._current_waypoint = None  # Reset exploration on new stimulus

        # Check if we've been idle long enough
        idle_duration = now - self._last_stimulus_time
        if idle_duration < self.idle_timeout:
            return None

        if not self.can_activate():
            return None

        # Check if we need a new waypoint
        need_new_waypoint = (
            self._current_waypoint is None or
            (now - self._waypoint_start_time) > self._waypoint_dwell_time
        )

        if need_new_waypoint:
            # Save previous target for interpolation
            self._previous_target = self._current_waypoint or self._frame_center

            # Pick new waypoint
            self._current_waypoint = self._pick_new_waypoint()
            self._waypoint_start_time = now
            self._waypoint_dwell_time = random.uniform(
                self.waypoint_dwell_min, self.waypoint_dwell_max
            )
            self._interpolation_progress = 0.0

            # Decide if this is a quick saccade or smooth movement
            is_saccade = random.random() < self.saccade_probability

            if is_saccade:
                # Quick saccade - jump directly to target
                self._interpolation_progress = 1.0

        # Calculate current target position (smooth interpolation)
        if self._interpolation_progress < 1.0:
            # Smooth ease-in-out interpolation
            self._interpolation_progress = min(1.0, self._interpolation_progress + 0.15)
            t = self._interpolation_progress
            # Smooth step function for natural movement
            t = t * t * (3.0 - 2.0 * t)

            px = self._previous_target[0] if self._previous_target else self._frame_center[0]
            py = self._previous_target[1] if self._previous_target else self._frame_center[1]
            tx, ty = self._current_waypoint

            target = (px + (tx - px) * t, py + (ty - py) * t)
        else:
            target = self._current_waypoint

        # Calculate delta from center for scan action
        cx, cy = self._frame_center
        dx = target[0] - cx
        dy = target[1] - cy

        return self._create_proposal(
            action_type="scan",
            target=target,
            priority_scale=1.0,
            confidence=0.5,
            idle_duration=idle_duration,
            waypoint=self._current_waypoint,
            interpolation=self._interpolation_progress,
            delta=(dx * 0.15, dy * 0.15),
        )

    def set_frame_center(self, center: tuple[float, float]) -> None:
        """Set the center point for scanning.

        Args:
            center: (x, y) pixel coordinates of frame center.
        """
        self._frame_center = center

    def set_frame_size(self, size: tuple[float, float]) -> None:
        """Set frame size for zone calculations.

        Args:
            size: (width, height) in pixels.
        """
        self._frame_size = size
        self._frame_center = (size[0] / 2, size[1] / 2)

    def reset(self) -> None:
        """Reset exploration state."""
        super().reset()
        self._last_stimulus_time = time.time()
        self._current_waypoint = None
        self._visited_zones = []
        self._interpolation_progress = 1.0


class Microsaccades(Behavior):
    """Small random eye movements during fixation.

    Real eyes make small involuntary movements even when fixating.
    This adds subtle movement to prevent an unnaturally frozen gaze.

    Attributes:
        fixation_timeout: Seconds of fixation before microsaccades (default 2.0).
        amplitude: Maximum displacement in pixels (default 20).
    """

    name = "microsaccades"
    base_priority = 0.1  # Very low - just background activity
    cooldown_seconds = 0.3

    def __init__(
        self,
        fixation_timeout: float = 2.0,
        amplitude: float = 20.0,
    ) -> None:
        """Initialize microsaccades.

        Args:
            fixation_timeout: How long to fixate before starting.
            amplitude: Size of random movements in pixels.
        """
        super().__init__()
        self.fixation_timeout = fixation_timeout
        self.amplitude = amplitude

        self._fixation_start: float = time.time()
        self._last_target: tuple[float, float] | None = None

    def evaluate(
        self,
        detections: list[dict],
        state: BehaviorState,
    ) -> ActionProposal | None:
        """Propose small random movement if fixating too long."""
        now = time.time()

        # Check if we've been fixating long enough
        if now - self._fixation_start < self.fixation_timeout:
            return None

        if not self.can_activate():
            return None

        # Generate small random displacement
        angle = random.uniform(0, 2 * math.pi)
        magnitude = random.uniform(0.3, 1.0) * self.amplitude

        dx = magnitude * math.cos(angle)
        dy = magnitude * math.sin(angle)

        return self._create_proposal(
            action_type="scan",
            target=None,  # Relative movement
            priority_scale=1.0,
            confidence=0.3,
            delta=(dx, dy),
        )

    def reset(self) -> None:
        """Reset state."""
        super().reset()
        self._fixation_start = time.time()
        self._last_target = None


class ReturnToCenter(Behavior):
    """Drift back toward neutral head position when at extremes.

    If the head is at an extreme position and there's no target,
    slowly return toward center for comfort and readiness.

    Uses 6-DOF position tracking with TRANSLATION as primary:
    - Monitors Y, Z translation and yaw, pitch rotation
    - Returns toward center using translation-first approach

    Attributes:
        threshold: Fraction of max range to trigger (default 0.7).
        return_speed: How fast to return (0-1 scale, default 0.3).
        bounds_learner: Optional WorkspaceBoundsLearner for learned limits.
    """

    name = "return_to_center"
    base_priority = 0.15
    cooldown_seconds = 1.0

    # 6D workspace limits (fallback if no bounds_learner)
    # Physical limits: Yaw ±55°, Pitch ±35°
    _DEFAULT_MAX_Y = 18.0   # mm (left/right translation)
    _DEFAULT_MAX_Z = 15.0   # mm (up/down translation)
    _DEFAULT_MAX_YAW = 55.0    # degrees (matches physical head limit)
    _DEFAULT_MAX_PITCH = 35.0  # degrees

    # Minimum limits - never use smaller than these even if bounds_learner suggests it
    # This prevents the behavior from over-triggering due to collapsed bounds
    _MIN_MAX_YAW = 27.5     # 50% of physical limit
    _MIN_MAX_PITCH = 17.5   # 50% of physical limit
    _MIN_MAX_Y = 5.4        # 30% of default
    _MIN_MAX_Z = 4.5        # 30% of default

    # Critical threshold - force return even with detections
    CRITICAL_THRESHOLD = 0.90  # Trigger earlier to prevent hitting limits

    def __init__(
        self,
        threshold: float = 0.7,
        return_speed: float = 0.3,
        bounds_learner: "WorkspaceBoundsLearner | None" = None,
    ) -> None:
        """Initialize return to center.

        Args:
            threshold: Fraction of range beyond which to return.
            return_speed: Speed of return motion (0-1).
            bounds_learner: Optional bounds learner for dynamic limits.
        """
        super().__init__()
        self.threshold = threshold
        self.return_speed = return_speed
        self._bounds_learner = bounds_learner

        # 6D head position (will be set by DN)
        self._head_position_6d: tuple[float, float, float, float, float, float] | None = None
        # Legacy 2D position for compatibility
        self._head_position: tuple[float, float] | None = None
        # Legacy max range (kept for compatibility)
        self._max_range: tuple[float, float] = (55.0, 35.0)

    def _get_limits(self) -> dict[str, float]:
        """Get workspace limits, preferring learned bounds over hardcoded.

        Enforces minimum floors to prevent collapsed bounds from causing
        constant CRITICAL triggers.

        Returns:
            Dict with y, z, yaw, pitch limits.
        """
        if self._bounds_learner is not None:
            # Get learned bounds but enforce minimum floors
            return {
                "y": max(self._MIN_MAX_Y, self._bounds_learner.get_bound("y")),
                "z": max(self._MIN_MAX_Z, self._bounds_learner.get_bound("z")),
                "yaw": max(self._MIN_MAX_YAW, self._bounds_learner.get_bound("yaw")),
                "pitch": max(self._MIN_MAX_PITCH, self._bounds_learner.get_bound("pitch")),
            }

        return {
            "y": self._DEFAULT_MAX_Y,
            "z": self._DEFAULT_MAX_Z,
            "yaw": self._DEFAULT_MAX_YAW,
            "pitch": self._DEFAULT_MAX_PITCH,
        }

    def set_bounds_learner(self, bounds_learner: "WorkspaceBoundsLearner | None") -> None:
        """Set the bounds learner for dynamic limits.

        Args:
            bounds_learner: WorkspaceBoundsLearner instance or None.
        """
        self._bounds_learner = bounds_learner

    def evaluate(
        self,
        detections: list[dict],
        state: BehaviorState,
    ) -> ActionProposal | None:
        """Propose returning to center if at extreme position.

        Uses 6D position tracking with TRANSLATION as primary concern.
        Calculates distance from center using both translation (Y, Z) and
        rotation (yaw, pitch) components.
        """
        if not self.can_activate():
            return None

        # Use 6D position if available, fall back to 2D
        if self._head_position_6d is not None:
            x, y, z, roll, pitch, yaw = self._head_position_6d
        elif self._head_position is not None:
            yaw, pitch = self._head_position
            _x, y, z, _roll = 0.0, 0.0, 0.0, 0.0
        else:
            return None

        # Get limits (learned or hardcoded)
        limits = self._get_limits()
        max_y = limits["y"]
        max_z = limits["z"]
        max_yaw = limits["yaw"]
        max_pitch = limits["pitch"]

        # SANITY CHECK: If yaw is outside physical limits (±90°), the reading is
        # likely corrupted (e.g., world-frame yaw instead of head-relative).
        # In this case, ignore the yaw component for distance calculation.
        # The head physically cannot rotate more than ±65° relative to body.
        yaw_is_valid = abs(yaw) <= 90.0
        pitch_is_valid = abs(pitch) <= 60.0

        # Calculate 6D distance from center using ellipsoid model
        # Translation components (primary concern for IK stability)
        norm_y = y / max_y if max_y > 0 else 0
        norm_z = z / max_z if max_z > 0 else 0
        trans_dist = math.sqrt(norm_y**2 + norm_z**2)

        # Rotation components (secondary) - only use if values are valid
        if yaw_is_valid:
            norm_yaw = yaw / max_yaw if max_yaw > 0 else 0
        else:
            norm_yaw = 0.0  # Don't trust invalid yaw reading
            logger.debug("ReturnToCenter: ignoring invalid yaw=%.1f (outside ±90°)", yaw)

        if pitch_is_valid:
            norm_pitch = pitch / max_pitch if max_pitch > 0 else 0
        else:
            norm_pitch = 0.0
            logger.debug("ReturnToCenter: ignoring invalid pitch=%.1f (outside ±60°)", pitch)

        rot_dist = math.sqrt(norm_yaw**2 + norm_pitch**2)

        # Combined 6D distance (translation weighted higher)
        combined_dist = math.sqrt(trans_dist**2 * 0.6 + rot_dist**2 * 0.4)

        # Check if we're at an extreme position
        is_extreme = combined_dist > self.threshold

        # Critical override: if very close to limits, return even with detections
        is_critical = combined_dist > self.CRITICAL_THRESHOLD
        # Extreme: way beyond limits (e.g., dist > 1.5 means 150% of safe zone)
        is_very_extreme = combined_dist > 1.5

        # Normal mode: only activate if no detections
        # Critical mode: activate regardless to prevent getting stuck
        if detections and not is_critical:
            return None

        if not is_extreme:
            return None

        # Calculate return direction - move all components toward center
        # Translation-first: stronger return force on Y, Z
        speed = self.return_speed * (2.0 if is_very_extreme else 1.5 if is_critical else 1.0)

        # Return deltas (will be scaled by move_relative)
        # Combining translation (Y, Z) and rotation (yaw, pitch) into 2D delta
        # dx affects horizontal (Y translation + yaw rotation)
        # dy affects vertical (Z translation + pitch rotation)
        dx = -(y * 0.3 + yaw * 0.7) * speed  # Blend Y and yaw
        dy = -(z * 0.3 + pitch * 0.7) * speed  # Blend Z and pitch

        # Priority scaling - must beat social tracking (0.9) when critical
        # base_priority (0.15) * scale = effective priority
        # Extreme: 0.15 * 8.0 = 1.2 (beats social 0.9)
        # Critical: 0.15 * 7.0 = 1.05 (beats social 0.9)
        # Normal: 0.15 * 1.0 = 0.15
        if is_very_extreme:
            priority_scale = 8.0
            confidence = 0.9
        elif is_critical:
            priority_scale = 7.0
            confidence = 0.7
        else:
            priority_scale = 1.0
            confidence = 0.4

        # DEBUG: Log position info only when extreme (reduce log spam)
        if is_critical:
            logger.debug(
                "RTC_DEBUG: yaw=%.1f pitch=%.1f y=%.1f z=%.1f | max_yaw=%.1f max_pitch=%.1f "
                "| norm_yaw=%.2f norm_pitch=%.2f | trans_dist=%.2f rot_dist=%.2f combined_dist=%.2f "
                "| very_extreme=%s critical=%s",
                yaw, pitch, y, z, max_yaw, max_pitch,
                norm_yaw, norm_pitch, trans_dist, rot_dist, combined_dist,
                is_very_extreme, is_critical
            )

        if is_very_extreme:
            logger.info(
                "ReturnToCenter EXTREME: trans(y=%.1f,z=%.1f) rot(yaw=%.1f,pitch=%.1f) "
                "dist=%.2f -> delta=(%.1f, %.1f) priority=%.2f",
                y, z, yaw, pitch, combined_dist, dx * 10, dy * 10,
                self.base_priority * priority_scale
            )
        elif is_critical:
            logger.info(
                "ReturnToCenter CRITICAL: dist=%.2f -> delta=(%.1f, %.1f) priority=%.2f",
                combined_dist, dx * 10, dy * 10, self.base_priority * priority_scale
            )

        return self._create_proposal(
            action_type="scan",
            target=None,
            priority_scale=priority_scale,
            confidence=confidence,
            delta=(dx * 10, dy * 10),
            head_position=self._head_position,
            head_position_6d=self._head_position_6d,
            is_critical=is_critical,
        )

    def set_head_position(self, yaw: float, pitch: float) -> None:
        """Update current head position (2D legacy).

        Args:
            yaw: Current yaw angle in degrees.
            pitch: Current pitch angle in degrees.
        """
        self._head_position = (yaw, pitch)

    def set_head_position_6d(
        self,
        x: float, y: float, z: float,
        roll: float, pitch: float, yaw: float,
    ) -> None:
        """Update current head position (full 6D).

        Args:
            x, y, z: Translation in mm.
            roll, pitch, yaw: Rotation in degrees.
        """
        self._head_position_6d = (x, y, z, roll, pitch, yaw)
        # Also update legacy 2D for compatibility
        self._head_position = (yaw, pitch)


__all__ = [
    "IdleScan",
    "Microsaccades",
    "ReturnToCenter",
]
