"""Gaze management mixin for DefaultNetwork.

Extracts salience-gated gaze control, exploration gaze logic,
scene context queries, and idle exploration target generation
into a reusable mixin.

All methods access state via ``self._*`` attributes initialised in
``DefaultNetwork.__init__``.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING

from maxim.attention import GazeCommand
from maxim.default_network.movement_utils import compute_dynamic_duration, compute_opposite_position

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class GazeManagerMixin:
    """Mixin providing gaze-management helpers for DefaultNetwork.

    Assumes the host class exposes at least:
        _config, _gaze_history, _spatial_map, _attention_network,
        _salience_map_unified, _gaze_controller, _scene_context,
        _last_interesting_time, _next_exploration_time, _maxim,
        _novelty_tracker, _interests
    """

    # ------------------------------------------------------------------
    # Public gaze query API
    # ------------------------------------------------------------------

    def look_opposite(self, duration: float | None = None) -> bool:
        """Look at the opposite side of the current view.

        Useful for exploration when the agent wants to see what's behind them
        or check the other side of the scene.

        Args:
            duration: Movement duration in seconds. If None, uses dynamic calculation.

        Returns:
            True if movement was executed, False if blocked or no current position.
        """
        current = self._gaze_history.get_current_position()
        if current is None:
            logger.debug("look_opposite: no current position")
            return False

        target = compute_opposite_position(
            current,
            self._config.movement.image_width,
            self._config.movement.image_height,
        )

        # Check reachability
        if self._config.reachability_check_enabled:
            reachability = 1.0
            if self._attention_network is not None:
                reachability = self._attention_network.get_reachability(target)
            elif self._spatial_map is not None:
                reachability = self._spatial_map.get_reachability(target)

            if reachability < self._config.min_reachability_threshold:
                logger.debug("look_opposite: target unreachable (%.2f)", reachability)
                return False

        # Compute duration if not specified
        if duration is None:
            duration = compute_dynamic_duration(
                current,
                target,
                self._config.movement,
                add_jitter=True,
            )

        # Execute movement
        try:
            if hasattr(self._maxim, "look_at_image"):
                self._maxim.look_at_image(target[0], target[1], duration=duration)
                self._gaze_history.record_gaze(target)

                if self._attention_network is not None:
                    self._attention_network.record_gaze(target, success=True)
                if self._spatial_map is not None:
                    self._spatial_map.record_movement(target, success=True)

                logger.debug("look_opposite: moved to (%.0f, %.0f)", target[0], target[1])
                return True
        except Exception as e:
            error_msg = str(e).lower()
            if "collision" in error_msg or "ik" in error_msg or "not achievable" in error_msg:
                if self._attention_network is not None:
                    self._attention_network.record_gaze(target, success=False)
                if self._spatial_map is not None:
                    self._spatial_map.record_movement(target, success=False)
            logger.warning("look_opposite failed: %s", e)

        return False

    def get_opposite_position(self) -> tuple[float, float] | None:
        """Get the opposite position from current gaze without moving.

        Returns:
            (u, v) coordinates of opposite position, or None if no current position.
        """
        current = self._gaze_history.get_current_position()
        if current is None:
            return None

        return compute_opposite_position(
            current,
            self._config.movement.image_width,
            self._config.movement.image_height,
        )

    def get_salience_target(
        self,
        mode: str = "peak",
        temperature: float = 1.0,
    ) -> tuple[float, float] | None:
        """Get a gaze target from the unified salience map.

        Args:
            mode: "peak" for most salient, "sample" for probabilistic.
            temperature: Sampling temperature (only for mode="sample").

        Returns:
            (x, y) pixel coordinates, or None if salience map is disabled.
        """
        if self._salience_map_unified is None:
            return None

        if mode == "peak":
            return self._salience_map_unified.get_peak_target()
        elif mode == "sample":
            return self._salience_map_unified.sample_target(temperature=temperature)
        else:
            logger.warning("Unknown salience target mode: %s", mode)
            return self._salience_map_unified.get_peak_target()

    def get_salience_info(self, position: tuple[float, float]) -> dict[str, float] | None:
        """Get detailed salience breakdown for a position.

        Args:
            position: (x, y) pixel coordinates.

        Returns:
            Dict with component salience values, or None if map is disabled.
        """
        if self._salience_map_unified is None:
            return None

        return self._salience_map_unified.get_cell_info(position)

    def get_gaze_command(
        self,
        force_target: tuple[float, float] | None = None,
    ) -> GazeCommand | None:
        """Get a gaze command from the saccade-fixate controller.

        Args:
            force_target: Force a saccade to this target if provided.

        Returns:
            GazeCommand if movement should occur, None to hold position.
        """
        if self._gaze_controller is None:
            return None

        current = self._gaze_history.get_current_position()
        return self._gaze_controller.update(
            current_gaze=current,
            force_target=force_target,
        )

    # ------------------------------------------------------------------
    # Scene context queries
    # ------------------------------------------------------------------

    def is_scene_scanning(self) -> bool:
        """Check if currently in scene scanning mode after a scene change."""
        if self._scene_context is None:
            return False
        return self._scene_context.is_scanning()

    def get_scene_age(self) -> float:
        """Get time since the last significant scene change.

        Returns:
            Seconds since scene changed, or inf if never changed.
        """
        if self._scene_context is None:
            return float("inf")
        return self._scene_context.get_scene_age()

    def force_scene_scan(self) -> None:
        """Force the system into scene scanning mode.

        Use when the robot has turned around or entered a new area.
        """
        if self._scene_context is not None:
            self._scene_context.force_scene_change()
            # Reset idle timer to trigger exploration
            self._last_interesting_time = 0.0
            self._next_exploration_time = time.time()
            logger.info("Forced scene scan triggered")

    # ------------------------------------------------------------------
    # Idle exploration helpers (called from _process_tick)
    # ------------------------------------------------------------------

    def _schedule_next_exploration(self) -> float:
        """Schedule the next idle exploration at a random future time.

        Returns:
            Timestamp when next exploration should occur.
        """
        delay = random.uniform(
            self._config.idle_exploration_min_seconds,
            self._config.idle_exploration_max_seconds,
        )
        return time.time() + delay

    def _has_interesting_content(self, detections: list[dict]) -> bool:
        """Check if any detections are interesting enough to reset idle timer.

        A detection is interesting if:
        - It's a person (class_id=0) - people are always interesting
        - It's in the interest set, OR
        - It has high novelty (not seen recently)

        Args:
            detections: List of detection dicts.

        Returns:
            True if at least one detection is interesting.
        """
        if not detections:
            return False

        # COCO class ID for person
        PERSON_CLASS_ID = 0

        for det in detections:
            class_id = det.get("class_id")

            # People are always interesting (social priority)
            if class_id == PERSON_CLASS_ID:
                return True

            # Check if in interests
            if class_id is not None and class_id in self._interests:
                return True

            # Check novelty (with class-level modulation)
            track_id = det.get("track_id")
            if track_id is not None:
                novelty = self._novelty_tracker.get_novelty(track_id, class_id=class_id)
                if novelty >= self._config.idle_novelty_threshold:
                    return True

        return False

    def _generate_exploration_target(self) -> tuple[float, float]:
        """Generate a safe target position for idle exploration.

        Uses the spatial map if available to select from known-reachable
        positions or conservatively-safe unexplored positions.

        Returns:
            (u, v) pixel coordinates for the target.
        """
        # Use spatial map for smart target selection if available
        if self._spatial_map is not None:
            target = self._spatial_map.get_safe_exploration_target(
                avoid_current=True,
                prefer_unexplored=True,
                prefer_interesting=True,
            )
            if target:
                return target

        # Fallback to random position in safe range
        u_min, u_max = self._config.idle_exploration_range
        v_min, v_max = self._config.idle_exploration_range

        u = random.uniform(u_min, u_max)
        v = random.uniform(v_min, v_max)

        # Convert to pixel coordinates (assuming 640x480 resolution)
        pixel_u = u * 640
        pixel_v = v * 480

        return (pixel_u, pixel_v)
