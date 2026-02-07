"""Gaze Controller - Human-like saccade-fixate-saccade dynamics.

This module replaces smooth tracking with human-like gaze patterns:
- Saccades: Rapid jumps between fixation points
- Fixations: Brief pauses where gaze is held (200-800ms)
- Exploration: Slow drift when nothing is salient

Humans don't smoothly scan - they jump and pause. This creates more
natural-looking gaze behavior.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.attention import SalienceMap

logger = logging.getLogger(__name__)


class GazeState(Enum):
    """Current state of the gaze controller."""
    FIXATING = "fixating"      # Holding gaze on target
    SACCADING = "saccading"    # Rapid movement to new target
    EXPLORING = "exploring"    # Slow scan when nothing salient


@dataclass
class GazeCommand:
    """Command to execute a gaze movement."""
    target: tuple[float, float]  # (x, y) pixel coordinates
    speed_multiplier: float      # Relative to normal tracking speed
    action_type: str             # "saccade", "fixate", "explore"
    priority: float              # For behavior arbitration
    metadata: dict[str, Any]     # Additional context


@dataclass
class GazeControllerConfig:
    """Configuration for gaze controller.

    Attributes:
        min_fixation_ms: Minimum fixation duration.
        max_fixation_ms: Maximum fixation duration.
        mean_fixation_ms: Mean fixation duration (for sampling).

        saccade_speed_multiplier: Speed boost for saccades.
        saccade_threshold_pixels: Minimum distance to trigger saccade.

        exploration_trigger_seconds: Idle time before exploration mode.
        exploration_speed_multiplier: Slow speed for exploration.

        interrupt_salience_ratio: How much more salient to interrupt fixation.
        sample_temperature: Temperature for salience sampling.
    """
    # Fixation dynamics (human average is ~250-350ms)
    min_fixation_ms: float = 200.0
    max_fixation_ms: float = 800.0
    mean_fixation_ms: float = 350.0
    fixation_std_ms: float = 100.0  # Standard deviation

    # Saccade dynamics
    saccade_speed_multiplier: float = 2.0  # Faster than normal tracking
    saccade_threshold_pixels: float = 30.0  # Min distance for saccade

    # Exploration mode
    exploration_trigger_seconds: float = 2.0  # Idle before exploring
    exploration_speed_multiplier: float = 0.5  # Slower, leisurely

    # Interruption threshold
    interrupt_salience_ratio: float = 1.5  # 50% more salient to interrupt

    # Sampling
    sample_temperature: float = 0.8  # Slightly deterministic


class GazeController:
    """Human-like gaze control with saccade-fixate dynamics.

    Instead of smooth tracking, this controller produces:
    1. Rapid saccades to new targets
    2. Fixations of variable duration
    3. Slow exploration when nothing is interesting

    Example:
        controller = GazeController(salience_map, config)

        # Each frame, update and get command if needed
        cmd = controller.update(current_gaze=(320, 240), dt=0.033)
        if cmd:
            execute_gaze_command(cmd)
    """

    def __init__(
        self,
        salience_map: "SalienceMap | None" = None,
        config: GazeControllerConfig | None = None,
    ) -> None:
        """Initialize gaze controller.

        Args:
            salience_map: SalienceMap for target selection.
            config: Controller configuration.
        """
        self._salience_map = salience_map
        self.config = config or GazeControllerConfig()

        # State
        self._state = GazeState.FIXATING
        self._current_target: tuple[float, float] | None = None
        self._fixation_start: float = 0.0
        self._fixation_duration: float = 0.0
        self._last_activity_time: float = time.time()
        self._last_saccade_time: float = 0.0

        # For smooth state transitions
        self._pending_target: tuple[float, float] | None = None

    def set_salience_map(self, salience_map: "SalienceMap") -> None:
        """Set or update the salience map.

        Args:
            salience_map: SalienceMap instance.
        """
        self._salience_map = salience_map

    def update(
        self,
        current_gaze: tuple[float, float] | None = None,
        dt: float = 0.033,
        force_target: tuple[float, float] | None = None,
    ) -> GazeCommand | None:
        """Update gaze state and return command if movement needed.

        Args:
            current_gaze: Current (x, y) gaze position.
            dt: Time delta since last update.
            force_target: If set, force saccade to this target.

        Returns:
            GazeCommand if movement should occur, None otherwise.
        """
        now = time.time()

        # Handle forced target (e.g., from external request)
        if force_target is not None:
            self._trigger_saccade(force_target, now)
            return self._create_saccade_command(force_target)

        # Check if salience map is available
        if self._salience_map is None:
            return None

        # State machine
        if self._state == GazeState.FIXATING:
            return self._update_fixation(current_gaze, now)
        elif self._state == GazeState.SACCADING:
            return self._update_saccade(current_gaze, now)
        else:  # EXPLORING
            return self._update_exploration(current_gaze, now)

    def _update_fixation(
        self,
        current_gaze: tuple[float, float] | None,
        now: float,
    ) -> GazeCommand | None:
        """Update during fixation state.

        Check if fixation should end due to:
        1. Fixation duration expired
        2. Much more salient target appeared
        """
        elapsed = now - self._fixation_start

        # Get current and peak salience
        current_salience = 0.0
        if current_gaze and self._salience_map:
            current_salience = self._salience_map.get_salience_at(current_gaze)

        peak_target = self._salience_map.get_peak_target()
        peak_salience = self._salience_map.get_salience_at(peak_target)

        # Check for interruption by very salient target
        min_fixation_sec = self.config.min_fixation_ms / 1000.0
        can_interrupt = elapsed > min_fixation_sec

        should_interrupt = (
            can_interrupt and
            peak_salience > current_salience * self.config.interrupt_salience_ratio
        )

        # Check if fixation expired
        fixation_expired = elapsed >= self._fixation_duration

        if should_interrupt or fixation_expired:
            # Sample new target (not always peak for variety)
            new_target = self._salience_map.sample_target(
                temperature=self.config.sample_temperature,
                avoid_current=True,
            )

            # Check if far enough for saccade
            if current_gaze:
                dist = self._distance(current_gaze, new_target)
                if dist < self.config.saccade_threshold_pixels:
                    # Too close - extend fixation slightly
                    self._fixation_duration += 0.1
                    return None

            self._trigger_saccade(new_target, now)
            return self._create_saccade_command(new_target)

        # Check for transition to exploration
        idle_time = now - self._last_activity_time
        if idle_time > self.config.exploration_trigger_seconds:
            self._state = GazeState.EXPLORING
            logger.debug("GazeController: entering exploration mode (idle %.1fs)", idle_time)

        return None  # Keep fixating

    def _update_saccade(
        self,
        current_gaze: tuple[float, float] | None,
        now: float,
    ) -> GazeCommand | None:
        """Update during saccade state.

        Saccades are brief - transition to fixation once target reached.
        """
        # Check if we've reached the target
        if current_gaze and self._current_target:
            dist = self._distance(current_gaze, self._current_target)
            if dist < 20.0:  # Close enough
                self._start_fixation(now)
                return None

        # Saccade timeout (shouldn't take more than 200ms)
        saccade_elapsed = now - self._last_saccade_time
        if saccade_elapsed > 0.2:
            self._start_fixation(now)
            return None

        # Still saccading - no new command
        return None

    def _update_exploration(
        self,
        current_gaze: tuple[float, float] | None,
        now: float,
    ) -> GazeCommand | None:
        """Update during exploration state.

        Slow, deliberate scanning of the scene.
        """
        # Check if something salient appeared
        peak_target = self._salience_map.get_peak_target()
        peak_salience = self._salience_map.get_salience_at(peak_target)

        # Exit exploration if something salient appears
        if peak_salience > 0.5:  # Threshold for "interesting"
            self._last_activity_time = now
            self._trigger_saccade(peak_target, now)
            return self._create_saccade_command(peak_target)

        # Sample exploration target
        target = self._salience_map.sample_target(
            temperature=1.5,  # More random during exploration
            avoid_current=True,
        )

        return GazeCommand(
            target=target,
            speed_multiplier=self.config.exploration_speed_multiplier,
            action_type="explore",
            priority=0.3,  # Low priority
            metadata={"state": "exploring"},
        )

    def _trigger_saccade(self, target: tuple[float, float], now: float) -> None:
        """Trigger a saccade to target."""
        self._state = GazeState.SACCADING
        self._current_target = target
        self._last_saccade_time = now
        self._last_activity_time = now

    def _start_fixation(self, now: float) -> None:
        """Start a new fixation."""
        self._state = GazeState.FIXATING
        self._fixation_start = now
        self._fixation_duration = self._sample_fixation_duration()

        logger.debug(
            "GazeController: starting fixation (%.0fms)",
            self._fixation_duration * 1000
        )

    def _sample_fixation_duration(self) -> float:
        """Sample a fixation duration from distribution.

        Uses a truncated normal distribution centered on mean.
        """
        # Sample from normal distribution
        duration_ms = random.gauss(
            self.config.mean_fixation_ms,
            self.config.fixation_std_ms,
        )

        # Clamp to valid range
        duration_ms = max(self.config.min_fixation_ms, duration_ms)
        duration_ms = min(self.config.max_fixation_ms, duration_ms)

        return duration_ms / 1000.0  # Convert to seconds

    def _create_saccade_command(self, target: tuple[float, float]) -> GazeCommand:
        """Create a saccade command."""
        return GazeCommand(
            target=target,
            speed_multiplier=self.config.saccade_speed_multiplier,
            action_type="saccade",
            priority=0.7,  # High priority for saccades
            metadata={
                "state": "saccading",
                "fixation_duration_ms": self._fixation_duration * 1000,
            },
        )

    def _distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx * dx + dy * dy) ** 0.5

    def note_activity(self) -> None:
        """Note that activity occurred (resets exploration timer)."""
        self._last_activity_time = time.time()

    def reset(self) -> None:
        """Reset controller state."""
        self._state = GazeState.FIXATING
        self._current_target = None
        self._fixation_start = time.time()
        self._fixation_duration = self._sample_fixation_duration()
        self._last_activity_time = time.time()

    @property
    def state(self) -> GazeState:
        """Current gaze state."""
        return self._state

    @property
    def current_target(self) -> tuple[float, float] | None:
        """Current gaze target."""
        return self._current_target

    def get_stats(self) -> dict[str, Any]:
        """Get controller statistics."""
        now = time.time()
        return {
            "state": self._state.value,
            "current_target": self._current_target,
            "fixation_elapsed_ms": (now - self._fixation_start) * 1000 if self._state == GazeState.FIXATING else 0,
            "fixation_duration_ms": self._fixation_duration * 1000,
            "idle_seconds": now - self._last_activity_time,
        }


__all__ = [
    "GazeState",
    "GazeCommand",
    "GazeControllerConfig",
    "GazeController",
]
