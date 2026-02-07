"""Movement energy tracker - Estimates motor energy expenditure.

Tracks energy used for robot movements based on:
- Movement distance (angular and translational)
- Movement speed (faster = more energy)
- Movement duration

Since direct motor current isn't available from the high-level SDK,
this uses physics-based estimation from movement parameters.

Future: If motor telemetry becomes available, this can integrate
actual current/voltage readings from Dynamixel servos.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from maxim.energy.signal import EnergySignal, EnergyType
from maxim.energy.tracker import EnergyConfig, EnergyTracker

logger = logging.getLogger(__name__)


@dataclass
class MovementEnergyConfig(EnergyConfig):
    """Configuration for movement energy tracking.

    Attributes:
        angular_energy_per_degree: Energy cost per degree of rotation.
        translation_energy_per_mm: Energy cost per mm of translation.
        speed_multiplier_base: Base for exponential speed cost.
        duration_cost_per_second: Energy cost for movement duration.
        antenna_energy_multiplier: Multiplier for antenna movements.
    """

    # Distance costs
    angular_energy_per_degree: float = 0.02      # 50 degrees = 1 energy
    translation_energy_per_mm: float = 0.05      # 20 mm = 1 energy

    # Speed affects energy (faster = more expensive)
    # Energy scales with velocity^2 (kinetic energy relationship)
    speed_multiplier_base: float = 1.0           # Baseline for normal speed
    fast_speed_threshold: float = 100.0          # deg/sec threshold for "fast"
    fast_speed_multiplier: float = 1.5           # Multiplier when above threshold

    # Duration cost (motor hold time)
    duration_cost_per_second: float = 0.1        # Holding position has cost

    # Component multipliers
    antenna_energy_multiplier: float = 0.3       # Antennas are lighter
    body_rotation_multiplier: float = 2.0        # Body rotation is heavier


class MovementEnergyTracker(EnergyTracker):
    """Tracks energy expenditure from motor movements.

    Estimates energy based on movement parameters since direct
    motor telemetry isn't available in the high-level SDK.

    Example:
        tracker = MovementEnergyTracker()

        # Record a head movement
        signal = tracker.record_head_movement(
            delta_yaw=45.0,
            delta_pitch=10.0,
            duration_s=0.5,
        )

        # Or from an action signature
        signal = tracker.record_from_signature(
            "look_at:dy=90:dp=30",
            duration_s=0.3,
        )
    """

    name = "movement"
    energy_types = {EnergyType.MOTOR_COMMAND}

    # Pattern for parsing action signatures
    _LOOK_AT_PATTERN = re.compile(r"look_at:dy=(-?[\d.]+):dp=(-?[\d.]+)")

    def __init__(self, config: MovementEnergyConfig | None = None) -> None:
        """Initialize the movement energy tracker.

        Args:
            config: Movement-specific configuration. Uses defaults if None.
        """
        super().__init__(config or MovementEnergyConfig())
        self._move_config = config or MovementEnergyConfig()

        # Movement-specific accumulators
        self._total_angular_distance = 0.0       # Total degrees moved
        self._total_translation_distance = 0.0   # Total mm moved
        self._total_movement_time = 0.0          # Total seconds of movement
        self._movement_count = 0

    def record(
        self,
        delta_yaw: float = 0.0,
        delta_pitch: float = 0.0,
        delta_roll: float = 0.0,
        delta_x: float = 0.0,
        delta_y: float = 0.0,
        delta_z: float = 0.0,
        duration_s: float = 0.0,
        movement_type: str = "head",
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> EnergySignal:
        """Record a movement command.

        Args:
            delta_yaw: Yaw change in degrees.
            delta_pitch: Pitch change in degrees.
            delta_roll: Roll change in degrees.
            delta_x: X translation in mm.
            delta_y: Y translation in mm.
            delta_z: Z translation in mm.
            duration_s: Movement duration in seconds.
            movement_type: Type of movement ("head", "antenna", "body").
            context: Additional context.

        Returns:
            EnergySignal representing this movement's energy cost.
        """
        # Calculate angular distance
        angular_distance = math.sqrt(
            delta_yaw**2 + delta_pitch**2 + delta_roll**2
        )

        # Calculate translation distance
        translation_distance = math.sqrt(
            delta_x**2 + delta_y**2 + delta_z**2
        )

        # Calculate angular velocity for speed multiplier
        if duration_s > 0:
            angular_velocity = angular_distance / duration_s
        else:
            angular_velocity = angular_distance / 0.1  # Assume 100ms if not specified

        # Calculate base energy
        angular_energy = angular_distance * self._move_config.angular_energy_per_degree
        translation_energy = translation_distance * self._move_config.translation_energy_per_mm
        duration_energy = duration_s * self._move_config.duration_cost_per_second

        # Apply speed multiplier
        if angular_velocity > self._move_config.fast_speed_threshold:
            speed_multiplier = self._move_config.fast_speed_multiplier
        else:
            speed_multiplier = self._move_config.speed_multiplier_base

        # Apply component multiplier
        if movement_type == "antenna":
            component_multiplier = self._move_config.antenna_energy_multiplier
        elif movement_type == "body":
            component_multiplier = self._move_config.body_rotation_multiplier
        else:
            component_multiplier = 1.0

        # Total energy
        total_energy = (
            (angular_energy + translation_energy) * speed_multiplier * component_multiplier
            + duration_energy
        )

        # Build context
        signal_context = context or {}
        signal_context.update({
            "delta_yaw": round(delta_yaw, 2),
            "delta_pitch": round(delta_pitch, 2),
            "delta_roll": round(delta_roll, 2),
            "angular_distance": round(angular_distance, 2),
            "translation_distance": round(translation_distance, 2),
            "angular_velocity": round(angular_velocity, 2),
            "duration_s": round(duration_s, 3),
            "movement_type": movement_type,
            "speed_multiplier": speed_multiplier,
        })

        signal = EnergySignal(
            energy_type=EnergyType.MOTOR_COMMAND,
            amount=total_energy,
            timestamp=time.time(),
            source=self.name,
            duration_ms=duration_s * 1000,
            context=signal_context,
        )

        # Update movement-specific accumulators
        with self._lock:
            self._total_angular_distance += angular_distance
            self._total_translation_distance += translation_distance
            self._total_movement_time += duration_s
            self._movement_count += 1

        self._record_signal(signal)
        return signal

    def record_head_movement(
        self,
        delta_yaw: float = 0.0,
        delta_pitch: float = 0.0,
        duration_s: float = 0.3,
        context: dict[str, Any] | None = None,
    ) -> EnergySignal:
        """Convenience method for recording head movements.

        Args:
            delta_yaw: Yaw change in degrees.
            delta_pitch: Pitch change in degrees.
            duration_s: Movement duration.
            context: Additional context.

        Returns:
            EnergySignal for this movement.
        """
        return self.record(
            delta_yaw=delta_yaw,
            delta_pitch=delta_pitch,
            duration_s=duration_s,
            movement_type="head",
            context=context,
        )

    def record_from_signature(
        self,
        action_signature: str,
        duration_s: float = 0.3,
        context: dict[str, Any] | None = None,
    ) -> EnergySignal | None:
        """Record energy from an action signature string.

        Parses signatures like "look_at:dy=90:dp=30" to extract
        movement parameters.

        Args:
            action_signature: Action signature to parse.
            duration_s: Movement duration.
            context: Additional context.

        Returns:
            EnergySignal if parsed successfully, None otherwise.
        """
        match = self._LOOK_AT_PATTERN.match(action_signature)
        if not match:
            return None

        delta_yaw = float(match.group(1))
        delta_pitch = float(match.group(2))

        ctx = context or {}
        ctx["action_signature"] = action_signature

        return self.record_head_movement(
            delta_yaw=delta_yaw,
            delta_pitch=delta_pitch,
            duration_s=duration_s,
            context=ctx,
        )

    def record_antenna_movement(
        self,
        left_delta: float = 0.0,
        right_delta: float = 0.0,
        duration_s: float = 0.3,
        context: dict[str, Any] | None = None,
    ) -> EnergySignal:
        """Record antenna movement energy.

        Args:
            left_delta: Left antenna angle change in degrees.
            right_delta: Right antenna angle change in degrees.
            duration_s: Movement duration.
            context: Additional context.

        Returns:
            EnergySignal for this movement.
        """
        ctx = context or {}
        ctx["left_delta"] = left_delta
        ctx["right_delta"] = right_delta

        # Combine antenna movements as pseudo-yaw/pitch
        return self.record(
            delta_yaw=left_delta,
            delta_pitch=right_delta,
            duration_s=duration_s,
            movement_type="antenna",
            context=ctx,
        )

    def record_body_rotation(
        self,
        angle: float,
        duration_s: float = 5.0,
        context: dict[str, Any] | None = None,
    ) -> EnergySignal:
        """Record body rotation (turn_around) energy.

        Body rotation uses the wheel base and is more expensive
        than head movements.

        Args:
            angle: Rotation angle in degrees.
            duration_s: Movement duration.
            context: Additional context.

        Returns:
            EnergySignal for this movement.
        """
        ctx = context or {}
        ctx["body_rotation_angle"] = angle

        return self.record(
            delta_yaw=angle,
            duration_s=duration_s,
            movement_type="body",
            context=ctx,
        )

    def get_movement_stats(self, window_seconds: float | None = None) -> dict[str, Any]:
        """Get movement-specific statistics.

        Args:
            window_seconds: Window for rate calculations.

        Returns:
            Dict with distances, times, and counts.
        """
        base_stats = self.get_window_stats(window_seconds)

        with self._lock:
            return {
                **base_stats,
                # Totals
                "total_angular_distance": round(self._total_angular_distance, 2),
                "total_translation_distance": round(self._total_translation_distance, 2),
                "total_movement_time": round(self._total_movement_time, 2),
                "movement_count": self._movement_count,
                # Averages
                "avg_angular_distance": round(
                    self._total_angular_distance / max(self._movement_count, 1), 2
                ),
                "avg_movement_duration": round(
                    self._total_movement_time / max(self._movement_count, 1), 3
                ),
            }

    def clear(self) -> None:
        """Clear history and reset movement-specific counters."""
        super().clear()
        with self._lock:
            self._total_angular_distance = 0.0
            self._total_translation_distance = 0.0
            self._total_movement_time = 0.0
            self._movement_count = 0


__all__ = [
    "MovementEnergyTracker",
    "MovementEnergyConfig",
]
