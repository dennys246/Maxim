"""Movement harm predictor - Predicts pain from movement commands.

Analyzes movement parameters BEFORE execution to predict if the movement
will cause pain (excessive velocity, acceleration, etc.).

This implements Option A: Predictive Pain with zero latency and zero
additional SDK calls.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

from maxim.harm.predictor import HarmCategory, HarmPrediction, HarmPredictor

logger = logging.getLogger(__name__)


@dataclass
class MovementHarmConfig:
    """Configuration for movement harm prediction.

    Attributes:
        angular_velocity_threshold: Velocity (deg/sec) that triggers harm prediction.
        translation_velocity_threshold: Translation velocity (mm/sec) threshold.
        default_duration: Default movement duration if not specified.
        intensity_scale_factor: Factor to convert excess velocity to intensity.
        confidence_base: Base confidence for predictions.
    """

    angular_velocity_threshold: float = 100.0  # deg/sec
    translation_velocity_threshold: float = 50.0  # mm/sec
    default_duration: float = 0.3  # seconds
    intensity_scale_factor: float = 0.005  # excess velocity -> intensity
    confidence_base: float = 0.8  # High confidence in physics-based prediction


class MovementHarmPredictor(HarmPredictor):
    """Predicts harm from movement commands.

    Analyzes the action_signature (e.g., "look_at:dy=90:dp=30") and duration
    to compute expected velocity. If velocity exceeds threshold, predicts harm.

    This is purely physics-based prediction with no SDK calls, making it
    extremely fast (~0.01ms per prediction).

    Example:
        predictor = MovementHarmPredictor(config=MovementHarmConfig(
            angular_velocity_threshold=100.0,
        ))

        prediction = predictor.predict(
            action_type="movement",
            action_params={
                "action_signature": "look_at:dy=90:dp=30",
                "duration": 0.3,
            },
        )

        if prediction:
            print(f"Predicted harm: {prediction.reason}")
    """

    name = "movement"
    categories = {
        HarmCategory.MOVEMENT_VELOCITY,
        HarmCategory.MOVEMENT_ACCELERATION,
        HarmCategory.JOINT_LIMIT,
    }

    # Patterns for parsing action signatures
    _LOOK_AT_PATTERN = re.compile(r"look_at:dy=(-?[\d.]+):dp=(-?[\d.]+)")
    _MOVE_PATTERN = re.compile(r"move:dx=(-?[\d.]+):dy=(-?[\d.]+):dz=(-?[\d.]+)")

    def __init__(self, config: MovementHarmConfig | None = None) -> None:
        """Initialize the movement harm predictor.

        Args:
            config: Configuration for thresholds. Uses defaults if None.
        """
        self.config = config or MovementHarmConfig()

    def predict(
        self,
        action_type: str,
        action_params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> HarmPrediction | None:
        """Predict harm from a movement command.

        Args:
            action_type: Should be "movement", "dn_movement", or similar.
            action_params: Must contain "action_signature" and optionally "duration".
            context: Optional context (unused currently).

        Returns:
            HarmPrediction if harm predicted, None otherwise.
        """
        # Only handle movement actions
        if action_type not in ("movement", "dn_movement", "look_at", "move"):
            return None

        action_sig = action_params.get("action_signature", "")
        if not action_sig:
            return None

        duration = action_params.get("duration", self.config.default_duration)
        if duration <= 0:
            duration = self.config.default_duration

        # Try to parse and predict harm
        prediction = self._predict_from_look_at(action_sig, duration, action_params)
        if prediction:
            return prediction

        prediction = self._predict_from_move(action_sig, duration, action_params)
        if prediction:
            return prediction

        return None

    def _predict_from_look_at(
        self,
        action_sig: str,
        duration: float,
        action_params: dict[str, Any],
    ) -> HarmPrediction | None:
        """Predict harm from look_at action signature."""
        match = self._LOOK_AT_PATTERN.match(action_sig)
        if not match:
            return None

        try:
            dy = abs(float(match.group(1)))  # yaw delta in degrees
            dp = abs(float(match.group(2)))  # pitch delta in degrees
        except (ValueError, IndexError):
            return None

        # Compute angular displacement and velocity
        angular_delta = math.sqrt(dy**2 + dp**2)
        angular_velocity = angular_delta / duration

        # Check against threshold
        threshold = self.config.angular_velocity_threshold
        if angular_velocity <= threshold:
            return None

        # Calculate intensity based on how much over threshold
        excess = angular_velocity - threshold
        intensity = min(1.0, excess * self.config.intensity_scale_factor)

        # Determine mitigation suggestion
        # Suggest slower movement that would be under threshold
        safe_duration = angular_delta / (threshold * 0.9)  # 90% of threshold
        mitigation = f"Increase duration to {safe_duration:.2f}s for safe velocity"

        return HarmPrediction(
            category=HarmCategory.MOVEMENT_VELOCITY,
            intensity=intensity,
            confidence=self.config.confidence_base,
            reason=(
                f"Expected angular velocity {angular_velocity:.0f}°/s "
                f"exceeds threshold {threshold:.0f}°/s "
                f"(delta={angular_delta:.0f}°, duration={duration:.2f}s)"
            ),
            source=self.name,
            action_type="movement",
            action_params=action_params,
            mitigation=mitigation,
            metadata={
                "angular_delta": angular_delta,
                "angular_velocity": angular_velocity,
                "threshold": threshold,
                "duration": duration,
                "dy": dy,
                "dp": dp,
            },
        )

    def _predict_from_move(
        self,
        action_sig: str,
        duration: float,
        action_params: dict[str, Any],
    ) -> HarmPrediction | None:
        """Predict harm from move action signature (translation)."""
        match = self._MOVE_PATTERN.match(action_sig)
        if not match:
            return None

        try:
            dx = abs(float(match.group(1)))  # x delta in mm
            dy = abs(float(match.group(2)))  # y delta in mm
            dz = abs(float(match.group(3)))  # z delta in mm
        except (ValueError, IndexError):
            return None

        # Compute translation displacement and velocity
        translation_delta = math.sqrt(dx**2 + dy**2 + dz**2)
        translation_velocity = translation_delta / duration

        # Check against threshold
        threshold = self.config.translation_velocity_threshold
        if translation_velocity <= threshold:
            return None

        # Calculate intensity
        excess = translation_velocity - threshold
        intensity = min(1.0, excess * self.config.intensity_scale_factor)

        # Mitigation
        safe_duration = translation_delta / (threshold * 0.9)
        mitigation = f"Increase duration to {safe_duration:.2f}s for safe velocity"

        return HarmPrediction(
            category=HarmCategory.MOVEMENT_VELOCITY,
            intensity=intensity,
            confidence=self.config.confidence_base,
            reason=(
                f"Expected translation velocity {translation_velocity:.0f}mm/s "
                f"exceeds threshold {threshold:.0f}mm/s "
                f"(delta={translation_delta:.0f}mm, duration={duration:.2f}s)"
            ),
            source=self.name,
            action_type="movement",
            action_params=action_params,
            mitigation=mitigation,
            metadata={
                "translation_delta": translation_delta,
                "translation_velocity": translation_velocity,
                "threshold": threshold,
                "duration": duration,
                "dx": dx,
                "dy": dy,
                "dz": dz,
            },
        )

    def can_predict(self, action_type: str) -> bool:
        """Check if this predictor handles the action type."""
        return action_type in ("movement", "dn_movement", "look_at", "move")


def parse_action_signature(action_sig: str) -> dict[str, float]:
    """Parse an action signature into its components.

    Utility function for external use.

    Args:
        action_sig: Action signature string.

    Returns:
        Dict with parsed values, or empty dict if parsing fails.
    """
    # Try look_at pattern
    match = MovementHarmPredictor._LOOK_AT_PATTERN.match(action_sig)
    if match:
        return {
            "dy": float(match.group(1)),
            "dp": float(match.group(2)),
            "type": "look_at",
        }

    # Try move pattern
    match = MovementHarmPredictor._MOVE_PATTERN.match(action_sig)
    if match:
        return {
            "dx": float(match.group(1)),
            "dy": float(match.group(2)),
            "dz": float(match.group(3)),
            "type": "move",
        }

    return {}


__all__ = [
    "MovementHarmPredictor",
    "MovementHarmConfig",
    "parse_action_signature",
]
