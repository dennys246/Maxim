"""Joint limit harm predictor - Predicts motor stall from workspace limits.

Analyzes movement targets against known workspace bounds to predict
whether the robot can reach the commanded position. Uses data from
WorkspaceBoundsLearner when available.

This catches "motor stall" situations where the robot commands a position
but physically cannot reach it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from maxim.harm.predictor import HarmCategory, HarmPrediction, HarmPredictor

if TYPE_CHECKING:
    from maxim.spatial import WorkspaceBoundsLearner

logger = logging.getLogger(__name__)


@dataclass
class JointLimitConfig:
    """Configuration for joint limit prediction.

    Attributes:
        yaw_limit: Maximum yaw angle (degrees, symmetric).
        pitch_limit_up: Maximum pitch up (degrees, positive).
        pitch_limit_down: Maximum pitch down (degrees, negative).
        y_limit: Maximum y translation (mm, symmetric).
        z_limit_up: Maximum z up (mm).
        z_limit_down: Maximum z down (mm).
        warning_threshold: Fraction of limit that triggers warning (0-1).
        danger_threshold: Fraction of limit that triggers high risk (0-1).
        use_learned_bounds: Whether to use WorkspaceBoundsLearner data.
        confidence_base: Base confidence for predictions.
    """

    # Default hardware limits (conservative estimates)
    yaw_limit: float = 45.0  # degrees
    pitch_limit_up: float = 30.0  # degrees
    pitch_limit_down: float = -30.0  # degrees
    y_limit: float = 20.0  # mm
    z_limit_up: float = 20.0  # mm
    z_limit_down: float = -20.0  # mm

    # Risk thresholds
    warning_threshold: float = 0.7  # 70% of limit = warning
    danger_threshold: float = 0.9  # 90% of limit = danger

    # Learning integration
    use_learned_bounds: bool = True

    # Confidence
    confidence_base: float = 0.75


class JointLimitHarmPredictor(HarmPredictor):
    """Predicts harm from movements near or beyond joint limits.

    Analyzes the target position (extracted from action signature or
    provided directly) against known workspace bounds. Returns harm
    predictions when movements target positions that:
    - Are near the edge of safe workspace (warning)
    - Are at or beyond physical limits (danger)
    - Have historically failed (learned from WorkspaceBoundsLearner)

    Example:
        predictor = JointLimitHarmPredictor(config=JointLimitConfig(
            yaw_limit=45.0,
        ))

        # With bounds learner for learned limits
        predictor = JointLimitHarmPredictor(
            config=JointLimitConfig(),
            bounds_learner=workspace_bounds_learner,
        )

        prediction = predictor.predict(
            action_type="movement",
            action_params={
                "target_yaw": 42.0,  # Near limit
                "target_pitch": 25.0,
            },
        )

        if prediction:
            print(f"Near limit: {prediction.reason}")
    """

    name = "joint_limit"
    categories = {
        HarmCategory.JOINT_LIMIT,
        HarmCategory.MOTOR_STALL,
    }

    # Pattern for parsing action signatures with absolute positions
    _LOOK_AT_PATTERN = re.compile(r"look_at:dy=(-?[\d.]+):dp=(-?[\d.]+)")

    def __init__(
        self,
        config: JointLimitConfig | None = None,
        bounds_learner: "WorkspaceBoundsLearner | None" = None,
    ) -> None:
        """Initialize the joint limit predictor.

        Args:
            config: Configuration for limits. Uses defaults if None.
            bounds_learner: Optional bounds learner for learned limits.
        """
        self.config = config or JointLimitConfig()
        self._bounds_learner = bounds_learner

    def set_bounds_learner(self, bounds_learner: "WorkspaceBoundsLearner") -> None:
        """Set or update the bounds learner.

        Args:
            bounds_learner: The bounds learner to use.
        """
        self._bounds_learner = bounds_learner

    def predict(
        self,
        action_type: str,
        action_params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> HarmPrediction | None:
        """Predict harm from movement near joint limits.

        Args:
            action_type: Should be "movement", "dn_movement", or similar.
            action_params: Can contain:
                - target_yaw, target_pitch: Absolute target positions
                - current_yaw, current_pitch: Current positions
                - action_signature: "look_at:dy=X:dp=Y" for relative movement
            context: Optional context (can include current position).

        Returns:
            HarmPrediction if harm predicted, None otherwise.
        """
        if action_type not in ("movement", "dn_movement", "look_at", "move"):
            return None

        # Try to get target position
        target = self._extract_target_position(action_params, context)
        if target is None:
            return None

        target_yaw, target_pitch = target

        # Check against limits
        prediction = self._check_limits(target_yaw, target_pitch, action_params)
        if prediction:
            return prediction

        # Check against learned bounds if available
        if self._bounds_learner and self.config.use_learned_bounds:
            prediction = self._check_learned_bounds(target_yaw, target_pitch, action_params)
            if prediction:
                return prediction

        return None

    def _extract_target_position(
        self,
        action_params: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> tuple[float, float] | None:
        """Extract target yaw/pitch from action params or context."""
        # Direct target position
        if "target_yaw" in action_params and "target_pitch" in action_params:
            return (
                float(action_params["target_yaw"]),
                float(action_params["target_pitch"]),
            )

        # Compute from current position + delta
        action_sig = action_params.get("action_signature", "")
        if action_sig:
            match = self._LOOK_AT_PATTERN.match(action_sig)
            if match:
                dy = float(match.group(1))
                dp = float(match.group(2))

                # Get current position from context or params
                current_yaw = 0.0
                current_pitch = 0.0

                if context:
                    current_yaw = float(context.get("current_yaw", 0.0))
                    current_pitch = float(context.get("current_pitch", 0.0))
                elif "current_yaw" in action_params:
                    current_yaw = float(action_params.get("current_yaw", 0.0))
                    current_pitch = float(action_params.get("current_pitch", 0.0))

                return (current_yaw + dy, current_pitch + dp)

        return None

    def _check_limits(
        self,
        target_yaw: float,
        target_pitch: float,
        action_params: dict[str, Any],
    ) -> HarmPrediction | None:
        """Check target against configured limits."""
        issues = []
        max_usage = 0.0

        # Check yaw
        yaw_usage = abs(target_yaw) / self.config.yaw_limit if self.config.yaw_limit > 0 else 0
        if yaw_usage >= self.config.danger_threshold:
            issues.append(f"yaw {target_yaw:.1f}° at {yaw_usage * 100:.0f}% of limit")
            max_usage = max(max_usage, yaw_usage)
        elif yaw_usage >= self.config.warning_threshold:
            issues.append(f"yaw {target_yaw:.1f}° near limit ({yaw_usage * 100:.0f}%)")
            max_usage = max(max_usage, yaw_usage)

        # Check pitch
        if target_pitch > 0:
            pitch_limit = self.config.pitch_limit_up
        else:
            pitch_limit = abs(self.config.pitch_limit_down)

        pitch_usage = abs(target_pitch) / pitch_limit if pitch_limit > 0 else 0
        if pitch_usage >= self.config.danger_threshold:
            issues.append(f"pitch {target_pitch:.1f}° at {pitch_usage * 100:.0f}% of limit")
            max_usage = max(max_usage, pitch_usage)
        elif pitch_usage >= self.config.warning_threshold:
            issues.append(f"pitch {target_pitch:.1f}° near limit ({pitch_usage * 100:.0f}%)")
            max_usage = max(max_usage, pitch_usage)

        if not issues:
            return None

        # Calculate intensity based on how close to/beyond limits
        if max_usage >= 1.0:
            intensity = 1.0
            category = HarmCategory.MOTOR_STALL
        elif max_usage >= self.config.danger_threshold:
            intensity = 0.7 + (max_usage - self.config.danger_threshold) * 3
            category = HarmCategory.JOINT_LIMIT
        else:
            intensity = 0.3 + (max_usage - self.config.warning_threshold) * 2
            category = HarmCategory.JOINT_LIMIT

        intensity = min(1.0, max(0.0, intensity))

        # Suggest mitigation
        safe_yaw = target_yaw * (self.config.warning_threshold / max(max_usage, 0.01))
        safe_pitch = target_pitch * (self.config.warning_threshold / max(max_usage, 0.01))
        mitigation = f"Reduce movement to yaw={safe_yaw:.1f}°, pitch={safe_pitch:.1f}°"

        return HarmPrediction(
            category=category,
            intensity=intensity,
            confidence=self.config.confidence_base,
            reason=f"Movement near joint limits: {', '.join(issues)}",
            source=self.name,
            action_type="movement",
            action_params=action_params,
            mitigation=mitigation,
            metadata={
                "target_yaw": target_yaw,
                "target_pitch": target_pitch,
                "max_usage": max_usage,
                "yaw_usage": yaw_usage,
                "pitch_usage": pitch_usage,
                "yaw_limit": self.config.yaw_limit,
                "pitch_limit_up": self.config.pitch_limit_up,
                "pitch_limit_down": self.config.pitch_limit_down,
            },
        )

    def _check_learned_bounds(
        self,
        target_yaw: float,
        target_pitch: float,
        action_params: dict[str, Any],
    ) -> HarmPrediction | None:
        """Check target against learned bounds from WorkspaceBoundsLearner."""
        if not self._bounds_learner:
            return None

        try:
            # Query bounds learner for reachability
            # Assume bounds_learner has methods like get_learned_limits() or is_reachable()
            learned_limits = getattr(self._bounds_learner, "get_learned_limits", None)
            if learned_limits:
                limits = learned_limits()
                if limits:
                    return self._check_against_learned_limits(target_yaw, target_pitch, limits, action_params)

            # Alternative: check historical success rate
            success_rate_fn = getattr(self._bounds_learner, "get_success_rate", None)
            if success_rate_fn:
                # Build approximate position dict
                position = {"yaw": target_yaw, "pitch": target_pitch}
                success_rate = success_rate_fn(position)

                if success_rate is not None and success_rate < 0.5:
                    # More than 50% failure rate for similar positions
                    intensity = 0.5 + (0.5 - success_rate)
                    return HarmPrediction(
                        category=HarmCategory.MOTOR_STALL,
                        intensity=min(1.0, intensity),
                        confidence=0.6,  # Lower confidence for learned predictions
                        reason=(
                            f"Historical failure rate {(1 - success_rate) * 100:.0f}% "
                            f"for positions near yaw={target_yaw:.1f}°, pitch={target_pitch:.1f}°"
                        ),
                        source=self.name,
                        action_type="movement",
                        action_params=action_params,
                        mitigation="Choose a position with higher historical success rate",
                        metadata={
                            "target_yaw": target_yaw,
                            "target_pitch": target_pitch,
                            "success_rate": success_rate,
                            "source": "learned",
                        },
                    )

        except Exception as e:
            logger.debug("Failed to query bounds learner: %s", e)

        return None

    def _check_against_learned_limits(
        self,
        target_yaw: float,
        target_pitch: float,
        limits: dict[str, Any],
        action_params: dict[str, Any],
    ) -> HarmPrediction | None:
        """Check against learned limits dict."""
        issues = []
        max_usage = 0.0

        # Extract learned limits
        yaw_min = limits.get("yaw_min", -self.config.yaw_limit)
        yaw_max = limits.get("yaw_max", self.config.yaw_limit)
        pitch_min = limits.get("pitch_min", self.config.pitch_limit_down)
        pitch_max = limits.get("pitch_max", self.config.pitch_limit_up)

        # Check yaw
        if target_yaw < yaw_min:
            yaw_excess = (yaw_min - target_yaw) / abs(yaw_min) if yaw_min != 0 else 1.0
            issues.append(f"yaw {target_yaw:.1f}° below learned minimum {yaw_min:.1f}°")
            max_usage = max(max_usage, 1.0 + yaw_excess)
        elif target_yaw > yaw_max:
            yaw_excess = (target_yaw - yaw_max) / yaw_max if yaw_max != 0 else 1.0
            issues.append(f"yaw {target_yaw:.1f}° above learned maximum {yaw_max:.1f}°")
            max_usage = max(max_usage, 1.0 + yaw_excess)
        else:
            # Calculate usage within learned range
            yaw_range = yaw_max - yaw_min
            if yaw_range > 0:
                if target_yaw >= 0:
                    yaw_usage = target_yaw / yaw_max if yaw_max > 0 else 0
                else:
                    yaw_usage = abs(target_yaw) / abs(yaw_min) if yaw_min < 0 else 0
                if yaw_usage >= self.config.warning_threshold:
                    issues.append(f"yaw {target_yaw:.1f}° at {yaw_usage * 100:.0f}% of learned range")
                    max_usage = max(max_usage, yaw_usage)

        # Check pitch
        if target_pitch < pitch_min:
            pitch_excess = (pitch_min - target_pitch) / abs(pitch_min) if pitch_min != 0 else 1.0
            issues.append(f"pitch {target_pitch:.1f}° below learned minimum {pitch_min:.1f}°")
            max_usage = max(max_usage, 1.0 + pitch_excess)
        elif target_pitch > pitch_max:
            pitch_excess = (target_pitch - pitch_max) / pitch_max if pitch_max != 0 else 1.0
            issues.append(f"pitch {target_pitch:.1f}° above learned maximum {pitch_max:.1f}°")
            max_usage = max(max_usage, 1.0 + pitch_excess)
        else:
            pitch_range = pitch_max - pitch_min
            if pitch_range > 0:
                if target_pitch >= 0:
                    pitch_usage = target_pitch / pitch_max if pitch_max > 0 else 0
                else:
                    pitch_usage = abs(target_pitch) / abs(pitch_min) if pitch_min < 0 else 0
                if pitch_usage >= self.config.warning_threshold:
                    issues.append(f"pitch {target_pitch:.1f}° at {pitch_usage * 100:.0f}% of learned range")
                    max_usage = max(max_usage, pitch_usage)

        if not issues:
            return None

        # Calculate intensity
        if max_usage >= 1.0:
            intensity = 0.8 + min(0.2, (max_usage - 1.0) * 0.5)
            category = HarmCategory.MOTOR_STALL
        else:
            intensity = 0.3 + (max_usage - self.config.warning_threshold) * 2
            category = HarmCategory.JOINT_LIMIT

        intensity = min(1.0, max(0.0, intensity))

        return HarmPrediction(
            category=category,
            intensity=intensity,
            confidence=0.7,  # Higher confidence for learned limits
            reason=f"Movement exceeds learned workspace: {', '.join(issues)}",
            source=self.name,
            action_type="movement",
            action_params=action_params,
            mitigation=f"Stay within learned bounds: yaw=[{yaw_min:.1f}°, {yaw_max:.1f}°], pitch=[{pitch_min:.1f}°, {pitch_max:.1f}°]",
            metadata={
                "target_yaw": target_yaw,
                "target_pitch": target_pitch,
                "max_usage": max_usage,
                "learned_yaw_min": yaw_min,
                "learned_yaw_max": yaw_max,
                "learned_pitch_min": pitch_min,
                "learned_pitch_max": pitch_max,
                "source": "learned",
            },
        )

    def can_predict(self, action_type: str) -> bool:
        """Check if this predictor handles the action type."""
        return action_type in ("movement", "dn_movement", "look_at", "move")


__all__ = [
    "JointLimitHarmPredictor",
    "JointLimitConfig",
]
