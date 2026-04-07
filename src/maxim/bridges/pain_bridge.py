"""PainCircuitBridge - Connects pain detection to NAc causal learning.

Enables the robot to learn from painful movement patterns and predict
future pain, allowing FearAgent to gate actions that would cause discomfort.

Supports two modes of pain prediction:
1. Predictive harm: Physics-based prediction from command parameters (zero latency)
2. Learned prediction: Queries NAc for patterns learned from past pain events

The predictive harm system provides zero-latency prediction by analyzing
movement commands before execution, without requiring historical data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from maxim.harm import HarmRegistry, MovementHarmPredictor, JointLimitHarmPredictor
from maxim.harm.movement import MovementHarmConfig
from maxim.harm.joint_limit import JointLimitConfig

if TYPE_CHECKING:
    from maxim.decisions.causal_link import OutcomePrediction
    from maxim.decisions.nac import NAc
    from maxim.harm.predictor import HarmPrediction
    from maxim.proprioception.pain import PainDetector, PainSignal
    from maxim.spatial import WorkspaceBoundsLearner

logger = logging.getLogger(__name__)


@dataclass
class PainBridgeConfig:
    """Configuration for pain bridge.

    Attributes:
        enable_learning: Whether to learn from pain events.
        enable_predictive_harm: Whether to use physics-based prediction (Option A).
        enable_joint_limit_prediction: Whether to predict joint limit violations.
        pain_to_valence_threshold: Pain intensity threshold for NEGATIVE valence.
        prediction_threshold: Confidence threshold for pain predictions.
        predictive_harm_threshold: Risk threshold for predictive harm gating.
        action_timeout_seconds: How long an action is considered "pending".
        angular_velocity_threshold: Threshold for movement velocity prediction.
        default_duration: Default movement duration for predictions.
        yaw_limit: Maximum yaw angle for joint limit prediction.
        pitch_limit: Maximum pitch angle for joint limit prediction.
        use_learned_bounds: Whether to use WorkspaceBoundsLearner for limits.
    """

    enable_learning: bool = True
    enable_predictive_harm: bool = True  # Physics-based prediction (Option A)
    enable_joint_limit_prediction: bool = True  # Joint limit prediction
    pain_to_valence_threshold: float = 0.5  # Pain intensity for NEGATIVE valence
    prediction_threshold: float = 0.4  # Min confidence for learned prediction
    predictive_harm_threshold: float = 0.3  # Min risk for predictive gating
    action_timeout_seconds: float = 5.0  # Max time to attribute pain to action
    angular_velocity_threshold: float = 100.0  # deg/sec for predictive harm
    default_duration: float = 0.3  # Default movement duration
    # Joint limit settings
    yaw_limit: float = 45.0  # degrees
    pitch_limit: float = 30.0  # degrees
    use_learned_bounds: bool = True  # Use WorkspaceBoundsLearner data


class PainCircuitBridge:
    """Connects pain detection to NAc causal learning.

    Learning flow:
    1. Action is about to execute -> record_action_start(action_signature)
    2. Position updates -> pain detector checks thresholds
    3. If pain detected: report to NAc as negative outcome
    4. Future actions: predict_pain() queries NAc before execution

    Example:
        bridge = PainCircuitBridge(nac=nac, pain_detector=detector)

        # Before movement
        bridge.record_action_start("look_at:dy=90:dp=10")

        # Position updates happen in sync_head_position
        # Pain detector triggers callback if thresholds exceeded

        # Before next similar movement
        should_gate, reason = bridge.should_gate_action("look_at:dy=85:dp=5")
        if should_gate:
            print(f"Movement gated: {reason}")
    """

    def __init__(
        self,
        nac: "NAc",
        pain_detector: "PainDetector",
        config: PainBridgeConfig | None = None,
        pain_bus: Any | None = None,
    ) -> None:
        """Initialize the pain circuit bridge.

        Args:
            nac: The NAc instance for causal learning.
            pain_detector: The pain detector instance (kept for movement tracking).
            config: Optional configuration.
            pain_bus: Optional PainBus for pain signal subscription. When
                provided, subscribes to the bus instead of using
                pain_detector.add_pain_callback().
        """
        self._nac = nac
        self._detector = pain_detector
        self.config = config or PainBridgeConfig()

        # Current action tracking
        self._pending_action: str | None = None
        self._pending_event_id: str | None = None
        self._action_start_time: float = 0.0
        self._action_context: dict[str, Any] = {}

        # Statistics
        self._total_pain_attributed = 0
        self._total_predictions = 0
        self._total_predictive_checks = 0
        self._predictive_gates = 0
        self._learned_gates = 0
        self._gated_actions = 0

        # Subscribe to pain signals via bus (preferred) or detector (legacy)
        if pain_bus is not None:
            pain_bus.subscribe(self._on_pain)
        else:
            self._detector.add_pain_callback(self._on_pain)

        # Initialize predictive harm system
        self._harm_registry: HarmRegistry | None = None
        self._joint_limit_predictor: JointLimitHarmPredictor | None = None

        if self.config.enable_predictive_harm or self.config.enable_joint_limit_prediction:
            self._harm_registry = HarmRegistry()

            # Movement velocity predictor (Option A)
            if self.config.enable_predictive_harm:
                movement_config = MovementHarmConfig(
                    angular_velocity_threshold=self.config.angular_velocity_threshold,
                    default_duration=self.config.default_duration,
                )
                self._harm_registry.register(MovementHarmPredictor(config=movement_config))
                logger.info(
                    "Movement velocity predictor enabled (threshold=%.1f deg/s)",
                    self.config.angular_velocity_threshold,
                )

            # Joint limit predictor
            if self.config.enable_joint_limit_prediction:
                joint_config = JointLimitConfig(
                    yaw_limit=self.config.yaw_limit,
                    pitch_limit_up=self.config.pitch_limit,
                    pitch_limit_down=-self.config.pitch_limit,
                    use_learned_bounds=self.config.use_learned_bounds,
                )
                self._joint_limit_predictor = JointLimitHarmPredictor(config=joint_config)
                self._harm_registry.register(self._joint_limit_predictor)
                logger.info(
                    "Joint limit predictor enabled (yaw=±%.1f°, pitch=±%.1f°)",
                    self.config.yaw_limit,
                    self.config.pitch_limit,
                )

        logger.info("PainCircuitBridge initialized")

    def record_action_start(
        self,
        action_signature: str,
        context: dict[str, Any] | None = None,
        *,
        target_yaw: float | None = None,
        target_pitch: float | None = None,
        target_x: float | None = None,
        target_y: float | None = None,
        target_z: float | None = None,
        target_roll: float | None = None,
    ) -> str:
        """Record that an action is starting.

        Call this before executing a movement command. If pain is
        detected afterward, it will be attributed to this action.

        Args:
            action_signature: Unique signature for the action type.
                Example: "look_at:dy=45:dp=-10"
            context: Optional context about the action.
            target_yaw, target_pitch, target_roll: Expected angular targets (degrees).
            target_x, target_y, target_z: Expected translation targets (mm).

        Returns:
            Event ID for tracking.
        """
        self._pending_action = action_signature
        self._action_start_time = time.time()
        self._action_context = context or {}

        # Register with NAc
        self._pending_event_id = self._nac.record_event(
            event_type="movement",
            event_signature=action_signature,
            context=context,
        )

        # Set movement target for failure detection (if any targets provided)
        has_target = any(t is not None for t in (target_yaw, target_pitch, target_x, target_y, target_z, target_roll))
        if has_target:
            self._detector.set_movement_target(
                yaw=target_yaw,
                pitch=target_pitch,
                x=target_x,
                y=target_y,
                z=target_z,
                roll=target_roll,
                action_signature=action_signature,
            )

        logger.debug("Action started: %s (event_id=%s)", action_signature, self._pending_event_id)

        return self._pending_event_id

    def record_action_complete(self, success: bool = True) -> None:
        """Record that an action completed without pain.

        Call this when a movement completes successfully without triggering
        pain. This provides positive feedback to the NAc.

        Args:
            success: Whether the action was successful.
        """
        if not self._pending_action or not self.config.enable_learning:
            # Still clear movement target even if not learning
            self._detector.clear_movement_target()
            return

        from maxim.decisions.causal_link import Valence

        # Clear movement target since action completed normally
        self._detector.clear_movement_target()

        # Determine valence based on success
        valence = Valence.POSITIVE if success else Valence.NEUTRAL

        self._nac.record_outcome(
            event_type="movement",
            event_id=self._pending_action,
            outcome_valence=valence,
            context={"success": success, **self._action_context},
        )

        logger.debug(
            "Action completed without pain: %s (valence=%s)",
            self._pending_action,
            valence.value,
        )

        self._clear_pending_action()

    def _on_pain(self, signal: "PainSignal") -> None:
        """Handle detected pain signal.

        Called by the pain detector when pain thresholds are exceeded.
        Attributes the pain to the pending action (if any) and reports
        to NAc for causal learning.
        """
        if not self.config.enable_learning:
            return

        # Check if we have a pending action and it's not too old
        if not self._pending_action:
            logger.debug("Pain detected but no pending action to attribute")
            return

        elapsed = time.time() - self._action_start_time
        if elapsed > self.config.action_timeout_seconds:
            logger.debug(
                "Pain detected but action too old (%.1fs > %.1fs)",
                elapsed,
                self.config.action_timeout_seconds,
            )
            self._clear_pending_action()
            return

        from maxim.decisions.causal_link import Valence

        # Determine valence from pain intensity
        if signal.intensity >= self.config.pain_to_valence_threshold:
            valence = Valence.NEGATIVE
        else:
            valence = Valence.NEUTRAL

        # Report to NAc
        self._nac.record_outcome(
            event_type="movement",
            event_id=self._pending_action,
            outcome_valence=valence,
            context={
                "pain_type": signal.pain_type.value,
                "intensity": signal.intensity,
                "angular_velocity": signal.angular_velocity,
                "translation_velocity": signal.translation_velocity,
                "direction_reversals": signal.direction_reversals,
                **self._action_context,
            },
        )

        self._total_pain_attributed += 1

        # Log at WARNING for negative valence (high pain), INFO otherwise
        log_msg = "Pain attributed to action: %s -> %s (intensity=%.2f, valence=%s)"
        log_args = (
            self._pending_action,
            signal.pain_type.value,
            signal.intensity,
            valence.value,
        )

        if valence.value == "negative":
            logger.warning(log_msg, *log_args)
        else:
            logger.info(log_msg, *log_args)

        self._clear_pending_action()

    def _clear_pending_action(self) -> None:
        """Clear the pending action tracking."""
        self._pending_action = None
        self._pending_event_id = None
        self._action_start_time = 0.0
        self._action_context = {}

    def predict_pain(
        self,
        action_signature: str,
        context: dict[str, Any] | None = None,
    ) -> "OutcomePrediction | None":
        """Predict if an action will cause pain.

        Queries the NAc for learned causal links between similar actions
        and pain outcomes.

        Args:
            action_signature: The action to predict.
            context: Optional context for the prediction.

        Returns:
            OutcomePrediction if pain is predicted, None otherwise.
        """
        from maxim.decisions.causal_link import Valence

        prediction = self._nac.predict(
            event_type="movement",
            event_signature=action_signature,
            context=context,
        )

        self._total_predictions += 1

        if prediction is None:
            return None

        # Only return if predicts negative outcome with confidence
        if (
            prediction.predicted_valence == Valence.NEGATIVE
            and prediction.confidence >= self.config.prediction_threshold
        ):
            logger.debug(
                "Pain predicted for %s (confidence=%.2f, value=%.2f)",
                action_signature,
                prediction.confidence,
                prediction.predicted_value,
            )
            return prediction

        return None

    def predict_harm(
        self,
        action_signature: str,
        duration: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> "HarmPrediction | None":
        """Predict harm using physics-based analysis (Option A).

        This provides zero-latency prediction by analyzing the action
        parameters directly, without requiring historical data.

        Args:
            action_signature: Action signature (e.g., "look_at:dy=90:dp=30").
            duration: Movement duration in seconds. Uses default if None.
            context: Optional context.

        Returns:
            HarmPrediction if harm predicted, None otherwise.
        """
        if not self._harm_registry:
            return None

        self._total_predictive_checks += 1

        action_params = {
            "action_signature": action_signature,
            "duration": duration or self.config.default_duration,
        }
        if context:
            action_params.update(context)

        prediction = self._harm_registry.predict_worst(
            action_type="movement",
            action_params=action_params,
            context=context,
        )

        if prediction and prediction.risk_score >= self.config.predictive_harm_threshold:
            logger.debug(
                "Predictive harm detected: %s (risk=%.2f, reason=%s)",
                action_signature,
                prediction.risk_score,
                prediction.reason,
            )
            return prediction

        return None

    def should_gate_action(
        self,
        action_signature: str,
        duration: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Check if an action should be gated due to predicted pain.

        Uses a two-tier prediction system:
        1. Predictive harm (Option A): Physics-based, zero latency
        2. Learned prediction: NAc-based, requires historical data

        Args:
            action_signature: The action to check.
            duration: Optional movement duration for predictive harm.
            context: Optional context.

        Returns:
            Tuple of (should_gate, reason).
        """
        # Tier 1: Predictive harm (Option A) - zero latency, physics-based
        harm_prediction = self.predict_harm(action_signature, duration, context)
        if harm_prediction and harm_prediction.should_gate:
            self._gated_actions += 1
            self._predictive_gates += 1
            reason = f"[Predictive] {harm_prediction.reason}"
            if harm_prediction.mitigation:
                reason += f" | Suggestion: {harm_prediction.mitigation}"

            # Log at WARNING for high risk, INFO for moderate
            if harm_prediction.risk_score >= 0.7:
                logger.warning(
                    "HIGH RISK action gated: %s (risk=%.2f, category=%s)",
                    action_signature,
                    harm_prediction.risk_score,
                    harm_prediction.category.value,
                )
            else:
                logger.info(
                    "Action gated by predictive harm: %s (risk=%.2f)",
                    action_signature,
                    harm_prediction.risk_score,
                )
            return True, reason

        # Tier 2: Learned prediction - NAc-based
        learned_prediction = self.predict_pain(action_signature, context)
        if learned_prediction:
            self._gated_actions += 1
            self._learned_gates += 1
            reason = (
                f"[Learned] Predicted pain: {learned_prediction.predicted_outcome} "
                f"(confidence={learned_prediction.confidence:.2f}, "
                f"value={learned_prediction.predicted_value:.2f})"
            )
            logger.info(
                "Action gated by learned pain: %s (confidence=%.2f)",
                action_signature,
                learned_prediction.confidence,
            )
            return True, reason

        return False, ""

    def get_pain_risk(
        self,
        action_signature: str,
        duration: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Get the predicted pain risk for an action (0-1).

        Combines predictive harm (Option A) and learned predictions.
        Returns the maximum risk from either source.

        Args:
            action_signature: The action to assess.
            duration: Optional movement duration for predictive harm.
            context: Optional context.

        Returns:
            Risk score from 0.0 (no risk) to 1.0 (high risk).
        """
        risk_scores = []

        # Predictive harm risk (Option A)
        if self._harm_registry:
            action_params = {
                "action_signature": action_signature,
                "duration": duration or self.config.default_duration,
            }
            predictive_risk = self._harm_registry.get_risk_score(
                action_type="movement",
                action_params=action_params,
                context=context,
            )
            if predictive_risk > 0:
                risk_scores.append(predictive_risk)

        # Learned prediction risk
        prediction = self._nac.predict(
            event_type="movement",
            event_signature=action_signature,
            context=context,
        )

        if prediction is not None:
            from maxim.decisions.causal_link import Valence

            if prediction.predicted_valence == Valence.NEGATIVE:
                # Risk = confidence * (1 - predicted_value)
                learned_risk = prediction.confidence * (1.0 - prediction.predicted_value)
                risk_scores.append(min(1.0, max(0.0, learned_risk)))

        # Return maximum risk from any source
        return max(risk_scores) if risk_scores else 0.0

    @property
    def detector(self) -> "PainDetector":
        """Access the underlying pain detector."""
        return self._detector

    @property
    def harm_registry(self) -> HarmRegistry | None:
        """Access the harm prediction registry.

        Returns None if predictive harm is disabled.
        """
        return self._harm_registry

    @property
    def joint_limit_predictor(self) -> JointLimitHarmPredictor | None:
        """Access the joint limit predictor.

        Returns None if joint limit prediction is disabled.
        """
        return self._joint_limit_predictor

    def set_bounds_learner(self, bounds_learner: "WorkspaceBoundsLearner") -> None:
        """Connect a WorkspaceBoundsLearner for learned limit prediction.

        When set, the joint limit predictor will use learned workspace bounds
        in addition to configured static limits, providing more accurate
        predictions based on actual robot experience.

        Args:
            bounds_learner: The workspace bounds learner to use.
        """
        if self._joint_limit_predictor:
            self._joint_limit_predictor.set_bounds_learner(bounds_learner)
            logger.info("Bounds learner connected to joint limit predictor")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about pain bridge operation."""
        stats = {
            "total_pain_attributed": self._total_pain_attributed,
            "total_predictions": self._total_predictions,
            "total_predictive_checks": self._total_predictive_checks,
            "gated_actions": self._gated_actions,
            "predictive_gates": self._predictive_gates,
            "learned_gates": self._learned_gates,
            "has_pending_action": self._pending_action is not None,
            "predictive_harm_enabled": self._harm_registry is not None,
            "joint_limit_prediction_enabled": self._joint_limit_predictor is not None,
            "detector_stats": self._detector.get_stats(),
        }
        if self._harm_registry:
            stats["harm_registry_stats"] = self._harm_registry.get_stats()
        return stats


__all__ = ["PainCircuitBridge", "PainBridgeConfig"]
