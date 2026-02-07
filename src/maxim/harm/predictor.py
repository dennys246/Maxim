"""Core abstractions for predictive harm detection.

Defines the protocol for harm predictors and common data structures
used across all subsystems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HarmCategory(Enum):
    """Categories of potential harm.

    These categories help organize and prioritize different types of
    harmful outcomes that the system can predict and mitigate.
    """

    # Physical/Motor harm
    MOVEMENT_VELOCITY = "movement_velocity"  # Too fast movement
    MOVEMENT_ACCELERATION = "movement_acceleration"  # Too rapid speed change
    MOVEMENT_THRASHING = "movement_thrashing"  # Rapid direction changes
    MOTOR_STALL = "motor_stall"  # Motor unable to reach position
    JOINT_LIMIT = "joint_limit"  # Near or at joint limits

    # Computational harm
    LLM_TIMEOUT = "llm_timeout"  # LLM likely to timeout
    MEMORY_EXHAUSTION = "memory_exhaustion"  # Memory pressure
    CONTEXT_OVERFLOW = "context_overflow"  # Context window exceeded

    # Social/Interaction harm
    USER_FRUSTRATION = "user_frustration"  # User likely frustrated
    REPEATED_FAILURE = "repeated_failure"  # Same action failing repeatedly
    CONVERSATION_ABANDON = "conversation_abandon"  # User disengaging

    # Sensory harm
    TRACKING_LOSS = "tracking_loss"  # Likely to lose visual tracking
    AUDIO_DISTORTION = "audio_distortion"  # Audio quality issues

    # Task/Goal harm
    TOOL_FAILURE = "tool_failure"  # Tool likely to fail
    GOAL_UNREACHABLE = "goal_unreachable"  # Goal cannot be achieved

    # Generic
    UNKNOWN = "unknown"


@dataclass
class HarmPrediction:
    """A prediction about potential harm from an action.

    Attributes:
        category: The type of harm predicted.
        intensity: Severity from 0.0 (none) to 1.0 (severe).
        confidence: How confident this prediction is (0.0-1.0).
        reason: Human-readable explanation.
        source: Which predictor generated this (for debugging).
        action_type: The action type that was evaluated.
        action_params: The action parameters that were evaluated.
        mitigation: Optional suggested mitigation action.
        metadata: Additional predictor-specific data.
    """

    category: HarmCategory
    intensity: float  # 0-1 severity
    confidence: float  # 0-1 confidence in prediction
    reason: str
    source: str  # Predictor name

    action_type: str = ""
    action_params: dict[str, Any] = field(default_factory=dict)
    mitigation: str = ""  # Suggested mitigation
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_score(self) -> float:
        """Combined risk score (intensity * confidence)."""
        return self.intensity * self.confidence

    @property
    def should_gate(self) -> bool:
        """Whether this prediction suggests gating the action.

        Default threshold is risk_score >= 0.3, but callers can
        use their own logic based on intensity/confidence.
        """
        return self.risk_score >= 0.3

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "category": self.category.value,
            "intensity": round(self.intensity, 3),
            "confidence": round(self.confidence, 3),
            "risk_score": round(self.risk_score, 3),
            "reason": self.reason,
            "source": self.source,
            "action_type": self.action_type,
            "mitigation": self.mitigation,
            "metadata": self.metadata,
        }


class HarmPredictor(ABC):
    """Abstract base class for harm predictors.

    Each subsystem (movement, computation, social, etc.) implements
    a predictor that analyzes action parameters and returns predictions
    about potential harmful outcomes.

    Predictors should be:
    - Fast: Called before every action, so must complete in <10ms
    - Stateless: No side effects, pure prediction
    - Specific: Clear about what types of harm they can predict

    Example implementation:
        class MovementHarmPredictor(HarmPredictor):
            name = "movement"
            categories = {HarmCategory.MOVEMENT_VELOCITY, HarmCategory.MOVEMENT_THRASHING}

            def predict(self, action_type, action_params, context):
                if action_type != "movement":
                    return None
                # Analyze movement parameters...
                return HarmPrediction(...)
    """

    # Subclasses should set these
    name: str = "base"
    categories: set[HarmCategory] = set()

    @abstractmethod
    def predict(
        self,
        action_type: str,
        action_params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> HarmPrediction | None:
        """Predict potential harm from an action.

        Args:
            action_type: Type of action (e.g., "movement", "tool_call").
            action_params: Parameters for the action.
            context: Optional context about current state.

        Returns:
            HarmPrediction if harm is predicted, None otherwise.
            If multiple harms are possible, return the most severe.
        """
        ...

    def can_predict(self, action_type: str) -> bool:
        """Check if this predictor handles the given action type.

        Override this for more complex action type matching.
        Default returns True (predictor decides in predict()).
        """
        return True


@dataclass
class AggregatedHarmPrediction:
    """Aggregated prediction from multiple predictors.

    When multiple predictors return predictions, this aggregates them
    into a single result with the most severe prediction highlighted.
    """

    predictions: list[HarmPrediction]
    action_type: str
    action_params: dict[str, Any]

    @property
    def worst(self) -> HarmPrediction | None:
        """Get the prediction with highest risk score."""
        if not self.predictions:
            return None
        return max(self.predictions, key=lambda p: p.risk_score)

    @property
    def max_intensity(self) -> float:
        """Maximum intensity across all predictions."""
        if not self.predictions:
            return 0.0
        return max(p.intensity for p in self.predictions)

    @property
    def max_risk_score(self) -> float:
        """Maximum risk score across all predictions."""
        if not self.predictions:
            return 0.0
        return max(p.risk_score for p in self.predictions)

    @property
    def should_gate(self) -> bool:
        """Whether any prediction suggests gating."""
        return any(p.should_gate for p in self.predictions)

    @property
    def reasons(self) -> list[str]:
        """All reasons from predictions that suggest gating."""
        return [p.reason for p in self.predictions if p.should_gate]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "action_type": self.action_type,
            "prediction_count": len(self.predictions),
            "max_intensity": round(self.max_intensity, 3),
            "max_risk_score": round(self.max_risk_score, 3),
            "should_gate": self.should_gate,
            "predictions": [p.to_dict() for p in self.predictions],
        }


__all__ = [
    "HarmCategory",
    "HarmPrediction",
    "HarmPredictor",
    "AggregatedHarmPrediction",
]
