"""Causal link data structures for NAc.

Represents learned causal relationships between events and outcomes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Valence(Enum):
    """Outcome quality classification."""

    POSITIVE = "positive"  # Goal achieved, user satisfied, success
    NEGATIVE = "negative"  # Error, failure, user frustrated
    NEUTRAL = "neutral"  # Expected outcome, no strong signal
    UNKNOWN = "unknown"  # Not yet classified


@dataclass(frozen=True, slots=True)
class TemporalDelta:
    """Learned temporal relationship between event and outcome.

    Rather than fixed delays, we learn the distribution of observed
    delays to enable flexible temporal predictions.
    """

    observed_deltas: tuple[float, ...]  # All observed delays (seconds)

    @property
    def mean(self) -> float:
        """Average observed delay."""
        if not self.observed_deltas:
            return 0.0
        return sum(self.observed_deltas) / len(self.observed_deltas)

    @property
    def min(self) -> float:
        """Fastest observed delay."""
        return min(self.observed_deltas) if self.observed_deltas else 0.0

    @property
    def max(self) -> float:
        """Slowest observed delay."""
        return max(self.observed_deltas) if self.observed_deltas else 0.0

    @property
    def variance(self) -> float:
        """Variance in observed delays (high = unpredictable timing)."""
        if len(self.observed_deltas) < 2:
            return 0.0
        mean = self.mean
        return sum((d - mean) ** 2 for d in self.observed_deltas) / len(self.observed_deltas)

    def add_observation(self, delta: float) -> TemporalDelta:
        """Return new TemporalDelta with added observation."""
        # Keep last 100 observations to prevent unbounded growth
        observations = list(self.observed_deltas)[-99:] + [delta]
        return TemporalDelta(observed_deltas=tuple(observations))

    def predict_delay(self, confidence_interval: float = 0.8) -> tuple[float, float, float]:
        """Predict expected delay with confidence bounds.

        Args:
            confidence_interval: Confidence level (0.0-1.0)

        Returns:
            Tuple of (expected_delay, lower_bound, upper_bound)
        """
        if not self.observed_deltas:
            return (0.0, 0.0, 0.0)

        sorted_deltas = sorted(self.observed_deltas)
        n = len(sorted_deltas)

        # Use percentiles for confidence bounds
        lower_idx = int((1 - confidence_interval) / 2 * n)
        upper_idx = int((1 + confidence_interval) / 2 * n) - 1

        return (
            self.mean,
            sorted_deltas[max(0, lower_idx)],
            sorted_deltas[min(n - 1, upper_idx)],
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"observed_deltas": list(self.observed_deltas)}

    @classmethod
    def from_dict(cls, data: dict) -> TemporalDelta:
        """Deserialize from dictionary."""
        return cls(observed_deltas=tuple(data.get("observed_deltas", [])))


@dataclass
class CausalLink:
    """A learned causal relationship between an event and an outcome.

    Events can be:
    - Tool executions (tool_name + params hash)
    - User inputs (intent classification)
    - Temporal events (SCN-triggered)
    - Environmental changes (state transitions)

    Outcomes can be:
    - Tool results (success/failure + output classification)
    - User responses (satisfaction signals)
    - State changes (mode transitions)
    - Follow-up events (subsequent actions)

    Attributes:
        id: Unique link identifier
        event_type: Category of event
        event_signature: Hashable identifier for the event
        event_context: Context when event occurred
        outcome_type: Category of outcome
        outcome_signature: Hashable identifier for the outcome
        outcome_valence: Quality of outcome
        temporal_delta: Learned delay distribution
        predicted_value: Rescorla-Wagner prediction (0.0-1.0)
        observation_count: How many times observed
        confidence: Confidence in this link (0.0-1.0)
        last_observed: Timestamp of last observation
        memory_ids: Hippocampus memory IDs where observed
    """

    id: str

    # Event identification (what triggered this)
    event_type: str  # "tool", "user_input", "temporal", "state_change"
    event_signature: str  # Hashable identifier for the event type
    event_context: dict[str, Any]  # Context when event occurred

    # Outcome identification (what resulted)
    outcome_type: str  # "tool_result", "user_response", "state_change", "followup"
    outcome_signature: str  # Hashable identifier for the outcome type
    outcome_valence: Valence  # Was it positive/negative/neutral?

    # Temporal learning (flexible delays)
    temporal_delta: TemporalDelta  # Learned delay distribution

    # Rescorla-Wagner prediction
    predicted_value: float = 0.5  # V - expected outcome (0=negative, 1=positive)
    prediction_history: list[float] = field(default_factory=list)  # Track V over time

    # Confidence tracking
    observation_count: int = 0  # How many times observed
    confidence: float = 0.5  # 0.0-1.0 (increases with observations)
    last_observed: float = field(default_factory=time.time)

    # Hippocampus integration
    memory_ids: list[str] = field(default_factory=list)  # Episodes where observed

    # Context sensitivity (same event, different context → different outcome)
    context_factors: dict[str, float] = field(default_factory=dict)

    # Most recent Rescorla-Wagner prediction error magnitude (set by update_prediction_rw)
    last_rpe: float | None = None

    def record_observation(
        self,
        delta_seconds: float,
        valence: Valence,
        memory_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a new observation of this causal relationship.

        Args:
            delta_seconds: Time between event and outcome
            valence: Quality of the outcome
            memory_id: Optional hippocampus memory ID
            context: Optional context for context-sensitive learning
        """
        # Update temporal delta
        self.temporal_delta = self.temporal_delta.add_observation(delta_seconds)

        # Update observation count and confidence
        self.observation_count += 1
        # Confidence grows logarithmically (diminishing returns)
        self.confidence = min(0.99, 0.5 + 0.1 * (self.observation_count**0.5))

        # Update valence (exponential moving average)
        valence_value = {
            "positive": 1.0,
            "negative": -1.0,
            "neutral": 0.0,
            "unknown": 0.0,
        }
        current_v = valence_value.get(self.outcome_valence.value, 0.0)
        new_v = valence_value.get(valence.value, 0.0)
        blended_v = 0.8 * current_v + 0.2 * new_v

        if blended_v > 0.3:
            self.outcome_valence = Valence.POSITIVE
        elif blended_v < -0.3:
            self.outcome_valence = Valence.NEGATIVE
        else:
            self.outcome_valence = Valence.NEUTRAL

        # Track memory reference
        if memory_id and memory_id not in self.memory_ids:
            self.memory_ids.append(memory_id)
            # Keep last 50 memory references
            if len(self.memory_ids) > 50:
                self.memory_ids = self.memory_ids[-50:]

        self.last_observed = time.time()

    def update_prediction_rw(
        self,
        actual_valence: Valence,
        learning_rate: float = 0.2,
        novelty: float = 0.5,
    ) -> float:
        """Update prediction using Rescorla-Wagner.

        Args:
            actual_valence: The actual outcome valence
            learning_rate: Base learning rate (α)
            novelty: How novel is this situation (0=familiar, 1=novel)

        Returns:
            Prediction error (for logging/analysis)
        """
        # Convert valence to reward value
        actual_reward = {
            Valence.POSITIVE: 1.0,
            Valence.NEGATIVE: 0.0,
            Valence.NEUTRAL: 0.5,
            Valence.UNKNOWN: 0.5,
        }.get(actual_valence, 0.5)

        # Adaptive learning rate based on novelty
        min_lr = 0.05
        max_lr = 0.5
        alpha = min_lr + (max_lr - min_lr) * novelty * learning_rate * 2

        # Rescorla-Wagner: ΔV = α × (R - V)
        error = actual_reward - self.predicted_value
        self.last_rpe = abs(error)  # Store magnitude for salience signalling
        delta_v = alpha * error
        new_prediction = self.predicted_value + delta_v

        # Clamp to valid range
        new_prediction = max(0.0, min(1.0, new_prediction))

        # Store history for analysis
        self.prediction_history.append(self.predicted_value)
        if len(self.prediction_history) > 50:
            self.prediction_history = self.prediction_history[-50:]

        self.predicted_value = new_prediction
        return error

    def decay(self, factor: float = 0.99) -> None:
        """Apply temporal decay to confidence (unused links fade)."""
        self.confidence *= factor

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "event_signature": self.event_signature,
            "event_context": self.event_context,
            "outcome_type": self.outcome_type,
            "outcome_signature": self.outcome_signature,
            "outcome_valence": self.outcome_valence.value,
            "temporal_delta": self.temporal_delta.to_dict(),
            "predicted_value": self.predicted_value,
            "prediction_history": self.prediction_history,
            "observation_count": self.observation_count,
            "confidence": self.confidence,
            "last_observed": self.last_observed,
            "memory_ids": self.memory_ids,
            "context_factors": self.context_factors,
            "last_rpe": self.last_rpe,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CausalLink:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            event_type=data["event_type"],
            event_signature=data["event_signature"],
            event_context=data.get("event_context", {}),
            outcome_type=data["outcome_type"],
            outcome_signature=data["outcome_signature"],
            outcome_valence=Valence(data["outcome_valence"]),
            temporal_delta=TemporalDelta.from_dict(data["temporal_delta"]),
            predicted_value=data.get("predicted_value", 0.5),
            prediction_history=data.get("prediction_history", []),
            observation_count=data.get("observation_count", 0),
            confidence=data.get("confidence", 0.5),
            last_observed=data.get("last_observed", time.time()),
            memory_ids=data.get("memory_ids", []),
            context_factors=data.get("context_factors", {}),
            last_rpe=data.get("last_rpe"),
        )


@dataclass
class OutcomePrediction:
    """A prediction about what will happen if an event occurs."""

    event_signature: str
    predicted_outcome: str  # Expected outcome signature
    predicted_valence: Valence  # Expected quality
    predicted_value: float  # Rescorla-Wagner value (0-1)
    predicted_delay: float  # Expected seconds until outcome
    delay_bounds: tuple[float, float]  # (lower, upper) confidence interval
    confidence: float  # How confident in this prediction
    contributing_links: list[str]  # CausalLink IDs that informed this
    context_match: float  # How well current context matches learned context

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "event_signature": self.event_signature,
            "predicted_outcome": self.predicted_outcome,
            "predicted_valence": self.predicted_valence.value,
            "predicted_value": self.predicted_value,
            "predicted_delay": self.predicted_delay,
            "delay_bounds": list(self.delay_bounds),
            "confidence": self.confidence,
            "contributing_links": self.contributing_links,
            "context_match": self.context_match,
        }


__all__ = [
    "Valence",
    "TemporalDelta",
    "CausalLink",
    "OutcomePrediction",
]
