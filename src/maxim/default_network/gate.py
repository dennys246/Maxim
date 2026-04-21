"""Thalamic Gate - Percept filtering for the Default Network.

The ThalamicGate determines which percepts require LLM deliberation
and which can be handled reactively by DN behaviors. This dramatically
reduces the load on the deliberative layer.

Inspired by the biological thalamus which gates sensory information
before it reaches the cortex.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from maxim.default_network.messages import EscalationResult, FilteredPercept

if TYPE_CHECKING:
    from maxim.agents.bus import Percept

logger = logging.getLogger(__name__)

# COCO class names for goal matching (subset for common objects)
COCO_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    15: "cat",
    16: "dog",
    39: "bottle",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    73: "book",
}


from maxim.runtime.gating import (  # noqa: E402  — extracted in G0
    AdaptiveThresholdConfig as _BaseAdaptiveThresholdConfig,
    AdaptiveThresholdController,
)


@dataclass
class AdaptiveThresholdConfig(_BaseAdaptiveThresholdConfig):
    """DN-specific adaptive threshold config with auto-resolved persist path.

    Extends the shared ``AdaptiveThresholdConfig`` from ``runtime/gating.py``
    with a ``__post_init__`` that auto-resolves the persistence path via
    ``maxim.utils.paths``.  The shared base class leaves ``persist_path``
    empty by default (ephemeral).
    """

    def __post_init__(self) -> None:
        if not self.persist_path:
            from maxim.utils.paths import resolve_user_state

            self.persist_path = str(resolve_user_state("util/adaptive_thresholds.json"))


@dataclass
class GateConfig:
    """Configuration for the ThalamicGate."""

    novelty_threshold: float = 0.7
    salience_threshold: float = 0.6
    anomaly_threshold: float = 0.7
    safety_velocity_threshold: float = 200.0  # pixels/frame for rapid approach
    max_recent_percepts: int = 100
    adaptive: bool = True


class ThalamicGate:
    """Filters percepts - decides what reaches the deliberative layer.

    Inspired by the thalamus which gates sensory information to cortex.
    Most percepts are handled reactively by DN behaviors. Only "significant"
    percepts are escalated for LLM deliberation.

    Escalation Criteria:
    - High novelty + salience combined score
    - Goal-relevant detection
    - Speech/audio input detected
    - Anomaly (unexpected disappearance, etc.)
    - Safety concern (rapid approach)
    - Attention lock (deliberative layer requested focus)
    """

    def __init__(
        self,
        config: GateConfig | None = None,
        anomaly_detector: Callable[["Percept", list["Percept"]], float] | None = None,
        goal_matcher: Callable[["Percept", str | None], bool] | None = None,
        adaptive_config: AdaptiveThresholdConfig | None = None,
    ) -> None:
        """Initialize the gate.

        Args:
            config: Gate configuration.
            anomaly_detector: Optional custom anomaly detection function.
            goal_matcher: Optional custom goal relevance matcher.
            adaptive_config: Configuration for adaptive thresholding.
        """
        self.config = config or GateConfig()
        self._anomaly_detector = anomaly_detector
        self._goal_matcher = goal_matcher

        # Adaptive thresholding (if enabled)
        self._adaptive = None
        if self.config.adaptive:
            self._adaptive = AdaptiveThresholdController(adaptive_config)

        # Recent percept history for anomaly detection
        self._recent_percepts: deque["Percept"] = deque(maxlen=self.config.max_recent_percepts)

        # Current active goal (set by deliberative layer)
        self._active_goal: str | None = None
        self._goal_keywords: list[str] = []

        # Attention locks - force escalation for specific tracks
        self._attention_locks: set[int] = set()

        # Thread safety
        self._lock = threading.Lock()

        # Interests - class IDs that get priority
        self._interests: frozenset[int] = frozenset()

    def evaluate(self, percept: "Percept") -> EscalationResult:
        """Evaluate whether this percept should be escalated to deliberation.

        Args:
            percept: The percept to evaluate.

        Returns:
            EscalationResult indicating whether to escalate and why.
        """
        # Get current thresholds (may be adaptive)
        if self._adaptive:
            novelty_thresh = self._adaptive.novelty_threshold
            salience_thresh = self._adaptive.salience_threshold
            combined_thresh = self._adaptive.combined_threshold
        else:
            novelty_thresh = self.config.novelty_threshold
            salience_thresh = self.config.salience_threshold
            combined_thresh = novelty_thresh * salience_thresh

        # Get percept attributes safely
        novelty = getattr(percept, "novelty", 0.5)
        salience = getattr(percept, "salience", 0.5)
        primary_track_id = getattr(percept, "primary_track_id", None)
        primary_class_id = getattr(percept, "primary_class_id", None)
        has_speech = getattr(percept, "has_speech", False)

        # Calculate fear factor from percept if available
        fear_factor = getattr(percept, "fear_factor", 0.0)

        # 1. Check attention locks (deliberative layer requested focus)
        with self._lock:
            if primary_track_id is not None and primary_track_id in self._attention_locks:
                result = EscalationResult(
                    escalated=True,
                    reason="attention_lock",
                    novelty_score=novelty,
                    salience_score=salience,
                    fear_factor=fear_factor,
                    threshold_used=combined_thresh,
                )
                self._record_escalation(True)
                return result

        # 2. Check goal relevance
        if self._active_goal and self._is_goal_relevant(percept):
            result = EscalationResult(
                escalated=True,
                reason="goal_relevant",
                novelty_score=novelty,
                salience_score=salience,
                fear_factor=fear_factor,
                threshold_used=combined_thresh,
            )
            self._record_escalation(True)
            return result

        # 3. Check for speech/audio (always escalate)
        if has_speech:
            result = EscalationResult(
                escalated=True,
                reason="speech_detected",
                novelty_score=novelty,
                salience_score=salience,
                fear_factor=fear_factor,
                threshold_used=combined_thresh,
            )
            self._record_escalation(True)
            return result

        # 4. Check interest classes (lower threshold for interests)
        if primary_class_id is not None and primary_class_id in self._interests:
            interest_thresh = combined_thresh * 0.5  # 50% lower threshold
            combined_score = novelty * salience
            if combined_score > interest_thresh:
                result = EscalationResult(
                    escalated=True,
                    reason="interest_detected",
                    novelty_score=novelty,
                    salience_score=salience,
                    fear_factor=fear_factor,
                    threshold_used=interest_thresh,
                )
                self._record_escalation(True)
                return result

        # 5. Check novelty + salience combined score
        combined_score = novelty * salience
        if combined_score > combined_thresh:
            result = EscalationResult(
                escalated=True,
                reason="high_novelty_salience",
                novelty_score=novelty,
                salience_score=salience,
                fear_factor=fear_factor,
                threshold_used=combined_thresh,
            )
            self._record_escalation(True)
            return result

        # 6. Check for anomalies (unexpected disappearance, etc.)
        anomaly_score = self._detect_anomaly(percept)
        if anomaly_score > self.config.anomaly_threshold:
            result = EscalationResult(
                escalated=True,
                reason="anomaly_detected",
                novelty_score=novelty,
                salience_score=salience,
                fear_factor=fear_factor,
                threshold_used=self.config.anomaly_threshold,
            )
            self._record_escalation(True)
            return result

        # 7. Check for safety concerns (rapid approach, etc.)
        if self._is_safety_concern(percept):
            result = EscalationResult(
                escalated=True,
                reason="safety_concern",
                novelty_score=novelty,
                salience_score=salience,
                fear_factor=max(fear_factor, 0.8),
                threshold_used=combined_thresh,
            )
            self._record_escalation(True)
            return result

        # Default: Don't escalate - DN handles reactively
        with self._lock:
            self._recent_percepts.append(percept)

        self._record_escalation(False)
        return EscalationResult(
            escalated=False,
            reason="routine",
            novelty_score=novelty,
            salience_score=salience,
            fear_factor=fear_factor,
            threshold_used=combined_thresh,
        )

    def create_filtered_percept(
        self,
        percept: "Percept",
        result: EscalationResult,
    ) -> FilteredPercept:
        """Create a FilteredPercept from an escalation result.

        Args:
            percept: The original percept.
            result: The escalation result.

        Returns:
            FilteredPercept ready for publishing to AgentBus.
        """
        urgency = self._calculate_urgency(result)
        return FilteredPercept(
            original=percept,
            escalation_reason=result.reason,
            dn_context={
                "novelty": result.novelty_score,
                "salience": result.salience_score,
                "fear_factor": result.fear_factor,
                "threshold_used": result.threshold_used,
            },
            urgency=urgency,
        )

    def _calculate_urgency(self, result: EscalationResult) -> float:
        """Calculate urgency score for an escalation."""
        # Base urgency by reason
        urgency_map = {
            "safety_concern": 1.0,
            "speech_detected": 0.95,
            "attention_lock": 0.9,
            "goal_relevant": 0.8,
            "high_novelty_salience": 0.7,
            "interest_detected": 0.6,
            "anomaly_detected": 0.75,
        }
        base = urgency_map.get(result.reason, 0.5)

        # Boost by fear factor
        if result.fear_factor > 0:
            base = min(1.0, base + result.fear_factor * 0.2)

        return base

    def _record_escalation(self, escalated: bool) -> None:
        """Record escalation for adaptive thresholding."""
        if self._adaptive:
            self._adaptive.record_escalation(escalated)

    def record_outcome(self, outcome: str) -> None:
        """Record outcome of an escalation for threshold adaptation.

        Args:
            outcome: One of 'action_taken', 'ignored', 'goal_created'.
        """
        if self._adaptive:
            self._adaptive.record_outcome(outcome)

    def adapt_thresholds(
        self,
        llm_queue_depth: int = 0,
        fear_risk: float = 0.0,
    ) -> None:
        """Trigger threshold adaptation if conditions are met."""
        if self._adaptive:
            self._adaptive.maybe_adapt(processing_load=llm_queue_depth, urgency=fear_risk)

    def set_active_goal(self, goal: str | None) -> None:
        """Set current goal for relevance matching.

        Args:
            goal: Goal description (e.g., "find the red cup") or None.
        """
        with self._lock:
            self._active_goal = goal
            if goal:
                # Extract keywords for matching
                self._goal_keywords = [
                    w.lower()
                    for w in goal.split()
                    if len(w) > 2 and w.lower() not in ("the", "a", "an", "find", "look")
                ]
            else:
                self._goal_keywords = []

    def set_interests(self, class_ids: frozenset[int]) -> None:
        """Set interest class IDs that get lower escalation threshold.

        Args:
            class_ids: Set of COCO class IDs that are currently interesting.
        """
        with self._lock:
            self._interests = class_ids

    def add_attention_lock(self, track_id: int) -> None:
        """Force escalation for specific track (deliberative override).

        Args:
            track_id: Track ID to always escalate.
        """
        with self._lock:
            self._attention_locks.add(track_id)

    def remove_attention_lock(self, track_id: int) -> None:
        """Release attention lock.

        Args:
            track_id: Track ID to stop forcing escalation.
        """
        with self._lock:
            self._attention_locks.discard(track_id)

    def clear_attention_locks(self) -> None:
        """Remove all attention locks."""
        with self._lock:
            self._attention_locks.clear()

    def _is_goal_relevant(self, percept: "Percept") -> bool:
        """Check if percept is relevant to the active goal."""
        if self._goal_matcher:
            return self._goal_matcher(percept, self._active_goal)

        # Simple keyword matching fallback
        if not self._goal_keywords:
            return False

        primary_class_id = getattr(percept, "primary_class_id", None)
        if primary_class_id is None:
            return False

        class_name = COCO_CLASSES.get(primary_class_id, "").lower()
        return any(kw in class_name for kw in self._goal_keywords)

    def _detect_anomaly(self, percept: "Percept") -> float:
        """Detect anomalies in the percept stream.

        Args:
            percept: Current percept.

        Returns:
            Anomaly score 0.0-1.0.
        """
        if self._anomaly_detector:
            with self._lock:
                recent = list(self._recent_percepts)
            return self._anomaly_detector(percept, recent)

        # Simple built-in anomaly detection:
        # Detect sudden disappearance of tracked objects
        # This is a placeholder - real implementation would be more sophisticated
        return 0.0

    def _is_safety_concern(self, percept: "Percept") -> bool:
        """Check if percept indicates a safety concern.

        Currently checks for rapid approach (high velocity toward camera).
        """
        # Check for velocity information
        velocity = getattr(percept, "velocity", None)
        if velocity is None:
            return False

        # Check if velocity indicates rapid approach
        # Negative y-velocity in image space often means approaching
        vx, vy = velocity if isinstance(velocity, tuple) else (0, 0)
        speed = (vx**2 + vy**2) ** 0.5

        return speed > self.config.safety_velocity_threshold


__all__ = [
    "AdaptiveThresholdConfig",
    "AdaptiveThresholdController",
    "GateConfig",
    "ThalamicGate",
]
