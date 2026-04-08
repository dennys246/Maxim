"""Thalamic Gate - Percept filtering for the Default Network.

The ThalamicGate determines which percepts require LLM deliberation
and which can be handled reactively by DN behaviors. This dramatically
reduces the load on the deliberative layer.

Inspired by the biological thalamus which gates sensory information
before it reaches the cortex.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from maxim.default_network.messages import EscalationResult, FilteredPercept

if TYPE_CHECKING:
    from maxim.agents.bus import Percept

logger = logging.getLogger(__name__)

# Default persistence path
DEFAULT_THRESHOLD_PERSIST_PATH = ""  # resolved via __post_init__

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


@dataclass
class AdaptiveThresholdConfig:
    """Configuration for adaptive threshold adjustment."""

    base_novelty_threshold: float = 0.7
    base_salience_threshold: float = 0.6
    min_threshold: float = 0.3
    max_threshold: float = 0.95
    adaptation_rate: float = 0.1
    escalation_rate_target: float = 0.05  # Target 5% escalation rate
    window_seconds: float = 60.0
    adaptation_interval: float = 5.0  # How often to adapt
    # Persistence
    persist_path: str = ""  # resolved via maxim.utils.paths at runtime
    auto_save_interval: float = 60.0  # Save every 60 seconds

    def __post_init__(self) -> None:
        if not self.persist_path:
            from maxim.utils.paths import resolve_user_state
            self.persist_path = str(resolve_user_state("util/adaptive_thresholds.json"))


class AdaptiveThresholdController:
    """Dynamically adjusts escalation thresholds based on feedback.

    Tracks escalation history and outcomes to tune thresholds:
    - Too many escalations -> raise threshold
    - Too few escalations -> lower threshold
    - Escalations leading to actions -> good, maintain
    - Escalations being ignored -> bad, raise threshold
    """

    def __init__(self, config: AdaptiveThresholdConfig | None = None) -> None:
        self.config = config or AdaptiveThresholdConfig()
        self._novelty_threshold = self.config.base_novelty_threshold
        self._salience_threshold = self.config.base_salience_threshold

        # History tracking
        self._escalation_history: deque[tuple[float, bool]] = deque(maxlen=1000)
        self._outcome_history: deque[tuple[float, str]] = deque(maxlen=100)
        self._last_adaptation = time.time()
        self._last_save_time = time.time()
        self._lock = threading.Lock()

        # Auto-load on init if path exists
        if self.config.persist_path and os.path.exists(self.config.persist_path):
            self.load(self.config.persist_path)

    @property
    def novelty_threshold(self) -> float:
        """Current novelty threshold."""
        return self._novelty_threshold

    @property
    def salience_threshold(self) -> float:
        """Current salience threshold."""
        return self._salience_threshold

    @property
    def combined_threshold(self) -> float:
        """Combined novelty * salience threshold for escalation."""
        return self._novelty_threshold * self._salience_threshold

    def record_escalation(self, escalated: bool) -> None:
        """Record whether a percept was escalated."""
        with self._lock:
            self._escalation_history.append((time.time(), escalated))

    def record_outcome(self, outcome: str) -> None:
        """Record outcome of an escalation.

        Args:
            outcome: One of 'action_taken', 'ignored', 'goal_created'.
        """
        with self._lock:
            self._outcome_history.append((time.time(), outcome))

    def maybe_adapt(
        self,
        llm_queue_depth: int = 0,
        fear_risk: float = 0.0,
    ) -> bool:
        """Adjust thresholds if enough time has passed.

        Args:
            llm_queue_depth: Number of items in LLM queue (0 = idle).
            fear_risk: Current fear/risk level from FearAgent (0-1).

        Returns:
            True if adaptation was performed.
        """
        now = time.time()
        if now - self._last_adaptation < self.config.adaptation_interval:
            return False

        with self._lock:
            self._last_adaptation = now
            self._adapt(llm_queue_depth, fear_risk)

        # Auto-save periodically
        self._maybe_auto_save()
        return True

    def _adapt(self, llm_queue_depth: int, fear_risk: float) -> None:
        """Internal adaptation logic (called with lock held)."""
        now = time.time()
        window_start = now - self.config.window_seconds

        # Calculate recent escalation rate
        recent = [(t, e) for t, e in self._escalation_history if t > window_start]
        if len(recent) < 10:
            return  # Not enough data

        escalation_count = sum(1 for _, e in recent if e)
        escalation_rate = escalation_count / len(recent)

        # Calculate outcome quality (what % of escalations led to actions)
        recent_outcomes = [o for t, o in self._outcome_history if t > window_start]
        if recent_outcomes:
            useful_outcomes = sum(1 for o in recent_outcomes if o in ("action_taken", "goal_created"))
            outcome_quality = useful_outcomes / len(recent_outcomes)
        else:
            outcome_quality = 0.5  # Neutral if no data

        # Determine adjustment direction
        target_rate = self.config.escalation_rate_target
        rate_error = escalation_rate - target_rate

        # Adjust for context
        adjustment = 0.0

        # Rate too high -> raise threshold
        if rate_error > 0.02:
            adjustment += self.config.adaptation_rate * rate_error * 2

        # Rate too low -> lower threshold
        elif rate_error < -0.02:
            adjustment += self.config.adaptation_rate * rate_error * 2

        # Outcomes being ignored -> raise threshold
        if outcome_quality < 0.3 and recent_outcomes:
            adjustment += self.config.adaptation_rate * 0.5

        # LLM queue busy -> raise threshold
        if llm_queue_depth > 3:
            adjustment += self.config.adaptation_rate * 0.3

        # High fear/risk -> lower threshold (be more cautious)
        if fear_risk > 0.5:
            adjustment -= self.config.adaptation_rate * fear_risk * 0.5

        # Apply adjustment to both thresholds
        self._novelty_threshold = max(
            self.config.min_threshold, min(self.config.max_threshold, self._novelty_threshold + adjustment)
        )
        self._salience_threshold = max(
            self.config.min_threshold, min(self.config.max_threshold, self._salience_threshold + adjustment)
        )

        logger.debug(
            "Adapted thresholds: novelty=%.3f, salience=%.3f (rate=%.3f, quality=%.3f)",
            self._novelty_threshold,
            self._salience_threshold,
            escalation_rate,
            outcome_quality,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> bool:
        """Save learned thresholds to disk.

        Args:
            path: Path to save to. Uses config.persist_path if None.

        Returns:
            True if save succeeded.
        """
        save_path = path or self.config.persist_path
        if not save_path:
            return False

        try:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                # Serialize history (keep recent 200 entries each)
                escalation_data = list(self._escalation_history)[-200:]
                outcome_data = list(self._outcome_history)[-100:]

                data = {
                    "version": 1,
                    "saved_at": time.time(),
                    "thresholds": {
                        "novelty": self._novelty_threshold,
                        "salience": self._salience_threshold,
                    },
                    "escalation_history": escalation_data,
                    "outcome_history": outcome_data,
                    "config": {
                        "base_novelty_threshold": self.config.base_novelty_threshold,
                        "base_salience_threshold": self.config.base_salience_threshold,
                    },
                }

            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

            self._last_save_time = time.time()
            logger.debug("AdaptiveThresholdController saved to %s", save_path)
            return True

        except Exception as e:
            logger.warning("Failed to save AdaptiveThresholdController: %s", e)
            return False

    def load(self, path: str | None = None) -> bool:
        """Load learned thresholds from disk.

        Args:
            path: Path to load from. Uses config.persist_path if None.

        Returns:
            True if load succeeded.
        """
        load_path = path or self.config.persist_path
        if not load_path or not os.path.exists(load_path):
            return False

        try:
            with open(load_path) as f:
                data = json.load(f)

            with self._lock:
                # Load thresholds
                thresholds = data.get("thresholds", {})
                self._novelty_threshold = thresholds.get("novelty", self.config.base_novelty_threshold)
                self._salience_threshold = thresholds.get("salience", self.config.base_salience_threshold)

                # Load history
                for ts, escalated in data.get("escalation_history", []):
                    self._escalation_history.append((ts, escalated))
                for ts, outcome in data.get("outcome_history", []):
                    self._outcome_history.append((ts, outcome))

            logger.info(
                "Loaded adaptive thresholds (novelty=%.3f, salience=%.3f) from %s",
                self._novelty_threshold,
                self._salience_threshold,
                load_path,
            )
            return True

        except Exception as e:
            logger.warning("Failed to load AdaptiveThresholdController: %s", e)
            return False

    def _maybe_auto_save(self) -> None:
        """Auto-save if enough time has passed."""
        if not self.config.persist_path:
            return
        now = time.time()
        if now - self._last_save_time >= self.config.auto_save_interval:
            self.save()

    def reset(self) -> None:
        """Reset thresholds to base values."""
        with self._lock:
            self._novelty_threshold = self.config.base_novelty_threshold
            self._salience_threshold = self.config.base_salience_threshold
            self._escalation_history.clear()
            self._outcome_history.clear()


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
            self._adaptive.maybe_adapt(llm_queue_depth, fear_risk)

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
