"""Base class for reactive behaviors in the Default Network.

Behaviors are lightweight, stateless-ish modules that evaluate current
detections and propose actions. Each behavior should execute in <10ms.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from maxim.default_network.messages import ActionProposal

if TYPE_CHECKING:
    from maxim.attention import SalienceMap
    from maxim.proprioception import FocusLearner


@dataclass
class BehaviorState:
    """Shared state passed to behaviors during evaluation.

    Contains context about the current network state, active goals,
    and any deliberative layer signals.

    Attributes:
        inhibited_behaviors: Set of behavior names currently inhibited.
        priority_modifiers: Dict of behavior name -> priority multiplier.
        current_goals: List of active goal keywords (e.g., "find red cup").
        interests: Set of class IDs that are currently of interest.
        fear_level: Current fear/caution level from FearAgent (0.0-1.0).
        frame_timestamp: Timestamp of the current frame being processed.
        salience_map: Unified salience map for spatial attention (if enabled).
        focus_learner: FocusLearner for adaptive movement correction (if enabled).
    """
    inhibited_behaviors: frozenset[str] = field(default_factory=frozenset)
    priority_modifiers: dict[str, float] = field(default_factory=dict)
    current_goals: list[str] = field(default_factory=list)
    interests: frozenset[int] = field(default_factory=frozenset)
    fear_level: float = 0.0
    frame_timestamp: float = 0.0
    salience_map: "SalienceMap | None" = None
    focus_learner: "FocusLearner | None" = None

    def get_tracking_bonus(self, position: tuple[float, float]) -> float:
        """Get spatial tracking bonus for a detection position.

        This provides centralized tracking hysteresis - a bonus for detections
        near where we were recently looking. Helps maintain focus when YOLO
        track_id fragments during head movement.

        Args:
            position: (x, y) pixel coordinates of the detection center.

        Returns:
            Bonus value to add to detection score (0.0 if no salience map or not near).
        """
        if self.salience_map is None:
            return 0.0
        return self.salience_map.get_tracking_bonus(position)

    def record_tracking_target(self, position: tuple[float, float]) -> None:
        """Record that we're actively tracking a target at this position.

        Call this when a behavior commits to tracking a detection.

        Args:
            position: (x, y) pixel coordinates of the tracked target.
        """
        if self.salience_map is not None:
            self.salience_map.record_tracking_target(position)


class Behavior(ABC):
    """Base class for reactive behaviors.

    Subclasses must implement `evaluate()` to analyze detections and
    optionally return an ActionProposal.

    Class Attributes:
        name: Unique identifier for this behavior.
        base_priority: Default priority (0.0-1.0) when behavior activates.
        cooldown_seconds: Minimum time between activations.
        enabled: Whether this behavior is active.

    Example:
        class OrientingResponse(Behavior):
            name = "orienting"
            base_priority = 0.8
            cooldown_seconds = 0.5

            def evaluate(self, detections, state):
                # Find most novel object and propose looking at it
                ...
    """

    name: str = "unnamed"
    base_priority: float = 0.5
    cooldown_seconds: float = 0.0
    enabled: bool = True

    def __init__(self) -> None:
        """Initialize behavior with timing state."""
        self._last_activation: float = 0.0
        self._activation_count: int = 0
        self._internal_state: dict[str, Any] = {}

    @abstractmethod
    def evaluate(
        self,
        detections: list[dict],
        state: BehaviorState,
    ) -> ActionProposal | None:
        """Evaluate current detections and return action proposal if triggered.

        This method should execute quickly (<10ms). Heavy computation
        should be avoided or amortized across frames.

        Args:
            detections: List of YOLO detection dicts with keys like
                'track_id', 'class_id', 'conf', 'bbox_xyxy', etc.
            state: Current network state with goals, inhibitions, etc.

        Returns:
            ActionProposal if the behavior wants to act, None otherwise.
        """
        pass

    def can_activate(self) -> bool:
        """Check if cooldown has elapsed since last activation."""
        if self.cooldown_seconds <= 0:
            return True
        return time.time() - self._last_activation >= self.cooldown_seconds

    def record_activation(self) -> None:
        """Record that this behavior just activated.

        Call this when returning an ActionProposal from evaluate().
        """
        self._last_activation = time.time()
        self._activation_count += 1

    def reset(self) -> None:
        """Reset behavior state (e.g., when DN restarts)."""
        self._last_activation = 0.0
        self._internal_state.clear()

    @property
    def time_since_activation(self) -> float:
        """Seconds since last activation (inf if never activated)."""
        if self._last_activation == 0.0:
            return float('inf')
        return time.time() - self._last_activation

    def _create_proposal(
        self,
        action_type: str,
        target: tuple[float, float] | None = None,
        priority_scale: float = 1.0,
        confidence: float = 1.0,
        **metadata: Any,
    ) -> ActionProposal:
        """Helper to create an ActionProposal with this behavior's defaults.

        Args:
            action_type: Type of action ("look_at", "track", "scan", etc.).
            target: Optional (u, v) pixel coordinates.
            priority_scale: Multiplier for base_priority (default 1.0).
            confidence: Confidence score (default 1.0).
            **metadata: Additional metadata to include.

        Returns:
            ActionProposal configured for this behavior.
        """
        self.record_activation()
        return ActionProposal(
            behavior_name=self.name,
            action_type=action_type,
            target=target,
            priority=self.base_priority * priority_scale,
            confidence=confidence,
            metadata=dict(metadata),
        )


__all__ = [
    "Behavior",
    "BehaviorState",
]
