"""Message types for the Default Network.

These dataclasses define the communication protocol between DN components
and the broader Maxim system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maxim.agents.bus import Percept


@dataclass(frozen=True, slots=True)
class ActionProposal:
    """A proposed action from a behavior module.

    Behaviors generate ActionProposals which are then arbitrated
    by the PriorityArbiter to select the winning action.

    Attributes:
        behavior_name: Name of the behavior that generated this proposal.
        action_type: Type of action ("look_at", "track", "scan", etc.).
        target: Optional (u, v) pixel coordinates for the action target.
        priority: Base priority score from 0.0-1.0.
        confidence: Confidence in this being the correct action (0.0-1.0).
        metadata: Behavior-specific data (track_id, novelty score, etc.).
    """

    behavior_name: str
    action_type: str
    target: tuple[float, float] | None
    priority: float
    confidence: float
    metadata: dict = field(default_factory=dict)

    def effective_score(self) -> float:
        """Calculate effective priority score (priority * confidence)."""
        return self.priority * self.confidence


@dataclass(frozen=True, slots=True)
class FilteredPercept:
    """A percept escalated by DN that requires deliberation.

    When the ThalamicGate determines a percept needs LLM involvement,
    it wraps it in a FilteredPercept with context about why it was escalated.

    Attributes:
        original: The original Percept that triggered escalation.
        escalation_reason: Why this percept was escalated
            ("high_novelty", "goal_relevant", "anomaly", "safety", etc.).
        dn_context: Context from DN about this percept (novelty scores,
            which behaviors evaluated it, etc.).
        urgency: Urgency score 0.0-1.0, affects ExecAgent priority queue.
    """

    original: "Percept"
    escalation_reason: str
    dn_context: dict
    urgency: float = 0.5

    def __post_init__(self) -> None:
        """Validate urgency is in valid range."""
        if not 0.0 <= self.urgency <= 1.0:
            object.__setattr__(self, "urgency", max(0.0, min(1.0, self.urgency)))


@dataclass(frozen=True, slots=True)
class DNActionProposal:
    """A reactive action taken by DN (for logging/learning).

    Published to AgentBus when DN executes an action, allowing other
    components to observe DN behavior for learning or analysis.

    Attributes:
        behavior: Name of the behavior that produced this action.
        action_type: Type of action executed.
        target: Target pixel coordinates, if applicable.
        timestamp: Unix timestamp when action was executed.
        confidence: Confidence score of the action.
        was_executed: Whether the action was actually executed
            (may be blocked by FearAgent or inhibition).
    """

    behavior: str
    action_type: str
    target: tuple[float, float] | None
    timestamp: float
    confidence: float = 1.0
    was_executed: bool = True


@dataclass(frozen=True, slots=True)
class EscalationResult:
    """Result of ThalamicGate escalation decision.

    Used internally by the gate to track escalation outcomes
    for adaptive threshold learning.

    Attributes:
        escalated: Whether the percept was escalated.
        reason: Reason for escalation (or why it was filtered).
        novelty_score: Novelty score at time of decision.
        salience_score: Salience score at time of decision.
        fear_factor: FearAgent influence on decision (0.0-1.0).
        threshold_used: The threshold that was applied.
    """

    escalated: bool
    reason: str
    novelty_score: float
    salience_score: float
    fear_factor: float = 0.0
    threshold_used: float = 0.7


__all__ = [
    "ActionProposal",
    "DNActionProposal",
    "EscalationResult",
    "FilteredPercept",
]
