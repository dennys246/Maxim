"""Behavior arbitration for the Default Network.

The PriorityArbiter selects the winning behavior from competing proposals
using priority scores with hysteresis to prevent rapid switching.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from maxim.default_network.messages import ActionProposal

logger = logging.getLogger(__name__)


@dataclass
class ArbiterConfig:
    """Configuration for the PriorityArbiter.

    Attributes:
        hysteresis_bonus: Priority bonus for the currently active behavior
            to prevent rapid switching (default 0.1).
        min_switch_interval: Minimum seconds between behavior switches
            (default 0.3s).
        score_threshold: Minimum effective score to act (default 0.1).
    """

    hysteresis_bonus: float = 0.1
    min_switch_interval: float = 0.3
    score_threshold: float = 0.1


class PriorityArbiter:
    """Winner-take-all arbitration with hysteresis.

    The arbiter selects the highest-scoring proposal while providing
    stability through:
    1. Hysteresis bonus: Current behavior gets a score bonus
    2. Switch cooldown: Minimum time before changing behaviors
    3. Score threshold: Minimum score required to act

    Example:
        arbiter = PriorityArbiter()
        proposals = [behavior.evaluate(detections, state) for behavior in behaviors]
        proposals = [p for p in proposals if p is not None]
        winner = arbiter.select(proposals)
        if winner:
            execute_action(winner)
    """

    def __init__(self, config: ArbiterConfig | None = None) -> None:
        """Initialize the arbiter.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or ArbiterConfig()
        self._current_behavior: str | None = None
        self._last_switch: float = 0.0
        self._consecutive_wins: int = 0

    def select(self, proposals: list[ActionProposal]) -> ActionProposal | None:
        """Select the winning action proposal.

        Args:
            proposals: List of ActionProposals from behaviors.

        Returns:
            The winning proposal, or None if no valid proposals.
        """
        if not proposals:
            return None

        now = time.time()

        # Score all proposals, applying hysteresis to current behavior
        scored: list[tuple[float, ActionProposal]] = []
        for p in proposals:
            score = p.effective_score()

            # Apply hysteresis bonus to current behavior
            if p.behavior_name == self._current_behavior:
                score += self.config.hysteresis_bonus

            scored.append((score, p))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Check if best score meets threshold
        best_score, best_proposal = scored[0]
        if best_score < self.config.score_threshold:
            return None

        # Check if this would be a behavior switch
        if best_proposal.behavior_name != self._current_behavior:
            # Enforce minimum switch interval
            if now - self._last_switch < self.config.min_switch_interval:
                # Try to find current behavior in proposals instead
                for score, p in scored:
                    if p.behavior_name == self._current_behavior:
                        if score >= self.config.score_threshold:
                            return p
                        break
                # Current behavior not available or below threshold
                # Still don't switch until cooldown expires
                return None

            # Perform the switch
            self._current_behavior = best_proposal.behavior_name
            self._last_switch = now
            self._consecutive_wins = 1
            logger.debug(
                "Arbiter switched to behavior '%s' (score=%.3f)",
                best_proposal.behavior_name,
                best_score,
            )
        else:
            self._consecutive_wins += 1

        return best_proposal

    def force_switch(self, behavior_name: str) -> None:
        """Force a behavior switch (bypasses cooldown).

        Used by deliberative layer to immediately change behavior.

        Args:
            behavior_name: Name of behavior to switch to.
        """
        self._current_behavior = behavior_name
        self._last_switch = time.time()
        self._consecutive_wins = 0
        logger.debug("Arbiter forced switch to '%s'", behavior_name)

    def reset(self) -> None:
        """Reset arbiter state."""
        self._current_behavior = None
        self._last_switch = 0.0
        self._consecutive_wins = 0

    @property
    def current_behavior(self) -> str | None:
        """Name of the currently active behavior."""
        return self._current_behavior

    @property
    def behavior_stability(self) -> float:
        """Measure of how stable the current behavior is (0-1).

        Higher values indicate the behavior has been winning consistently.
        """
        if self._consecutive_wins == 0:
            return 0.0
        # Asymptotic approach to 1.0
        return 1.0 - (1.0 / (1.0 + self._consecutive_wins * 0.2))


__all__ = [
    "ArbiterConfig",
    "PriorityArbiter",
]
