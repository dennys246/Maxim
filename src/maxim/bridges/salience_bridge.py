"""SalienceMemoryBridge - Enriches salience with interaction history.

Connects: Hippocampus <-> EC <-> SalienceNetwork

Problem: Salience is purely novelty/recency. No memory of which objects
led to positive interactions.

Solution: Boost salience for objects with positive interaction history.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.memory.hippocampus import Hippocampus
    from maxim.salience.salience_network import SalienceNetwork
    from maxim.similarity.ec import EntorhinalCortex

logger = logging.getLogger(__name__)


@dataclass
class InteractionRecord:
    """Record of past interactions with an object class."""

    object_class: str
    success_count: int = 0
    failure_count: int = 0
    total_interactions: int = 0
    last_interaction: float = 0.0
    positive_goals: list[str] = field(default_factory=list)


@dataclass
class SalienceMemoryBridge:
    """Enriches salience with interaction history from Hippocampus.

    Features:
    - Interaction history: Track success/failure with object classes
    - Salience boosting: Increase salience for objects with positive history
    - Goal-aware salience: Boost objects relevant to current goal

    Example:
        bridge = SalienceMemoryBridge(
            hippocampus=hippocampus,
            ec=ec,
            salience_network=salience_network,
        )

        # Enrich salience scores with history
        enriched = bridge.enrich_salience(
            detections=[{"label": "mug", "salience": 0.5}],
            goal="find the mug",
        )
        # enriched[0]["salience"] might be 0.75 if mug has positive history
    """

    hippocampus: "Hippocampus"
    ec: "EntorhinalCortex"
    salience_network: "SalienceNetwork"

    # Interaction history: object_class -> InteractionRecord
    _interaction_history: dict[str, InteractionRecord] = field(
        default_factory=lambda: defaultdict(lambda: InteractionRecord(object_class=""))
    )

    # Configuration
    history_boost_factor: float = 0.3  # Max boost from history
    goal_match_boost: float = 0.2  # Boost for goal-relevant objects
    decay_days: float = 30.0  # Time for history influence to decay

    # Health tracking
    _healthy: bool = True
    _error_count: int = 0
    _max_errors: int = 5

    def __post_init__(self) -> None:
        """Initialize default factory fields."""
        if not hasattr(self, "_interaction_history") or self._interaction_history is None:
            self._interaction_history = defaultdict(lambda: InteractionRecord(object_class=""))

    # ─────────────────────────────────────────────────────────────────────────
    # Session Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def on_session_start(self) -> int:
        """Restore interaction history from Hippocampus.

        Returns:
            Number of object classes with history restored
        """
        try:
            # Query successful and failed memories
            successful = self.hippocampus.recall(limit=500, success=True)
            failed = self.hippocampus.recall(limit=200, success=False)
            # D56 (g): associated memories that carry no success attribute at
            # all are UNKNOWN, not failures. They are recorded so they dilute
            # confidence without asserting a direction.
            neutral: list[Any] = []

            # Expand with associative recall for richer context
            if successful:
                seed_ids = [m.id for m in successful[:20]]
                try:
                    associated = self.hippocampus.recall_associated(seed_ids, limit=50)
                    seen_ids = {m.id for m in successful} | {m.id for m in failed}
                    for mem, _score in associated:
                        if mem.id not in seen_ids:
                            seen_ids.add(mem.id)
                            # Classify associated memory by its success status
                            # D56 (g), REACHABLE half. The trailing `else
                            # False` booked "this memory carries no success
                            # attribute at all" as a FAILURE — an absence of
                            # evidence recorded as evidence of harm, and the
                            # one branch here that a real recall can hit. It is
                            # UNKNOWN, so it belongs in neither decisive
                            # bucket. `record_interaction`'s `None` tier and
                            # the decisive-denominator rate exist for exactly
                            # this case.
                            if hasattr(mem, "success"):
                                success_val = mem.success
                            elif hasattr(mem, "outcome"):
                                success_val = mem.outcome.success
                            else:
                                success_val = None
                            if success_val is None:
                                neutral.append(mem)
                            elif success_val:
                                successful.append(mem)
                            else:
                                failed.append(mem)
                except Exception:
                    pass  # Associative recall is best-effort

            # Build interaction records
            for memory in successful:
                if hasattr(memory, "perception"):
                    goal = ""
                    if hasattr(memory, "context") and memory.context.active_goal:
                        goal = memory.context.active_goal

                    for obj in memory.perception.detected_objects:
                        record = self._get_or_create_record(obj)
                        record.success_count += 1
                        record.total_interactions += 1
                        record.last_interaction = max(record.last_interaction, memory.timestamp)
                        if goal and goal not in record.positive_goals:
                            record.positive_goals.append(goal)
                            # Keep only last 10 goals
                            record.positive_goals = record.positive_goals[-10:]

            for memory in failed:
                if hasattr(memory, "perception"):
                    for obj in memory.perception.detected_objects:
                        record = self._get_or_create_record(obj)
                        record.failure_count += 1
                        record.total_interactions += 1
                        record.last_interaction = max(record.last_interaction, memory.timestamp)

            for memory in neutral:
                if hasattr(memory, "perception"):
                    for obj in memory.perception.detected_objects:
                        record = self._get_or_create_record(obj)
                        # total only — neither decisive bucket. This is what
                        # makes `decisive < total_interactions` reachable in
                        # production, and therefore what makes the
                        # decisive-denominator rate mean anything at all.
                        record.total_interactions += 1
                        record.last_interaction = max(record.last_interaction, memory.timestamp)

            logger.info(
                "Restored interaction history for %d object classes",
                len(self._interaction_history),
            )
            return len(self._interaction_history)

        except Exception as e:
            self._record_error(e)
            return 0

    # ─────────────────────────────────────────────────────────────────────────
    # Salience Enrichment
    # ─────────────────────────────────────────────────────────────────────────

    def enrich_salience(
        self,
        detections: list[dict[str, Any]],
        goal: str | None = None,
    ) -> list[dict[str, Any]]:
        """Enrich detection salience scores with interaction history.

        Args:
            detections: List of detection dicts with at least "label" and "salience"
            goal: Current goal for goal-aware boosting

        Returns:
            Same detections with enriched "salience" values and "history_boost" added
        """
        if not self._healthy:
            return detections

        try:
            now = time.time()

            for detection in detections:
                label = detection.get("label", "").lower()
                base_salience = detection.get("salience", 0.5)

                # Calculate history boost
                history_boost = self._calculate_history_boost(label, now)

                # Calculate goal match boost
                goal_boost = 0.0
                if goal and label in goal.lower():
                    goal_boost = self.goal_match_boost

                # Also check if this object led to success with similar goals
                record = self._interaction_history.get(label)
                if record and goal:
                    for past_goal in record.positive_goals:
                        if self._goals_similar(goal, past_goal):
                            goal_boost = max(goal_boost, self.goal_match_boost * 0.8)
                            break

                # Apply boosts
                total_boost = history_boost + goal_boost
                detection["salience"] = min(1.0, base_salience + total_boost)
                detection["history_boost"] = history_boost
                detection["goal_boost"] = goal_boost

            return detections

        except Exception as e:
            self._record_error(e)
            return detections

    def get_interaction_history(
        self,
        object_class: str,
    ) -> InteractionRecord | None:
        """Get interaction history for an object class.

        Args:
            object_class: Object class to query

        Returns:
            InteractionRecord or None if no history
        """
        return self._interaction_history.get(object_class.lower())

    def get_success_rate(self, object_class: str) -> float:
        """Get success rate for interactions with an object class.

        Returns:
            Success rate (0-1), or 0.5 if no history
        """
        record = self._interaction_history.get(object_class.lower())
        if not record:
            return 0.5

        # D56 (c)/(g): DECISIVE interactions only. Dividing by
        # ``total_interactions`` counts neutrals in the denominator and nowhere
        # in the numerator, so a class you interacted with fifty times and
        # learned nothing from reads 0.0 — identical to fifty failures, and
        # further from the truth than the 0.5 returned for no history at all.
        decisive = record.success_count + record.failure_count
        if decisive == 0:
            return 0.5

        return record.success_count / decisive

    # ─────────────────────────────────────────────────────────────────────────
    # Recording
    # ─────────────────────────────────────────────────────────────────────────

    def record_interaction(
        self,
        object_class: str,
        success: bool | None,
        goal: str | None = None,
    ) -> None:
        """Record an interaction with an object.

        D56 (g). ``success`` was a plain ``bool``, so a NEUTRAL outcome — the
        interaction happened and taught nothing directional — was unrepresentable
        and fell into the ``else`` branch as a FAILURE. That is not a rounding
        error: it is the difference between "this object hurt me" and "nothing
        came of it", and the consumer scales sensitization on exactly that
        distinction.

        Args:
            object_class: Class of object interacted with
            success: ``True`` success, ``False`` failure, ``None`` NEUTRAL —
                counted in ``total_interactions`` but in neither decisive
                bucket, so it dilutes confidence without asserting a direction.
            goal: Goal being pursued (optional)
        """
        if not self._healthy:
            return

        try:
            record = self._get_or_create_record(object_class)
            record.total_interactions += 1
            record.last_interaction = time.time()

            if success is None:
                pass  # neutral: observed, but neither bucket — see the docstring
            elif success:
                record.success_count += 1
                if goal and goal not in record.positive_goals:
                    record.positive_goals.append(goal)
                    record.positive_goals = record.positive_goals[-10:]
            else:
                record.failure_count += 1

        except Exception as e:
            self._record_error(e)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_or_create_record(self, object_class: str) -> InteractionRecord:
        """Get or create an interaction record for an object class."""
        key = object_class.lower()
        if key not in self._interaction_history:
            self._interaction_history[key] = InteractionRecord(object_class=key)
        return self._interaction_history[key]

    def _calculate_history_boost(self, object_class: str, now: float) -> float:
        """Calculate salience boost from interaction history."""
        record = self._interaction_history.get(object_class.lower())
        if not record:
            return 0.0

        # Base boost from success rate over DECISIVE interactions — the same
        # denominator `get_success_rate` uses. This is the LIVE consumer
        # (`enrich_salience` <- `MemoryHub.enrich_salience`), so leaving it on
        # the total would have made 5 successes among 50 neutrals boost
        # salience by 0.03 instead of 0.30: the neutrals dilute a rate they
        # carry no evidence about.
        decisive = record.success_count + record.failure_count
        if decisive == 0:
            return 0.0
        success_rate = record.success_count / decisive
        base_boost = success_rate * self.history_boost_factor

        # Decay based on time since last interaction
        age_days = (now - record.last_interaction) / 86400.0
        decay = max(0.0, 1.0 - (age_days / self.decay_days))

        return base_boost * decay

    def _goals_similar(self, goal1: str, goal2: str) -> bool:
        """Check if two goals are similar (simple word overlap)."""
        words1 = set(goal1.lower().split())
        words2 = set(goal2.lower().split())

        # Remove common words
        common_words = {"the", "a", "an", "find", "get", "pick", "up"}
        words1 -= common_words
        words2 -= common_words

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        return overlap / min(len(words1), len(words2)) > 0.3

    def _record_error(self, error: Exception) -> None:
        """Record an error and potentially disable the bridge."""
        self._error_count += 1
        logger.warning(
            "SalienceMemoryBridge error (%d/%d): %s",
            self._error_count,
            self._max_errors,
            error,
        )

        if self._error_count >= self._max_errors:
            self._healthy = False
            logger.error("SalienceMemoryBridge disabled after %d errors", self._error_count)

    @property
    def is_healthy(self) -> bool:
        """Check if bridge is operational."""
        return self._healthy

    def stats(self) -> dict[str, Any]:
        """Return bridge statistics."""
        total_interactions = sum(r.total_interactions for r in self._interaction_history.values())
        total_successes = sum(r.success_count for r in self._interaction_history.values())
        total_decisive = total_successes + sum(r.failure_count for r in self._interaction_history.values())
        return {
            "healthy": self._healthy,
            "error_count": self._error_count,
            "object_classes_tracked": len(self._interaction_history),
            "total_interactions": total_interactions,
            # Reporting only, but the same denominator rule — a fleet summary
            # that counts neutrals as failures reads as a much worse agent.
            "overall_success_rate": (total_successes / total_decisive if total_decisive > 0 else 0.5),
        }


__all__ = ["SalienceMemoryBridge", "InteractionRecord"]
