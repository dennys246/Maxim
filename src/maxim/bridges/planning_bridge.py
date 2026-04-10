"""PlanHistoryBridge - Retrieves successful plan templates from memory.

Connects: Hippocampus <-> NAc <-> DecisionEngine

Problem: Planner generates plans from scratch. No memory of what worked before.

Solution: Query similar past plans and use outcomes to guide new planning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.decisions.nac import NAc
    from maxim.memory.hippocampus import Hippocampus
    from maxim.similarity.ec import EntorhinalCortex

logger = logging.getLogger(__name__)


@dataclass
class PlanTemplate:
    """A successful plan template from memory.

    Represents a historical plan that achieved a similar goal.
    """

    memory_id: str
    goal: str
    tool_sequence: list[str]
    success_count: int
    failure_count: int
    avg_execution_time_ms: float
    last_used: float
    similarity: float  # Similarity to current goal


@dataclass
class PlanHistoryBridge:
    """Retrieves successful plan templates from Hippocampus.

    Features:
    - Template retrieval: Find plans that achieved similar goals
    - Success prediction: Predict plan success via NAc causal inference
    - Outcome recording: Update predictions after execution

    Example:
        bridge = PlanHistoryBridge(
            hippocampus=hippocampus,
            nac=nac,
            ec=ec,
        )

        # Find templates for current goal
        templates = bridge.get_plan_templates("find the mug")
        for template in templates:
            print(f"Goal: {template.goal}")
            print(f"Tools: {template.tool_sequence}")
            print(f"Success rate: {template.success_count / (template.success_count + template.failure_count):.1%}")

        # Predict success for a potential plan
        prediction = bridge.get_predicted_success(
            goal="find mug",
            tool_sequence=["look_around", "grasp"],
        )

        # Record outcome after execution
        bridge.record_plan_outcome(
            goal="find mug",
            tool_sequence=["look_around", "grasp"],
            success=True,
            execution_time_ms=1500,
        )
    """

    hippocampus: "Hippocampus"
    nac: "NAc"
    ec: "EntorhinalCortex"

    # Cached templates: goal_hash -> list[PlanTemplate]
    _template_cache: dict[str, list[PlanTemplate]] = field(default_factory=dict)
    _cache_expiry: float = 300.0  # Cache TTL in seconds
    _cache_timestamp: float = 0.0

    # Configuration
    min_template_confidence: float = 0.3
    max_templates_per_query: int = 10

    # Health tracking
    _healthy: bool = True
    _error_count: int = 0
    _max_errors: int = 5

    def __post_init__(self) -> None:
        """Initialize default factory fields."""
        if not hasattr(self, "_template_cache") or self._template_cache is None:
            self._template_cache = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Template Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def get_plan_templates(
        self,
        goal: str,
        limit: int = 5,
    ) -> list[PlanTemplate]:
        """Find successful plan templates for a goal.

        Queries Hippocampus for memories with similar goals that
        resulted in successful outcomes.

        Args:
            goal: Current goal text
            limit: Maximum templates to return

        Returns:
            List of PlanTemplate, sorted by relevance/success
        """
        if not self._healthy:
            return self._get_fallback_templates(goal)

        try:
            templates: list[PlanTemplate] = []

            # Query successful memories with matching goal patterns
            memories = self.hippocampus.recall(
                limit=100,
                goal=goal,
                success=True,
            )

            # Expand results with associative recall for richer context
            if memories:
                seed_ids = [m.id for m in memories[:10]]
                try:
                    associated = self.hippocampus.recall_associated(seed_ids, limit=20)
                    seen_ids = {m.id for m in memories}
                    for mem, _score in associated:
                        if mem.id not in seen_ids:
                            # Only include successful associated memories
                            success_val = (
                                mem.success
                                if hasattr(mem, "success")
                                else mem.outcome.success
                                if hasattr(mem, "outcome")
                                else False
                            )
                            if success_val:
                                seen_ids.add(mem.id)
                                memories.append(mem)
                except Exception:
                    pass  # Associative recall is best-effort

            # If exact match fails, try broader search
            if not memories:
                memories = self.hippocampus.recall(
                    limit=100,
                    success=True,
                )

            # Build templates from memories
            template_map: dict[str, PlanTemplate] = {}

            for memory in memories:
                if not hasattr(memory, "action"):
                    continue

                tool_name = memory.action.tool_name if hasattr(memory, "action") else ""
                if not tool_name:
                    continue

                # Get goal from memory
                mem_goal = ""
                if hasattr(memory, "context") and memory.context.active_goal:
                    mem_goal = memory.context.active_goal

                # Calculate similarity to current goal
                similarity = self._calculate_goal_similarity(goal, mem_goal)
                if similarity < self.min_template_confidence:
                    continue

                # Create or update template
                template_key = f"{mem_goal}:{tool_name}"
                if template_key in template_map:
                    template = template_map[template_key]
                    template.success_count += 1
                    template.last_used = max(template.last_used, memory.timestamp)
                    template.similarity = max(template.similarity, similarity)
                else:
                    template_map[template_key] = PlanTemplate(
                        memory_id=memory.id,
                        goal=mem_goal,
                        tool_sequence=[tool_name],
                        success_count=1,
                        failure_count=0,
                        avg_execution_time_ms=0.0,
                        last_used=memory.timestamp,
                        similarity=similarity,
                    )

            templates = list(template_map.values())

            # Sort by combined score (success * similarity)
            templates.sort(
                key=lambda t: t.success_count * t.similarity,
                reverse=True,
            )

            return templates[:limit]

        except Exception as e:
            self._record_error(e)
            return self._get_fallback_templates(goal)

    def _get_fallback_templates(self, goal: str) -> list[PlanTemplate]:
        """Return generic fallback templates when no history exists."""
        # Common action patterns that usually work
        return [
            PlanTemplate(
                memory_id="",
                goal=goal,
                tool_sequence=["look_around"],
                success_count=1,
                failure_count=0,
                avg_execution_time_ms=500.0,
                last_used=0.0,
                similarity=0.3,
            ),
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Success Prediction
    # ─────────────────────────────────────────────────────────────────────────

    def get_predicted_success(
        self,
        goal: str,
        tool_sequence: list[str],
        context: dict[str, Any] | None = None,
    ) -> float:
        """Predict success probability for a plan.

        Uses NAc causal inference to predict whether the tool sequence
        will succeed for the given goal.

        Args:
            goal: Goal to achieve
            tool_sequence: Sequence of tools to use
            context: Current context (optional)

        Returns:
            Predicted success probability (0-1)
        """
        if not self._healthy:
            return 0.5

        try:
            if not tool_sequence:
                return 0.0

            # Query NAc for predictions on each tool
            total_confidence = 0.0
            predictions = []

            for tool in tool_sequence:
                prediction = self.nac.predict(
                    event_type="tool",
                    event_signature=tool,
                    context={"goal": goal, **(context or {})},
                )
                if prediction:
                    predictions.append(prediction)
                    total_confidence += prediction.confidence

            if not predictions:
                return 0.5  # No history

            # Combine predictions
            # Use minimum predicted value (weakest link)
            min_value = min(p.predicted_value for p in predictions)
            avg_confidence = total_confidence / len(predictions)

            # Weight by confidence
            return min_value * avg_confidence + 0.5 * (1 - avg_confidence)

        except Exception as e:
            self._record_error(e)
            return 0.5

    def get_tool_success_rate(
        self,
        tool_name: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[float, int]:
        """Get historical success rate for a tool.

        Args:
            tool_name: Tool to query
            context: Optional context filter

        Returns:
            (success_rate, sample_count) tuple
        """
        try:
            from maxim.decisions.causal_link import Valence

            # Query NAc for tool outcomes
            links = self.nac.get_links_for_event(tool_name)

            if not links:
                return (0.5, 0)

            # Valence.value is a string ("positive"/"negative"/"neutral"/"unknown"),
            # not a number — compare against the enum directly.
            positive = sum(1 for link in links if link.outcome_valence == Valence.POSITIVE)
            return (positive / len(links), len(links))

        except Exception as e:
            self._record_error(e)
            return (0.5, 0)

    # ─────────────────────────────────────────────────────────────────────────
    # Outcome Recording
    # ─────────────────────────────────────────────────────────────────────────

    def record_plan_outcome(
        self,
        goal: str,
        tool_sequence: list[str],
        success: bool,
        execution_time_ms: float = 0.0,
        memory_id: str | None = None,
    ) -> None:
        """Record the outcome of a plan execution.

        Updates NAc causal links for each tool in the sequence. Uses
        :meth:`NAc.observe` (not ``record_outcome``) because we already
        know both the event (the tool call) and the outcome (the plan's
        success/failure) — there's no temporal-window matching needed.
        ``record_outcome`` would silently no-op when called outside an
        active event window, which previously caused
        ``test_record_plan_outcome`` to fail.

        Args:
            goal: Goal that was pursued
            tool_sequence: Tools that were used
            success: Whether the plan succeeded
            execution_time_ms: Total execution time per tool (used as
                ``delta_seconds`` for the temporal delta distribution)
            memory_id: Associated Hippocampus memory ID
        """
        if not self._healthy:
            return

        try:
            from maxim.decisions.causal_link import Valence

            valence = Valence.POSITIVE if success else Valence.NEGATIVE
            # Convert to seconds; default to 0.1s when caller didn't measure.
            delta_s = max(0.001, execution_time_ms / 1000.0) if execution_time_ms > 0 else 0.1

            for tool in tool_sequence:
                self.nac.observe(
                    event_type="tool",
                    event_signature=tool,
                    outcome_type="plan_result",
                    outcome_signature=f"plan:{goal}:{'success' if success else 'failure'}",
                    outcome_valence=valence,
                    delta_seconds=delta_s,
                    context={"goal": goal},
                    memory_id=memory_id,
                )

            # Clear template cache to pick up new data
            self._template_cache.clear()

        except Exception as e:
            self._record_error(e)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_goal_similarity(self, goal1: str, goal2: str) -> float:
        """Calculate similarity between two goals.

        Simple word overlap for now. Phase 4 will add semantic similarity.
        """
        if not goal1 or not goal2:
            return 0.0

        words1 = set(goal1.lower().split())
        words2 = set(goal2.lower().split())

        # Remove common words
        common_words = {"the", "a", "an", "to", "and", "or", "in", "on", "at"}
        words1 -= common_words
        words2 -= common_words

        if not words1 or not words2:
            return 0.3  # Some baseline similarity

        overlap = len(words1 & words2)
        total = len(words1 | words2)

        return overlap / total if total > 0 else 0.0

    def _record_error(self, error: Exception) -> None:
        """Record an error and potentially disable the bridge."""
        self._error_count += 1
        logger.warning(
            "PlanHistoryBridge error (%d/%d): %s",
            self._error_count,
            self._max_errors,
            error,
        )

        if self._error_count >= self._max_errors:
            self._healthy = False
            logger.error("PlanHistoryBridge disabled after %d errors", self._error_count)

    @property
    def is_healthy(self) -> bool:
        """Check if bridge is operational."""
        return self._healthy

    def stats(self) -> dict[str, Any]:
        """Return bridge statistics."""
        return {
            "healthy": self._healthy,
            "error_count": self._error_count,
            "cached_templates": sum(len(t) for t in self._template_cache.values()),
        }


__all__ = ["PlanHistoryBridge", "PlanTemplate"]
