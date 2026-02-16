"""MathBridge — connects mathematical cognition to the memory system.

Bridges the Angular Gyrus (math memory + computation) and the
NumericalWorkspace to the broader memory ecosystem:

- Caches computation results as episodic memories in Hippocampus
- Retrieves relevant past computations for similar contexts
- Promotes frequently-used math patterns to Angular Gyrus memory
- Provides numerical context summaries for agent prompts

Follows the same lifecycle pattern as SalienceMemoryBridge:
on_session_start(), on_session_end(), enrich_context(), stats().
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.math.angular_gyrus import AngularGyrus
    from maxim.math.workspace import NumericalWorkspace
    from maxim.memory.hippocampus import Hippocampus

logger = logging.getLogger(__name__)

# Minimum times a computation pattern must repeat before promotion
_PROMOTION_THRESHOLD = 5


@dataclass
class MathBridge:
    """Bridges mathematical cognition to the memory system.

    - Caches computation results as episodic memories
    - Retrieves relevant past computations for similar contexts
    - Promotes frequently-used math patterns to Angular Gyrus memory
    """

    hippocampus: Hippocampus
    angular_gyrus: AngularGyrus
    workspace: NumericalWorkspace

    # Health tracking (follows bridge convention)
    _healthy: bool = field(default=True, repr=False)
    _error_count: int = field(default=0, repr=False)
    _max_errors: int = field(default=5, repr=False)

    # Pattern tracking for promotion
    _pattern_counts: dict[str, int] = field(default_factory=lambda: collections.defaultdict(int), repr=False)

    def on_session_start(self) -> int:
        """Load workspace state from last session.

        Returns number of workspace values restored.
        """
        try:
            # Workspace loading is handled by MemoryHub passing
            # persisted data to workspace.load_from_dict()
            count = len(self.workspace.recall_all())
            logger.debug("MathBridge: session started with %d workspace values", count)
            return count
        except Exception as e:
            self._record_error(e)
            return 0

    def on_session_end(self) -> None:
        """Promote patterns and clean up.

        The actual persistence of workspace and angular_gyrus is handled
        by the mandatory consolidation cycle in MemoryHub.on_session_end().
        """
        try:
            promoted = self.promote_patterns()
            if promoted:
                logger.debug("MathBridge: promoted %d math patterns", len(promoted))
        except Exception as e:
            self._record_error(e)

    def enrich_context(self, goal: str | None = None) -> dict[str, Any]:
        """Retrieve numerically relevant context for agent prompts.

        Returns workspace summary + recent computation history.
        """
        try:
            context: dict[str, Any] = {
                "workspace_summary": self.workspace.summarize(),
                "recent_computations": self.workspace.get_history(limit=5),
            }

            # If there's a goal, try to find relevant math facts
            if goal and self.angular_gyrus:
                method = self.angular_gyrus.recall_method(goal)
                if method:
                    context["suggested_method"] = {
                        "name": method.name,
                        "verbal": method.verbal,
                        "code": method.code,
                    }

            return context
        except Exception as e:
            self._record_error(e)
            return {"workspace_summary": "Workspace: unavailable"}

    def cache_result(self, operation: str, inputs: dict[str, Any], result: Any) -> None:
        """Track a computation pattern for potential promotion."""
        try:
            # Build a pattern signature from the operation
            pattern_key = f"{operation}"
            self._pattern_counts[pattern_key] += 1
        except Exception as e:
            self._record_error(e)

    def promote_patterns(self) -> list[str]:
        """Scan computation history for repeated patterns.

        Frequently-used computations (>= threshold) that don't already
        exist in Angular Gyrus are promoted to MathMemorys.

        Returns list of promoted record IDs.
        """
        promoted: list[str] = []

        try:
            for pattern_key, count in self._pattern_counts.items():
                if count < _PROMOTION_THRESHOLD:
                    continue

                # Check if this pattern already exists in angular gyrus
                existing = self.angular_gyrus.recall(
                    limit=1,
                    name=pattern_key,
                    category="PATTERN",
                )
                if existing:
                    # Reinforce existing record
                    record = existing[0]
                    record.touch()
                    continue

                # Promote: create a new MathMemory from the pattern
                from maxim.math.math_types import MathMemory
                from maxim.math.types import MathCategory

                now = time.time()
                record = MathMemory(
                    id=f"promoted_{pattern_key}_{int(now)}",
                    timestamp=now,
                    created_at=now,
                    accessed_at=now,
                    name=pattern_key,
                    category=MathCategory.PATTERN,
                    domain="learned",
                    verbal=f"Frequently used computation pattern: {pattern_key} (used {count} times)",
                    code=pattern_key,
                    source="learned",
                    confidence=min(0.99, 0.5 + 0.1 * (count**0.5)),
                    observation_count=count,
                )
                rid = self.angular_gyrus.store(record)
                promoted.append(rid)

            # Reset counts for promoted patterns
            for rid in promoted:
                pattern_key = rid.split("_", 1)[-1].rsplit("_", 1)[0] if "_" in rid else rid
                self._pattern_counts.pop(pattern_key, None)

        except Exception as e:
            self._record_error(e)

        return promoted

    def stats(self) -> dict[str, Any]:
        return {
            "healthy": self._healthy,
            "error_count": self._error_count,
            "workspace_values": len(self.workspace.recall_all()),
            "tracked_patterns": len(self._pattern_counts),
            "history_entries": len(self.workspace.get_history(limit=9999)),
        }

    def is_healthy(self) -> bool:
        return self._healthy

    def _record_error(self, error: Exception) -> None:
        self._error_count += 1
        if self._error_count >= self._max_errors:
            self._healthy = False
        logger.debug("MathBridge error (%d/%d): %s", self._error_count, self._max_errors, error)
