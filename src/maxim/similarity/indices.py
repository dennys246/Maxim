"""Inverted indices for O(1) structural feature lookup.

Enables fast filtering by tool, outcome, mode, and temporal bins.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from maxim.similarity.signature import SituationSignature


@dataclass
class InvertedIndices:
    """Inverted indices for O(1) structural feature lookup.

    Maintains separate indices for:
    - tool_name → memory_ids
    - outcome_type → memory_ids
    - mode → memory_ids
    - temporal bins → memory_ids

    Attributes:
        _by_tool: Index by tool name
        _by_outcome: Index by outcome type
        _by_mode: Index by mode
        _by_hour: Index by hour bin
        _by_day: Index by day bin
    """

    _by_tool: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _by_outcome: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _by_mode: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _by_hour: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    _by_day: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, memory_id: str, signature: SituationSignature) -> None:
        """Add a memory to all relevant indices.

        Args:
            memory_id: Unique memory identifier
            signature: Situation signature with features
        """
        if signature.tool_name:
            self._by_tool[signature.tool_name].add(memory_id)
        if signature.outcome_type:
            self._by_outcome[signature.outcome_type].add(memory_id)
        if signature.mode:
            self._by_mode[signature.mode].add(memory_id)

        # Temporal indices
        hour_bin, day_bin, _, _ = signature.temporal_hash
        self._by_hour[hour_bin].add(memory_id)
        self._by_day[day_bin].add(memory_id)

    def remove(self, memory_id: str, signature: SituationSignature) -> None:
        """Remove a memory from all indices.

        Args:
            memory_id: Memory to remove
            signature: Signature to find index entries
        """
        if signature.tool_name and memory_id in self._by_tool.get(signature.tool_name, set()):
            self._by_tool[signature.tool_name].discard(memory_id)
        if signature.outcome_type and memory_id in self._by_outcome.get(signature.outcome_type, set()):
            self._by_outcome[signature.outcome_type].discard(memory_id)
        if signature.mode and memory_id in self._by_mode.get(signature.mode, set()):
            self._by_mode[signature.mode].discard(memory_id)

        hour_bin, day_bin, _, _ = signature.temporal_hash
        self._by_hour[hour_bin].discard(memory_id)
        self._by_day[day_bin].discard(memory_id)

    def query_tool(self, tool_name: str) -> set[str]:
        """Get all memory_ids using a specific tool."""
        return self._by_tool.get(tool_name, set()).copy()

    def query_outcome(self, outcome_type: str) -> set[str]:
        """Get all memory_ids with a specific outcome type."""
        return self._by_outcome.get(outcome_type, set()).copy()

    def query_mode(self, mode: str) -> set[str]:
        """Get all memory_ids in a specific mode."""
        return self._by_mode.get(mode, set()).copy()

    def query_hour(self, hour: int) -> set[str]:
        """Get all memory_ids from a specific hour (0-23)."""
        return self._by_hour.get(hour % 24, set()).copy()

    def query_day(self, day: int) -> set[str]:
        """Get all memory_ids from a specific day (0=Mon, 6=Sun)."""
        return self._by_day.get(day % 7, set()).copy()

    def query_intersection(
        self,
        tool: str | None = None,
        outcome: str | None = None,
        mode: str | None = None,
        hour: int | None = None,
        day: int | None = None,
    ) -> set[str]:
        """Find memories matching ALL specified criteria.

        Args:
            tool: Tool name filter
            outcome: Outcome type filter
            mode: Mode filter
            hour: Hour bin filter (0-23)
            day: Day bin filter (0-6)

        Returns:
            Set of memory_ids matching all criteria
        """
        sets_to_intersect: list[set[str]] = []

        if tool is not None:
            sets_to_intersect.append(self._by_tool.get(tool, set()))
        if outcome is not None:
            sets_to_intersect.append(self._by_outcome.get(outcome, set()))
        if mode is not None:
            sets_to_intersect.append(self._by_mode.get(mode, set()))
        if hour is not None:
            sets_to_intersect.append(self._by_hour.get(hour % 24, set()))
        if day is not None:
            sets_to_intersect.append(self._by_day.get(day % 7, set()))

        if not sets_to_intersect:
            return set()

        # Start with smallest set for efficiency
        sets_to_intersect.sort(key=len)
        result = sets_to_intersect[0].copy()
        for s in sets_to_intersect[1:]:
            result &= s
            if not result:
                break

        return result

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        return {
            "tools": len(self._by_tool),
            "outcomes": len(self._by_outcome),
            "modes": len(self._by_mode),
            "hours_used": len([h for h in self._by_hour.values() if h]),
            "days_used": len([d for d in self._by_day.values() if d]),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "by_tool": {k: list(v) for k, v in self._by_tool.items()},
            "by_outcome": {k: list(v) for k, v in self._by_outcome.items()},
            "by_mode": {k: list(v) for k, v in self._by_mode.items()},
            "by_hour": {str(k): list(v) for k, v in self._by_hour.items()},
            "by_day": {str(k): list(v) for k, v in self._by_day.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvertedIndices:
        """Deserialize from dictionary."""
        instance = cls()
        instance._by_tool = defaultdict(set, {k: set(v) for k, v in data.get("by_tool", {}).items()})
        instance._by_outcome = defaultdict(set, {k: set(v) for k, v in data.get("by_outcome", {}).items()})
        instance._by_mode = defaultdict(set, {k: set(v) for k, v in data.get("by_mode", {}).items()})
        instance._by_hour = defaultdict(set, {int(k): set(v) for k, v in data.get("by_hour", {}).items()})
        instance._by_day = defaultdict(set, {int(k): set(v) for k, v in data.get("by_day", {}).items()})
        return instance


__all__ = ["InvertedIndices"]
