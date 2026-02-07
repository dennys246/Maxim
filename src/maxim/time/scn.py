"""Suprachiasmatic Nucleus (SCN) - Temporal rhythm indexing.

Maintains binned indices for fast temporal queries across multiple time scales.
Inspired by the biological SCN that serves as the brain's master circadian pacemaker.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from maxim.time.temporal_signature import TemporalSignature

logger = logging.getLogger(__name__)


@dataclass
class SCN:
    """Suprachiasmatic Nucleus - temporal rhythm indexing.

    Maintains binned indices for fast temporal queries:
    - Circadian: 24 hourly bins (hour 0-23)
    - Weekly: 7 daily bins (Monday=0 through Sunday=6)
    - Monthly: 4 weekly bins (week 1-4 of month)
    - Annual: 12 monthly bins (January=0 through December=11)

    Memory Footprint (10K memories):
    - 47 total bins × ~10K/47 ≈ 213 memories/bin average
    - Total: ~500KB for indices (memory_ids are shared refs)

    Example:
        scn = SCN()

        # Register a memory with its temporal signature
        sig = TemporalSignature.from_timestamp(memory.timestamp)
        scn.register(memory.id, sig)

        # Query by time
        morning_memories = scn.query_hour(9)  # 9am memories
        monday_memories = scn.query_day(0)    # Monday memories

        # Find patterns
        patterns = scn.find_rhythmic_patterns(min_occurrences=5)
    """

    _circadian_bins: dict[int, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    _weekly_bins: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    _monthly_bins: dict[int, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    _annual_bins: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    _signatures: dict[str, TemporalSignature] = field(default_factory=dict)

    # Temporal priors for cold start
    _priors: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))

    def register(self, memory_id: str, signature: TemporalSignature) -> None:
        """Register a memory's temporal signature in all indices.

        Args:
            memory_id: Unique memory identifier
            signature: Temporal signature to register
        """
        hour_bin, day_bin, week_bin, month_bin = signature.to_bins()
        self._circadian_bins[hour_bin].add(memory_id)
        self._weekly_bins[day_bin].add(memory_id)
        self._monthly_bins[week_bin].add(memory_id)
        self._annual_bins[month_bin].add(memory_id)
        self._signatures[memory_id] = signature

    def unregister(self, memory_id: str) -> None:
        """Remove a memory from all temporal indices.

        Args:
            memory_id: Memory to remove
        """
        if memory_id not in self._signatures:
            return

        sig = self._signatures.pop(memory_id)
        hour_bin, day_bin, week_bin, month_bin = sig.to_bins()
        self._circadian_bins[hour_bin].discard(memory_id)
        self._weekly_bins[day_bin].discard(memory_id)
        self._monthly_bins[week_bin].discard(memory_id)
        self._annual_bins[month_bin].discard(memory_id)

    def remove_memory(self, memory_id: str) -> None:
        """Alias for unregister for deletion callback compatibility."""
        self.unregister(memory_id)

    def query_hour(self, hour: int) -> set[str]:
        """Get all memory_ids from a specific hour (0-23)."""
        return self._circadian_bins.get(hour % 24, set()).copy()

    def query_day(self, day: int) -> set[str]:
        """Get all memory_ids from a specific day (0=Monday, 6=Sunday)."""
        return self._weekly_bins.get(day % 7, set()).copy()

    def query_week_of_month(self, week: int) -> set[str]:
        """Get all memory_ids from a specific week of month (0-3)."""
        return self._monthly_bins.get(week % 4, set()).copy()

    def query_month(self, month: int) -> set[str]:
        """Get all memory_ids from a specific month (0=Jan, 11=Dec)."""
        return self._annual_bins.get(month % 12, set()).copy()

    def query_similar_time(
        self,
        signature: TemporalSignature,
        tolerance: int = 1,
    ) -> set[str]:
        """Find memories at similar times across all rhythms.

        Args:
            signature: Reference temporal signature
            tolerance: How many adjacent bins to include (default=1)

        Returns:
            Set of memory_ids that match in ANY rhythm within tolerance
        """
        hour_bin, day_bin, week_bin, month_bin = signature.to_bins()
        result: set[str] = set()

        # Collect from circadian bins with tolerance
        for h in range(hour_bin - tolerance, hour_bin + tolerance + 1):
            result.update(self._circadian_bins.get(h % 24, set()))

        return result

    def query_intersection(
        self,
        hour: int | None = None,
        day: int | None = None,
        week_of_month: int | None = None,
        month: int | None = None,
    ) -> set[str]:
        """Find memories matching ALL specified temporal criteria.

        Example: query_intersection(hour=9, day=0) returns memories
        from Monday mornings at 9am.

        Args:
            hour: Hour of day (0-23) or None for any
            day: Day of week (0-6, Mon=0) or None for any
            week_of_month: Week of month (0-3) or None for any
            month: Month (0-11, Jan=0) or None for any

        Returns:
            Set of memory_ids matching all criteria
        """
        sets_to_intersect: list[set[str]] = []

        if hour is not None:
            sets_to_intersect.append(self._circadian_bins.get(hour % 24, set()))
        if day is not None:
            sets_to_intersect.append(self._weekly_bins.get(day % 7, set()))
        if week_of_month is not None:
            sets_to_intersect.append(self._monthly_bins.get(week_of_month % 4, set()))
        if month is not None:
            sets_to_intersect.append(self._annual_bins.get(month % 12, set()))

        if not sets_to_intersect:
            return set()

        # Start with smallest set for efficiency
        sets_to_intersect.sort(key=len)
        result = sets_to_intersect[0].copy()
        for s in sets_to_intersect[1:]:
            result &= s

        return result

    def get_signature(self, memory_id: str) -> TemporalSignature | None:
        """Retrieve the temporal signature for a memory."""
        return self._signatures.get(memory_id)

    def find_rhythmic_patterns(
        self,
        min_occurrences: int = 3,
    ) -> dict[str, list[tuple[int, int]]]:
        """Identify bins with repeated activity patterns.

        Args:
            min_occurrences: Minimum memories in a bin to be considered a pattern

        Returns:
            Dict mapping rhythm type to list of (bin_id, count) tuples
            where count >= min_occurrences.
        """
        patterns: dict[str, list[tuple[int, int]]] = {
            "circadian": [],
            "weekly": [],
            "monthly": [],
            "annual": [],
        }

        for hour, memories in self._circadian_bins.items():
            if len(memories) >= min_occurrences:
                patterns["circadian"].append((hour, len(memories)))

        for day, memories in self._weekly_bins.items():
            if len(memories) >= min_occurrences:
                patterns["weekly"].append((day, len(memories)))

        for week, memories in self._monthly_bins.items():
            if len(memories) >= min_occurrences:
                patterns["monthly"].append((week, len(memories)))

        for month, memories in self._annual_bins.items():
            if len(memories) >= min_occurrences:
                patterns["annual"].append((month, len(memories)))

        return patterns

    def add_temporal_prior(self, pattern_name: str, hour_bin: int) -> None:
        """Add a temporal prior for cold start.

        Args:
            pattern_name: Name of the pattern (e.g., "morning_greeting")
            hour_bin: Hour bin (0-23) when this pattern typically occurs
        """
        self._priors[pattern_name].add(hour_bin % 24)

    def get_threshold_adjustment(self, signature: TemporalSignature) -> float:
        """Get threshold adjustment factor for a given time.

        Used by EscalationLearningBridge to adjust thresholds based on
        temporal context.

        Args:
            signature: Current temporal signature

        Returns:
            Adjustment factor (1.0 = no adjustment, <1.0 = lower threshold)
        """
        hour_bin, _, _, _ = signature.to_bins()

        # Fewer memories at this hour = higher threshold (less confident)
        hour_count = len(self._circadian_bins.get(hour_bin, set()))
        if hour_count < 5:
            return 1.2  # Higher threshold when we have little data
        elif hour_count > 50:
            return 0.9  # Lower threshold when we have lots of data

        return 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal Clustering Support (for sleep consolidation)
    # ─────────────────────────────────────────────────────────────────────────

    def get_bin_populations(self) -> dict[tuple[int, int], int]:
        """Get population counts for each (hour, day) bin.

        Returns:
            Dict mapping (hour_bin, day_bin) to count of memories.
            Pre-computed for O(1) lookups during sleep consolidation.
        """
        populations: dict[tuple[int, int], int] = defaultdict(int)

        for memory_id, sig in self._signatures.items():
            hour_bin, day_bin, _, _ = sig.to_bins()
            populations[(hour_bin, day_bin)] += 1

        return dict(populations)

    def get_bins(self, memory_id: str) -> tuple[int, int] | None:
        """Get (hour_bin, day_bin) for a memory.

        Args:
            memory_id: Memory to look up

        Returns:
            (hour_bin, day_bin) tuple, or None if not registered
        """
        sig = self._signatures.get(memory_id)
        if sig is None:
            return None
        hour_bin, day_bin, _, _ = sig.to_bins()
        return (hour_bin, day_bin)

    def get_temporal_cluster(
        self,
        hour: int,
        day: int,
    ) -> set[str]:
        """Get all memories in a specific (hour, day) cluster.

        Args:
            hour: Hour of day (0-23)
            day: Day of week (0-6, Mon=0)

        Returns:
            Set of memory_ids in this temporal cluster
        """
        hour_memories = self._circadian_bins.get(hour % 24, set())
        day_memories = self._weekly_bins.get(day % 7, set())
        return hour_memories & day_memories

    def get_all_clusters(self) -> dict[tuple[int, int], set[str]]:
        """Get all non-empty temporal clusters.

        Returns:
            Dict mapping (hour, day) to set of memory_ids.
            Only includes clusters with at least one memory.
        """
        clusters: dict[tuple[int, int], set[str]] = {}

        for memory_id, sig in self._signatures.items():
            hour_bin, day_bin, _, _ = sig.to_bins()
            key = (hour_bin, day_bin)
            if key not in clusters:
                clusters[key] = set()
            clusters[key].add(memory_id)

        return clusters

    def is_sole_representative(self, memory_id: str) -> bool:
        """Check if memory is the only one in its temporal bin.

        Memories that are sole representatives of a time slot
        should be protected from removal to maintain temporal coverage.

        Args:
            memory_id: Memory to check

        Returns:
            True if this is the only memory in its (hour, day) cluster
        """
        bins = self.get_bins(memory_id)
        if bins is None:
            return False

        hour, day = bins
        cluster = self.get_temporal_cluster(hour, day)
        return len(cluster) == 1 and memory_id in cluster

    def is_rhythmic_bin(self, memory_id: str, min_occurrences: int = 5) -> bool:
        """Check if memory belongs to a rhythmic pattern bin.

        Memories in rhythmic bins (time slots with many occurrences)
        represent learned behavioral patterns and should be preserved.

        Args:
            memory_id: Memory to check
            min_occurrences: Threshold for considering a bin "rhythmic"

        Returns:
            True if the memory's hour bin has >= min_occurrences
        """
        sig = self._signatures.get(memory_id)
        if sig is None:
            return False

        hour_bin, _, _, _ = sig.to_bins()
        return len(self._circadian_bins.get(hour_bin, set())) >= min_occurrences

    def stats(self) -> dict[str, Any]:
        """Return SCN statistics."""
        return {
            "total_signatures": len(self._signatures),
            "circadian_bins_used": len(
                [b for b in self._circadian_bins.values() if b]
            ),
            "weekly_bins_used": len([b for b in self._weekly_bins.values() if b]),
            "monthly_bins_used": len([b for b in self._monthly_bins.values() if b]),
            "annual_bins_used": len([b for b in self._annual_bins.values() if b]),
        }

    def __len__(self) -> int:
        """Number of memories with temporal signatures."""
        return len(self._signatures)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save SCN state to JSON file."""
        data = {
            "version": "1.0",
            "circadian_bins": {k: list(v) for k, v in self._circadian_bins.items()},
            "weekly_bins": {k: list(v) for k, v in self._weekly_bins.items()},
            "monthly_bins": {k: list(v) for k, v in self._monthly_bins.items()},
            "annual_bins": {k: list(v) for k, v in self._annual_bins.items()},
            "signatures": {k: v.to_dict() for k, v in self._signatures.items()},
            "priors": {k: list(v) for k, v in self._priors.items()},
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved SCN to %s (%d signatures)", path, len(self._signatures))

    def load(self, path: str) -> None:
        """Load SCN state from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", "0.0")
        if version != "1.0":
            raise ValueError(f"Unsupported SCN version: {version}")

        self._circadian_bins = defaultdict(
            set, {int(k): set(v) for k, v in data.get("circadian_bins", {}).items()}
        )
        self._weekly_bins = defaultdict(
            set, {int(k): set(v) for k, v in data.get("weekly_bins", {}).items()}
        )
        self._monthly_bins = defaultdict(
            set, {int(k): set(v) for k, v in data.get("monthly_bins", {}).items()}
        )
        self._annual_bins = defaultdict(
            set, {int(k): set(v) for k, v in data.get("annual_bins", {}).items()}
        )
        self._signatures = {
            k: TemporalSignature.from_dict(v)
            for k, v in data.get("signatures", {}).items()
        }
        self._priors = defaultdict(
            set, {k: set(v) for k, v in data.get("priors", {}).items()}
        )

        logger.info("Loaded SCN from %s (%d signatures)", path, len(self._signatures))

    def get_version(self) -> str:
        """Return data format version."""
        return "1.0"


__all__ = ["SCN"]