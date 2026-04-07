"""Suprachiasmatic Nucleus (SCN) - Temporal rhythm indexing.

Maintains binned indices for fast temporal queries across multiple time scales.
Inspired by the biological SCN that serves as the brain's master circadian pacemaker.
"""

from __future__ import annotations

import json
import logging
import time as _time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from maxim.time.temporal_signature import TemporalSignature

logger = logging.getLogger(__name__)


@dataclass
class BinEntry:
    """A memory entry in an SCN time bin, ordered by insertion time."""

    memory_id: str
    significance: float  # From significance score or salience
    registered_at: float  # Timestamp of registration


class BoundedBin:
    """A time bin with max capacity and significance-based eviction.

    Entries are stored in insertion order (newest at front).
    When full, evicts the least significant entry from the older half.

    Compatible with the old ``set[str]`` interface via ``get_ids()``,
    ``__contains__``, ``__len__``, and ``__iter__``.
    """

    def __init__(self, max_size: int = 200) -> None:
        self.max_size = max_size
        self.entries: list[BinEntry] = []  # Ordered by insertion (newest first)
        self._id_set: set[str] = set()  # For O(1) membership checks

    # ── set-compatible interface ──────────────────────────────────────────

    def add(self, memory_id: str, significance: float = 0.5) -> None:
        """Add a memory to the bin, evicting least-significant old entry if full."""
        if memory_id in self._id_set:
            return  # Already present

        entry = BinEntry(memory_id, significance, _time.time())

        if len(self.entries) >= self.max_size:
            # Find least significant entry, searching from back (oldest)
            min_idx = len(self.entries) - 1
            min_sig = self.entries[min_idx].significance
            for i in range(len(self.entries) - 2, len(self.entries) // 2, -1):
                if self.entries[i].significance < min_sig:
                    min_sig = self.entries[i].significance
                    min_idx = i
            # Only evict if new entry is more significant
            if significance > min_sig:
                evicted = self.entries.pop(min_idx)
                self._id_set.discard(evicted.memory_id)
            else:
                return  # New entry isn't significant enough

        self.entries.insert(0, entry)  # Newest at front
        self._id_set.add(memory_id)

    def discard(self, memory_id: str) -> None:
        """Remove a memory from the bin (no-op if absent). ``set.discard`` compat."""
        self.remove(memory_id)

    def remove(self, memory_id: str) -> None:
        """Remove a memory from the bin."""
        if memory_id not in self._id_set:
            return
        self.entries = [e for e in self.entries if e.memory_id != memory_id]
        self._id_set.discard(memory_id)

    def get_ids(self) -> set[str]:
        """Return a copy of all memory IDs in this bin."""
        return self._id_set.copy()

    def copy(self) -> set[str]:
        """Return a copy of IDs — ``set.copy()`` compat for query methods."""
        return self._id_set.copy()

    def __contains__(self, item: object) -> bool:
        return item in self._id_set

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):  # type: ignore[override]
        return iter(self._id_set)

    def __bool__(self) -> bool:
        return len(self.entries) > 0

    def __and__(self, other: set[str]) -> set[str]:  # type: ignore[override]
        return self._id_set & other

    def __rand__(self, other: set[str]) -> set[str]:
        return other & self._id_set

    # ── Persistence helpers ──────────────────────────────────────────────

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize entries for JSON persistence."""
        return [
            {
                "memory_id": e.memory_id,
                "significance": e.significance,
                "registered_at": e.registered_at,
            }
            for e in self.entries
        ]

    @classmethod
    def from_list(cls, data: list[Any], max_size: int = 200) -> BoundedBin:
        """Deserialize from JSON. Accepts list[dict] (v3) or list[str] (v2 compat)."""
        bb = cls(max_size=max_size)
        for item in data:
            if isinstance(item, dict):
                entry = BinEntry(
                    memory_id=item["memory_id"],
                    significance=item.get("significance", 0.5),
                    registered_at=item.get("registered_at", 0.0),
                )
                if entry.memory_id not in bb._id_set:
                    bb.entries.append(entry)
                    bb._id_set.add(entry.memory_id)
            else:
                # v2 compat: plain string memory IDs
                mid = str(item)
                if mid not in bb._id_set:
                    bb.entries.append(BinEntry(mid, 0.5, 0.0))
                    bb._id_set.add(mid)
        return bb


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

    _circadian_bins: dict[int, BoundedBin] = field(default_factory=lambda: defaultdict(BoundedBin))
    _weekly_bins: dict[int, BoundedBin] = field(default_factory=lambda: defaultdict(BoundedBin))
    _monthly_bins: dict[int, BoundedBin] = field(default_factory=lambda: defaultdict(BoundedBin))
    _annual_bins: dict[int, BoundedBin] = field(default_factory=lambda: defaultdict(BoundedBin))
    _signatures: dict[str, TemporalSignature] = field(default_factory=dict)

    # Temporal priors for cold start
    _priors: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))

    # Coupled oscillator network (optional, disabled by default)
    _oscillator: Any = field(default=None, repr=False)

    def register(
        self,
        memory_id: str,
        signature: TemporalSignature,
        significance: float = 0.5,
    ) -> None:
        """Register a memory's temporal signature in all indices.

        Args:
            memory_id: Unique memory identifier
            signature: Temporal signature to register
            significance: Significance score for bounded-bin eviction (0-1)
        """
        hour_bin, day_bin, week_bin, month_bin = signature.to_bins()
        self._circadian_bins[hour_bin].add(memory_id, significance)
        self._weekly_bins[day_bin].add(memory_id, significance)
        self._monthly_bins[week_bin].add(memory_id, significance)
        self._annual_bins[month_bin].add(memory_id, significance)
        self._signatures[memory_id] = signature

        # Feed coupled oscillator (if enabled)
        if self._oscillator is not None:
            self._oscillator.observe(signature)

    def register_external(
        self,
        memory_id: str,
        signature: TemporalSignature,
        peer_id: str,
        clock_estimator: Any,
        significance: float = 0.5,
    ) -> None:
        """Register a memory from a peer with clock skew correction.

        Corrects the TemporalSignature using the PeerClockEstimator before
        registering, so bins align with local time. If no clock estimate
        is available yet, registers with the uncorrected signature.

        Args:
            memory_id: Unique memory identifier (should include peer prefix)
            signature: Peer's TemporalSignature (uncorrected)
            peer_id: The peer's node_id
            clock_estimator: PeerClockEstimator instance
            significance: Significance score (0-1)
        """
        try:
            corrected = clock_estimator.correct_signature(peer_id, signature)
        except Exception:
            corrected = signature  # Fall back to uncorrected
        self.register(memory_id, corrected, significance)

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
        b = self._circadian_bins.get(hour % 24)
        return b.get_ids() if b else set()

    def query_day(self, day: int) -> set[str]:
        """Get all memory_ids from a specific day (0=Monday, 6=Sunday)."""
        b = self._weekly_bins.get(day % 7)
        return b.get_ids() if b else set()

    def query_week_of_month(self, week: int) -> set[str]:
        """Get all memory_ids from a specific week of month (0-3)."""
        b = self._monthly_bins.get(week % 4)
        return b.get_ids() if b else set()

    def query_month(self, month: int) -> set[str]:
        """Get all memory_ids from a specific month (0=Jan, 11=Dec)."""
        b = self._annual_bins.get(month % 12)
        return b.get_ids() if b else set()

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
            b = self._circadian_bins.get(h % 24)
            if b:
                result.update(b.get_ids())

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
            b = self._circadian_bins.get(hour % 24)
            sets_to_intersect.append(b.get_ids() if b else set())
        if day is not None:
            b = self._weekly_bins.get(day % 7)
            sets_to_intersect.append(b.get_ids() if b else set())
        if week_of_month is not None:
            b = self._monthly_bins.get(week_of_month % 4)
            sets_to_intersect.append(b.get_ids() if b else set())
        if month is not None:
            b = self._annual_bins.get(month % 12)
            sets_to_intersect.append(b.get_ids() if b else set())

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
        b = self._circadian_bins.get(hour_bin)
        hour_count = len(b) if b else 0
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
        hb = self._circadian_bins.get(hour % 24)
        db = self._weekly_bins.get(day % 7)
        hour_ids = hb.get_ids() if hb else set()
        day_ids = db.get_ids() if db else set()
        return hour_ids & day_ids

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
        b = self._circadian_bins.get(hour_bin)
        return len(b) >= min_occurrences if b else False

    def stats(self) -> dict[str, Any]:
        """Return SCN statistics."""
        s: dict[str, Any] = {
            "total_signatures": len(self._signatures),
            "circadian_bins_used": len([b for b in self._circadian_bins.values() if b]),
            "weekly_bins_used": len([b for b in self._weekly_bins.values() if b]),
            "monthly_bins_used": len([b for b in self._monthly_bins.values() if b]),
            "annual_bins_used": len([b for b in self._annual_bins.values() if b]),
        }
        if self._oscillator is not None:
            s["oscillator_enabled"] = True
            s["oscillator_observations"] = self._oscillator._observation_count
            s["oscillator_coherence"] = self._oscillator.phase_coherence()
        else:
            s["oscillator_enabled"] = False
        return s

    def __len__(self) -> int:
        """Number of memories with temporal signatures."""
        return len(self._signatures)

    # ─────────────────────────────────────────────────────────────────────────
    # Coupled Oscillator Network
    # ─────────────────────────────────────────────────────────────────────────

    def enable_oscillator(self, config: Any = None) -> None:
        """Enable the coupled oscillator network.

        Args:
            config: OscillatorConfig or None for defaults.
        """
        from maxim.time.oscillator import OscillatorNetwork

        self._oscillator = OscillatorNetwork(config)
        logger.info("SCN oscillator network enabled")

    @property
    def oscillator(self) -> Any:
        """Access the oscillator network (None if disabled)."""
        return self._oscillator

    def predict_next_occurrence(self, target_hour: float, max_hours: float = 168.0) -> float | None:
        """Predict hours until the circadian oscillator reaches target phase.

        Args:
            target_hour: Target hour of day (0-23.999).
            max_hours: Maximum lookahead in hours.

        Returns:
            Hours until target, or None if oscillator disabled/not found.
        """
        if self._oscillator is None:
            return None
        target_phase = target_hour / 24.0
        return self._oscillator.predict_next_occurrence(target_phase, oscillator=0, max_hours=max_hours)

    def phase_coherence(self) -> float | None:
        """Kuramoto order parameter — how synchronized are the oscillators?

        Returns 0.0-1.0, or None if oscillator disabled.
        """
        if self._oscillator is None:
            return None
        return self._oscillator.phase_coherence()

    def coupling_strength(self, osc_a: int, osc_b: int) -> float | None:
        """Read coupling weight between two oscillators.

        Returns weight, or None if oscillator disabled.
        """
        if self._oscillator is None:
            return None
        return self._oscillator.coupling_strength_between(osc_a, osc_b)

    def temporal_anomaly_score(self, signature: TemporalSignature) -> float | None:
        """Score how anomalous a temporal signature is vs predicted phases.

        Returns 0.0-1.0, or None if oscillator disabled.
        """
        if self._oscillator is None:
            return None
        return self._oscillator.temporal_anomaly_score(signature)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save SCN state to JSON file (v3.0 with bounded bins)."""
        data: dict[str, Any] = {
            "version": "3.0",
            "circadian_bins": {str(k): v.to_list() for k, v in self._circadian_bins.items() if v},
            "weekly_bins": {str(k): v.to_list() for k, v in self._weekly_bins.items() if v},
            "monthly_bins": {str(k): v.to_list() for k, v in self._monthly_bins.items() if v},
            "annual_bins": {str(k): v.to_list() for k, v in self._annual_bins.items() if v},
            "signatures": {k: v.to_dict() for k, v in self._signatures.items()},
            "priors": {k: list(v) for k, v in self._priors.items()},
        }

        if self._oscillator is not None:
            data["oscillator"] = self._oscillator.to_dict()

        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        import os

        os.replace(tmp_path, path)
        logger.info("Saved SCN to %s (%d signatures)", path, len(self._signatures))

    def load(self, path: str) -> None:
        """Load SCN state from JSON file (supports v1.0, v2.0, and v3.0)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", "0.0")
        if version not in ("1.0", "2.0", "3.0"):
            raise ValueError(f"Unsupported SCN version: {version}")

        # BoundedBin.from_list handles both v2 (list[str]) and v3 (list[dict])
        self._circadian_bins = defaultdict(
            BoundedBin,
            {int(k): BoundedBin.from_list(v) for k, v in data.get("circadian_bins", {}).items()},
        )
        self._weekly_bins = defaultdict(
            BoundedBin,
            {int(k): BoundedBin.from_list(v) for k, v in data.get("weekly_bins", {}).items()},
        )
        self._monthly_bins = defaultdict(
            BoundedBin,
            {int(k): BoundedBin.from_list(v) for k, v in data.get("monthly_bins", {}).items()},
        )
        self._annual_bins = defaultdict(
            BoundedBin,
            {int(k): BoundedBin.from_list(v) for k, v in data.get("annual_bins", {}).items()},
        )
        self._signatures = {k: TemporalSignature.from_dict(v) for k, v in data.get("signatures", {}).items()}
        self._priors = defaultdict(set, {k: set(v) for k, v in data.get("priors", {}).items()})

        # Restore oscillator if present (v2.0)
        osc_data = data.get("oscillator")
        if osc_data is not None:
            from maxim.time.oscillator import OscillatorNetwork

            self._oscillator = OscillatorNetwork.from_dict(osc_data)
            logger.info("Restored oscillator (%d observations)", self._oscillator._observation_count)

        logger.info("Loaded SCN v%s from %s (%d signatures)", version, path, len(self._signatures))

    def get_version(self) -> str:
        """Return data format version."""
        return "3.0"


__all__ = ["BinEntry", "BoundedBin", "SCN"]
