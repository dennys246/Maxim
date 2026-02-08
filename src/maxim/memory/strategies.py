"""Memory management strategies for sleep consolidation.

This module provides different strategies for scoring memories and
deciding which to retain, compress, or remove during sleep consolidation.

Strategies:
- AccessBasedStrategy: Prioritizes frequently and recently accessed memories
- ImportanceBasedStrategy: Prioritizes high-salience and novel memories
- CompositeStrategy: Weighted combination of multiple strategies
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maxim.memory.types import CompressedMemory, EpisodicMemory
    from maxim.time.scn import SCN

# Type alias for memory types
MemoryRecord = "EpisodicMemory | CompressedMemory"


class MemoryStrategy(ABC):
    """Abstract base class for memory management strategies.

    Strategies determine how memories are scored for retention and
    whether they should be compressed to save space.

    Memory lifecycle during sleep():
    1. score_for_retention() evaluates each memory
    2. Low scores → removal (permanent deletion)
    3. Medium scores → compression (if should_compress() agrees)
    4. High scores → preservation (full record kept)
    """

    @abstractmethod
    def score_for_retention(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> float:
        """Score a memory for retention (0.0 = remove, 1.0 = definitely keep).

        Args:
            record: The memory record to evaluate (full or compressed).
            now: Current timestamp for age calculations.
            degree: Graph degree (number of connections) for centrality boost.

        Returns:
            Retention score 0.0-1.0 where:
            - < retention_threshold → remove
            - < compression_threshold → compress
            - >= compression_threshold → preserve
        """

    @abstractmethod
    def should_compress(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> bool:
        """Determine if a full memory should be compressed.

        Only called for EpisodicMemory records that score between
        retention_threshold and compression_threshold.

        Args:
            record: The memory record to evaluate.
            now: Current timestamp.
            degree: Graph degree (number of connections).

        Returns:
            True if should compress to CompressedMemory.
        """


class AccessBasedStrategy(MemoryStrategy):
    """Memory strategy based on access recency and frequency.

    Memories that haven't been accessed recently and are infrequently
    accessed are candidates for removal. High-access memories are
    preserved regardless of age.

    Good for: Systems with heavy query loads where frequently-used
    memories should always be available.
    """

    def __init__(
        self,
        max_age_without_access: float = 7 * 24 * 3600,  # 1 week default
        compression_age: float = 24 * 3600,  # Compress after 1 day
        access_count_threshold: int = 5,  # High-access threshold
        centrality_weight: float = 0.3,  # How much connectivity matters
    ):
        """Initialize AccessBasedStrategy.

        Args:
            max_age_without_access: Seconds without access before removal.
            compression_age: Seconds after creation before compression eligible.
            access_count_threshold: Access count to consider "frequently used".
            centrality_weight: Weight for graph centrality boost (0-1).
        """
        self.max_age_without_access = max_age_without_access
        self.compression_age = compression_age
        self.access_count_threshold = access_count_threshold
        self.centrality_weight = centrality_weight

    def score_for_retention(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> float:
        """Score based on access patterns and graph importance."""
        from maxim.memory.types import CompressedMemory, EpisodicMemory

        # Time since last access
        time_since_access = now - record.accessed_at
        max_age = self.max_age_without_access

        # Base score from access recency (exponential decay)
        if time_since_access >= max_age:
            recency_score = 0.0
        else:
            recency_score = 1.0 - (time_since_access / max_age)

        # Boost from access frequency
        access_boost = min(1.0, record.access_count / self.access_count_threshold)

        # Boost from graph centrality
        # Normalize: assume max useful degree is ~20
        centrality_boost = min(1.0, degree / 20) if degree > 0 else 0.0

        # Combine scores
        score = (
            recency_score * 0.5
            + access_boost * 0.2
            + centrality_boost * self.centrality_weight
        )

        # Always keep if accessed many times
        if record.access_count >= self.access_count_threshold * 2:
            score = max(score, 0.8)

        # Always keep successful user interactions (check for EpisodicMemory)
        if isinstance(record, EpisodicMemory):
            if record.perception.cli_input or record.perception.transcript:
                if record.outcome.success:
                    score = max(score, 0.7)
        elif isinstance(record, CompressedMemory):
            if record.had_user_input and record.success:
                score = max(score, 0.7)

        return min(1.0, score)

    def should_compress(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> bool:
        """Compress based on age and access patterns."""
        from maxim.memory.types import CompressedMemory, EpisodicMemory

        # Already compressed
        if isinstance(record, CompressedMemory):
            return False

        age = now - record.created_at
        if age < self.compression_age:
            return False

        # Keep full records for frequently accessed memories
        if record.access_count >= self.access_count_threshold:
            return False

        # Keep full records for user interactions longer
        if isinstance(record, EpisodicMemory):
            if record.perception.cli_input or record.perception.transcript:
                return age > self.compression_age * 3  # 3 days

        return True


class ImportanceBasedStrategy(MemoryStrategy):
    """Memory strategy based on importance and outcomes.

    Preserves memories that are:
    - High novelty (rare/unique experiences)
    - High salience (contextually important)
    - Successful outcomes (worth learning from)
    - User interactions (direct commands/conversations)

    Good for: Learning systems where memorable experiences
    matter more than recent ones.
    """

    def __init__(
        self,
        max_age: float = 30 * 24 * 3600,  # 30 days default
        compression_age: float = 3 * 24 * 3600,  # Compress after 3 days
        novelty_weight: float = 0.2,
        success_weight: float = 0.2,
    ):
        """Initialize ImportanceBasedStrategy.

        Args:
            max_age: Maximum age before hard removal (seconds).
            compression_age: Age before compression eligible (seconds).
            novelty_weight: Weight for novelty score (0-1).
            success_weight: Weight for success bonus (0-1).
        """
        self.max_age = max_age
        self.compression_age = compression_age
        self.novelty_weight = novelty_weight
        self.success_weight = success_weight

    def score_for_retention(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> float:
        """Score based on importance and outcomes."""
        from maxim.memory.types import CompressedMemory, EpisodicMemory

        age = now - record.created_at

        # Hard cutoff for very old, rarely accessed memories
        if age > self.max_age and record.access_count < 3:
            return 0.0

        # Base score from age (linear decay)
        age_score = max(0.0, 1.0 - (age / self.max_age))

        # Success bonus
        if isinstance(record, EpisodicMemory):
            success_score = 1.0 if record.outcome.success else 0.3
            novelty_score = record.perception.novelty
            had_user_input = bool(record.perception.cli_input or record.perception.transcript)
        elif isinstance(record, CompressedMemory):
            success_score = 1.0 if record.success else 0.3
            novelty_score = record.novelty
            had_user_input = record.had_user_input
        else:
            success_score = 0.5
            novelty_score = 0.5
            had_user_input = False

        # User interaction bonus
        user_bonus = 0.3 if had_user_input else 0.0

        # Combine
        score = (
            age_score * 0.4
            + success_score * self.success_weight
            + novelty_score * self.novelty_weight
            + user_bonus
        )

        return min(1.0, score)

    def should_compress(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> bool:
        """Compress based on age and importance."""
        from maxim.memory.types import CompressedMemory, EpisodicMemory

        if isinstance(record, CompressedMemory):
            return False

        age = now - record.created_at
        if age < self.compression_age:
            return False

        if isinstance(record, EpisodicMemory):
            # Keep full records for user interactions
            if record.perception.cli_input or record.perception.transcript:
                return age > self.compression_age * 3  # 9 days

            # Keep full records for high-novelty events
            if record.perception.novelty > 0.8:
                return age > self.compression_age * 2  # 6 days

        return True


class CompositeStrategy(MemoryStrategy):
    """Combines multiple strategies with configurable weights.

    Useful for balancing multiple concerns (e.g., both access patterns
    and importance should matter).
    """

    def __init__(
        self,
        strategies: list[tuple[MemoryStrategy, float]],
    ):
        """Initialize CompositeStrategy.

        Args:
            strategies: List of (strategy, weight) tuples.
                Weights are normalized automatically.
        """
        total_weight = sum(w for _, w in strategies)
        if total_weight == 0:
            raise ValueError("Total weight must be > 0")

        # Normalize weights
        self.strategies = [(s, w / total_weight) for s, w in strategies]

    def score_for_retention(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> float:
        """Weighted average of all strategy scores."""
        total = 0.0
        for strategy, weight in self.strategies:
            total += strategy.score_for_retention(record, now, degree) * weight
        return total

    def should_compress(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> bool:
        """Compress if majority of strategies agree."""
        votes = sum(
            1
            for strategy, _ in self.strategies
            if strategy.should_compress(record, now, degree)
        )
        return votes > len(self.strategies) / 2


class TemporalAwareStrategy(MemoryStrategy):
    """SCN-integrated strategy for temporally-aware memory consolidation.

    Wraps a base strategy and adds temporal context from SCN:
    - Protects sole representatives of time slots
    - Boosts memories in rhythmic patterns
    - Enables temporal clustering for batch operations

    This is the most efficient strategy for long-running systems with
    established temporal patterns.
    """

    def __init__(
        self,
        scn: "SCN",
        base_strategy: MemoryStrategy | None = None,
        sole_representative_boost: float = 1.5,
        rhythmic_pattern_boost: float = 1.3,
        min_rhythmic_occurrences: int = 5,
    ):
        """Initialize TemporalAwareStrategy.

        Args:
            scn: SCN instance for temporal lookups
            base_strategy: Underlying strategy (defaults to AccessBasedStrategy)
            sole_representative_boost: Score multiplier for sole representatives
            rhythmic_pattern_boost: Score multiplier for rhythmic patterns
            min_rhythmic_occurrences: Min memories in bin to consider rhythmic
        """
        self.scn = scn
        self.base = base_strategy or AccessBasedStrategy()
        self.sole_representative_boost = sole_representative_boost
        self.rhythmic_pattern_boost = rhythmic_pattern_boost
        self.min_rhythmic_occurrences = min_rhythmic_occurrences

        # Pre-computed bin populations (set during prepare())
        self._bin_populations: dict[tuple[int, int], int] = {}
        self._prepared = False

    def prepare(self) -> None:
        """Pre-compute bin populations for efficient lookups.

        Call this once at the start of sleep() for O(1) lookups.
        """
        self._bin_populations = self.scn.get_bin_populations()
        self._prepared = True

    def score_for_retention(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> float:
        """Score with temporal context boosts."""
        base_score = self.base.score_for_retention(record, now, degree)

        # Get temporal bins for this memory
        bins = self.scn.get_bins(record.id)
        if bins is None:
            return base_score  # No temporal data, use base score

        # Apply temporal boosts
        score = base_score

        # Boost 1: Sole representative of time slot (preserve coverage)
        if self._prepared:
            population = self._bin_populations.get(bins, 0)
            if population == 1:
                score *= self.sole_representative_boost
        else:
            if self.scn.is_sole_representative(record.id):
                score *= self.sole_representative_boost

        # Boost 2: Part of rhythmic pattern (learned behavior)
        if self.scn.is_rhythmic_bin(record.id, self.min_rhythmic_occurrences):
            score *= self.rhythmic_pattern_boost

        return min(1.0, score)

    def should_compress(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> bool:
        """Defer to base strategy for compression decisions."""
        return self.base.should_compress(record, now, degree)

    def select_cluster_representative(
        self,
        cluster: set[str],
        records: dict[str, MemoryRecord],
        now: float,
    ) -> str | None:
        """Select best representative from a temporal cluster.

        Used for temporal clustering: instead of keeping all memories
        in a cluster, keep the best one and remove/compress the rest.

        Args:
            cluster: Set of memory_ids in the cluster
            records: Dict of memory_id -> record for scoring
            now: Current timestamp

        Returns:
            memory_id of best representative, or None if cluster is empty
        """
        if not cluster:
            return None

        best_id = None
        best_score = -1.0

        for memory_id in cluster:
            record = records.get(memory_id)
            if record is None:
                continue

            score = self.base.score_for_retention(record, now, 0)

            # Prefer long-term memories
            if getattr(record, "long_term", False):
                score *= 1.5

            # Prefer user interactions
            from maxim.memory.types import CompressedMemory, EpisodicMemory

            if isinstance(record, EpisodicMemory):
                if record.perception.cli_input or record.perception.transcript:
                    score *= 1.3
            elif isinstance(record, CompressedMemory):
                if record.had_user_input:
                    score *= 1.3

            if score > best_score:
                best_score = score
                best_id = memory_id

        return best_id


__all__ = [
    "MemoryStrategy",
    "AccessBasedStrategy",
    "ImportanceBasedStrategy",
    "CompositeStrategy",
    "TemporalAwareStrategy",
]