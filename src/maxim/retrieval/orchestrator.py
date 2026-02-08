"""Retrieval orchestrator - Combines multiple retrieval signals.

Coordinates multiple RetrievalSignal implementations to produce a unified
ranked list of memory candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from maxim.retrieval.signals import RetrievalCandidate, RetrievalSignal

if TYPE_CHECKING:
    from maxim.memory.hippocampus import Hippocampus
    from maxim.memory.types import CompressedMemory, EpisodicMemory

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation.

    Attributes:
        memory_id: ID of the retrieved memory
        memory: The actual memory object (if fetched)
        combined_score: Weighted combination of all signal scores
        signal_scores: Individual scores from each signal
        metadata: Combined metadata from all signals
    """

    memory_id: str
    memory: "EpisodicMemory | CompressedMemory | None" = None
    combined_score: float = 0.0
    signal_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorConfig:
    """Configuration for the retrieval orchestrator.

    Attributes:
        max_candidates: Maximum candidates to return
        min_score: Minimum combined score to include
        fetch_memories: Whether to fetch full memory objects
        signal_weights: Custom weights per signal name (default 1.0)
        merge_strategy: How to combine scores ("max", "sum", "weighted_avg")
    """

    max_candidates: int = 20
    min_score: float = 0.1
    fetch_memories: bool = True
    signal_weights: dict[str, float] = field(default_factory=dict)
    merge_strategy: str = "weighted_avg"


class RetrievalOrchestrator:
    """Orchestrates multiple retrieval signals to find relevant memories.

    Combines candidates from multiple signals, merges scores, and returns
    a ranked list of retrieval results.

    Signals can be:
    - Temporal: Find memories from mentioned times
    - Semantic: Find memories similar in meaning (future)
    - Keyword: Find memories matching specific terms (future)
    - Contextual: Find memories from similar situations (future)

    Example:
        # Setup
        orchestrator = RetrievalOrchestrator(hippocampus=hippocampus)
        orchestrator.register_signal(TemporalReferenceSignal(scn=scn))
        orchestrator.register_signal(KeywordSignal(hippocampus=hippocampus))

        # Query
        results = orchestrator.retrieve("What did we discuss last Tuesday?")

        for result in results:
            print(f"Memory: {result.memory_id}")
            print(f"Score: {result.combined_score:.2f}")
            print(f"Signals: {result.signal_scores}")
    """

    def __init__(
        self,
        hippocampus: "Hippocampus | None" = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            hippocampus: Hippocampus for fetching memories (optional)
            config: Orchestrator configuration
        """
        self._hippocampus = hippocampus
        self._config = config or OrchestratorConfig()
        self._signals: list[RetrievalSignal] = []
        self._stats = {
            "queries": 0,
            "candidates_found": 0,
            "memories_returned": 0,
        }

    def register_signal(
        self,
        signal: RetrievalSignal,
        weight: float | None = None,
    ) -> None:
        """Register a retrieval signal.

        Args:
            signal: The signal to register
            weight: Optional custom weight (overrides config)
        """
        self._signals.append(signal)
        # Sort by priority (descending)
        self._signals.sort(key=lambda s: s.priority, reverse=True)

        if weight is not None:
            self._config.signal_weights[signal.name] = weight

        logger.debug(
            "Registered signal: %s (priority=%d, weight=%.2f)",
            signal.name,
            signal.priority,
            weight or self._config.signal_weights.get(signal.name, 1.0),
        )

    def unregister_signal(self, name: str) -> bool:
        """Remove a signal by name.

        Args:
            name: Signal name to remove

        Returns:
            True if removed, False if not found
        """
        original_len = len(self._signals)
        self._signals = [s for s in self._signals if s.name != name]
        return len(self._signals) < original_len

    def retrieve(
        self,
        text: str,
        *,
        limit: int | None = None,
        min_score: float | None = None,
        signals: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant memories based on input text.

        Runs all registered signals (or specified subset), merges results,
        and returns ranked candidates.

        Args:
            text: Input text to analyze
            limit: Maximum results (overrides config)
            min_score: Minimum score threshold (overrides config)
            signals: Only use these signals by name (None = all)

        Returns:
            List of RetrievalResult, sorted by combined_score descending
        """
        if not text or not text.strip():
            return []

        self._stats["queries"] += 1
        limit = limit or self._config.max_candidates
        min_score = min_score if min_score is not None else self._config.min_score

        # Select signals to use
        active_signals = self._signals
        if signals:
            active_signals = [s for s in self._signals if s.name in signals]

        if not active_signals:
            return []

        # Collect candidates from all signals
        all_candidates: dict[str, list[RetrievalCandidate]] = {}

        for signal in active_signals:
            try:
                candidates = signal.extract(text)
                for candidate in candidates:
                    if candidate.memory_id not in all_candidates:
                        all_candidates[candidate.memory_id] = []
                    all_candidates[candidate.memory_id].append(candidate)
            except Exception as e:
                logger.warning("Signal %s failed: %s", signal.name, e)

        self._stats["candidates_found"] += len(all_candidates)

        if not all_candidates:
            return []

        # Merge candidates into results
        results = self._merge_candidates(all_candidates)

        # Filter by minimum score
        results = [r for r in results if r.combined_score >= min_score]

        # Sort by combined score
        results.sort(key=lambda r: r.combined_score, reverse=True)

        # Apply limit
        results = results[:limit]

        # Fetch memories if configured
        if self._config.fetch_memories and self._hippocampus:
            for result in results:
                result.memory = self._hippocampus.get(result.memory_id)

        self._stats["memories_returned"] += len(results)
        return results

    def retrieve_memory_ids(
        self,
        text: str,
        *,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> set[str]:
        """Convenience method to get just memory IDs.

        Useful when you only need IDs for further processing.

        Args:
            text: Input text to analyze
            limit: Maximum results
            min_score: Minimum score threshold

        Returns:
            Set of memory IDs
        """
        results = self.retrieve(text, limit=limit, min_score=min_score)
        return {r.memory_id for r in results}

    def _merge_candidates(
        self,
        candidates: dict[str, list[RetrievalCandidate]],
    ) -> list[RetrievalResult]:
        """Merge candidates from multiple signals into results."""
        results: list[RetrievalResult] = []

        for memory_id, candidate_list in candidates.items():
            signal_scores: dict[str, float] = {}
            metadata: dict[str, Any] = {}

            for candidate in candidate_list:
                signal_scores[candidate.source] = candidate.score
                # Merge metadata, prefixed by source
                for key, value in candidate.metadata.items():
                    metadata[f"{candidate.source}.{key}"] = value

            # Calculate combined score
            combined_score = self._calculate_combined_score(signal_scores)

            results.append(
                RetrievalResult(
                    memory_id=memory_id,
                    combined_score=combined_score,
                    signal_scores=signal_scores,
                    metadata=metadata,
                )
            )

        return results

    def _calculate_combined_score(
        self,
        signal_scores: dict[str, float],
    ) -> float:
        """Calculate combined score from individual signal scores."""
        if not signal_scores:
            return 0.0

        strategy = self._config.merge_strategy

        if strategy == "max":
            return max(signal_scores.values())

        elif strategy == "sum":
            return min(1.0, sum(signal_scores.values()))

        elif strategy == "weighted_avg":
            total_weight = 0.0
            weighted_sum = 0.0

            for name, score in signal_scores.items():
                weight = self._config.signal_weights.get(name, 1.0)
                weighted_sum += score * weight
                total_weight += weight

            return weighted_sum / total_weight if total_weight > 0 else 0.0

        else:
            # Default to weighted average
            return sum(signal_scores.values()) / len(signal_scores)

    def warmup(self) -> None:
        """Warmup all registered signals."""
        for signal in self._signals:
            try:
                signal.warmup()
            except Exception as e:
                logger.warning("Signal %s warmup failed: %s", signal.name, e)

    @property
    def signal_names(self) -> list[str]:
        """Get names of all registered signals."""
        return [s.name for s in self._signals]

    def stats(self) -> dict[str, Any]:
        """Return orchestrator statistics."""
        return {
            **self._stats,
            "registered_signals": len(self._signals),
            "signal_names": self.signal_names,
        }


__all__ = ["RetrievalOrchestrator", "RetrievalResult", "OrchestratorConfig"]