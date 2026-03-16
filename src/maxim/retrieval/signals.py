"""Retrieval signal protocol and base types.

Defines the interface for retrieval signals - components that extract
memory retrieval candidates from input text using different strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class RetrievalCandidate:
    """A candidate memory surfaced by a retrieval signal.

    Attributes:
        memory_id: ID of the candidate memory
        score: Relevance score (0.0-1.0, higher = more relevant)
        source: Name of the signal that found this candidate
        metadata: Additional context about why this was selected
    """

    memory_id: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate score is in range."""
        if not 0.0 <= self.score <= 1.0:
            object.__setattr__(self, "score", max(0.0, min(1.0, self.score)))


class RetrievalSignal(ABC):
    """Abstract base for retrieval signals.

    A retrieval signal extracts memory candidates from input text using
    a specific strategy (temporal references, semantic similarity, etc.).

    Signals are designed to be:
    - Stateless: Can be called repeatedly without side effects
    - Composable: Multiple signals can be combined via RetrievalOrchestrator
    - Lightweight: Should complete quickly (<50ms) for real-time use

    Example implementation:
        class KeywordSignal(RetrievalSignal):
            def __init__(self, hippocampus: Hippocampus):
                self._hippocampus = hippocampus

            @property
            def name(self) -> str:
                return "keyword"

            def extract(self, text: str) -> list[RetrievalCandidate]:
                # Extract keywords and query hippocampus
                keywords = self._extract_keywords(text)
                candidates = []
                for keyword in keywords:
                    memories = self._hippocampus.recall(goal=keyword)
                    for mem in memories:
                        candidates.append(RetrievalCandidate(
                            memory_id=mem.id,
                            score=0.7,
                            source=self.name,
                            metadata={"keyword": keyword},
                        ))
                return candidates
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this signal type."""
        ...

    @abstractmethod
    def extract(self, text: str) -> list[RetrievalCandidate]:
        """Extract retrieval candidates from input text.

        Args:
            text: Input text to analyze (user message, CLI input, etc.)

        Returns:
            List of RetrievalCandidate with memory_ids and scores.
            May return empty list if no relevant signals found.
        """
        ...

    def extract_batch(self, texts: list[str]) -> list[list[RetrievalCandidate]]:
        """Extract candidates from multiple texts.

        Default implementation calls extract() for each text.
        Override for batch-optimized implementations.

        Args:
            texts: List of input texts

        Returns:
            List of candidate lists, one per input text
        """
        return [self.extract(text) for text in texts]

    @property
    def priority(self) -> int:
        """Priority for signal ordering (higher = processed first).

        Default is 50. Override to adjust relative priority.
        """
        return 50

    def warmup(self) -> None:
        """Optional warmup to preload resources.

        Called once at startup if the signal needs initialization.
        """
        pass


__all__ = ["RetrievalCandidate", "RetrievalSignal"]