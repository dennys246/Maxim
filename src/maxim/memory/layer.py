"""MemoryLayer ABC — abstract protocol for bio-inspired memory layers.

Designed for three simultaneous implementations:
- Hippocampus (episodic memory)
- ATL/Anterior Temporal Lobe (semantic memory)
- Angular Gyrus (mathematical knowledge)

Each layer stores MemoryRecord subclasses, maintains a DependencyGraph
for associative recall, and supports persistence and consolidation.

Does NOT replace the Memory ABC in base.py (used by InMemoryMemory
for simple contexts).  This is a richer protocol for the bio-inspired
memory layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    from maxim.agents.bus import DependencyGraph
    from maxim.memory.types import CompressedRecord, MemoryRecord
    from maxim.models.bio_context import RetrievalContext


class MemoryLayer(ABC):
    """Abstract protocol for bio-inspired memory layers.

    Every layer manages MemoryRecord subclasses with:
    - store/get/remove for CRUD operations
    - recall with layer-specific filters
    - recall_associated for spreading-activation graph queries
    - save/load for JSON persistence
    - consolidate for sleep-cycle memory management
    - graph for cross-layer linking (via DependencyGraph)
    """

    @property
    @abstractmethod
    def layer_name(self) -> str:
        """Unique layer identifier: 'hippocampus', 'atl', 'angular_gyrus'."""
        ...

    @abstractmethod
    def store(self, record: "MemoryRecord", **kwargs: Any) -> str:
        """Store a record, return its ID."""
        ...

    @abstractmethod
    def get(self, record_id: str) -> "MemoryRecord | CompressedRecord | None":
        """Retrieve a record by ID. Updates access tracking."""
        ...

    @abstractmethod
    def remove(self, record_id: str) -> None:
        """Delete a record and clean up graph edges."""
        ...

    @abstractmethod
    def recall(
        self,
        limit: int = 10,
        *,
        retrieval_context: "RetrievalContext | None" = None,
        **filters: Any,
    ) -> "list[MemoryRecord | CompressedRecord]":
        """Retrieve records matching layer-specific filters."""
        ...

    @abstractmethod
    def recall_by_ids(
        self,
        record_ids: list[str],
    ) -> "list[MemoryRecord | CompressedRecord]":
        """Bulk-load records by ID. Skips missing IDs.

        More efficient than iterating get() in a loop — single lock
        acquisition for the entire batch.
        """
        ...

    @abstractmethod
    def recall_associated(
        self,
        seed_ids: list[str],
        limit: int = 10,
        **kwargs: Any,
    ) -> "list[tuple[MemoryRecord | CompressedRecord, float]]":
        """Spreading-activation recall from seed memories.

        Returns (record, activation_score) pairs sorted by score.
        """
        ...

    @property
    @abstractmethod
    def graph(self) -> "DependencyGraph":
        """Internal association graph for cross-layer linking."""
        ...

    @abstractmethod
    def save(self, path: str | None = None) -> None:
        """Persist layer state to disk."""
        ...

    @abstractmethod
    def load(self, path: str | None = None) -> None:
        """Restore layer state from disk."""
        ...

    @abstractmethod
    def consolidate(self, **kwargs: Any) -> dict[str, int]:
        """Run consolidation cycle (compress, prune, promote).

        Returns stats dict with keys like 'compressed', 'removed', 'promoted'.
        """
        ...

    @abstractmethod
    def register_capture_callback(
        self,
        callback: "Callable[[str, MemoryRecord], None]",
    ) -> None:
        """Register callback invoked when a record is stored."""
        ...

    @abstractmethod
    def register_deletion_callback(
        self,
        callback: "Callable[[str], None]",
    ) -> None:
        """Register callback invoked when a record is removed."""
        ...

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return layer statistics."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Total number of records (full + compressed)."""
        ...

    @abstractmethod
    def __iter__(self) -> "Iterator[MemoryRecord | CompressedRecord]":
        """Iterate over all records."""
        ...


__all__ = ["MemoryLayer"]
