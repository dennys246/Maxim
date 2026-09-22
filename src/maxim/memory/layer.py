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
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator

from maxim.memory.types import ACTIVATION_SOURCES
from maxim.utils.logging import log_swallowed_exception

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

    def activate(self, record_ids: Iterable[str], *, source: str) -> int:
        """Record that these records were USED (memory-strength plan Phase 1). Returns how many.

        The one activation path for every store. Call it at a CONSUMPTION point -- content that
        reached a prompt, a prediction or a decision -- never for a bookkeeping read (echo
        filters, bulk loads, deletion callbacks, neighbour lookups); those use ``recall_by_ids``
        and stay uncounted. Unknown ids are skipped, like ``recall_by_ids``.

        **Never call it while holding ANY lock of this store, or inside ``for r in store:``**
        (``Hippocampus.__iter__`` holds the read lock for the whole loop). It re-enters the store
        through ``recall_by_ids``, and the store's ``RWLock`` is writer-priority and not
        re-entrant: a second read blocks behind any waiting writer (the capture worker), which is
        waiting on the first read -- a deadlock. Call it after the read has returned. It takes the
        store read lock once, releases it, then the per-record locks, so it never upgrades.
        Consumers should go through ``activate_after_use``, which cannot cost them their content.
        """
        if source not in ACTIVATION_SOURCES:
            raise ValueError(f"unknown activation source {source!r}; expected one of {sorted(ACTIVATION_SOURCES)}")
        records = self.recall_by_ids(list(dict.fromkeys(record_ids)))
        for record in records:
            record.activate(source)
        return len(records)

    def __bool__(self) -> bool:
        """A store that EXISTS is truthy even when EMPTY (#839).

        Defining ``__len__`` alone makes an empty store falsy, so ``if store:`` / ``if not store:``
        silently skipped it until something else wrote its first entry (e.g. MemoryAgent dropped
        every capture into an empty hippocampus). Presence is ``is not None``; emptiness is
        ``len(store) == 0``.
        """
        return True

    @abstractmethod
    def __len__(self) -> int:
        """Total number of records (full + compressed)."""
        ...

    @abstractmethod
    def __iter__(self) -> "Iterator[MemoryRecord | CompressedRecord]":
        """Iterate over all records."""
        ...


def activate_after_use(store: MemoryLayer | None, record_ids: Iterable[str], *, source: str) -> None:
    """Count a use at a consumption point without ever costing the consumer its content.

    Call it AFTER the consumer's content is built. A wrong ``source`` is a programming error and
    raises; a store-side failure is logged and swallowed, so counting can never remove what the
    consumer already received. ``record_ids`` may be a generator: it is consumed inside the guard,
    so a malformed record cannot abort the caller's own loop either.
    """
    if source not in ACTIVATION_SOURCES:
        raise ValueError(f"unknown activation source {source!r}; expected one of {sorted(ACTIVATION_SOURCES)}")
    if store is None:
        return
    try:
        store.activate(record_ids, source=source)
    except Exception as e:
        log_swallowed_exception(e, operation="memory_activate", context={"source": source})


__all__ = ["MemoryLayer", "activate_after_use"]
