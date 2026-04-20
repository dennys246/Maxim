"""Session-scoped imagination cache — prevents duplicate imagination within a session.

Thread-safe via RLock. Shared between AUT and orchestrator in sim mode.
Check the cache BEFORE ComponentIndex lookup to avoid repeated embedding
computations for phrases we've already resolved.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImaginationResult:
    """Outcome of an imagination lookup for a single entity phrase.

    Attributes:
        phrase: The normalized entity phrase that was queried.
        ref: The component ref (existing or newly imagined).
        imagined: True if this entity was freshly designed (not found in index).
        score: Similarity score if found via index, 1.0 if imagined.
        spec: The entity spec dict if imagined, None if found existing.
        validation_warnings: Warnings from quick validation (empty if clean).
    """

    phrase: str
    ref: str
    imagined: bool
    score: float = 1.0
    spec: dict[str, Any] | None = None
    validation_warnings: tuple[str, ...] = ()


@dataclass
class ImaginationCache:
    """Session-scoped cache keyed by normalized entity phrase.

    Stores both "found via index" (existing ref) and "imagined" (new ref)
    results. Thread-safe for concurrent access from AUT + orchestrator.
    """

    _entries: dict[str, ImaginationResult] = field(default_factory=dict)
    _mention_counts: dict[str, int] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @staticmethod
    def normalize(phrase: str) -> str:
        """Normalize a phrase for cache lookup: lowercase, strip, collapse whitespace."""
        return " ".join(phrase.lower().split())

    def get(self, phrase: str) -> ImaginationResult | None:
        """Return cached result for *phrase*, or None if not cached."""
        key = self.normalize(phrase)
        with self._lock:
            return self._entries.get(key)

    def put(self, result: ImaginationResult) -> None:
        """Cache an imagination result."""
        key = self.normalize(result.phrase)
        with self._lock:
            self._entries[key] = result

    def record_mention(self, phrase: str) -> int:
        """Increment and return the mention count for *phrase*."""
        key = self.normalize(phrase)
        with self._lock:
            self._mention_counts[key] = self._mention_counts.get(key, 0) + 1
            return self._mention_counts[key]

    def mention_count(self, phrase: str) -> int:
        """Return the current mention count for *phrase*."""
        key = self.normalize(phrase)
        with self._lock:
            return self._mention_counts.get(key, 0)

    def has(self, phrase: str) -> bool:
        """Check if *phrase* has a cached result."""
        key = self.normalize(phrase)
        with self._lock:
            return key in self._entries

    @property
    def size(self) -> int:
        """Number of cached results."""
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        """Clear all cached results and mention counts."""
        with self._lock:
            self._entries.clear()
            self._mention_counts.clear()
            log.debug("ImaginationCache cleared")

    def imagined_refs(self) -> list[str]:
        """Return refs of all imagined (not found-existing) entries."""
        with self._lock:
            return [r.ref for r in self._entries.values() if r.imagined]
