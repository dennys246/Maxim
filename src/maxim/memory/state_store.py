"""StateStore - LRU cache for full state snapshots.

Memories store only a state_ref (hash), not the full snapshot.
This keeps EpisodicMemory objects lightweight (~200 bytes vs ~2KB).
Full snapshots are stored here and evicted LRU when capacity is reached.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any


class StateStore:
    """LRU cache for full state snapshots, referenced by hash.

    Thread-safe with its own lock (separate from Hippocampus lock).

    Example:
        store = StateStore(max_entries=1000)

        # Store a state and get its hash
        state = {"goal": "find cup", "mode": "explore"}
        state_ref = store.store(state)  # Returns "a1b2c3d4..."

        # Later, retrieve by hash
        state = store.get(state_ref)
    """

    def __init__(self, max_entries: int = 1000):
        """Initialize the state store.

        Args:
            max_entries: Maximum number of state snapshots to cache.
                         Older entries are evicted LRU when exceeded.
        """
        self._max_entries = max_entries
        self._store: dict[str, dict[str, Any]] = {}
        self._access_order: list[str] = []  # LRU order (most recent at end)
        self._lock = threading.Lock()

    def store(self, state: dict[str, Any]) -> str:
        """Store a state snapshot and return its hash reference.

        If the state already exists (same hash), just updates access order.
        Duplicate states share a single entry.

        Args:
            state: The state dictionary to store.

        Returns:
            16-character hash reference to the stored state.
        """
        # Create deterministic hash of state
        try:
            state_json = json.dumps(state, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fallback for non-serializable state
            state_json = repr(state)

        state_hash = hashlib.sha256(state_json.encode()).hexdigest()[:16]

        with self._lock:
            if state_hash in self._store:
                # Already exists - update access order
                try:
                    self._access_order.remove(state_hash)
                except ValueError:
                    pass
                self._access_order.append(state_hash)
                return state_hash

            # Store new state
            self._store[state_hash] = state
            self._access_order.append(state_hash)

            # Evict LRU if over capacity
            while len(self._store) > self._max_entries:
                lru_hash = self._access_order.pop(0)
                self._store.pop(lru_hash, None)

            return state_hash

    def get(self, state_ref: str) -> dict[str, Any] | None:
        """Retrieve a state snapshot by hash reference.

        Updates access order (marks as recently used).

        Args:
            state_ref: The hash reference from store().

        Returns:
            The state dictionary, or None if evicted from cache.
        """
        if not state_ref:
            return None

        with self._lock:
            state = self._store.get(state_ref)
            if state is not None:
                # Update access order
                try:
                    self._access_order.remove(state_ref)
                except ValueError:
                    pass
                self._access_order.append(state_ref)
            return state

    def contains(self, state_ref: str) -> bool:
        """Check if a state reference exists in the store."""
        with self._lock:
            return state_ref in self._store

    def __len__(self) -> int:
        """Return number of cached states."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Clear all cached states."""
        with self._lock:
            self._store.clear()
            self._access_order.clear()

    def stats(self) -> dict[str, Any]:
        """Return statistics about the store."""
        with self._lock:
            return {
                "entries": len(self._store),
                "max_entries": self._max_entries,
                "utilization": len(self._store) / self._max_entries if self._max_entries > 0 else 0,
            }


__all__ = ["StateStore"]
