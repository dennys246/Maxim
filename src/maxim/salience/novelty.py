"""Thread-safe wrapper for NoveltyTracker.

The underlying NoveltyTracker is not thread-safe, but the Default Network
needs to access it from multiple threads (perception thread, DN thread).
This wrapper provides safe concurrent access.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from maxim.inference.segment_vision import NoveltyTracker


class ThreadSafeNoveltyTracker:
    """Thread-safe wrapper around NoveltyTracker.

    Provides the same interface as NoveltyTracker but with locking
    for safe concurrent access from multiple threads.

    Example:
        tracker = NoveltyTracker()
        safe_tracker = ThreadSafeNoveltyTracker(tracker)

        # Safe to call from any thread
        novelty = safe_tracker.get_novelty(track_id)
        safe_tracker.update(track_id)
    """

    def __init__(self, tracker: NoveltyTracker | None = None) -> None:
        """Initialize with an existing or new NoveltyTracker.

        Args:
            tracker: Existing NoveltyTracker to wrap. If None, creates a new one.
        """
        self._tracker = tracker or NoveltyTracker()
        self._lock = threading.Lock()

    def get_novelty(self, track_id: Any, class_id: int | None = None) -> float:
        """Get novelty score for a track_id (thread-safe).

        Args:
            track_id: The track ID to look up.
            class_id: Optional COCO class ID for class-level modulation.

        Returns:
            Novelty score (higher = more novel).
        """
        with self._lock:
            return self._tracker.get_novelty(track_id, class_id=class_id)

    def update(self, track_id: Any) -> None:
        """Mark a track_id as seen now (thread-safe).

        Args:
            track_id: The track ID that was seen.
        """
        with self._lock:
            self._tracker.update(track_id)

    def update_with_class(self, track_id: Any, class_id: int) -> None:
        """Mark a track_id as seen and register its class (thread-safe).

        If this is a new track_id, increments the unique instance count
        for its class, driving class-level habituation.

        Args:
            track_id: The track ID that was seen.
            class_id: COCO class ID for this detection.
        """
        with self._lock:
            self._tracker.update_with_class(track_id, class_id)

    def get_class_novelty(self, class_id: int) -> float:
        """Get class-level novelty score (thread-safe).

        Args:
            class_id: COCO class ID.

        Returns:
            Class novelty score (higher = less familiar category).
        """
        with self._lock:
            return self._tracker.get_class_novelty(class_id)

    def update_batch(self, track_ids: list[Any]) -> None:
        """Mark multiple track_ids as seen now (thread-safe).

        Args:
            track_ids: List of track IDs that were seen.
        """
        with self._lock:
            self._tracker.update_batch(track_ids)

    def get_novelty_batch(self, track_ids: list[Any], class_ids: dict[Any, int] | None = None) -> dict[Any, float]:
        """Get novelty scores for multiple track_ids in one lock acquisition.

        More efficient than calling get_novelty() repeatedly.

        Args:
            track_ids: List of track IDs to look up.
            class_ids: Optional mapping of track_id -> class_id for class modulation.

        Returns:
            Dict mapping track_id to novelty score.
        """
        with self._lock:
            if class_ids:
                return {tid: self._tracker.get_novelty(tid, class_id=class_ids.get(tid)) for tid in track_ids}
            return {tid: self._tracker.get_novelty(tid) for tid in track_ids}

    def focus(self, track_id: Any) -> None:
        """Mark a track_id as the current focus target (thread-safe).

        This resets novelty to max and starts the slow decay.
        Call this when the object is being actively attended to.

        Args:
            track_id: The track ID being focused on.
        """
        with self._lock:
            self._tracker.focus(track_id)

    def focus_and_get_novelty(self, track_id: Any) -> float:
        """Atomically get novelty and mark as focused.

        Useful when you want to check novelty and focus in one operation.

        Args:
            track_id: The track ID to process.

        Returns:
            Novelty score (before the focus update).
        """
        with self._lock:
            novelty = self._tracker.get_novelty(track_id)
            self._tracker.focus(track_id)
            return novelty

    def update_and_get_novelty(self, track_id: Any) -> float:
        """Atomically get novelty and mark as seen (not focused).

        Note: This does NOT affect novelty - use focus_and_get_novelty() instead.

        Args:
            track_id: The track ID to process.

        Returns:
            Novelty score.
        """
        with self._lock:
            novelty = self._tracker.get_novelty(track_id)
            self._tracker.update(track_id)
            return novelty

    def set_modulation_lookup(self, lookup: Callable[[int], float] | None) -> None:
        """Set sensitization modulation callback (thread-safe).

        Args:
            lookup: Callable taking class_id (int), returning float >= 1.0. None to disable.
        """
        with self._lock:
            self._tracker.set_modulation_lookup(lookup)

    @property
    def sensitization_ceiling(self) -> float:
        """Max multiplier for sensitized class novelty."""
        return self._tracker.sensitization_ceiling

    @property
    def focus_decay_seconds(self) -> float:
        """Time for novelty to decay from max to min while focused."""
        return self._tracker.focus_decay_seconds

    @property
    def recovery_seconds(self) -> float:
        """Time for novelty to recover from min to ~63% when not focused."""
        return self._tracker.recovery_seconds

    @property
    def decay_seconds(self) -> float:
        """Alias for recovery_seconds (backward compatibility)."""
        return self._tracker.decay_seconds

    @property
    def max_novelty(self) -> float:
        """Maximum novelty score for never-seen objects."""
        return self._tracker.max_novelty

    @property
    def min_novelty(self) -> float:
        """Minimum novelty score for frequently-seen objects."""
        return self._tracker.min_novelty

    @property
    def tracked_count(self) -> int:
        """Number of track IDs currently being tracked."""
        with self._lock:
            return len(self._tracker._track_times)


__all__ = [
    "ThreadSafeNoveltyTracker",
]
