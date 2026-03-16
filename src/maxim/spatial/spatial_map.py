"""Spatial map for tracking reachable positions and spatial memory.

Implements a learned workspace model using an efficient spatial hash grid that:
1. Tracks which positions are reachable (successful movements)
2. Tracks which positions are unreachable (IK failures)
3. Remembers what was seen at different gaze positions (spatial memory)
4. Provides safe exploration targets within learned boundaries

Data Structure:
- Spatial hash grid for O(1) cell lookup by position
- Compact numpy arrays for reachability data (if available)
- Neighbor lookup via grid coordinate arithmetic
- Memory entries with spatial indexing for fast "what's nearby" queries
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Try to use numpy for efficient array operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class SpatialMapConfig:
    """Configuration for spatial map.

    Attributes:
        grid_resolution: Size of grid cells in normalized UV coordinates (0-1).
        failure_decay_seconds: Time for failure penalty to decay.
        memory_decay_seconds: Time for spatial memories to decay.
        max_memories_per_cell: Max interesting objects to remember per cell.
        min_reachable_confidence: Minimum confidence to consider position reachable.
        default_reachable_range: Default UV range assumed reachable before learning.
        neighbor_search_radius: Grid cells to search for nearby memories.
    """
    grid_resolution: float = 0.1  # 10x10 grid over image space
    failure_decay_seconds: float = 60.0  # Failures decay over 1 minute
    memory_decay_seconds: float = 300.0  # Memories decay over 5 minutes
    max_memories_per_cell: int = 5
    min_reachable_confidence: float = 0.3
    # Conservative default range (center of image is usually safe)
    default_reachable_range: tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.75)
    neighbor_search_radius: int = 2  # Search 2 cells in each direction


@dataclass(slots=True)
class SpatialMemory:
    """A memory of something interesting seen at a location.

    Uses __slots__ for memory efficiency.
    """
    class_id: int | None
    label: str | None
    track_id: int | None
    confidence: float
    timestamp: float
    novelty: float = 1.0
    pixel_u: float = 0.0  # Exact position within cell
    pixel_v: float = 0.0


class ReachabilityGrid:
    """Efficient grid for tracking movement success/failure.

    Uses numpy arrays if available for compact storage and fast operations.
    Falls back to dict-based storage otherwise.
    """

    def __init__(self, grid_size: int, decay_seconds: float) -> None:
        self.grid_size = grid_size
        self.decay_seconds = decay_seconds

        if HAS_NUMPY:
            # Numpy arrays for efficient storage
            # Shape: (grid_size, grid_size)
            self._success_count = np.zeros((grid_size, grid_size), dtype=np.int32)
            self._failure_count = np.zeros((grid_size, grid_size), dtype=np.int32)
            self._last_success = np.zeros((grid_size, grid_size), dtype=np.float64)
            self._last_failure = np.zeros((grid_size, grid_size), dtype=np.float64)
            self._use_numpy = True
        else:
            # Dict-based fallback
            self._data: dict[tuple[int, int], list[int | float]] = {}
            # [success_count, failure_count, last_success, last_failure]
            self._use_numpy = False

    def record(self, grid_u: int, grid_v: int, success: bool) -> None:
        """Record a movement attempt."""
        now = time.time()

        if self._use_numpy:
            if success:
                self._success_count[grid_u, grid_v] += 1
                self._last_success[grid_u, grid_v] = now
            else:
                self._failure_count[grid_u, grid_v] += 1
                self._last_failure[grid_u, grid_v] = now
        else:
            key = (grid_u, grid_v)
            if key not in self._data:
                self._data[key] = [0, 0, 0.0, 0.0]

            if success:
                self._data[key][0] += 1
                self._data[key][2] = now
            else:
                self._data[key][1] += 1
                self._data[key][3] = now

    def get_reachability(self, grid_u: int, grid_v: int) -> float:
        """Get reachability score (0=unreachable, 1=reachable)."""
        now = time.time()

        if self._use_numpy:
            success = int(self._success_count[grid_u, grid_v])
            failure = int(self._failure_count[grid_u, grid_v])
            last_success = float(self._last_success[grid_u, grid_v])
            last_failure = float(self._last_failure[grid_u, grid_v])
        else:
            key = (grid_u, grid_v)
            if key not in self._data:
                return 0.5  # Unknown
            success, failure, last_success, last_failure = self._data[key]

        if success == 0 and failure == 0:
            return 0.5  # Unknown

        # Decay failure influence over time
        time_since_failure = now - last_failure if last_failure > 0 else float('inf')
        failure_weight = math.exp(-time_since_failure / self.decay_seconds) if time_since_failure < 1000 else 0.0

        total = success + failure
        success_ratio = success / total if total > 0 else 0.5

        # Recent failures reduce reachability
        return max(0.0, min(1.0, success_ratio - failure_weight * (1 - success_ratio) * 0.5))

    def get_all_reachable(self, min_confidence: float) -> list[tuple[int, int, float]]:
        """Get all cells with reachability above threshold."""
        results = []

        if self._use_numpy:
            # Vectorized check for any data
            has_data = (self._success_count > 0) | (self._failure_count > 0)
            indices = np.argwhere(has_data)

            for idx in indices:
                grid_u, grid_v = int(idx[0]), int(idx[1])
                reach = self.get_reachability(grid_u, grid_v)
                if reach >= min_confidence:
                    results.append((grid_u, grid_v, reach))
        else:
            for (grid_u, grid_v), data in self._data.items():
                if data[0] > 0 or data[1] > 0:
                    reach = self.get_reachability(grid_u, grid_v)
                    if reach >= min_confidence:
                        results.append((grid_u, grid_v, reach))

        return results


class SpatialMemoryIndex:
    """Spatial index for memories using grid-based hashing.

    Provides O(1) insertion and O(k) neighbor queries where k is
    the number of cells in the search radius.
    """

    def __init__(self, max_per_cell: int, decay_seconds: float) -> None:
        self.max_per_cell = max_per_cell
        self.decay_seconds = decay_seconds

        # Grid cell -> list of memories
        self._cells: dict[tuple[int, int], list[SpatialMemory]] = defaultdict(list)

        # Track ID -> grid cell (for fast lookup by track)
        self._track_index: dict[int, tuple[int, int]] = {}

        self._total_memories = 0

    def add(
        self,
        grid_u: int,
        grid_v: int,
        memory: SpatialMemory,
    ) -> None:
        """Add a memory to the index."""
        cell_memories = self._cells[(grid_u, grid_v)]

        # Remove old memory for same track if exists
        if memory.track_id is not None:
            old_cell = self._track_index.get(memory.track_id)
            if old_cell:
                old_memories = self._cells[old_cell]
                self._cells[old_cell] = [m for m in old_memories if m.track_id != memory.track_id]
                if not self._cells[old_cell]:
                    del self._cells[old_cell]
            self._track_index[memory.track_id] = (grid_u, grid_v)

        cell_memories.append(memory)
        self._total_memories += 1

        # Prune if over limit
        if len(cell_memories) > self.max_per_cell:
            # Remove oldest
            cell_memories.sort(key=lambda m: m.timestamp)
            removed = cell_memories.pop(0)
            self._total_memories -= 1
            if removed.track_id in self._track_index:
                del self._track_index[removed.track_id]

    def get_nearby(
        self,
        grid_u: int,
        grid_v: int,
        radius: int = 1,
        max_age: float | None = None,
    ) -> Iterator[SpatialMemory]:
        """Get memories in nearby cells."""
        now = time.time()
        max_age = max_age or self.decay_seconds

        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                cell_key = (grid_u + du, grid_v + dv)
                if cell_key in self._cells:
                    for mem in self._cells[cell_key]:
                        if now - mem.timestamp <= max_age:
                            yield mem

    def get_by_class(
        self,
        class_ids: set[int],
        max_age: float | None = None,
    ) -> list[tuple[tuple[int, int], SpatialMemory]]:
        """Get memories matching class IDs."""
        now = time.time()
        max_age = max_age or self.decay_seconds
        results = []

        for cell_key, memories in self._cells.items():
            for mem in memories:
                if mem.class_id in class_ids and now - mem.timestamp <= max_age:
                    results.append((cell_key, mem))

        return results

    def cleanup_old(self, max_age: float | None = None) -> int:
        """Remove memories older than max_age. Returns count removed."""
        now = time.time()
        max_age = max_age or self.decay_seconds
        removed = 0

        cells_to_delete = []
        for cell_key, memories in self._cells.items():
            old_len = len(memories)
            self._cells[cell_key] = [m for m in memories if now - m.timestamp <= max_age]
            new_len = len(self._cells[cell_key])
            removed += old_len - new_len

            if new_len == 0:
                cells_to_delete.append(cell_key)

        for cell_key in cells_to_delete:
            del self._cells[cell_key]

        self._total_memories -= removed
        return removed

    @property
    def total_memories(self) -> int:
        return self._total_memories


class SpatialMap:
    """Spatial map tracking reachable positions and memories.

    Uses efficient data structures:
    - ReachabilityGrid: numpy-backed grid for movement tracking
    - SpatialMemoryIndex: Hash-grid index for spatial memories

    Example:
        spatial_map = SpatialMap()

        # Record movement outcome
        spatial_map.record_movement(target=(320, 240), success=True)

        # Record what was seen
        spatial_map.record_observation(
            position=(320, 240),
            detections=[{"class_id": 0, "label": "person", "track_id": 1}],
        )

        # Get a safe exploration target
        target = spatial_map.get_safe_exploration_target()
    """

    def __init__(self, config: SpatialMapConfig | None = None) -> None:
        """Initialize spatial map."""
        self.config = config or SpatialMapConfig()
        self._lock = threading.RLock()  # Reentrant for nested calls

        # Calculate grid size
        self._grid_size = int(1.0 / self.config.grid_resolution)

        # Efficient data structures
        self._reachability = ReachabilityGrid(
            self._grid_size,
            self.config.failure_decay_seconds,
        )
        self._memories = SpatialMemoryIndex(
            self.config.max_memories_per_cell,
            self.config.memory_decay_seconds,
        )

        # Image dimensions
        self._image_width: int = 640
        self._image_height: int = 480

        # Current head position estimate
        self._current_position: tuple[float, float] | None = None
        self._current_grid: tuple[int, int] | None = None

        # Statistics
        self._stats = {
            "movements_recorded": 0,
            "successes": 0,
            "failures": 0,
            "observations_recorded": 0,
        }

    def _pixel_to_grid(self, pixel_u: float, pixel_v: float) -> tuple[int, int]:
        """Convert pixel coordinates to grid cell indices. O(1)."""
        norm_u = max(0.0, min(1.0, pixel_u / self._image_width))
        norm_v = max(0.0, min(1.0, pixel_v / self._image_height))

        grid_u = min(int(norm_u / self.config.grid_resolution), self._grid_size - 1)
        grid_v = min(int(norm_v / self.config.grid_resolution), self._grid_size - 1)

        return (grid_u, grid_v)

    def _grid_to_pixel(self, grid_u: int, grid_v: int) -> tuple[float, float]:
        """Convert grid cell to pixel coordinates (center of cell). O(1)."""
        norm_u = (grid_u + 0.5) * self.config.grid_resolution
        norm_v = (grid_v + 0.5) * self.config.grid_resolution

        return (norm_u * self._image_width, norm_v * self._image_height)

    def _is_in_default_safe_range(self, grid_u: int, grid_v: int) -> bool:
        """Check if grid cell is in default safe range. O(1)."""
        norm_u = (grid_u + 0.5) * self.config.grid_resolution
        norm_v = (grid_v + 0.5) * self.config.grid_resolution
        u_min, v_min, u_max, v_max = self.config.default_reachable_range
        return u_min <= norm_u <= u_max and v_min <= norm_v <= v_max

    def record_movement(
        self,
        target: tuple[float, float],
        success: bool,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        """Record a movement attempt outcome. O(1)."""
        with self._lock:
            if image_size:
                self._image_width, self._image_height = image_size

            grid_key = self._pixel_to_grid(target[0], target[1])
            self._reachability.record(grid_key[0], grid_key[1], success)

            if success:
                self._current_position = target
                self._current_grid = grid_key
                self._stats["successes"] += 1
            else:
                self._stats["failures"] += 1

            self._stats["movements_recorded"] += 1

        if not success:
            logger.debug("Movement to %s failed (grid %s)", target, grid_key)

    def record_observation(
        self,
        position: tuple[float, float],
        detections: list[dict[str, Any]],
        novelty_scores: dict[Any, float] | None = None,
    ) -> None:
        """Record interesting detections at a position. O(d) where d=detections."""
        now = time.time()
        novelty_scores = novelty_scores or {}

        with self._lock:
            grid_key = self._pixel_to_grid(position[0], position[1])

            for det in detections:
                track_id = det.get("track_id")
                novelty = novelty_scores.get(track_id, 0.5)
                conf = det.get("conf", 0.0)

                # Only store interesting detections
                if conf < 0.5 and novelty < 0.3:
                    continue

                memory = SpatialMemory(
                    class_id=det.get("class_id"),
                    label=det.get("label"),
                    track_id=track_id,
                    confidence=conf,
                    timestamp=now,
                    novelty=novelty,
                    pixel_u=position[0],
                    pixel_v=position[1],
                )

                self._memories.add(grid_key[0], grid_key[1], memory)

            self._stats["observations_recorded"] += 1

    def get_reachability(self, target: tuple[float, float]) -> float:
        """Get reachability score for a target position. O(1)."""
        with self._lock:
            grid_key = self._pixel_to_grid(target[0], target[1])
            base_reach = self._reachability.get_reachability(grid_key[0], grid_key[1])

            # If unknown (0.5), check default safe range
            if 0.4 < base_reach < 0.6:
                if self._is_in_default_safe_range(grid_key[0], grid_key[1]):
                    return 0.7  # Probably safe
                else:
                    return 0.3  # Probably not safe

            return base_reach

    def get_safe_exploration_target(
        self,
        avoid_current: bool = True,
        prefer_unexplored: bool = True,
        prefer_interesting: bool = True,
    ) -> tuple[float, float] | None:
        """Get a safe target for idle exploration.

        Uses weighted random selection favoring:
        1. Known reachable positions
        2. Positions with recent interesting memories
        3. Unexplored positions within safe default range

        Returns (u, v) pixel coordinates.
        """
        candidates: list[tuple[int, int, float]] = []

        with self._lock:
            # Get all known reachable positions
            reachable = self._reachability.get_all_reachable(
                self.config.min_reachable_confidence
            )

            for grid_u, grid_v, reach_score in reachable:
                # Skip current position if requested
                if avoid_current and self._current_grid == (grid_u, grid_v):
                    continue

                score = reach_score

                # Boost for interesting memories
                if prefer_interesting:
                    for mem in self._memories.get_nearby(grid_u, grid_v, radius=0):
                        score += mem.novelty * 0.2

                candidates.append((grid_u, grid_v, score))

            # Add unexplored cells in safe range
            if prefer_unexplored:
                explored = {(c[0], c[1]) for c in candidates}
                for grid_u in range(self._grid_size):
                    for grid_v in range(self._grid_size):
                        if (grid_u, grid_v) in explored:
                            continue
                        if avoid_current and self._current_grid == (grid_u, grid_v):
                            continue
                        if self._is_in_default_safe_range(grid_u, grid_v):
                            candidates.append((grid_u, grid_v, 0.5))

            if not candidates:
                # Fallback to center
                return (self._image_width / 2, self._image_height / 2)

            # Weighted random selection
            total_score = sum(c[2] for c in candidates)
            if total_score <= 0:
                selected = random.choice(candidates)
            else:
                r = random.uniform(0, total_score)
                cumulative = 0.0
                selected = candidates[0]
                for c in candidates:
                    cumulative += c[2]
                    if cumulative >= r:
                        selected = c
                        break

            # Add small random offset within cell for variety
            pixel_u, pixel_v = self._grid_to_pixel(selected[0], selected[1])
            cell_size_u = self.config.grid_resolution * self._image_width
            cell_size_v = self.config.grid_resolution * self._image_height
            pixel_u += random.uniform(-cell_size_u * 0.3, cell_size_u * 0.3)
            pixel_v += random.uniform(-cell_size_v * 0.3, cell_size_v * 0.3)

            return (pixel_u, pixel_v)

    def get_interesting_positions(
        self,
        class_ids: set[int] | None = None,
        max_age_seconds: float | None = None,
    ) -> list[tuple[tuple[float, float], SpatialMemory]]:
        """Get positions where interesting things were seen."""
        with self._lock:
            if class_ids:
                results = self._memories.get_by_class(class_ids, max_age_seconds)
                return [
                    (self._grid_to_pixel(cell[0], cell[1]), mem)
                    for cell, mem in results
                ]
            else:
                # Return all recent memories
                results = []
                time.time()
                max_age = max_age_seconds or self.config.memory_decay_seconds

                for grid_u in range(self._grid_size):
                    for grid_v in range(self._grid_size):
                        for mem in self._memories.get_nearby(grid_u, grid_v, radius=0, max_age=max_age):
                            results.append((self._grid_to_pixel(grid_u, grid_v), mem))

                return results

    def get_nearby_memories(
        self,
        position: tuple[float, float],
        radius_cells: int | None = None,
    ) -> list[SpatialMemory]:
        """Get memories near a position."""
        with self._lock:
            grid_key = self._pixel_to_grid(position[0], position[1])
            radius = radius_cells or self.config.neighbor_search_radius
            return list(self._memories.get_nearby(grid_key[0], grid_key[1], radius))

    def update_image_size(self, width: int, height: int) -> None:
        """Update the expected image dimensions."""
        with self._lock:
            self._image_width = width
            self._image_height = height

    def cleanup(self) -> int:
        """Remove old memories. Returns count removed."""
        with self._lock:
            return self._memories.cleanup_old()

    def get_stats(self) -> dict[str, Any]:
        """Get spatial map statistics."""
        with self._lock:
            return {
                **self._stats,
                "total_memories": self._memories.total_memories,
                "grid_size": self._grid_size,
                "image_size": (self._image_width, self._image_height),
                "current_position": self._current_position,
                "using_numpy": HAS_NUMPY,
            }

    def clear(self) -> None:
        """Clear all spatial data."""
        with self._lock:
            self._reachability = ReachabilityGrid(
                self._grid_size,
                self.config.failure_decay_seconds,
            )
            self._memories = SpatialMemoryIndex(
                self.config.max_memories_per_cell,
                self.config.memory_decay_seconds,
            )
            self._current_position = None
            self._current_grid = None


__all__ = [
    "SpatialMap",
    "SpatialMapConfig",
    "SpatialMemory",
]
