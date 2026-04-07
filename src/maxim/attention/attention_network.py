"""AttentionNetwork - Manages WHERE the robot attends in visual space.

A DataFrame-backed attention system that tracks:
- Spatial attention grid (where we can/have looked)
- Current focus position
- Dwell time and visit history
- Reachability constraints

Designed for efficient queries by agents/LLMs via to_context_str().
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Use numpy for efficient array operations
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for AttentionNetwork.

    Attributes:
        grid_size: Number of cells per dimension (grid_size x grid_size).
        image_width: Expected image width in pixels.
        image_height: Expected image height in pixels.
        dwell_decay_seconds: Time for dwell/visit salience to decay.
        reachability_decay_seconds: Time for failure penalties to decay.
        min_dwell_seconds: Minimum time to stay at a position.
        default_safe_range: (u_min, v_min, u_max, v_max) in normalized coords.
    """

    grid_size: int = 10
    image_width: int = 640
    image_height: int = 480
    dwell_decay_seconds: float = 2.0
    reachability_decay_seconds: float = 60.0
    min_dwell_seconds: float = 0.5
    default_safe_range: tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.75)


@dataclass
class AttentionCell:
    """A single cell in the attention grid (for non-numpy fallback)."""

    grid_u: int
    grid_v: int
    visit_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_visit: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    total_dwell: float = 0.0


class AttentionNetwork:
    """DataFrame-backed attention system for spatial attention management.

    Maintains a grid of attention cells tracking:
    - Visit history (where we've looked)
    - Reachability (where we can look)
    - Dwell time (how long we've looked)

    Example:
        attention = AttentionNetwork()

        # Record a gaze
        attention.record_gaze((320, 240), success=True)

        # Get next target
        target = attention.get_next_target()

        # Get context for LLM
        context = attention.to_context_str()
    """

    def __init__(self, config: AttentionConfig | None = None) -> None:
        """Initialize AttentionNetwork."""
        self.config = config or AttentionConfig()
        self._lock = threading.RLock()

        n = self.config.grid_size

        if HAS_NUMPY:
            # DataFrame-like numpy arrays (column-major storage)
            # Each array is a "column" in our attention DataFrame
            self._grid_u = np.repeat(np.arange(n), n).astype(np.int32)
            self._grid_v = np.tile(np.arange(n), n).astype(np.int32)
            self._visit_count = np.zeros(n * n, dtype=np.int32)
            self._success_count = np.zeros(n * n, dtype=np.int32)
            self._failure_count = np.zeros(n * n, dtype=np.int32)
            self._last_visit = np.zeros(n * n, dtype=np.float64)
            self._last_success = np.zeros(n * n, dtype=np.float64)
            self._last_failure = np.zeros(n * n, dtype=np.float64)
            self._total_dwell = np.zeros(n * n, dtype=np.float64)
        else:
            # Fallback to dict-based storage
            self._cells: dict[tuple[int, int], AttentionCell] = {}

        # Current focus state
        self._current_focus: tuple[float, float] | None = None
        self._current_grid: tuple[int, int] | None = None
        self._focus_start_time: float = 0.0
        self._last_move_time: float = 0.0

        # Statistics
        self._total_visits = 0
        self._total_successes = 0
        self._total_failures = 0

    def _idx(self, grid_u: int, grid_v: int) -> int:
        """Convert grid coords to flat array index."""
        return grid_u * self.config.grid_size + grid_v

    def _pixel_to_grid(self, pixel_u: float, pixel_v: float) -> tuple[int, int]:
        """Convert pixel coordinates to grid cell. O(1)."""
        norm_u = max(0.0, min(1.0, pixel_u / self.config.image_width))
        norm_v = max(0.0, min(1.0, pixel_v / self.config.image_height))

        res = 1.0 / self.config.grid_size
        grid_u = min(int(norm_u / res), self.config.grid_size - 1)
        grid_v = min(int(norm_v / res), self.config.grid_size - 1)

        return (grid_u, grid_v)

    def _grid_to_pixel(self, grid_u: int, grid_v: int) -> tuple[float, float]:
        """Convert grid cell to pixel coordinates (center). O(1)."""
        res = 1.0 / self.config.grid_size
        norm_u = (grid_u + 0.5) * res
        norm_v = (grid_v + 0.5) * res

        return (norm_u * self.config.image_width, norm_v * self.config.image_height)

    def _is_in_safe_range(self, grid_u: int, grid_v: int) -> bool:
        """Check if grid cell is in default safe range."""
        res = 1.0 / self.config.grid_size
        norm_u = (grid_u + 0.5) * res
        norm_v = (grid_v + 0.5) * res
        u_min, v_min, u_max, v_max = self.config.default_safe_range
        return u_min <= norm_u <= u_max and v_min <= norm_v <= v_max

    def record_gaze(
        self,
        position: tuple[float, float],
        success: bool = True,
        duration: float = 0.0,
    ) -> None:
        """Record a gaze/movement to a position.

        Args:
            position: (u, v) pixel coordinates.
            success: Whether the movement succeeded (False = IK failure).
            duration: How long we dwelled at this position.
        """
        now = time.time()

        with self._lock:
            grid_u, grid_v = self._pixel_to_grid(position[0], position[1])
            idx = self._idx(grid_u, grid_v)

            if HAS_NUMPY:
                self._visit_count[idx] += 1
                self._last_visit[idx] = now

                if success:
                    self._success_count[idx] += 1
                    self._last_success[idx] = now
                    self._total_dwell[idx] += duration
                    self._current_focus = position
                    self._current_grid = (grid_u, grid_v)
                    self._focus_start_time = now
                    self._total_successes += 1
                else:
                    self._failure_count[idx] += 1
                    self._last_failure[idx] = now
                    self._total_failures += 1
            else:
                key = (grid_u, grid_v)
                if key not in self._cells:
                    self._cells[key] = AttentionCell(grid_u=grid_u, grid_v=grid_v)

                cell = self._cells[key]
                cell.visit_count += 1
                cell.last_visit = now

                if success:
                    cell.success_count += 1
                    cell.last_success = now
                    cell.total_dwell += duration
                    self._current_focus = position
                    self._current_grid = (grid_u, grid_v)
                    self._focus_start_time = now
                    self._total_successes += 1
                else:
                    cell.failure_count += 1
                    cell.last_failure = now
                    self._total_failures += 1

            self._total_visits += 1
            self._last_move_time = now

    def get_reachability(self, position: tuple[float, float]) -> float:
        """Get reachability score (0=unreachable, 1=reachable) for a position."""
        now = time.time()

        with self._lock:
            grid_u, grid_v = self._pixel_to_grid(position[0], position[1])
            idx = self._idx(grid_u, grid_v)

            if HAS_NUMPY:
                success = int(self._success_count[idx])
                failure = int(self._failure_count[idx])
                last_failure = float(self._last_failure[idx])
            else:
                key = (grid_u, grid_v)
                if key not in self._cells:
                    # Unknown - check safe range
                    return 0.7 if self._is_in_safe_range(grid_u, grid_v) else 0.3
                cell = self._cells[key]
                success = cell.success_count
                failure = cell.failure_count
                last_failure = cell.last_failure

            if success == 0 and failure == 0:
                return 0.7 if self._is_in_safe_range(grid_u, grid_v) else 0.3

            # Decay failure influence
            time_since_failure = now - last_failure if last_failure > 0 else 1000.0
            failure_weight = math.exp(-time_since_failure / self.config.reachability_decay_seconds)

            total = success + failure
            success_ratio = success / total if total > 0 else 0.5

            return max(0.0, min(1.0, success_ratio - failure_weight * (1 - success_ratio) * 0.5))

    def get_visit_salience(self, position: tuple[float, float]) -> float:
        """Get salience based on visit history (0=recently visited, 1=not visited).

        Implements inhibition-of-return - recently visited positions have low salience.
        """
        now = time.time()

        with self._lock:
            grid_u, grid_v = self._pixel_to_grid(position[0], position[1])
            idx = self._idx(grid_u, grid_v)

            if HAS_NUMPY:
                last_visit = float(self._last_visit[idx])
            else:
                key = (grid_u, grid_v)
                if key not in self._cells:
                    return 1.0  # Never visited = fully salient
                last_visit = self._cells[key].last_visit

            if last_visit == 0:
                return 1.0

            # Exponential recovery
            time_since = now - last_visit
            return 1.0 - math.exp(-time_since / self.config.dwell_decay_seconds)

    def should_move(self, target: tuple[float, float]) -> tuple[bool, str]:
        """Check if we should move to a target position.

        Returns:
            (should_move, reason) tuple.
        """
        now = time.time()

        with self._lock:
            # Check dwell time
            time_since_move = now - self._last_move_time
            if time_since_move < self.config.min_dwell_seconds:
                return False, f"dwell_time ({self.config.min_dwell_seconds - time_since_move:.2f}s remaining)"

            # Check reachability
            reach = self.get_reachability(target)
            if reach < 0.3:
                return False, f"low_reachability ({reach:.2f})"

            # Check visit salience
            salience = self.get_visit_salience(target)
            if salience < 0.3:
                return False, f"recently_visited ({salience:.2f})"

            return True, "ok"

    def get_next_target(
        self,
        avoid_current: bool = True,
        prefer_unexplored: bool = True,
    ) -> tuple[float, float]:
        """Get next attention target using weighted selection.

        Prioritizes:
        1. Reachable positions
        2. Positions with high visit salience (not recently visited)
        3. Unexplored positions in safe range
        """
        now = time.time()
        n = self.config.grid_size

        with self._lock:
            candidates: list[tuple[int, int, float]] = []

            for grid_u in range(n):
                for grid_v in range(n):
                    if avoid_current and self._current_grid == (grid_u, grid_v):
                        continue

                    idx = self._idx(grid_u, grid_v)

                    # Get reachability
                    if HAS_NUMPY:
                        success = int(self._success_count[idx])
                        failure = int(self._failure_count[idx])
                        last_visit = float(self._last_visit[idx])
                        last_failure = float(self._last_failure[idx])
                    else:
                        key = (grid_u, grid_v)
                        if key in self._cells:
                            cell = self._cells[key]
                            success = cell.success_count
                            failure = cell.failure_count
                            last_visit = cell.last_visit
                            last_failure = cell.last_failure
                        else:
                            success = failure = 0
                            last_visit = last_failure = 0.0

                    # Calculate reachability
                    if success == 0 and failure == 0:
                        reach = 0.7 if self._is_in_safe_range(grid_u, grid_v) else 0.3
                    else:
                        time_since_failure = now - last_failure if last_failure > 0 else 1000.0
                        failure_weight = math.exp(-time_since_failure / self.config.reachability_decay_seconds)
                        total = success + failure
                        success_ratio = success / total
                        reach = max(0.0, success_ratio - failure_weight * (1 - success_ratio) * 0.5)

                    if reach < 0.3:
                        continue

                    # Calculate visit salience
                    if last_visit == 0:
                        salience = 1.0
                    else:
                        time_since = now - last_visit
                        salience = 1.0 - math.exp(-time_since / self.config.dwell_decay_seconds)

                    # Combined score
                    score = reach * salience

                    if prefer_unexplored and success == 0 and failure == 0:
                        score += 0.3  # Bonus for unexplored

                    if score > 0.1:
                        candidates.append((grid_u, grid_v, score))

            if not candidates:
                # Fallback to center
                return (self.config.image_width / 2, self.config.image_height / 2)

            # Weighted random selection
            total_score = sum(c[2] for c in candidates)
            r = random.uniform(0, total_score)
            cumulative = 0.0
            selected = candidates[0]

            for c in candidates:
                cumulative += c[2]
                if cumulative >= r:
                    selected = c
                    break

            # Add jitter within cell
            pixel_u, pixel_v = self._grid_to_pixel(selected[0], selected[1])
            cell_size = self.config.image_width / self.config.grid_size
            pixel_u += random.uniform(-cell_size * 0.3, cell_size * 0.3)
            pixel_v += random.uniform(-cell_size * 0.3, cell_size * 0.3)

            return (pixel_u, pixel_v)

    def get_current_focus(self) -> tuple[float, float] | None:
        """Get current focus position."""
        with self._lock:
            return self._current_focus

    def get_dwell_time(self) -> float:
        """Get time spent at current focus position."""
        with self._lock:
            if self._focus_start_time == 0:
                return 0.0
            return time.time() - self._focus_start_time

    # ─────────────────────────────────────────────────────────────────────────
    # DataFrame-like query interface
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        min_reachability: float = 0.0,
        min_salience: float = 0.0,
        in_safe_range: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Query attention grid with filters.

        Args:
            min_reachability: Minimum reachability score.
            min_salience: Minimum visit salience.
            in_safe_range: If True/False, filter by safe range.

        Returns:
            List of dicts with cell data matching filters.
        """
        now = time.time()
        results = []
        n = self.config.grid_size

        with self._lock:
            for grid_u in range(n):
                for grid_v in range(n):
                    idx = self._idx(grid_u, grid_v)

                    if HAS_NUMPY:
                        visit_count = int(self._visit_count[idx])
                        success = int(self._success_count[idx])
                        failure = int(self._failure_count[idx])
                        last_visit = float(self._last_visit[idx])
                        last_failure = float(self._last_failure[idx])
                        total_dwell = float(self._total_dwell[idx])
                    else:
                        key = (grid_u, grid_v)
                        if key not in self._cells:
                            visit_count = success = failure = 0
                            last_visit = last_failure = total_dwell = 0.0
                        else:
                            cell = self._cells[key]
                            visit_count = cell.visit_count
                            success = cell.success_count
                            failure = cell.failure_count
                            last_visit = cell.last_visit
                            last_failure = cell.last_failure
                            total_dwell = cell.total_dwell

                    # Calculate metrics
                    if success == 0 and failure == 0:
                        reach = 0.7 if self._is_in_safe_range(grid_u, grid_v) else 0.3
                    else:
                        time_since_failure = now - last_failure if last_failure > 0 else 1000.0
                        failure_weight = math.exp(-time_since_failure / self.config.reachability_decay_seconds)
                        total = success + failure
                        success_ratio = success / total
                        reach = max(0.0, success_ratio - failure_weight * (1 - success_ratio) * 0.5)

                    if last_visit == 0:
                        salience = 1.0
                    else:
                        time_since = now - last_visit
                        salience = 1.0 - math.exp(-time_since / self.config.dwell_decay_seconds)

                    is_safe = self._is_in_safe_range(grid_u, grid_v)

                    # Apply filters
                    if reach < min_reachability:
                        continue
                    if salience < min_salience:
                        continue
                    if in_safe_range is not None and is_safe != in_safe_range:
                        continue

                    pixel_u, pixel_v = self._grid_to_pixel(grid_u, grid_v)

                    results.append(
                        {
                            "grid_u": grid_u,
                            "grid_v": grid_v,
                            "pixel_u": pixel_u,
                            "pixel_v": pixel_v,
                            "reachability": reach,
                            "salience": salience,
                            "visit_count": visit_count,
                            "success_count": success,
                            "failure_count": failure,
                            "total_dwell": total_dwell,
                            "in_safe_range": is_safe,
                            "is_current": self._current_grid == (grid_u, grid_v),
                        }
                    )

            return results

    def to_context_str(self, max_items: int = 5) -> str:
        """Generate concise context string for LLM consumption.

        Returns a human-readable summary of attention state.
        """
        with self._lock:
            lines = ["[ATTENTION STATE]"]

            # Current focus
            if self._current_focus:
                dwell = self.get_dwell_time()
                lines.append(f"Focus: ({self._current_focus[0]:.0f}, {self._current_focus[1]:.0f}) for {dwell:.1f}s")
            else:
                lines.append("Focus: None")

            # Top reachable positions
            reachable = self.query(min_reachability=0.5, min_salience=0.3)
            reachable.sort(key=lambda x: x["reachability"] * x["salience"], reverse=True)

            if reachable:
                lines.append(f"Available targets: {len(reachable)}")
                for item in reachable[:max_items]:
                    marker = "*" if item["is_current"] else " "
                    lines.append(
                        f" {marker}({item['pixel_u']:.0f},{item['pixel_v']:.0f}) "
                        f"reach={item['reachability']:.2f} sal={item['salience']:.2f}"
                    )

            # Unreachable positions
            unreachable = self.query(min_reachability=0.0)
            unreachable = [x for x in unreachable if x["reachability"] < 0.3 and x["failure_count"] > 0]
            if unreachable:
                lines.append(f"Blocked positions: {len(unreachable)}")

            # Stats
            lines.append(f"Stats: {self._total_visits} visits, {self._total_failures} failures")

            return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """Get network statistics."""
        with self._lock:
            return {
                "total_visits": self._total_visits,
                "total_successes": self._total_successes,
                "total_failures": self._total_failures,
                "current_focus": self._current_focus,
                "current_grid": self._current_grid,
                "dwell_time": self.get_dwell_time(),
                "grid_size": self.config.grid_size,
            }

    def clear(self) -> None:
        """Clear all attention data."""
        with self._lock:
            if HAS_NUMPY:
                self._visit_count.fill(0)
                self._success_count.fill(0)
                self._failure_count.fill(0)
                self._last_visit.fill(0)
                self._last_success.fill(0)
                self._last_failure.fill(0)
                self._total_dwell.fill(0)
            else:
                self._cells.clear()

            self._current_focus = None
            self._current_grid = None
            self._focus_start_time = 0.0
            self._last_move_time = 0.0
            self._total_visits = 0
            self._total_successes = 0
            self._total_failures = 0


__all__ = [
    "AttentionNetwork",
    "AttentionConfig",
]
