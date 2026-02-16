"""Unified Salience Map - Combines all attention factors into spatial probability.

This module provides a unified view of WHERE attention should go by combining:
- Object salience (novelty, movement, interest)
- Social salience (face/person presence)
- Spatial salience (unexplored areas, inhibition of return)
- Center bias (tendency to return to neutral)

The map can be sampled probabilistically for natural gaze variety,
or queried for the peak location for reactive attention.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from maxim.attention import GazeHistory
    from maxim.salience import MovementDetector, ThreadSafeNoveltyTracker

logger = logging.getLogger(__name__)

# COCO class ID for person
PERSON_CLASS_ID = 0


@dataclass
class SalienceMapConfig:
    """Configuration for the unified salience map.

    Attributes:
        grid_width: Number of horizontal cells.
        grid_height: Number of vertical cells.
        frame_width: Image width in pixels.
        frame_height: Image height in pixels.

        novelty_weight: Weight for object novelty (0-1).
        social_weight: Weight for face/person presence (0-1).
        movement_weight: Weight for object motion (0-1).
        unexplored_weight: Weight for unvisited areas (0-1).
        center_bias_weight: Weight for center preference (0-1).

        inhibition_decay_seconds: Time for visited locations to recover salience.
        social_radius_cells: How many cells a person "covers" in the map.
        min_salience_floor: Minimum salience for any cell (exploration baseline).
        temperature_default: Default sampling temperature (higher = more random).
    """
    grid_width: int = 16
    grid_height: int = 12
    frame_width: float = 640.0
    frame_height: float = 480.0

    # Factor weights (should roughly sum to 1.0 for interpretability)
    novelty_weight: float = 0.25
    social_weight: float = 0.35
    movement_weight: float = 0.15
    unexplored_weight: float = 0.15
    center_bias_weight: float = 0.10

    # Temporal dynamics
    inhibition_decay_seconds: float = 1.5

    # Spatial parameters
    social_radius_cells: int = 2  # Person influence spreads this many cells
    min_salience_floor: float = 0.1  # Everything has at least this salience

    # Sampling
    temperature_default: float = 1.0

    # Tracking hysteresis - centralized spatial persistence
    # Helps maintain focus when track_id fragments during head movement
    tracking_hysteresis_radius: float = 200.0  # pixels - bonus within this radius
    tracking_hysteresis_bonus: float = 0.25  # salience bonus for nearby detections


@dataclass
class SalienceCell:
    """Salience state for a single grid cell."""
    # Component contributions (before weighting)
    novelty: float = 0.0
    social: float = 0.0
    movement: float = 0.0
    unexplored: float = 1.0  # Start as unexplored
    center_bias: float = 0.0

    # Temporal state
    last_visited: float = 0.0
    visit_count: int = 0

    # Computed total
    total_salience: float = 0.0


class SalienceMap:
    """Unified spatial salience combining all attention factors.

    The map divides the visual field into a grid and computes per-cell
    salience from multiple factors. This enables:

    1. **Probabilistic sampling**: Natural variety in gaze targets
    2. **Peak detection**: Reactive attention to most salient location
    3. **Exploration guidance**: Preference for unvisited areas
    4. **Inhibition of return**: Recently visited areas are less salient

    Example:
        salience_map = SalienceMap(config)

        # Each frame, update with current perception
        salience_map.update(
            detections=detections,
            gaze_history=gaze_history,
            novelty_tracker=novelty_tracker,
            movement_detector=movement_detector,
        )

        # Get most salient location (reactive)
        peak = salience_map.get_peak_target()

        # Or sample probabilistically (exploratory)
        target = salience_map.sample_target(temperature=0.8)
    """

    def __init__(self, config: SalienceMapConfig | None = None) -> None:
        """Initialize the salience map.

        Args:
            config: Map configuration.
        """
        self.config = config or SalienceMapConfig()

        # Initialize grid
        self._grid: list[list[SalienceCell]] = [
            [SalienceCell() for _ in range(self.config.grid_width)]
            for _ in range(self.config.grid_height)
        ]

        # Precompute center bias for each cell
        self._precompute_center_bias()

        # Cache for current salience values (as numpy array for sampling)
        self._salience_array: np.ndarray = np.zeros(
            (self.config.grid_height, self.config.grid_width)
        )

        # Track current gaze for inhibition
        self._current_gaze_cell: tuple[int, int] | None = None

        # Centralized tracking hysteresis - shared across all behaviors
        # Stores the last actively tracked position to help maintain focus
        # when YOLO track_id fragments during head movement
        self._last_tracked_pos: tuple[float, float] | None = None
        self._last_tracked_time: float = 0.0

    def _precompute_center_bias(self) -> None:
        """Precompute center bias values for all cells."""
        cx = self.config.grid_width / 2
        cy = self.config.grid_height / 2
        max_dist = math.sqrt(cx * cx + cy * cy)

        for row in range(self.config.grid_height):
            for col in range(self.config.grid_width):
                # Distance from center (normalized 0-1)
                dx = (col + 0.5 - cx) / cx
                dy = (row + 0.5 - cy) / cy
                dist = math.sqrt(dx * dx + dy * dy)

                # Center bias: high at center, low at edges
                # Use Gaussian-like falloff
                bias = math.exp(-dist * dist * 2)
                self._grid[row][col].center_bias = bias

    def _pixel_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Convert pixel coordinates to grid cell indices.

        Args:
            x: X pixel coordinate.
            y: Y pixel coordinate.

        Returns:
            (row, col) grid indices.
        """
        col = int(x / self.config.frame_width * self.config.grid_width)
        row = int(y / self.config.frame_height * self.config.grid_height)

        # Clamp to valid range
        col = max(0, min(self.config.grid_width - 1, col))
        row = max(0, min(self.config.grid_height - 1, row))

        return (row, col)

    def _cell_to_pixel(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid cell to pixel coordinates (center of cell).

        Args:
            row: Grid row index.
            col: Grid column index.

        Returns:
            (x, y) pixel coordinates.
        """
        x = (col + 0.5) / self.config.grid_width * self.config.frame_width
        y = (row + 0.5) / self.config.grid_height * self.config.frame_height
        return (x, y)

    def update(
        self,
        detections: list[dict[str, Any]],
        gaze_history: "GazeHistory | None" = None,
        novelty_tracker: "ThreadSafeNoveltyTracker | None" = None,
        movement_detector: "MovementDetector | None" = None,
        current_time: float | None = None,
    ) -> None:
        """Update salience map with current perception.

        Args:
            detections: List of detection dicts with bbox, track_id, class_id.
            gaze_history: GazeHistory for inhibition of return.
            novelty_tracker: NoveltyTracker for object novelty scores.
            movement_detector: MovementDetector for motion scores.
            current_time: Current timestamp (defaults to time.time()).
        """
        now = current_time or time.time()

        # Reset per-frame components
        for row in self._grid:
            for cell in row:
                cell.novelty = 0.0
                cell.social = 0.0
                cell.movement = 0.0

        # Update current gaze position
        if gaze_history is not None:
            current_pos = gaze_history.get_current_position()
            if current_pos is not None:
                self._current_gaze_cell = self._pixel_to_cell(current_pos[0], current_pos[1])
                # Mark as visited
                row, col = self._current_gaze_cell
                self._grid[row][col].last_visited = now
                self._grid[row][col].visit_count += 1

        # Process detections
        for det in detections:
            bbox = det.get("bbox_xyxy") or det.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue

            # Get center of detection
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            row, col = self._pixel_to_cell(cx, cy)

            track_id = det.get("track_id")
            class_id = det.get("class_id")

            # Get novelty score (with class-level modulation)
            novelty = 0.5  # Default
            if novelty_tracker is not None and track_id is not None:
                novelty = novelty_tracker.get_novelty(track_id, class_id=class_id)
                # Normalize to 0-1 range (novelty is typically 0.5-2.0)
                novelty = (novelty - 0.5) / 1.5

            # Get movement score
            movement = det.get("movement_score", 0.0)

            # Check if person (social salience)
            is_person = class_id == PERSON_CLASS_ID

            # Apply to cell and neighbors
            self._apply_detection_to_grid(
                row, col,
                novelty=novelty,
                movement=movement,
                is_person=is_person,
            )

        # Compute unexplored salience based on visit recency
        self._update_unexplored_salience(now)

        # Compute total salience for each cell
        self._compute_total_salience()

    def _apply_detection_to_grid(
        self,
        center_row: int,
        center_col: int,
        novelty: float,
        movement: float,
        is_person: bool,
    ) -> None:
        """Apply detection salience to grid cells with spatial spread.

        Args:
            center_row: Center row of detection.
            center_col: Center column of detection.
            novelty: Novelty score (0-1).
            movement: Movement score (0-1).
            is_person: Whether this is a person detection.
        """
        # For social salience, spread across multiple cells
        radius = self.config.social_radius_cells if is_person else 1

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                row = center_row + dr
                col = center_col + dc

                if 0 <= row < self.config.grid_height and 0 <= col < self.config.grid_width:
                    # Distance-based falloff
                    dist = math.sqrt(dr * dr + dc * dc)
                    if radius > 0:
                        falloff = max(0, 1 - dist / (radius + 1))
                    else:
                        falloff = 1.0

                    cell = self._grid[row][col]

                    # Take max (don't accumulate to avoid overflow)
                    cell.novelty = max(cell.novelty, novelty * falloff)
                    cell.movement = max(cell.movement, movement * falloff)

                    if is_person:
                        cell.social = max(cell.social, falloff)

    def _update_unexplored_salience(self, now: float) -> None:
        """Update unexplored salience based on visit recency.

        Cells that haven't been visited recently have high unexplored salience.
        Recently visited cells have low unexplored salience (inhibition of return).

        Args:
            now: Current timestamp.
        """
        decay = self.config.inhibition_decay_seconds

        for row in self._grid:
            for cell in row:
                if cell.last_visited == 0:
                    # Never visited - fully unexplored
                    cell.unexplored = 1.0
                else:
                    # Recover based on time since visit
                    time_since = now - cell.last_visited
                    # Exponential recovery
                    recovery = 1 - math.exp(-time_since / decay)
                    cell.unexplored = recovery

    def _compute_total_salience(self) -> None:
        """Compute weighted total salience for each cell."""
        cfg = self.config

        for row_idx, row in enumerate(self._grid):
            for col_idx, cell in enumerate(row):
                # Weighted sum of components
                total = (
                    cell.novelty * cfg.novelty_weight +
                    cell.social * cfg.social_weight +
                    cell.movement * cfg.movement_weight +
                    cell.unexplored * cfg.unexplored_weight +
                    cell.center_bias * cfg.center_bias_weight
                )

                # Apply floor
                total = max(cfg.min_salience_floor, total)

                cell.total_salience = total
                self._salience_array[row_idx, col_idx] = total

    def get_peak_target(self) -> tuple[float, float]:
        """Get the most salient location (winner-take-all).

        Returns:
            (x, y) pixel coordinates of peak salience.
        """
        # Find max cell
        max_salience = -1.0
        best_row, best_col = 0, 0

        for row_idx, row in enumerate(self._grid):
            for col_idx, cell in enumerate(row):
                if cell.total_salience > max_salience:
                    max_salience = cell.total_salience
                    best_row = row_idx
                    best_col = col_idx

        return self._cell_to_pixel(best_row, best_col)

    def sample_target(
        self,
        temperature: float | None = None,
        avoid_current: bool = True,
    ) -> tuple[float, float]:
        """Probabilistically sample a gaze target from salience distribution.

        Higher temperature = more random (uniform distribution).
        Lower temperature = more deterministic (peaks dominate).

        Args:
            temperature: Sampling temperature (default from config).
            avoid_current: If True, reduce probability of current gaze location.

        Returns:
            (x, y) pixel coordinates of sampled target.
        """
        if temperature is None:
            temperature = self.config.temperature_default

        # Create probability distribution
        probs = self._salience_array.copy()

        # Reduce probability of current gaze location
        if avoid_current and self._current_gaze_cell is not None:
            row, col = self._current_gaze_cell
            probs[row, col] *= 0.1  # 90% reduction

        # Apply temperature
        if temperature != 1.0:
            # Temperature scaling: p = p^(1/T)
            # High T -> uniform, Low T -> peaky
            probs = np.power(probs, 1.0 / temperature)

        # Normalize to probability distribution
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            # Fallback to uniform
            probs = np.ones_like(probs) / probs.size

        # Flatten and sample
        flat_probs = probs.flatten()
        idx = np.random.choice(len(flat_probs), p=flat_probs)

        # Convert back to grid coordinates
        row = idx // self.config.grid_width
        col = idx % self.config.grid_width

        return self._cell_to_pixel(row, col)

    def get_salience_at(self, position: tuple[float, float]) -> float:
        """Get salience value at a specific position.

        Args:
            position: (x, y) pixel coordinates.

        Returns:
            Salience value at that position.
        """
        row, col = self._pixel_to_cell(position[0], position[1])
        return self._grid[row][col].total_salience

    def get_cell_info(self, position: tuple[float, float]) -> dict[str, float]:
        """Get detailed salience breakdown for a position.

        Args:
            position: (x, y) pixel coordinates.

        Returns:
            Dict with component salience values.
        """
        row, col = self._pixel_to_cell(position[0], position[1])
        cell = self._grid[row][col]

        return {
            "novelty": cell.novelty,
            "social": cell.social,
            "movement": cell.movement,
            "unexplored": cell.unexplored,
            "center_bias": cell.center_bias,
            "total": cell.total_salience,
            "visit_count": cell.visit_count,
        }

    def get_heatmap(self) -> np.ndarray:
        """Get salience as a 2D numpy array for visualization.

        Returns:
            Array of shape (grid_height, grid_width) with salience values.
        """
        return self._salience_array.copy()

    def get_top_n_targets(self, n: int = 5) -> list[tuple[tuple[float, float], float]]:
        """Get the N most salient locations.

        Args:
            n: Number of targets to return.

        Returns:
            List of ((x, y), salience) tuples, sorted by salience descending.
        """
        # Collect all cells with salience
        cells = []
        for row_idx, row in enumerate(self._grid):
            for col_idx, cell in enumerate(row):
                pos = self._cell_to_pixel(row_idx, col_idx)
                cells.append((pos, cell.total_salience))

        # Sort by salience descending
        cells.sort(key=lambda x: x[1], reverse=True)

        return cells[:n]

    # -------------------------------------------------------------------------
    # Tracking Hysteresis - Centralized spatial persistence for all behaviors
    # -------------------------------------------------------------------------

    def record_tracking_target(self, position: tuple[float, float]) -> None:
        """Record that we're actively tracking a target at this position.

        This should be called when a behavior commits to tracking a detection.
        The position is used to provide spatial bonus to nearby detections,
        helping maintain focus when YOLO track_id fragments during head movement.

        Args:
            position: (x, y) pixel coordinates of the tracked target.
        """
        self._last_tracked_pos = position
        self._last_tracked_time = time.time()

    def get_tracking_bonus(self, position: tuple[float, float]) -> float:
        """Get spatial tracking bonus for a position.

        Returns a bonus value if the position is near the last tracked target.
        This helps maintain focus on the same object even when track_id changes.

        Args:
            position: (x, y) pixel coordinates to check.

        Returns:
            Bonus value (0.0 if not near, config.tracking_hysteresis_bonus if near).
        """
        if self._last_tracked_pos is None:
            return 0.0

        dx = position[0] - self._last_tracked_pos[0]
        dy = position[1] - self._last_tracked_pos[1]
        dist = (dx * dx + dy * dy) ** 0.5

        if dist < self.config.tracking_hysteresis_radius:
            return self.config.tracking_hysteresis_bonus
        return 0.0

    def get_last_tracked_position(self) -> tuple[float, float] | None:
        """Get the last actively tracked position.

        Returns:
            (x, y) pixel coordinates, or None if no recent tracking.
        """
        return self._last_tracked_pos

    def clear_tracking(self) -> None:
        """Clear the tracking hysteresis state."""
        self._last_tracked_pos = None
        self._last_tracked_time = 0.0

    def reset(self) -> None:
        """Reset all salience state."""
        for row in self._grid:
            for cell in row:
                cell.novelty = 0.0
                cell.social = 0.0
                cell.movement = 0.0
                cell.unexplored = 1.0
                cell.last_visited = 0.0
                cell.visit_count = 0
                cell.total_salience = 0.0

        self._salience_array.fill(0.0)
        self._current_gaze_cell = None
        self._last_tracked_pos = None
        self._last_tracked_time = 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get map statistics.

        Returns:
            Dict with statistics.
        """
        flat = self._salience_array.flatten()
        visited_count = sum(
            1 for row in self._grid for cell in row if cell.visit_count > 0
        )

        return {
            "grid_size": (self.config.grid_height, self.config.grid_width),
            "mean_salience": float(np.mean(flat)),
            "max_salience": float(np.max(flat)),
            "min_salience": float(np.min(flat)),
            "visited_cells": visited_count,
            "total_cells": self.config.grid_height * self.config.grid_width,
            "coverage": visited_count / (self.config.grid_height * self.config.grid_width),
        }

    def to_context_str(self, max_targets: int = 3) -> str:
        """Generate context string for LLM consumption.

        Args:
            max_targets: Number of top targets to include.

        Returns:
            Human-readable summary of salience state.
        """
        lines = ["[SALIENCE MAP]"]

        stats = self.get_stats()
        lines.append(f"Coverage: {stats['coverage']*100:.0f}% explored")
        lines.append(f"Salience range: {stats['min_salience']:.2f} - {stats['max_salience']:.2f}")

        # Top targets
        top = self.get_top_n_targets(max_targets)
        if top:
            lines.append("Top attention targets:")
            for i, (pos, sal) in enumerate(top, 1):
                lines.append(f"  {i}. ({pos[0]:.0f}, {pos[1]:.0f}) salience={sal:.2f}")

        return "\n".join(lines)


__all__ = [
    "SalienceMapConfig",
    "SalienceCell",
    "SalienceMap",
]
