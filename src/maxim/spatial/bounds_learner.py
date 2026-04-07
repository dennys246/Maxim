"""WorkspaceBoundsLearner - Learns actual workspace limits from commanded vs actual position comparison.

Tracks 6-DOF joint/translation limits:
- yaw: Head rotation left/right (degrees)
- pitch: Head rotation up/down (degrees)
- y: Translation left/right (mm)
- z: Translation up/down (mm)
- roll: Head tilt (degrees)
- x: Translation forward/back (mm)

Learning strategy:
1. Bootstrap with conservative hardcoded limits
2. Track commanded vs actual for each movement
3. If actual < commanded at boundary, tighten the bound
4. If successful at boundary, cautiously expand (with confidence threshold)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BoundState:
    """Learned bounds for a single dimension.

    Tracks both positive and negative direction limits separately,
    since robot workspace may be asymmetric.
    """

    min_value: float  # Learned minimum (negative direction limit)
    max_value: float  # Learned maximum (positive direction limit)
    min_confidence: int = 0  # Successful movements at min boundary
    max_confidence: int = 0  # Successful movements at max boundary
    failure_count_min: int = 0  # Failures in negative direction
    failure_count_max: int = 0  # Failures in positive direction
    last_update: float = 0.0  # Timestamp of last update

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundState:
        """Create from dict."""
        return cls(
            min_value=float(data.get("min_value", 0)),
            max_value=float(data.get("max_value", 0)),
            min_confidence=int(data.get("min_confidence", 0)),
            max_confidence=int(data.get("max_confidence", 0)),
            failure_count_min=int(data.get("failure_count_min", 0)),
            failure_count_max=int(data.get("failure_count_max", 0)),
            last_update=float(data.get("last_update", 0)),
        )


@dataclass
class WorkspaceBoundsConfig:
    """Configuration for bounds learning.

    Attributes:
        initial_*_limit: Starting limits (also hard ceiling - never exceeded)
        expansion_threshold: Number of successes at boundary before expanding
        expansion_step: Fraction to expand by (0.05 = 5%)
        contraction_factor: Factor to contract to on failure (0.9 = 90%)
        position_match_threshold: Tolerance for considering movement complete
        boundary_threshold: Fraction of limit considered "at boundary"
        persist_path: Path for saving learned bounds
        save_interval: Seconds between auto-saves
    """

    # Initial hardcoded limits (hard ceiling - never exceed these)
    initial_yaw_limit: float = 55.0  # degrees
    initial_pitch_limit: float = 35.0  # degrees
    initial_y_limit: float = 18.0  # mm
    initial_z_limit: float = 15.0  # mm
    initial_roll_limit: float = 35.0  # degrees
    initial_x_limit: float = 15.0  # mm

    # Learning parameters
    expansion_threshold: int = 5  # Successes at boundary before expansion
    expansion_step: float = 0.05  # Expand by 5% of current limit
    contraction_factor: float = 0.9  # Contract to 90% on failure
    position_match_threshold: float = 2.0  # Degrees/mm tolerance
    boundary_threshold: float = 0.85  # Consider at boundary if > 85% of limit

    # Persistence
    persist_path: str = "data/util/learned_bounds.json"
    save_interval: float = 60.0  # Save every 60 seconds if changed


# Dimension names and their initial limit config keys
DIMENSIONS = {
    "yaw": "initial_yaw_limit",
    "pitch": "initial_pitch_limit",
    "y": "initial_y_limit",
    "z": "initial_z_limit",
    "roll": "initial_roll_limit",
    "x": "initial_x_limit",
}


class WorkspaceBoundsLearner:
    """Learns and tracks actual workspace boundaries.

    Compares commanded vs actual positions after each movement to detect
    when the robot is constrained. Learns tighter bounds when movements
    fail, and cautiously expands when consistently successful at boundaries.

    Example:
        learner = WorkspaceBoundsLearner()

        # After a movement
        learner.record_movement_outcome(
            commanded={"yaw": 45.0, "pitch": -5.0, "y": 10.0, "z": 3.0},
            actual={"yaw": 42.5, "pitch": -5.0, "y": 10.0, "z": 3.0},
            ik_failure=False,
        )

        # Get current bound
        yaw_limit = learner.get_bound("yaw")

        # Clamp a pose
        clamped = learner.clamp_to_bounds(x, y, z, roll, pitch, yaw)
    """

    def __init__(self, config: WorkspaceBoundsConfig | None = None) -> None:
        """Initialize the bounds learner.

        Args:
            config: Configuration for learning parameters. Uses defaults if None.
        """
        self.config = config or WorkspaceBoundsConfig()
        self._bounds: dict[str, BoundState] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._last_save = 0.0

        self._initialize_bounds()
        self._load_persisted()

    def _initialize_bounds(self) -> None:
        """Initialize bounds with hardcoded limits as starting values."""
        now = time.time()

        for dim, config_key in DIMENSIONS.items():
            initial_limit = getattr(self.config, config_key, 0.0)
            self._bounds[dim] = BoundState(
                min_value=-initial_limit,
                max_value=initial_limit,
                last_update=now,
            )

    def _load_persisted(self) -> None:
        """Load persisted bounds from file if available."""
        persist_path = Path(self.config.persist_path)

        if not persist_path.exists():
            logger.debug("No persisted bounds found at %s", persist_path)
            return

        try:
            with open(persist_path, "r") as f:
                data = json.load(f)

            with self._lock:
                for dim, bound_data in data.get("bounds", {}).items():
                    if dim in self._bounds:
                        loaded = BoundState.from_dict(bound_data)

                        # Validate loaded bounds don't exceed hard ceiling
                        initial_limit = getattr(self.config, DIMENSIONS.get(dim, ""), 0.0)
                        loaded.min_value = max(loaded.min_value, -initial_limit)
                        loaded.max_value = min(loaded.max_value, initial_limit)

                        # Also enforce minimum floor (50% for rotation, 30% for translation)
                        if dim in ("yaw", "pitch", "roll"):
                            min_floor = initial_limit * 0.5
                        else:
                            min_floor = initial_limit * 0.3

                        if loaded.max_value < min_floor:
                            logger.warning(
                                "Resetting %s max bound: %.2f -> %.2f (below minimum floor)",
                                dim,
                                loaded.max_value,
                                min_floor,
                            )
                            loaded.max_value = min_floor
                        if loaded.min_value > -min_floor:
                            logger.warning(
                                "Resetting %s min bound: %.2f -> %.2f (below minimum floor)",
                                dim,
                                loaded.min_value,
                                -min_floor,
                            )
                            loaded.min_value = -min_floor

                        self._bounds[dim] = loaded

            logger.info("Loaded persisted bounds from %s", persist_path)

        except Exception as e:
            logger.warning("Failed to load persisted bounds: %s", e)

    def _save_persisted(self, force: bool = False) -> None:
        """Save bounds to file if dirty and interval elapsed.

        Args:
            force: If True, save regardless of interval.
        """
        now = time.time()

        if not force and (now - self._last_save) < self.config.save_interval:
            return

        if not self._dirty and not force:
            return

        with self._lock:
            data = {
                "bounds": {dim: bound.to_dict() for dim, bound in self._bounds.items()},
                "saved_at": now,
                "version": 1,
            }
            self._dirty = False
            self._last_save = now

        persist_path = Path(self.config.persist_path)

        try:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(persist_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved bounds to %s", persist_path)
        except Exception as e:
            logger.warning("Failed to save bounds: %s", e)

    def record_movement_outcome(
        self,
        commanded: dict[str, float],
        actual: dict[str, float],
        ik_failure: bool = False,
    ) -> None:
        """Record the outcome of a movement command.

        Compares commanded vs actual positions to detect constraints.
        Updates learned bounds based on outcome.

        Args:
            commanded: Dict of commanded positions {dim: value}
            actual: Dict of actual positions after movement {dim: value}
            ik_failure: Whether an IK failure was detected
        """
        now = time.time()
        threshold = self.config.position_match_threshold

        with self._lock:
            for dim in DIMENSIONS:
                cmd = commanded.get(dim)
                act = actual.get(dim)

                if cmd is None or act is None:
                    continue

                bound = self._bounds.get(dim)
                if bound is None:
                    continue

                initial_limit = getattr(self.config, DIMENSIONS[dim], 0.0)
                error = abs(cmd - act)

                # Determine direction
                is_positive = cmd > 0
                is_at_boundary = self._is_at_boundary(cmd, bound, is_positive)

                if ik_failure or error > threshold:
                    # Movement was constrained - tighten bound
                    self._contract_bound(dim, bound, act, is_positive, initial_limit, now)
                elif is_at_boundary:
                    # Successful movement at boundary - build confidence
                    self._build_confidence(dim, bound, is_positive, initial_limit, now)

            self._dirty = True

        # Periodic save
        self._save_persisted()

    def _is_at_boundary(self, commanded: float, bound: BoundState, is_positive: bool) -> bool:
        """Check if commanded position is near the boundary."""
        if is_positive:
            limit = bound.max_value
        else:
            limit = abs(bound.min_value)

        if limit == 0:
            return False

        usage = abs(commanded) / limit
        return usage >= self.config.boundary_threshold

    def _contract_bound(
        self,
        dim: str,
        bound: BoundState,
        actual: float,
        is_positive: bool,
        initial_limit: float,
        now: float,
    ) -> None:
        """Contract bound after a failed movement."""
        # Use actual position as new limit (slightly inside for safety)
        new_limit = abs(actual) * self.config.contraction_factor

        # Never go below a minimum (50% of initial for rotation, 30% for translation)
        # Previous 10% floor was too aggressive - resulted in unusable bounds
        if dim in ("yaw", "pitch", "roll"):
            min_limit = initial_limit * 0.5  # 50% for rotation (e.g., 27.5° for yaw)
        else:
            min_limit = initial_limit * 0.3  # 30% for translation
        new_limit = max(new_limit, min_limit)

        if is_positive:
            if new_limit < bound.max_value:
                logger.info(
                    "Contracting %s max bound: %.2f -> %.2f",
                    dim,
                    bound.max_value,
                    new_limit,
                )
                bound.max_value = new_limit
                bound.max_confidence = 0  # Reset confidence
                bound.failure_count_max += 1
        else:
            if -new_limit > bound.min_value:
                logger.info(
                    "Contracting %s min bound: %.2f -> %.2f",
                    dim,
                    bound.min_value,
                    -new_limit,
                )
                bound.min_value = -new_limit
                bound.min_confidence = 0
                bound.failure_count_min += 1

        bound.last_update = now

    def _build_confidence(
        self,
        dim: str,
        bound: BoundState,
        is_positive: bool,
        initial_limit: float,
        now: float,
    ) -> None:
        """Build confidence at boundary, potentially expanding."""
        if is_positive:
            bound.max_confidence += 1
            confidence = bound.max_confidence
        else:
            bound.min_confidence += 1
            confidence = bound.min_confidence

        # Check if we should expand
        if confidence >= self.config.expansion_threshold:
            self._expand_bound(dim, bound, is_positive, initial_limit, now)

    def _expand_bound(
        self,
        dim: str,
        bound: BoundState,
        is_positive: bool,
        initial_limit: float,
        now: float,
    ) -> None:
        """Cautiously expand bound after repeated success."""
        step = self.config.expansion_step

        if is_positive:
            current = bound.max_value
            new_limit = current * (1 + step)

            # Never exceed hard ceiling
            new_limit = min(new_limit, initial_limit)

            if new_limit > current:
                logger.info("Expanding %s max bound: %.2f -> %.2f", dim, current, new_limit)
                bound.max_value = new_limit
                bound.max_confidence = 0  # Reset for next expansion cycle
        else:
            current = bound.min_value
            new_limit = current * (1 + step)  # More negative

            # Never exceed hard ceiling
            new_limit = max(new_limit, -initial_limit)

            if new_limit < current:
                logger.info("Expanding %s min bound: %.2f -> %.2f", dim, current, new_limit)
                bound.min_value = new_limit
                bound.min_confidence = 0

        bound.last_update = now

    def get_bound(self, dimension: str, direction: str = "both") -> float:
        """Get current learned bound for a dimension.

        Args:
            dimension: One of "yaw", "pitch", "y", "z", "roll", "x"
            direction: "positive", "negative", or "both" (returns max of abs values)

        Returns:
            The learned bound value (always positive for "both")
        """
        with self._lock:
            bound = self._bounds.get(dimension)

            if bound is None:
                # Fallback to initial limit
                config_key = DIMENSIONS.get(dimension, "")
                return getattr(self.config, config_key, 0.0)

            if direction == "positive":
                return bound.max_value
            elif direction == "negative":
                return abs(bound.min_value)
            else:  # "both"
                return max(bound.max_value, abs(bound.min_value))

    def clamp_to_bounds(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Clamp 6-DOF pose to learned bounds.

        Args:
            x, y, z: Translation components (mm)
            roll, pitch, yaw: Rotation components (degrees)

        Returns:
            Tuple of clamped (x, y, z, roll, pitch, yaw)
        """
        with self._lock:
            x = self._clamp_value(x, "x")
            y = self._clamp_value(y, "y")
            z = self._clamp_value(z, "z")
            roll = self._clamp_value(roll, "roll")
            pitch = self._clamp_value(pitch, "pitch")
            yaw = self._clamp_value(yaw, "yaw")

        return (x, y, z, roll, pitch, yaw)

    def _clamp_value(self, value: float, dimension: str) -> float:
        """Clamp a single value to its learned bounds."""
        bound = self._bounds.get(dimension)

        if bound is None:
            # Fallback to initial limit
            config_key = DIMENSIONS.get(dimension, "")
            limit = getattr(self.config, config_key, 0.0)
            return max(-limit, min(limit, value))

        return max(bound.min_value, min(bound.max_value, value))

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about learned bounds."""
        with self._lock:
            stats = {}

            for dim, bound in self._bounds.items():
                initial_limit = getattr(self.config, DIMENSIONS[dim], 0.0)
                stats[dim] = {
                    "min": round(bound.min_value, 2),
                    "max": round(bound.max_value, 2),
                    "initial_limit": initial_limit,
                    "min_confidence": bound.min_confidence,
                    "max_confidence": bound.max_confidence,
                    "failures_min": bound.failure_count_min,
                    "failures_max": bound.failure_count_max,
                    "usage_min": (round(abs(bound.min_value) / initial_limit, 2) if initial_limit > 0 else 0),
                    "usage_max": (round(bound.max_value / initial_limit, 2) if initial_limit > 0 else 0),
                }

            return stats

    def to_context_str(self) -> str:
        """Generate concise context string for logging/LLM consumption."""
        with self._lock:
            lines = ["[WORKSPACE BOUNDS]"]

            for dim in ["yaw", "pitch", "y", "z"]:
                bound = self._bounds.get(dim)
                if bound:
                    initial = getattr(self.config, DIMENSIONS[dim], 0.0)
                    unit = "deg" if dim in ("yaw", "pitch", "roll") else "mm"
                    usage = max(bound.max_value, abs(bound.min_value)) / initial if initial > 0 else 0
                    lines.append(
                        f"  {dim}: [{bound.min_value:+.1f}, {bound.max_value:+.1f}] {unit} ({usage:.0%} of max)"
                    )

            return "\n".join(lines)

    def clear(self) -> None:
        """Reset all bounds to initial hardcoded values."""
        with self._lock:
            self._initialize_bounds()
            self._dirty = True

        # Delete persisted file
        persist_path = Path(self.config.persist_path)
        if persist_path.exists():
            try:
                persist_path.unlink()
                logger.info("Deleted persisted bounds file")
            except Exception as e:
                logger.warning("Failed to delete persisted bounds: %s", e)

    def save(self) -> None:
        """Force save current bounds to disk."""
        self._save_persisted(force=True)


__all__ = [
    "WorkspaceBoundsLearner",
    "WorkspaceBoundsConfig",
    "BoundState",
]
