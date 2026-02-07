"""SpatialMemoryBridge - Consolidates spatial learning across sessions.

Connects: Hippocampus <-> SpatialMap <-> AttentionNetwork

Problem: SpatialMap rebuilds from scratch each session (~5 min decay).
Robot forgets where objects appear.

Solution: Multi-session spatial priors from Hippocampus.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.attention.attention_network import AttentionNetwork
    from maxim.memory.hippocampus import Hippocampus
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.spatial.spatial_map import SpatialMap

logger = logging.getLogger(__name__)


@dataclass
class ObjectLocationPrior:
    """Prior probability for an object appearing at a location."""

    object_class: str
    grid_u: int
    grid_v: int
    probability: float
    observation_count: int = 0
    last_seen: float = 0.0
    success_count: int = 0  # Times we found it here


@dataclass
class SpatialMemoryBridge:
    """Consolidates spatial learning across sessions via Hippocampus.

    Features:
    - Location priors: Learn where object classes typically appear
    - Attention boosting: Direct attention to historically successful positions
    - Memory enrichment: Record spatial context with episodic memories

    Example:
        bridge = SpatialMemoryBridge(
            hippocampus=hippocampus,
            spatial_map=spatial_map,
            ec=ec,
        )

        # On session start, restore priors from memory
        bridge.on_session_start()

        # Boost attention for current goal
        boosts = bridge.boost_attention_for_goal("find mug")
        for position, boost in boosts:
            attention.apply_boost(position, boost)

        # Record successful find
        bridge.record_success(
            object_class="mug",
            position=(320, 240),
            goal="find mug",
        )
    """

    hippocampus: "Hippocampus"
    spatial_map: "SpatialMap"
    ec: "EntorhinalCortex"
    attention: "AttentionNetwork | None" = None

    # Location priors: object_class -> list of ObjectLocationPrior
    _location_priors: dict[str, list[ObjectLocationPrior]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Configuration
    min_observations_for_prior: int = 3
    prior_decay_days: float = 30.0
    max_priors_per_class: int = 10

    # Health tracking
    _healthy: bool = True
    _error_count: int = 0
    _max_errors: int = 5

    def __post_init__(self) -> None:
        """Initialize default factory fields."""
        if not hasattr(self, "_location_priors") or self._location_priors is None:
            self._location_priors = defaultdict(list)

    # ─────────────────────────────────────────────────────────────────────────
    # Session Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def on_session_start(self) -> int:
        """Restore spatial priors from Hippocampus at session start.

        Queries memories for successful object interactions and builds
        location priors from historical data.

        Returns:
            Number of priors restored
        """
        try:
            restored = 0

            # Query successful memories with detected objects
            memories = self.hippocampus.recall(
                limit=500,
                success=True,
            )

            # Extract location patterns
            location_counts: dict[tuple[str, int, int], dict[str, Any]] = {}

            for memory in memories:
                if hasattr(memory, "perception"):
                    for obj in memory.perception.detected_objects:
                        # Get position from memory if available
                        if hasattr(memory.perception, "observations"):
                            obs = memory.perception.observations
                            if "position" in obs:
                                pos = obs["position"]
                                grid_u, grid_v = self._pixel_to_grid(pos[0], pos[1])
                                key = (obj, grid_u, grid_v)

                                if key not in location_counts:
                                    location_counts[key] = {
                                        "count": 0,
                                        "success_count": 0,
                                        "last_seen": 0.0,
                                    }

                                location_counts[key]["count"] += 1
                                location_counts[key]["success_count"] += 1
                                location_counts[key]["last_seen"] = max(
                                    location_counts[key]["last_seen"],
                                    memory.timestamp,
                                )

            # Build priors from counts
            for (obj_class, grid_u, grid_v), data in location_counts.items():
                if data["count"] >= self.min_observations_for_prior:
                    prior = ObjectLocationPrior(
                        object_class=obj_class,
                        grid_u=grid_u,
                        grid_v=grid_v,
                        probability=min(0.9, data["count"] / 20),  # Cap at 0.9
                        observation_count=data["count"],
                        last_seen=data["last_seen"],
                        success_count=data["success_count"],
                    )
                    self._location_priors[obj_class].append(prior)
                    restored += 1

            # Enforce limits per class
            for obj_class in self._location_priors:
                priors = self._location_priors[obj_class]
                if len(priors) > self.max_priors_per_class:
                    # Keep highest probability priors
                    priors.sort(key=lambda p: p.probability, reverse=True)
                    self._location_priors[obj_class] = priors[: self.max_priors_per_class]

            logger.info(
                "Restored %d spatial priors from memory (%d object classes)",
                restored,
                len(self._location_priors),
            )
            return restored

        except Exception as e:
            self._record_error(e)
            return 0

    def on_session_end(self) -> None:
        """Called at session end to consolidate spatial learning."""
        # Priors are captured through record_success calls
        # No explicit action needed at session end
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Attention Integration
    # ─────────────────────────────────────────────────────────────────────────

    def boost_attention_for_goal(
        self,
        goal: str,
        boost_factor: float = 1.5,
    ) -> list[tuple[tuple[int, int], float]]:
        """Get attention boosts for positions relevant to a goal.

        Analyzes the goal text to identify object classes, then returns
        grid positions where those objects have historically appeared.

        Args:
            goal: Current goal text (e.g., "find the mug")
            boost_factor: Multiplier for attention (1.0 = no boost)

        Returns:
            List of ((grid_u, grid_v), boost_value) tuples
        """
        if not self._healthy:
            return []

        try:
            boosts: list[tuple[tuple[int, int], float]] = []

            # Extract potential object classes from goal
            goal_lower = goal.lower()
            for obj_class, priors in self._location_priors.items():
                if obj_class.lower() in goal_lower:
                    for prior in priors:
                        boost = prior.probability * boost_factor
                        boosts.append(((prior.grid_u, prior.grid_v), boost))

            return boosts

        except Exception as e:
            self._record_error(e)
            return []

    def get_likely_positions(
        self,
        object_class: str,
        top_k: int = 5,
    ) -> list[tuple[int, int, float]]:
        """Get most likely grid positions for an object class.

        Args:
            object_class: Object class to find (e.g., "mug", "person")
            top_k: Maximum positions to return

        Returns:
            List of (grid_u, grid_v, probability) tuples, sorted by probability
        """
        if not self._healthy:
            return [(0, 0, 0.1)]  # Center fallback

        try:
            priors = self._location_priors.get(object_class.lower(), [])

            if not priors:
                # No history - return center with low confidence
                return [(5, 5, 0.1)]

            # Sort by probability and return top k
            priors.sort(key=lambda p: p.probability, reverse=True)
            return [
                (p.grid_u, p.grid_v, p.probability) for p in priors[:top_k]
            ]

        except Exception as e:
            self._record_error(e)
            return [(5, 5, 0.1)]

    # ─────────────────────────────────────────────────────────────────────────
    # Recording
    # ─────────────────────────────────────────────────────────────────────────

    def record_success(
        self,
        object_class: str,
        position: tuple[float, float],
        goal: str | None = None,
    ) -> None:
        """Record a successful object interaction at a position.

        Updates location priors for the object class.

        Args:
            object_class: Class of object found
            position: (u, v) pixel coordinates where found
            goal: Goal that was being pursued (optional)
        """
        if not self._healthy:
            return

        try:
            now = time.time()
            grid_u, grid_v = self._pixel_to_grid(position[0], position[1])

            # Find or create prior
            priors = self._location_priors[object_class.lower()]
            existing = None
            for p in priors:
                if p.grid_u == grid_u and p.grid_v == grid_v:
                    existing = p
                    break

            if existing:
                existing.observation_count += 1
                existing.success_count += 1
                existing.last_seen = now
                # Update probability with exponential moving average
                existing.probability = min(
                    0.95,
                    existing.probability * 0.8 + 0.2 * 1.0,
                )
            else:
                priors.append(
                    ObjectLocationPrior(
                        object_class=object_class.lower(),
                        grid_u=grid_u,
                        grid_v=grid_v,
                        probability=0.3,  # Initial probability
                        observation_count=1,
                        last_seen=now,
                        success_count=1,
                    )
                )

            # Enforce limits
            if len(priors) > self.max_priors_per_class:
                priors.sort(key=lambda p: p.probability, reverse=True)
                self._location_priors[object_class.lower()] = priors[
                    : self.max_priors_per_class
                ]

        except Exception as e:
            self._record_error(e)

    def record_failure(
        self,
        object_class: str,
        position: tuple[float, float],
    ) -> None:
        """Record a failed object search at a position.

        Decreases probability for the location.
        """
        if not self._healthy:
            return

        try:
            grid_u, grid_v = self._pixel_to_grid(position[0], position[1])

            priors = self._location_priors.get(object_class.lower(), [])
            for prior in priors:
                if prior.grid_u == grid_u and prior.grid_v == grid_v:
                    # Decrease probability
                    prior.probability *= 0.9
                    prior.observation_count += 1
                    break

        except Exception as e:
            self._record_error(e)

    def add_location_prior(
        self,
        object_class: str,
        location: str,
        probability: float,
    ) -> None:
        """Add a cold-start location prior.

        Used during initialization to inject prior knowledge.

        Args:
            object_class: Object class (e.g., "mug")
            location: Location description (parsed to grid coords)
            probability: Prior probability (0-1)
        """
        # Parse location string to grid coordinates
        # For now, accept "center", "left", "right", etc.
        grid_u, grid_v = self._parse_location(location)

        priors = self._location_priors[object_class.lower()]
        priors.append(
            ObjectLocationPrior(
                object_class=object_class.lower(),
                grid_u=grid_u,
                grid_v=grid_v,
                probability=probability,
                observation_count=0,  # Prior, not observed
                last_seen=0.0,
                success_count=0,
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _pixel_to_grid(self, pixel_u: float, pixel_v: float) -> tuple[int, int]:
        """Convert pixel coordinates to grid cell indices."""
        # Use spatial map's grid if available
        if self.spatial_map:
            return self.spatial_map._pixel_to_grid(pixel_u, pixel_v)

        # Fallback: assume 10x10 grid over 640x480 image
        grid_u = min(9, max(0, int(pixel_u / 64)))
        grid_v = min(9, max(0, int(pixel_v / 48)))
        return (grid_u, grid_v)

    def _parse_location(self, location: str) -> tuple[int, int]:
        """Parse location string to grid coordinates."""
        location = location.lower()
        # Assume 10x10 grid
        locations = {
            "center": (5, 5),
            "left": (2, 5),
            "right": (8, 5),
            "top": (5, 2),
            "bottom": (5, 8),
            "top-left": (2, 2),
            "top-right": (8, 2),
            "bottom-left": (2, 8),
            "bottom-right": (8, 8),
        }
        return locations.get(location, (5, 5))

    def _record_error(self, error: Exception) -> None:
        """Record an error and potentially disable the bridge."""
        self._error_count += 1
        logger.warning("SpatialMemoryBridge error (%d/%d): %s",
                       self._error_count, self._max_errors, error)

        if self._error_count >= self._max_errors:
            self._healthy = False
            logger.error("SpatialMemoryBridge disabled after %d errors",
                         self._error_count)

    @property
    def is_healthy(self) -> bool:
        """Check if bridge is operational."""
        return self._healthy

    def stats(self) -> dict[str, Any]:
        """Return bridge statistics."""
        total_priors = sum(len(p) for p in self._location_priors.values())
        return {
            "healthy": self._healthy,
            "error_count": self._error_count,
            "object_classes": len(self._location_priors),
            "total_priors": total_priors,
        }


__all__ = ["SpatialMemoryBridge", "ObjectLocationPrior"]