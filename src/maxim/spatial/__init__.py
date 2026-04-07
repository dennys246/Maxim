"""Spatial map module - Tracks reachable positions and spatial memory."""

from maxim.spatial.spatial_map import (
    ReachabilityGrid,
    SpatialMap,
    SpatialMapConfig,
    SpatialMemory,
    SpatialMemoryIndex,
)
from maxim.spatial.bounds_learner import (
    BoundState,
    WorkspaceBoundsConfig,
    WorkspaceBoundsLearner,
)

__all__ = [
    "BoundState",
    "ReachabilityGrid",
    "SpatialMap",
    "SpatialMapConfig",
    "SpatialMemory",
    "SpatialMemoryIndex",
    "WorkspaceBoundsConfig",
    "WorkspaceBoundsLearner",
]
