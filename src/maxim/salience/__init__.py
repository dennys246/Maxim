"""Salience module - Manages WHAT objects are salient in the visual scene.

This module provides:
- SalienceNetwork: Object-level salience tracking with novelty and interest matching
- ThreadSafeNoveltyTracker: Thread-safe wrapper for novelty scoring
- MovementDetector: Track object motion for salience boosting
"""

from maxim.salience.salience_network import (
    SalienceConfig,
    SalienceNetwork,
    TrackedObject,
)
from maxim.salience.novelty import (
    ThreadSafeNoveltyTracker,
)
from maxim.salience.movement_detector import (
    MovementConfig,
    MovementDetector,
)

__all__ = [
    # Salience Network
    "SalienceConfig",
    "SalienceNetwork",
    "TrackedObject",
    # Novelty
    "ThreadSafeNoveltyTracker",
    # Movement Detection
    "MovementConfig",
    "MovementDetector",
]
