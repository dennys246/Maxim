"""Proprioception module - Body awareness and pain detection.

Provides movement tracking and pain detection for proprioceptive awareness,
enabling the robot to learn from and avoid harmful movement patterns.
"""

from maxim.proprioception.focus_learner import (
    FocusError,
    FocusLearner,
    FocusLearnerConfig,
    FocusSample,
)
from maxim.proprioception.movement_tracker import (
    MovementMetrics,
    MovementSample,
    MovementTracker,
)
from maxim.proprioception.pain import PainConfig, PainDetector, PainSignal, PainType

__all__ = [
    # Focus learning
    "FocusError",
    "FocusLearner",
    "FocusLearnerConfig",
    "FocusSample",
    # Movement tracking
    "MovementMetrics",
    "MovementSample",
    "MovementTracker",
    # Pain detection
    "PainConfig",
    "PainDetector",
    "PainSignal",
    "PainType",
]
