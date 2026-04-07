"""Attention module - Manages WHERE the robot attends in visual space.

This module provides:
- AttentionNetwork: Grid-based spatial attention tracking
- GazeHistory: Temporal tracking with inhibition-of-return
- SalienceMap: Unified spatial salience combining all attention factors
- GazeController: Human-like saccade-fixate dynamics
- SceneContextDetector: Detect scene changes for exploration
"""

from maxim.attention.attention_network import (
    AttentionCell,
    AttentionConfig,
    AttentionNetwork,
)
from maxim.attention.gaze_history import (
    GazeHistory,
    GazeHistoryConfig,
    GazePosition,
)
from maxim.attention.salience_map import (
    SalienceCell,
    SalienceMap,
    SalienceMapConfig,
)
from maxim.attention.gaze_controller import (
    GazeCommand,
    GazeController,
    GazeControllerConfig,
    GazeState,
)
from maxim.attention.scene_context import (
    SceneContextConfig,
    SceneContextDetector,
    SceneSnapshot,
)

__all__ = [
    # Attention Network
    "AttentionCell",
    "AttentionConfig",
    "AttentionNetwork",
    # Gaze History
    "GazeHistory",
    "GazeHistoryConfig",
    "GazePosition",
    # Salience Map
    "SalienceCell",
    "SalienceMap",
    "SalienceMapConfig",
    # Gaze Controller
    "GazeCommand",
    "GazeController",
    "GazeControllerConfig",
    "GazeState",
    # Scene Context
    "SceneContextConfig",
    "SceneContextDetector",
    "SceneSnapshot",
]
