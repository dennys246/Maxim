"""Default Network - Reactive behavior layer for Maxim.

The Default Network (DN) is a biologically-inspired reactive layer that:
1. Handles routine visual behaviors without LLM involvement
2. Acts as a Thalamic Gate, filtering percepts before they reach deliberation
3. Provides responsive, naturalistic movement through behavior arbitration

Key components:
- DefaultNetwork: Main coordinator class
- ThalamicGate: Filters percepts for escalation to deliberative layer
- Behaviors: Reactive behavior modules (orienting, social, idle, etc.)
- PriorityArbiter: Selects winning behavior with hysteresis
"""

from __future__ import annotations

from maxim.default_network.messages import (
    ActionProposal,
    DNActionProposal,
    EscalationResult,
    FilteredPercept,
)
from maxim.default_network.arbiter import ArbiterConfig, PriorityArbiter
from maxim.attention import AttentionConfig, AttentionNetwork
from maxim.default_network.background_tasks import (
    BackgroundTaskConfig,
    BackgroundTaskManager,
    ThrottleConfig,
    OperationThrottler,
    create_throttlers,
)
from maxim.default_network.gate import (
    AdaptiveThresholdConfig,
    GateConfig,
    ThalamicGate,
)
from maxim.default_network.movement_utils import (
    MovementConfig,
    compute_dynamic_duration,
    compute_opposite_position,
    get_direction_name,
    get_quadrant,
    suggest_exploration_direction,
)
from maxim.default_network.network import DefaultNetwork, DefaultNetworkConfig
from maxim.salience import ThreadSafeNoveltyTracker
from maxim.attention import GazeHistory, GazeHistoryConfig, GazePosition
from maxim.salience import SalienceConfig, SalienceNetwork
from maxim.spatial import SpatialMap, SpatialMapConfig, SpatialMemory
from maxim.default_network.behaviors import (
    Behavior,
    BehaviorState,
    IdleScan,
    Microsaccades,
    MotionTracking,
    OrientingResponse,
    ReturnToCenter,
    SocialAttention,
    StartleResponse,
    create_default_behaviors,
)
from maxim.default_network.config import (
    DNConfig,
    BehaviorsConfig,
    load_dn_config,
    save_dn_config,
    create_behaviors_from_config,
)

__all__ = [
    # Core
    "DefaultNetwork",
    "DefaultNetworkConfig",
    # Messages
    "ActionProposal",
    "DNActionProposal",
    "EscalationResult",
    "FilteredPercept",
    # Behaviors - Base
    "Behavior",
    "BehaviorState",
    "create_default_behaviors",
    # Behaviors - Concrete
    "IdleScan",
    "Microsaccades",
    "MotionTracking",
    "OrientingResponse",
    "ReturnToCenter",
    "SocialAttention",
    "StartleResponse",
    # Arbiter
    "ArbiterConfig",
    "PriorityArbiter",
    # Attention Network
    "AttentionConfig",
    "AttentionNetwork",
    # Background Tasks
    "BackgroundTaskConfig",
    "BackgroundTaskManager",
    "ThrottleConfig",
    "OperationThrottler",
    "create_throttlers",
    # Salience Network
    "SalienceConfig",
    "SalienceNetwork",
    # Spatial Map
    "SpatialMap",
    "SpatialMapConfig",
    "SpatialMemory",
    # Movement Utils
    "MovementConfig",
    "compute_dynamic_duration",
    "compute_opposite_position",
    "get_direction_name",
    "get_quadrant",
    "suggest_exploration_direction",
    # Gate
    "AdaptiveThresholdConfig",
    "GateConfig",
    "ThalamicGate",
    # Novelty
    "ThreadSafeNoveltyTracker",
    # Gaze History
    "GazeHistory",
    "GazeHistoryConfig",
    "GazePosition",
    # Config
    "DNConfig",
    "BehaviorsConfig",
    "load_dn_config",
    "save_dn_config",
    "create_behaviors_from_config",
]
