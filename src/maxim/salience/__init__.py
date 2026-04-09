"""Salience module — manages WHAT is salient across all modalities.

Core abstractions (modality-agnostic):
- SalienceItem: Frozen dataclass representing one salient entity
- WhereCoord: Protocol for spatial/contextual coordinates
- PerceptSource: Enum of input modalities (vision, narrative, SEM, etc.)
- AttentionField, ChangeDetector, FocusController: abstract protocols

Concrete implementations:
- SalienceNetwork: Object-level salience tracking with novelty and interest matching
- ThreadSafeNoveltyTracker: Thread-safe wrapper for novelty scoring
- MovementDetector: Track object motion for salience boosting (vision)
- PixelWhere, NarrativeWhere, EntityWhere, etc.: WhereCoord implementations
"""

from maxim.salience.protocols import (
    AttentionField,
    ChangeDetector,
    FocusCommand,
    FocusController,
    FocusState,
    PerceptSource,
    SalienceItem,
    WhereCoord,
)
from maxim.salience.where import (
    EntityWhere,
    NarrativeWhere,
    NullWhere,
    PixelWhere,
    SemanticWhere,
    TemporalWhere,
)
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
    # Protocols (modality-agnostic)
    "AttentionField",
    "ChangeDetector",
    "FocusCommand",
    "FocusController",
    "FocusState",
    "PerceptSource",
    "SalienceItem",
    "WhereCoord",
    # WhereCoord implementations
    "EntityWhere",
    "NarrativeWhere",
    "NullWhere",
    "PixelWhere",
    "SemanticWhere",
    "TemporalWhere",
    # Salience Network
    "SalienceConfig",
    "SalienceNetwork",
    "TrackedObject",
    # Novelty
    "ThreadSafeNoveltyTracker",
    # Movement Detection (vision-specific)
    "MovementConfig",
    "MovementDetector",
]
