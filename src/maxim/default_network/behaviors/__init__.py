"""Reactive behavior modules for the Default Network.

Each behavior evaluates current detections and proposes actions.
Behaviors are lightweight and execute in <10ms.

Available behaviors:
- OrientingResponse: Look at novel objects
- SocialAttention: Track people and faces
- MotionTracking: Follow moving objects
- IdleScan: Scan environment when idle
- Microsaccades: Small random movements during fixation
- ReturnToCenter: Drift back to neutral position
- StartleResponse: Quick reaction to sudden appearances
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from maxim.default_network.behaviors.base import Behavior, BehaviorState

if TYPE_CHECKING:
    from maxim.default_network.novelty import ThreadSafeNoveltyTracker
from maxim.default_network.behaviors.orienting import OrientingResponse
from maxim.default_network.behaviors.social import SocialAttention
from maxim.default_network.behaviors.motion import MotionTracking
from maxim.default_network.behaviors.idle import (
    IdleScan,
    Microsaccades,
    ReturnToCenter,
)
from maxim.default_network.behaviors.startle import StartleResponse

__all__ = [
    # Base
    "Behavior",
    "BehaviorState",
    # Behaviors
    "IdleScan",
    "Microsaccades",
    "MotionTracking",
    "OrientingResponse",
    "ReturnToCenter",
    "SocialAttention",
    "StartleResponse",
]


def create_default_behaviors(
    novelty_tracker: "ThreadSafeNoveltyTracker | None" = None,
    frame_size: tuple[int, int] = (640, 480),
) -> list[Behavior]:
    """Create a standard set of behaviors for the Default Network.

    Args:
        novelty_tracker: Shared novelty tracker for orienting behavior.
        frame_size: Video frame dimensions for peripheral calculations.

    Returns:
        List of configured behavior instances.
    """
    return [
        OrientingResponse(novelty_tracker=novelty_tracker),
        SocialAttention(prefer_faces=True),
        MotionTracking(velocity_threshold=50.0),
        StartleResponse(frame_size=frame_size),
        IdleScan(idle_timeout=5.0),
        Microsaccades(fixation_timeout=2.0),
        ReturnToCenter(threshold=0.7),
    ]
