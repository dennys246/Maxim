"""Movement utilities for natural, dynamic head movements.

Provides:
- Distance-based duration calculation (farther = slower)
- Random jitter for natural variation
- Look-opposite helper for exploration
- Movement parameter computation
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MovementConfig:
    """Configuration for movement dynamics.

    Attributes:
        base_duration: Minimum duration for any movement (seconds).
        max_duration: Maximum duration cap (seconds).
        duration_per_pixel: Additional seconds per pixel of distance.
        jitter_min: Minimum random jitter to add (seconds).
        jitter_max: Maximum random jitter to add (seconds).
        image_width: Expected image width for distance calculations.
        image_height: Expected image height for distance calculations.
    """
    base_duration: float = 0.3
    max_duration: float = 2.0
    duration_per_pixel: float = 0.001  # 1ms per pixel
    jitter_min: float = 0.0
    jitter_max: float = 1.0
    image_width: int = 640
    image_height: int = 480


def compute_distance(
    from_pos: Tuple[float, float],
    to_pos: Tuple[float, float],
) -> float:
    """Compute Euclidean distance between two positions.

    Args:
        from_pos: Starting (u, v) position.
        to_pos: Target (u, v) position.

    Returns:
        Distance in pixels.
    """
    du = to_pos[0] - from_pos[0]
    dv = to_pos[1] - from_pos[1]
    return math.sqrt(du * du + dv * dv)


def compute_dynamic_duration(
    from_pos: Tuple[float, float] | None,
    to_pos: Tuple[float, float],
    config: MovementConfig | None = None,
    add_jitter: bool = True,
) -> float:
    """Compute movement duration based on distance with optional jitter.

    Longer distances result in longer durations for more natural movement.
    Random jitter adds organic variation.

    Args:
        from_pos: Current position (u, v). If None, uses image center.
        to_pos: Target position (u, v).
        config: Movement configuration. Uses defaults if None.
        add_jitter: Whether to add random jitter.

    Returns:
        Duration in seconds.

    Example:
        >>> duration = compute_dynamic_duration((320, 240), (100, 100))
        >>> # Returns ~0.5-1.5s depending on distance and jitter
    """
    cfg = config or MovementConfig()

    # Default from center if no current position
    if from_pos is None:
        from_pos = (cfg.image_width / 2, cfg.image_height / 2)

    # Calculate distance
    distance = compute_distance(from_pos, to_pos)

    # Base duration + distance-scaled duration
    duration = cfg.base_duration + (distance * cfg.duration_per_pixel)

    # Add random jitter for natural variation
    if add_jitter:
        jitter = random.uniform(cfg.jitter_min, cfg.jitter_max)
        duration += jitter

    # Clamp to max duration
    return min(duration, cfg.max_duration)


def compute_opposite_position(
    current_pos: Tuple[float, float],
    image_width: int = 640,
    image_height: int = 480,
    margin: float = 0.15,
) -> Tuple[float, float]:
    """Compute the opposite side of the view from current position.

    Useful for exploration when the agent wants to look "the other way".

    Args:
        current_pos: Current (u, v) position.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        margin: Margin from edges as fraction (0.15 = 15% from edge).

    Returns:
        (u, v) position on the opposite side of the frame.

    Example:
        >>> # If looking at top-left, returns bottom-right area
        >>> opposite = compute_opposite_position((100, 100))
        >>> # Returns approximately (540, 380)
    """
    center_u = image_width / 2
    center_v = image_height / 2

    # Vector from center to current position
    du = current_pos[0] - center_u
    dv = current_pos[1] - center_v

    # Opposite direction
    opposite_u = center_u - du
    opposite_v = center_v - dv

    # Clamp to safe margins
    min_u = image_width * margin
    max_u = image_width * (1 - margin)
    min_v = image_height * margin
    max_v = image_height * (1 - margin)

    opposite_u = max(min_u, min(max_u, opposite_u))
    opposite_v = max(min_v, min(max_v, opposite_v))

    # Add small random offset for variety
    offset_u = random.uniform(-30, 30)
    offset_v = random.uniform(-30, 30)

    final_u = max(min_u, min(max_u, opposite_u + offset_u))
    final_v = max(min_v, min(max_v, opposite_v + offset_v))

    return (final_u, final_v)


def get_direction_name(
    from_pos: Tuple[float, float],
    to_pos: Tuple[float, float],
) -> str:
    """Get human-readable direction name for a movement.

    Args:
        from_pos: Starting (u, v) position.
        to_pos: Target (u, v) position.

    Returns:
        Direction string like "upper-left", "right", "center", etc.
    """
    du = to_pos[0] - from_pos[0]
    dv = to_pos[1] - from_pos[1]

    # Threshold for significant movement
    threshold = 50

    horizontal = ""
    if du < -threshold:
        horizontal = "left"
    elif du > threshold:
        horizontal = "right"

    vertical = ""
    if dv < -threshold:
        vertical = "upper"
    elif dv > threshold:
        vertical = "lower"

    if horizontal and vertical:
        return f"{vertical}-{horizontal}"
    elif horizontal:
        return horizontal
    elif vertical:
        return vertical
    else:
        return "center"


def get_quadrant(
    position: Tuple[float, float],
    image_width: int = 640,
    image_height: int = 480,
) -> str:
    """Get which quadrant of the image a position is in.

    Args:
        position: (u, v) position.
        image_width: Image width.
        image_height: Image height.

    Returns:
        Quadrant name: "top-left", "top-right", "bottom-left", "bottom-right", or "center".
    """
    center_u = image_width / 2
    center_v = image_height / 2

    # Check if near center
    if abs(position[0] - center_u) < image_width * 0.2 and abs(position[1] - center_v) < image_height * 0.2:
        return "center"

    horizontal = "left" if position[0] < center_u else "right"
    vertical = "top" if position[1] < center_v else "bottom"

    return f"{vertical}-{horizontal}"


def suggest_exploration_direction(
    current_pos: Tuple[float, float] | None,
    recent_positions: list[Tuple[float, float]] | None = None,
    image_width: int = 640,
    image_height: int = 480,
) -> Tuple[float, float]:
    """Suggest a new position for exploration.

    Avoids recently visited areas and prefers unexplored quadrants.

    Args:
        current_pos: Current position (or None).
        recent_positions: List of recently visited positions to avoid.
        image_width: Image width.
        image_height: Image height.

    Returns:
        Suggested (u, v) position for exploration.
    """
    margin = 0.2
    min_u = image_width * margin
    max_u = image_width * (1 - margin)
    min_v = image_height * margin
    max_v = image_height * (1 - margin)

    # Define exploration targets (5 points: center + 4 corners)
    targets = [
        (image_width * 0.5, image_height * 0.5),  # center
        (min_u, min_v),  # top-left
        (max_u, min_v),  # top-right
        (min_u, max_v),  # bottom-left
        (max_u, max_v),  # bottom-right
    ]

    recent = recent_positions or []
    if current_pos:
        recent = recent + [current_pos]

    # Score each target by distance from recent positions
    best_target = targets[0]
    best_score = -1

    for target in targets:
        min_dist = float('inf')
        for recent_pos in recent:
            dist = compute_distance(recent_pos, target)
            min_dist = min(min_dist, dist)

        # Higher score = farther from recent positions
        if min_dist > best_score:
            best_score = min_dist
            best_target = target

    # Add jitter
    jitter_u = random.uniform(-40, 40)
    jitter_v = random.uniform(-40, 40)

    final_u = max(min_u, min(max_u, best_target[0] + jitter_u))
    final_v = max(min_v, min(max_v, best_target[1] + jitter_v))

    return (final_u, final_v)


# Default configuration instance
DEFAULT_MOVEMENT_CONFIG = MovementConfig()


__all__ = [
    "MovementConfig",
    "compute_distance",
    "compute_dynamic_duration",
    "compute_opposite_position",
    "get_direction_name",
    "get_quadrant",
    "suggest_exploration_direction",
    "DEFAULT_MOVEMENT_CONFIG",
]
