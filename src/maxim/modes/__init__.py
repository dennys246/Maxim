"""Modes package — autonomy levels and processing states.

Modes define safety constraints and permissions (passive/active/singularity).
The agent pipeline handles behavioral flexibility naturally.
"""

from maxim.modes.definitions import (
    ModeDefinition,
    MODES,
    OPERATIONAL_MODES,
    get_mode,
)

__all__ = [
    "ModeDefinition",
    "MODES",
    "OPERATIONAL_MODES",
    "get_mode",
]
