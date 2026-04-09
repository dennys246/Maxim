"""Modes package — autonomy levels and processing states.

Modes define safety constraints and permissions (planning/supervised/autonomous).
The agent pipeline handles behavioral flexibility naturally.
"""

from maxim.modes.definitions import (
    ModeDefinition,
    OPERATIONAL_MODES,
    get_mode,
)

__all__ = [
    "ModeDefinition",
    "OPERATIONAL_MODES",
    "get_mode",
]
