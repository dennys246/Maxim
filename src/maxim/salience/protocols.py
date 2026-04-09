"""Salience protocols — modality-agnostic primitives for attention.

Defines the core abstractions that decouple salience from vision:

- ``PerceptSource``: Enum of input modalities (vision, narrative, SEM, etc.)
- ``WhereCoord``: Protocol for spatial/contextual coordinates
- ``SalienceItem``: Frozen dataclass representing one salient entity
- ``AttentionField``: Protocol for spatial attention tracking (WHERE)
- ``ChangeDetector``: Protocol for detecting state changes (motion, sensor deltas)
- ``FocusController``: Protocol for attention dynamics (saccade/fixate/scan)

The key design: ``WhereCoord`` is always present on every ``SalienceItem``
(never optional), but the coordinate system is pluggable. Vision uses pixels,
text uses narrative positions, SEM uses entity-relative coordinates.

Example::

    from maxim.salience.protocols import SalienceItem, PerceptSource
    from maxim.salience.where import NarrativeWhere

    item = SalienceItem(
        id="guard_0",
        label="guard",
        source=PerceptSource.NARRATIVE,
        confidence=0.8,
        where=NarrativeWhere("entrance", "tavern_interior"),
    )
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Percept source — which modality produced this item
# ---------------------------------------------------------------------------


class PerceptSource(enum.Enum):
    """Input modality that produced a salience item."""

    VISION = "vision"
    NARRATIVE = "narrative"
    SEM = "sem"
    AUDIO = "audio"
    TEMPORAL = "temporal"
    PROPRIOCEPTION = "proprioception"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# WhereCoord — pluggable coordinate protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class WhereCoord(Protocol):
    """Spatial/contextual coordinate for a salient item.

    Every ``SalienceItem`` carries a ``WhereCoord``. The coordinate system
    is modality-specific but the protocol is uniform, enabling the salience
    system to compute distances and label regions without knowing whether
    it's dealing with pixels, narrative positions, or entity trees.

    Implementations:
        ``PixelWhere``  — vision/YOLO bounding boxes
        ``NarrativeWhere`` — named positions in a scene
        ``EntityWhere`` — SEM entity tree coordinates
        ``TemporalWhere`` — SCN circadian phase
        ``SemanticWhere`` — ATL concept space
        ``NullWhere`` — headless/benchmark (no spatial dimension)
    """

    def distance_to(self, other: WhereCoord) -> float:
        """Distance to another coordinate in this space.

        Returns a float in [0, inf). Same-location returns 0.
        Implementations define what "distance" means for their space.
        """
        ...

    def region(self) -> str:
        """Human-readable region label for LLM context.

        Examples: "top-left", "entrance", "guard.hp", "morning".
        """
        ...

    def as_dict(self) -> dict[str, Any]:
        """Serialize for persistence and logging."""
        ...


# ---------------------------------------------------------------------------
# SalienceItem — one salient entity (modality-agnostic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SalienceItem:
    """A single salient entity detected by any modality.

    Replaces the loose ``dict[str, Any]`` detection dicts that previously
    flowed through the salience system. Frozen to prevent mutation —
    updates go through ``SalienceNetwork`` methods.

    Attributes:
        id: Stable tracking identifier (persists across updates).
        label: Human-readable label (e.g., "guard", "rusty_sword").
        source: Which modality produced this item.
        confidence: Detection/extraction confidence in [0, 1].
        where: Spatial/contextual coordinate (always present).
        class_id: Numeric class for compatibility with YOLO/COCO pipelines.
        interest_matched: Whether this item matches the current interest set.
        concept_id: ATL concept ID if recognized (set by S-3 integration).
        metadata: Additional modality-specific data.
    """

    id: str
    label: str
    source: PerceptSource
    confidence: float
    where: WhereCoord
    class_id: int = 0
    interest_matched: bool = False
    concept_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AttentionField — abstract spatial tracking (WHERE to attend)
# ---------------------------------------------------------------------------


@runtime_checkable
class AttentionField(Protocol):
    """Tracks where attention has been directed and suggests where to go next.

    The "where" equivalent of ``SalienceNetwork`` (which tracks "what").
    Implementations maintain a history of attended locations and provide
    inhibition-of-return and exploration suggestions.
    """

    def record_focus(self, where: WhereCoord, *, success: bool = True) -> None:
        """Record that attention was directed to this location.

        Args:
            where: The coordinate that was attended to.
            success: Whether the attention resulted in a useful outcome.
        """
        ...

    def get_inhibition(self, where: WhereCoord) -> float:
        """Get inhibition-of-return score for a location.

        Returns a float in [0, 1] where 1 = fully inhibited (just looked
        there), 0 = no inhibition (haven't looked there recently).
        """
        ...

    def suggest_next(self) -> WhereCoord | None:
        """Suggest the next location to attend to.

        Returns None if no suggestion is available (e.g., all locations
        equally explored).
        """
        ...

    def get_exploration_score(self, where: WhereCoord) -> float:
        """How unexplored is this location? Higher = less visited."""
        ...


# ---------------------------------------------------------------------------
# ChangeDetector — abstract state-change detection
# ---------------------------------------------------------------------------


@runtime_checkable
class ChangeDetector(Protocol):
    """Detects changes between updates that should boost salience.

    Replaces the vision-specific ``MovementDetector`` with a generic
    interface. Vision tracks pixel motion; narrative tracks new entities;
    SEM tracks sensor deltas.
    """

    def update(self, items: list[SalienceItem]) -> dict[str, float]:
        """Process new items and return change scores.

        Args:
            items: Current salient items.

        Returns:
            Dict mapping item IDs to change scores in [0, 1].
            Items not in the dict have zero change.
        """
        ...

    def get_change_score(self, item_id: str) -> float:
        """Get the current change score for a specific item."""
        ...


# ---------------------------------------------------------------------------
# FocusController — abstract attention dynamics
# ---------------------------------------------------------------------------


class FocusState(enum.Enum):
    """Current state of the attention system."""

    ATTENDING = "attending"  # Focused on a target
    SHIFTING = "shifting"  # Moving to a new target
    SCANNING = "scanning"  # Exploring when nothing is salient


@dataclass
class FocusCommand:
    """Command to direct attention to a target.

    Modality-agnostic replacement for ``GazeCommand``. In vision mode,
    this becomes a saccade; in narrative mode, it becomes "examine this
    entity"; in SEM mode, it becomes "read this sensor."
    """

    target: WhereCoord
    priority: float
    action_type: FocusState
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class FocusController(Protocol):
    """Controls attention dynamics — when to shift focus, when to hold.

    Vision: saccade/fixation timing.
    Narrative: which entity to examine next.
    SEM: which sensor to monitor.
    Null: no-op for headless.
    """

    def update(self, dt: float, top_salience: float) -> FocusCommand | None:
        """Advance the focus state machine.

        Args:
            dt: Time since last update in seconds.
            top_salience: Salience of the most salient item (for interrupt decisions).

        Returns:
            A FocusCommand if attention should move, None if holding.
        """
        ...

    @property
    def state(self) -> FocusState:
        """Current focus state."""
        ...


__all__ = [
    "AttentionField",
    "ChangeDetector",
    "FocusCommand",
    "FocusController",
    "FocusState",
    "PerceptSource",
    "SalienceItem",
    "WhereCoord",
]
