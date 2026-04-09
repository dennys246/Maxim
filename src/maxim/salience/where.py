"""WhereCoord implementations — pluggable coordinate systems for salience.

Each implementation satisfies the ``WhereCoord`` protocol with modality-
specific distance semantics:

- ``PixelWhere``: 2D pixel coordinates from vision/YOLO
- ``NarrativeWhere``: Named positions in a narrative scene
- ``EntityWhere``: SEM entity tree coordinates
- ``TemporalWhere``: SCN circadian temporal coordinates
- ``SemanticWhere``: ATL concept space coordinates
- ``NullWhere``: No spatial dimension (headless/benchmark)

Example::

    a = NarrativeWhere("entrance", "tavern")
    b = NarrativeWhere("bar", "tavern")
    c = NarrativeWhere("entrance", "dungeon")

    a.distance_to(b)  # 1.0 (same scene, different position)
    a.distance_to(c)  # 2.0 (different scene)
    a.region()         # "entrance"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from maxim.salience.protocols import WhereCoord


# ---------------------------------------------------------------------------
# PixelWhere — vision / YOLO bounding boxes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelWhere:
    """2D pixel coordinate with optional bounding box dimensions.

    Distance is Euclidean in pixel space. Region is derived from
    position relative to frame dimensions.

    Attributes:
        u: Horizontal pixel coordinate (center of bbox).
        v: Vertical pixel coordinate (center of bbox).
        w: Bounding box width (0 if point, not bbox).
        h: Bounding box height (0 if point, not bbox).
        frame_w: Frame width for region calculation.
        frame_h: Frame height for region calculation.
    """

    u: float
    v: float
    w: float = 0.0
    h: float = 0.0
    frame_w: float = 640.0
    frame_h: float = 480.0

    def distance_to(self, other: WhereCoord) -> float:
        if isinstance(other, PixelWhere):
            return math.sqrt((self.u - other.u) ** 2 + (self.v - other.v) ** 2)
        return float("inf")

    def region(self) -> str:
        col = "left" if self.u < self.frame_w / 3 else "right" if self.u > 2 * self.frame_w / 3 else "center"
        row = "top" if self.v < self.frame_h / 3 else "bottom" if self.v > 2 * self.frame_h / 3 else "middle"
        if row == "middle":
            return col
        return f"{row}-{col}"

    def as_dict(self) -> dict[str, Any]:
        return {"type": "pixel", "u": self.u, "v": self.v, "w": self.w, "h": self.h}

    @classmethod
    def from_bbox(cls, bbox: tuple | list, frame_w: float = 640.0, frame_h: float = 480.0) -> PixelWhere:
        """Create from a (x, y, w, h) or (x1, y1, x2, y2) bounding box."""
        if len(bbox) >= 4:
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            # Heuristic: if w > x and h > y, it's likely xyxy format
            if w > x * 1.5 and h > y * 1.5:
                cx = (x + w) / 2
                cy = (y + h) / 2
                bw = w - x
                bh = h - y
                return cls(u=cx, v=cy, w=bw, h=bh, frame_w=frame_w, frame_h=frame_h)
            return cls(u=x + w / 2, v=y + h / 2, w=w, h=h, frame_w=frame_w, frame_h=frame_h)
        return cls(u=0, v=0, frame_w=frame_w, frame_h=frame_h)


# ---------------------------------------------------------------------------
# NarrativeWhere — named positions in a narrative scene
# ---------------------------------------------------------------------------

# Ordinal distance: positions within a scene have distance 1,
# positions in different scenes have distance 2.

_SAME_SCENE_DISTANCE = 1.0
_DIFFERENT_SCENE_DISTANCE = 2.0


@dataclass(frozen=True)
class NarrativeWhere:
    """Named position within a narrative scene.

    Distance is ordinal:
    - Same position, same scene → 0
    - Different position, same scene → 1
    - Different scene → 2

    Attributes:
        position: Named position (e.g., "entrance", "bar", "center").
        scene: Scene identifier (e.g., "tavern_interior", "back_alley").
    """

    position: str
    scene: str = ""

    def distance_to(self, other: WhereCoord) -> float:
        if isinstance(other, NarrativeWhere):
            if self.scene and other.scene and self.scene != other.scene:
                return _DIFFERENT_SCENE_DISTANCE
            if self.position == other.position:
                return 0.0
            return _SAME_SCENE_DISTANCE
        return float("inf")

    def region(self) -> str:
        if self.scene:
            return f"{self.position} ({self.scene})"
        return self.position

    def as_dict(self) -> dict[str, Any]:
        return {"type": "narrative", "position": self.position, "scene": self.scene}


# ---------------------------------------------------------------------------
# EntityWhere — SEM entity tree coordinates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityWhere:
    """Coordinate in the SEM entity tree.

    Distance is graph hops in the entity hierarchy. Entities with a
    common parent are closer than entities in different branches.

    Attributes:
        entity_name: Name of the SEM entity.
        sensor_name: Specific sensor (optional — for sensor-level attention).
        path: Dot-separated path in the entity tree (e.g., "runner.megarm_v3").
    """

    entity_name: str
    sensor_name: str = ""
    path: str = ""

    def distance_to(self, other: WhereCoord) -> float:
        if isinstance(other, EntityWhere):
            if self.entity_name == other.entity_name:
                if self.sensor_name == other.sensor_name:
                    return 0.0
                return 0.5  # Same entity, different sensor
            # Use path prefix matching for tree distance
            if self.path and other.path:
                self_parts = self.path.split(".")
                other_parts = other.path.split(".")
                common = 0
                for a, b in zip(self_parts, other_parts):
                    if a == b:
                        common += 1
                    else:
                        break
                total = max(len(self_parts), len(other_parts))
                return float(total - common)
            return 1.0  # No path info, assume neighbors
        return float("inf")

    def region(self) -> str:
        if self.sensor_name:
            return f"{self.entity_name}.{self.sensor_name}"
        return self.entity_name

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "entity",
            "entity_name": self.entity_name,
            "sensor_name": self.sensor_name,
            "path": self.path,
        }


# ---------------------------------------------------------------------------
# TemporalWhere — SCN circadian temporal coordinates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalWhere:
    """Temporal coordinate from the SCN circadian system.

    Distance is circular on a 24-hour clock (6am→6pm = 12, not 12).
    Region returns human-readable time-of-day labels.

    Attributes:
        hour: Hour of day (0-23).
        day_of_week: Day of week (0-6, Monday=0).
        scn_phase: Circadian phase from SCN oscillator (0-1).
    """

    hour: int = 0
    day_of_week: int = 0
    scn_phase: float = 0.0

    def distance_to(self, other: WhereCoord) -> float:
        if isinstance(other, TemporalWhere):
            # Circular distance on 24h clock
            diff = abs(self.hour - other.hour)
            hour_dist = min(diff, 24 - diff) / 12.0  # Normalize to [0, 1]
            # Day distance (linear, normalized)
            day_diff = abs(self.day_of_week - other.day_of_week)
            day_dist = min(day_diff, 7 - day_diff) / 3.5
            return hour_dist * 0.7 + day_dist * 0.3
        return float("inf")

    def region(self) -> str:
        if self.hour < 6:
            return "night"
        if self.hour < 12:
            return "morning"
        if self.hour < 18:
            return "afternoon"
        return "evening"

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "temporal",
            "hour": self.hour,
            "day_of_week": self.day_of_week,
            "scn_phase": self.scn_phase,
        }


# ---------------------------------------------------------------------------
# SemanticWhere — ATL concept space coordinates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticWhere:
    """Coordinate in ATL concept space.

    Distance is based on concept category matching and relationship
    proximity in the ATL graph. Exact similarity requires ATL lookup
    (deferred to S-3); this provides structural distance.

    Attributes:
        concept_id: ATL concept identifier.
        category: Concept category (e.g., "character", "weapon", "location").
        related_concepts: Set of directly related concept IDs.
    """

    concept_id: str
    category: str = ""
    related_concepts: frozenset[str] = field(default_factory=frozenset)

    def distance_to(self, other: WhereCoord) -> float:
        if isinstance(other, SemanticWhere):
            if self.concept_id == other.concept_id:
                return 0.0
            # Check if directly related
            if other.concept_id in self.related_concepts or self.concept_id in other.related_concepts:
                return 0.5
            # Same category = closer than different category
            if self.category and other.category and self.category == other.category:
                return 1.0
            return 2.0
        return float("inf")

    def region(self) -> str:
        if self.category:
            return f"{self.concept_id} ({self.category})"
        return self.concept_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "semantic",
            "concept_id": self.concept_id,
            "category": self.category,
            "related_concepts": sorted(self.related_concepts),
        }


# ---------------------------------------------------------------------------
# NullWhere — no spatial dimension (headless / benchmark)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NullWhere:
    """No spatial coordinate — for headless or benchmark mode.

    All instances are equidistant (distance 0). Satisfies the protocol
    so SalienceNetwork doesn't need special-casing for no-spatial mode.
    """

    label: str = ""

    def distance_to(self, other: WhereCoord) -> float:
        return 0.0

    def region(self) -> str:
        return self.label or "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {"type": "null", "label": self.label}


__all__ = [
    "EntityWhere",
    "NarrativeWhere",
    "NullWhere",
    "PixelWhere",
    "SemanticWhere",
    "TemporalWhere",
]
