"""Spatial context types for episodic memory (POG-0d prep).

Defines the serialization format for ``Perception.location`` before v0.2.0
so that the wire format is stable from publication. Adding spatial context
post-publication is then a 3-line change to ``Perception.to_dict()`` and
``Perception.from_dict()`` — no EpisodicMemory changes needed.

Current status: Type-only. No integration with Perception yet (that ships
when the Embodiment Hardware Adapter or SEM location sensing is wired).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpatialContext:
    """Where an event happened — from robot odometry, SEM, or campaign scene.

    Frozen so it's safe to share across threads and store in memory without
    defensive copies. All fields are optional; partially-known locations
    (e.g., room name without coordinates) are valid.
    """

    # Symbolic location (from SEM scene, campaign encounter, or user annotation)
    room: str = ""  # "kitchen", "throne_room", "neon_alley"
    zone: str = ""  # "workspace", "combat_area", "market_square"
    scene_id: str = ""  # Campaign encounter ID if from DM runtime

    # Metric coordinates (from robot odometry or SEM position sensors)
    x: float | None = None
    y: float | None = None
    z: float | None = None
    heading_deg: float | None = None  # Facing direction (0=north, 90=east)

    # Confidence / source
    source: str = "unknown"  # "odometry" | "sem" | "campaign" | "user" | "inferred"
    confidence: float = 1.0  # How sure we are about this location (0-1)

    # Metadata
    tags: tuple[str, ...] = ()  # Searchable tags ("indoor", "hazardous", "crowded")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.room:
            d["room"] = self.room
        if self.zone:
            d["zone"] = self.zone
        if self.scene_id:
            d["scene_id"] = self.scene_id
        if self.x is not None:
            d["x"] = self.x
        if self.y is not None:
            d["y"] = self.y
        if self.z is not None:
            d["z"] = self.z
        if self.heading_deg is not None:
            d["heading_deg"] = self.heading_deg
        if self.source != "unknown":
            d["source"] = self.source
        if self.confidence != 1.0:
            d["confidence"] = self.confidence
        if self.tags:
            d["tags"] = list(self.tags)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpatialContext:
        return cls(
            room=d.get("room", ""),
            zone=d.get("zone", ""),
            scene_id=d.get("scene_id", ""),
            x=d.get("x"),
            y=d.get("y"),
            z=d.get("z"),
            heading_deg=d.get("heading_deg"),
            source=d.get("source", "unknown"),
            confidence=d.get("confidence", 1.0),
            tags=tuple(d.get("tags", ())),
        )
