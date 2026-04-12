"""Sensory modality types for entity-modulated perception.

Defines the modality system that tags percepts with sensory metadata.
Existing consumers check ``percept.source`` (string) and ignore
``sensory`` — backward compatible. New consumers (DM campaigns,
embodiment, novelty tracking) use ``percept.sensory`` for richer
processing.

Modality surface
================

+----------------+--------------------+------------------------------+--------------------------------------+
| Modality       | Typical submodality| Fields used                  | Producer location                    |
+================+====================+==============================+======================================+
| SIGHT          | "detection",       | spatial_source, intensity,   | agents/perception_agent.py           |
|                | "scene"            | entity_source                | (_on_captured_frame,                 |
|                |                    |                              |  process_captured_frame,             |
|                |                    |                              |  process_observation vision branch)  |
+----------------+--------------------+------------------------------+--------------------------------------+
| NARRATIVE      | "cli", "transcript"| (none required)              | agents/perception_agent.py           |
|                |                    |                              | (process_observation text branch),   |
|                |                    |                              | agents/percept_factory.py            |
|                |                    |                              | (make_text_percept,                  |
|                |                    |                              |  make_scene_percept),                |
|                |                    |                              | comms/gateway.py,                    |
|                |                    |                              | comms/conversation.py                |
+----------------+--------------------+------------------------------+--------------------------------------+
| INTEROCEPTION  | "vital", "pain",   | intensity, entity_source     | embodiment/percepts.py               |
|                | "fatigue"          |                              | (EmbodimentPerceptSource),           |
|                |                    |                              | agents/percept_factory.py            |
|                |                    |                              | (make_intero_percept),               |
|                |                    |                              | simulation/conversational_source.py  |
|                |                    |                              | (inject_pain legacy path)            |
+----------------+--------------------+------------------------------+--------------------------------------+
| SOUND          | "speech",          | spatial_source, intensity,   | (stub — future audio pipeline)       |
|                | "ambient"          | entity_source                |                                      |
+----------------+--------------------+------------------------------+--------------------------------------+
| TOUCH          | "pressure", "pain",| spatial_source, intensity    | (stub — future tactile sensors)      |
|                | "texture"          |                              |                                      |
+----------------+--------------------+------------------------------+--------------------------------------+
| SMELL          | "environmental",   | spatial_source, intensity    | (stub — future olfactory)            |
|                | "tracking"         |                              |                                      |
+----------------+--------------------+------------------------------+--------------------------------------+
| ABSTRACT       | "tool_result",     | (none required)              | mesh/peer_channel.py,                |
|                | "system"           |                              | runtime/agent_loop.py                |
+----------------+--------------------+------------------------------+--------------------------------------+

Example::

    from maxim.agents.modality import SensoryModality, SensoryTag

    tag = SensoryTag(
        modality=SensoryModality.SOUND,
        submodality="speech",
        spatial_source="ahead",
        intensity=0.7,
        entity_source="guard_captain",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SensoryModality(Enum):
    """What sense channel produced this percept."""

    SIGHT = "sight"  # Visual — detections, scene descriptions
    SOUND = "sound"  # Auditory — speech, ambient, alerts
    TOUCH = "touch"  # Tactile — proprioception, pain, texture
    SMELL = "smell"  # Olfactory — environmental, tracking
    INTEROCEPTION = "intero"  # Internal — hunger, fatigue, emotional state
    NARRATIVE = "narrative"  # Meta — DM scene-setting, exposition (not a real sense)
    ABSTRACT = "abstract"  # Non-sensory — tool results, system messages


@dataclass(frozen=True)
class SensoryTag:
    """Rich sensory metadata attached to a Percept.

    Attributes:
        modality: Which sense channel produced this percept.
        submodality: Finer distinction within a modality (e.g., "speech"
            vs "ambient" for SOUND, "pain" vs "pressure" for TOUCH).
        spatial_source: Where the stimulus came from relative to the
            perceiver ("behind", "left", "overhead", "ahead").
        intensity: Raw signal strength before entity modulation (0-1).
        entity_source: SEM entity name that produced this percept
            (e.g., "guard_captain", "longsword"). Empty if no entity.
        perceived_intensity: Signal strength after entity sensor
            filtering. Set by SensoryGate. None if unmodulated.
        modulated_by: Which entity sensor filtered this percept
            (e.g., "derek.perception.sight_acuity"). Empty if unmodulated.
    """

    modality: SensoryModality
    submodality: str = ""
    spatial_source: str = ""
    intensity: float = 0.5
    entity_source: str = ""
    # Set by SensoryGate (entity modulation):
    perceived_intensity: float | None = None
    modulated_by: str = ""

    def to_dict(self) -> dict:
        """Serialize for JSON persistence."""
        return {
            "modality": self.modality.value,
            "submodality": self.submodality,
            "spatial_source": self.spatial_source,
            "intensity": self.intensity,
            "entity_source": self.entity_source,
            "perceived_intensity": self.perceived_intensity,
            "modulated_by": self.modulated_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SensoryTag:
        """Deserialize from dict."""
        return cls(
            modality=SensoryModality(data["modality"]),
            submodality=data.get("submodality", ""),
            spatial_source=data.get("spatial_source", ""),
            intensity=data.get("intensity", 0.5),
            entity_source=data.get("entity_source", ""),
            perceived_intensity=data.get("perceived_intensity"),
            modulated_by=data.get("modulated_by", ""),
        )
