"""Narrative Percept Transcriber — convert text to structured detections.

Converts narrative scene descriptions into structured ``SalienceItem``
objects with ``NarrativeWhere`` coordinates, enabling the full bio-stack
(SalienceNetwork, NoveltyTracker, AttentionNetwork) to process narrative
content without faking pixel coordinates.

Uses the small-tier LLM (smollm 1.7B) for entity extraction when
available, with regex/keyword fallback when no LLM is configured.

Example::

    transcriber = NarrativeTranscriber(router=llm_router)
    items = transcriber.transcribe_to_items(
        "A massive silver elm with a stone door and a carved face",
        scene="forest_clearing",
    )
    # Returns: [
    #   SalienceItem(id="silver_elm_0", label="silver_elm",
    #                where=NarrativeWhere("center", "forest_clearing"), ...),
    #   SalienceItem(id="stone_door_1", label="stone_door",
    #                where=NarrativeWhere("center", "forest_clearing"), ...),
    # ]

    # Legacy dict format still available via transcribe()
    detections = transcriber.transcribe("A guard at the entrance")
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Narrative class IDs start at 900 to avoid collision with COCO classes (0-80)
_NEXT_CLASS_ID = 900
_class_registry: dict[str, int] = {}
_class_registry_lock = threading.Lock()

# Legacy pixel position map — kept for backward compatibility with
# code that still calls transcribe() and expects bbox_xyxy dicts.
# New code should use transcribe_to_items() which produces NarrativeWhere.
_POSITION_MAP = {
    "left": [50, 100, 250, 400],
    "center": [200, 50, 440, 430],
    "right": [400, 100, 600, 400],
    "background": [100, 50, 540, 300],
    "center-bottom": [200, 250, 440, 450],
    "center-top": [200, 30, 440, 250],
}


def _get_class_id(label: str) -> int:
    """Get or assign a stable class ID for a narrative entity label.

    Thread-safe: multiple agents may call this concurrently.
    """
    global _NEXT_CLASS_ID
    with _class_registry_lock:
        if label not in _class_registry:
            _class_registry[label] = _NEXT_CLASS_ID
            _NEXT_CLASS_ID += 1
        return _class_registry[label]


def _position_to_bbox(hint: str) -> list[int]:
    """Convert a spatial hint to a bounding box [x1, y1, x2, y2]."""
    return _POSITION_MAP.get(hint, _POSITION_MAP["center"])


class NarrativeTranscriber:
    """Convert narrative text into structured perceptual detections.

    When an LLM router is provided, uses the small tier for entity
    extraction (richer, handles implicit entities). Falls back to
    regex-based noun phrase extraction when no LLM is available.
    """

    def __init__(self, router: Any = None, *, function: str = "narrative_transcription") -> None:
        self._router = router
        self._function = function
        self._entity_ids: dict[str, str] = {}

    def transcribe(self, text: str) -> list[dict[str, Any]]:
        """Extract structured detections from narrative text.

        Returns a list of detection dicts compatible with
        ``SalienceNetwork.update_from_detections()``.
        """
        if not text or not text.strip():
            return []

        # Try LLM extraction first (richer, handles implicit entities)
        if self._router is not None:
            try:
                return self._transcribe_llm(text)
            except Exception as e:
                logger.debug("LLM transcription failed, using regex fallback: %s", e)

        # Regex fallback — extract capitalized names and quoted phrases
        return self._transcribe_regex(text)

    def _transcribe_llm(self, text: str) -> list[dict[str, Any]]:
        """Use small-tier LLM for entity extraction."""
        result = self._router.generate_json(
            f"Extract entities from this scene description. "
            f"Return a JSON list of objects with: label (snake_case), "
            f"type (object/character/sound/location), "
            f"confidence (0.0-1.0), spatial_hint (left/center/right/background).\n\n"
            f"Scene: {text}",
            function=self._function,
            max_tokens=200,
        )
        if not isinstance(result, list):
            return []
        return self._to_detections(result)

    def _transcribe_regex(self, text: str) -> list[dict[str, Any]]:
        """Regex fallback — extract proper nouns and notable noun phrases."""
        entities: list[dict[str, Any]] = []

        # Extract capitalized proper nouns (character/place names)
        proper_nouns = re.findall(r"\b([A-Z][a-z]{2,})\b", text)
        # Deduplicate while preserving order
        seen: set[str] = set()
        for noun in proper_nouns:
            lower = noun.lower()
            if lower not in seen and lower not in {"the", "you", "your", "this", "that", "what"}:
                seen.add(lower)
                entities.append(
                    {
                        "label": lower,
                        "type": "character",
                        "confidence": 0.7,
                        "spatial_hint": "center",
                    }
                )

        # Extract quoted speech (indicates active NPC)
        quotes = re.findall(r'"([^"]{5,50})"', text)
        for q in quotes[:2]:
            label = "speech_" + re.sub(r"[^a-z]", "_", q[:20].lower()).strip("_")
            if label not in seen:
                seen.add(label)
                entities.append(
                    {
                        "label": label,
                        "type": "sound",
                        "confidence": 0.5,
                        "spatial_hint": "center",
                    }
                )

        return self._to_detections(entities)

    def _to_detections(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert parsed entities to SalienceNetwork-compatible detection dicts."""
        detections = []
        for entity in entities:
            label = entity.get("label", "unknown")
            track_id = self._get_stable_id(label)
            class_id = _get_class_id(label)
            spatial_hint = entity.get("spatial_hint", "center")
            detections.append(
                {
                    "track_id": track_id,
                    "class_id": class_id,
                    "label": label,
                    "conf": entity.get("confidence", 0.5),
                    "bbox_xyxy": _position_to_bbox(spatial_hint),
                    "_spatial_hint": spatial_hint,  # Preserved for transcribe_to_items()
                }
            )
        return detections

    def _get_stable_id(self, label: str) -> str:
        """Return a stable track_id for an entity across turns.

        Same entity label always gets the same track_id, enabling
        NoveltyTracker to correctly compute novelty decay when
        entities reappear in later turns.
        """
        if label not in self._entity_ids:
            self._entity_ids[label] = f"{label}_{len(self._entity_ids)}"
        return self._entity_ids[label]

    def transcribe_to_items(
        self,
        text: str,
        *,
        scene: str = "",
    ) -> list:
        """Extract structured SalienceItems from narrative text.

        Produces ``SalienceItem`` objects with ``NarrativeWhere`` coordinates
        instead of fake pixel bounding boxes.  This is the preferred path
        for simulation/DM runtime code.

        Args:
            text: Narrative scene description.
            scene: Scene identifier (e.g., encounter name) for the
                ``NarrativeWhere`` coordinate.

        Returns:
            List of ``SalienceItem`` instances.
        """
        from maxim.salience.protocols import PerceptSource, SalienceItem
        from maxim.salience.where import NarrativeWhere

        if not text or not text.strip():
            return []

        # Use existing extraction pipeline (LLM or regex)
        if self._router is not None:
            try:
                entities = self._transcribe_llm(text)
            except Exception as e:
                logger.debug("LLM transcription failed, using regex fallback: %s", e)
                entities = self._transcribe_regex(text)
        else:
            entities = self._transcribe_regex(text)

        # Convert to raw entity dicts (pre-detection format)
        # _transcribe_llm/_transcribe_regex return detection dicts via _to_detections,
        # but we want the intermediate entity list. Re-extract from text.
        items = []
        for det in entities:
            label = det.get("label", "unknown")
            track_id = det.get("track_id", self._get_stable_id(label))
            class_id = det.get("class_id", _get_class_id(label))
            confidence = det.get("conf", det.get("confidence", 0.5))

            # Use spatial_hint if available from the entity dict, else "center"
            position = "center"
            # Check if the detection was built from an entity with spatial_hint
            spatial = det.get("_spatial_hint", "center")
            if spatial:
                position = spatial

            where = NarrativeWhere(position=position, scene=scene)

            items.append(
                SalienceItem(
                    id=track_id,
                    label=label,
                    source=PerceptSource.NARRATIVE,
                    confidence=confidence,
                    where=where,
                    class_id=class_id,
                )
            )

        return items

    def reset(self) -> None:
        """Reset entity tracking (for new benchmark run)."""
        self._entity_ids.clear()
