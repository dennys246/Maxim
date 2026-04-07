"""Narrative Percept Transcriber — convert text to structured detections.

Converts narrative scene descriptions into the same structured detection
format that the camera/YOLO pipeline produces, enabling the full bio-stack
(SalienceNetwork, NoveltyTracker, AttentionNetwork) to process narrative
content.

Uses the small-tier LLM (smollm 1.7B) for entity extraction when
available, with regex/keyword fallback when no LLM is configured.

Example::

    transcriber = NarrativeTranscriber(router=llm_router)
    detections = transcriber.transcribe(
        "A massive silver elm with a stone door and a carved face"
    )
    # Returns: [
    #   {"track_id": "silver_elm_0", "class_id": 900, "label": "silver_elm", ...},
    #   {"track_id": "stone_door_1", "class_id": 901, "label": "stone_door", ...},
    # ]
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Narrative class IDs start at 900 to avoid collision with COCO classes (0-80)
_NEXT_CLASS_ID = 900
_class_registry: dict[str, int] = {}

# Spatial positions mapped to approximate pixel coordinates (640x480 frame)
_POSITION_MAP = {
    "left": [50, 100, 250, 400],
    "center": [200, 50, 440, 430],
    "right": [400, 100, 600, 400],
    "background": [100, 50, 540, 300],
    "center-bottom": [200, 250, 440, 450],
    "center-top": [200, 30, 440, 250],
}


def _get_class_id(label: str) -> int:
    """Get or assign a stable class ID for a narrative entity label."""
    global _NEXT_CLASS_ID
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
            detections.append(
                {
                    "track_id": track_id,
                    "class_id": class_id,
                    "label": label,
                    "conf": entity.get("confidence", 0.5),
                    "bbox_xyxy": _position_to_bbox(entity.get("spatial_hint", "center")),
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

    def reset(self) -> None:
        """Reset entity tracking (for new benchmark run)."""
        self._entity_ids.clear()
