"""Backward compatibility - re-exports from segment_vision.py."""

from maxim.inference.segment_vision import (
    DEFAULT_CLASS_WEIGHTS,
    NoveltyTracker,
    display_detections,
    get_default_novelty_tracker,
    passive_listening,
    passive_observation,
    score_detection_weighted,
)

__all__ = [
    "DEFAULT_CLASS_WEIGHTS",
    "NoveltyTracker",
    "display_detections",
    "get_default_novelty_tracker",
    "passive_listening",
    "passive_observation",
    "score_detection_weighted",
]
