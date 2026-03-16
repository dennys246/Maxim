"""SalienceNetwork - Manages WHAT objects are salient in the visual scene.

A DataFrame-backed salience system that tracks:
- Object detections and their properties
- Novelty scores (new vs. familiar objects)
- Interest matching (user-specified interests)
- Temporal decay of salience

Designed for efficient queries by agents/LLMs via to_context_str().
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Use numpy for efficient array operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class SalienceConfig:
    """Configuration for SalienceNetwork.

    Attributes:
        max_tracked_objects: Maximum number of objects to track.
        novelty_decay_seconds: Time for novelty to decay from max to ~37%.
        recency_decay_seconds: Time for recency salience to decay.
        min_confidence: Minimum detection confidence to track.
        interest_labels: Labels that get boosted salience.
        interest_boost: Multiplier for interest-matched objects.
        min_salience_threshold: Minimum salience to consider object important.
    """
    max_tracked_objects: int = 100
    novelty_decay_seconds: float = 30.0
    recency_decay_seconds: float = 5.0
    min_confidence: float = 0.3
    interest_labels: set[str] = field(default_factory=lambda: {"person", "face"})
    interest_boost: float = 1.5
    min_salience_threshold: float = 0.2


@dataclass
class TrackedObject:
    """A tracked object in the scene (for non-numpy fallback)."""
    track_id: Any
    class_id: int
    label: str
    confidence: float
    first_seen: float
    last_seen: float
    times_seen: int
    position_u: float
    position_v: float
    bbox_w: float
    bbox_h: float
    interest_matched: bool = False


class SalienceNetwork:
    """DataFrame-backed salience system for object tracking.

    Maintains a collection of tracked objects with:
    - Novelty (new objects are more salient)
    - Recency (recently seen objects are more salient)
    - Interest matching (user-specified labels boost salience)
    - Confidence weighting

    Example:
        salience = SalienceNetwork()

        # Update from detections
        salience.update_from_detections(detections)

        # Get top salient objects
        top = salience.get_top_salient(n=5)

        # Get context for LLM
        context = salience.to_context_str()
    """

    def __init__(self, config: SalienceConfig | None = None) -> None:
        """Initialize SalienceNetwork."""
        self.config = config or SalienceConfig()
        self._lock = threading.RLock()

        # DataFrame-like storage using numpy arrays as columns
        max_n = self.config.max_tracked_objects

        if HAS_NUMPY:
            # Track IDs (object dtype for flexibility)
            self._track_ids: np.ndarray = np.empty(max_n, dtype=object)
            self._class_ids = np.zeros(max_n, dtype=np.int32)
            self._labels: np.ndarray = np.empty(max_n, dtype=object)
            self._confidence = np.zeros(max_n, dtype=np.float32)
            self._first_seen = np.zeros(max_n, dtype=np.float64)
            self._last_seen = np.zeros(max_n, dtype=np.float64)
            self._times_seen = np.zeros(max_n, dtype=np.int32)
            self._position_u = np.zeros(max_n, dtype=np.float32)
            self._position_v = np.zeros(max_n, dtype=np.float32)
            self._bbox_w = np.zeros(max_n, dtype=np.float32)
            self._bbox_h = np.zeros(max_n, dtype=np.float32)
            self._interest_matched = np.zeros(max_n, dtype=np.bool_)
            self._valid = np.zeros(max_n, dtype=np.bool_)  # Marks used slots
        else:
            # Fallback to dict-based storage
            self._objects: dict[Any, TrackedObject] = {}

        # Index for O(1) track_id lookup
        self._track_id_to_idx: dict[Any, int] = {}
        self._next_idx = 0
        self._object_count = 0

        # Statistics
        self._total_detections = 0
        self._total_unique_objects = 0
        self._interest_detections = 0

    def _find_slot(self) -> int:
        """Find next available slot, evicting oldest if full. O(n) worst case."""
        if HAS_NUMPY:
            if self._object_count < self.config.max_tracked_objects:
                # Find first invalid slot
                invalid = np.where(~self._valid)[0]
                if len(invalid) > 0:
                    return int(invalid[0])

            # Evict oldest (least recently seen)
            oldest_idx = int(np.argmin(np.where(self._valid, self._last_seen, np.inf)))
            old_track_id = self._track_ids[oldest_idx]
            if old_track_id in self._track_id_to_idx:
                del self._track_id_to_idx[old_track_id]
            self._valid[oldest_idx] = False
            self._object_count -= 1
            return oldest_idx
        else:
            # Dict storage doesn't need slots
            return -1

    def update_from_detections(
        self,
        detections: list[dict[str, Any]],
        timestamp: float | None = None,
    ) -> None:
        """Update salience from detection results.

        Args:
            detections: List of detection dicts with keys:
                - track_id (optional): Unique tracking ID
                - class_id: Object class ID
                - label: Human-readable label
                - confidence: Detection confidence
                - bbox: (x, y, w, h) bounding box
            timestamp: Detection timestamp (defaults to now).
        """
        now = timestamp or time.time()

        with self._lock:
            seen_ids: set[Any] = set()

            for det in detections:
                confidence = det.get("confidence", 0.5)
                if confidence < self.config.min_confidence:
                    continue

                track_id = det.get("track_id", id(det))
                class_id = det.get("class_id", 0)
                label = det.get("label", "unknown")
                bbox = det.get("bbox", (0, 0, 0, 0))

                # Compute position from bbox center
                pos_u = bbox[0] + bbox[2] / 2 if len(bbox) >= 4 else 0
                pos_v = bbox[1] + bbox[3] / 2 if len(bbox) >= 4 else 0

                is_interest = label.lower() in {lbl.lower() for lbl in self.config.interest_labels}

                seen_ids.add(track_id)
                self._total_detections += 1

                if HAS_NUMPY:
                    if track_id in self._track_id_to_idx:
                        # Update existing
                        idx = self._track_id_to_idx[track_id]
                        self._last_seen[idx] = now
                        self._times_seen[idx] += 1
                        self._confidence[idx] = confidence
                        self._position_u[idx] = pos_u
                        self._position_v[idx] = pos_v
                        self._bbox_w[idx] = bbox[2] if len(bbox) >= 4 else 0
                        self._bbox_h[idx] = bbox[3] if len(bbox) >= 4 else 0
                    else:
                        # New object
                        idx = self._find_slot()
                        self._track_ids[idx] = track_id
                        self._class_ids[idx] = class_id
                        self._labels[idx] = label
                        self._confidence[idx] = confidence
                        self._first_seen[idx] = now
                        self._last_seen[idx] = now
                        self._times_seen[idx] = 1
                        self._position_u[idx] = pos_u
                        self._position_v[idx] = pos_v
                        self._bbox_w[idx] = bbox[2] if len(bbox) >= 4 else 0
                        self._bbox_h[idx] = bbox[3] if len(bbox) >= 4 else 0
                        self._interest_matched[idx] = is_interest
                        self._valid[idx] = True
                        self._track_id_to_idx[track_id] = idx
                        self._object_count += 1
                        self._total_unique_objects += 1

                        if is_interest:
                            self._interest_detections += 1
                else:
                    if track_id in self._objects:
                        obj = self._objects[track_id]
                        obj.last_seen = now
                        obj.times_seen += 1
                        obj.confidence = confidence
                        obj.position_u = pos_u
                        obj.position_v = pos_v
                        obj.bbox_w = bbox[2] if len(bbox) >= 4 else 0
                        obj.bbox_h = bbox[3] if len(bbox) >= 4 else 0
                    else:
                        self._objects[track_id] = TrackedObject(
                            track_id=track_id,
                            class_id=class_id,
                            label=label,
                            confidence=confidence,
                            first_seen=now,
                            last_seen=now,
                            times_seen=1,
                            position_u=pos_u,
                            position_v=pos_v,
                            bbox_w=bbox[2] if len(bbox) >= 4 else 0,
                            bbox_h=bbox[3] if len(bbox) >= 4 else 0,
                            interest_matched=is_interest,
                        )
                        self._total_unique_objects += 1
                        if is_interest:
                            self._interest_detections += 1

                        # Evict oldest if over limit
                        if len(self._objects) > self.config.max_tracked_objects:
                            oldest_id = min(
                                self._objects.keys(),
                                key=lambda k: self._objects[k].last_seen
                            )
                            del self._objects[oldest_id]

    def get_salience(self, track_id: Any) -> float:
        """Get salience score for a tracked object.

        Salience combines:
        - Novelty (new objects = higher)
        - Recency (recently seen = higher)
        - Interest match (boosted if matching interest labels)
        - Confidence

        Returns:
            Salience score in [0, 1+] (can exceed 1 with interest boost).
        """
        now = time.time()

        with self._lock:
            if HAS_NUMPY:
                if track_id not in self._track_id_to_idx:
                    return 0.0
                idx = self._track_id_to_idx[track_id]
                if not self._valid[idx]:
                    return 0.0

                first_seen = float(self._first_seen[idx])
                last_seen = float(self._last_seen[idx])
                confidence = float(self._confidence[idx])
                interest = bool(self._interest_matched[idx])
            else:
                if track_id not in self._objects:
                    return 0.0
                obj = self._objects[track_id]
                first_seen = obj.first_seen
                last_seen = obj.last_seen
                confidence = obj.confidence
                interest = obj.interest_matched

            # Novelty: decays from 1.0 based on time since first seen
            time_known = now - first_seen
            novelty = math.exp(-time_known / self.config.novelty_decay_seconds)

            # Recency: decays from 1.0 based on time since last seen
            time_since_seen = now - last_seen
            recency = math.exp(-time_since_seen / self.config.recency_decay_seconds)

            # Combined salience
            salience = (novelty * 0.3 + recency * 0.5 + confidence * 0.2)

            # Apply interest boost
            if interest:
                salience *= self.config.interest_boost

            return salience

    def get_novelty(self, track_id: Any) -> float:
        """Get novelty score for an object (1=never seen, 0=very familiar)."""
        now = time.time()

        with self._lock:
            if HAS_NUMPY:
                if track_id not in self._track_id_to_idx:
                    return 1.0  # Unknown = fully novel
                idx = self._track_id_to_idx[track_id]
                if not self._valid[idx]:
                    return 1.0
                first_seen = float(self._first_seen[idx])
            else:
                if track_id not in self._objects:
                    return 1.0
                first_seen = self._objects[track_id].first_seen

            time_known = now - first_seen
            return math.exp(-time_known / self.config.novelty_decay_seconds)

    def get_top_salient(
        self,
        n: int = 5,
        min_salience: float | None = None,
    ) -> list[dict[str, Any]]:
        """Get top N most salient objects.

        Args:
            n: Number of objects to return.
            min_salience: Minimum salience threshold (defaults to config).

        Returns:
            List of object dicts sorted by salience (highest first).
        """
        threshold = min_salience if min_salience is not None else self.config.min_salience_threshold

        with self._lock:
            candidates: list[tuple[float, dict[str, Any]]] = []

            if HAS_NUMPY:
                valid_indices = np.where(self._valid)[0]
                for idx in valid_indices:
                    track_id = self._track_ids[idx]
                    salience = self.get_salience(track_id)
                    if salience < threshold:
                        continue

                    candidates.append((salience, {
                        "track_id": track_id,
                        "class_id": int(self._class_ids[idx]),
                        "label": str(self._labels[idx]),
                        "confidence": float(self._confidence[idx]),
                        "salience": salience,
                        "novelty": self.get_novelty(track_id),
                        "position": (float(self._position_u[idx]), float(self._position_v[idx])),
                        "bbox_size": (float(self._bbox_w[idx]), float(self._bbox_h[idx])),
                        "times_seen": int(self._times_seen[idx]),
                        "interest_matched": bool(self._interest_matched[idx]),
                    }))
            else:
                for track_id, obj in self._objects.items():
                    salience = self.get_salience(track_id)
                    if salience < threshold:
                        continue

                    candidates.append((salience, {
                        "track_id": track_id,
                        "class_id": obj.class_id,
                        "label": obj.label,
                        "confidence": obj.confidence,
                        "salience": salience,
                        "novelty": self.get_novelty(track_id),
                        "position": (obj.position_u, obj.position_v),
                        "bbox_size": (obj.bbox_w, obj.bbox_h),
                        "times_seen": obj.times_seen,
                        "interest_matched": obj.interest_matched,
                    }))

            # Sort by salience descending
            candidates.sort(key=lambda x: x[0], reverse=True)

            return [c[1] for c in candidates[:n]]

    def get_objects_at_position(
        self,
        position: tuple[float, float],
        radius: float = 50.0,
    ) -> list[dict[str, Any]]:
        """Get objects near a specific position.

        Args:
            position: (u, v) pixel coordinates.
            radius: Search radius in pixels.

        Returns:
            List of objects within radius, sorted by distance.
        """
        target_u, target_v = position

        with self._lock:
            results: list[tuple[float, dict[str, Any]]] = []

            if HAS_NUMPY:
                valid_indices = np.where(self._valid)[0]
                for idx in valid_indices:
                    pos_u = float(self._position_u[idx])
                    pos_v = float(self._position_v[idx])
                    dist = math.sqrt((pos_u - target_u)**2 + (pos_v - target_v)**2)

                    if dist <= radius:
                        track_id = self._track_ids[idx]
                        results.append((dist, {
                            "track_id": track_id,
                            "label": str(self._labels[idx]),
                            "position": (pos_u, pos_v),
                            "distance": dist,
                            "salience": self.get_salience(track_id),
                        }))
            else:
                for track_id, obj in self._objects.items():
                    dist = math.sqrt((obj.position_u - target_u)**2 + (obj.position_v - target_v)**2)
                    if dist <= radius:
                        results.append((dist, {
                            "track_id": track_id,
                            "label": obj.label,
                            "position": (obj.position_u, obj.position_v),
                            "distance": dist,
                            "salience": self.get_salience(track_id),
                        }))

            results.sort(key=lambda x: x[0])
            return [r[1] for r in results]

    # ─────────────────────────────────────────────────────────────────────────
    # DataFrame-like query interface
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        min_salience: float = 0.0,
        min_confidence: float = 0.0,
        labels: set[str] | None = None,
        interest_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Query tracked objects with filters.

        Args:
            min_salience: Minimum salience score.
            min_confidence: Minimum detection confidence.
            labels: Only include objects with these labels.
            interest_only: Only include interest-matched objects.

        Returns:
            List of object dicts matching filters.
        """
        with self._lock:
            results: list[dict[str, Any]] = []

            if HAS_NUMPY:
                valid_indices = np.where(self._valid)[0]
                for idx in valid_indices:
                    track_id = self._track_ids[idx]
                    label = str(self._labels[idx])
                    confidence = float(self._confidence[idx])
                    interest = bool(self._interest_matched[idx])
                    salience = self.get_salience(track_id)

                    # Apply filters
                    if salience < min_salience:
                        continue
                    if confidence < min_confidence:
                        continue
                    if labels is not None and label.lower() not in {lbl.lower() for lbl in labels}:
                        continue
                    if interest_only and not interest:
                        continue

                    results.append({
                        "track_id": track_id,
                        "class_id": int(self._class_ids[idx]),
                        "label": label,
                        "confidence": confidence,
                        "salience": salience,
                        "novelty": self.get_novelty(track_id),
                        "position": (float(self._position_u[idx]), float(self._position_v[idx])),
                        "bbox_size": (float(self._bbox_w[idx]), float(self._bbox_h[idx])),
                        "times_seen": int(self._times_seen[idx]),
                        "interest_matched": interest,
                        "first_seen": float(self._first_seen[idx]),
                        "last_seen": float(self._last_seen[idx]),
                    })
            else:
                for track_id, obj in self._objects.items():
                    salience = self.get_salience(track_id)

                    if salience < min_salience:
                        continue
                    if obj.confidence < min_confidence:
                        continue
                    if labels is not None and obj.label.lower() not in {lbl.lower() for lbl in labels}:
                        continue
                    if interest_only and not obj.interest_matched:
                        continue

                    results.append({
                        "track_id": track_id,
                        "class_id": obj.class_id,
                        "label": obj.label,
                        "confidence": obj.confidence,
                        "salience": salience,
                        "novelty": self.get_novelty(track_id),
                        "position": (obj.position_u, obj.position_v),
                        "bbox_size": (obj.bbox_w, obj.bbox_h),
                        "times_seen": obj.times_seen,
                        "interest_matched": obj.interest_matched,
                        "first_seen": obj.first_seen,
                        "last_seen": obj.last_seen,
                    })

            return results

    def to_context_str(self, max_items: int = 5) -> str:
        """Generate concise context string for LLM consumption.

        Returns a human-readable summary of salient objects.
        """
        time.time()

        with self._lock:
            lines = ["[SALIENCE STATE]"]

            # Get top salient objects
            top = self.get_top_salient(n=max_items)

            if not top:
                lines.append("No salient objects detected")
            else:
                lines.append(f"Top salient objects ({len(top)}):")
                for obj in top:
                    marker = "*" if obj.get("interest_matched") else " "
                    pos = obj["position"]
                    lines.append(
                        f" {marker}{obj['label']} at ({pos[0]:.0f},{pos[1]:.0f}) "
                        f"sal={obj['salience']:.2f} nov={obj['novelty']:.2f}"
                    )

            # Interest summary
            interest_objs = self.query(interest_only=True)
            if interest_objs:
                labels = {}
                for obj in interest_objs:
                    label = obj["label"]
                    labels[label] = labels.get(label, 0) + 1
                interest_summary = ", ".join(f"{v}x {k}" for k, v in labels.items())
                lines.append(f"Interest matches: {interest_summary}")

            # Stats
            lines.append(
                f"Stats: {self._total_unique_objects} unique objects, "
                f"{self._interest_detections} interest matches"
            )

            return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """Get network statistics."""
        with self._lock:
            return {
                "total_detections": self._total_detections,
                "total_unique_objects": self._total_unique_objects,
                "interest_detections": self._interest_detections,
                "current_object_count": self._object_count if HAS_NUMPY else len(self._objects),
                "max_tracked_objects": self.config.max_tracked_objects,
            }

    def clear(self) -> None:
        """Clear all tracked objects."""
        with self._lock:
            if HAS_NUMPY:
                self._valid.fill(False)
                self._times_seen.fill(0)
            else:
                self._objects.clear()

            self._track_id_to_idx.clear()
            self._object_count = 0
            self._total_detections = 0
            self._total_unique_objects = 0
            self._interest_detections = 0


__all__ = [
    "SalienceNetwork",
    "SalienceConfig",
]
