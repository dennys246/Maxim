"""Vision observation and display utilities.

Provides display and scoring functions for object detection:
- display_detections(): Pure display function for annotated frames
- passive_observation(): Display-only observation (no movement)
- NoveltyTracker: Tracks object novelty for attention prioritization
- score_detection_weighted(): Weighted scoring combining conf, area, class, novelty

Movement decisions are handled by the agentic runtime, not this module.
"""

import math
import os
import time
from typing import Any

import numpy as np

from maxim.data.camera.display import ensure_bgr, show_frame
from maxim.utils.detections import score_detection_conf_area
from maxim.utils.logging import warn

try:
    from scipy.io.wavfile import write as write_wav
except ImportError:
    write_wav = None


# Default class weights (can be overridden dynamically)
# Higher = more likely to be selected as primary target
DEFAULT_CLASS_WEIGHTS: dict[int, float] = {
    0: 1.2,   # person - slight preference for social interaction
    # All other classes default to 1.0
}


class NoveltyTracker:
    """Tracks object novelty based on focus recency.

    Objects that haven't been focused recently get a novelty boost.
    When focused, novelty slowly decays over focus_decay_seconds.
    When not focused (even if still in frame), novelty recovers over recovery_seconds.
    """

    def __init__(
        self,
        focus_decay_seconds: float = 10.0,
        recovery_seconds: float = 20.0,
        max_novelty: float = 2.0,
        min_novelty: float = 0.5,
        cleanup_interval: int = 50,
        max_entries: int = 1000,
        max_age: float = 60.0,
        adaptive_cleanup: bool = True,
        *,
        decay_seconds: float | None = None,  # Backward compatibility alias
    ):
        """Initialize novelty tracker.

        Args:
            focus_decay_seconds: Time for novelty to decay from max to min while focused
            recovery_seconds: Time for novelty to recover from min to ~63% when not focused
            max_novelty: Novelty score for never-seen objects
            min_novelty: Minimum novelty score for frequently-focused objects
            cleanup_interval: How often to clean up old entries (default 50)
            max_entries: Hard cap on tracked entries to prevent unbounded growth
            max_age: Max age in seconds before entries are cleaned up
            adaptive_cleanup: If True, cleanup more frequently when near capacity
            decay_seconds: Deprecated alias for recovery_seconds (backward compatibility)
        """
        # Handle backward compatibility: if old decay_seconds is passed, use it for recovery
        if decay_seconds is not None:
            recovery_seconds = decay_seconds

        self.focus_decay_seconds = focus_decay_seconds
        self.recovery_seconds = recovery_seconds
        # Keep decay_seconds as alias for backward compatibility
        self.decay_seconds = recovery_seconds
        self.max_novelty = max_novelty
        self.min_novelty = min_novelty
        self.cleanup_interval = cleanup_interval
        self._adaptive_cleanup = adaptive_cleanup

        # track_id -> last_seen_timestamp (for cleanup/memory management)
        self._track_times: dict[Any, float] = {}
        # track_id -> last_focused_timestamp (for novelty calculation)
        self._focus_times: dict[Any, float] = {}
        # Counter for cleanup scheduling
        self._update_count = 0
        # Max age before cleanup (reduced from 300s to prevent memory bloat)
        self._max_age = max_age
        # Hard cap on entries to prevent unbounded growth
        self._max_entries = max_entries

    def get_novelty(self, track_id: Any) -> float:
        """Get novelty score for a track_id.

        Returns max_novelty for new objects. While focused (recently attended to),
        novelty stays at max. After not being focused for focus_decay_seconds,
        novelty decays. After a long absence, novelty recovers.
        """
        if track_id is None:
            return self.max_novelty  # Unknown objects are novel

        now = time.time()
        last_focused = self._focus_times.get(track_id)

        if last_focused is None:
            # Never focused before - maximum novelty
            return self.max_novelty

        time_since_focus = now - last_focused

        if time_since_focus <= self.focus_decay_seconds:
            # Currently focused or recently focused - novelty stays at max
            # This keeps objects interesting while being attended to
            return self.max_novelty
        else:
            # Not focused - decay from max toward min exponentially (habituation)
            # Decay starts after focus_decay_seconds grace period
            decay_time = time_since_focus - self.focus_decay_seconds
            decay_factor = math.exp(-decay_time / self.recovery_seconds)
            # decay_factor: 1.0 at start, approaches 0 over time
            novelty = self.min_novelty + (self.max_novelty - self.min_novelty) * decay_factor
            return novelty

    def focus(self, track_id: Any) -> None:
        """Mark a track_id as the current focus target.

        This resets novelty to max and starts the slow decay.
        Call this when the object is being actively attended to.
        """
        if track_id is None:
            return
        now = time.time()
        self._focus_times[track_id] = now
        self._track_times[track_id] = now  # Also update for cleanup tracking

    def update(self, track_id: Any) -> None:
        """Mark a track_id as seen in frame (but not focused).

        This only updates tracking for cleanup purposes.
        Novelty is NOT affected - call focus() to affect novelty.
        """
        if track_id is None:
            return

        self._track_times[track_id] = time.time()
        self._update_count += 1

        # Adaptive cleanup: when over 80% capacity, cleanup every update
        if self._adaptive_cleanup and len(self._track_times) > self._max_entries * 0.8:
            self._cleanup()
            self._update_count = 0
        # Periodic cleanup of stale entries
        elif self._update_count >= self.cleanup_interval:
            self._cleanup()
            self._update_count = 0

    def update_batch(self, track_ids: list[Any]) -> None:
        """Mark multiple track_ids as seen in frame (but not focused).

        This only updates tracking for cleanup purposes.
        Novelty is NOT affected - call focus() to affect novelty.
        """
        now = time.time()
        for tid in track_ids:
            if tid is not None:
                self._track_times[tid] = now

        self._update_count += len(track_ids)

        # Adaptive cleanup: when over 80% capacity, cleanup every update
        if self._adaptive_cleanup and len(self._track_times) > self._max_entries * 0.8:
            self._cleanup()
            self._update_count = 0
        elif self._update_count >= self.cleanup_interval:
            self._cleanup()
            self._update_count = 0

    def forget(self, track_id: Any) -> bool:
        """Immediately forget a track_id (mark as non-salient).

        Use this to quickly remove entries that are no longer of interest.

        Args:
            track_id: The track_id to forget.

        Returns:
            True if the track_id was found and removed.
        """
        if track_id is None:
            return False
        return self._track_times.pop(track_id, None) is not None

    def forget_batch(self, track_ids: list[Any]) -> int:
        """Immediately forget multiple track_ids.

        Args:
            track_ids: List of track_ids to forget.

        Returns:
            Number of entries that were actually removed.
        """
        removed = 0
        for tid in track_ids:
            if tid is not None and self._track_times.pop(tid, None) is not None:
                removed += 1
        return removed

    def force_cleanup(self) -> int:
        """Force an immediate cleanup regardless of update count.

        Returns:
            Number of entries before cleanup (for diagnostics).
        """
        count_before = len(self._track_times)
        self._cleanup()
        self._update_count = 0
        return count_before

    def _cleanup(self) -> None:
        """Remove entries older than max_age and enforce hard cap.

        More aggressive cleanup: removes stale entries first, then if still
        over the hard cap, removes oldest entries until under the limit.
        """
        now = time.time()
        cutoff = now - self._max_age

        # First pass: remove entries older than max_age
        self._track_times = {
            k: v for k, v in self._track_times.items() if v > cutoff
        }
        self._focus_times = {
            k: v for k, v in self._focus_times.items() if v > cutoff
        }

        # Second pass: if still over hard cap, remove oldest entries
        if len(self._track_times) > self._max_entries:
            # Sort by timestamp (oldest first) and keep only the newest entries
            sorted_items = sorted(self._track_times.items(), key=lambda x: x[1])
            excess = len(sorted_items) - self._max_entries
            # Keep only the most recent entries
            keys_to_remove = {k for k, _ in sorted_items[:excess]}
            self._track_times = dict(sorted_items[excess:])
            # Also remove from focus_times
            self._focus_times = {
                k: v for k, v in self._focus_times.items() if k not in keys_to_remove
            }

    @property
    def tracked_count(self) -> int:
        """Return number of currently tracked entries."""
        return len(self._track_times)


def score_detection_weighted(
    det: tuple,
    novelty_tracker: NoveltyTracker | None = None,
    class_weights: dict[int, float] | None = None,
) -> tuple[float, float]:
    """Score a detection combining confidence, area, class weight, and novelty.

    Args:
        det: Detection tuple (track_id, frame_ind, x1, y1, x2, y2, conf, cls_id, ...)
        novelty_tracker: Optional tracker for novelty scoring
        class_weights: Optional dict of class_id -> weight multiplier

    Returns:
        Tuple of (weighted_conf, weighted_area) for comparison (higher = more important)
    """
    # Base score from confidence and area - returns (conf, area) tuple
    conf, area = score_detection_conf_area(det)

    # Class weight
    class_weight = 1.0
    if class_weights and len(det) > 7 and det[7] is not None:
        try:
            class_id = int(det[7])
            class_weight = class_weights.get(class_id, 1.0)
        except (ValueError, TypeError):
            pass

    # Novelty boost
    novelty = 1.0
    if novelty_tracker and len(det) > 0:
        track_id = det[0]
        novelty = novelty_tracker.get_novelty(track_id)

    # Apply weights to both conf and area
    weight = class_weight * novelty
    return (conf * weight, area * weight)


# Global default novelty tracker (can be overridden per-maxim instance)
_default_novelty_tracker: NoveltyTracker | None = None


def get_default_novelty_tracker() -> NoveltyTracker:
    """Get or create the default novelty tracker."""
    global _default_novelty_tracker
    if _default_novelty_tracker is None:
        _default_novelty_tracker = NoveltyTracker()
    return _default_novelty_tracker


def _normalize_detection(det: Any) -> tuple | None:
    """Normalize detection from various formats to tuple format.

    Supports:
    - Tuple/list: (track_id, frame_ind, x1, y1, x2, y2, conf, cls_id, ...)
    - Dict: {"track_id": ..., "bbox_xyxy": [x1, y1, x2, y2], "conf": ..., "class_id": ...}

    Returns:
        Tuple in standard format (track_id, 0, x1, y1, x2, y2, conf, cls_id)
    """
    if isinstance(det, (tuple, list)) and len(det) >= 8:
        return tuple(det)

    if isinstance(det, dict):
        try:
            bbox = det.get("bbox_xyxy", [0, 0, 0, 0])
            return (
                det.get("track_id"),
                0,  # frame_ind placeholder
                float(bbox[0]) if len(bbox) > 0 else 0.0,
                float(bbox[1]) if len(bbox) > 1 else 0.0,
                float(bbox[2]) if len(bbox) > 2 else 0.0,
                float(bbox[3]) if len(bbox) > 3 else 0.0,
                float(det.get("conf", 0.0)),
                det.get("class_id"),
            )
        except Exception:
            return None

    return None


def display_detections(
    frame: np.ndarray,
    detections: list[Any],
    *,
    segmenter: Any = None,
    window_name: str = "Maxim",
    wait_ms: int = 1,
    show_pose: bool = True,
    target_class_id: int | None = None,
    novelty_tracker: NoveltyTracker | None = None,
    class_weights: dict[int, float] | None = None,
) -> dict[str, Any] | None:
    """Display frame with detection annotations.

    This is the pure display function - no movement, no side effects.
    Returns detection info for the agentic system to act on.

    Uses weighted scoring combining confidence, area, class weights, and novelty.
    Novel objects (new track_ids) are prioritized over familiar ones.

    Args:
        frame: BGR image frame
        detections: List of detection tuples from segmenter
        segmenter: Optional segmenter for pose refinement
        window_name: OpenCV window name
        wait_ms: waitKey delay
        show_pose: Whether to compute and display pose info
        target_class_id: If specified, filter to only this class
        novelty_tracker: Optional tracker for novelty-based scoring
        class_weights: Optional class_id -> weight multiplier dict

    Returns:
        Dict with target info if a primary target was identified, else None
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return None

    frame = ensure_bgr(frame)
    height, width = frame.shape[:2]

    if not detections:
        show_frame(frame, window_name=window_name, wait_ms=wait_ms)
        return None

    # Normalize detections to tuple format (supports both tuple and dict inputs)
    normalized = []
    for det in detections:
        norm = _normalize_detection(det)
        if norm is not None:
            normalized.append(norm)

    if not normalized:
        show_frame(frame, window_name=window_name, wait_ms=wait_ms)
        return None

    detections = normalized

    # If target_class_id is specified, filter to only that class
    if target_class_id is not None:
        class_filtered = []
        for det in detections:
            try:
                if len(det) > 7 and det[7] is not None and int(det[7]) == target_class_id:
                    class_filtered.append(det)
            except Exception:
                continue
        if class_filtered:
            detections = class_filtered
        # If no detections of target class, fall through to use all detections

    if not detections:
        show_frame(frame, window_name=window_name, wait_ms=wait_ms)
        return None

    # Use default novelty tracker if none provided
    tracker = novelty_tracker if novelty_tracker is not None else get_default_novelty_tracker()
    weights = class_weights if class_weights is not None else DEFAULT_CLASS_WEIGHTS

    # Create scoring function that combines conf, area, class weight, and novelty
    def score_fn(det: tuple) -> float:
        return score_detection_weighted(det, tracker, weights)

    # Find primary target using weighted scoring
    primary = max(detections, key=score_fn)

    # Update novelty tracker with all seen detections
    # This ensures frequently-seen objects decay in novelty
    if tracker:
        track_ids = [det[0] for det in detections if len(det) > 0]
        tracker.update_batch(track_ids)

    x1, y1, x2, y2 = primary[2], primary[3], primary[4], primary[5]
    is_person = False
    try:
        is_person = len(primary) > 7 and int(primary[7]) == 0
    except Exception:
        pass

    # Target point (center of bbox by default)
    u = float((x1 + x2) / 2)
    v = float((y1 + y2) / 2)
    target_method = "bbox"
    pose_box = None
    pose_info = None

    # Refine with pose if available
    if is_person and show_pose and segmenter is not None:
        try:
            pose_info = segmenter.pose_targets_for_box(frame, (x1, y1, x2, y2))
            if pose_info and "target" in pose_info:
                u, v = map(float, pose_info["target"])
                target_method = str(pose_info.get("method", "pose"))
                pose_box = pose_info.get("pose_box")
        except Exception:
            pass

    u_int = int(np.clip(round(u), 1, width - 1))
    v_int = int(np.clip(round(v), 1, height - 1))

    # Build display annotations
    boxes = list(detections)
    text_lines = []

    if is_person:
        text_lines.append(f"target: {target_method}")
        if pose_info and pose_info.get("iou") is not None:
            try:
                text_lines.append(f"pose iou: {float(pose_info['iou']):.2f}")
            except Exception:
                pass

    if pose_box is not None:
        try:
            px1, py1, px2, py2 = pose_box
            boxes.append({"x1": px1, "y1": py1, "x2": px2, "y2": py2, "label": "pose"})
        except Exception:
            pass

    target_box = (x1, y1, x2, y2)
    if pose_box is not None:
        try:
            target_box = tuple(map(float, pose_box))
        except Exception:
            pass

    # Display
    show_frame(
        frame,
        boxes=boxes,
        target_box=target_box,
        center=(width / 2, height / 2),
        target_point=(u_int, v_int),
        text_lines=text_lines if text_lines else None,
        window_name=window_name,
        wait_ms=wait_ms,
    )

    # Compute novelty for the selected target (for reporting)
    target_novelty = 1.0
    if tracker and len(primary) > 0:
        target_novelty = tracker.get_novelty(primary[0])

    # Return target info for agentic system
    return {
        "target_u": u_int,
        "target_v": v_int,
        "target_method": target_method,
        "bbox": (x1, y1, x2, y2),
        "is_person": is_person,
        "frame_center": (width / 2, height / 2),
        "detection": {
            "track_id": primary[0] if len(primary) > 0 else None,
            "class_id": int(primary[7]) if len(primary) > 7 and primary[7] is not None else None,
            "conf": float(primary[6]) if len(primary) > 6 and primary[6] is not None else None,
            "novelty": target_novelty,
        },
        "pose_info": pose_info,
    }


def passive_observation(
    maxim,
    photos,
    *,
    show: bool = True,
    window_name: str = "Maxim Observation",
    target_class_id: int | None = None,
    novelty_tracker: NoveltyTracker | None = None,
    class_weights: dict[int, float] | None = None,
) -> dict[str, Any] | None:
    """Display-only observation (no movement).

    This function does not control movement. It only:
    1. Runs segmentation on the frame
    2. Displays annotated results
    3. Returns target info for the agentic system

    Movement decisions are handled by the agentic runtime via tools.
    Uses novelty-weighted scoring to prioritize new/interesting objects.

    Args:
        maxim: Maxim instance (for segmenter access)
        photos: Single frame or list of frames
        show: Whether to display the frame
        window_name: OpenCV window name
        target_class_id: If specified, filter to only this COCO class ID when selecting primary target
        novelty_tracker: Optional tracker for novelty-based scoring (uses maxim.novelty_tracker or default if None)
        class_weights: Optional class_id -> weight multiplier dict

    Returns:
        Target info dict if detections found, else None
    """
    if isinstance(photos, np.ndarray):
        photos = [photos]
    if not photos:
        return None

    frame = photos[-1]
    if not isinstance(frame, np.ndarray):
        return None

    # Get segmenter
    segmenter = getattr(maxim, "segmenter", None)
    if segmenter is None:
        if show:
            show_frame(ensure_bgr(frame), window_name=window_name, wait_ms=1)
        return None

    # Run segmentation
    try:
        observations = segmenter.segment_photos(
            photos,
            interests=list(getattr(maxim, "interests", []) or []),
        )
    except Exception as e:
        warn("Segmentation failed: %s", e, logger=getattr(maxim, "log", None))
        if show:
            show_frame(ensure_bgr(frame), window_name=window_name, wait_ms=1)
        return None

    # Filter to most recent frame
    frame_ind = len(photos) - 1
    candidates = [obs for obs in (observations or []) if obs[1] == frame_ind]

    # Get novelty tracker from maxim instance if not provided
    tracker = novelty_tracker
    if tracker is None:
        tracker = getattr(maxim, "novelty_tracker", None)

    # Get class weights from maxim instance if not provided
    weights = class_weights
    if weights is None:
        weights = getattr(maxim, "class_weights", None)

    if not show:
        # Just return detection info without display
        if not candidates:
            return None
        return display_detections(
            frame,
            candidates,
            segmenter=segmenter,
            window_name=window_name,
            wait_ms=1,
            show_pose=True,
            target_class_id=target_class_id,
            novelty_tracker=tracker,
            class_weights=weights,
        )

    # Display with annotations
    return display_detections(
        frame,
        candidates,
        segmenter=segmenter,
        window_name=window_name,
        wait_ms=1,
        show_pose=True,
        target_class_id=target_class_id,
        novelty_tracker=tracker,
        class_weights=weights,
    )


def passive_listening(maxim, save_file: str | None = None):
    """Capture and optionally save audio sample.

    Args:
        maxim: Maxim instance
        save_file: Optional path to save audio

    Returns:
        Audio sample array
    """
    sample = maxim.listen()

    if save_file and sample is not None and write_wav is not None:
        os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
        try:
            write_wav(save_file, maxim.mini.media.get_output_audio_samplerate(), sample)
        except Exception as e:
            warn("Failed to write audio to '%s': %s", save_file, e, logger=maxim.log)

    return sample


__all__ = [
    "DEFAULT_CLASS_WEIGHTS",
    "NoveltyTracker",
    "display_detections",
    "get_default_novelty_tracker",
    "passive_listening",
    "passive_observation",
    "score_detection_weighted",
]
