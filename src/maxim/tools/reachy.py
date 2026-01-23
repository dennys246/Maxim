from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from maxim.inference.observation_passive import passive_observation
from maxim.tools.base import Tool, ToolResult
from maxim.utils.logging import warn
from maxim.utils.structured_logging import log_agentic


# ─────────────────────────────────────────────────────────────────────────────
# Novelty Tracking Data Structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NoveltyRecord:
    """Record of a detection's novelty state."""

    track_id: int
    class_id: int
    class_name: str
    first_seen: float
    last_seen: float
    seen_count: int = 1
    total_frames: int = 1
    bbox_history: list[tuple[float, float, float, float]] = field(default_factory=list)
    confidence_history: list[float] = field(default_factory=list)

    @property
    def novelty_score(self) -> float:
        """Compute novelty: decays as object is seen more often."""
        # Novelty decays with exposure: 1.0 for new, approaches 0 with repeated sightings
        decay_factor = 0.85  # How quickly novelty decays per sighting
        return decay_factor ** (self.seen_count - 1)

    @property
    def persistence(self) -> float:
        """How consistently this object appears (0-1)."""
        if self.total_frames == 0:
            return 0.0
        return self.seen_count / self.total_frames

    @property
    def age_seconds(self) -> float:
        """Time since first seen."""
        return time.time() - self.first_seen


@dataclass
class NoveltyInfo:
    """Novelty information for a single detection."""

    track_id: int
    class_id: int
    class_name: str
    novelty_score: float  # 0.0 (familiar) to 1.0 (novel)
    age_seconds: float
    seen_count: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    confidence: float
    is_new: bool  # True if this is the first time seeing this track_id


class FocusInterestsTool(Tool):
    """
    Focus Reachy Mini's attention on YOLOv8 detections matching the current interests list.

    Requires a live `Maxim` instance (used for the latest frame, vision model, and motor queue).
    """

    name = "focus_interests"
    description = "Run YOLOv8 on the latest frame and move to center an interesting target."

    input_schema = {
        "deadzone_px": (int, 20),  # optional
        "duration_s": (float, None),  # optional
    }

    def __init__(self, maxim: Any) -> None:
        super().__init__()
        self._maxim = maxim
        self._last_frame_ts: float | None = None

    def execute(self, **kwargs: Any) -> ToolResult:
        maxim = self._maxim
        if maxim is None:
            return ToolResult(success=False, error="No Maxim context available.")

        frame = getattr(maxim, "_last_frame", None)
        frame_ts = getattr(maxim, "_last_frame_ts", None)
        if frame is None or not isinstance(frame, np.ndarray):
            return ToolResult(success=False, error="No camera frame available.")

        ts: float | None = None
        try:
            ts = float(frame_ts) if frame_ts is not None else None
        except Exception:
            ts = None

        if ts is not None and self._last_frame_ts is not None and ts <= float(self._last_frame_ts):
            return ToolResult(success=True, output={"skipped": True, "reason": "no_new_frame"})
        if ts is not None:
            self._last_frame_ts = float(ts)

        if getattr(maxim, "segmenter", None) is None:
            return ToolResult(success=False, error="Vision segmenter not initialized.")

        deadzone_px = int(kwargs.get("deadzone_px", 20) or 20)
        duration_s = kwargs.get("duration_s", None)
        duration: float | None = None
        if duration_s is not None:
            try:
                duration = float(duration_s)
            except Exception:
                duration = None

        paused = getattr(maxim, "_training_paused", None)
        pause_training = bool(getattr(maxim, "train", False)) and paused is not None
        lock = getattr(maxim, "_observation_lock", None)

        try:
            if pause_training:
                try:
                    paused.set()
                except Exception:
                    pass

            if lock is None:
                passive_observation(maxim, frame, duration=duration, deadzone_px=deadzone_px, show=False)
            else:
                with lock:
                    passive_observation(maxim, frame, duration=duration, deadzone_px=deadzone_px, show=False)
        except Exception as e:
            warn("focus_interests failed: %s", e, logger=getattr(maxim, "log", None))
            return ToolResult(success=False, error=str(e))
        finally:
            if pause_training:
                try:
                    paused.clear()
                except Exception:
                    pass

        return ToolResult(
            success=True,
            output={
                "focused": True,
                "frame_ts": ts,
                "deadzone_px": int(deadzone_px),
            },
        )


class TrackTargetTool(Tool):
    """
    Move head to center on detected objects of interest.

    Uses target info from CaptureManager/PerceptionAgent detections to keep
    interesting objects centered in view. This enables active visual tracking
    behavior for the agentic system.
    """

    name = "track_target"
    description = "Move head to center on a detected object of interest."

    input_schema = {
        "deadzone_px": (int, 40),  # Minimum offset from center to trigger movement
        "duration_s": (float, 0.3),  # Movement duration
        "prefer_people": (bool, True),  # Prioritize people over other objects
    }

    def __init__(self, maxim: Any) -> None:
        super().__init__()
        self._maxim = maxim
        self._last_track_time: float = 0.0
        self._min_interval: float = 0.1  # Rate limit: max 10 Hz

        # Position tracking for movement gating
        self._current_look_u: float | None = None
        self._current_look_v: float | None = None
        self._movement_threshold: float = MIN_MOVEMENT_THRESHOLD_PX

    def execute(self, **kwargs: Any) -> ToolResult:
        maxim = self._maxim
        if maxim is None:
            return ToolResult(success=False, error="No Maxim context available.")

        # Rate limit
        now = time.time()
        if now - self._last_track_time < self._min_interval:
            log_agentic("track_target", "rate_limited", level="DEBUG")
            return ToolResult(
                success=True,
                output={"skipped": True, "reason": "rate_limited"},
            )

        # Get parameters
        deadzone_px = int(kwargs.get("deadzone_px", 40) or 40)
        duration_s = float(kwargs.get("duration_s", 0.3) or 0.3)
        prefer_people = bool(kwargs.get("prefer_people", True))

        # Try to get target from CaptureManager first (Phase 3 path)
        target_info = None
        frame_width = DEFAULT_FRAME_WIDTH
        frame_height = DEFAULT_FRAME_HEIGHT
        capture_manager = getattr(maxim, "_capture_manager", None)
        if capture_manager is not None:
            captured = capture_manager.get_latest_frame()
            if captured is not None and captured.detections:
                frame_shape = captured.frame.shape if captured.frame is not None else (1080, 1920)
                frame_height = frame_shape[0] if len(frame_shape) > 0 else 1080
                frame_width = frame_shape[1] if len(frame_shape) > 1 else 1920
                target_info = self._compute_target_from_detections(
                    captured.detections,
                    frame_shape,
                    prefer_people=prefer_people,
                )

        # Fall back to stored detection target (Phase 2 path)
        if target_info is None:
            target_info = getattr(maxim, "_last_detection_target", None)

        if target_info is None or not isinstance(target_info, dict):
            log_agentic("track_target", "no_target", level="DEBUG")
            return ToolResult(
                success=True,
                output={"skipped": True, "reason": "no_target"},
            )

        # Extract target coordinates
        target_u = target_info.get("target_u")
        target_v = target_info.get("target_v")
        frame_center = target_info.get("frame_center", (frame_width / 2, frame_height / 2))

        if target_u is None or target_v is None:
            log_agentic("track_target", "no_target", {"reason": "no_coords"}, level="DEBUG")
            return ToolResult(
                success=True,
                output={"skipped": True, "reason": "no_target_coords"},
            )

        # Gate 1: Clamp to safe vision range
        clamped_u, clamped_v, was_clamped = clamp_to_vision_range(
            target_u, target_v, frame_width, frame_height
        )

        # Gate 2: Check if movement is significant
        if not is_significant_movement(
            self._current_look_u,
            self._current_look_v,
            clamped_u,
            clamped_v,
            self._movement_threshold,
        ):
            log_agentic(
                "track_target",
                "insignificant_movement",
                {
                    "current_u": round(self._current_look_u or 0, 1),
                    "current_v": round(self._current_look_v or 0, 1),
                    "target_u": round(clamped_u, 1),
                    "target_v": round(clamped_v, 1),
                },
                level="DEBUG",
            )
            return ToolResult(
                success=True,
                output={
                    "skipped": True,
                    "reason": "insignificant_movement",
                    "current_position": (
                        round(self._current_look_u or 0, 1),
                        round(self._current_look_v or 0, 1),
                    ),
                    "target_position": (round(clamped_u, 1), round(clamped_v, 1)),
                },
            )

        # Gate 3: Check deadzone
        center_u, center_v = frame_center
        offset_u = abs(clamped_u - center_u)
        offset_v = abs(clamped_v - center_v)

        if offset_u < deadzone_px and offset_v < deadzone_px:
            log_agentic(
                "track_target",
                "within_deadzone",
                {"offset_u": round(offset_u, 1), "offset_v": round(offset_v, 1)},
                level="DEBUG",
            )
            return ToolResult(
                success=True,
                output={
                    "skipped": True,
                    "reason": "within_deadzone",
                    "offset_u": offset_u,
                    "offset_v": offset_v,
                },
            )

        # All gates passed - execute movement
        try:
            maxim.look_at_image(
                int(clamped_u),
                int(clamped_v),
                duration=duration_s,
                perform_movement=True,
            )

            # Update position tracking
            self._current_look_u = clamped_u
            self._current_look_v = clamped_v
            self._last_track_time = now

            # Log successful tracking
            log_agentic(
                "track_target",
                "detection",
                {
                    "target_u": int(clamped_u),
                    "target_v": int(clamped_v),
                    "offset_u": round(offset_u, 1),
                    "offset_v": round(offset_v, 1),
                    "is_person": target_info.get("is_person", False),
                    "was_clamped": was_clamped,
                },
            )

            return ToolResult(
                success=True,
                output={
                    "tracked": True,
                    "target_u": int(clamped_u),
                    "target_v": int(clamped_v),
                    "original_u": int(target_u),
                    "original_v": int(target_v),
                    "was_clamped": was_clamped,
                    "offset_u": offset_u,
                    "offset_v": offset_v,
                    "is_person": target_info.get("is_person", False),
                    "duration_s": duration_s,
                },
            )
        except Exception as e:
            log_agentic(
                "track_target",
                "error",
                {"error": str(e)},
                level="ERROR",
            )
            warn("track_target failed: %s", e, logger=getattr(maxim, "log", None))
            return ToolResult(success=False, error=str(e))

    def _compute_target_from_detections(
        self,
        detections: list[dict],
        frame_shape: tuple,
        prefer_people: bool = True,
    ) -> dict | None:
        """Compute tracking target from detection list."""
        if not detections:
            return None

        height = frame_shape[0] if len(frame_shape) > 0 else 1080
        width = frame_shape[1] if len(frame_shape) > 1 else 1920

        # Separate people from other detections
        people = []
        others = []
        for det in detections:
            class_id = det.get("class_id")
            if class_id == 0:  # COCO person class
                people.append(det)
            else:
                others.append(det)

        # Choose target based on preference
        if prefer_people and people:
            target_list = people
        elif others:
            target_list = others
        elif people:
            target_list = people
        else:
            return None

        # Select best target (highest confidence)
        best = max(target_list, key=lambda d: float(d.get("conf", 0)))

        # Compute center of bounding box
        bbox = best.get("bbox_xyxy", [0, 0, 0, 0])
        if len(bbox) < 4:
            return None

        x1, y1, x2, y2 = bbox
        target_u = (x1 + x2) / 2
        target_v = (y1 + y2) / 2

        return {
            "target_u": target_u,
            "target_v": target_v,
            "frame_center": (width / 2, height / 2),
            "is_person": best.get("class_id") == 0,
            "bbox": (x1, y1, x2, y2),
            "detection": {
                "track_id": best.get("track_id"),
                "class_id": best.get("class_id"),
                "conf": best.get("conf"),
            },
        }


class MaximCommandTool(Tool):
    """
    Execute a small allowlisted set of actions on a live `Maxim` instance.
    """

    name = "maxim_command"
    description = "Execute an allowlisted Maxim command (side effects on Reachy/runtime)."
    input_schema = {
        "command": str,
        "params": (dict, None),  # optional
        "note": (str, None),  # optional (e.g., for label_outcome)
    }

    _ALLOWED: set[str] = {
        "center_vision",
        "goto_pose",
        "look_at_image",
        "mark_trainable_moment",
        "move",
        "move_antenna",
        "label_outcome",
        "request_sleep",
        "request_observe",
        "request_shutdown",
        "update_interests",
    }

    def __init__(self, maxim: Any) -> None:
        super().__init__()
        self._maxim = maxim

    def execute(self, **kwargs: Any) -> ToolResult:
        maxim = self._maxim
        if maxim is None:
            return ToolResult(success=False, error="No Maxim context available.")

        command = kwargs.get("command")
        if not isinstance(command, str) or not command:
            return ToolResult(success=False, error="Missing command.")
        command = command.strip()
        if command not in self._ALLOWED:
            return ToolResult(success=False, error=f"Unsupported command: {command}")

        params = kwargs.get("params") if isinstance(kwargs.get("params"), dict) else {}
        note = kwargs.get("note")

        paused = getattr(maxim, "_training_paused", None)
        pause_training = bool(getattr(maxim, "train", False)) and paused is not None and command in {
            "center_vision",
            "mark_trainable_moment",
            "label_outcome",
        }

        try:
            if pause_training:
                try:
                    paused.set()
                except Exception:
                    pass

            if command == "label_outcome":
                code = params.get("code", 0)
                try:
                    maxim.label_outcome(int(code), source="llm", trigger="maxim", note=str(note) if note else None)
                except Exception as e:
                    return ToolResult(success=False, error=str(e))
                return ToolResult(success=True, output={"command": command, "code": int(code)})
            if command == "update_interests":
                add = params.get("add") if isinstance(params.get("add"), list) else None
                remove = params.get("remove") if isinstance(params.get("remove"), list) else None
                try:
                    maxim.update_interests(add=add, remove=remove)
                except Exception as e:
                    return ToolResult(success=False, error=str(e))
                return ToolResult(
                    success=True,
                    output={
                        "command": command,
                        "interests": list(getattr(maxim, "interests", []) or []),
                    },
                )
            if command == "goto_pose":
                name = params.get("name", "centered")
                duration = params.get("duration")
                try:
                    maxim.goto_pose(str(name), duration=duration)
                except Exception as e:
                    return ToolResult(success=False, error=str(e))
                return ToolResult(success=True, output={"command": command, "name": str(name)})
            if command == "look_at_image":
                try:
                    u = int(params.get("u"))
                    v = int(params.get("v"))
                except Exception:
                    return ToolResult(success=False, error="Missing u/v for look_at_image.")
                duration = params.get("duration")
                perform_movement = params.get("perform_movement", True)
                try:
                    maxim.look_at_image(u, v, duration=duration, perform_movement=bool(perform_movement))
                except Exception as e:
                    return ToolResult(success=False, error=str(e))
                return ToolResult(success=True, output={"command": command, "u": u, "v": v})
            if command == "move":
                duration = params.get("duration")
                try:
                    maxim.move(
                        x=params.get("x"),
                        y=params.get("y"),
                        z=params.get("z"),
                        roll=params.get("roll"),
                        pitch=params.get("pitch"),
                        yaw=params.get("yaw"),
                        duration=duration,
                    )
                except Exception as e:
                    return ToolResult(success=False, error=str(e))
                return ToolResult(success=True, output={"command": command})
            if command == "move_antenna":
                try:
                    maxim.move_antenna(
                        right=params.get("right"),
                        left=params.get("left"),
                        angle=params.get("angle"),
                        duration=params.get("duration"),
                        method=params.get("method", "minjerk"),
                        degrees=bool(params.get("degrees", True)),
                        relative=bool(params.get("relative", False)),
                    )
                except Exception as e:
                    return ToolResult(success=False, error=str(e))
                return ToolResult(success=True, output={"command": command})

            fn = getattr(maxim, command, None)
            if not callable(fn):
                return ToolResult(success=False, error=f"Maxim missing method: {command}")
            fn()
            return ToolResult(success=True, output={"command": command})
        except Exception as e:
            warn("maxim_command failed: %s", e, logger=getattr(maxim, "log", None))
            return ToolResult(success=False, error=str(e))
        finally:
            if pause_training:
                try:
                    paused.clear()
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Vision Range Constants
# ─────────────────────────────────────────────────────────────────────────────

# Frame dimensions (standard Reachy Mini camera)
DEFAULT_FRAME_WIDTH = 1920
DEFAULT_FRAME_HEIGHT = 1080

# Safe viewing margins - don't look too close to edges where head limits are reached
# These are pixel margins from the frame edges
VISION_MARGIN_HORIZONTAL = 200  # ~10% margin on each side
VISION_MARGIN_VERTICAL = 150  # ~14% margin top/bottom

# Effective viewing range (targets outside this are clamped)
VISION_MIN_U = VISION_MARGIN_HORIZONTAL
VISION_MAX_U = DEFAULT_FRAME_WIDTH - VISION_MARGIN_HORIZONTAL
VISION_MIN_V = VISION_MARGIN_VERTICAL
VISION_MAX_V = DEFAULT_FRAME_HEIGHT - VISION_MARGIN_VERTICAL

# Minimum movement threshold - don't move if target is within this distance of current look position
MIN_MOVEMENT_THRESHOLD_PX = 50  # pixels


def clamp_to_vision_range(
    u: float,
    v: float,
    frame_width: int = DEFAULT_FRAME_WIDTH,
    frame_height: int = DEFAULT_FRAME_HEIGHT,
) -> tuple[float, float, bool]:
    """
    Clamp target coordinates to safe vision range.

    Args:
        u: Horizontal pixel coordinate
        v: Vertical pixel coordinate
        frame_width: Frame width for scaling margins
        frame_height: Frame height for scaling margins

    Returns:
        Tuple of (clamped_u, clamped_v, was_clamped)
    """
    # Scale margins proportionally if frame size differs from default
    margin_h = int(VISION_MARGIN_HORIZONTAL * frame_width / DEFAULT_FRAME_WIDTH)
    margin_v = int(VISION_MARGIN_VERTICAL * frame_height / DEFAULT_FRAME_HEIGHT)

    min_u = margin_h
    max_u = frame_width - margin_h
    min_v = margin_v
    max_v = frame_height - margin_v

    clamped_u = max(min_u, min(max_u, u))
    clamped_v = max(min_v, min(max_v, v))

    was_clamped = (clamped_u != u) or (clamped_v != v)
    return clamped_u, clamped_v, was_clamped


def is_significant_movement(
    current_u: float | None,
    current_v: float | None,
    target_u: float,
    target_v: float,
    threshold: float = MIN_MOVEMENT_THRESHOLD_PX,
) -> bool:
    """
    Check if target position represents a significant movement from current.

    Args:
        current_u: Current look position u (None if unknown)
        current_v: Current look position v (None if unknown)
        target_u: Target look position u
        target_v: Target look position v
        threshold: Minimum distance to be considered significant

    Returns:
        True if movement is significant, False if too small to bother
    """
    if current_u is None or current_v is None:
        return True  # Always move if we don't know current position

    distance = ((target_u - current_u) ** 2 + (target_v - current_v) ** 2) ** 0.5
    return distance >= threshold


# COCO class names for common objects (subset)
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    15: "cat",
    16: "dog",
    17: "horse",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    73: "book",
    74: "clock",
}


class NoveltyTrackTool(Tool):
    """
    Track and center vision on novel objects not recently seen.

    Maintains a memory of seen track IDs and their novelty scores. Novel objects
    (high novelty score) are prioritized for visual attention. Other agents can
    query novelty information via the query_novelty() method.

    Features:
    - Per-detection novelty scoring based on exposure history
    - Automatic forgetting of old tracks (configurable timeout)
    - Query interface for other agents to get novelty rankings
    - Optional automatic centering on most novel object
    """

    name = "novelty_track"
    description = "Center vision on the most novel detected object, or query novelty info."

    input_schema = {
        "action": (str, "track"),  # "track", "query", "reset"
        "novelty_threshold": (float, 0.5),  # Min novelty to trigger movement
        "deadzone_px": (int, 40),  # Deadzone for movement
        "duration_s": (float, 0.3),  # Movement duration
        "top_k": (int, 5),  # Number of results for query action
        "class_filter": (list, None),  # Optional class IDs to filter
    }

    def __init__(
        self,
        maxim: Any,
        *,
        memory_duration: float = 60.0,
        max_tracked: int = 100,
        frame_counter_decay: int = 30,
    ) -> None:
        """
        Initialize NoveltyTrackTool.

        Args:
            maxim: The Maxim instance for camera control
            memory_duration: Seconds before forgetting old tracks
            max_tracked: Maximum number of tracks to remember
            frame_counter_decay: Frames before incrementing total_frames for decay
        """
        super().__init__()
        self._maxim = maxim
        self._memory_duration = memory_duration
        self._max_tracked = max_tracked
        self._frame_counter_decay = frame_counter_decay

        # Track memory: track_id -> NoveltyRecord
        self._track_memory: dict[int, NoveltyRecord] = {}
        self._frame_count = 0
        self._last_track_time: float = 0.0
        self._min_interval: float = 0.1  # Rate limit: max 10 Hz

        # Position tracking - remember where we're currently looking
        self._current_look_u: float | None = None
        self._current_look_v: float | None = None
        self._movement_threshold: float = MIN_MOVEMENT_THRESHOLD_PX

    def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "track")).lower()

        if action == "query":
            return self._query_novelty(**kwargs)
        elif action == "reset":
            return self._reset_memory()
        elif action == "track":
            return self._track_novel(**kwargs)
        else:
            return ToolResult(
                success=False,
                error=f"Unknown action: {action}. Use 'track', 'query', or 'reset'.",
            )

    def _track_novel(self, **kwargs: Any) -> ToolResult:
        """Track and center on the most novel detection.

        Includes gating for:
        - Rate limiting (max 10 Hz)
        - Minimum movement threshold (skip if too close to current position)
        - Vision range clamping (don't look outside safe head range)
        - Deadzone check (skip if target already centered)
        """
        maxim = self._maxim
        if maxim is None:
            return ToolResult(success=False, error="No Maxim context available.")

        # Rate limit
        now = time.time()
        if now - self._last_track_time < self._min_interval:
            return ToolResult(
                success=True,
                output={"skipped": True, "reason": "rate_limited"},
            )

        # Get parameters
        novelty_threshold = float(kwargs.get("novelty_threshold", 0.5))
        deadzone_px = int(kwargs.get("deadzone_px", 40))
        duration_s = float(kwargs.get("duration_s", 0.3))
        class_filter = kwargs.get("class_filter")

        # Get detections from CaptureManager or fallback
        detections, frame_shape = self._get_current_detections()
        if not detections:
            return ToolResult(
                success=True,
                output={"skipped": True, "reason": "no_detections"},
            )

        # Get frame dimensions
        height = frame_shape[0] if len(frame_shape) > 0 else DEFAULT_FRAME_HEIGHT
        width = frame_shape[1] if len(frame_shape) > 1 else DEFAULT_FRAME_WIDTH

        # Update track memory and compute novelty for all detections
        novelty_infos = self._update_and_score(detections, class_filter)
        if not novelty_infos:
            return ToolResult(
                success=True,
                output={"skipped": True, "reason": "no_valid_detections"},
            )

        # Find most novel above threshold
        novel_candidates = [n for n in novelty_infos if n.novelty_score >= novelty_threshold]
        if not novel_candidates:
            # Return info about what was seen but below threshold
            best = max(novelty_infos, key=lambda x: x.novelty_score)
            return ToolResult(
                success=True,
                output={
                    "skipped": True,
                    "reason": "below_threshold",
                    "best_novelty": round(best.novelty_score, 3),
                    "best_track_id": best.track_id,
                    "best_class": best.class_name,
                    "threshold": novelty_threshold,
                },
            )

        # Select most novel
        target = max(novel_candidates, key=lambda x: x.novelty_score)
        target_u, target_v = target.center

        # Gate 1: Clamp to safe vision range (don't exceed head limits)
        clamped_u, clamped_v, was_clamped = clamp_to_vision_range(
            target_u, target_v, width, height
        )

        # Gate 2: Check if movement is significant (avoid micro-movements)
        if not is_significant_movement(
            self._current_look_u,
            self._current_look_v,
            clamped_u,
            clamped_v,
            self._movement_threshold,
        ):
            log_agentic(
                "novelty_track",
                "insignificant_movement",
                {
                    "track_id": target.track_id,
                    "current_u": round(self._current_look_u or 0, 1),
                    "current_v": round(self._current_look_v or 0, 1),
                    "target_u": round(clamped_u, 1),
                    "target_v": round(clamped_v, 1),
                    "threshold": self._movement_threshold,
                },
                level="DEBUG",
            )
            return ToolResult(
                success=True,
                output={
                    "skipped": True,
                    "reason": "insignificant_movement",
                    "target_track_id": target.track_id,
                    "target_class": target.class_name,
                    "novelty": round(target.novelty_score, 3),
                    "current_position": (
                        round(self._current_look_u or 0, 1),
                        round(self._current_look_v or 0, 1),
                    ),
                    "target_position": (round(clamped_u, 1), round(clamped_v, 1)),
                    "movement_threshold": self._movement_threshold,
                },
            )

        # Gate 3: Check deadzone (target already near center)
        center_u, center_v = width / 2, height / 2
        offset_u = abs(clamped_u - center_u)
        offset_v = abs(clamped_v - center_v)

        if offset_u < deadzone_px and offset_v < deadzone_px:
            log_agentic(
                "novelty_track",
                "within_deadzone",
                {
                    "track_id": target.track_id,
                    "novelty": round(target.novelty_score, 3),
                    "offset_u": round(offset_u, 1),
                    "offset_v": round(offset_v, 1),
                },
                level="DEBUG",
            )
            return ToolResult(
                success=True,
                output={
                    "skipped": True,
                    "reason": "within_deadzone",
                    "target_track_id": target.track_id,
                    "target_class": target.class_name,
                    "novelty": round(target.novelty_score, 3),
                    "offset_u": offset_u,
                    "offset_v": offset_v,
                },
            )

        # All gates passed - execute movement
        try:
            maxim.look_at_image(
                int(clamped_u),
                int(clamped_v),
                duration=duration_s,
                perform_movement=True,
            )

            # Update position tracking
            self._current_look_u = clamped_u
            self._current_look_v = clamped_v
            self._last_track_time = now

            log_agentic(
                "novelty_track",
                "centered",
                {
                    "track_id": target.track_id,
                    "class": target.class_name,
                    "novelty": round(target.novelty_score, 3),
                    "is_new": target.is_new,
                    "seen_count": target.seen_count,
                    "was_clamped": was_clamped,
                },
            )

            return ToolResult(
                success=True,
                output={
                    "tracked": True,
                    "target_track_id": target.track_id,
                    "target_class": target.class_name,
                    "target_class_id": target.class_id,
                    "novelty": round(target.novelty_score, 3),
                    "is_new": target.is_new,
                    "seen_count": target.seen_count,
                    "age_seconds": round(target.age_seconds, 2),
                    "target_u": int(clamped_u),
                    "target_v": int(clamped_v),
                    "original_u": int(target_u),
                    "original_v": int(target_v),
                    "was_clamped": was_clamped,
                    "duration_s": duration_s,
                },
            )
        except Exception as e:
            log_agentic("novelty_track", "error", {"error": str(e)}, level="ERROR")
            warn("novelty_track failed: %s", e, logger=getattr(maxim, "log", None))
            return ToolResult(success=False, error=str(e))

    def _query_novelty(self, **kwargs: Any) -> ToolResult:
        """Query novelty information without moving."""
        top_k = int(kwargs.get("top_k", 5))
        class_filter = kwargs.get("class_filter")

        detections, _ = self._get_current_detections()
        if not detections:
            return ToolResult(
                success=True,
                output={
                    "novelty_rankings": [],
                    "total_tracked": len(self._track_memory),
                    "reason": "no_current_detections",
                },
            )

        # Update and score
        novelty_infos = self._update_and_score(detections, class_filter)

        # Sort by novelty (highest first)
        sorted_infos = sorted(novelty_infos, key=lambda x: x.novelty_score, reverse=True)[:top_k]

        rankings = [
            {
                "track_id": n.track_id,
                "class_id": n.class_id,
                "class_name": n.class_name,
                "novelty": round(n.novelty_score, 3),
                "is_new": n.is_new,
                "seen_count": n.seen_count,
                "age_seconds": round(n.age_seconds, 2),
                "center": (round(n.center[0], 1), round(n.center[1], 1)),
                "confidence": round(n.confidence, 3),
            }
            for n in sorted_infos
        ]

        # Also return summary stats
        novel_count = sum(1 for n in novelty_infos if n.novelty_score >= 0.5)
        new_count = sum(1 for n in novelty_infos if n.is_new)

        return ToolResult(
            success=True,
            output={
                "novelty_rankings": rankings,
                "total_current": len(novelty_infos),
                "total_tracked": len(self._track_memory),
                "novel_count": novel_count,
                "new_count": new_count,
            },
        )

    def _reset_memory(self) -> ToolResult:
        """Reset all novelty tracking memory."""
        count = len(self._track_memory)
        self._track_memory.clear()
        self._frame_count = 0
        return ToolResult(
            success=True,
            output={"reset": True, "cleared_tracks": count},
        )

    def _get_current_detections(self) -> tuple[list[dict], tuple]:
        """Get current detections from CaptureManager or fallback."""
        maxim = self._maxim
        detections = []
        frame_shape = (1080, 1920)

        # Try CaptureManager first (Phase 3)
        capture_manager = getattr(maxim, "_capture_manager", None)
        if capture_manager is not None:
            captured = capture_manager.get_latest_frame()
            if captured is not None:
                detections = captured.detections or []
                if captured.frame is not None:
                    frame_shape = captured.frame.shape

        # Fallback to stored detections
        if not detections:
            last_det = getattr(maxim, "_last_detection_target", None)
            if isinstance(last_det, dict) and "detection" in last_det:
                # Single detection format
                detections = [last_det["detection"]]

        return detections, frame_shape

    def _update_and_score(
        self,
        detections: list[dict],
        class_filter: list[int] | None = None,
    ) -> list[NoveltyInfo]:
        """Update track memory and return novelty info for current detections."""
        now = time.time()
        self._frame_count += 1

        # Prune old tracks
        self._prune_old_tracks(now)

        # Increment total_frames for decay calculation
        if self._frame_count % self._frame_counter_decay == 0:
            for record in self._track_memory.values():
                record.total_frames += 1

        novelty_infos = []

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            class_id = det.get("class_id", -1)

            # Apply class filter if specified
            if class_filter is not None and class_id not in class_filter:
                continue

            class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
            conf = float(det.get("conf", 0.0))
            bbox = det.get("bbox_xyxy", [0, 0, 0, 0])

            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            bbox_tuple = (float(x1), float(y1), float(x2), float(y2))

            # Check if new or update existing
            is_new = track_id not in self._track_memory

            if is_new:
                # New track
                record = NoveltyRecord(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    first_seen=now,
                    last_seen=now,
                    seen_count=1,
                    total_frames=1,
                    bbox_history=[bbox_tuple],
                    confidence_history=[conf],
                )
                self._track_memory[track_id] = record

                # Enforce max tracked limit
                if len(self._track_memory) > self._max_tracked:
                    self._evict_oldest()
            else:
                # Update existing
                record = self._track_memory[track_id]
                record.last_seen = now
                record.seen_count += 1
                record.bbox_history.append(bbox_tuple)
                record.confidence_history.append(conf)

                # Keep history bounded
                if len(record.bbox_history) > 30:
                    record.bbox_history = record.bbox_history[-30:]
                    record.confidence_history = record.confidence_history[-30:]

            # Build novelty info
            novelty_infos.append(
                NoveltyInfo(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    novelty_score=record.novelty_score,
                    age_seconds=record.age_seconds,
                    seen_count=record.seen_count,
                    bbox=bbox_tuple,
                    center=center,
                    confidence=conf,
                    is_new=is_new,
                )
            )

        return novelty_infos

    def _prune_old_tracks(self, now: float) -> None:
        """Remove tracks not seen for memory_duration seconds."""
        to_remove = [
            track_id
            for track_id, record in self._track_memory.items()
            if (now - record.last_seen) > self._memory_duration
        ]
        for track_id in to_remove:
            del self._track_memory[track_id]

    def _evict_oldest(self) -> None:
        """Evict the oldest track when at capacity."""
        if not self._track_memory:
            return
        oldest_id = min(
            self._track_memory.keys(),
            key=lambda tid: self._track_memory[tid].last_seen,
        )
        del self._track_memory[oldest_id]

    # ─────────────────────────────────────────────────────────────────────────
    # Public API for other agents
    # ─────────────────────────────────────────────────────────────────────────

    def query_novelty(
        self,
        top_k: int = 5,
        class_filter: list[int] | None = None,
        min_novelty: float = 0.0,
    ) -> list[NoveltyInfo]:
        """
        Query novelty rankings for other agents (direct API, no tool call).

        Args:
            top_k: Number of results to return
            class_filter: Optional list of class IDs to include
            min_novelty: Minimum novelty threshold

        Returns:
            List of NoveltyInfo sorted by novelty (highest first)
        """
        detections, _ = self._get_current_detections()
        if not detections:
            return []

        novelty_infos = self._update_and_score(detections, class_filter)
        filtered = [n for n in novelty_infos if n.novelty_score >= min_novelty]
        return sorted(filtered, key=lambda x: x.novelty_score, reverse=True)[:top_k]

    def get_most_novel(self, class_filter: list[int] | None = None) -> NoveltyInfo | None:
        """
        Get the single most novel detection.

        Args:
            class_filter: Optional list of class IDs to include

        Returns:
            NoveltyInfo for most novel detection, or None if no detections
        """
        results = self.query_novelty(top_k=1, class_filter=class_filter)
        return results[0] if results else None

    def get_track_history(self, track_id: int) -> NoveltyRecord | None:
        """
        Get the full history record for a specific track.

        Args:
            track_id: The track ID to look up

        Returns:
            NoveltyRecord if found, None otherwise
        """
        return self._track_memory.get(track_id)

    def get_new_detections(self, class_filter: list[int] | None = None) -> list[NoveltyInfo]:
        """
        Get only brand-new detections (first time seen).

        Args:
            class_filter: Optional list of class IDs to include

        Returns:
            List of NoveltyInfo for new detections only
        """
        detections, _ = self._get_current_detections()
        if not detections:
            return []

        novelty_infos = self._update_and_score(detections, class_filter)
        return [n for n in novelty_infos if n.is_new]

    @property
    def tracked_count(self) -> int:
        """Number of tracks currently in memory."""
        return len(self._track_memory)

    @property
    def track_ids(self) -> set[int]:
        """Set of all track IDs currently in memory."""
        return set(self._track_memory.keys())
