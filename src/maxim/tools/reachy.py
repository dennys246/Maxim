from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from maxim.inference.segment_vision import passive_observation
from maxim.tools.base import Tool, ToolResult
from maxim.utils.logging import warn
from maxim.tools.novelty import NoveltyInfo, NoveltyRecord
from maxim.utils.structured_logging import log_agentic

if TYPE_CHECKING:
    from maxim.hardware import RobotController


def _get_robot_from_registry(
    robot_id: str | None,
    maxim: Any | None = None,
) -> "RobotController | None":
    """Get a robot controller by ID, or the primary robot.

    Args:
        robot_id: Optional robot ID. If None, uses primary robot.
        maxim: Optional Maxim instance (for backward compatibility).

    Returns:
        RobotController instance, or None if not found.
    """
    # Try to get from maxim's robot first (most common case)
    if robot_id is None and maxim is not None:
        robot = getattr(maxim, "_robot", None)
        if robot is not None:
            return robot

    # Look up from global registry
    try:
        from maxim.hardware import RobotRegistry

        registry = RobotRegistry()

        if robot_id is not None:
            return registry.get_robot(robot_id)
        else:
            return registry.primary
    except Exception:
        return None


class FocusInterestsTool(Tool):
    """
    Focus Reachy Mini's attention on vision detections matching specified or default interests.

    Requires a live `Maxim` instance (used for the latest frame, vision model, and motor queue).
    If target_class is provided, detections are filtered to that class for centering.
    """

    name = "focus_interests"
    description = "Focus attention on objects in the current frame. Use target_class to prioritize a specific object class (e.g., 'cup', 'backpack', 'person')."

    input_schema = {
        "target_class": (str, None),  # optional - specific object class to focus on (e.g., "backpack", "person")
        "deadzone_px": (int, 20),  # optional - pixels from center before triggering movement
    }

    def __init__(self, maxim: Any) -> None:
        super().__init__()
        self._maxim = maxim
        self._last_frame_ts: float | None = None

    def execute(self, **kwargs: Any) -> ToolResult:
        import time as _time
        import logging

        _logger = logging.getLogger(__name__)
        _exec_start = _time.time()
        _logger.info("focus_interests: starting execution")

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

        target_class = kwargs.get("target_class")
        deadzone_px = int(kwargs.get("deadzone_px", 20) or 20)

        # Look up class ID for target_class filtering
        # Vision model detects all 80 COCO classes; target_class_id is used for
        # post-detection filtering in passive_observation, not model filtering.
        target_class_id: int | None = None
        if target_class:
            class_name_to_id = {v.lower(): k for k, v in COCO_CLASSES.items()}
            target_class_id = class_name_to_id.get(str(target_class).lower().strip())
            _logger.info("focus_interests: target_class='%s' mapped to class_id=%s", target_class, target_class_id)

        paused = getattr(maxim, "_training_paused", None)
        pause_training = bool(getattr(maxim, "train", False)) and paused is not None
        lock = getattr(maxim, "_observation_lock", None)

        _logger.info(
            "focus_interests: setup took %.2fs, acquiring lock=%s", _time.time() - _exec_start, lock is not None
        )

        detection_info = None
        try:
            if pause_training:
                try:
                    paused.set()
                except Exception:
                    pass

            _lock_start = _time.time()
            if lock is None:
                _logger.info("focus_interests: calling passive_observation (no lock)")
                detection_info = passive_observation(maxim, frame, show=False, target_class_id=target_class_id)
            else:
                _logger.info("focus_interests: waiting for observation lock")
                with lock:
                    _logger.info(
                        "focus_interests: lock acquired in %.2fs, running detection", _time.time() - _lock_start
                    )
                    detection_info = passive_observation(maxim, frame, show=False, target_class_id=target_class_id)
            _logger.info("focus_interests: detection complete in %.2fs", _time.time() - _lock_start)
        except Exception as e:
            warn("focus_interests failed: %s", e, logger=getattr(maxim, "log", None))
            return ToolResult(success=False, error=str(e))
        finally:
            if pause_training:
                try:
                    paused.clear()
                except Exception:
                    pass

        # If we got detection info with a target, attempt to center on it
        if detection_info and detection_info.get("detection"):
            _logger.info("focus_interests: detection found: %s", detection_info.get("detection"))
            # passive_observation returns target_u and target_v as separate keys
            target_u = detection_info.get("target_u")
            target_v = detection_info.get("target_v")
            frame_center = detection_info.get("frame_center")
            _logger.info("focus_interests: target_u=%s, target_v=%s, frame_center=%s", target_u, target_v, frame_center)
            if target_u is not None and target_v is not None and frame_center:
                offset_x = target_u - frame_center[0]
                offset_y = target_v - frame_center[1]
                _logger.info("focus_interests: offset=(%d, %d), deadzone=%d", offset_x, offset_y, deadzone_px)
                # Check if outside deadzone
                if abs(offset_x) > deadzone_px or abs(offset_y) > deadzone_px:
                    # Use look_at_image which properly enqueues motor commands
                    look_at_fn = getattr(maxim, "look_at_image", None)
                    _logger.info(
                        "focus_interests: outside deadzone, look_at_image=%s",
                        "available" if look_at_fn is not None else "None",
                    )
                    if look_at_fn is not None:
                        try:
                            look_at_fn(target_u, target_v, duration=0.3)
                            _logger.info("focus_interests: look_at_image called for (%d, %d)", target_u, target_v)
                            log_agentic(
                                "focus_interests",
                                "movement_queued",
                                {"target_u": target_u, "target_v": target_v},
                            )
                        except Exception as e:
                            _logger.warning("focus_interests: failed to call look_at_image: %s", e)
                    else:
                        _logger.warning("focus_interests: no look_at_image method available")
                else:
                    _logger.info("focus_interests: within deadzone, no movement needed")
            else:
                _logger.warning("focus_interests: missing target coordinates")
        else:
            # Log what was returned to help debug
            if detection_info:
                _logger.warning(
                    "focus_interests: detection_info returned but no 'detection' key, got: %s",
                    list(detection_info.keys()),
                )
            else:
                _logger.warning(
                    "focus_interests: passive_observation returned None (target=%s, class_id=%s)",
                    target_class,
                    target_class_id,
                )

        _logger.info("focus_interests: total execution took %.2fs", _time.time() - _exec_start)
        return ToolResult(
            success=True,
            output={
                "focused": detection_info is not None,
                "frame_ts": ts,
                "target_class": target_class,
                "deadzone_px": int(deadzone_px),
                "detection": detection_info.get("detection") if detection_info else None,
            },
        )


class TrackTargetTool(Tool):
    """
    Move head to center on detected objects of interest.

    Uses target info from CaptureManager/PerceptionAgent detections to keep
    interesting objects centered in view. This enables active visual tracking
    behavior for the agentic system.

    Multi-robot support:
    - robot_id: Optional robot ID to target. If not specified, uses the primary robot.
    """

    name = "track_target"
    description = "Move head to center on a detected object. Use target_class to specify what to track (e.g., 'person', 'backpack', 'cup'). Optionally specify robot_id to target a specific robot."

    input_schema = {
        "target_class": (str, None),  # optional - specific object class to track (e.g., "backpack", "person", "cup")
        "deadzone_px": (int, 40),  # Minimum offset from center to trigger movement
        "duration_s": (float, 0.3),  # Movement duration
        "prefer_people": (bool, True),  # Prioritize people over other objects (ignored if target_class specified)
        "robot_id": (str, None),  # Optional robot ID for multi-robot support
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

        # Tracking hysteresis - prefer to stay on same target
        self._current_track_id: int | None = None
        self._track_hysteresis: float = 0.15  # Bonus confidence for current target

        # Spatial hysteresis - prefer detections near last tracked position
        # This helps when track_id changes due to head movement
        self._last_tracked_pos: tuple[float, float] | None = None
        self._spatial_hysteresis_radius: float = 200.0  # pixels - bonus within this radius
        self._spatial_hysteresis_bonus: float = 0.20  # confidence bonus for nearby detections

    def execute(self, **kwargs: Any) -> ToolResult:
        maxim = self._maxim
        if maxim is None:
            return ToolResult(success=False, error="No Maxim context available.")

        # Rate limit
        now = time.time()
        if now - self._last_track_time < self._min_interval:
            log_agentic("track_target", "rate_limited", level="DEBUG")
            return ToolResult(success=True, output={"skipped": True, "reason": "rate_limited"})

        target_class = kwargs.get("target_class")
        deadzone_px = int(kwargs.get("deadzone_px", 40) or 40)
        duration_s = float(kwargs.get("duration_s", 0.3) or 0.3)
        prefer_people = bool(kwargs.get("prefer_people", True))

        # Resolve a target from the active capture manager (or stored fallback)
        target_info, frame_width, frame_height = self._resolve_track_target(maxim, prefer_people, target_class)
        if target_info is None or not isinstance(target_info, dict):
            log_agentic("track_target", "no_target", level="DEBUG")
            return ToolResult(success=True, output={"skipped": True, "reason": "no_target"})

        target_u = target_info.get("target_u")
        target_v = target_info.get("target_v")
        frame_center = target_info.get("frame_center", (frame_width / 2, frame_height / 2))
        if target_u is None or target_v is None:
            log_agentic("track_target", "no_target", {"reason": "no_coords"}, level="DEBUG")
            return ToolResult(success=True, output={"skipped": True, "reason": "no_target_coords"})

        # Three gates: vision-range clamp, significant-movement check, deadzone check
        clamped_u, clamped_v, was_clamped = clamp_to_vision_range(target_u, target_v, frame_width, frame_height)

        skip_result = self._check_movement_gates(clamped_u, clamped_v, frame_center, deadzone_px)
        if skip_result is not None:
            return skip_result

        # All gates passed — perform the movement
        return self._perform_track_movement(
            maxim=maxim,
            kwargs=kwargs,
            clamped_u=clamped_u,
            clamped_v=clamped_v,
            target_u=target_u,
            target_v=target_v,
            duration_s=duration_s,
            frame_center=frame_center,
            target_info=target_info,
            was_clamped=was_clamped,
            now=now,
        )

    def _resolve_track_target(
        self,
        maxim: Any,
        prefer_people: bool,
        target_class: str | None,
    ) -> tuple[dict | None, int, int]:
        """Pick the best target from the latest capture frame, or fall back to stored target."""
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
                    target_class=target_class,
                )
        if target_info is None:
            target_info = getattr(maxim, "_last_detection_target", None)
        return target_info, frame_width, frame_height

    def _check_movement_gates(
        self,
        clamped_u: float,
        clamped_v: float,
        frame_center: tuple,
        deadzone_px: int,
    ) -> ToolResult | None:
        """Apply significant-movement and deadzone gates. Returns a skip ``ToolResult`` if blocked."""
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
        return None

    def _perform_track_movement(
        self,
        *,
        maxim: Any,
        kwargs: dict[str, Any],
        clamped_u: float,
        clamped_v: float,
        target_u: float,
        target_v: float,
        duration_s: float,
        frame_center: tuple,
        target_info: dict,
        was_clamped: bool,
        now: float,
    ) -> ToolResult:
        """Issue the look-at command via RobotController or Maxim, then update tracking state."""
        robot_id = kwargs.get("robot_id")
        try:
            if robot_id is not None:
                robot = _get_robot_from_registry(robot_id, maxim)
                if robot is None:
                    return ToolResult(success=False, error=f"Robot not found: {robot_id}")

                from maxim.hardware import PixelTarget

                target = PixelTarget(
                    u=int(clamped_u),
                    v=int(clamped_v),
                    duration=duration_s,
                )
                success = robot.look_at_pixel(target)
                if not success:
                    return ToolResult(success=False, error="Look-at command failed")
            else:
                maxim.look_at_image(
                    int(clamped_u),
                    int(clamped_v),
                    duration=duration_s,
                    perform_movement=True,
                )

            self._current_look_u = clamped_u
            self._current_look_v = clamped_v
            self._last_track_time = now

            center_u, center_v = frame_center
            offset_u = abs(clamped_u - center_u)
            offset_v = abs(clamped_v - center_v)

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
            log_agentic("track_target", "error", {"error": str(e)}, level="ERROR")
            warn("track_target failed: %s", e, logger=getattr(maxim, "log", None))
            return ToolResult(success=False, error=str(e))

    def _compute_target_from_detections(
        self,
        detections: list[dict],
        frame_shape: tuple,
        prefer_people: bool = True,
        target_class: str | None = None,
    ) -> dict | None:
        """Compute tracking target from detection list.

        Args:
            detections: List of detection dicts with class_id, bbox_xyxy, conf
            frame_shape: (height, width) tuple
            prefer_people: If True and no target_class, prioritize person detections
            target_class: Optional class name to filter by (e.g., "backpack", "person")
        """
        if not detections:
            return None

        height = frame_shape[0] if len(frame_shape) > 0 else 1080
        width = frame_shape[1] if len(frame_shape) > 1 else 1920

        # If target_class specified, filter detections by class name
        if target_class:
            # Build reverse lookup: class name -> class ID
            class_name_to_id = {v.lower(): k for k, v in COCO_CLASSES.items()}
            target_class_lower = target_class.lower().strip()
            target_class_id = class_name_to_id.get(target_class_lower)

            if target_class_id is not None:
                # Filter to only detections matching the target class
                filtered = [d for d in detections if d.get("class_id") == target_class_id]
                if filtered:
                    target_list = filtered
                else:
                    # No detections of the specified class found
                    return None
            else:
                # Unknown class name - try fuzzy matching
                for name, cid in class_name_to_id.items():
                    if target_class_lower in name or name in target_class_lower:
                        filtered = [d for d in detections if d.get("class_id") == cid]
                        if filtered:
                            target_list = filtered
                            break
                else:
                    # No match found, fall through to default behavior
                    target_list = detections
        else:
            # Default behavior: separate people from other detections
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

        # Try to get centralized salience map for spatial hysteresis
        salience_map = None
        try:
            dn = getattr(self._maxim, "_default_network", None)
            if dn is not None:
                salience_map = getattr(dn, "_salience_map_unified", None)
        except Exception:
            pass

        # Select best target with hysteresis (prefer current target AND nearby positions)
        def score_target(d: dict) -> float:
            conf = float(d.get("conf", 0))
            # Give bonus to currently tracked target to prevent rapid switching
            if d.get("track_id") == self._current_track_id and self._current_track_id is not None:
                conf += self._track_hysteresis

            # Spatial hysteresis: bonus for detections near last tracked position
            # Use centralized salience map if available, fallback to local tracking
            bbox = d.get("bbox_xyxy", [])
            if len(bbox) >= 4:
                det_cx = (bbox[0] + bbox[2]) / 2
                det_cy = (bbox[1] + bbox[3]) / 2

                if salience_map is not None:
                    # Use centralized tracking bonus
                    conf += salience_map.get_tracking_bonus((det_cx, det_cy))
                elif self._last_tracked_pos is not None:
                    # Fallback to local tracking
                    dx = det_cx - self._last_tracked_pos[0]
                    dy = det_cy - self._last_tracked_pos[1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < self._spatial_hysteresis_radius:
                        conf += self._spatial_hysteresis_bonus

            return conf

        best = max(target_list, key=score_target)

        # DEBUG: Log all people detections when multiple exist
        if len(target_list) > 1:
            log_agentic(
                "track_target",
                "multiple_targets",
                {
                    "count": len(target_list),
                    "current_track_id": self._current_track_id,
                    "last_tracked_pos": self._last_tracked_pos,
                    "targets": [
                        {
                            "track_id": d.get("track_id"),
                            "conf": round(float(d.get("conf", 0)), 2),
                            "score": round(score_target(d), 2),
                            "center": (
                                round(
                                    (d.get("bbox_xyxy", [0, 0, 0, 0])[0] + d.get("bbox_xyxy", [0, 0, 0, 0])[2]) / 2, 1
                                ),
                                round(
                                    (d.get("bbox_xyxy", [0, 0, 0, 0])[1] + d.get("bbox_xyxy", [0, 0, 0, 0])[3]) / 2, 1
                                ),
                            ),
                        }
                        for d in target_list[:5]  # Limit to 5
                    ],
                    "selected_track_id": best.get("track_id"),
                },
                level="WARNING",
            )

        # Update tracked target
        self._current_track_id = best.get("track_id")

        # Compute center of bounding box
        bbox = best.get("bbox_xyxy", [0, 0, 0, 0])
        if len(bbox) < 4:
            return None

        x1, y1, x2, y2 = bbox
        target_u = (x1 + x2) / 2
        target_v = (y1 + y2) / 2

        # Update last tracked position for spatial hysteresis
        # Only update for person detections to prevent locking onto non-person targets
        if best.get("class_id") == 0:  # Person class
            self._last_tracked_pos = (target_u, target_v)
            # Also update centralized salience map if available
            if salience_map is not None:
                salience_map.record_tracking_target((target_u, target_v))

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


class MoveTool(Tool):
    """
    Direct head movement control for Maxim.

    This tool provides direct control over Maxim's head position using
    relative or absolute coordinates. It is marked as always-allowed,
    meaning it bypasses autonomy approval requirements for responsive
    movement control.

    Coordinates (2026-08-03 mirror-turn fix — the SIGN CONVENTIONS below are
    hardware-verified; an undocumented yaw sign made the LLM guess, and its
    natural compass prior (+ = right) is the OPPOSITE of the stack's
    convention, producing mirror-image orienting):
    - target_x / target_y: normalized GAZE direction. They TURN the head
      (mapped to yaw/pitch), they do not translate it. target_x: -1 = look
      full LEFT ... +1 = look full RIGHT — the same sign convention as a
      heard sound's azimuth, so "look toward the sound" is target_x ≈ azimuth.
      target_y: -1 = up ... +1 = down.
    - yaw: degrees, POSITIVE = LEFT, negative = RIGHT (verified 2026-08-03:
      +30° physically turns the head left).
    - pitch: degrees, POSITIVE = DOWN. roll: degrees.
    - x/y/z: raw head TRANSLATION (mm-scale platform offsets) — rarely what
      a caller wants; use target_x/target_y for gaze.

    Multi-robot support:
    - robot_id: Optional robot ID to target. If not specified, uses the primary robot.
    """

    name = "move"
    description = (
        "Turn Maxim's head to look in a direction. Preferred: target_x (-1 = look full LEFT, "
        "+1 = look full RIGHT — same sign as a heard sound's azimuth, so to face a sound pass "
        "target_x ≈ its azimuth) and target_y (-1 = up, +1 = down). Or raw angles in degrees: "
        "yaw (POSITIVE = LEFT, negative = RIGHT), pitch (positive = down), roll. "
        "Optionally robot_id for a specific robot."
    )

    # Mark as always allowed - bypasses autonomy approval
    always_allowed = True

    input_schema = {
        "target_x": (float, None),  # Normalized target X (-1 to 1, left to right)
        "target_y": (float, None),  # Normalized target Y (-1 to 1, up to down)
        "x": (float, None),  # Raw X movement
        "y": (float, None),  # Raw Y movement
        "z": (float, None),  # Raw Z movement
        "roll": (float, None),  # Roll angle (degrees)
        "pitch": (float, None),  # Pitch angle (degrees)
        "yaw": (float, None),  # Yaw angle (degrees)
        "duration": (float, None),  # Movement duration in seconds
        "robot_id": (str, None),  # Optional robot ID for multi-robot support
    }

    def __init__(self, maxim: Any) -> None:
        super().__init__()
        self._maxim = maxim

    def execute(self, **kwargs: Any) -> ToolResult:
        maxim = self._maxim
        robot_id = kwargs.get("robot_id")

        # Handle target_x/target_y (normalized -1 to 1)
        target_x = kwargs.get("target_x")
        target_y = kwargs.get("target_y")

        x = kwargs.get("x")
        y = kwargs.get("y")
        z = kwargs.get("z")
        roll = kwargs.get("roll")
        pitch = kwargs.get("pitch")
        yaw = kwargs.get("yaw")
        duration = kwargs.get("duration", 1.0)

        # Gaze mapping (2026-08-03 mirror-turn fix). target_x/target_y are the
        # LLM-facing GAZE parameters and must TURN the head — the pre-fix code
        # mapped them onto the x/y TRANSLATION axes, so "look right" slid the
        # head a few millimetres instead of rotating it, and the model fell
        # back to raw yaw whose sign it had to guess (its compass prior,
        # + = right, is the OPPOSITE of the stack's +yaw = LEFT — verified on
        # hardware 2026-08-03). Mapping: target_x +1 (look full RIGHT, the
        # azimuth sign convention) → yaw −45°; target_y +1 (down) → pitch
        # +30°. Ranges match the joint-limit predictor (±45° yaw, ±30° pitch).
        # Explicit raw yaw/pitch win over the normalized targets.
        _MAX_GAZE_YAW_DEG = 45.0
        _MAX_GAZE_PITCH_DEG = 30.0
        if target_x is not None and yaw is None:
            tx = max(-1.0, min(1.0, float(target_x)))
            yaw = -tx * _MAX_GAZE_YAW_DEG  # +yaw = LEFT, so look-right (+tx) is negative yaw
        if target_y is not None and pitch is None:
            ty = max(-1.0, min(1.0, float(target_y)))
            pitch = ty * _MAX_GAZE_PITCH_DEG  # +pitch = DOWN, matching -1=up/+1=down

        try:
            # If robot_id specified, use RobotController directly
            if robot_id is not None:
                robot = _get_robot_from_registry(robot_id, maxim)
                if robot is None:
                    return ToolResult(success=False, error=f"Robot not found: {robot_id}")

                from maxim.hardware import MotionTarget
                import math

                # Convert degrees to radians for controller
                target = MotionTarget(
                    head_roll=math.radians(roll) if roll is not None else None,
                    head_pitch=math.radians(pitch) if pitch is not None else None,
                    head_yaw=math.radians(yaw) if yaw is not None else None,
                    duration=float(duration) if duration else 1.0,
                )

                success = robot.goto_target(target)
                if not success:
                    return ToolResult(success=False, error="Motion command failed")

                # Honesty (2026-08-07 safety fold): echoing the COMMANDED
                # angles after the controller clamped them is the
                # "accepted dispatch is a promise, not a motion" class PR
                # #459 fixed for focus_on_sound. Surface what was clamped
                # so the LLM does not re-issue an impossible pose.
                clamped_axes = tuple(getattr(robot, "last_clamped_axes", ()) or ())
                output: dict[str, Any] = {
                    "moved": True,
                    "robot_id": robot_id,
                    "target_x": target_x,
                    "target_y": target_y,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                    "duration": duration,
                }
                if clamped_axes:
                    output["clamped_axes"] = list(clamped_axes)
                    output["note"] = (
                        "command exceeded the robot's physical limits on: "
                        + ", ".join(clamped_axes)
                        + " — the head moved to the clamped pose, NOT the requested angles"
                    )
                return ToolResult(success=True, output=output)

            # Fallback to Maxim's move method for backward compatibility
            if maxim is None:
                return ToolResult(success=False, error="No Maxim context available.")

            move_fn = getattr(maxim, "move", None)
            if move_fn is None or not callable(move_fn):
                return ToolResult(success=False, error="Maxim instance does not support move()")

            move_fn(
                x=x,
                y=y,
                z=z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                duration=duration,
            )

            return ToolResult(
                success=True,
                output={
                    "moved": True,
                    "target_x": target_x,
                    "target_y": target_y,
                    "x": x,
                    "y": y,
                    "z": z,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                    "duration": duration,
                },
            )
        except Exception as e:
            warn("move failed: %s", e, logger=getattr(maxim, "log", None))
            return ToolResult(success=False, error=str(e))


def _focus_result_note(
    az: float,
    clamped: bool,
    reached: bool | None,
    side: str,
    turn_tool: str | None = None,
) -> str | None:
    """LLM-facing outcome note for focus_on_sound — honest per situation.

    The 2026-08-03 live session showed Qwen re-issuing the identical
    clamped call eight times because the output claimed faced_sound=True
    with nothing suggesting a different strategy. The note is the
    actionable half of the honesty fix: say WHY the sound is not faced —
    and NAME the real body-turn tool when one is registered (the follow-up
    session showed the un-named advice made the LLM hallucinate a
    plausible tool name, sem_motor_binding.md).
    """
    ambiguous = "front/back ambiguous (linear array)" if abs(az) <= 0.1 else None
    if clamped:
        body_advice = (
            f"use the {turn_tool} tool to rotate your body toward it"
            if turn_tool
            else "turning the body toward the sound would help"
        )
        return (
            f"sound beyond neck reach on the {side} — head pointed as far as the neck "
            f"allows but is NOT facing it; {body_advice}"
        )
    if reached is False:
        # Name REAL actions here too (2026-08-04 live session: the
        # un-named fell-short advice made the LLM hallucinate an
        # `adjust_yaw` tool, which parked the loop at an approval prompt).
        body_hint = f" or use the {turn_tool} tool to rotate your body toward it" if turn_tool else ""
        return (
            "head fell short of the target (see achieved_yaw_deg) — call "
            f"focus_on_sound again to close the remaining gap{body_hint}"
        )
    if reached is None:
        return (
            f"{ambiguous}; motion not verified (no pose readback)"
            if ambiguous
            else "motion not verified (no pose readback)"
        )
    return ambiguous


class FocusOnSoundTool(Tool):
    """Turn the head to face the sound currently being heard — closed-loop.

    The zero-numeric orient action (2026-08-03, designed off the mirror-turn
    post-mortem): the LLM decides WHETHER to attend to a sound; this tool owns
    HOW FAR to turn. No signed scalar ever crosses the LLM interface — the
    failure mode that produced the mirror robot — and the azimuth used is the
    live reading at EXECUTION time (the model's own copy is seconds stale by
    the time a decision lands).

    Sensor: the DoA feed's speech-gated cache (``maxim._doa_feed.latest``,
    live_audio_orient_wiring.md Stage 2). Convention (hardware-verified):
    azimuth -1 = full left ... +1 = full right; +yaw = LEFT in degrees. A
    head-relative azimuth spans ±90°, so the relative turn is ``azimuth·90°``
    toward the sound, applied to the CURRENT head yaw and clamped to the
    ±45° head-yaw envelope (a farther sound gets the fullest turn the neck
    allows; body rotation is the Stage-5 reflex layer's job).

    Fails soft when no sound has been heard yet — a silent room is not an
    error, there is just nothing to face.
    """

    name = "focus_on_sound"
    description = (
        "Turn Maxim's head to face the sound it is currently hearing. No direction or angle "
        "parameters — reads the live sound direction at execution time and turns the right "
        "amount in the right direction (optional: duration, seconds). Use when you hear a "
        "sound and want to look toward it. Fails softly if no sound has been heard recently."
    )

    # Responsive orienting with no side effects beyond head pose — same
    # class as move/track_target/focus_interests (autonomy ALWAYS_ALLOWED).
    always_allowed = True

    input_schema = {
        "duration": (float, None),  # Optional movement duration in seconds (default 1.0)
    }

    # Head-relative azimuth ±1 spans ±90°. The default head-yaw envelope
    # matches the joint-limit predictor's ±45°; the LEARNED workspace bound
    # (bounds learner, probed at execute time) can tighten it further but
    # never widen it.
    _AZIMUTH_TO_DEG = 90.0
    _MAX_HEAD_YAW_DEG = 45.0
    # A reading older than this is a sound that FADED — turning toward it
    # would face a memory, not a stimulus (and during silence the feed
    # holds the last value forever). Fail soft instead.
    _MAX_READING_AGE_S = 15.0
    # Post-motion readback tolerance: |achieved − target| within this counts
    # as reached (minjerk settling + pose-estimate noise; well under the
    # 15° azimuth quantum a DoA step represents).
    _REACH_TOLERANCE_DEG = 5.0

    def __init__(self, maxim: Any) -> None:
        super().__init__()
        self._maxim = maxim

    def _yaw_envelope_deg(self, maxim: Any) -> float:
        """±45° default, tightened (never widened) by the learned workspace
        bound when the runtime exposes one (pre-merge review fold: the
        bounds learner exists precisely to shrink limits that caused pain —
        a frozen 45 would ignore it)."""
        cap = self._MAX_HEAD_YAW_DEG
        get_limits = getattr(maxim, "_get_workspace_limits", None)
        if callable(get_limits):
            try:
                learned = float((get_limits() or {}).get("yaw", cap))
                if 0.0 < learned < cap:
                    cap = learned
            except Exception:
                pass
        return cap

    def execute(self, **kwargs: Any) -> ToolResult:
        maxim = self._maxim
        if maxim is None:
            return ToolResult(success=False, error="No Maxim context available.")

        feed = getattr(maxim, "_doa_feed", None)
        latest = getattr(feed, "latest", None) if feed is not None else None
        if latest is None:
            return ToolResult(
                success=False,
                error="No sound has been heard yet (no DoA reading available — is the audio feed running?)",
            )

        # (azimuth, monotonic ts, capture-time head yaw) — the third element
        # is the FRAME the head-relative azimuth was measured in. Tolerate
        # the pre-fold 2-tuple shape (capture frame unknown → fall back to
        # the current yaw, accepting the pre-fold behavior).
        azimuth, ts = float(latest[0]), float(latest[1])
        capture_yaw = latest[2] if len(latest) > 2 else None
        capture_body = latest[3] if len(latest) > 3 else None

        age_s = max(0.0, time.monotonic() - ts)
        if age_s > self._MAX_READING_AGE_S:
            return ToolResult(
                success=False,
                error=(
                    f"The last sound faded {age_s:.0f}s ago — nothing current to face. "
                    "(Stale readings are not re-used: during silence the direction cache "
                    "holds the final value indefinitely.)"
                ),
            )

        az = max(-1.0, min(1.0, azimuth))
        cur_yaw = float(getattr(maxim, "yaw", 0.0) or 0.0)
        base_yaw = float(capture_yaw) if capture_yaw is not None else cur_yaw

        # +yaw = LEFT (hardware-verified 2026-08-03); azimuth +1 = right —
        # so turning TOWARD the sound subtracts. Computed against the
        # CAPTURE-time yaw: the azimuth is head-relative in THAT frame, so
        # the target is a stable absolute pose and re-invocation on the
        # same reading is idempotent (pre-merge review fold: computing
        # against the current yaw re-subtracts the delta every call and
        # marches the head to the limit stop).
        envelope = self._yaw_envelope_deg(maxim)
        target_yaw = base_yaw - az * self._AZIMUTH_TO_DEG
        # Body-rotation correction (sem_motor_binding.md Phase 1): the
        # dispatched head_yaw is BODY-RELATIVE against the body's CURRENT
        # yaw, but the capture frame was the body's CAPTURE-time yaw. Once
        # SEM turns really rotate the body between capture and execute,
        # the same world direction requires shifting the body-relative
        # target by (capture_body − current_body). Best-effort: without a
        # body stamp or a readable pose, behave as before (fixed-body
        # assumption — correct whenever the body hasn't moved).
        if capture_body is not None:
            try:
                import math as _math_frame

                robot_for_frame = _get_robot_from_registry(None, maxim)
                pose_now = robot_for_frame.get_current_pose() if robot_for_frame is not None else None
                if pose_now and "body_yaw" in pose_now:
                    current_body_deg = _math_frame.degrees(float(pose_now["body_yaw"]))
                    target_yaw += float(capture_body) - current_body_deg
            except Exception:
                # A silent revert to the fixed-body assumption aims wrong by
                # exactly the body rotation — log it (review fold F1).
                __import__("logging").getLogger(__name__).debug(
                    "focus_on_sound: body-frame correction failed", exc_info=True
                )
        clamped = abs(target_yaw) > envelope
        target_yaw = max(-envelope, min(envelope, target_yaw))

        duration = kwargs.get("duration")
        duration_s = float(duration) if duration is not None else 1.0
        if duration_s <= 0:
            duration_s = 1.0

        # Dispatch via the CONTROLLER's goto_target — the hardware-verified
        # one-shot path (pre-merge review fold): minjerk-interpolated over
        # ``duration`` (speed governed by time, with the DN movement-velocity
        # predictor as envelope), ships the explicit head matrix composed
        # with the body's ACTUAL yaw per the head-frame invariant, and is
        # NOT subject to maxim.move()'s per-call step clamp (2°/call — a
        # smoothness guard for streaming gaze loops that made a single
        # move() call under-turn a 45° orient by 43°).
        robot = _get_robot_from_registry(None, maxim)
        if robot is None:
            return ToolResult(success=False, error="No robot controller available.")
        try:
            import math as _math

            from maxim.hardware import MotionTarget

            ok = robot.goto_target(MotionTarget(head_yaw=_math.radians(target_yaw), duration=duration_s))
            if not ok:
                return ToolResult(success=False, error="Motion command rejected by controller")
        except Exception as e:
            warn("focus_on_sound failed: %s", e, logger=getattr(maxim, "log", None))
            return ToolResult(success=False, error=str(e))

        # READ THE FRAME BACK (verify-actuation discipline, 2026-08-04): the
        # goto blocks for ``duration``, so the controller's pose after it
        # returns is the ACHIEVED pose. The 2026-08-03 live session showed
        # eight consecutive edge-of-envelope commands where the daemon
        # accepted the goto, nothing moved, and the tool still reported
        # faced_sound=True — an "accepted" dispatch is a promise, not a
        # motion. Best-effort: a controller without get_current_pose (or a
        # failed read) leaves ``achieved`` unknown rather than fabricated.
        achieved_yaw: float | None = None
        reached: bool | None = None
        try:
            get_pose = getattr(robot, "get_current_pose", None)
            pose = get_pose() if callable(get_pose) else None
            # BOTH frames must be present: the controller's joint read is
            # best-effort, so a pose can carry world "yaw" without
            # "body_yaw" — folding a missing body angle to 0 would compute
            # achieved in the wrong frame with the body turned (a false
            # "[FELL SHORT]" by the full body angle: the exact
            # frame-folding class this readback exists to kill). Missing
            # either → unknown, not failed (review fold, 2026-08-04).
            if pose and "yaw" in pose and "body_yaw" in pose:
                world_deg = _math.degrees(float(pose["yaw"]))
                body_deg = _math.degrees(float(pose["body_yaw"]))
                achieved_yaw = world_deg - body_deg
                reached = abs(achieved_yaw - target_yaw) <= self._REACH_TOLERANCE_DEG
        except Exception:
            pass

        # Clamp always negates "faced" (the sound is beyond the neck, period);
        # a CONFIRMED shortfall negates it; an unavailable readback stays
        # optimistic (unknown ≠ failed) but the note says so.
        faced = (not clamped) and (reached is not False)
        side = "left" if az < 0 else ("right" if az > 0 else "ahead")
        # Every aim decision at INFO: the 2026-08-03 live sessions were
        # undiagnosable from the console because the tool's direction math
        # was invisible — only "success=True" showed, whether the turn was
        # toward or away from the sound.
        _log = getattr(maxim, "log", None) or __import__("logging").getLogger(__name__)
        _log.info(
            "focus_on_sound: az=%+.2f (%s, %.1fs old) turning %.1f° → %.1f°%s%s",
            az,
            side,
            age_s,
            cur_yaw,
            target_yaw,
            " [clamped]" if clamped else "",
            (
                f" achieved {achieved_yaw:.1f}°" + ("" if reached else " [FELL SHORT]")
                if achieved_yaw is not None
                else ""
            ),
        )
        # Resolve the REGISTERED body-turn tool name for the note (the
        # tools are entity-prefixed: reachy_mini_turn_left_big). Clamped
        # means the sound is far — recommend the big step on the sound's
        # side. Best-effort: no wired body → generic advice.
        turn_tool: str | None = None
        if clamped or reached is False:
            try:
                _emb = getattr(feed, "_embodiment", None)
                _ent_name = getattr(getattr(_emb, "root", None), "name", None)
                if _ent_name:
                    turn_tool = f"{_ent_name}_turn_{'left' if az < 0 else 'right'}_big"
            except Exception:
                __import__("logging").getLogger(__name__).debug(
                    "focus_on_sound: turn-tool name resolution failed", exc_info=True
                )

        return ToolResult(
            success=True,
            output={
                # Honest by measurement, not by dispatch: True only when the
                # post-motion readback confirms the head is at an UNCLAMPED
                # target. A clamped target means the sound lies beyond the
                # neck's reach — the head points as far as it can, but it is
                # NOT facing the sound, and saying so lets the LLM choose a
                # body-turn strategy instead of re-issuing the same call.
                "faced_sound": faced,
                "azimuth": az,
                "sound_side": side,
                "reading_age_s": round(age_s, 2),
                "from_yaw_deg": round(cur_yaw, 1),
                "to_yaw_deg": round(target_yaw, 1),
                "achieved_yaw_deg": round(achieved_yaw, 1) if achieved_yaw is not None else None,
                "reached_target": reached,
                "clamped_to_head_limit": clamped,
                "note": _focus_result_note(az, clamped, reached, side, turn_tool),
            },
        )


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
        "turn_around",
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
        pause_training = (
            bool(getattr(maxim, "train", False))
            and paused is not None
            and command
            in {
                "center_vision",
                "mark_trainable_moment",
                "label_outcome",
            }
        )

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
            if command == "turn_around":
                # Rotate the body when head is at yaw limits
                try:
                    angle = float(params.get("angle", 90.0))
                except (TypeError, ValueError):
                    angle = 90.0
                try:
                    duration = float(params.get("duration", 5.0))
                except (TypeError, ValueError):
                    duration = 5.0
                recenter_head = bool(params.get("recenter_head", True))
                try:
                    maxim.turn_around(angle, duration=duration, recenter_head=recenter_head)
                except Exception as e:
                    return ToolResult(success=False, error=str(e))
                return ToolResult(
                    success=True,
                    output={
                        "command": command,
                        "angle": angle,
                        "duration": duration,
                        "recenter_head": recenter_head,
                    },
                )

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


# Complete COCO class mapping (all 80 classes)
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
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
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
        clamped_u, clamped_v, was_clamped = clamp_to_vision_range(target_u, target_v, width, height)

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
