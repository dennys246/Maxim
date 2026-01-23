"""Passive observation and display utilities.

Phase 2 refactor: Separates display logic from movement control.
- display_detections(): Pure display function for annotated frames
- passive_observation(): Legacy fallback (display only, no movement)
- Movement decisions are now handled by the agentic runtime
"""

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
) -> dict[str, Any] | None:
    """Display frame with detection annotations.

    This is the pure display function - no movement, no side effects.
    Returns detection info for the agentic system to act on.

    Args:
        frame: BGR image frame
        detections: List of detection tuples from segmenter
        segmenter: Optional segmenter for pose refinement
        window_name: OpenCV window name
        wait_ms: waitKey delay
        show_pose: Whether to compute and display pose info

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

    # Find primary target (prefer people)
    people = []
    for det in detections:
        try:
            if len(det) > 7 and int(det[7]) == 0:
                people.append(det)
        except Exception:
            continue

    if not detections:
        show_frame(frame, window_name=window_name, wait_ms=wait_ms)
        return None

    primary = max(people, key=score_detection_conf_area) if people else max(detections, key=score_detection_conf_area)

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
        },
        "pose_info": pose_info,
    }


def passive_observation(
    maxim,
    photos,
    *,
    show: bool = True,
    window_name: str = "Maxim Observation",
) -> dict[str, Any] | None:
    """Display-only observation (no movement).

    Phase 2: This function no longer controls movement. It only:
    1. Runs segmentation on the frame
    2. Displays annotated results
    3. Returns target info for the agentic system

    Movement decisions are handled by the agentic runtime via tools.

    Args:
        maxim: Maxim instance (for segmenter access)
        photos: Single frame or list of frames
        show: Whether to display the frame
        window_name: OpenCV window name

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
        )

    # Display with annotations
    return display_detections(
        frame,
        candidates,
        segmenter=segmenter,
        window_name=window_name,
        wait_ms=1,
        show_pose=True,
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
