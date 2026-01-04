import time, os
import numpy as np

from maxim.data.camera.display import ensure_bgr, show_frame
from maxim.utils.detections import score_detection_conf_area
from maxim.utils.logging import warn
from scipy.io.wavfile import write


def passive_observation(
    maxim,
    photos,
    *,
    duration: float | None = None,
    deadzone_px: int = 20,
    show: bool = True,
    window_name: str = "Maxim Observation",
):

    if isinstance(photos, np.ndarray):
        photos = [photos]
    if not photos:
        return

    # Segment photos
    observations = maxim.segmenter.segment_photos(photos, interests=maxim.interests)
    if not observations:
        if show:
            show_frame(photos[-1], window_name=window_name, wait_ms=1)
        return

    frame_ind = len(photos) - 1
    photo = photos[frame_ind]
    if not isinstance(photo, np.ndarray):
        return
    photo = ensure_bgr(photo)
    photo_height, photo_width = photo.shape[:2]

    # Only act on detections from the most recent frame to avoid chasing stale boxes.
    candidates = [obs for obs in observations if obs[1] == frame_ind]
    if not candidates:
        if show:
            show_frame(photo, window_name=window_name, wait_ms=1)
        return

    # Prefer people (COCO class 0) when present; otherwise fallback to any detection.
    people = []
    for obs in candidates:
        try:
            if len(obs) > 7 and int(obs[7]) == 0:
                people.append(obs)
        except Exception:
            continue

    observation = max(people, key=score_detection_conf_area) if people else max(candidates, key=score_detection_conf_area)

    x1, y1, x2, y2 = observation[2], observation[3], observation[4], observation[5]

    is_person = False
    try:
        is_person = len(observation) > 7 and int(observation[7]) == 0
    except Exception:
        is_person = False

    target_method = "bbox"
    pose_box = None
    pose_info = None

    u = float((x1 + x2) / 2)
    v = float((y1 + y2) / 2)

    # If a person is detected, refine target to the midpoint between their eyes using YOLOv8-pose.
    if is_person:
        try:
            pose_info = maxim.segmenter.pose_targets_for_box(photo, (x1, y1, x2, y2))
        except Exception:
            pose_info = None

        if pose_info and "target" in pose_info:
            try:
                u, v = map(float, pose_info["target"])
                target_method = str(pose_info.get("method", "pose"))
                pose_box = pose_info.get("pose_box")
            except Exception:
                pass

    u_int = int(np.clip(round(u), 1, photo_width - 1))
    v_int = int(np.clip(round(v), 1, photo_height - 1))

    if show:
        boxes = list(candidates)
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
                target_box = (x1, y1, x2, y2)

        show_frame(
            photos[frame_ind],
            boxes=boxes,
            target_box=target_box,
            center=(photo_width / 2, photo_height / 2),
            target_point=(u_int, v_int),
            text_lines=text_lines if text_lines else None,
            window_name=window_name,
            wait_ms=1,
        )

    # Small deadzone to avoid jitter (still displays).
    if abs(u_int - (photo_width / 2)) < deadzone_px and abs(v_int - (photo_height / 2)) < deadzone_px:
        return

    if duration is None:
        duration = getattr(maxim, "duration", 0.5)

    # Use the Reachy SDK camera model to look directly at the target pixel when available.
    try:
        maxim.look_at_image(u_int, v_int, duration=float(duration), perform_movement=True)
        return
    except Exception:
        pass

    # Fallback: proportional control in image space (less accurate but keeps things moving).
    x_diff = u_int - (photo_width / 2)
    y_diff = v_int - (photo_height / 2)
    yaw_delta = (x_diff / photo_width) * 10
    pitch_delta = (y_diff / photo_height) * 10
    maxim.move(
        yaw=getattr(maxim, "yaw", 0.0) + yaw_delta,
        pitch=getattr(maxim, "pitch", 0.0) + pitch_delta,
    )


def passive_listening(maxim, save_file):
    sample = maxim.listen()

    if save_file:
        # Save audio samples to file
        os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
        try:
            write(save_file, maxim.mini.media.get_output_audio_samplerate(), sample)
        except Exception as e:
            warn("Failed to write audio to '%s': %s", save_file, e, logger=maxim.log)
    # Return audio samples
    return sample
