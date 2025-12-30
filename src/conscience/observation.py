import numpy as np
from src.camera.display import show_frame


def passive_observation(
    maxim,
    photos,
    *,
    duration: float | None = None,
    deadzone_px: int = 20,
    show: bool = True,
    window_name: str = "Maxim Observation",
):
    if photos is None:
        return

    if isinstance(photos, np.ndarray):
        photos = [photos]
    if not photos:
        return

    # Segment photos
    observations = maxim.segmenter.segment_photos(photos, interests = maxim.interests)
    if not observations:
        if show:
            show_frame(photos[-1], window_name=window_name, wait_ms=1)
        return

    frame_ind = len(photos) - 1
    photo_height, photo_width = photos[frame_ind].shape[:2]

    # Only act on detections from the most recent frame to avoid chasing stale boxes.
    candidates = [obs for obs in observations if obs[1] == frame_ind]
    if not candidates:
        if show:
            show_frame(photos[frame_ind], window_name=window_name, wait_ms=1)
        return

    def _score(obs):
        x1, y1, x2, y2, conf = obs[2], obs[3], obs[4], obs[5], obs[6]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return (conf, area)

    observation = max(candidates, key=_score)

    x1, y1, x2, y2 = observation[2], observation[3], observation[4], observation[5]
    u = int(np.clip(round((x1 + x2) / 2), 1, photo_width - 1))
    v = int(np.clip(round((y1 + y2) / 2), 1, photo_height - 1))

    if show:
        show_frame(
            photos[frame_ind],
            boxes=candidates,
            target_box=(x1, y1, x2, y2),
            center=(photo_width / 2, photo_height / 2),
            target_point=(u, v),
            window_name=window_name,
            wait_ms=1,
        )

    # Small deadzone to avoid jitter (still displays).
    if abs(u - (photo_width / 2)) < deadzone_px and abs(v - (photo_height / 2)) < deadzone_px:
        return

    if duration is None:
        duration = getattr(maxim, "duration", 0.5)

    # Use the Reachy SDK camera model to look directly at the target pixel when available.
    try:
        maxim.mini.look_at_image(u, v, duration=float(duration), perform_movement=True)
        return
    except Exception:
        pass

    # Fallback: proportional control in image space (less accurate but keeps things moving).
    x_diff = u - (photo_width / 2)
    y_diff = v - (photo_height / 2)
    yaw_delta = (x_diff / photo_width) * 10
    pitch_delta = (y_diff / photo_height) * 10
    maxim.move(
        yaw=getattr(maxim, "yaw", 0.0) + yaw_delta,
        pitch=getattr(maxim, "pitch", 0.0) + pitch_delta,
    )
