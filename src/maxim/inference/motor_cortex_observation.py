import time, os
import numpy as np

from maxim.data.camera.display import show_frame
from maxim.utils.detections import maybe_scale_normalized_xyxy, score_detection_conf_area
from maxim.utils.logging import warn

def _coerce_model_xy(prediction):
    arr = np.asarray(prediction).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Expected 2 coordinates, got shape {arr.shape}")
    return float(arr[0]), float(arr[1])


def _coerce_pred_vector(prediction):
    arr = np.asarray(prediction).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Empty prediction with shape {np.asarray(prediction).shape}")
    return arr


def _infer_model_hw(movement_model):
    cfg = getattr(movement_model, "config", None)
    input_shape = getattr(cfg, "input_shape", None) if cfg is not None else None

    if isinstance(input_shape, int):
        size = int(input_shape)
        if size > 0:
            return size, size

    if isinstance(input_shape, (list, tuple)):
        shape = tuple(input_shape)
        if len(shape) >= 2:
            try:
                h = int(shape[0])
                w = int(shape[1])
                if h > 0 and w > 0:
                    return h, w
            except (TypeError, ValueError):
                pass

    model_obj = getattr(movement_model, "model", movement_model)
    model_input_shape = getattr(model_obj, "input_shape", None)
    if isinstance(model_input_shape, (list, tuple)) and len(model_input_shape) >= 3:
        h, w = model_input_shape[1], model_input_shape[2]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return h, w

    return None, None


def _range_from_limits(pose_limits, key: str, fallback: float) -> float:
    try:
        lo, hi = pose_limits.get(key, (-fallback / 2.0, fallback / 2.0))
        return abs(float(hi) - float(lo))
    except Exception:
        return float(fallback)


def _pose_range(pose_limits, key: str, default: tuple[float, float]) -> tuple[float, float]:
    try:
        lo, hi = pose_limits.get(key, default)
        return float(lo), float(hi)
    except Exception:
        return default


def motor_cortex_control(
    maxim,
    movement_model,
    photos,
    *,
    duration: float | None = None,
    train: bool = False,
    deadzone_px: int = 20,
    show: bool = True,
    window_name: str = "Maxim Observation",
    callbacks = None):

    if isinstance(photos, np.ndarray):
        photos = [photos]
    if not photos:
        return
    
    photo = photos[-1]

    cfg = getattr(movement_model, "config", None) if movement_model is not None else None

    callback_list = None
    if train and callbacks is not None:
        try:
            from maxim.training.callbacks import as_callback_list

            callback_list = as_callback_list(callbacks)
        except Exception as e:
            callback_list = None
            warn("Failed to initialize callbacks: %s", e, logger=getattr(maxim, "log", None))

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

    # Prefer people (COCO class 0) when present; otherwise fallback to any detection.
    people = []
    for obs in candidates:
        try:
            if len(obs) > 7 and int(obs[7]) == 0:
                people.append(obs)
        except Exception:
            continue

    observation = (
        max(people, key=score_detection_conf_area) if people else max(candidates, key=score_detection_conf_area)
    )

    x1, y1, x2, y2 = observation[2], observation[3], observation[4], observation[5]

    x1, y1, x2, y2, scaled_box = maybe_scale_normalized_xyxy(x1, y1, x2, y2, photo_width, photo_height)
    if scaled_box and getattr(maxim, "verbosity", 0) >= 2 and getattr(maxim, "log", None) is not None:
        try:
            maxim.log.debug("Scaled normalized bbox to pixels: (%.3f, %.3f, %.3f, %.3f)", x1, y1, x2, y2)
        except Exception:
            pass

    is_person = False
    try:
        is_person = len(observation) > 7 and int(observation[7]) == 0
    except Exception:
        is_person = False

    target_method = "bbox"
    pose_box = None
    pose_info = None

    # Calculate point between eyes
    u = float((x1 + x2) / 2)
    v = float((y1 + y2) / 2)

    # If a person is detected, refine target to the midpoint between their eyes using YOLOv8-pose.
    if is_person:
        try:
            pose_info = maxim.segmenter.pose_targets_for_box(photos[frame_ind], (x1, y1, x2, y2))
        except Exception:
            pose_info = None

        if pose_info and "target" in pose_info:
            try:
                u, v = map(float, pose_info["target"])
                target_method = str(pose_info.get("method", "pose"))
                pose_box = pose_info.get("pose_box")
            except Exception:
                pass

    if 0.0 <= float(u) <= 1.0 and 0.0 <= float(v) <= 1.0 and photo_width > 2 and photo_height > 2:
        # Defensive: some pose/keypoint APIs can return normalized coordinates.
        u = float(u) * float(photo_width - 1)
        v = float(v) * float(photo_height - 1)
        if getattr(maxim, "verbosity", 0) >= 2 and getattr(maxim, "log", None) is not None:
            try:
                maxim.log.debug("Scaled normalized target to pixels: (%.3f, %.3f)", u, v)
            except Exception:
                pass

    u_int = int(np.clip(round(u), 1, photo_width - 1))
    v_int = int(np.clip(round(v), 1, photo_height - 1))

    model_h, model_w = _infer_model_hw(movement_model)
    scale_x = 1.0
    scale_y = 1.0

    model_photo = photo
    if model_h is not None and model_w is not None:
        if model_photo.shape[0] != model_h or model_photo.shape[1] != model_w:
            try:
                import cv2

                model_photo = cv2.resize(model_photo, (model_w, model_h), interpolation=cv2.INTER_AREA)
            except Exception:
                model_photo = photo
        try:
            if int(model_w) > 1 and int(photo_width) > 1:
                scale_x = float(int(model_w) - 1) / float(int(photo_width) - 1)
            else:
                scale_x = 1.0
            if int(model_h) > 1 and int(photo_height) > 1:
                scale_y = float(int(model_h) - 1) / float(int(photo_height) - 1)
            else:
                scale_y = 1.0
        except Exception:
            scale_x = 1.0
            scale_y = 1.0

    model_batch = np.expand_dims(model_photo, axis=0)

    final_activation = getattr(cfg, "final_activation", None) if cfg is not None else None

    if duration is None:
        duration = getattr(maxim, "duration", 0.5)
    base_duration = float(duration)

    default_delta_limits = [5.0, 5.0, 10.0, 10.0, 10.0, 10.0]
    delta_limits = getattr(cfg, "movement_delta_limits", None) if cfg is not None else None
    if not isinstance(delta_limits, (list, tuple)) or len(delta_limits) < 6:
        delta_limits = default_delta_limits
    delta_limits = [float(v) for v in list(delta_limits)[:6]]

    default_pose_limits = {
        "x": (-30.0, 30.0),
        "y": (-30.0, 30.0),
        "z": (-60.0, 60.0),
        "roll": (-30.0, 30.0),
        "pitch": (-30.0, 30.0),
        "yaw": (-45.0, 45.0),
    }
    pose_limits = getattr(cfg, "movement_pose_limits", None) if cfg is not None else None
    if not isinstance(pose_limits, dict):
        pose_limits = default_pose_limits

    # Teacher control signal from pixel error (simple proportional mapping).
    # Use the available pose range as a rough proxy for camera FOV so large pixel errors saturate faster.
    center_u = float(photo_width) / 2.0
    center_v = float(photo_height) / 2.0

    yaw_gain = _range_from_limits(pose_limits, "yaw", 90.0)
    pitch_gain = _range_from_limits(pose_limits, "pitch", 60.0)
    teacher_yaw_delta = ((float(u_int) - center_u) / float(photo_width)) * float(yaw_gain)
    teacher_pitch_delta = ((float(v_int) - center_v) / float(photo_height)) * float(pitch_gain)

    pixel_error_px = float(np.hypot(float(u_int) - center_u, float(v_int) - center_v))

    duration_limits = getattr(cfg, "movement_duration_limits", None) if cfg is not None else None
    if not isinstance(duration_limits, (list, tuple)) or len(duration_limits) < 2:
        duration_limits = [0.1, 2.0]
    duration_min = float(duration_limits[0])
    duration_max = float(duration_limits[1])

    pred_mode = "teacher"
    pred_vec = None
    try:
        if movement_model is not None:
            try:
                pred = movement_model.predict(model_batch, verbose=0)
            except TypeError:
                pred = movement_model.predict(model_batch)
            pred_vec = _coerce_pred_vector(pred)
    except Exception:
        pred_vec = None

    # Default command: teacher deltas.
    cmd_dx = 0.0
    cmd_dy = 0.0
    cmd_dz = 0.0
    cmd_droll = 0.0
    cmd_dpitch = float(teacher_pitch_delta)
    cmd_dyaw = float(teacher_yaw_delta)
    cmd_duration = float(np.clip(base_duration, duration_min, duration_max))

    u_pred_frame = None
    v_pred_frame = None

    if pred_vec is not None and pred_vec.size >= 7:
        pred_mode = "movement"
        raw = np.asarray(pred_vec[:7], dtype=float)

        if str(final_activation).lower() == "tanh":
            raw = np.clip(raw, -1.0, 1.0)
            cmd_dx = float(raw[0]) * delta_limits[0]
            cmd_dy = float(raw[1]) * delta_limits[1]
            cmd_dz = float(raw[2]) * delta_limits[2]
            cmd_droll = float(raw[3]) * delta_limits[3]
            cmd_dpitch = float(raw[4]) * delta_limits[4]
            cmd_dyaw = float(raw[5]) * delta_limits[5]
            cmd_duration = duration_min + (float(raw[6]) + 1.0) * 0.5 * (duration_max - duration_min)
        else:
            cmd_dx = float(raw[0])
            cmd_dy = float(raw[1])
            cmd_dz = float(raw[2])
            cmd_droll = float(raw[3])
            cmd_dpitch = float(raw[4])
            cmd_dyaw = float(raw[5])
            cmd_duration = float(raw[6])

        cmd_dx = float(np.clip(cmd_dx, -delta_limits[0], delta_limits[0]))
        cmd_dy = float(np.clip(cmd_dy, -delta_limits[1], delta_limits[1]))
        cmd_dz = float(np.clip(cmd_dz, -delta_limits[2], delta_limits[2]))
        cmd_droll = float(np.clip(cmd_droll, -delta_limits[3], delta_limits[3]))
        cmd_dpitch = float(np.clip(cmd_dpitch, -delta_limits[4], delta_limits[4]))
        cmd_dyaw = float(np.clip(cmd_dyaw, -delta_limits[5], delta_limits[5]))
        cmd_duration = float(np.clip(cmd_duration, duration_min, duration_max))

    elif pred_vec is not None and pred_vec.size >= 2:
        pred_mode = "pixel"
        u_pred, v_pred = _coerce_model_xy(pred_vec)
        if str(final_activation).lower() == "tanh" and model_h is not None and model_w is not None:
            u_pred = (u_pred + 1.0) * 0.5 * float(model_w - 1)
            v_pred = (v_pred + 1.0) * 0.5 * float(model_h - 1)

        u_pred_frame = u_pred / scale_x if scale_x else float(u_int)
        v_pred_frame = v_pred / scale_y if scale_y else float(v_int)
        if not (np.isfinite(u_pred_frame) and np.isfinite(v_pred_frame)):
            u_pred_frame = float(u_int)
            v_pred_frame = float(v_int)

        cmd_dyaw = ((float(u_pred_frame) - center_u) / float(photo_width)) * yaw_gain
        cmd_dpitch = ((float(v_pred_frame) - center_v) / float(photo_height)) * pitch_gain
        cmd_dpitch = float(np.clip(cmd_dpitch, -delta_limits[4], delta_limits[4]))
        cmd_dyaw = float(np.clip(cmd_dyaw, -delta_limits[5], delta_limits[5]))

    # Final safety clamp (all modes).
    cmd_dx = float(np.clip(cmd_dx, -delta_limits[0], delta_limits[0]))
    cmd_dy = float(np.clip(cmd_dy, -delta_limits[1], delta_limits[1]))
    cmd_dz = float(np.clip(cmd_dz, -delta_limits[2], delta_limits[2]))
    cmd_droll = float(np.clip(cmd_droll, -delta_limits[3], delta_limits[3]))
    cmd_dpitch = float(np.clip(cmd_dpitch, -delta_limits[4], delta_limits[4]))
    cmd_dyaw = float(np.clip(cmd_dyaw, -delta_limits[5], delta_limits[5]))
    cmd_duration = float(np.clip(cmd_duration, duration_min, duration_max))

    if not np.all(np.isfinite([cmd_dx, cmd_dy, cmd_dz, cmd_droll, cmd_dpitch, cmd_dyaw, cmd_duration])):
        pred_mode = "teacher"
        cmd_dx = 0.0
        cmd_dy = 0.0
        cmd_dz = 0.0
        cmd_droll = 0.0
        cmd_dpitch = float(np.clip(teacher_pitch_delta, -delta_limits[4], delta_limits[4]))
        cmd_dyaw = float(np.clip(teacher_yaw_delta, -delta_limits[5], delta_limits[5]))
        cmd_duration = float(np.clip(base_duration, duration_min, duration_max))

    if train and movement_model is not None:
        loss_value = None
        try:
            import tensorflow as tf
            from maxim.training import losses

            x = tf.convert_to_tensor(model_batch, dtype=tf.float32)

            step = int(getattr(movement_model, "_train_step", 0) or 0) + 1
            try:
                setattr(movement_model, "_train_step", step)
            except Exception:
                pass

            if callback_list is not None:
                callback_list.on_train_begin(model=movement_model, maxim=maxim)
                callback_list.on_train_step_begin(
                    step,
                    logs={
                        "mode": pred_mode,
                        "u_true": float(u_int),
                        "v_true": float(v_int),
                        "pixel_error_px": float(pixel_error_px),
                        "pitch_delta_true": float(teacher_pitch_delta),
                        "yaw_delta_true": float(teacher_yaw_delta),
                        "x_delta_pred": float(cmd_dx),
                        "y_delta_pred": float(cmd_dy),
                        "z_delta_pred": float(cmd_dz),
                        "roll_delta_pred": float(cmd_droll),
                        "pitch_delta_pred": float(cmd_dpitch),
                        "yaw_delta_pred": float(cmd_dyaw),
                        "duration_pred": float(cmd_duration),
                    },
                    model=movement_model,
                    maxim=maxim,
                )

            optimizer = getattr(movement_model, "optimizer", None)
            if optimizer is None:
                lr = float(getattr(cfg, "learning_rate", 1e-5) or 1e-5) if cfg is not None else 1e-5
                beta_1 = float(getattr(cfg, "beta_1", 0.9) or 0.9) if cfg is not None else 0.9
                beta_2 = float(getattr(cfg, "beta_2", 0.999) or 0.999) if cfg is not None else 0.999
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
                movement_model.optimizer = optimizer

            with tf.GradientTape() as tape:
                y_pred = movement_model(x, training=True)
                try:
                    out_dim = int(y_pred.shape[-1])
                except Exception:
                    out_dim = 0

                if out_dim >= 7:
                    # Train against the teacher deltas (normalized when using tanh head).
                    teacher_dx = 0.0
                    teacher_dy = 0.0
                    teacher_dz = 0.0
                    teacher_droll = 0.0
                    teacher_dpitch = float(teacher_pitch_delta)
                    teacher_dyaw = float(teacher_yaw_delta)
                    teacher_duration = float(np.clip(base_duration, duration_min, duration_max))

                    if str(final_activation).lower() == "tanh":
                        denom = [v if abs(v) > 1e-9 else 1.0 for v in delta_limits]
                        dx_n = np.clip(teacher_dx / denom[0], -1.0, 1.0)
                        dy_n = np.clip(teacher_dy / denom[1], -1.0, 1.0)
                        dz_n = np.clip(teacher_dz / denom[2], -1.0, 1.0)
                        droll_n = np.clip(teacher_droll / denom[3], -1.0, 1.0)
                        dpitch_n = np.clip(teacher_dpitch / denom[4], -1.0, 1.0)
                        dyaw_n = np.clip(teacher_dyaw / denom[5], -1.0, 1.0)
                        if duration_max > duration_min:
                            dur_n = ((teacher_duration - duration_min) / (duration_max - duration_min)) * 2.0 - 1.0
                            dur_n = float(np.clip(dur_n, -1.0, 1.0))
                        else:
                            dur_n = 0.0
                        y_true = tf.constant([[dx_n, dy_n, dz_n, droll_n, dpitch_n, dyaw_n, dur_n]], dtype=tf.float32)
                        y_pred_use = y_pred[:, :7]
                    else:
                        y_true = tf.constant(
                            [[teacher_dx, teacher_dy, teacher_dz, teacher_droll, teacher_dpitch, teacher_dyaw, teacher_duration]],
                            dtype=tf.float32,
                        )
                        y_pred_use = y_pred[:, :7]

                    loss_vec = losses.euclidian_distance(y_true, y_pred_use)
                    loss_value = tf.reduce_mean(loss_vec)
                else:
                    # Legacy 2D (u,v) target training.
                    y_true_u = float(u_int) * scale_x
                    y_true_v = float(v_int) * scale_y
                    if str(final_activation).lower() == "tanh" and model_h is not None and model_w is not None:
                        denom_w = float(model_w - 1) if model_w and model_w > 1 else 1.0
                        denom_h = float(model_h - 1) if model_h and model_h > 1 else 1.0
                        y_true_u = (y_true_u / denom_w) * 2.0 - 1.0
                        y_true_v = (y_true_v / denom_h) * 2.0 - 1.0
                    y_true = tf.constant([[y_true_u, y_true_v]], dtype=tf.float32)
                    y_pred_use = y_pred[:, :2]
                    loss_vec = losses.euclidian_distance(y_true, y_pred_use)
                    loss_value = tf.reduce_mean(loss_vec)

            variables = getattr(movement_model, "trainable_variables", None) or []
            grads = tape.gradient(loss_value, variables)
            grads_and_vars = [(g, v) for g, v in zip(grads, variables) if g is not None]
            if grads_and_vars:
                optimizer.apply_gradients(grads_and_vars)

            try:
                loss_scalar = float(loss_value)
            except Exception:
                loss_scalar = None

            step_logs = {
                "loss": loss_scalar,
                "mode": pred_mode,
                "u_true": float(u_int),
                "v_true": float(v_int),
                "pixel_error_px": float(pixel_error_px),
                "pitch_delta_true": float(teacher_pitch_delta),
                "yaw_delta_true": float(teacher_yaw_delta),
                "x_delta_pred": float(cmd_dx),
                "y_delta_pred": float(cmd_dy),
                "z_delta_pred": float(cmd_dz),
                "roll_delta_pred": float(cmd_droll),
                "pitch_delta_pred": float(cmd_dpitch),
                "yaw_delta_pred": float(cmd_dyaw),
                "duration_pred": float(cmd_duration),
            }
            if u_pred_frame is not None and v_pred_frame is not None:
                step_logs["u_pred"] = float(u_pred_frame)
                step_logs["v_pred"] = float(v_pred_frame)

            if callback_list is not None:
                callback_list.on_train_step_end(step, logs=step_logs, model=movement_model, maxim=maxim)

            history_list = None
            try:
                history_list = getattr(maxim, "motor_history", None)
                if isinstance(history_list, list):
                    history_list.append({"time": time.time(), "step": int(step), **step_logs})
            except Exception:
                history_list = None

            if loss_scalar is not None and isinstance(history_list, list):
                try:
                    from maxim.utils.plotting import update_motor_cortex_loss_plot
                    from maxim.utils.plotting import update_motor_cortex_pixel_error_plot

                    save_dir = getattr(cfg, "save_dir", None) if cfg is not None else None
                    update_motor_cortex_loss_plot(history_list, save_dir=save_dir)
                    update_motor_cortex_pixel_error_plot(history_list, save_dir=save_dir)
                except Exception as e:
                    warn("Failed to update motor plots: %s", e, logger=getattr(maxim, "log", None))

        except Exception as e:
            loss_value = None
            warn("Motor training step failed: %s", e, logger=getattr(maxim, "log", None))


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

        text_lines.append(f"motor: {pred_mode}")
        try:
            text_lines.append(f"err: {float(pixel_error_px):.0f}px")
        except Exception:
            pass
        text_lines.append(f"Δ yaw={cmd_dyaw:.2f}° pitch={cmd_dpitch:.2f}° dur={cmd_duration:.2f}s")
        if pred_mode == "pixel" and u_pred_frame is not None and v_pred_frame is not None:
            try:
                text_lines.append(f"pred px: ({int(round(float(u_pred_frame)))}, {int(round(float(v_pred_frame)))})")
            except Exception:
                pass
        if train and "loss_value" in locals() and loss_value is not None:
            try:
                text_lines.append(f"loss: {float(loss_value):.4f}")
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
            photo,
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

    x_min, x_max = _pose_range(pose_limits, "x", (-30.0, 30.0))
    y_min, y_max = _pose_range(pose_limits, "y", (-30.0, 30.0))
    z_min, z_max = _pose_range(pose_limits, "z", (-60.0, 60.0))
    roll_min, roll_max = _pose_range(pose_limits, "roll", (-30.0, 30.0))
    pitch_min, pitch_max = _pose_range(pose_limits, "pitch", (-30.0, 30.0))
    yaw_min, yaw_max = _pose_range(pose_limits, "yaw", (-45.0, 45.0))

    x_cmd = float(getattr(maxim, "x", 0.0) or 0.0) + float(cmd_dx)
    y_cmd = float(getattr(maxim, "y", 0.0) or 0.0) + float(cmd_dy)
    z_cmd = float(getattr(maxim, "z", 0.0) or 0.0) + float(cmd_dz)
    roll_cmd = float(getattr(maxim, "roll", 0.0) or 0.0) + float(cmd_droll)
    pitch_cmd = float(getattr(maxim, "pitch", 0.0) or 0.0) + float(cmd_dpitch)
    yaw_cmd = float(getattr(maxim, "yaw", 0.0) or 0.0) + float(cmd_dyaw)

    x_cmd = float(np.clip(x_cmd, x_min, x_max))
    y_cmd = float(np.clip(y_cmd, y_min, y_max))
    z_cmd = float(np.clip(z_cmd, z_min, z_max))
    roll_cmd = float(np.clip(roll_cmd, roll_min, roll_max))
    pitch_cmd = float(np.clip(pitch_cmd, pitch_min, pitch_max))
    yaw_cmd = float(np.clip(yaw_cmd, yaw_min, yaw_max))

    try:
        training_logger = getattr(maxim, "_training_logger", None)
        if training_logger is not None:
            sample_id = int(getattr(maxim, "_training_sample_seq", 0) or 0) + 1
            try:
                setattr(maxim, "_training_sample_seq", sample_id)
            except Exception:
                pass

            frame_ts = getattr(maxim, "_last_frame_ts", None)
            run_start_ts = getattr(maxim, "run_start_ts", None)
            t_rel_s = None
            try:
                if frame_ts is not None and run_start_ts is not None:
                    t_rel_s = float(frame_ts) - float(run_start_ts)
            except Exception:
                t_rel_s = None

            record = {
                "kind": "motor_sample",
                "time": time.time(),
                "sample_id": int(sample_id),
                "run_id": getattr(maxim, "run_id", None),
                "mode": getattr(maxim, "mode", None),
                "epoch": int(getattr(maxim, "current_epoch", 0) or 0),
                "frame_ts": float(frame_ts) if frame_ts is not None else None,
                "t_rel_s": t_rel_s,
                "video_path": getattr(maxim, "video_path", None),
                "audio_path": getattr(maxim, "audio_path", None),
                "transcript_path": getattr(maxim, "transcript_path", None),
                "photo": {"width": int(photo_width), "height": int(photo_height)},
                "detection": {
                    "track_id": observation[0] if len(observation) > 0 else None,
                    "class_id": int(observation[7]) if len(observation) > 7 and observation[7] is not None else None,
                    "conf": float(observation[6]) if len(observation) > 6 and observation[6] is not None else None,
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "is_person": bool(is_person),
                },
                "target": {
                    "method": str(target_method),
                    "u": int(u_int),
                    "v": int(v_int),
                    "center_u": float(center_u),
                    "center_v": float(center_v),
                    "pixel_error_px": float(pixel_error_px),
                },
                "teacher": {
                    "yaw_delta": float(teacher_yaw_delta),
                    "pitch_delta": float(teacher_pitch_delta),
                },
                "pred_mode": str(pred_mode),
                "command": {
                    "dx": float(cmd_dx),
                    "dy": float(cmd_dy),
                    "dz": float(cmd_dz),
                    "droll": float(cmd_droll),
                    "dpitch": float(cmd_dpitch),
                    "dyaw": float(cmd_dyaw),
                    "duration_s": float(cmd_duration),
                    "x": float(x_cmd),
                    "y": float(y_cmd),
                    "z": float(z_cmd),
                    "roll": float(roll_cmd),
                    "pitch": float(pitch_cmd),
                    "yaw": float(yaw_cmd),
                },
                "user_marked": False,
            }

            if isinstance(pose_info, dict):
                pose_box_val = pose_info.get("pose_box")
                record["pose"] = {
                    "method": pose_info.get("method"),
                    "iou": pose_info.get("iou"),
                    "conf": pose_info.get("conf"),
                    "pose_box_xyxy": list(pose_box_val) if isinstance(pose_box_val, (list, tuple)) else None,
                }

            try:
                setattr(maxim, "_last_motor_sample", record)
            except Exception:
                pass
            training_logger.log_motor_sample(record)
    except Exception as e:
        warn("Failed to log training sample: %s", e, logger=getattr(maxim, "log", None))

    maxim.move(
        x=x_cmd,
        y=y_cmd,
        z=z_cmd,
        roll=roll_cmd,
        pitch=pitch_cmd,
        yaw=yaw_cmd,
        duration=float(cmd_duration),
    )
