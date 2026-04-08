"""RTM vision engine — ONNX Runtime inference for RTMDet + RTMPose.

Uses pre-exported ONNX models from OpenMMLab (Apache 2.0 licensed) and
runs inference with ``onnxruntime`` (MIT licensed).  No PyTorch, mmcv,
mmengine, mmdet, or mmpose packages are required at runtime.

Models:
    RTMDet-m   — 80-class COCO object detection, 640×640 input, ~49.4 mAP
    RTMPose-m  — 17 COCO keypoint pose estimation, 256×192 input, ~75.8 AP
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np


def _import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required for vision features. "
            "Install with: pip install pymaxim[vision]"
        ) from None

from maxim.models.vision.engine import (
    COCO_KEYPOINTS,
    COCO_NAMES,
    Detection,
    PoseResult,
    VisionEngine,
)
from maxim.models.vision.tracker import IoUTracker

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RTMDet preprocessing / postprocessing
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet-style normalisation used by RTMDet (BGR order)
_DET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
_DET_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)


def _det_preprocess(
    img: np.ndarray,
    input_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Letterbox-resize and normalise a BGR image for RTMDet.

    Returns:
        (blob, ratio, (pad_w, pad_h))
        where *blob* has shape (1, 3, H, W) float32.
    """
    cv2 = _import_cv2()
    h, w = img.shape[:2]
    target_h, target_w = input_size
    ratio = min(target_h / h, target_w / w)

    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size, centred
    pad_w = (target_w - new_w) / 2.0
    pad_h = (target_h - new_h) / 2.0
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    # Ensure exact target size
    padded = padded[:target_h, :target_w]

    # Normalise (BGR order, float32)
    blob = padded.astype(np.float32)
    blob = (blob - _DET_MEAN) / _DET_STD
    # HWC → CHW → NCHW
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(blob), ratio, (pad_w, pad_h)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS.  Returns indices to keep."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def _det_postprocess(
    outputs: np.ndarray,
    ratio: float,
    pad_wh: tuple[float, float],
    orig_shape: tuple[int, int],
    conf_thr: float = 0.5,
    iou_thr: float = 0.45,
    max_det: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode RTMDet ONNX output into boxes, scores, class_ids.

    Handles both end2end exports (NMS baked in, last dim == 5) and raw
    exports (last dim == 4, requiring grid decode + NMS).

    Returns:
        boxes (N, 4) xyxy in original image coords,
        scores (N,),
        class_ids (N,) int.
    """
    pad_w, pad_h = pad_wh
    oh, ow = orig_shape

    # ── End-to-end export (NMS already applied) ─────────────────────────
    if outputs.ndim == 3 and outputs.shape[-1] == 5:
        # Shape: (batch, num_dets, 5) → [x1, y1, x2, y2, score]
        dets = outputs[0]
        scores = dets[:, 4]
        mask = scores > conf_thr
        dets = dets[mask]
        scores = scores[mask]
        if len(dets) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
        boxes = dets[:, :4]
        # Undo letterbox
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh)
        # End-to-end exports are single-class (person) — class 0
        class_ids = np.zeros(len(boxes), dtype=int)
        return boxes, scores, class_ids

    # ── Raw export (no baked NMS) ────────────────────────────────────────
    # Expected shape: (1, num_anchors, 4 + num_classes)  or
    #                 (1, 4 + num_classes, num_anchors) — transpose if needed
    pred = outputs[0] if outputs.ndim == 3 else outputs
    if pred.ndim == 2:
        pass  # (num_anchors, features)
    elif pred.shape[0] < pred.shape[1]:
        # (features, num_anchors) → transpose
        pred = pred.T

    num_features = pred.shape[1]
    if num_features <= 4:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # Columns: cx, cy, w, h, class_scores...
    cx = pred[:, 0]
    cy = pred[:, 1]
    w = pred[:, 2]
    h = pred[:, 3]
    class_scores = pred[:, 4:]

    # Best class per anchor
    class_ids = class_scores.argmax(axis=1)
    best_scores = class_scores[np.arange(len(class_scores)), class_ids]

    # Confidence filter
    mask = best_scores > conf_thr
    if not mask.any():
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    best_scores = best_scores[mask]
    class_ids = class_ids[mask]

    # cxcywh → xyxy
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Undo letterbox
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh)

    # Per-class NMS (offset trick: shift boxes per class to avoid inter-class suppression)
    max_dim = max(ow, oh)
    offset_boxes = boxes.copy()
    offset_boxes[:, [0, 2]] += class_ids * max_dim
    offset_boxes[:, [1, 3]] += class_ids * max_dim
    keep = _nms(offset_boxes, best_scores, iou_thr)
    keep = keep[:max_det]

    return boxes[keep], best_scores[keep], class_ids[keep].astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# RTMPose preprocessing / postprocessing
# ─────────────────────────────────────────────────────────────────────────────

_POSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_POSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _bbox_xyxy2cs(
    bbox: np.ndarray,
    padding: float = 1.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert xyxy bbox to centre + scale representation.

    Args:
        bbox: (4,) array [x1, y1, x2, y2].
        padding: Scale multiplier for context around the box.

    Returns:
        center (2,), scale (2,)  — scale = padded (w, h).
    """
    x1, y1, x2, y2 = bbox[:4]
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    w = (x2 - x1) * padding
    h = (y2 - y1) * padding
    scale = np.array([w, h], dtype=np.float32)
    return center, scale


def _get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Compute 2×3 affine matrix that maps *center*/*scale* ROI → *output_size*."""
    cv2 = _import_cv2()
    dst_w, dst_h = output_size
    src_w, src_h = scale

    src = np.array(
        [[center[0], center[1]], [center[0], center[1] - src_h * 0.5], [center[0] + src_w * 0.5, center[1]]],
        dtype=np.float32,
    )
    dst = np.array(
        [[dst_w * 0.5, dst_h * 0.5], [dst_w * 0.5, 0.0], [dst_w, dst_h * 0.5]],
        dtype=np.float32,
    )
    return cv2.getAffineTransform(src, dst)


def _pose_preprocess(
    img: np.ndarray,
    bbox: np.ndarray,
    input_size: tuple[int, int] = (192, 256),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop + resize + normalise a person region for RTMPose.

    Args:
        img: Full BGR image.
        bbox: (4,) xyxy bounding box.
        input_size: (width, height) for the pose model.

    Returns:
        (blob, center, scale)
    """
    cv2 = _import_cv2()
    center, scale = _bbox_xyxy2cs(bbox, padding=1.25)
    warp_mat = _get_warp_matrix(center, scale, input_size)

    cropped = cv2.warpAffine(
        img,
        warp_mat,
        input_size,
        flags=cv2.INTER_LINEAR,
    )
    blob = cropped.astype(np.float32)
    blob = (blob - _POSE_MEAN) / _POSE_STD
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(blob), center, scale


def _simcc_decode(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    simcc_split_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode SimCC heatmaps into keypoint coordinates and scores.

    Args:
        simcc_x: (N, K, Wx) logits along x-axis.
        simcc_y: (N, K, Wy) logits along y-axis.

    Returns:
        locs (N, K, 2) in model-input pixel space,
        scores (N, K).
    """
    n, k, _ = simcc_x.shape
    flat_x = simcc_x.reshape(n * k, -1)
    flat_y = simcc_y.reshape(n * k, -1)

    x_locs = np.argmax(flat_x, axis=1).astype(np.float32)
    y_locs = np.argmax(flat_y, axis=1).astype(np.float32)

    max_val_x = np.amax(flat_x, axis=1)
    max_val_y = np.amax(flat_y, axis=1)
    scores = 0.5 * (max_val_x + max_val_y)

    locs = np.stack([x_locs, y_locs], axis=-1)
    locs[scores <= 0.0] = -1
    locs = locs.reshape(n, k, 2) / simcc_split_ratio

    return locs, scores.reshape(n, k)


def _pose_postprocess(
    locs: np.ndarray,
    scores: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map keypoint coordinates from model space back to original image.

    Args:
        locs: (N, K, 2) keypoint locations in model-input pixel coords.
        scores: (N, K) keypoint confidence.
        center: (2,) bbox centre in image coords.
        scale: (2,) bbox scale (padded w, h).
        input_size: (width, height) of model input.

    Returns:
        keypoints (N, K, 2) in image coords,
        scores (N, K).
    """
    inp_w, inp_h = input_size
    kp = locs.copy()
    # Scale from model input to bbox ROI
    kp[..., 0] = kp[..., 0] / inp_w * scale[0]
    kp[..., 1] = kp[..., 1] / inp_h * scale[1]
    # Translate to image coords
    kp[..., 0] += center[0] - scale[0] / 2.0
    kp[..., 1] += center[1] - scale[1] / 2.0
    return kp, scores


# ─────────────────────────────────────────────────────────────────────────────
# ONNX session wrapper
# ─────────────────────────────────────────────────────────────────────────────


class _OnnxSession:
    """Thin wrapper around ``onnxruntime.InferenceSession``."""

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort

        providers = []
        available = ort.get_available_providers()
        for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"):
            if p in available:
                providers.append(p)
        if not providers:
            providers = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.log_severity_level = 3  # suppress ORT info/warning spam

        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def run(self, blob: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: blob})


# ─────────────────────────────────────────────────────────────────────────────
# Model path resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_model_path(filename: str, model_dir: str) -> str:
    """Locate an ONNX model file, downloading it if necessary.

    Search order:
    1. ``model_dir/filename``
    2. ``data/models/YOLO/filename``  (legacy path)
    3. Auto-download via ``maxim.models.download``

    Raises:
        FileNotFoundError: If the model cannot be found or downloaded.
    """
    local = os.path.join(model_dir, filename)
    if os.path.exists(local):
        return local

    # Legacy / alternate paths — check ~/.maxim/models/ tree
    from maxim.utils.paths import model_dir as _model_dir
    _mdir = _model_dir()
    for subdir in ("YOLO", "vision"):
        alt = str(_mdir / subdir / filename)
        if os.path.exists(alt):
            return alt

    # Attempt auto-download
    try:
        from maxim.models.download import download_vision

        dest_dir = Path(model_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        if download_vision(filename.replace(".onnx", ""), str(dest_dir)):
            if os.path.exists(local):
                return local
    except Exception as exc:
        logger.warning("Auto-download failed for %s: %s", filename, exc)

    raise FileNotFoundError(
        f"Vision model '{filename}' not found at {local}. Run: python -m maxim.models.download --vision"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RTMEngine
# ─────────────────────────────────────────────────────────────────────────────


class RTMEngine(VisionEngine):
    """ONNX Runtime engine using RTMDet (detection) + RTMPose (pose).

    All model weights are Apache 2.0 licensed from OpenMMLab.
    Inference uses only ``onnxruntime`` (MIT) — no PyTorch or mmcv needed.

    Args:
        det_model: Filename of the RTMDet ONNX model.
        pose_model: Filename of the RTMPose ONNX model.
        model_dir: Directory containing the ONNX files.
        load_pose: Whether to load the pose model immediately.
        det_input_size: (H, W) input size for the detection model.
        pose_input_size: (W, H) input size for the pose model.
    """

    def __init__(
        self,
        det_model: str = "rtmdet-m.onnx",
        pose_model: str = "rtmpose-m.onnx",
        model_dir: str | None = None,
        load_pose: bool = False,
        det_input_size: tuple[int, int] = (640, 640),
        pose_input_size: tuple[int, int] = (192, 256),
    ) -> None:
        if model_dir is None:
            from maxim.utils.paths import model_dir as _model_dir
            model_dir = str(_model_dir() / "YOLO")

        det_path = _resolve_model_path(det_model, model_dir)
        self._det_session = _OnnxSession(det_path)
        self._det_input_size = det_input_size

        self._pose_session: _OnnxSession | None = None
        self._pose_model_name = pose_model
        self._pose_model_dir = model_dir
        self._pose_input_size = pose_input_size
        self._pose_load_error: str | None = None

        self._tracker = IoUTracker(iou_threshold=0.3, max_age=30)

        if load_pose:
            self._load_pose()

    def _load_pose(self) -> bool:
        if self._pose_session is not None:
            return True
        if self._pose_load_error is not None:
            return False
        try:
            path = _resolve_model_path(self._pose_model_name, self._pose_model_dir)
            self._pose_session = _OnnxSession(path)
            return True
        except Exception as exc:
            self._pose_load_error = str(exc)
            logger.warning("Failed to load pose model: %s", exc)
            return False

    # ── VisionEngine interface ───────────────────────────────────────────

    def detect_and_track(
        self,
        frames: list[np.ndarray],
        conf: float = 0.5,
    ) -> list[list[Detection]]:
        cv2 = _import_cv2()
        all_detections: list[list[Detection]] = []

        for frame_idx, frame in enumerate(frames):
            if not isinstance(frame, np.ndarray):
                all_detections.append([])
                continue
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if not (frame.ndim == 3 and frame.shape[2] == 3):
                all_detections.append([])
                continue

            blob, ratio, pad_wh = _det_preprocess(frame, self._det_input_size)

            try:
                outputs = self._det_session.run(blob)
            except Exception:
                all_detections.append([])
                continue

            raw = outputs[0]
            boxes, scores, cls_ids = _det_postprocess(
                raw,
                ratio,
                pad_wh,
                frame.shape[:2],
                conf_thr=conf,
            )

            if len(boxes) == 0:
                all_detections.append([])
                continue

            track_ids = self._tracker.update(boxes, scores, cls_ids)

            dets: list[Detection] = []
            for i in range(len(boxes)):
                dets.append(
                    Detection(
                        track_id=int(track_ids[i]),
                        frame_index=frame_idx,
                        x1=float(boxes[i, 0]),
                        y1=float(boxes[i, 1]),
                        x2=float(boxes[i, 2]),
                        y2=float(boxes[i, 3]),
                        confidence=float(scores[i]),
                        class_id=int(cls_ids[i]),
                    )
                )
            all_detections.append(dets)

        return all_detections

    def estimate_pose(
        self,
        frame: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        *,
        pose_conf: float = 0.25,
        keypoint_conf: float = 0.25,
        min_iou: float = 0.1,
    ) -> PoseResult | None:
        cv2 = _import_cv2()
        if not self._load_pose():
            return None

        if not isinstance(frame, np.ndarray):
            return None
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if not (frame.ndim == 3 and frame.shape[2] == 3):
            return None

        # First, detect all people in the frame
        blob_det, ratio, pad_wh = _det_preprocess(frame, self._det_input_size)
        try:
            det_out = self._det_session.run(blob_det)
        except Exception:
            return None

        det_boxes, det_scores, det_cls = _det_postprocess(
            det_out[0],
            ratio,
            pad_wh,
            frame.shape[:2],
            conf_thr=pose_conf,
        )
        # Filter to person class (0)
        person_mask = det_cls == 0
        if not person_mask.any():
            return None
        person_boxes = det_boxes[person_mask]
        person_scores = det_scores[person_mask]

        # Find the person box with best IoU to the target bbox
        target = np.array(bbox_xyxy, dtype=np.float32)
        from maxim.models.vision.tracker import _iou_matrix

        ious = _iou_matrix(target.reshape(1, 4), person_boxes).flatten()
        best_idx = int(ious.argmax())
        best_iou = float(ious[best_idx])

        if best_iou < min_iou:
            return None

        best_box = person_boxes[best_idx]

        # Run pose estimation on the best person box
        blob_pose, center, scale = _pose_preprocess(
            frame,
            best_box,
            self._pose_input_size,
        )
        try:
            pose_out = self._pose_session.run(blob_pose)  # type: ignore[union-attr]
        except Exception:
            return None

        # SimCC decode: outputs are [simcc_x, simcc_y]
        simcc_x, simcc_y = pose_out[0], pose_out[1]
        locs, scores = _simcc_decode(simcc_x, simcc_y)
        kp_img, kp_scores = _pose_postprocess(
            locs,
            scores,
            center,
            scale,
            self._pose_input_size,
        )

        # Build keypoints dict (first person in batch — N=1)
        kp_map: dict[str, tuple[float, float, float | None]] = {}
        for j, name in enumerate(COCO_KEYPOINTS):
            x, y = float(kp_img[0, j, 0]), float(kp_img[0, j, 1])
            c = float(kp_scores[0, j])
            kp_map[name] = (x, y, c)

        # Select gaze target (eyes midpoint or nose)
        def _kp(name: str) -> tuple[float, float, float] | None:
            item = kp_map.get(name)
            if item is None:
                return None
            x, y, c = item
            if c is None or c < keypoint_conf:
                return None
            return x, y, c

        left = _kp("left_eye")
        right = _kp("right_eye")
        nose = _kp("nose")

        if left is not None and right is not None:
            target_pt = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
            method = "eyes"
        elif nose is not None:
            target_pt = (nose[0], nose[1])
            method = "nose"
        else:
            return None

        return PoseResult(
            method=method,
            target=target_pt,
            iou=best_iou,
            pose_box=(float(best_box[0]), float(best_box[1]), float(best_box[2]), float(best_box[3])),
            keypoints=kp_map,
            confidence=float(person_scores[best_idx]),
        )

    def warmup(self, *, include_pose: bool = True) -> None:
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        blob, _, _ = _det_preprocess(dummy, self._det_input_size)
        try:
            self._det_session.run(blob)
        except Exception:
            pass

        if include_pose and self._pose_session is not None:
            bbox = np.array([10, 10, 54, 54], dtype=np.float32)
            blob_p, _, _ = _pose_preprocess(dummy, bbox, self._pose_input_size)
            try:
                self._pose_session.run(blob_p)
            except Exception:
                pass

    @property
    def class_names(self) -> dict[int, str]:
        return COCO_NAMES
