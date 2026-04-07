"""Ultralytics vision engine — optional YOLOv8 backend.

Requires the ``ultralytics`` package (AGPL-3.0), which is NOT included in
core dependencies.  Install with::

    pip install "maxim[yolo]"

This engine wraps the existing YOLOv8 detection + ByteTrack tracking and
pose estimation behind the :class:`VisionEngine` interface.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from maxim.models.vision.engine import (
    COCO_KEYPOINTS,
    COCO_NAMES,
    Detection,
    PoseResult,
    VisionEngine,
)

logger = logging.getLogger(__name__)


def _first_existing(paths: list[str]) -> str | None:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _ensure_weight_file(
    local_path: str,
    asset_name: str,
    *,
    alternates: list[str] | None = None,
) -> str:
    """Ensure a YOLO weight file exists at *local_path*."""
    if local_path and os.path.exists(local_path):
        return local_path

    dest_dir = os.path.dirname(local_path) or "."
    os.makedirs(dest_dir, exist_ok=True)

    for alt in alternates or []:
        if alt and os.path.exists(alt):
            try:
                shutil.copy2(alt, local_path)
                return local_path
            except Exception:
                return alt

    try:
        from ultralytics.utils.downloads import attempt_download_asset  # type: ignore

        downloaded = attempt_download_asset(asset_name)
        if downloaded and os.path.exists(downloaded):
            try:
                shutil.copy2(downloaded, local_path)
                return local_path
            except Exception:
                return str(downloaded)
    except Exception as e:
        logger.warning("Failed to download YOLO weights '%s': %s", asset_name, e)

    return asset_name


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        return value.item()
    except Exception:
        pass
    try:
        arr = np.asarray(value)
        if arr.size == 1:
            return arr.reshape(()).item()
    except Exception:
        pass
    return value


class UltralyticsEngine(VisionEngine):
    """YOLOv8 engine using the ``ultralytics`` package.

    .. warning:: Requires ``pip install "maxim[yolo]"`` (AGPL-3.0).

    Args:
        load_pose: Whether to load the pose model immediately.
    """

    def __init__(self, load_pose: bool = False) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "The 'ultralytics' package is required for the YOLO engine. "
                'Install it with: pip install "maxim[yolo]"'
            ) from None

        repo_root = Path(__file__).resolve().parents[4]
        local_dir = (repo_root / "data" / "models" / "YOLO").as_posix()
        legacy_dir = (repo_root / "experiments" / "models" / "YOLO").as_posix()
        legacy_models_dir = (repo_root / "experiments" / "models").as_posix()

        # Segmentation model
        seg_name = "yolov8m-seg.pt"
        seg_local = os.path.join(local_dir, seg_name)
        seg_alt = _first_existing(
            [
                os.path.join(local_dir, "yolo8m-seg.pt"),
                os.path.join(legacy_dir, seg_name),
                os.path.join(legacy_dir, "yolo8m-seg.pt"),
                os.path.join(legacy_models_dir, "yolo8m-seg.pt"),
                os.path.join(legacy_models_dir, "yolov8m-seg.pt"),
                "yolov8m-seg.pt",
                "yolo8m-seg.pt",
            ]
        )
        seg_path = _ensure_weight_file(
            seg_local,
            seg_name,
            alternates=[seg_alt] if seg_alt else [],
        )
        self._model = YOLO(seg_path)

        # Thresholds
        self._conf = 0.5
        self._pose_conf = 0.25
        self._keypoint_conf = 0.25

        # Pose model (lazy-loadable)
        self._pose_model: Any = None
        self._pose_load_error: str | None = None
        pose_name = "yolov8m-pose.pt"
        pose_local = os.path.join(local_dir, pose_name)
        pose_alt = _first_existing(
            [
                os.path.join(local_dir, "yolo8m-pose.pt"),
                os.path.join(legacy_dir, pose_name),
                os.path.join(legacy_dir, "yolo8m-pose.pt"),
                os.path.join(legacy_models_dir, "yolo8m-pose.pt"),
                os.path.join(legacy_models_dir, "yolov8m-pose.pt"),
                "yolov8m-pose.pt",
                "yolo8m-pose.pt",
            ]
        )
        self._pose_model_path = _ensure_weight_file(
            pose_local,
            pose_name,
            alternates=[pose_alt] if pose_alt else [],
        )

        if load_pose:
            self._try_load_pose()

    def _try_load_pose(self) -> bool:
        if self._pose_model is not None:
            return True
        if self._pose_load_error is not None:
            return False
        try:
            from ultralytics import YOLO

            self._pose_model = YOLO(self._pose_model_path)
            return True
        except Exception as e:
            self._pose_load_error = str(e)
            logger.warning("Failed to load YOLOv8 pose model: %s", e)
            return False

    # ── VisionEngine interface ───────────────────────────────────────────

    def detect_and_track(
        self,
        frames: list[np.ndarray],
        conf: float = 0.5,
    ) -> list[list[Detection]]:
        all_detections: list[list[Detection]] = []

        for frame_idx, photo in enumerate(frames):
            if not isinstance(photo, np.ndarray):
                all_detections.append([])
                continue
            if photo.ndim == 2:
                photo = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
            elif photo.ndim == 3 and photo.shape[2] == 4:
                photo = cv2.cvtColor(photo, cv2.COLOR_BGRA2BGR)
            if not (photo.ndim == 3 and photo.shape[2] == 3):
                all_detections.append([])
                continue

            try:
                results = self._model.track(
                    photo,
                    conf=conf,
                    persist=True,
                    verbose=False,
                )
            except TypeError:
                try:
                    results = self._model.track(
                        photo,
                        conf=conf,
                        persist=True,
                    )
                except Exception:
                    results = []
            except Exception:
                results = []

            if not results:
                all_detections.append([])
                continue

            boxes = getattr(results[0], "boxes", None)
            if boxes is None:
                all_detections.append([])
                continue
            xyxy = getattr(boxes, "xyxy", None)
            if xyxy is None:
                all_detections.append([])
                continue

            try:
                xyxy_arr = xyxy.cpu().numpy()
            except Exception:
                all_detections.append([])
                continue

            if xyxy_arr.ndim == 1:
                xyxy_arr = xyxy_arr.reshape(1, -1)

            try:
                box_count = int(xyxy_arr.shape[0])
            except Exception:
                box_count = 0

            conf_raw = getattr(boxes, "conf", None)
            cls_raw = getattr(boxes, "cls", None)
            track_raw = getattr(boxes, "id", None)

            try:
                conf_arr = conf_raw.cpu().numpy().reshape(-1) if conf_raw is not None else None
            except Exception:
                conf_arr = None
            try:
                cls_arr = cls_raw.cpu().numpy().reshape(-1) if cls_raw is not None else None
            except Exception:
                cls_arr = None
            try:
                track_arr = track_raw.cpu().numpy().reshape(-1) if track_raw is not None else None
            except Exception:
                track_arr = None

            dets: list[Detection] = []
            for i in range(box_count):
                row = xyxy_arr[i]
                if row is None or len(row) < 4:
                    continue

                x1, y1, x2, y2 = map(float, row[:4])

                track_id = None
                if track_arr is not None and i < int(track_arr.shape[0]):
                    try:
                        track_id = int(_scalar(track_arr[i]))
                    except Exception:
                        track_id = None

                cls_id = None
                if cls_arr is not None and i < int(cls_arr.shape[0]):
                    try:
                        cls_id = int(_scalar(cls_arr[i]))
                    except Exception:
                        cls_id = None

                det_conf = 0.0
                if conf_arr is not None and i < int(conf_arr.shape[0]):
                    try:
                        det_conf = float(_scalar(conf_arr[i]))
                    except Exception:
                        det_conf = 0.0

                dets.append(
                    Detection(
                        track_id=track_id,
                        frame_index=frame_idx,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=det_conf,
                        class_id=cls_id if cls_id is not None else 0,
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
        if not self._try_load_pose():
            return None

        if not isinstance(frame, np.ndarray):
            return None
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if not (frame.ndim == 3 and frame.shape[2] == 3):
            return None

        try:
            pose_results = self._pose_model.predict(
                frame,
                conf=pose_conf,
                verbose=False,
            )
        except TypeError:
            try:
                pose_results = self._pose_model.predict(frame, conf=pose_conf)
            except Exception:
                return None
        except Exception:
            return None

        if not pose_results:
            return None

        result0 = pose_results[0]
        boxes_obj = getattr(result0, "boxes", None)
        kp_obj = getattr(result0, "keypoints", None)
        if boxes_obj is None or kp_obj is None:
            return None

        try:
            pose_xyxy = boxes_obj.xyxy.detach().cpu().numpy()
        except Exception:
            return None
        try:
            pose_conf_arr = boxes_obj.conf.detach().cpu().numpy().reshape(-1)
        except Exception:
            pose_conf_arr = None
        try:
            kp_xy = kp_obj.xy.detach().cpu().numpy()
        except Exception:
            return None

        kp_conf_arr = None
        if hasattr(kp_obj, "conf") and getattr(kp_obj, "conf") is not None:
            try:
                kp_conf_arr = kp_obj.conf.detach().cpu().numpy()
            except Exception:
                kp_conf_arr = None

        # Find best IoU match
        def _iou(a: tuple, b: tuple) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                return 0.0
            aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = aa + ab - inter
            return inter / union if union > 0 else 0.0

        best_iou = 0.0
        best_idx: int | None = None
        for i in range(int(pose_xyxy.shape[0])):
            px1, py1, px2, py2 = map(float, pose_xyxy[i][:4])
            iou = _iou(bbox_xyxy, (px1, py1, px2, py2))
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx is None or best_iou < min_iou:
            return None

        i = best_idx
        px1, py1, px2, py2 = map(float, pose_xyxy[i][:4])

        kp_map: dict[str, tuple[float, float, float | None]] = {}
        for j, name in enumerate(COCO_KEYPOINTS):
            try:
                x, y = map(float, kp_xy[i, j, :2])
            except Exception:
                continue
            c = None
            if kp_conf_arr is not None:
                try:
                    c = float(kp_conf_arr[i, j])
                except Exception:
                    c = None
            kp_map[name] = (x, y, c)

        def _get(name: str) -> tuple[float, float, float] | None:
            item = kp_map.get(name)
            if item is None:
                return None
            x, y, c = item
            if c is None or c < keypoint_conf:
                return None
            return x, y, c

        left = _get("left_eye")
        right = _get("right_eye")
        nose = _get("nose")

        if left is not None and right is not None:
            target = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
            method = "eyes"
        elif nose is not None:
            target = (nose[0], nose[1])
            method = "nose"
        else:
            return None

        confidence = None
        if pose_conf_arr is not None and i < int(pose_conf_arr.shape[0]):
            try:
                confidence = float(pose_conf_arr[i])
            except Exception:
                pass

        return PoseResult(
            method=method,
            target=target,
            iou=best_iou,
            pose_box=(px1, py1, px2, py2),
            keypoints=kp_map,
            confidence=confidence,
        )

    def warmup(self, *, include_pose: bool = True) -> None:
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            self._model.predict(dummy, conf=0.5, verbose=False)
        except Exception:
            pass
        if include_pose and self._pose_model is not None:
            try:
                self._pose_model.predict(dummy, conf=0.25, verbose=False)
            except Exception:
                pass

    @property
    def class_names(self) -> dict[int, str]:
        names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            return names
        return COCO_NAMES
