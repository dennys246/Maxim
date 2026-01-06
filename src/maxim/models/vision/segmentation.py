from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from maxim.utils.logging import warn

COCO_KEYPOINTS: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _ensure_bgr(photo: np.ndarray) -> np.ndarray:
    if photo.ndim == 2:
        return cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
    if photo.ndim == 3 and photo.shape[2] == 1:
        return cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
    if photo.ndim == 3 and photo.shape[2] == 4:
        return cv2.cvtColor(photo, cv2.COLOR_BGRA2BGR)
    return photo


def _area_xyxy(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0.0:
        return 0.0

    union = _area_xyxy(a) + _area_xyxy(b) - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


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
    """
    Ensure a YOLO weight file exists at `local_path`.

    Strategy (best-effort, minimal dependencies):
    1) If local_path exists -> use it.
    2) If an alternate path exists -> copy it into local_path and use it.
    3) Try Ultralytics' downloader (requires network) and copy into local_path.
    4) Fallback to `asset_name` so Ultralytics can still use its cache.
    """
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
        warn("Failed to download YOLO weights '%s': %s", asset_name, e)

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

class YOLO8:
    def __init__(self, pose_model: bool = False):
        # Initialize segmentation model.
        repo_root = Path(__file__).resolve().parents[4]
        local_dir = (repo_root / "data" / "models" / "YOLO").as_posix()
        legacy_dir = (repo_root / "experiments" / "models" / "YOLO").as_posix()
        legacy_models_dir = (repo_root / "experiments" / "models").as_posix()
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
        seg_path = _ensure_weight_file(seg_local, seg_name, alternates=[seg_alt] if seg_alt else [])
        self.model = YOLO(seg_path)

        # Thresholds.
        self.conf = 0.5
        self.pose_conf = 0.25
        self.keypoint_conf = 0.25

        # Pose model (lazy-loadable).
        self.pose_model: YOLO | None = None
        self._pose_model_load_error: str | None = None
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
        self._pose_model_path = _ensure_weight_file(pose_local, pose_name, alternates=[pose_alt] if pose_alt else [])

        if pose_model:
            self._try_load_pose_model()

    def _try_load_pose_model(self) -> bool:
        if self.pose_model is not None:
            return True
        if self._pose_model_load_error is not None:
            return False
        try:
            self.pose_model = YOLO(self._pose_model_path)
            return True
        except Exception as e:
            self._pose_model_load_error = str(e)
            warn(
                "Failed to load YOLOv8 pose model ('%s'): %s",
                self._pose_model_path,
                self._pose_model_load_error,
            )
            self.pose_model = None
            return False

    def ensure_pose_model(self) -> bool:
        return self._try_load_pose_model()

    def segment_photos(self, photos, interests = [0, 1, 2, 3, 4], display = False, save_video = False):

        observations: list[list[Any]] = [] # Things of interest
        if photos is None:
            return observations

        if isinstance(photos, np.ndarray):
            photos = [photos]

        for frame_ind, photo in enumerate(photos):
            if not isinstance(photo, np.ndarray):
                continue

            photo = _ensure_bgr(photo)
            if not (photo.ndim == 3 and photo.shape[2] == 3):
                continue

            # Track people in this frame
            try:
                results = self.model.track(
                    photo,
                    classes=interests,
                    conf=self.conf,
                    persist=True,
                    verbose=False,
                )
            except TypeError:
                try:
                    results = self.model.track(
                        photo,
                        classes=interests,
                        conf=self.conf,
                        persist=True,
                    )
                except Exception:
                    results = []
            except Exception:
                results = []

            if not results:
                continue

            boxes = getattr(results[0], "boxes", None)
            if boxes is None:
                continue
            xyxy = getattr(boxes, "xyxy", None)
            if xyxy is None:
                continue

            try:
                xyxy_arr = xyxy.cpu().numpy()
            except Exception:
                xyxy_arr = None

            if xyxy_arr is None:
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


            for box_index in range(box_count):
                row = xyxy_arr[box_index]
                if row is None or len(row) < 4:
                    continue


                x1, y1, x2, y2 = map(float, row[:4])

                track_id = None
                if track_arr is not None and box_index < int(track_arr.shape[0]):
                    try:
                        track_id = int(_scalar(track_arr[box_index]))
                    except Exception:
                        track_id = None

                cls_id = None
                if cls_arr is not None and box_index < int(cls_arr.shape[0]):
                    try:
                        cls_id = int(_scalar(cls_arr[box_index]))
                    except Exception:
                        cls_id = None

                conf = 0.0
                if conf_arr is not None and box_index < int(conf_arr.shape[0]):
                    try:
                        conf = float(_scalar(conf_arr[box_index]))
                    except Exception:
                        conf = 0.0

                # Preserve observation: [track_id, frame_ind, x1, y1, x2, y2, conf, cls_id]
                observations.append([track_id, frame_ind, float(x1), float(y1), float(x2), float(y2), conf, cls_id])


        return observations

    def pose_targets_for_box(
        self,
        photo: np.ndarray,
        box_xyxy: tuple[float, float, float, float],
        *,
        keypoint_conf: float | None = None,
        min_iou: float = 0.1) -> dict[str, Any] | None:
        """
        Run YOLOv8 pose on `photo` and return the best-matching pose for `box_xyxy`.

        Returns a dict containing:
        - method: "eyes" | "nose"
        - target: (x, y)
        - iou: float
        - pose_box: (x1, y1, x2, y2)
        - keypoints: {name: (x, y, conf)}
        """
        if keypoint_conf is None:
            keypoint_conf = self.keypoint_conf

        if not isinstance(photo, np.ndarray):
            return None

        photo = _ensure_bgr(photo)
        if not (photo.ndim == 3 and photo.shape[2] == 3):
            return None

        if not self.ensure_pose_model():
            return None

        try:
            pose_results = self.pose_model.predict(photo, conf=self.pose_conf, verbose=False)  # type: ignore[union-attr]
        except TypeError:
            try:
                pose_results = self.pose_model.predict(photo, conf=self.pose_conf)  # type: ignore[union-attr]
            except Exception:
                return None
        except Exception:
            return None

        if not pose_results:
            return None

        result0 = pose_results[0]
        boxes = getattr(result0, "boxes", None)
        keypoints = getattr(result0, "keypoints", None)
        if boxes is None or keypoints is None:
            return None

        try:
            pose_xyxy = boxes.xyxy.detach().cpu().numpy()
        except Exception:
            return None

        try:
            pose_conf = boxes.conf.detach().cpu().numpy().reshape(-1)
        except Exception:
            pose_conf = None

        try:
            kp_xy = keypoints.xy.detach().cpu().numpy()
        except Exception:
            return None

        kp_conf = None
        if hasattr(keypoints, "conf") and getattr(keypoints, "conf") is not None:
            try:
                kp_conf = keypoints.conf.detach().cpu().numpy()
            except Exception:
                kp_conf = None

        best_iou = 0.0
        best_idx: int | None = None
        tx1, ty1, tx2, ty2 = map(float, box_xyxy)
        target_box = (tx1, ty1, tx2, ty2)

        for i in range(int(pose_xyxy.shape[0])):
            px1, py1, px2, py2 = map(float, pose_xyxy[i][:4])
            iou = _iou_xyxy(target_box, (px1, py1, px2, py2))
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx is None or best_iou < float(min_iou):
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
            if kp_conf is not None:
                try:
                    c = float(kp_conf[i, j])
                except Exception:
                    c = None
            kp_map[name] = (x, y, c)

        def _get(name: str) -> tuple[float, float, float] | None:
            item = kp_map.get(name)
            if item is None:
                return None
            x, y, c = item
            if c is None:
                return None
            if float(c) < float(keypoint_conf):
                return None
            return float(x), float(y), float(c)

        left = _get("left_eye")
        right = _get("right_eye")
        nose = _get("nose")

        if left is not None and right is not None:
            lx, ly, _ = left
            rx, ry, _ = right
            target = ((lx + rx) / 2.0, (ly + ry) / 2.0)
            method = "eyes"
        elif nose is not None:
            nx, ny, _ = nose
            target = (nx, ny)
            method = "nose"
        else:
            return None

        out: dict[str, Any] = {
            "method": method,
            "target": target,
            "iou": float(best_iou),
            "pose_box": (px1, py1, px2, py2),
            "keypoints": kp_map,
        }
        if pose_conf is not None and i < int(pose_conf.shape[0]):
            try:
                out["conf"] = float(pose_conf[i])
            except Exception:
                pass
        return out
