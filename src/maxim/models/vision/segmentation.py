"""Segmentation model adapter — delegates to a pluggable VisionEngine.

This module preserves the ``YOLO8`` class name and the exact
``segment_photos()`` / ``pose_targets_for_box()`` interface used by the
rest of the codebase so that no downstream code needs to change.

The actual inference is performed by a :class:`VisionEngine` backend
(``RTMEngine`` by default, ``UltralyticsEngine`` with ``pip install maxim[yolo]``).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from maxim.models.vision.engine import VisionEngine


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (preserved from original module — used downstream)
# ─────────────────────────────────────────────────────────────────────────────


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


def _iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
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


# ─────────────────────────────────────────────────────────────────────────────
# Namespace proxy for backward compatibility
# ─────────────────────────────────────────────────────────────────────────────


class _NamespaceProxy:
    """Expose engine attributes via dotted access (e.g. ``model.names``)."""

    def __init__(self, engine: VisionEngine) -> None:
        self._engine = engine

    @property
    def names(self) -> dict[int, str]:
        return self._engine.class_names


# ─────────────────────────────────────────────────────────────────────────────
# YOLO8 — thin adapter around any VisionEngine
# ─────────────────────────────────────────────────────────────────────────────


class YOLO8:
    """Object detection + pose estimation adapter.

    Delegates all inference to a :class:`VisionEngine` backend while
    preserving the legacy list-based output format consumed by the rest
    of the codebase.

    Args:
        pose_model: Whether to eagerly load the pose model.
        engine: A ``VisionEngine`` instance.  When ``None`` the default
            RTMEngine is created automatically.
    """

    def __init__(
        self,
        pose_model: bool = False,
        engine: VisionEngine | None = None,
    ) -> None:
        if engine is None:
            from maxim.models.vision.rtm_engine import RTMEngine

            engine = RTMEngine(load_pose=pose_model)

        self._engine = engine

        # Backward compat: some code accesses ``segmenter.model.names``
        self.model = _NamespaceProxy(engine)

        # Thresholds (kept for callers that read them directly)
        self.conf: float = 0.5
        self.pose_conf: float = 0.25
        self.keypoint_conf: float = 0.25

    # Expose engine directly for callers that need it
    @property
    def engine(self) -> VisionEngine:
        return self._engine

    def ensure_pose_model(self) -> bool:
        """Attempt to load the pose model.  Returns ``True`` on success."""
        try:
            self._engine.warmup(include_pose=True)
            return True
        except Exception:
            return False

    def warmup(self, *, include_pose: bool = True) -> None:
        """Run dummy inference to reduce first-use latency."""
        self._engine.warmup(include_pose=include_pose)

    # ── Detection + tracking (legacy list format) ────────────────────────

    def segment_photos(
        self,
        photos: Any,
        interests: Any = None,
        display: bool = False,
        save_video: bool = False,
    ) -> list[list[Any]]:
        """Run detection + tracking and return the legacy observation lists.

        Each inner list: ``[track_id, frame_ind, x1, y1, x2, y2, conf, cls_id]``
        """
        if photos is None:
            return []

        if isinstance(photos, np.ndarray):
            photos = [photos]

        frames = [_ensure_bgr(p) for p in photos if isinstance(p, np.ndarray)]
        if not frames:
            return []

        frame_dets = self._engine.detect_and_track(frames, conf=self.conf)

        observations: list[list[Any]] = []
        for dets in frame_dets:
            for det in dets:
                observations.append([
                    det.track_id,
                    det.frame_index,
                    det.x1, det.y1, det.x2, det.y2,
                    det.confidence,
                    det.class_id,
                ])
        return observations

    # ── Pose estimation (legacy dict format) ─────────────────────────────

    def pose_targets_for_box(
        self,
        photo: np.ndarray,
        box_xyxy: tuple[float, float, float, float],
        *,
        keypoint_conf: float | None = None,
        min_iou: float = 0.1,
    ) -> dict[str, Any] | None:
        """Estimate pose and return the best match for *box_xyxy*.

        Returns a dict with keys: method, target, iou, pose_box, keypoints
        (and optionally conf).
        """
        if keypoint_conf is None:
            keypoint_conf = self.keypoint_conf

        result = self._engine.estimate_pose(
            photo,
            box_xyxy,
            pose_conf=self.pose_conf,
            keypoint_conf=keypoint_conf,
            min_iou=min_iou,
        )
        if result is None:
            return None

        out: dict[str, Any] = {
            "method": result.method,
            "target": result.target,
            "iou": result.iou,
            "pose_box": result.pose_box,
            "keypoints": result.keypoints,
        }
        if result.confidence is not None:
            out["conf"] = result.confidence
        return out
