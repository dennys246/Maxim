"""IoU-based multi-object tracker.

Replaces the ByteTrack tracker bundled with ultralytics (which required
the ``lap`` package for the Jonker-Volgenant algorithm).  Uses
``scipy.optimize.linear_sum_assignment`` (Hungarian algorithm) instead —
MIT-licensed and already a project dependency.

For the Reachy Mini robotics use case (typically < 20 objects per frame at
~10 FPS), the Hungarian algorithm's O(n³) cost is negligible and IoU
matching between consecutive frames is sufficient without Kalman prediction.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of xyxy boxes.

    Args:
        boxes_a: (M, 4) array of [x1, y1, x2, y2].
        boxes_b: (N, 4) array of [x1, y1, x2, y2].

    Returns:
        (M, N) IoU matrix.
    """
    # Intersection coordinates
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    # Avoid division by zero
    return np.where(union > 0, inter / union, 0.0)


class IoUTracker:
    """Frame-to-frame IoU tracker with persistent track IDs.

    Each call to :meth:`update` matches new detections against existing
    tracks using IoU overlap, assigns stable integer IDs, and retires
    tracks that have not been matched for ``max_age`` consecutive frames.

    Args:
        iou_threshold: Minimum IoU to consider a valid match.
        max_age: Frames to keep an unmatched track before deletion.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
    ) -> None:
        self._next_id: int = 1
        self._tracks: dict[int, np.ndarray] = {}  # track_id → last box
        self._ages: dict[int, int] = {}  # track_id → frames since last match
        self._iou_threshold = iou_threshold
        self._max_age = max_age

    def reset(self) -> None:
        """Clear all tracked state."""
        self._next_id = 1
        self._tracks.clear()
        self._ages.clear()

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> np.ndarray:
        """Match detections to existing tracks and return assigned IDs.

        Args:
            boxes: (N, 4) xyxy bounding boxes.
            scores: (N,) confidence scores (unused for matching, reserved
                for future score-gated logic).
            class_ids: (N,) integer class IDs (unused for matching,
                reserved for class-aware tracking).

        Returns:
            (N,) int array of track IDs, one per input detection.
        """
        n_det = len(boxes)
        if n_det == 0:
            # No detections — age all tracks
            self._age_tracks()
            return np.array([], dtype=int)

        track_ids_out = np.zeros(n_det, dtype=int)

        if not self._tracks:
            # No existing tracks — assign new IDs to everything
            for i in range(n_det):
                tid = self._next_id
                self._next_id += 1
                track_ids_out[i] = tid
                self._tracks[tid] = boxes[i].copy()
                self._ages[tid] = 0
            return track_ids_out

        # Build cost matrix (negative IoU for minimisation)
        existing_ids = list(self._tracks.keys())
        existing_boxes = np.array([self._tracks[tid] for tid in existing_ids])
        iou = _iou_matrix(boxes, existing_boxes)  # (n_det, n_tracks)

        # Hungarian assignment
        det_indices, trk_indices = linear_sum_assignment(-iou)

        matched_det: set[int] = set()
        matched_trk: set[int] = set()

        for d_idx, t_idx in zip(det_indices, trk_indices):
            if iou[d_idx, t_idx] >= self._iou_threshold:
                tid = existing_ids[t_idx]
                track_ids_out[d_idx] = tid
                self._tracks[tid] = boxes[d_idx].copy()
                self._ages[tid] = 0
                matched_det.add(d_idx)
                matched_trk.add(t_idx)

        # Unmatched detections → new tracks
        for i in range(n_det):
            if i not in matched_det:
                tid = self._next_id
                self._next_id += 1
                track_ids_out[i] = tid
                self._tracks[tid] = boxes[i].copy()
                self._ages[tid] = 0

        # Age unmatched tracks; remove stale ones
        for t_idx, tid in enumerate(existing_ids):
            if t_idx not in matched_trk:
                self._ages[tid] = self._ages.get(tid, 0) + 1
                if self._ages[tid] > self._max_age:
                    del self._tracks[tid]
                    del self._ages[tid]

        return track_ids_out

    def _age_tracks(self) -> None:
        """Increment age of all tracks and prune expired ones."""
        expired = []
        for tid in list(self._ages):
            self._ages[tid] += 1
            if self._ages[tid] > self._max_age:
                expired.append(tid)
        for tid in expired:
            self._tracks.pop(tid, None)
            self._ages.pop(tid, None)
