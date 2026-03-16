"""Tests for the vision engine abstraction, IoU tracker, and RTM post-processing."""

from __future__ import annotations


import numpy as np
import pytest

from maxim.models.vision.engine import COCO_NAMES, Detection, PoseResult, VisionEngine
from maxim.models.vision.tracker import IoUTracker, _iou_matrix


# ─────────────────────────────────────────────────────────────────────────────
# IoU matrix
# ─────────────────────────────────────────────────────────────────────────────


class TestIoUMatrix:
    def test_identical_boxes(self):
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        iou = _iou_matrix(boxes, boxes)
        assert iou.shape == (1, 1)
        assert pytest.approx(iou[0, 0], abs=1e-5) == 1.0

    def test_non_overlapping_boxes(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[20, 20, 30, 30]], dtype=np.float32)
        iou = _iou_matrix(a, b)
        assert iou[0, 0] == 0.0

    def test_partial_overlap(self):
        a = np.array([[0, 0, 20, 20]], dtype=np.float32)
        b = np.array([[10, 10, 30, 30]], dtype=np.float32)
        iou = _iou_matrix(a, b)
        # Intersection: 10×10=100, Union: 400+400-100=700
        assert pytest.approx(iou[0, 0], abs=1e-3) == 100 / 700

    def test_multiple_boxes(self):
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        b = np.array([[5, 5, 15, 15]], dtype=np.float32)
        iou = _iou_matrix(a, b)
        assert iou.shape == (2, 1)
        assert iou[0, 0] > 0  # overlaps with first
        assert iou[1, 0] == 0  # no overlap with second


# ─────────────────────────────────────────────────────────────────────────────
# IoU Tracker
# ─────────────────────────────────────────────────────────────────────────────


class TestIoUTracker:
    def test_new_detections_get_unique_ids(self):
        tracker = IoUTracker()
        boxes = np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        cls_ids = np.array([0, 1])

        ids = tracker.update(boxes, scores, cls_ids)
        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_matching_detection_preserves_id(self):
        tracker = IoUTracker(iou_threshold=0.3)
        boxes1 = np.array([[10, 10, 50, 50]], dtype=np.float32)
        boxes2 = np.array([[12, 12, 52, 52]], dtype=np.float32)  # small shift
        scores = np.array([0.9])
        cls_ids = np.array([0])

        ids1 = tracker.update(boxes1, scores, cls_ids)
        ids2 = tracker.update(boxes2, scores, cls_ids)
        assert ids1[0] == ids2[0]

    def test_distant_detection_gets_new_id(self):
        tracker = IoUTracker(iou_threshold=0.3)
        boxes1 = np.array([[10, 10, 50, 50]], dtype=np.float32)
        boxes2 = np.array([[200, 200, 300, 300]], dtype=np.float32)  # far away
        scores = np.array([0.9])
        cls_ids = np.array([0])

        ids1 = tracker.update(boxes1, scores, cls_ids)
        ids2 = tracker.update(boxes2, scores, cls_ids)
        assert ids1[0] != ids2[0]

    def test_unmatched_track_removed_after_max_age(self):
        tracker = IoUTracker(max_age=2)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9])
        cls_ids = np.array([0])

        ids = tracker.update(boxes, scores, cls_ids)
        original_id = ids[0]

        # Send empty frames to age it out
        empty = np.empty((0, 4), dtype=np.float32)
        empty_s = np.empty(0)
        empty_c = np.empty(0, dtype=int)
        tracker.update(empty, empty_s, empty_c)
        tracker.update(empty, empty_s, empty_c)
        tracker.update(empty, empty_s, empty_c)  # exceeds max_age=2

        # Re-detect at same position — should get a new ID
        ids2 = tracker.update(boxes, scores, cls_ids)
        assert ids2[0] != original_id

    def test_empty_frame_returns_empty(self):
        tracker = IoUTracker()
        ids = tracker.update(
            np.empty((0, 4), dtype=np.float32),
            np.empty(0),
            np.empty(0, dtype=int),
        )
        assert len(ids) == 0

    def test_reset_clears_state(self):
        tracker = IoUTracker()
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        tracker.update(boxes, np.array([0.9]), np.array([0]))

        tracker.reset()

        ids2 = tracker.update(boxes, np.array([0.9]), np.array([0]))
        # After reset, IDs restart from 1
        assert ids2[0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# RTM post-processing (no model files needed)
# ─────────────────────────────────────────────────────────────────────────────


class TestDetPreprocess:
    def test_output_shape(self):
        from maxim.models.vision.rtm_engine import _det_preprocess

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        blob, ratio, (pw, ph) = _det_preprocess(img, (640, 640))
        assert blob.shape == (1, 3, 640, 640)
        assert blob.dtype == np.float32
        assert ratio > 0

    def test_square_image_no_padding(self):
        from maxim.models.vision.rtm_engine import _det_preprocess

        img = np.zeros((640, 640, 3), dtype=np.uint8)
        _, ratio, (pw, ph) = _det_preprocess(img, (640, 640))
        assert pytest.approx(ratio, abs=1e-3) == 1.0


class TestNMS:
    def test_empty_input(self):
        from maxim.models.vision.rtm_engine import _nms

        keep = _nms(
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            0.5,
        )
        assert len(keep) == 0

    def test_single_detection_passes(self):
        from maxim.models.vision.rtm_engine import _nms

        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9])
        keep = _nms(boxes, scores, 0.5)
        assert list(keep) == [0]

    def test_overlapping_boxes_suppressed(self):
        from maxim.models.vision.rtm_engine import _nms

        boxes = np.array([
            [10, 10, 50, 50],
            [12, 12, 52, 52],  # nearly identical
        ], dtype=np.float32)
        scores = np.array([0.9, 0.7])
        keep = _nms(boxes, scores, 0.5)
        assert len(keep) == 1
        assert keep[0] == 0  # higher score kept

    def test_non_overlapping_preserved(self):
        from maxim.models.vision.rtm_engine import _nms

        boxes = np.array([
            [10, 10, 50, 50],
            [200, 200, 250, 250],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        keep = _nms(boxes, scores, 0.5)
        assert len(keep) == 2


class TestSimCCDecode:
    def test_basic_decode(self):
        from maxim.models.vision.rtm_engine import _simcc_decode

        # 1 person, 17 keypoints, 384 bins (192 * split_ratio=2)
        N, K, W = 1, 17, 384
        simcc_x = np.random.randn(N, K, W).astype(np.float32)
        simcc_y = np.random.randn(N, K, W).astype(np.float32)

        locs, scores = _simcc_decode(simcc_x, simcc_y, simcc_split_ratio=2.0)
        assert locs.shape == (1, 17, 2)
        assert scores.shape == (1, 17)

    def test_argmax_location(self):
        from maxim.models.vision.rtm_engine import _simcc_decode

        # Single keypoint with a clear peak
        simcc_x = np.zeros((1, 1, 100), dtype=np.float32)
        simcc_y = np.zeros((1, 1, 100), dtype=np.float32)
        simcc_x[0, 0, 50] = 10.0  # peak at bin 50
        simcc_y[0, 0, 30] = 10.0  # peak at bin 30

        locs, scores = _simcc_decode(simcc_x, simcc_y, simcc_split_ratio=2.0)
        assert pytest.approx(locs[0, 0, 0], abs=0.1) == 25.0  # 50 / 2.0
        assert pytest.approx(locs[0, 0, 1], abs=0.1) == 15.0  # 30 / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation model adapter (YOLO8 class)
# ─────────────────────────────────────────────────────────────────────────────


class _MockEngine(VisionEngine):
    """Fake engine for testing the YOLO8 adapter without models."""

    def detect_and_track(self, frames, conf=0.5):
        results = []
        for i, _ in enumerate(frames):
            results.append([
                Detection(track_id=1, frame_index=i,
                          x1=10, y1=20, x2=50, y2=60,
                          confidence=0.9, class_id=0),
            ])
        return results

    def estimate_pose(self, frame, bbox_xyxy, *, pose_conf=0.25,
                      keypoint_conf=0.25, min_iou=0.1):
        return PoseResult(
            method="eyes",
            target=(30.0, 25.0),
            iou=0.85,
            pose_box=(10, 20, 50, 60),
            keypoints={"left_eye": (25.0, 25.0, 0.9),
                        "right_eye": (35.0, 25.0, 0.9),
                        "nose": (30.0, 30.0, 0.8)},
            confidence=0.95,
        )

    def warmup(self, *, include_pose=True):
        pass

    @property
    def class_names(self):
        return COCO_NAMES


class TestYOLO8Adapter:
    def test_segment_photos_output_format(self):
        from maxim.models.vision.segmentation import YOLO8

        model = YOLO8(engine=_MockEngine())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        obs = model.segment_photos([frame])

        assert len(obs) == 1
        row = obs[0]
        assert len(row) == 8
        track_id, frame_ind, x1, y1, x2, y2, conf, cls_id = row
        assert track_id == 1
        assert frame_ind == 0
        assert conf == 0.9
        assert cls_id == 0

    def test_segment_photos_none_returns_empty(self):
        from maxim.models.vision.segmentation import YOLO8

        model = YOLO8(engine=_MockEngine())
        assert model.segment_photos(None) == []

    def test_pose_targets_returns_dict(self):
        from maxim.models.vision.segmentation import YOLO8

        model = YOLO8(engine=_MockEngine())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = model.pose_targets_for_box(frame, (10, 20, 50, 60))

        assert result is not None
        assert result["method"] == "eyes"
        assert "target" in result
        assert "keypoints" in result
        assert "conf" in result

    def test_model_names_attribute(self):
        from maxim.models.vision.segmentation import YOLO8

        model = YOLO8(engine=_MockEngine())
        assert hasattr(model, "model")
        assert hasattr(model.model, "names")
        assert model.model.names[0] == "person"


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_normalize_default(self):
        from maxim.models.vision.registry import normalize_engine_name

        assert normalize_engine_name(None) == "rtm"
        assert normalize_engine_name("") == "rtm"

    def test_normalize_aliases(self):
        from maxim.models.vision.registry import normalize_engine_name

        assert normalize_engine_name("yolo8") == "yolo"
        assert normalize_engine_name("yolov8") == "yolo"
        assert normalize_engine_name("ultralytics") == "yolo"
        assert normalize_engine_name("rtmdet") == "rtm"
        assert normalize_engine_name("mmlab") == "rtm"
        assert normalize_engine_name("onnx") == "rtm"

    def test_normalize_canonical(self):
        from maxim.models.vision.registry import normalize_engine_name

        assert normalize_engine_name("rtm") == "rtm"
        assert normalize_engine_name("yolo") == "yolo"

    def test_list_engines(self):
        from maxim.models.vision.registry import list_engines

        engines = list_engines()
        assert "rtm" in engines
        assert "yolo" in engines
