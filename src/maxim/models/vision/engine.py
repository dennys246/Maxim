"""Vision engine abstraction for pluggable object detection and pose estimation.

Defines the VisionEngine interface and shared data types. Concrete engines
(RTMEngine, UltralyticsEngine) implement this interface so backends can be
swapped without changing downstream code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Detection:
    """A single object detection with optional tracking ID."""

    track_id: int | None
    frame_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


@dataclass(slots=True)
class PoseResult:
    """Pose estimation result for a single person detection."""

    method: str  # "eyes" | "nose"
    target: tuple[float, float]
    iou: float
    pose_box: tuple[float, float, float, float]
    keypoints: dict[str, tuple[float, float, float | None]]
    confidence: float | None = None


# Complete COCO-80 class names shared across all engines.
COCO_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

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


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Engine
# ─────────────────────────────────────────────────────────────────────────────


class VisionEngine(ABC):
    """Abstract interface for vision detection and pose estimation backends.

    Concrete implementations provide the actual inference logic.  The
    ``SegmentationModel`` wrapper (segmentation.py) delegates to an engine
    and converts outputs to the legacy list format consumed by the rest of
    the codebase.
    """

    @abstractmethod
    def detect_and_track(
        self,
        frames: list[np.ndarray],
        conf: float = 0.5,
    ) -> list[list[Detection]]:
        """Run detection + tracking on one or more BGR frames.

        Args:
            frames: List of BGR uint8 images.
            conf: Confidence threshold.

        Returns:
            Outer list has one entry per frame.  Inner list contains
            ``Detection`` instances for that frame.
        """

    @abstractmethod
    def estimate_pose(
        self,
        frame: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        *,
        pose_conf: float = 0.25,
        keypoint_conf: float = 0.25,
        min_iou: float = 0.1,
    ) -> PoseResult | None:
        """Estimate human pose and return the best match for *bbox_xyxy*.

        Args:
            frame: BGR uint8 image.
            bbox_xyxy: Target bounding box (x1, y1, x2, y2).
            pose_conf: Minimum detection confidence for pose model.
            keypoint_conf: Minimum per-keypoint confidence.
            min_iou: Minimum IoU between target box and a pose detection.

        Returns:
            ``PoseResult`` for the best-matching person, or ``None``.
        """

    @abstractmethod
    def warmup(self, *, include_pose: bool = True) -> None:
        """Run a dummy inference pass to reduce first-use latency."""

    @property
    @abstractmethod
    def class_names(self) -> dict[int, str]:
        """Mapping of class ID → human-readable name (e.g. COCO-80)."""
