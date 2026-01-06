from __future__ import annotations

from typing import Any


_SEGMENTATION_ALIASES: dict[str, str] = {
    "yolo8": "YOLO8",
    "yolov8": "YOLO8",
    "yolov8seg": "YOLO8",
    "yolov8-seg": "YOLO8",
}


def normalize_segmentation_model(name: Any) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    key = raw.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    return _SEGMENTATION_ALIASES.get(key, raw.strip())


def list_segmentation_models() -> list[str]:
    return ["YOLO8"]


def build_segmentation_model(name: Any, *, pose_model: bool = False) -> Any:
    model = normalize_segmentation_model(name) or "YOLO8"
    if model == "YOLO8":
        from maxim.models.vision.segmentation import YOLO8

        return YOLO8(pose_model=pose_model)
    raise ValueError(f"Unknown segmentation model: {model}")

