"""Vision engine registry — factory for building segmentation models.

Maps human-friendly engine names (e.g. ``"rtm"``, ``"yolo"``) to
:class:`VisionEngine` implementations and constructs a ``YOLO8`` adapter
around the selected engine.
"""

from __future__ import annotations

from typing import Any

from maxim.models.vision.engine import VisionEngine


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

# Canonical engine name → lazy builder (import deferred to avoid heavy deps
# at import time).
_ENGINE_BUILDERS: dict[str, str] = {
    "rtm": "maxim.models.vision.rtm_engine.RTMEngine",
    "yolo": "maxim.models.vision.ultralytics_engine.UltralyticsEngine",
}

# Aliases → canonical name
_ALIASES: dict[str, str] = {
    # RTM / default
    "rtmdet": "rtm",
    "rtmpose": "rtm",
    "mmlab": "rtm",
    "onnx": "rtm",
    "default": "rtm",
    # YOLO / ultralytics
    "yolo8": "yolo",
    "yolov8": "yolo",
    "yolov8seg": "yolo",
    "yolov8-seg": "yolo",
    "ultralytics": "yolo",
}

DEFAULT_ENGINE = "rtm"


def normalize_engine_name(name: Any) -> str:
    """Resolve a user-supplied name to a canonical engine key."""
    raw = str(name or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if not raw:
        return DEFAULT_ENGINE
    if raw in _ENGINE_BUILDERS:
        return raw
    return _ALIASES.get(raw, raw)


def list_engines() -> list[str]:
    """Return canonical engine names."""
    return list(_ENGINE_BUILDERS.keys())


def _build_engine(name: str, *, pose_model: bool = False) -> VisionEngine:
    """Instantiate an engine by canonical name."""
    canonical = normalize_engine_name(name)
    fqn = _ENGINE_BUILDERS.get(canonical)

    if fqn is None:
        available = ", ".join(sorted(set(list(_ENGINE_BUILDERS.keys()) + list(_ALIASES.keys()))))
        raise ValueError(f"Unknown vision engine: {name!r}. Available engines: {available}")

    module_path, class_name = fqn.rsplit(".", 1)

    try:
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except ImportError as exc:
        if canonical == "yolo":
            raise ImportError(
                "The YOLO engine requires the 'ultralytics' package. Install it with: pip install \"maxim[yolo]\""
            ) from exc
        raise

    return cls(load_pose=pose_model)


# ─────────────────────────────────────────────────────────────────────────────
# Public factory (used by cli.py, selfy.py, etc.)
# ─────────────────────────────────────────────────────────────────────────────


def build_segmentation_model(name: Any = None, *, pose_model: bool = False) -> Any:
    """Build a ``YOLO8`` adapter wrapping the requested engine.

    Args:
        name: Engine name (``"rtm"`` or ``"yolo"``).  ``None`` → default.
        pose_model: Whether to eagerly load the pose model.

    Returns:
        A ``YOLO8`` instance.
    """
    engine = _build_engine(name or DEFAULT_ENGINE, pose_model=pose_model)

    from maxim.models.vision.segmentation import YOLO8

    return YOLO8(pose_model=pose_model, engine=engine)
