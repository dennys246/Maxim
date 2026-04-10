"""Vision event streaming mixin for Maxim.

Complete subsystem with its own thread lifecycle for continuous
vision event logging via segmentation model inference.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np

from maxim.utils.data_management import VisionEventLogger
from maxim.utils.logging import warn


class VisionStreamMixin:
    """Mixin providing vision event streaming for the Maxim class."""

    # Type hints for attributes accessed from the main Maxim class
    log: Any
    home_dir: str
    _vision_event_logger: VisionEventLogger | None
    _vision_event_thread: threading.Thread | None
    _vision_event_stop_event: threading.Event | None
    _vision_event_last_frame_ts: float | None
    vision_events_path: str

    def _ensure_vision_logger(self) -> VisionEventLogger | None:
        if self._vision_event_logger is not None:
            return self._vision_event_logger

        run_id = getattr(self, "run_id", None) or time.strftime("%Y-%m-%d_%H%M%S")
        if not self.vision_events_path:
            self.vision_events_path = os.path.join(
                str(getattr(self, "home_dir", "data") or "data"),
                "vision",
                f"vision_events_{run_id}.jsonl",
            )

        try:
            logger = VisionEventLogger(self.vision_events_path)
            logger.start()
            self._vision_event_logger = logger
            return logger
        except Exception as e:
            warn("Failed to start vision event logger: %s", e, logger=self.log)
            self._vision_event_logger = None
            return None

    def _start_vision_event_stream(self) -> None:
        existing = getattr(self, "_vision_event_thread", None)
        if existing is not None and getattr(existing, "is_alive", lambda: False)():
            return

        logger = self._ensure_vision_logger()
        if logger is None:
            return

        stop_event = threading.Event()
        self._vision_event_stop_event = stop_event
        self._vision_event_last_frame_ts = None

        def _worker() -> None:
            try:
                hz = float(os.getenv("MAXIM_VISION_EVENT_HZ", "2.0") or 2.0)
            except Exception:
                hz = 2.0
            if hz <= 0:
                hz = 2.0
            sleep_s = 1.0 / hz

            while not stop_event.is_set():
                frame = getattr(self, "_last_frame", None)
                if frame is None or not isinstance(frame, np.ndarray):
                    time.sleep(sleep_s)
                    continue

                frame_ts = getattr(self, "_last_frame_ts", None)
                ts_val = None
                try:
                    ts_val = float(frame_ts) if frame_ts is not None else None
                except Exception:
                    ts_val = None

                if ts_val is not None and self._vision_event_last_frame_ts is not None:
                    if ts_val <= float(self._vision_event_last_frame_ts):
                        time.sleep(sleep_s)
                        continue

                segmenter = getattr(self, "segmenter", None)
                if segmenter is None:
                    try:
                        self._ensure_segmenter()
                    except Exception:
                        segmenter = None
                    segmenter = getattr(self, "segmenter", None)
                if segmenter is None:
                    time.sleep(sleep_s)
                    continue

                lock = getattr(self, "_observation_lock", None)
                acquired = False
                if lock is not None:
                    try:
                        acquired = lock.acquire(blocking=False)
                    except Exception:
                        acquired = False
                if lock is not None and not acquired:
                    time.sleep(sleep_s)
                    continue

                try:
                    observations = segmenter.segment_photos(frame)
                except Exception as e:
                    warn("Vision event segmentation failed: %s", e, logger=self.log)
                    observations = []
                finally:
                    if lock is not None and acquired:
                        try:
                            lock.release()
                        except Exception:
                            pass

                dets: list[dict[str, Any]] = []
                names = getattr(getattr(segmenter, "model", None), "names", None)
                for obs in observations or []:
                    if not isinstance(obs, (list, tuple)) or len(obs) < 8:
                        continue
                    try:
                        track_id = int(obs[0]) if obs[0] is not None else None
                    except Exception:
                        track_id = None
                    try:
                        cls_id = int(obs[7]) if obs[7] is not None else None
                    except Exception:
                        cls_id = None
                    try:
                        conf = float(obs[6]) if obs[6] is not None else 0.0
                    except Exception:
                        conf = 0.0

                    label = None
                    try:
                        if isinstance(names, dict) and cls_id in names:
                            label = names.get(cls_id)
                        elif isinstance(names, (list, tuple)) and cls_id is not None:
                            if 0 <= cls_id < len(names):
                                label = names[cls_id]
                    except Exception:
                        label = None

                    dets.append(
                        {
                            "track_id": track_id,
                            "class_id": cls_id,
                            "label": label,
                            "conf": conf,
                            "bbox_xyxy": [float(obs[2]), float(obs[3]), float(obs[4]), float(obs[5])],
                        }
                    )

                record = {
                    "kind": "vision_event",
                    "time": float(time.time()),
                    "run_id": getattr(self, "run_id", None),
                    "frame_ts": ts_val,
                    "model": getattr(self, "_segmenter_model", None),
                    "interests": list(getattr(self, "interests", []) or []),
                    "detections": dets,
                }
                try:
                    shape = getattr(frame, "shape", None)
                    if isinstance(shape, tuple) and len(shape) >= 2:
                        record["frame_shape"] = [int(shape[0]), int(shape[1])]
                except Exception:
                    pass

                logger.log_event(record)
                if ts_val is not None:
                    self._vision_event_last_frame_ts = float(ts_val)

                time.sleep(sleep_s)

        t = threading.Thread(target=_worker, name="maxim.vision.events", daemon=True)
        self._vision_event_thread = t
        self.register_thread("maxim.vision.events", t)
        t.start()

    def _stop_vision_event_stream(self, *, timeout: float = 2.0) -> None:
        try:
            ev = getattr(self, "_vision_event_stop_event", None)
            if ev is not None:
                ev.set()
        except Exception:
            pass
        t = getattr(self, "_vision_event_thread", None)
        if t is not None:
            try:
                t.join(timeout=float(timeout))
            except Exception:
                pass
        self._vision_event_thread = None
        self._vision_event_stop_event = None

        logger = getattr(self, "_vision_event_logger", None)
        if logger is not None:
            try:
                logger.stop(timeout=float(timeout))
            except Exception:
                pass
        self._vision_event_logger = None
