from __future__ import annotations

from typing import Any

import atexit
import multiprocessing as mp
import os
import sys
import queue as queue_module
import threading

import cv2
import numpy as np

_IMSHOW_FAILED = False
_IMSHOW_DISABLED_WARNED = False
_DISPLAY_QUEUE = None
_DISPLAY_PROCESS = None
_DISPLAY_LOCK = threading.Lock()
_IMSHOW_MODE = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    return bool(default)


def _imshow_mode() -> str:
    global _IMSHOW_MODE
    if _IMSHOW_MODE is not None:
        return _IMSHOW_MODE
    default = "direct"
    if os.name != "nt" and (sys.platform.startswith("linux") or _is_wsl()):
        default = "process"
    raw = str(os.getenv("MAXIM_IMSHOW_MODE", default) or default).strip().lower()
    if raw not in ("direct", "process"):
        raw = "direct"
    _IMSHOW_MODE = raw
    return raw


def _display_disabled() -> bool:
    if _env_flag("MAXIM_DISABLE_IMSHOW", False) or _env_flag("MAXIM_HEADLESS", False):
        return True
    if os.name != "nt":
        if not os.getenv("DISPLAY") and not os.getenv("WAYLAND_DISPLAY"):
            return True
    return False


def _is_wsl() -> bool:
    if os.getenv("WSL_DISTRO_NAME") or os.getenv("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8") as handle:
            return "microsoft" in handle.read().lower()
    except Exception:
        return False


def _warn_display_disabled(reason: str) -> None:
    global _IMSHOW_DISABLED_WARNED
    if _IMSHOW_DISABLED_WARNED:
        return
    print(f"[WARN] OpenCV display disabled ({reason}). Set MAXIM_DISABLE_IMSHOW=0 to re-enable.")
    _IMSHOW_DISABLED_WARNED = True


def _display_process_main(frame_queue) -> None:
    window_names: set[str] = set()
    while True:
        try:
            item = frame_queue.get(timeout=0.1)
        except queue_module.Empty:
            try:
                cv2.waitKey(1)
            except Exception:
                pass
            continue

        if item is None:
            break

        window_name, frame, wait_ms = item
        try:
            if window_name not in window_names:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                window_names.add(window_name)
            cv2.imshow(window_name, frame)
            cv2.waitKey(int(wait_ms) if wait_ms is not None else 1)
        except Exception:
            continue

    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:
        pass


def _shutdown_display_process() -> None:
    global _DISPLAY_QUEUE, _DISPLAY_PROCESS
    try:
        if _DISPLAY_QUEUE is not None:
            _DISPLAY_QUEUE.put_nowait(None)
    except Exception:
        pass
    try:
        if _DISPLAY_PROCESS is not None:
            _DISPLAY_PROCESS.join(timeout=1.0)
    except Exception:
        pass
    _DISPLAY_QUEUE = None
    _DISPLAY_PROCESS = None


def _ensure_display_process() -> bool:
    global _DISPLAY_QUEUE, _DISPLAY_PROCESS
    if threading.current_thread() is not threading.main_thread():
        return _DISPLAY_PROCESS is not None and _DISPLAY_PROCESS.is_alive()
    with _DISPLAY_LOCK:
        if _DISPLAY_PROCESS is not None and _DISPLAY_PROCESS.is_alive():
            return True
        try:
            ctx = mp.get_context("spawn")
            _DISPLAY_QUEUE = ctx.Queue(maxsize=2)
            _DISPLAY_PROCESS = ctx.Process(
                target=_display_process_main,
                args=(_DISPLAY_QUEUE,),
                daemon=True,
            )
            _DISPLAY_PROCESS.start()
            atexit.register(_shutdown_display_process)
            return True
        except Exception:
            _DISPLAY_QUEUE = None
            _DISPLAY_PROCESS = None
            return False


def _enqueue_display(window_name: str, frame: np.ndarray, wait_ms: int) -> None:
    if _DISPLAY_QUEUE is None:
        return
    try:
        _DISPLAY_QUEUE.put_nowait((window_name, frame, wait_ms))
    except queue_module.Full:
        try:
            _DISPLAY_QUEUE.get_nowait()
            _DISPLAY_QUEUE.put_nowait((window_name, frame, wait_ms))
        except Exception:
            pass


def prepare_display() -> None:
    if _display_disabled():
        return
    if _imshow_mode() != "process":
        return
    _ensure_display_process()


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def close_windows() -> None:
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:
        pass


def show_photo(photo: np.ndarray, window_name: str = "Camera", wait: bool = True) -> None:
    try:
        if _display_disabled():
            _warn_display_disabled("headless or MAXIM_DISABLE_IMSHOW=1")
            return
        mode = _imshow_mode()
        if mode == "process":
            if not _ensure_display_process():
                _warn_display_disabled("display process unavailable")
                return
            frame = ensure_bgr(photo)
            _enqueue_display(window_name, frame, 1)
            return
        if threading.current_thread() is not threading.main_thread():
            _warn_display_disabled("non-main thread")
            return
        frame = ensure_bgr(photo)
        cv2.imshow(window_name, frame)
        if wait:
            cv2.waitKey(0)
            close_windows()
        else:
            cv2.waitKey(1)
    except Exception as e:
        global _IMSHOW_FAILED
        if not _IMSHOW_FAILED:
            print(f"[WARN] OpenCV display failed (imshow). Install a GUI-enabled OpenCV build. ({e})")
            _IMSHOW_FAILED = True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number))


def _as_numpy(value: Any) -> np.ndarray:
    try:
        return value.detach().cpu().numpy()
    except Exception:
        return np.asarray(value)


def iter_bounding_boxes(boxes: Any) -> list[dict[str, Any]]:
    """
    Normalize many common bounding-box formats into dictionaries.

    Supported inputs (best-effort):
    - Maxim observations: [track_id, frame_ind, x1, y1, x2, y2, conf]
    - xyxy(+extras): [x1, y1, x2, y2] or [x1, y1, x2, y2, conf]
    - dict with x1/y1/x2/y2 and optional track_id/id/conf/label
    - Ultralytics Results/Boxes (duck-typed via .boxes/.xyxy/.conf/.id)
    """
    if boxes is None:
        return []

    # List of Ultralytics Results -> flatten
    if isinstance(boxes, (list, tuple)) and boxes and hasattr(boxes[0], "boxes"):
        out: list[dict[str, Any]] = []
        for item in boxes:
            out.extend(iter_bounding_boxes(item))
        return out

    # Ultralytics Results -> Boxes
    names = None
    if hasattr(boxes, "boxes"):
        try:
            names = getattr(boxes, "names", None)
            boxes = boxes.boxes
        except Exception:
            pass

    # Ultralytics Boxes
    if hasattr(boxes, "xyxy"):
        try:
            xyxy = _as_numpy(getattr(boxes, "xyxy"))
            conf = getattr(boxes, "conf", None)
            ids = getattr(boxes, "id", None)
            clss = getattr(boxes, "cls", None)

            conf_arr = _as_numpy(conf).reshape(-1) if conf is not None else None
            id_arr = _as_numpy(ids).reshape(-1) if ids is not None else None
            cls_arr = _as_numpy(clss).reshape(-1) if clss is not None else None

            out: list[dict[str, Any]] = []
            for i in range(int(xyxy.shape[0])):
                x1, y1, x2, y2 = map(float, xyxy[i][:4])
                entry: dict[str, Any] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                if conf_arr is not None and i < conf_arr.shape[0]:
                    entry["conf"] = float(conf_arr[i])
                if id_arr is not None and i < id_arr.shape[0]:
                    entry["track_id"] = int(id_arr[i])
                if cls_arr is not None and i < cls_arr.shape[0]:
                    cls_id = int(cls_arr[i])
                    entry["cls"] = cls_id
                    if names is not None:
                        try:
                            entry["label"] = names.get(cls_id) if isinstance(names, dict) else names[cls_id]
                        except Exception:
                            pass
                out.append(entry)
            return out
        except Exception:
            pass

    if isinstance(boxes, dict):
        boxes = [boxes]

    if isinstance(boxes, np.ndarray):
        if boxes.ndim == 2 and boxes.shape[1] >= 4:
            boxes = boxes.tolist()
        else:
            return []

    if not isinstance(boxes, (list, tuple)):
        return []

    out: list[dict[str, Any]] = []
    for box in boxes:
        if box is None:
            continue

        if isinstance(box, dict):
            if all(k in box for k in ("x1", "y1", "x2", "y2")):
                entry = dict(box)
                out.append(entry)
            continue

        if isinstance(box, (list, tuple, np.ndarray)):
            seq = list(box)
            if len(seq) >= 7 and (_is_number(seq[2]) and _is_number(seq[5])):
                track_id, frame_ind, x1, y1, x2, y2, conf = seq[:7]
                entry = {
                    "track_id": int(track_id) if track_id is not None else None,
                    "frame_ind": int(frame_ind) if frame_ind is not None else None,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "conf": float(conf),
                }
                out.append(entry)
            elif len(seq) >= 4 and all(_is_number(v) for v in seq[:4]):
                x1, y1, x2, y2 = map(float, seq[:4])
                entry = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                if len(seq) >= 5 and _is_number(seq[4]):
                    entry["conf"] = float(seq[4])
                out.append(entry)

    return out


def annotate_frame(
    frame: np.ndarray,
    *,
    boxes: Any = None,
    target_box: tuple[float, float, float, float] | None = None,
    center: tuple[float, float] | None = None,
    target_point: tuple[float, float] | None = None,
    text_lines: list[str] | None = None,
) -> np.ndarray:
    frame = ensure_bgr(frame)
    height, width = frame.shape[:2]

    norm_boxes = iter_bounding_boxes(boxes)
    needs_draw = (
        bool(norm_boxes)
        or target_box is not None
        or center is not None
        or target_point is not None
        or bool(text_lines)
    )
    if needs_draw:
        frame = frame.copy()

    target_int = None
    if target_box is not None:
        tx1, ty1, tx2, ty2 = target_box
        target_int = (
            int(np.clip(round(tx1), 0, width - 1)),
            int(np.clip(round(ty1), 0, height - 1)),
            int(np.clip(round(tx2), 0, width - 1)),
            int(np.clip(round(ty2), 0, height - 1)),
        )

    for b in norm_boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        ix1 = int(np.clip(round(x1), 0, width - 1))
        iy1 = int(np.clip(round(y1), 0, height - 1))
        ix2 = int(np.clip(round(x2), 0, width - 1))
        iy2 = int(np.clip(round(y2), 0, height - 1))

        is_target = target_int == (ix1, iy1, ix2, iy2) if target_int is not None else False
        color = (0, 0, 255) if is_target else (0, 255, 0)
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)

        track_id = b.get("track_id")
        conf = b.get("conf")
        label = b.get("label")

        parts: list[str] = []
        if label is not None:
            parts.append(str(label))
        if track_id is not None:
            parts.append(f"id:{track_id}")
        if conf is not None:
            parts.append(f"{float(conf):.2f}")

        if parts:
            cv2.putText(
                frame,
                " ".join(parts),
                (ix1, max(0, iy1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    if center is not None:
        cx, cy = center
        cv2.drawMarker(
            frame,
            (int(round(cx)), int(round(cy))),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            18,
            1,
        )

    if target_point is not None:
        tx, ty = target_point
        cv2.drawMarker(
            frame,
            (int(round(tx)), int(round(ty))),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            18,
            2,
        )

    if text_lines:
        x, y = 10, 20
        for line in text_lines:
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 18

    return frame


def show_frame(
    frame: np.ndarray,
    *,
    boxes: Any = None,
    target_box: tuple[float, float, float, float] | None = None,
    center: tuple[float, float] | None = None,
    target_point: tuple[float, float] | None = None,
    text_lines: list[str] | None = None,
    window_name: str = "Camera",
    wait_ms: int = 1,
) -> None:
    try:
        if _display_disabled():
            _warn_display_disabled("headless or MAXIM_DISABLE_IMSHOW=1")
            return
        mode = _imshow_mode()
        annotated = annotate_frame(
            frame,
            boxes=boxes,
            target_box=target_box,
            center=center,
            target_point=target_point,
            text_lines=text_lines,
        )
        if mode == "process":
            if not _ensure_display_process():
                _warn_display_disabled("display process unavailable")
                return
            _enqueue_display(window_name, annotated, wait_ms)
            return
        if threading.current_thread() is not threading.main_thread():
            _warn_display_disabled("non-main thread")
            return
        cv2.imshow(window_name, annotated)
        cv2.waitKey(wait_ms)
    except Exception as e:
        global _IMSHOW_FAILED
        if not _IMSHOW_FAILED:
            print(f"[WARN] OpenCV display failed (imshow). Install a GUI-enabled OpenCV build. ({e})")
            _IMSHOW_FAILED = True
