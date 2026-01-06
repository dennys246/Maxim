from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

from maxim.utils.logging import warn

def build_home(home_dir):
    for name in (
        "images",
        "videos",
        "audio",
        os.path.join("audio", "chunks"),
        "transcript",
        "logs",
        "models",
        "training",
    ):
        os.makedirs(os.path.join(home_dir, name), exist_ok=True)


class TrainingSampleLogger:
    """
    Background JSONL writer for training samples.

    Uses an internal queue to keep the control loop responsive.
    """

    def __init__(
        self,
        training_dir: str | os.PathLike[str],
        *,
        motor_filename: str = "motor_training_set.jsonl",
        events_filename: str = "action_events.jsonl",
        max_queue: int = 2048,
    ) -> None:
        self.training_dir = Path(training_dir)
        self.motor_path = self.training_dir / motor_filename
        self.events_path = self.training_dir / events_filename
        self._q: queue.Queue[tuple[str, dict[str, Any], bool]] = queue.Queue(maxsize=int(max_queue))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="maxim.training.logger", daemon=True)

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(("__stop__", {}, True))
        except Exception:
            pass
        try:
            self._thread.join(timeout=float(timeout))
        except Exception:
            pass

    def log_motor_sample(self, record: dict[str, Any], *, flush: bool = False) -> None:
        self._enqueue("motor", record, flush=flush)

    def log_event(self, record: dict[str, Any], *, flush: bool = False) -> None:
        self._enqueue("event", record, flush=flush)

    def _enqueue(self, kind: str, record: dict[str, Any], *, flush: bool) -> None:
        try:
            self._q.put_nowait((kind, record, bool(flush)))
            return
        except queue.Full:
            pass

        # Avoid blocking the control loop: drop the oldest record and retry once.
        try:
            _ = self._q.get_nowait()
        except Exception:
            return
        try:
            self._q.put_nowait((kind, record, bool(flush)))
        except Exception:
            return

    def _worker(self) -> None:
        motor_fp = None
        event_fp = None
        last_flush = 0.0
        flush_every_s = 1.0

        try:
            self.training_dir.mkdir(parents=True, exist_ok=True)
            motor_fp = open(self.motor_path, "a", encoding="utf-8")
            event_fp = open(self.events_path, "a", encoding="utf-8")

            while True:
                if self._stop.is_set() and self._q.empty():
                    break
                try:
                    kind, record, force_flush = self._q.get(timeout=0.25)
                except queue.Empty:
                    continue

                try:
                    if kind == "__stop__":
                        break
                    if kind == "motor":
                        motor_fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    elif kind == "event":
                        event_fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    else:
                        continue

                    now = time.time()
                    if force_flush or (now - last_flush) >= flush_every_s:
                        try:
                            motor_fp.flush()
                            event_fp.flush()
                        except Exception:
                            pass
                        last_flush = now
                except Exception as e:
                    warn("Failed to write training sample: %s", e)
                finally:
                    try:
                        self._q.task_done()
                    except Exception:
                        pass
        finally:
            for fp in (motor_fp, event_fp):
                if fp is None:
                    continue
                try:
                    fp.flush()
                except Exception:
                    pass
                try:
                    fp.close()
                except Exception:
                    pass
