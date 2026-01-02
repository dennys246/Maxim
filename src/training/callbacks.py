from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils.logging import warn


class Callback:
    def on_train_begin(self, **kwargs: Any) -> None:
        return

    def on_train_end(self, **kwargs: Any) -> None:
        return

    def on_train_step_begin(self, step: int, *, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        return

    def on_train_step_end(self, step: int, *, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        return


class CallbackList:
    def __init__(self, callbacks: Iterable[Callback] | None = None):
        self.callbacks: list[Callback] = list(callbacks or [])
        self._started = False

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_begin(self, **kwargs: Any) -> None:
        if self._started:
            return
        self._started = True
        for cb in self.callbacks:
            try:
                cb.on_train_begin(**kwargs)
            except Exception as e:
                warn("Callback %s.on_train_begin failed: %s", type(cb).__name__, e)

    def on_train_end(self, **kwargs: Any) -> None:
        for cb in self.callbacks:
            try:
                cb.on_train_end(**kwargs)
            except Exception as e:
                warn("Callback %s.on_train_end failed: %s", type(cb).__name__, e)

    def on_train_step_begin(self, step: int, *, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        for cb in self.callbacks:
            try:
                cb.on_train_step_begin(step, logs=logs, **kwargs)
            except Exception as e:
                warn("Callback %s.on_train_step_begin failed: %s", type(cb).__name__, e)

    def on_train_step_end(self, step: int, *, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        for cb in self.callbacks:
            try:
                cb.on_train_step_end(step, logs=logs, **kwargs)
            except Exception as e:
                warn("Callback %s.on_train_step_end failed: %s", type(cb).__name__, e)


def as_callback_list(callbacks: Any) -> CallbackList | None:
    if callbacks is None:
        return None
    if isinstance(callbacks, CallbackList):
        return callbacks
    if isinstance(callbacks, Callback):
        return CallbackList([callbacks])
    if isinstance(callbacks, (list, tuple)):
        return CallbackList([cb for cb in callbacks if cb is not None])
    raise TypeError(f"Unsupported callbacks type: {type(callbacks).__name__}")


class History(Callback):
    def __init__(self):
        self.steps: list[int] = []
        self.logs: list[dict[str, Any]] = []

    def on_train_step_end(self, step: int, *, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.steps.append(int(step))
        self.logs.append(dict(logs or {}))


class PrintLogger(Callback):
    def __init__(self, *, every: int = 1):
        self.every = max(1, int(every))

    def on_train_step_end(self, step: int, *, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if int(step) % self.every != 0:
            return
        logs = logs or {}
        loss = logs.get("loss")
        if loss is None:
            print(f"[train step {step}]")
        else:
            try:
                print(f"[train step {step}] loss={float(loss):.6f}")
            except Exception:
                print(f"[train step {step}] loss={loss}")


class JSONLLogger(Callback):
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._fp = None

    def on_train_begin(self, **kwargs: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def on_train_end(self, **kwargs: Any) -> None:
        if self._fp is None:
            return
        try:
            self._fp.flush()
        finally:
            self._fp.close()
            self._fp = None

    def on_train_step_end(self, step: int, *, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if self._fp is None:
            self.on_train_begin()
        record = {
            "time": time.time(),
            "step": int(step),
            **(logs or {}),
        }
        self._fp.write(json.dumps(record) + "\n")
        self._fp.flush()


@dataclass
class ModelCheckpoint(Callback):
    filepath: str | os.PathLike[str]
    monitor: str = "loss"
    mode: str = "min"  # "min" or "max"
    save_best_only: bool = True
    every: int = 1

    def __post_init__(self) -> None:
        self.filepath = Path(self.filepath)
        self.every = max(1, int(self.every))
        self._best: float | None = None

        mode = str(self.mode).lower().strip()
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode

    def _is_improved(self, value: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return value < self._best
        return value > self._best

    def _extract_model_to_save(self, model: Any):
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "save"):
            return inner
        return model

    def on_train_step_end(self, step: int, *, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if int(step) % self.every != 0:
            return

        logs = logs or {}
        raw = logs.get(self.monitor)
        if raw is None:
            return
        try:
            value = float(raw)
        except Exception:
            return

        if self.save_best_only and not self._is_improved(value):
            return

        model = kwargs.get("model")
        if model is None:
            return

        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        to_save = self._extract_model_to_save(model)
        try:
            to_save.save(self.filepath)
            self._best = value
        except Exception as e:
            warn("Failed to save checkpoint to '%s': %s", self.filepath, e)
