from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeState:
    max_steps: int = 100
    confirmed: bool = False
    steps_taken: int = 0
    done: bool = False
    last_error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def update(self, observation: dict[str, Any] | None) -> None:
        if not observation:
            return
        if isinstance(observation, dict):
            self.data.update(observation)

    def snapshot(self) -> dict[str, Any]:
        return {
            "steps_taken": int(self.steps_taken),
            "max_steps": int(self.max_steps),
            "confirmed": bool(self.confirmed),
            "done": bool(self.done),
            "last_error": self.last_error,
            "data": dict(self.data),
        }

    def save_json(self, path: str, *, meta: dict[str, Any] | None = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload: dict[str, Any] = {"saved_at": time.time(), **self.snapshot()}
        if meta:
            payload.update(meta)

        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, default=str)
        os.replace(tmp, path)

    @classmethod
    def load_json(cls, path: str) -> "RuntimeState":
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            data = {}

        return cls(
            max_steps=int(payload.get("max_steps", 100) or 100),
            confirmed=bool(payload.get("confirmed", False)),
            steps_taken=int(payload.get("steps_taken", 0) or 0),
            done=bool(payload.get("done", False)),
            last_error=payload.get("last_error"),
            data=data,
        )

    def mark_failure(self, error: str | None) -> None:
        self.last_error = str(error) if error is not None else None

    def is_done(self) -> bool:
        if self.done:
            return True
        if int(self.steps_taken) >= int(self.max_steps):
            return True
        return False
