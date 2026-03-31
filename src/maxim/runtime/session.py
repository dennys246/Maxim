"""Session persistence for crash recovery and session replay."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maxim.agents.bus import StopReason


@dataclass
class AgentSession:
    """Persistent session state across agent loop runs."""

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    plan_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    stop_reason: StopReason | None = None
    total_steps: int = 0

    def log_event(self, category: str, title: str, detail: str = "") -> None:
        """Log a session-level event (goal changes, plan transitions, tool failures)."""
        self.events.append({
            "ts": time.time(),
            "cat": category,
            "title": title,
            "detail": detail,
        })

    def add_message(self, role: str, content: str) -> None:
        """Add a conversation turn."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        self.last_active_at = time.time()

    def save(self, sessions_dir: str | Path) -> None:
        """Persist session to JSON file."""
        sessions_dir = Path(sessions_dir)
        sessions_dir.mkdir(parents=True, exist_ok=True)
        path = sessions_dir / f"{self.session_id}.json"

        payload = {
            "version": "1.0",
            "session_id": self.session_id,
            "messages": self.messages,
            "events": self.events,
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "total_steps": self.total_steps,
        }

        # Atomic write
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, path)

    @classmethod
    def from_saved(cls, path: str | Path) -> "AgentSession":
        """Restore session from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        stop_reason = None
        if payload.get("stop_reason"):
            try:
                stop_reason = StopReason(payload["stop_reason"])
            except ValueError:
                pass

        return cls(
            session_id=payload["session_id"],
            messages=payload.get("messages", []),
            events=payload.get("events", []),
            plan_id=payload.get("plan_id"),
            created_at=payload.get("created_at", time.time()),
            last_active_at=payload.get("last_active_at", time.time()),
            stop_reason=stop_reason,
            total_steps=payload.get("total_steps", 0),
        )
