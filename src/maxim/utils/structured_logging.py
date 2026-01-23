"""Structured logging for the abstraction stream.

Provides compact, JSON-serializable log records that flow to LLM context
without raw images or excessive data.

Verbosity Levels for Agentic Information Flow:
- 0 (QUIET): Errors and critical warnings only
- 1 (NORMAL): Key events: goal proposals, tool executions, mode changes
- 2 (VERBOSE): + Perception events, memory updates, autonomy decisions
- 3 (DEBUG): + Loop iterations, rate limiting, internal state changes
"""

from __future__ import annotations

import enum
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class AgenticVerbosity(enum.IntEnum):
    """Verbosity levels for agentic information flow."""

    QUIET = 0  # Errors only
    NORMAL = 1  # Key events (goals, tools, mode changes)
    VERBOSE = 2  # + Perception, memory, autonomy
    DEBUG = 3  # + Loop iterations, internal state


# Event categories for filtering
class EventCategory(enum.Enum):
    """Categories of agentic events for verbosity filtering."""

    ERROR = "error"  # Always shown (level 0+)
    GOAL = "goal"  # Goal proposals, completions (level 1+)
    TOOL = "tool"  # Tool calls, results (level 1+)
    MODE = "mode"  # Mode changes (level 1+)
    PERCEPTION = "perception"  # Percepts, detections (level 2+)
    MEMORY = "memory"  # Memory updates (level 2+)
    AUTONOMY = "autonomy"  # Autonomy decisions (level 2+)
    LOOP = "loop"  # Loop iterations (level 3+)
    INTERNAL = "internal"  # Internal state (level 3+)


# Map events to their minimum verbosity level
EVENT_VERBOSITY: dict[str, int] = {
    # Level 0: Always shown
    "error": 0,
    "critical": 0,
    "hard_stop": 0,
    "emergency_halt": 0,
    # Level 1: Key events
    "goal_proposed": 1,
    "goal_completed": 1,
    "goal_failed": 1,
    "goal_cancelled": 1,
    "tool_called": 1,
    "tool_result": 1,
    "mode_change": 1,
    "startup": 1,
    "shutdown": 1,
    "action_executed": 1,
    "action_rejected": 1,
    # Level 2: Detailed events
    "percept": 2,
    "detection": 2,
    "speech_detected": 2,
    "memory_store": 2,
    "memory_promote": 2,
    "memory_decay": 2,
    "autonomy_check": 2,
    "autonomy_level_change": 2,
    "context_built": 2,
    "intent_proposed": 2,
    # Level 3: Debug events
    "loop_iteration": 3,
    "rate_limited": 3,
    "no_target": 3,
    "within_deadzone": 3,
    "skipped": 3,
    "idle": 3,
    "state_persist": 3,
}


@dataclass
class LogRecord:
    """Structured log record for abstraction stream."""

    timestamp: float
    level: str
    source: str  # agent name or component
    event: str  # action type: "percept", "goal_proposed", "tool_called", etc.
    data: dict[str, Any] = field(default_factory=dict)

    def to_compact(self) -> dict[str, Any]:
        """Compact representation for LLM context."""
        return {
            "t": round(self.timestamp, 2),
            "l": self.level[0] if self.level else "I",  # "I", "W", "E", "D"
            "s": self.source,
            "e": self.event,
            **{k: v for k, v in self.data.items() if v is not None},
        }

    def to_verbose(self) -> dict[str, Any]:
        """Verbose representation with full field names."""
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "source": self.source,
            "event": self.event,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_compact(), separators=(",", ":"))

    def to_human(self) -> str:
        """Human-readable format for console output."""
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        ms = int((self.timestamp % 1) * 1000)
        data_str = ""
        if self.data:
            # Format key data fields nicely
            parts = []
            for k, v in self.data.items():
                if v is not None:
                    if isinstance(v, float):
                        parts.append(f"{k}={v:.2f}")
                    elif isinstance(v, dict):
                        parts.append(f"{k}={{...}}")
                    elif isinstance(v, list):
                        parts.append(f"{k}=[{len(v)}]")
                    else:
                        parts.append(f"{k}={v}")
            if parts:
                data_str = " | " + ", ".join(parts)
        return f"{ts}.{ms:03d} [{self.source}] {self.event}{data_str}"


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs compact JSON for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = LogRecord(
            timestamp=record.created,
            level=record.levelname,
            source=record.name.split(".")[-1],
            event=getattr(record, "event", "log"),
            data=getattr(record, "data", {}),
        )
        return log_record.to_json()


class AbstractionBuffer:
    """Thread-safe ring buffer for recent abstraction stream entries.

    Supports verbosity-aware filtering and multiple output formats.
    """

    def __init__(
        self,
        max_entries: int = 200,
        verbosity: int = 1,
        console_output: bool = False,
    ) -> None:
        self._entries: list[LogRecord] = []
        self._max = max_entries
        self._lock = threading.Lock()
        self._verbosity = verbosity
        self._console_output = console_output
        self._callbacks: list[Callable[[LogRecord], None]] = []

    @property
    def verbosity(self) -> int:
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value: int) -> None:
        self._verbosity = max(0, min(3, int(value)))

    @property
    def console_output(self) -> bool:
        return self._console_output

    @console_output.setter
    def console_output(self, value: bool) -> None:
        self._console_output = bool(value)

    def add_callback(self, callback: Callable[[LogRecord], None]) -> None:
        """Add a callback for real-time log streaming."""
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[LogRecord], None]) -> None:
        """Remove a callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def append(self, record: LogRecord) -> None:
        """Append a record, respecting verbosity filtering."""
        # Check if this event passes the verbosity filter
        event_min_verbosity = EVENT_VERBOSITY.get(record.event, 1)
        if event_min_verbosity > self._verbosity:
            return  # Skip this event based on verbosity

        with self._lock:
            self._entries.append(record)
            if len(self._entries) > self._max:
                self._entries = self._entries[-self._max:]

            # Console output if enabled
            if self._console_output:
                print(f"[AGENTIC] {record.to_human()}")

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(record)
                except Exception:
                    pass

    def get_recent(self, n: int = 20, verbose: bool = False) -> list[dict[str, Any]]:
        """Get N most recent entries."""
        with self._lock:
            if verbose:
                return [e.to_verbose() for e in self._entries[-n:]]
            return [e.to_compact() for e in self._entries[-n:]]

    def get_recent_human(self, n: int = 20) -> list[str]:
        """Get N most recent entries in human-readable format."""
        with self._lock:
            return [e.to_human() for e in self._entries[-n:]]

    def get_by_source(self, source: str, n: int = 10) -> list[dict[str, Any]]:
        """Get recent entries from a specific source."""
        with self._lock:
            filtered = [e for e in self._entries if e.source == source]
            return [e.to_compact() for e in filtered[-n:]]

    def get_by_event(self, event: str, n: int = 10) -> list[dict[str, Any]]:
        """Get recent entries of a specific event type."""
        with self._lock:
            filtered = [e for e in self._entries if e.event == event]
            return [e.to_compact() for e in filtered[-n:]]

    def get_by_verbosity(self, max_verbosity: int, n: int = 20) -> list[dict[str, Any]]:
        """Get entries that would be shown at a given verbosity level."""
        with self._lock:
            filtered = [
                e for e in self._entries
                if EVENT_VERBOSITY.get(e.event, 1) <= max_verbosity
            ]
            return [e.to_compact() for e in filtered[-n:]]

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the buffer contents."""
        with self._lock:
            event_counts: dict[str, int] = {}
            source_counts: dict[str, int] = {}
            for e in self._entries:
                event_counts[e.event] = event_counts.get(e.event, 0) + 1
                source_counts[e.source] = source_counts.get(e.source, 0) + 1
            return {
                "total_entries": len(self._entries),
                "verbosity": self._verbosity,
                "console_output": self._console_output,
                "events": event_counts,
                "sources": source_counts,
            }

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# Global abstraction buffer (singleton)
_abstraction_buffer: AbstractionBuffer | None = None
_buffer_lock = threading.Lock()
_global_verbosity: int = 1


def get_abstraction_buffer() -> AbstractionBuffer:
    """Get or create the global abstraction buffer."""
    global _abstraction_buffer
    with _buffer_lock:
        if _abstraction_buffer is None:
            verbosity = int(os.environ.get("MAXIM_AGENTIC_VERBOSITY", _global_verbosity))
            console = os.environ.get("MAXIM_AGENTIC_CONSOLE", "").lower() in ("1", "true", "yes")
            _abstraction_buffer = AbstractionBuffer(
                max_entries=500,
                verbosity=verbosity,
                console_output=console,
            )
        return _abstraction_buffer


def configure_agentic_verbosity(
    verbosity: int = 1,
    console_output: bool = False,
) -> None:
    """Configure the global agentic verbosity level.

    Args:
        verbosity: 0=quiet, 1=normal, 2=verbose, 3=debug
        console_output: If True, print all events to console
    """
    global _global_verbosity
    _global_verbosity = max(0, min(3, int(verbosity)))

    buf = get_abstraction_buffer()
    buf.verbosity = _global_verbosity
    buf.console_output = console_output


def log_structured(
    logger: logging.Logger,
    level: int,
    event: str,
    data: dict[str, Any] | None = None,
    msg: str = "",
) -> None:
    """Log a structured event that flows to the abstraction stream."""
    record = LogRecord(
        timestamp=time.time(),
        level=logging.getLevelName(level),
        source=logger.name.split(".")[-1],
        event=event,
        data=data or {},
    )
    get_abstraction_buffer().append(record)

    # Also log normally (respecting standard logging verbosity)
    event_min = EVENT_VERBOSITY.get(event, 1)
    if event_min <= 1:
        # Key events always go to standard log
        logger.log(level, msg or event, extra={"event": event, "data": data or {}})
    elif event_min == 2:
        logger.debug(msg or event, extra={"event": event, "data": data or {}})
    # Level 3 events only go to abstraction buffer


def log_agentic(
    source: str,
    event: str,
    data: dict[str, Any] | None = None,
    level: str = "INFO",
) -> None:
    """Log an agentic event directly to the abstraction stream.

    This is a lighter-weight alternative to log_structured() that doesn't
    require a logger instance.

    Args:
        source: Component name (e.g., "agent_loop", "exec_agent")
        event: Event type (e.g., "goal_proposed", "tool_called")
        data: Additional event data
        level: Log level (INFO, DEBUG, WARNING, ERROR)
    """
    record = LogRecord(
        timestamp=time.time(),
        level=level,
        source=source,
        event=event,
        data=data or {},
    )
    get_abstraction_buffer().append(record)


def compact_detection(det: dict[str, Any]) -> dict[str, Any]:
    """Compact a YOLO detection for LLM context."""
    return {
        "id": det.get("track_id"),
        "cls": det.get("class_id"),
        "lbl": det.get("label", ""),
        "c": round(det.get("conf", 0), 2),
        "box": [round(x, 1) for x in det.get("bbox_xyxy", [0, 0, 0, 0])],
    }


def compact_vision_event(event: dict[str, Any]) -> dict[str, Any]:
    """Compact a vision event for LLM context."""
    dets = event.get("detections", [])
    return {
        "t": round(event.get("timestamp", 0), 2),
        "n": len(dets),
        "dets": [compact_detection(d) for d in dets[:10]],  # Limit to 10
    }
