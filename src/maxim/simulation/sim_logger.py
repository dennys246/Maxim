"""Simulation verbosity — traces bio-inspired subsystem activity during --sim.

When simulation mode is active, this logger prints human-readable traces
of what each biological subsystem is doing, making it easy to follow the
percept → memory → goal → action pipeline.

Subsystem labels use bio-inspired naming:
  [PERCEPT]       Incoming sensory input (ScenarioSource)
  [HIPPOCAMPUS]   Memory formation, recall, consolidation
  [NAc]           Reward prediction, causal learning
  [FEAR]          FearAgent safety review
  [PAIN]          Pain signal detection and routing
  [EXEC]          ExecAgent goal proposal
  [MOTOR]         Motor command / tool execution
  [SALIENCE]      Attention and novelty tracking
  [SCN]           Temporal indexing
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
_COLORS = {
    "PERCEPT": "\033[36m",      # Cyan
    "HIPPOCAMPUS": "\033[35m",  # Magenta
    "NAc": "\033[33m",          # Yellow
    "FEAR": "\033[31m",         # Red
    "PAIN": "\033[31;1m",       # Bold red
    "EXEC": "\033[32m",         # Green
    "MOTOR": "\033[34m",        # Blue
    "SALIENCE": "\033[33;1m",   # Bold yellow
    "SCN": "\033[37m",          # White
    "PIPELINE": "\033[37;2m",   # Dim white
    "RESULT": "\033[32;1m",     # Bold green
    "BLOCKED": "\033[31;1m",    # Bold red
}
_RESET = "\033[0m"

_sim_active = False
_sim_start: float = 0.0
_use_color = True
_log_file = None
_log_records: list[dict[str, Any]] = []


def enable_sim_logging(use_color: bool = True, log_path: str | None = None) -> None:
    """Enable simulation verbosity mode.

    Args:
        use_color: Use ANSI colors in terminal output.
        log_path: Path to save JSONL log file for future reference.
            Saved logs can be used for system refinement and as input
            to sleep mode's dream function for offline analysis.
    """
    global _sim_active, _sim_start, _use_color, _log_file, _log_records
    _sim_active = True
    _sim_start = time.time()
    _log_records = []
    _use_color = use_color and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    if log_path:
        import os
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            _log_file = open(log_path, "w")
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to open sim log file %s: %s", log_path, e)
            _log_file = None

    sim_log("PIPELINE", "Simulation logging enabled")


def disable_sim_logging() -> str | None:
    """Disable simulation verbosity mode.

    Returns:
        Path to the saved log file, or None if no log was saved.
    """
    global _sim_active, _log_file
    _sim_active = False
    log_path = None
    if _log_file is not None:
        log_path = _log_file.name
        _log_file.close()
        _log_file = None
    return log_path


def get_sim_records() -> list[dict[str, Any]]:
    """Get all simulation log records (for programmatic access)."""
    return list(_log_records)


def sim_log(subsystem: str, message: str, data: dict[str, Any] | None = None) -> None:
    """Log a simulation event with subsystem label and timestamp.

    Events are printed to the terminal AND persisted to a JSONL file
    (if log_path was provided to enable_sim_logging). Persisted logs
    can be used for system refinement, replay analysis, and as input
    to sleep mode's dream function for offline pattern discovery.

    Args:
        subsystem: Bio-inspired subsystem name (PERCEPT, HIPPOCAMPUS, etc.)
        message: Human-readable description of what happened
        data: Optional structured data for detailed inspection
    """
    if not _sim_active:
        return

    elapsed = time.time() - _sim_start
    timestamp = f"{elapsed:7.2f}s"

    # Persist structured record
    record = {
        "t": round(elapsed, 3),
        "subsystem": subsystem,
        "message": message,
        "data": data or {},
    }
    _log_records.append(record)

    if _log_file is not None:
        import json
        _log_file.write(json.dumps(record) + "\n")
        _log_file.flush()

    # Terminal output
    if _use_color:
        color = _COLORS.get(subsystem, "")
        label = f"{color}[{subsystem:12s}]{_RESET}"
    else:
        label = f"[{subsystem:12s}]"

    line = f"  {timestamp} {label} {message}"

    if data:
        # Format key data inline (compact)
        details = ", ".join(f"{k}={v}" for k, v in data.items() if v is not None)
        if details:
            line += f"  ({details})"

    print(line, flush=True)


def sim_percept(source: str, summary: str, **kwargs: Any) -> None:
    """Log an incoming percept."""
    sim_log("PERCEPT", f"[{source}] {summary}", kwargs if kwargs else None)


def sim_memory(action: str, **kwargs: Any) -> None:
    """Log a hippocampus/memory event."""
    sim_log("HIPPOCAMPUS", action, kwargs if kwargs else None)


def sim_pain(pain_type: str, intensity: float, **kwargs: Any) -> None:
    """Log a pain signal."""
    sim_log("PAIN", f"{pain_type} (intensity={intensity:.2f})", kwargs if kwargs else None)


def sim_fear(tool: str, allowed: bool, reason: str = "") -> None:
    """Log a FearAgent review."""
    if allowed:
        sim_log("FEAR", f"ALLOWED: {tool}")
    else:
        sim_log("BLOCKED", f"BLOCKED: {tool} — {reason}")


def sim_action(tool: str, success: bool, summary: str = "") -> None:
    """Log a tool execution."""
    status = "OK" if success else "FAIL"
    sim_log("MOTOR", f"[{status}] {tool}: {summary}" if summary else f"[{status}] {tool}")


def sim_result(scenario_name: str, passed: bool, met: int, failed: int) -> None:
    """Log final scenario result."""
    status = "PASS" if passed else "FAIL"
    subsystem = "RESULT" if passed else "BLOCKED"
    sim_log(subsystem, f"{status}: {scenario_name} ({met} passed, {failed} failed)")
