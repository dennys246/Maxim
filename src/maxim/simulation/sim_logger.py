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
    "PERCEPT": "\033[36m",  # Cyan
    "HIPPOCAMPUS": "\033[35m",  # Magenta
    "NAc": "\033[33m",  # Yellow
    "FEAR": "\033[31m",  # Red
    "PAIN": "\033[31;1m",  # Bold red
    "EXEC": "\033[32m",  # Green
    "MOTOR": "\033[34m",  # Blue
    "SALIENCE": "\033[33;1m",  # Bold yellow
    "SCN": "\033[37m",  # White
    "PIPELINE": "\033[37;2m",  # Dim white
    "RESULT": "\033[32;1m",  # Bold green
    "BLOCKED": "\033[31;1m",  # Bold red
    "CEREBELLUM": "\033[34;1m",  # Bold blue
    "ATL": "\033[35;1m",  # Bold magenta
    "SENSORY": "\033[36;1m",  # Bold cyan
    "BODY": "\033[37;1m",  # Bold white
}
_RESET = "\033[0m"

_sim_active = False
_sim_start: float = 0.0
_use_color = True
_log_file = None
_log_records: list[dict[str, Any]] = []
_debug_mode = False
_show_channels: set[str] | None = None  # None = show all, set = filter

# Channel → subsystem mapping for --show flag
_CHANNEL_MAP: dict[str, set[str]] = {
    "bio": {"HIPPOCAMPUS", "NAc", "SCN", "ATL", "FEAR", "PAIN", "MOTOR", "SENSORY", "BODY_STATE"},
    "exec": {"EXEC", "PIPELINE"},
    "sim": {"PERCEPT", "SCENE", "NPC", "CHOICE"},
    "memory": {"HIPPOCAMPUS", "NAc", "SCN", "ATL"},
    "safety": {"FEAR", "PAIN"},
}


def set_show_channels(channels: str | None) -> None:
    """Set which subsystem channels to show in terminal output.

    Args:
        channels: Comma-separated channel names (``"bio"``, ``"exec"``,
            ``"sim"``, ``"memory"``, ``"safety"``, ``"all"``).
            ``None`` or ``"all"`` shows everything.

    Examples::

        set_show_channels("bio")          # Only bio-system events
        set_show_channels("bio,exec")     # Bio + execution
        set_show_channels("all")          # Everything
        set_show_channels(None)           # Everything (default)
    """
    global _show_channels
    if channels is None or channels.strip().lower() == "all":
        _show_channels = None
        return

    allowed: set[str] = set()
    for ch in channels.split(","):
        ch = ch.strip().lower()
        if ch in _CHANNEL_MAP:
            allowed |= _CHANNEL_MAP[ch]
        else:
            # Treat as a raw subsystem name (e.g., "HIPPOCAMPUS")
            allowed.add(ch.upper())
    _show_channels = allowed if allowed else None


def _cleanup_log_file() -> None:
    """atexit handler — close log file if still open."""
    global _log_file
    if _log_file is not None:
        try:
            _log_file.close()
        except Exception:
            pass
        _log_file = None


import atexit
atexit.register(_cleanup_log_file)

# Subsystems that only print in debug mode (always persisted to JSONL log)
_DEBUG_ONLY_SUBSYSTEMS = {"PIPELINE"}


def enable_sim_logging(
    use_color: bool = True,
    log_path: str | None = None,
    debug: bool = False,
) -> None:
    """Enable simulation verbosity mode.

    Args:
        use_color: Use ANSI colors in terminal output.
        log_path: Path to save JSONL log file for future reference.
            Saved logs can be used for system refinement and as input
            to sleep mode's dream function for offline analysis.
        debug: If True, show all subsystem traces including PIPELINE
            internal polling. If False (default), PIPELINE traces are
            silenced from terminal but still persisted to the JSONL log.
    """
    global _sim_active, _sim_start, _use_color, _log_file, _log_records, _debug_mode
    _sim_active = True
    _sim_start = time.time()
    _log_records = []
    _debug_mode = debug
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

    import threading

    elapsed = time.time() - _sim_start
    timestamp = f"{elapsed:7.2f}s"
    # Tag with thread role so AUT vs orchestrator logs are distinguishable
    thread_name = threading.current_thread().name
    if "aut" in thread_name.lower() or "Thread-" in thread_name:
        thread_tag = "[AUT]"
    elif "Main" in thread_name:
        thread_tag = "[ORCH]"
    else:
        thread_tag = ""

    # Always persist structured record (JSONL log + in-memory)
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

    # Terminal output — skip debug-only subsystems unless debug mode
    if subsystem in _DEBUG_ONLY_SUBSYSTEMS and not _debug_mode:
        return

    # Channel filter — skip subsystems not in the active show set
    if _show_channels is not None and subsystem not in _show_channels:
        return

    if _use_color:
        color = _COLORS.get(subsystem, "")
        label = f"{color}[{subsystem:12s}]{_RESET}"
    else:
        label = f"[{subsystem:12s}]"

    line = f"  {timestamp} {label}{(' ' + thread_tag) if thread_tag else ''} {message}"

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


def sim_nac(event: str, outcome: str, rpe: float, confidence: float) -> None:
    """Log a NAc causal learning observation."""
    sim_log("NAc", f"Causal link: {event} -> {outcome} (RPE={rpe:.2f}, confidence={confidence:.2f})")


def sim_scn(memory_id: str, phase: str, significance: float) -> None:
    """Log an SCN temporal bin registration."""
    sim_log("SCN", f"Registered {memory_id[:8]} in {phase} (significance={significance:.2f})")


def sim_cerebellum(entity: str, affordance: str, confidence: float, error: float | None = None) -> None:
    """Log a Cerebellum forward model observation."""
    if error is not None:
        sim_log("CEREBELLUM", f"Observed {entity}.{affordance} (conf={confidence:.2f}, pred_error={error:.3f})")
    else:
        sim_log("CEREBELLUM", f"New model: {entity}.{affordance} (conf={confidence:.2f})")


def sim_sensory(modality: str, entity: str, acuity: float, dropped: bool = False) -> None:
    """Log a SensoryGate modulation event."""
    if dropped:
        sim_log("SENSORY", f"Dropped {modality} percept from {entity} (acuity={acuity:.2f})")
    else:
        sim_log("SENSORY", f"Modulated {modality} from {entity} (acuity={acuity:.2f})")


def sim_body_state(entity_count: int, active_failures: int) -> None:
    """Log body state injection into prompt."""
    sim_log("BODY", f"State: {entity_count} entities, {active_failures} active failures")
