"""Simulation verbosity — traces bio-inspired subsystem activity during --sim.

When simulation mode is active, this logger prints human-readable traces
of what each biological subsystem is doing, making it easy to follow the
percept → memory → goal → action pipeline.

Display tier system (v1.0):
  CLEAN       Scenes, agent actions, prompts only (default)
  BIO         + condensed bio-system annotations (memory, causal, pain)
  DEBUG       + full subsystem traces (same as legacy --show all)

Interactive mode (orthogonal to display tier):
  AUTO        DM campaigns → prompt, generative sims → no prompt (default)
  ON          Always prompt at choice/confirmation points
  OFF         Never prompt — use policy defaults
"""

from __future__ import annotations

import enum
import logging
import sys
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Display tier — controls what appears on the user's console
# ─────────────────────────────────────────────────────────────────────────────


class DisplayTier(enum.IntEnum):
    """Output detail level for the user's console.

    CLEAN: narrative only — scenes, actions, prompts, summaries.
    BIO: + condensed bio annotations (memory, causal learning, pain).
    DEBUG: + full subsystem traces, pipeline timing, internal state.
    """

    CLEAN = 0
    BIO = 1
    DEBUG = 2


class InteractiveMode(enum.Enum):
    """Whether the system pauses for user input.

    AUTO: context-dependent (DM campaigns → yes, generative → no).
    ON: always prompt at choice/confirmation points.
    OFF: never prompt — use policy defaults (NonInteractiveHandler).
    """

    AUTO = "auto"
    ON = "on"
    OFF = "off"


# Global display state
_display_tier: DisplayTier = DisplayTier.CLEAN
_interactive_mode: InteractiveMode = InteractiveMode.AUTO
_display_floor: DisplayTier = DisplayTier.CLEAN  # User's --display setting (agent can't go below)


def set_display_tier(tier: DisplayTier | str) -> None:
    """Set the display tier globally."""
    global _display_tier, _display_floor
    if isinstance(tier, str):
        tier = DisplayTier[tier.upper()]
    _display_tier = tier
    _display_floor = tier  # User-set tier is also the floor


def get_display_tier() -> DisplayTier:
    """Get the current display tier."""
    return _display_tier


def set_interactive_mode(mode: InteractiveMode | str) -> None:
    """Set the interactive mode globally."""
    global _interactive_mode
    if isinstance(mode, str):
        mode = InteractiveMode(mode.lower())
    _interactive_mode = mode


def get_interactive_mode() -> InteractiveMode:
    """Get the current interactive mode."""
    return _interactive_mode


def agent_escalate_display(tier: DisplayTier) -> bool:
    """Allow the agent to temporarily escalate display tier.

    Returns True if escalation was applied, False if tier is below floor.
    """
    global _display_tier
    if tier < _display_floor:
        return False  # Agent can't suppress below user's floor
    _display_tier = tier
    return True


def revert_display_to_floor() -> None:
    """Revert display tier to user's --display setting."""
    global _display_tier
    _display_tier = _display_floor


_CRITICAL_CONTEXTS = frozenset(
    {
        "plan_approval",
        "safety_escalation",
        "autonomy_escalation",
    }
)
"""Contexts that override OFF mode — these represent decisions too
important to auto-resolve silently."""


def should_prompt(context: str = "", *, force: bool = False) -> bool:
    """Determine whether to prompt the user for input.

    Args:
        context: What's requesting the prompt — "dm_campaign",
            "agentic_mode", "autonomy_escalation", "plan_approval",
            "safety_escalation", "agent_request", "confirmation", etc.
        force: If True, prompt regardless of interactive mode. Use
            sparingly — only for decisions that must not be auto-resolved.

    Returns:
        True if the prompt should be shown to the user.
    """
    if force:
        return True
    if _interactive_mode == InteractiveMode.ON:
        return True
    if _interactive_mode == InteractiveMode.OFF:
        # Critical contexts still prompt even when OFF
        return context in _CRITICAL_CONTEXTS
    # AUTO: derive from context
    return context in (
        "dm_campaign",
        "agentic_mode",
        "autonomy_escalation",
        "plan_approval",
        "safety_escalation",
        "agent_request",
        "confirmation",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Display output functions — tier-aware
# ─────────────────────────────────────────────────────────────────────────────

_DISPLAY_COLORS = {
    "scene": "\033[37;1m",  # Bold white
    "action": "\033[32m",  # Green
    "response": "\033[36m",  # Cyan
    "turn": "\033[37;2m",  # Dim white
    "summary": "\033[33m",  # Yellow
    "bio": "\033[35;2m",  # Dim magenta
    "entity": "\033[34;2m",  # Dim blue
}


def _emit(text: str, color_key: str | None = None) -> None:
    """Emit a line to the active display or stdout.

    When a ``MaximDisplay`` is active, routes through ``display.log()``
    to avoid corrupting the rich ``Live`` panel with raw ANSI output.
    Falls back to ``print()`` when no display is active.
    """
    display = get_active_display()
    if display is not None:
        # Route through the display's log panel. Use color_key as
        # subsystem label so the display can apply its own styling.
        display.log(color_key or "info", text)
        return
    if color_key and _use_color:
        c = _DISPLAY_COLORS.get(color_key, "")
        print(f"{c}{text}{_RESET}", flush=True)
    else:
        print(text, flush=True)


def display_scene(text: str) -> None:
    """Show scene/percept text (CLEAN tier)."""
    if _display_tier >= DisplayTier.CLEAN:
        _emit(text, "scene")


def display_action(tool: str, params: dict[str, Any] | None = None) -> None:
    """Show agent action (CLEAN tier)."""
    if _display_tier >= DisplayTier.CLEAN:
        if params:
            param_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in params.items())
            _emit(f"  > Agent uses {tool}({param_str})", "action")
        else:
            _emit(f"  > Agent uses {tool}()", "action")


def display_response(text: str) -> None:
    """Show agent response text (CLEAN tier)."""
    if _display_tier >= DisplayTier.CLEAN:
        _emit(f"  Agent: {text}", "response")


def display_entity_state(name: str, sensors: dict[str, Any]) -> None:
    """Show entity sensor state (CLEAN tier)."""
    if _display_tier >= DisplayTier.CLEAN and sensors:
        parts = ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in sensors.items())
        _emit(f"  [{name}: {parts}]", "entity")


def display_turn(n: int) -> None:
    """Show turn marker (CLEAN tier)."""
    if _display_tier >= DisplayTier.CLEAN:
        display = get_active_display()
        if display is not None:
            display.set_status(turn=str(n))
        else:
            _emit(f"\n{'─' * 2} Turn {n} {'─' * 40}", "turn")


def display_summary(lines: list[str]) -> None:
    """Show final summary (CLEAN tier)."""
    if _display_tier >= DisplayTier.CLEAN:
        for line in lines:
            _emit(line, "summary")


def display_status(message: str) -> None:
    """Show system status message (DEBUG tier only)."""
    if _display_tier >= DisplayTier.DEBUG:
        _emit(f"  {message}", "turn")


# ─────────────────────────────────────────────────────────────────────────────
# sim_log subsystem traces — wired across the runtime, gated per-subsystem
# ─────────────────────────────────────────────────────────────────────────────

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

# Per-subsystem minimum display tier. An event for subsystem S surfaces to the
# terminal only if the active display tier >= _SUBSYSTEM_TIERS[S]. All events
# are persisted to the JSONL log regardless of tier — this map only controls
# terminal visibility.
#
# BIO-level subsystems are the "bio annotations" users want at --display bio:
# memory events, causal learning, safety gates, pain, tool execution, high-
# level perception and choice. DEBUG-level subsystems are granular pipeline
# traces that would flood bio mode. Unknown subsystems default to BIO so new
# code emitting to sim_log surfaces by default (opt-out rather than opt-in).
_SUBSYSTEM_TIERS: dict[str, "DisplayTier"] = {
    # Bio-tier: surface at --display bio and above. All named biological
    # subsystems belong here — they are the "bio annotations" users expect
    # when they pass --display bio. The _CHANNEL_MAP below already groups
    # all of these under the "bio" --show channel; the tier map must agree.
    "HIPPOCAMPUS": DisplayTier.BIO,
    "NAc": DisplayTier.BIO,
    "ATL": DisplayTier.BIO,
    "SCN": DisplayTier.BIO,
    "CEREBELLUM": DisplayTier.BIO,
    "SENSORY": DisplayTier.BIO,
    "BODY": DisplayTier.BIO,
    "BODY_STATE": DisplayTier.BIO,
    "FEAR": DisplayTier.BIO,
    "BLOCKED": DisplayTier.BIO,
    "PAIN": DisplayTier.BIO,
    "MOTOR": DisplayTier.BIO,
    "EXEC": DisplayTier.BIO,
    "PERCEPT": DisplayTier.BIO,
    "SCENE": DisplayTier.BIO,
    "CHOICE": DisplayTier.BIO,
    "NPC": DisplayTier.BIO,
    "RESULT": DisplayTier.BIO,
    # Debug-tier: granular pipeline internals that would flood bio mode.
    # These are implementation traces, not biological subsystem events.
    "PIPELINE": DisplayTier.DEBUG,
    "SALIENCE": DisplayTier.DEBUG,
}

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


# ─────────────────────────────────────────────────────────────────────────────
# Active display — when set, terminal output routes through MaximDisplay
# instead of raw print(). JSONL file writes are unchanged (always persist).
# ─────────────────────────────────────────────────────────────────────────────

_active_display: Any = None  # MaximDisplay | None
_display_lock = threading.Lock()
_display_log_handler: logging.Handler | None = None


class _DisplayLoggingHandler(logging.Handler):
    """Routes Python logging WARNING+ through MaximDisplay when active.

    Prevents standard logging output (CostTracker, role_divergence, etc.)
    from writing raw text to stderr and corrupting the Rich Live panel.
    """

    def __init__(self, display: Any) -> None:
        super().__init__(level=logging.WARNING)
        self._display = display

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._display.log("info", msg)
        except Exception:
            pass


def set_active_display(display: Any) -> None:
    """Set the active MaximDisplay for terminal output routing.

    When set, ``_emit()`` routes through ``display.log()`` instead of
    ``print()``. A logging handler is installed to intercept WARNING+
    messages that would otherwise corrupt the Rich Live panel.
    Pass ``None`` to revert to direct ANSI printing.
    """
    global _active_display, _display_log_handler
    with _display_lock:
        root = logging.getLogger()
        # Remove previous display handler if any
        if _display_log_handler is not None:
            root.removeHandler(_display_log_handler)
            _display_log_handler = None
        _active_display = display
        if display is not None:
            # Install handler that routes warnings through the display
            # and suppresses them from the stderr StreamHandler.
            handler = _DisplayLoggingHandler(display)
            handler.setFormatter(logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
            root.addHandler(handler)
            # Suppress WARNING+ from stderr StreamHandlers while display is active.
            # Remove any existing display filters first to prevent accumulation
            # on repeated set_active_display() calls.
            for h in root.handlers:
                if isinstance(h, logging.StreamHandler) and not isinstance(
                    h, (logging.FileHandler, _DisplayLoggingHandler)
                ):
                    for f in list(h.filters):
                        if isinstance(f, _DisplayStreamFilter):
                            h.removeFilter(f)
                    h.addFilter(_DisplayStreamFilter())
            _display_log_handler = handler
        else:
            # Remove suppression filters from StreamHandlers
            for h in root.handlers:
                for f in list(h.filters):
                    if isinstance(f, _DisplayStreamFilter):
                        h.removeFilter(f)


class _DisplayStreamFilter(logging.Filter):
    """Suppresses WARNING+ from stderr while the display is active."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING and get_active_display() is not None:
            return False
        return True


def get_active_display() -> Any:
    """Return the active MaximDisplay, or None."""
    with _display_lock:
        return _active_display


def _cleanup_display() -> None:
    """atexit handler — stop active display if still running."""
    display = get_active_display()
    if display is not None:
        try:
            display.stop()
        except Exception:
            pass


atexit.register(_cleanup_display)


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

    # Apply env var channel filter if no explicit --show was set
    import os

    if _show_channels is None:
        env_channels = os.environ.get("MAXIM_SHOW_CHANNELS", "").strip()
        if env_channels:
            set_show_channels(env_channels)

    if log_path:
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


def sim_log(
    subsystem: str,
    message: str,
    data: dict[str, Any] | None = None,
    *,
    _force_debug: bool = False,
) -> None:
    """Log a simulation event with subsystem label and timestamp.

    Events are printed to the terminal AND persisted to a JSONL file
    (if log_path was provided to enable_sim_logging). Persisted logs
    can be used for system refinement, replay analysis, and as input
    to sleep mode's dream function for offline pattern discovery.

    Args:
        subsystem: Bio-inspired subsystem name (PERCEPT, HIPPOCAMPUS, etc.)
        message: Human-readable description of what happened
        data: Optional structured data for detailed inspection
        _force_debug: If True, only show on terminal in debug mode
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

    # Display-tier gate. Each subsystem declares its minimum visible tier via
    # _SUBSYSTEM_TIERS (unknown subsystems default to BIO). _force_debug
    # escalates an event to DEBUG tier regardless of its subsystem's baseline.
    # The legacy --debug runtime flag (_debug_mode) bypasses the tier gate
    # entirely for backward compatibility with verbose-mode CLI users.
    if not _debug_mode:
        min_tier = DisplayTier.DEBUG if _force_debug else _SUBSYSTEM_TIERS.get(subsystem, DisplayTier.BIO)
        if _display_tier < min_tier:
            return

    # Channel filter — skip subsystems not in the active show set
    if _show_channels is not None and subsystem not in _show_channels:
        return

    # Build the display line
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

    # Route terminal output through active display or direct print
    display = get_active_display()
    if display is not None:
        # Strip ANSI codes — the display applies its own styling
        plain_msg = f"{timestamp} [{subsystem}] {message}"
        if data:
            details = ", ".join(f"{k}={v}" for k, v in data.items() if v is not None)
            if details:
                plain_msg += f"  ({details})"
        display.log(subsystem.lower(), plain_msg)
    else:
        print(line, flush=True)


def sim_percept(source: str, summary: str, **kwargs: Any) -> None:
    """Log an incoming percept."""
    sim_log("PERCEPT", f"[{source}] {summary}", kwargs if kwargs else None)


def sim_memory(action: str, **kwargs: Any) -> None:
    """Log a hippocampus/memory event."""
    sim_log("HIPPOCAMPUS", action, kwargs if kwargs else None)


def sim_debug(subsystem: str, action: str, **kwargs: Any) -> None:
    """Log an event that only appears in debug mode (--debug).

    Always persisted to JSONL; terminal output suppressed unless debug=True.
    """
    sim_log(subsystem, action, kwargs if kwargs else None, _force_debug=True)


def sim_reaction(kind: str, intensity: float, source: str, **kwargs: Any) -> None:
    """Log a Reaction from the ReactionBus.

    Generalizes sim_pain for all reaction kinds. Replaces the sim_pain
    call lost when route_pain_percept was deleted in Phase 2a.
    """
    sim_log("REACTION", f"{kind} (intensity={intensity:.2f}) from {source}", kwargs if kwargs else None)


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
