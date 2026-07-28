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

import contextvars
import enum
import logging
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)

# Dedicated bridge logger for sim_log → MAXIM_LOG_FILE unification. Configured
# at module import: never propagates to the root logger (which would double-
# print to stdout) and accepts DEBUG so the JSONL FileHandler attached to
# root by ``configure_logging`` captures everything via root.handlers — but
# only via handlers that opt into traversal. The pattern:
#  - propagate=False  → no double-printing to the root stdout StreamHandler
#  - we manually mirror onto root.handlers that are JSONL-tagged
# This keeps human stdout clean while feeding the unified post-mortem trail.
_sim_bridge_logger = logging.getLogger("maxim.sim")
_sim_bridge_logger.setLevel(logging.DEBUG)
_sim_bridge_logger.propagate = False


def _ensure_sim_bridge_handlers() -> None:
    """Reconcile bridge handlers with root's JSONL handlers.

    Called lazily on each sim_log emission.  ``configure_logging(force=True)``
    closes existing root handlers and attaches new ones (same baseFilename,
    different FD); without reconciliation the bridge keeps writing through
    the closed handler's FD and lines silently vanish (or, worse, write
    into a rotated archive after RotatingFileHandler triggers).  Pre-merge
    review caught this as a HIGH-severity bug.

    The reconciliation:
      1. Drop bridge handlers that are no longer present in root (closed).
      2. Add root handlers tagged ``_maxim_jsonl`` that the bridge lacks.

    Identity match on ``id(handler)`` — same baseFilename with a different
    handler instance means root rebuilt it; we want to follow.
    """
    root = logging.getLogger()
    root_jsonl = {
        id(h): h for h in root.handlers if isinstance(h, logging.FileHandler) and getattr(h, "_maxim_jsonl", False)
    }
    bridge_ids = {id(h) for h in _sim_bridge_logger.handlers}

    # Drop stale (closed) handlers — those whose id no longer appears in root.
    for h in list(_sim_bridge_logger.handlers):
        if id(h) not in root_jsonl:
            _sim_bridge_logger.removeHandler(h)

    # Add fresh handlers from root that aren't already on the bridge.
    for handler_id, h in root_jsonl.items():
        if handler_id not in bridge_ids:
            _sim_bridge_logger.addHandler(h)


# ContextVar for implicit agent_id threading. The agent loop sets this
# once per turn via ``sim_agent_context(agent_id)``; all nested sim_log
# calls within that scope automatically pick it up without requiring
# every bio-system to accept and forward agent_id explicitly.
_current_agent_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("_current_agent_id", default=None)


@contextmanager
def sim_agent_context(agent_id: str | None) -> Generator[None, None, None]:
    """Set the implicit agent_id for all sim_log calls within this scope."""
    token = _current_agent_id.set(agent_id)
    try:
        yield
    finally:
        _current_agent_id.reset(token)


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

# Agent display escalations are TEMPORARY — epoch deadline after which
# maybe_auto_revert_display() drops back to the floor (0.0 = no escalation).
_ESCALATION_HOLD_S = 120.0
_escalation_expires_at: float = 0.0

# Agent nickname registry — maps agent_id → short display name
_agent_nicknames: dict[str, str] = {}
_nickname_lock = threading.Lock()


def register_agent_nickname(agent_id: str, nickname: str) -> None:
    """Register a display nickname for an agent_id.

    Also registers the nickname with the active display's agent roster
    (for future agent-focus filtering in Stage 2).
    """
    with _nickname_lock:
        _agent_nicknames[agent_id] = nickname
    # Register with display roster (outside nickname lock to avoid deadlock)
    display = get_active_display()
    if display is not None:
        display.register_agent(nickname)


def get_agent_nickname(agent_id: str | None) -> str | None:
    """Look up display nickname for an agent_id. Returns None if unregistered."""
    if agent_id is None:
        return None
    with _nickname_lock:
        return _agent_nicknames.get(agent_id)


def clear_agent_nicknames() -> None:
    """Clear all registered nicknames (call between sim sessions)."""
    with _nickname_lock:
        _agent_nicknames.clear()


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


def agent_escalate_display(tier: DisplayTier, reason: str = "", *, hold_s: float = _ESCALATION_HOLD_S) -> bool:
    """Allow the agent to temporarily escalate display tier.

    The escalation is TEMPORARY: it auto-reverts to the user's floor after
    ``hold_s`` seconds, via :func:`maybe_auto_revert_display` (ticked by the
    agent loop). Time, not turns, is the unit — a loop iteration runs at
    2-30Hz, so a turn count is not available at the tick point.

    Returns True if escalation was applied, False if tier is below floor.
    """
    global _display_tier, _escalation_expires_at
    if tier < _display_floor:
        return False  # Agent can't suppress below user's floor
    changed = _display_tier != tier
    _display_tier = tier
    _escalation_expires_at = (time.time() + hold_s) if tier > _display_floor else 0.0
    # EVENT seam: surface the agent's display suggestion on the record stream
    # (kind="display" on the wire). The web UI keeps the floor semantics
    # client-side — the wire carries the suggestion, never enforces it.
    # Only on an actual change — repeat tool calls at the same tier would be
    # stream noise (review fold).
    if changed:
        sim_log(
            "DISPLAY",
            f"agent escalated display → {tier.name.lower()}",
            {"tier": tier.name.lower(), "reason": reason, "action": "escalate"},
        )
    return True


def maybe_auto_revert_display() -> bool:
    """Revert an EXPIRED agent escalation to the user's floor. Loop-ticked.

    The production producer of the ``display``/revert wire event: without it
    an escalation would stick forever (and ``DisplayModeTool``'s documented
    "auto-revert" would be a lie). Cheap: one float compare on the common
    no-escalation path. Returns True if a revert happened.
    """
    if _escalation_expires_at <= 0.0 or time.time() < _escalation_expires_at:
        return False
    revert_display_to_floor()
    return True


def revert_display_to_floor() -> None:
    """Revert display tier to user's --display setting."""
    global _display_tier, _escalation_expires_at
    reverted = _display_tier != _display_floor
    _display_tier = _display_floor
    _escalation_expires_at = 0.0
    if reverted:
        sim_log(
            "DISPLAY",
            f"display reverted to floor ({_display_floor.name.lower()})",
            {"tier": _display_floor.name.lower(), "reason": "", "action": "revert"},
        )


def reset_sim_display_state() -> None:
    """Reset all display/interactive globals to their defaults.

    Call after a sim ends to prevent state leaking into the next sim
    (e.g., in the menu loop where ``start_simulation_mode`` is called
    repeatedly in the same process).
    """
    global _display_tier, _display_floor, _interactive_mode, _escalation_expires_at
    _display_tier = DisplayTier.CLEAN
    _display_floor = DisplayTier.CLEAN
    _interactive_mode = InteractiveMode.AUTO
    _escalation_expires_at = 0.0
    _sensor_last_logged.clear()
    clear_agent_nicknames()


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
    "THOUGHT": "\033[32;1m",  # Bold green
    "DELIBERATION": "\033[32;1m",  # Bold green
    "EXEC": "\033[32m",  # Green
    "MOTOR": "\033[36;1m",  # Bold cyan
    "REACTION": "\033[35m",  # Magenta
    "SCN": "\033[37m",  # White
    "PIPELINE": "\033[37;2m",  # Dim white
    "RESULT": "\033[32;1m",  # Bold green
    "BLOCKED": "\033[31;1m",  # Bold red
    "CEREBELLUM": "\033[34;1m",  # Bold blue
    "ATL": "\033[35;1m",  # Bold magenta
    "SENSORY": "\033[36;1m",  # Bold cyan
    "SENSOR": "\033[36;1m",  # Bold cyan
    "BODY": "\033[36;2m",  # Dim cyan
    "ENRICHMENT": "\033[35;1m",  # Bold magenta
    "IMAGINATION": "\033[33;1m",  # Bold yellow
    "DISCOVERY": "\033[34;1m",  # Bold blue
    "GATE": "\033[37;1m",  # Bold white
    "SCENE": "\033[33;1m",  # Bold yellow
    "NPC": "\033[33;1m",  # Bold yellow
    "CHOICE": "\033[36;1m",  # Bold cyan
    "TURN": "\033[34;1m",  # Bold blue
    "ACTION": "\033[36;1m",  # Bold cyan
    "RESPONSE": "\033[32;1m",  # Bold green
    "INFO": "\033[37;2m",  # Dim white
    "USER": "\033[32;1m",  # Bold green
    # Substrate-learning headline events. Bold bright magenta — distinct
    # from any other channel. These are CLEAN-tier so users see them even
    # with bio annotations off.
    "LEARN": "\033[95;1m",
    # Drive/reflex BIO channels for embodiment runs.
    "DRIVE": "\033[33;2m",  # Dim yellow
    "REFLEX": "\033[31;1m",  # Bold red
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
    # Clean-tier: narrative events visible at ALL display levels including clean.
    # Also includes user-facing UI events (turn, info, action, response, summary)
    # so user input/output is never hidden.
    "SCENE": DisplayTier.CLEAN,
    "NPC": DisplayTier.CLEAN,
    "CHOICE": DisplayTier.CLEAN,
    "RESULT": DisplayTier.CLEAN,
    "BLOCKED": DisplayTier.CLEAN,
    "TURN": DisplayTier.CLEAN,
    "INFO": DisplayTier.CLEAN,
    "ACTION": DisplayTier.CLEAN,
    "RESPONSE": DisplayTier.CLEAN,
    "SUMMARY": DisplayTier.CLEAN,
    "USER": DisplayTier.CLEAN,
    # Substrate-learning headline events: reward_bias updates, tier
    # promotions, sleep consolidation, anticipatory pre-activation.
    # CLEAN-tier so they surface even when bio annotations are off —
    # these are the "agent just learned" moments.
    "LEARN": DisplayTier.CLEAN,
    # Bio-tier: biological subsystem annotations visible at --display bio+.
    "HIPPOCAMPUS": DisplayTier.BIO,
    "NAc": DisplayTier.BIO,
    "ATL": DisplayTier.BIO,
    "SCN": DisplayTier.BIO,
    "CEREBELLUM": DisplayTier.BIO,
    "SENSORY": DisplayTier.BIO,
    "SENSOR": DisplayTier.BIO,
    "BODY": DisplayTier.BIO,
    "BODY_STATE": DisplayTier.BIO,
    "FEAR": DisplayTier.BIO,
    "PAIN": DisplayTier.BIO,
    "REACTION": DisplayTier.BIO,
    "PERCEPT": DisplayTier.BIO,
    "THOUGHT": DisplayTier.BIO,
    "DELIBERATION": DisplayTier.BIO,
    "MOTOR": DisplayTier.BIO,
    "ENRICHMENT": DisplayTier.BIO,
    "IMAGINATION": DisplayTier.BIO,
    "DISCOVERY": DisplayTier.BIO,
    "GATE": DisplayTier.BIO,
    "DRIVE": DisplayTier.BIO,
    "REFLEX": DisplayTier.BIO,
    # Debug-tier: implementation details, pipeline traces.
    "EXEC": DisplayTier.DEBUG,
    "PIPELINE": DisplayTier.DEBUG,
    # Display-suggestion events (EVENT seam): persisted + streamed always
    # (sinks/JSONL fire before this gate); terminal only in debug — the
    # terminal already SHOWS the tier change itself, a log line would be noise.
    "DISPLAY": DisplayTier.DEBUG,
}

_sim_active = False
_sim_start: float = 0.0
_use_color = True
_log_file = None
_log_records: list[dict[str, Any]] = []
_debug_mode = False
_show_channels: set[str] | None = None  # None = show all, set = filter

# ─────────────────────────────────────────────────────────────────────────────
# Sim sinks — the EVENT seam's Maxim-side hook (reachy_app_maxim_seams.md
# § EVENT). Registered callables receive every structured sim_log record
# (same all-events rule as JSONL persistence: dispatch happens BEFORE the
# display-tier gate, regardless of terminal tier / --show filters; per-client
# filtering is the consumer's job). Sinks run ON THE PUBLISHING THREAD — they
# must be non-blocking (enqueue-and-return) and must not mutate the record.
# A raising sink is logged once (per sink) and never propagates into sim_log's
# caller. The console server's /ws bridge is the first consumer.
# ─────────────────────────────────────────────────────────────────────────────

_sim_sinks: list["Callable[[dict[str, Any]], None]"] = []
_sim_sinks_lock = threading.Lock()
_warned_sinks: set[int] = set()


def register_sim_sink(sink: "Callable[[dict[str, Any]], None]") -> None:
    """Register a callable receiving every sim_log record dict.

    The record shape is the persisted JSONL shape: ``t`` (sim-elapsed seconds),
    ``subsystem`` (UPPERCASE), ``message``, ``data``, plus ``agent_id`` /
    ``agent`` when present. Treat it as read-only.
    """
    with _sim_sinks_lock:
        if sink not in _sim_sinks:
            _sim_sinks.append(sink)


def unregister_sim_sink(sink: "Callable[[dict[str, Any]], None]") -> None:
    """Remove a previously-registered sink (no-op if absent).

    Matching is by ``==`` (a fresh bound-method object like ``hub.sink``
    compares equal to the stored one), so the warned-set entry must be
    discarded by the STORED element's id — discarding ``id(sink)`` would be a
    no-op for bound methods and could suppress a future sink's first warning
    on id reuse (cross-confirmed review finding).
    """
    with _sim_sinks_lock:
        for i, stored in enumerate(_sim_sinks):
            if stored == sink:
                _warned_sinks.discard(id(stored))
                del _sim_sinks[i]
                break


def _dispatch_to_sinks(record: dict[str, Any]) -> None:
    with _sim_sinks_lock:
        sinks = list(_sim_sinks)
    for sink in sinks:
        try:
            sink(record)
        except Exception:
            # A broken sink must not take down the agent loop — but don't
            # swallow silently either: warn ONCE per sink (sim_log is a hot
            # path; per-event warnings would flood).
            if id(sink) not in _warned_sinks:
                _warned_sinks.add(id(sink))
                logger.warning("sim sink %r raised; suppressing further warnings for it", sink, exc_info=True)


# Channel → subsystem mapping for --show flag
_CHANNEL_MAP: dict[str, set[str]] = {
    "bio": {
        "HIPPOCAMPUS",
        "NAc",
        "SCN",
        "ATL",
        "FEAR",
        "PAIN",
        "REACTION",
        "SENSORY",
        "SENSOR",
        "BODY_STATE",
        "CEREBELLUM",
        "THOUGHT",
        "DELIBERATION",
        "BODY",
        "PERCEPT",
        "ENRICHMENT",
        "IMAGINATION",
        "DISCOVERY",
        "GATE",
    },
    "bio-only": {
        "HIPPOCAMPUS",
        "NAc",
        "SCN",
        "ATL",
        "FEAR",
        "PAIN",
        "REACTION",
        "SENSORY",
        "THOUGHT",
        "DELIBERATION",
        "CEREBELLUM",
        "ENRICHMENT",
        "GATE",
    },
    "exec": {"EXEC", "MOTOR", "PIPELINE"},
    "sim": {"SCENE", "NPC", "CHOICE", "RESULT", "BLOCKED"},
    "memory": {"HIPPOCAMPUS", "NAc", "SCN", "ATL"},
    "safety": {"FEAR", "PAIN", "BLOCKED"},
    "sem": {"SENSOR", "IMAGINATION", "DISCOVERY", "CEREBELLUM", "BODY", "BODY_STATE"},
    "enrichment": {"ENRICHMENT", "GATE"},
}


def expand_channels(channels: "list[str] | tuple[str, ...]") -> set[str]:
    """Expand channel names to subsystem names via ``_CHANNEL_MAP``.

    Unknown names are treated as raw subsystem names (uppercased). NOTE: the
    returned set is NOT uniformly uppercase — canonical subsystem spellings
    like ``"NAc"`` come through mixed-case from the map — so consumers MUST
    compare case-insensitively (uppercase both sides; the terminal gate and
    the ``/ws`` filter both do). Shared by the terminal's ``--show`` filter
    and the console ``/ws`` SubscribeFrame (EVENT seam) so the two filter
    surfaces cannot drift.
    """
    allowed: set[str] = set()
    for ch in channels:
        c = ch.strip().lower()
        if c in _CHANNEL_MAP:
            allowed |= _CHANNEL_MAP[c]
        else:
            allowed.add(ch.strip().upper())
    return allowed


def subsystem_wire_tier(subsystem: str) -> str:
    """Wire ``tier`` for a subsystem (EVENT seam): ``clean``/``bio``/``debug``.

    Unknown subsystems → ``"bio"`` — the same opt-out-not-opt-in default the
    terminal gate uses (``_SUBSYSTEM_TIERS.get(..., BIO)``).
    """
    return _SUBSYSTEM_TIERS.get(subsystem, DisplayTier.BIO).name.lower()


def set_show_channels(channels: str | None) -> None:
    """Set which subsystem channels to show in terminal output.

    Args:
        channels: Comma-separated channel names (``"bio"``, ``"bio-only"``,
            ``"exec"``, ``"sim"``, ``"memory"``, ``"safety"``, ``"all"``).
            ``None`` or ``"all"`` shows everything.
            ``"bio-only"`` shows only biological learning mechanisms
            (hippo, NAc, ATL, SCN, fear, pain, reaction, sensory, cerebellum).

    Examples::

        set_show_channels("bio")          # Bio-system + percept events
        set_show_channels("bio-only")     # Only learning mechanisms
        set_show_channels("bio,exec")     # Bio + execution
        set_show_channels("all")          # Everything
        set_show_channels(None)           # Everything (default)
    """
    global _show_channels
    if channels is None or channels.strip().lower() == "all":
        _show_channels = None
        return

    # Stored uppercased; the sim_log gate compares subsystem.upper() — makes
    # `--show nac` work (pre-fix, raw-name "nac"→"NAC" never matched the
    # mixed-case canonical "NAc"; cross-confirmed review finding).
    allowed = {s.upper() for s in expand_channels(channels.split(","))}
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

    Repetitive messages (same text within a short window) are suppressed
    to prevent log floods from sources like CostTracker.
    """

    # Messages containing these substrings are shown at most once per
    # dedup window to prevent flooding the display.
    _DEDUP_SUBSTRINGS = ("CostTracker.record failed", "Missing pricing for model")
    _DEDUP_WINDOW_S = 30.0

    def __init__(self, display: Any) -> None:
        super().__init__(level=logging.WARNING)
        self._display = display
        self._seen: dict[str, float] = {}  # message_key → last_shown_time

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)

            # Route noisy repeated messages to the persistent warnings
            # panel instead of the scrolling log.
            for substr in self._DEDUP_SUBSTRINGS:
                if substr in msg:
                    if hasattr(self._display, "warn"):
                        # Show a concise version in the fixed warnings panel
                        self._display.warn(record.getMessage())
                    return

            if record.levelno >= logging.ERROR:
                self._display.log("blocked", msg)
            else:
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
    agent_id: str | None = None,
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
        agent_id: Optional agent identifier — resolved to a nickname via
            the registry for display. Falls back to the implicit
            ``_current_agent_id`` contextvar (set by ``sim_agent_context``).
            Persisted as-is in JSONL records.
        _force_debug: If True, only show on terminal in debug mode
    """
    if not _sim_active:
        return

    # Resolve agent_id: explicit parameter > contextvar > None
    if agent_id is None:
        agent_id = _current_agent_id.get(None)

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

    # Resolve agent nickname for display
    nickname = get_agent_nickname(agent_id)

    # Always persist structured record (JSONL log + in-memory)
    record: dict[str, Any] = {
        "t": round(elapsed, 3),
        "subsystem": subsystem,
        "message": message,
        "data": data or {},
    }
    if agent_id is not None:
        record["agent_id"] = agent_id
    if nickname is not None:
        record["agent"] = nickname
    _log_records.append(record)

    if _log_file is not None:
        import json

        _log_file.write(json.dumps(record) + "\n")
        _log_file.flush()

    # EVENT seam: fan out to registered sinks (console /ws bridge). Same rule
    # as JSONL persistence — before the display-tier gate, all events.
    _dispatch_to_sinks(record)

    # Bridge into the unified MAXIM_LOG_FILE stream. The sim-session JSONL
    # above stays the canonical session artifact; this second emission lets
    # users who set MAXIM_LOG_FILE see bio-system events alongside HTTP /
    # peer-backend / role events in one post-mortem trail. StructuredFormatter
    # picks up `event` + `data` from the `extra=` dict; we route through a
    # dedicated logger with propagate=False so stdout handlers never see
    # these emissions (they have their own terminal path below).
    _ensure_sim_bridge_handlers()
    _sim_bridge_logger.info(
        message,
        extra={
            "event": f"sim_{subsystem.lower()}",
            "data": {
                "subsystem": subsystem,
                "message": message,
                "elapsed_s": round(elapsed, 3),
                "agent_id": agent_id,
                "agent": nickname,
                **(data or {}),
            },
        },
    )

    # Display-tier gate. Each subsystem declares its minimum visible tier via
    # _SUBSYSTEM_TIERS (unknown subsystems default to BIO). _force_debug
    # escalates an event to DEBUG tier regardless of its subsystem's baseline.
    # The legacy --debug runtime flag (_debug_mode) bypasses the tier gate
    # entirely for backward compatibility with verbose-mode CLI users.
    if not _debug_mode:
        min_tier = DisplayTier.DEBUG if _force_debug else _SUBSYSTEM_TIERS.get(subsystem, DisplayTier.BIO)
        if _display_tier < min_tier:
            return

    # Channel filter — skip subsystems not in the active show set (stored
    # uppercased; compare case-insensitively so mixed-case canonical
    # spellings like "NAc" are reachable).
    if _show_channels is not None and subsystem.upper() not in _show_channels:
        return

    # Build the display line
    if _use_color:
        color = _COLORS.get(subsystem, "")
        label = f"{color}[{subsystem:12s}]{_RESET}"
    else:
        label = f"[{subsystem:12s}]"

    agent_tag = f" [{nickname}]" if nickname else ""
    line = f"  {timestamp} {label}{agent_tag}{(' ' + thread_tag) if thread_tag else ''} {message}"

    if data:
        # Format key data inline (compact)
        details = ", ".join(f"{k}={v}" for k, v in data.items() if v is not None)
        if details:
            line += f"  ({details})"

    # Route terminal output through active display or direct print
    display = get_active_display()
    if display is not None:
        # Strip ANSI codes — the display applies its own styling
        agent_prefix = f"[{nickname}] " if nickname else ""
        plain_msg = f"{agent_prefix}{timestamp} [{subsystem}] {message}"
        if data:
            details = ", ".join(f"{k}={v}" for k, v in data.items() if v is not None)
            if details:
                plain_msg += f"  ({details})"
        display.log(subsystem.lower(), plain_msg, agent=nickname)
    else:
        print(line, flush=True)


def sim_percept(source: str, summary: str, *, agent_id: str | None = None, **kwargs: Any) -> None:
    """Log an incoming percept."""
    sim_log("PERCEPT", f"👁️ [{source}] {summary}", kwargs if kwargs else None, agent_id=agent_id)


def sim_memory(action: str, *, agent_id: str | None = None, **kwargs: Any) -> None:
    """Log a hippocampus/memory event."""
    sim_log("HIPPOCAMPUS", f"💾 {action}", kwargs if kwargs else None, agent_id=agent_id)


def sim_debug(subsystem: str, action: str, *, agent_id: str | None = None, **kwargs: Any) -> None:
    """Log an event that only appears in debug mode (--debug).

    Always persisted to JSONL; terminal output suppressed unless debug=True.
    """
    sim_log(subsystem, action, kwargs if kwargs else None, agent_id=agent_id, _force_debug=True)


def sim_reaction(kind: str, intensity: float, source: str, *, agent_id: str | None = None, **kwargs: Any) -> None:
    """Log a Reaction from the ReactionBus.

    Generalizes sim_pain for all reaction kinds. Replaces the sim_pain
    call lost when route_pain_percept was deleted in Phase 2a.
    """
    _REACTION_ICONS = {
        "pain": "🔴",
        "fear": "😨",
        "hunger": "🍽️",
        "surprise": "❗",
        "fatigue": "😴",
        "satiation": "✅",
    }
    icon = _REACTION_ICONS.get(kind.lower(), "⚡")
    sim_log(
        "REACTION",
        f"{icon} {kind} (intensity={intensity:.2f}) from {source}",
        kwargs if kwargs else None,
        agent_id=agent_id,
    )


def sim_pain(pain_type: str, intensity: float, *, agent_id: str | None = None, **kwargs: Any) -> None:
    """Log a pain signal."""
    sim_log("PAIN", f"🔴 {pain_type} (intensity={intensity:.2f})", kwargs if kwargs else None, agent_id=agent_id)


def sim_fear(tool: str, allowed: bool, reason: str = "", *, agent_id: str | None = None) -> None:
    """Log a FearAgent review."""
    if allowed:
        sim_log("FEAR", f"🛡️ ALLOWED: {tool}", agent_id=agent_id)
    else:
        sim_log("BLOCKED", f"🚫 BLOCKED: {tool} — {reason}", agent_id=agent_id)


def sim_action(
    tool: str,
    success: bool,
    summary: str = "",
    *,
    agent_id: str | None = None,
    entity_class: str | None = None,
    **kwargs: Any,
) -> None:
    """Log a tool execution.

    Stage 0b (release_0_9_1.md): accepts ``entity_class`` for
    exposure-count normalization in Roy-3 analysis. Falls into the
    structured ``data`` dict alongside any other kwargs. None → field
    omitted from the persisted record. The plain ``sim_action(tool,
    success)`` call shape from earlier callers continues to work
    unchanged (entity_class defaults to None and is dropped before
    the dict is passed to sim_log).
    """
    icon = "⚔️" if success else "❌"
    status = "OK" if success else "FAIL"
    data: dict[str, Any] = dict(kwargs)
    if entity_class is not None:
        data["entity_class"] = entity_class
    sim_log(
        "MOTOR",
        f"{icon} [{status}] {tool}: {summary}" if summary else f"{icon} [{status}] {tool}",
        data if data else None,
        agent_id=agent_id,
    )


def sim_result(scenario_name: str, passed: bool, met: int, failed: int) -> None:
    """Log final scenario result."""
    status = "PASS" if passed else "FAIL"
    subsystem = "RESULT" if passed else "BLOCKED"
    icon = "✅" if passed else "❌"
    sim_log(subsystem, f"{icon} {status}: {scenario_name} ({met} passed, {failed} failed)")


def sim_nac(event: str, outcome: str, rpe: float, confidence: float, *, agent_id: str | None = None) -> None:
    """Log a NAc causal learning observation."""
    icon = "📈" if rpe >= 0 else "📉"
    sim_log(
        "NAc", f"🧠 Causal link: {event} → {outcome} (RPE={rpe:+.2f}, conf={confidence:.2f}) {icon}", agent_id=agent_id
    )


def sim_scn(memory_id: str, phase: str, significance: float, *, agent_id: str | None = None) -> None:
    """Log an SCN temporal bin registration."""
    sim_log("SCN", f"🕐 Registered {memory_id[:8]} in {phase} (significance={significance:.2f})", agent_id=agent_id)


def sim_cerebellum(
    entity: str, affordance: str, confidence: float, error: float | None = None, *, agent_id: str | None = None
) -> None:
    """Log a Cerebellum forward model observation."""
    if error is not None:
        icon = "🎯" if error < 0.1 else "❌"
        sim_log(
            "CEREBELLUM",
            f"{icon} Observed {entity}.{affordance} (conf={confidence:.2f}, pred_error={error:.3f})",
            agent_id=agent_id,
        )
    else:
        sim_log("CEREBELLUM", f"📊 New model: {entity}.{affordance} (conf={confidence:.2f})", agent_id=agent_id)


def sim_sensory(
    modality: str, entity: str, acuity: float, dropped: bool = False, *, agent_id: str | None = None
) -> None:
    """Log a SensoryGate modulation event."""
    if dropped:
        sim_log("SENSORY", f"📡 Dropped {modality} percept from {entity} (acuity={acuity:.2f})", agent_id=agent_id)
    else:
        sim_log("SENSORY", f"📡 Modulated {modality} from {entity} (acuity={acuity:.2f})", agent_id=agent_id)


def sim_body_state(entity_count: int, active_failures: int, *, agent_id: str | None = None) -> None:
    """Log body state injection into prompt."""
    icon = "🔥" if active_failures > 0 else "🫀"
    sim_log("BODY", f"{icon} State: {entity_count} entities, {active_failures} active failures", agent_id=agent_id)


# ─────────────────────────────────────────────────────────────────────────────
# LEARN — substrate-learning headline events (CLEAN tier).
# Bold bright magenta + ▲ icon so the "agent just learned" moments stand out
# even when bio annotations are off.  Producers: NAc.credit_node, hippocampus
# tier promotion, sleep consolidation, TemporalCreditDistributor phase
# fallback, SCN anticipatory pre-activation.
# ─────────────────────────────────────────────────────────────────────────────


def sim_learn(
    headline: str,
    detail: str = "",
    *,
    source: str = "",
    agent_id: str | None = None,
    **kwargs: Any,
) -> None:
    """Log a substrate-learning headline event.

    These are the moments that make the bio-inspired architecture feel alive:
    a reward_bias just changed, a memory just promoted to long-term, a sleep
    cycle just consolidated, a phase-similarity fallback just credited a node
    whose eligibility trace had decayed, an oscillator prediction just primed
    eligibility for an imminent event.

    Routed at CLEAN display tier so users see them by default.
    """
    msg = f"▲ {headline}"
    if detail:
        msg += f" — {detail}"
    if source:
        msg += f" [{source}]"
    data = {"headline": headline, "detail": detail, "source": source, **kwargs}
    sim_log("LEARN", msg, data, agent_id=agent_id)


def sim_drive(
    name: str,
    value: float,
    set_point: float | None = None,
    *,
    pain: float = 0.0,
    agent_id: str | None = None,
) -> None:
    """Log a homeostatic / entropic drive state sample.

    Args:
        name: Drive name (e.g. "hunger", "thirst", "thermal").
        value: Current drive value.
        set_point: Optional homeostatic target (homeostatic drives only).
        pain: Pain magnitude derived from the drive, if any.
    """
    parts = [f"{name}={value:.2f}"]
    if set_point is not None:
        parts.append(f"set={set_point:.2f}")
    if pain > 0:
        parts.append(f"pain={pain:.2f}")
    icon = "🔥" if pain > 0.3 else "⚠️" if pain > 0 else "🫀"
    sim_log(
        "DRIVE",
        f"{icon} drive {name}: " + ", ".join(parts),
        {"name": name, "value": value, "set_point": set_point, "pain": pain},
        agent_id=agent_id,
    )


def sim_reflex(
    reflex_name: str,
    tool: str,
    intensity: float,
    *,
    raw_intensity: float | None = None,
    agent_id: str | None = None,
) -> None:
    """Log an innate reflex firing.

    Reflexes are pre-deliberative responses (thermal withdrawal, pain wince,
    startle).  Surfacing the firing at BIO tier makes embodiment runs
    legible — without this users see drive/sensor changes but no signal that
    the reflex layer is mediating them.
    """
    msg = f"⚡ reflex {reflex_name} → {tool} (intensity={intensity:.2f})"
    if raw_intensity is not None and abs(raw_intensity - intensity) > 0.05:
        msg += f" raw={raw_intensity:.2f}"
    sim_log(
        "REFLEX",
        msg,
        {
            "reflex": reflex_name,
            "tool": tool,
            "intensity": intensity,
            "raw_intensity": raw_intensity,
        },
        agent_id=agent_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Track 1: New bio-system event loggers — surface previously-silent events
# ─────────────────────────────────────────────────────────────────────────────


def sim_nac_learn(
    event: str,
    outcome: str,
    confidence: float,
    rpe: float,
    *,
    link_type: str = "updated",
    agent_id: str | None = None,
) -> None:
    """Log a NAc causal link formation or update.

    Called when ``record_outcome`` / ``record_outcome_full`` / ``observe``
    creates or modifies a CausalLink.  Distinct from ``sim_nac`` which is
    only called from ToolPainBridge today — this captures ALL NAc learning.
    """
    icon = "📈" if rpe >= 0 else "📉"
    sim_log(
        "NAc",
        f"🧠 {link_type}: {event} → {outcome} (RPE={rpe:+.2f}, conf={confidence:.2f}) {icon}",
        {"event": event, "outcome": outcome, "confidence": confidence, "rpe": rpe, "link_type": link_type},
        agent_id=agent_id,
    )


def sim_nac_predict(
    event: str,
    outcomes: list[tuple[str, float]],
    *,
    agent_id: str | None = None,
) -> None:
    """Log a NAc causal prediction query.

    ``outcomes`` is a list of (outcome_signature, confidence) tuples.
    """
    if not outcomes:
        sim_log("NAc", f"🔮 predict({event}): no known outcomes", agent_id=agent_id)
        return
    top = outcomes[0]
    others = len(outcomes) - 1
    more = f" +{others} more" if others > 0 else ""
    sim_log(
        "NAc",
        f"🔮 predict({event}): {top[0]} (conf={top[1]:.2f}){more}",
        {"event": event, "outcomes": outcomes[:5]},
        agent_id=agent_id,
    )


def sim_cerebellum_train(
    entity: str,
    affordance: str,
    pred_error: float,
    confidence: float,
    *,
    agent_id: str | None = None,
) -> None:
    """Log a Cerebellum forward model training update."""
    icon = "🎯" if pred_error < 0.1 else "📊"
    sim_log(
        "CEREBELLUM",
        f"{icon} train: {entity}.{affordance} (error={pred_error:.3f}, conf={confidence:.2f})",
        {"entity": entity, "affordance": affordance, "pred_error": pred_error, "confidence": confidence},
        agent_id=agent_id,
    )


def sim_cerebellum_predict(
    entity: str,
    affordance: str,
    predicted: dict[str, float] | None,
    confidence: float,
    *,
    agent_id: str | None = None,
) -> None:
    """Log a Cerebellum forward model prediction (not just new model creation)."""
    if predicted:
        values = ", ".join(f"{k}={v:.2f}" for k, v in list(predicted.items())[:3])
        sim_log(
            "CEREBELLUM",
            f"🔮 predict: {entity}.{affordance} → [{values}] (conf={confidence:.2f})",
            {"entity": entity, "affordance": affordance, "confidence": confidence},
            agent_id=agent_id,
        )
    else:
        sim_log(
            "CEREBELLUM",
            f"🔮 predict: {entity}.{affordance} — no confident model",
            agent_id=agent_id,
        )


def sim_hippocampus_episode(
    action: str,
    episode_id: str = "",
    node_count: int = 0,
    valence: float = 0.0,
    *,
    agent_id: str | None = None,
) -> None:
    """Log a hippocampus episode lifecycle event (open/close/boundary)."""
    _EPISODE_ICONS = {"open": "📖", "close": "📕", "boundary": "📑", "bind": "🔗"}
    icon = _EPISODE_ICONS.get(action, "📖")
    parts = [f"{icon} episode {action}"]
    if episode_id:
        parts.append(f"E-{episode_id[:8]}")
    if node_count:
        parts.append(f"{node_count} nodes")
    if valence != 0.0:
        parts.append(f"valence={valence:+.2f}")
    sim_log(
        "HIPPOCAMPUS",
        " ".join(parts),
        {"action": action, "episode_id": episode_id, "node_count": node_count, "valence": valence},
        agent_id=agent_id,
    )


def sim_ec_pattern(
    action: str,
    concept: str,
    similarity: float,
    threshold: float,
    *,
    agent_id: str | None = None,
) -> None:
    """Log an EC pattern completion or separation event."""
    icon = "🔗" if action == "complete" else "✂️"
    sim_log(
        "ATL",
        f"{icon} pattern {action}: '{concept}' (sim={similarity:.2f}, thresh={threshold:.2f})",
        {"action": action, "concept": concept, "similarity": similarity, "threshold": threshold},
        agent_id=agent_id,
    )


def sim_imagination(
    phase: str,
    phrase: str,
    result: str = "",
    *,
    agent_id: str | None = None,
) -> None:
    """Log an imagination pipeline event (extract/cache/index/design/register)."""
    _IMAGINATION_ICONS = {
        "extract": "💡",
        "cache_hit": "📋",
        "index_hit": "🗺️",
        "design": "✨",
        "register": "🏗️",
        "gate_rejected": "⛔",
    }
    icon = _IMAGINATION_ICONS.get(phase, "💡")
    msg = f"{icon} imagination {phase}: '{phrase}'"
    if result:
        msg += f" → {result}"
    sim_log(
        "IMAGINATION",
        msg,
        {"phase": phase, "phrase": phrase, "result": result},
        agent_id=agent_id,
    )


def sim_discovery(
    query: str,
    matched: int,
    activated: int,
    source: str = "",
    *,
    agent_id: str | None = None,
) -> None:
    """Log a tool discovery event."""
    sim_log(
        "DISCOVERY",
        f"🔎 discover({query}): {matched} matched, {activated} activated" + (f" via {source}" if source else ""),
        {"query": query, "matched": matched, "activated": activated, "source": source},
        agent_id=agent_id,
    )


def sim_enrichment(
    system: str,
    summary: str,
    *,
    agent_id: str | None = None,
) -> None:
    """Log what a specific bio-system contributed during enrichment.

    Called from BioEnrichmentPipeline to surface WHAT each system contributed
    (not just that it contributed).
    """
    _ENRICHMENT_ICONS = {
        "hippocampus": "💾",
        "nac": "🧠",
        "atl": "🏷️",
        "cerebellum": "🎯",
        "component_index": "🗺️",
        "working_memory": "📝",
        "fear": "😨",
    }
    icon = _ENRICHMENT_ICONS.get(system.lower(), "🔬")
    sim_log(
        "ENRICHMENT",
        f"{icon} [{system}] {summary}",
        {"system": system, "summary": summary},
        agent_id=agent_id,
    )


def sim_gate(
    gate_name: str,
    passed: bool,
    score: float,
    threshold: float,
    reason: str = "",
    *,
    agent_id: str | None = None,
) -> None:
    """Log a gating decision (ThoughtGate, AdaptiveThreshold, novelty)."""
    icon = "✅" if passed else "⛔"
    action = "passed" if passed else "rejected"
    msg = f"⚖️ {gate_name} {icon} {action} (score={score:.2f}, threshold={threshold:.2f})"
    if reason:
        msg += f" — {reason}"
    sim_log(
        "GATE",
        msg,
        {"gate": gate_name, "passed": passed, "score": score, "threshold": threshold, "reason": reason},
        agent_id=agent_id,
    )


# Per-(entity, sensor) sampling gate: log at most once per _SENSOR_SAMPLE_S
# unless drift exceeds the significant threshold.  Prevents flooding in long
# sessions where evaluate_failures() fires every tick (10-150 calls/sec/entity).
_SENSOR_SAMPLE_S = 2.0
_SENSOR_SIGNIFICANT_DRIFT_PCT = 15.0
_sensor_last_logged: dict[tuple[str, str], float] = {}


def sim_sensor(
    entity: str,
    sensor: str,
    value: float,
    baseline: float | None = None,
    *,
    agent_id: str | None = None,
) -> None:
    """Log a SEM entity sensor reading.

    Sampled: at most one log per entity+sensor per ``_SENSOR_SAMPLE_S``
    unless drift from baseline exceeds ``_SENSOR_SIGNIFICANT_DRIFT_PCT``.
    """
    drift_pct = 0.0
    if baseline is not None and baseline != 0:
        drift_pct = ((value - baseline) / abs(baseline)) * 100

    # Sampling gate: skip unless significant drift or enough time elapsed
    key = (entity, sensor)
    now = time.time()
    if abs(drift_pct) < _SENSOR_SIGNIFICANT_DRIFT_PCT:
        last = _sensor_last_logged.get(key, 0.0)
        if now - last < _SENSOR_SAMPLE_S:
            return
    _sensor_last_logged[key] = now

    if baseline is not None and baseline != 0:
        icon = "🔥" if abs(drift_pct) > 30 else "⚠️" if abs(drift_pct) > 15 else "📡"
        sim_log(
            "SENSOR",
            f"{icon} {entity}.{sensor}={value:.2f} (baseline={baseline:.2f}, drift={drift_pct:+.0f}%)",
            {"entity": entity, "sensor": sensor, "value": value, "baseline": baseline, "drift_pct": drift_pct},
            agent_id=agent_id,
        )
    else:
        sim_log(
            "SENSOR",
            f"📡 {entity}.{sensor}={value:.2f}",
            {"entity": entity, "sensor": sensor, "value": value},
            agent_id=agent_id,
        )


def sim_thought(
    passed: bool,
    reason: str,
    score: float = 0.0,
    threshold: float = 0.0,
    *,
    agent_id: str | None = None,
) -> None:
    """Log a ThoughtGate deliberation decision.

    Shows at BIO display tier.  Passed decisions show score/threshold;
    rejected decisions show the rejection reason.
    """
    if passed:
        sim_log(
            "THOUGHT",
            f"💭 deliberation approved (score={score:.2f} >= {threshold:.2f})",
            {"score": score, "threshold": threshold, "reason": reason},
            agent_id=agent_id,
        )
    else:
        sim_log(
            "THOUGHT",
            f"💭 deliberation skipped — {reason}",
            {"score": score, "threshold": threshold, "reason": reason},
            agent_id=agent_id,
        )


def sim_pre_deliberation(
    gate_passed: bool,
    score: float,
    threshold: float,
    enrichment_sections: int,
    *,
    agent_id: str | None = None,
) -> None:
    """Log a Layer 1 pre-deliberation decision + enrichment summary.

    Shows at BIO display tier.  When the gate passes, shows how many
    bio-system sections contributed enrichment.  When rejected, shows
    the score vs threshold.
    """
    if gate_passed:
        sim_log(
            "THOUGHT",
            f"⚖️ pre-deliberation: gate passed (score={score:.2f} >= {threshold:.2f}), "
            f"{enrichment_sections} enrichment section(s)",
            {
                "gate_passed": True,
                "score": score,
                "threshold": threshold,
                "enrichment_sections": enrichment_sections,
            },
            agent_id=agent_id,
        )
    else:
        sim_log(
            "THOUGHT",
            f"⚖️ pre-deliberation: gate rejected (score={score:.2f} < {threshold:.2f})",
            {
                "gate_passed": False,
                "score": score,
                "threshold": threshold,
                "enrichment_sections": 0,
            },
            agent_id=agent_id,
        )


def sim_contemplation(
    gate_passed: bool,
    refined: bool,
    score: float,
    *,
    agent_id: str | None = None,
) -> None:
    """Log a Layer 3 post-proposal contemplation decision.

    Shows at BIO display tier.  Distinguishes gate-rejected,
    contemplated-but-kept, and contemplated-and-refined outcomes.
    """
    if not gate_passed:
        sim_log(
            "DELIBERATION",
            f"💭 contemplation skipped (score={score:.2f})",
            {"gate_passed": False, "refined": False, "score": score},
            agent_id=agent_id,
        )
    elif refined:
        sim_log(
            "DELIBERATION",
            f"✨ contemplation refined plan (score={score:.2f})",
            {"gate_passed": True, "refined": True, "score": score},
            agent_id=agent_id,
        )
    else:
        sim_log(
            "DELIBERATION",
            f"💭 contemplation kept original (score={score:.2f})",
            {"gate_passed": True, "refined": False, "score": score},
            agent_id=agent_id,
        )


def sim_deliberation_update(
    reasoning: str,
    cycle: int,
    max_cycles: int,
    enrichment_tags: list[str] | None = None,
    *,
    agent_id: str | None = None,
    salience: float | None = None,
    enrichment_details: dict[str, str] | None = None,
) -> None:
    """Push deliberation reasoning text to the thinking panel.

    Called at each deliberation cycle boundary to update the live
    thinking panel with the LLM's full reasoning text and enrichment
    metadata.  Also logs a DELIBERATION entry for JSONL persistence.

    Args:
        enrichment_details: Map of bio-system name → summary of what it
            contributed (e.g. ``{"hippocampus": "2 episodes: ..."}``.
    """
    if agent_id is None:
        agent_id = _current_agent_id.get(None)
    nickname = get_agent_nickname(agent_id)

    display = get_active_display()
    if display is not None:
        display.set_thinking(
            reasoning=reasoning,
            cycle=cycle,
            max_cycles=max_cycles,
            enrichment_tags=enrichment_tags,
            agent=nickname,
            salience=salience,
            enrichment_details=enrichment_details,
        )

    # EVENT seam (work item 7): the thinking-panel content previously reached
    # ONLY display.set_thinking — no record, so JSONL and the /ws stream missed
    # all deliberation. Emit the record the docstring always claimed.
    # _force_debug keeps the terminal path unchanged at CLEAN/BIO (the panel
    # renders the reasoning; a scrolling log line would duplicate it); in
    # debug modes the reasoning ALSO appears as a log line — opt-in verbosity.
    # JSONL + sinks always fire regardless. `text` is the reasoning key the
    # web ThinkingPanel reads.
    sim_log(
        "DELIBERATION",
        f"deliberation cycle {cycle}/{max_cycles}",
        {
            "text": reasoning,
            "cycle": cycle,
            "max_cycles": max_cycles,
            "enrichment_tags": list(enrichment_tags) if enrichment_tags else [],
            "salience": salience,
            "enrichment_details": enrichment_details or {},
            "completed": False,
        },
        agent_id=agent_id,
        _force_debug=True,
    )


def sim_deliberation_end(
    cycle: int,
    max_cycles: int,
    summary: str = "",
    *,
    agent_id: str | None = None,
) -> None:
    """Mark deliberation as completed in the thinking panel.

    Called when deliberation finishes (ready_to_act, convergence, or
    max cycles).  Sets the thinking panel to a completed summary view.
    """
    if agent_id is None:
        agent_id = _current_agent_id.get(None)
    nickname = get_agent_nickname(agent_id)

    display = get_active_display()
    if display is not None:
        display.set_thinking(
            reasoning=summary or f"Deliberation complete after {cycle} cycle(s)",
            cycle=cycle,
            max_cycles=max_cycles,
            agent=nickname,
            completed=True,
        )

    # EVENT seam (work item 7): mirror the completion onto the record stream.
    sim_log(
        "DELIBERATION",
        f"deliberation complete after {cycle} cycle(s)",
        {
            "text": summary or f"Deliberation complete after {cycle} cycle(s)",
            "cycle": cycle,
            "max_cycles": max_cycles,
            "completed": True,
        },
        agent_id=agent_id,
        _force_debug=True,
    )
