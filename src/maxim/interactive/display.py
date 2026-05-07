"""Rich-based live terminal display with structured panels.

Provides a split-panel terminal UI with scrolling agent log, status bar,
and input area.  Extensible via :class:`DisplayExtension` for adding
custom panels (character sheet, inventory, encounter info, etc.).

Graceful degradation: if ``rich`` is not installed or stdout is not a
TTY, the display silently does nothing (all methods are no-ops).

Example::

    display = create_display()
    if display:
        display.start()
        display.log("hippo", "Captured: sword encounter")
        display.set_status(mode="simulation", turn="3")
        display.stop()
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

# Check for rich availability at module level (no side effects)
try:
    from rich.console import Console  # noqa: F401
    from rich.live import Live  # noqa: F401
    from rich.layout import Layout  # noqa: F401
    from rich.panel import Panel  # noqa: F401
    from rich.text import Text  # noqa: F401

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Display Extension Protocol
# ---------------------------------------------------------------------------


class DisplayExtension(ABC):
    """Protocol for adding custom panels to the display.

    Implement this to add DM campaign panels (character sheet, inventory),
    research panels (experiment progress), robot panels (joint status), etc.
    """

    @abstractmethod
    def panel_name(self) -> str:
        """Name shown in panel header."""

    @abstractmethod
    def render(self) -> Any:
        """Return a rich renderable for this panel."""

    def key_bindings(self) -> dict[str, Callable]:
        """Optional key bindings this extension handles."""
        return {}


# ---------------------------------------------------------------------------
# Log entry + thinking state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _LogEntry:
    """Structured log line with agent identity for filtering."""

    agent: str | None  # nickname or None (system)
    markup: str  # full Rich markup string


@dataclass
class _ThinkingState:
    """State for the thinking panel — tracks active or completed deliberation.

    ``history`` accumulates (cycle, reasoning, enrichment_tags) tuples
    across deliberation cycles so the user sees the full chain building,
    not just the latest cycle's text.
    """

    reasoning: str  # Latest cycle's reasoning text (or summary if completed)
    cycle: int  # Current cycle number
    max_cycles: int  # Hard cap
    enrichment_tags: list[str] = field(default_factory=list)
    enrichment_details: dict[str, str] = field(default_factory=dict)  # system → summary
    started_at: float = 0.0  # time.monotonic() for elapsed display
    agent: str | None = None  # Agent nickname
    completed: bool = False  # True = show summary instead of live text
    # Each tuple: (cycle, reasoning, enrichment_tags, salience_or_None)
    history: list[tuple[int, str, list[str], float | None]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MaximDisplay
# ---------------------------------------------------------------------------


class MaximDisplay:
    """Rich-based live terminal display with structured panels.

    Layout::

        +- Status -----------------------------------------------+
        | Mode: simulation  Goal: test memory  Turn: 7           |
        +- Agent Log --------------------------------------------+
        | [hippo] Captured: "merchant offered healing potion"     |
        | [nac]   Link: threaten -> hostility (conf 0.82)        |
        | [exec]  Tool: memory_recall("healing") -> 2 hits       |
        +--------------------------------------------------------+
        | > What do you do? [fight / negotiate / flee]            |
        | > _                                                     |
        +--------------------------------------------------------+

    Thread-safe: all mutations acquire ``_lock``. The orchestrator runs
    3+ concurrent threads (sim.aut, sim.stdin, sim.stall + main) that
    all emit ``sim_log()`` events routed through ``log()``.
    """

    # Bio subsystems get dim styling; scene/dialogue stays bright.
    # PFC subsystems (thought, deliberation) get their own styling —
    # NOT dimmed, since deliberation events are user-relevant.
    _BIO_SUBSYSTEMS = frozenset(
        {
            "hippo",
            "hippocampus",
            "nac",
            "fear",
            "pain",
            "exec",
            "cerebellum",
            "atl",
            "sensory",
            "sensor",
            "body",
            "percept",
            "reaction",
            "scn",
            "ec",
            "pipeline",
            "enrichment",
            "imagination",
            "discovery",
            "gate",
        }
    )

    def __init__(self, title: str = "Maxim", max_log_lines: int = 500) -> None:
        self._title = title
        self._log_lines: deque[_LogEntry] = deque(maxlen=max_log_lines)
        self._status: dict[str, str] = {}
        self._prompt_text: str = ""
        self._extensions: list[DisplayExtension] = []
        self._live: Any = None  # rich.live.Live when active
        self._console: Any = None
        self._lock = threading.RLock()
        self._prompt_urgent: bool = False  # Gold border when agent is asking a question
        self._scene_title: str = "Constructing Simulation"
        self._scene_description: str = ""
        self._status_style: str = "normal"  # "normal" (gold), "error" (red), "stalled" (grey)
        self._warnings: list[str] = []  # Persistent warnings shown below status bar

        # Thinking panel state
        self._thinking_state: _ThinkingState | None = None
        self._thinking_scroll_offset: int = 0  # 0 = bottom, positive = scrolled up

        # Agent focus (Stage 2)
        self._agent_roster: list[str] = []
        self._focused_agent: str | None = None  # None = ALL (no filter)
        self._scroll_offsets: dict[str | None, int] = {}  # per-agent scroll positions

        # Resize presets (Stage 3): (log_ratio, thinking_ratio)
        self._resize_presets: list[tuple[int, int]] = [
            (5, 1),
            (3, 1),
            (2, 1),
            (1, 1),
            (1, 2),
            (1, 3),
        ]
        self._resize_index: int = 1  # default 3:1
        self._default_resize_index: int = 1  # for collapse-back

        # Scroll target: which panel arrows control
        self._scroll_target: str = "log"  # "log" or "thinking"

        if _RICH_AVAILABLE:
            self._console = Console()

    @property
    def is_active(self) -> bool:
        return self._live is not None

    def start(self) -> None:
        """Begin live display.  No-op if rich unavailable."""
        if not _RICH_AVAILABLE or not sys.stdout.isatty():
            return
        try:
            self._live = Live(
                self._build_layout(),
                console=self._console,
                refresh_per_second=4,
                screen=False,
            )
            self._live.start()
        except Exception as e:
            log.debug("Failed to start rich display: %s", e)
            self._live = None

    def stop(self) -> None:
        """End live display, restore terminal."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def log(self, subsystem: str, message: str, level: str = "info", *, agent: str | None = None) -> None:
        """Add a line to the scrolling log panel (thread-safe).

        Args:
            subsystem: Subsystem tag (e.g. "hippocampus", "scene").
            message: Human-readable log content.
            level: Severity level (unused currently, reserved).
            agent: Optional agent nickname — rendered as a colored prefix
                before the subsystem tag so multi-agent logs are legible.
        """
        # Scene/dialogue: bold + bright color for tag, white message
        # Bio subsystems: colored tag, grey message (readable but subdued)
        tag_colors = {
            # User input — distinctive so it stands out after Enter
            "user": "bold green",
            # Scene/dialogue — bold tags (always bright, never dimmed)
            "scene": "bold yellow",
            "npc": "bold yellow",
            "choice": "bold cyan",
            "result": "bold green",
            "blocked": "bold red",
            "turn": "bold blue",
            "summary": "bold cyan",
            "response": "bold green",
            "action": "bold cyan",
            "info": "dim white",
            # Substrate-learning headlines (NAc reward bias, tier promotion,
            # sleep consolidation, anticipatory pre-activation). Bold bright
            # magenta — visually distinct from every other channel so the
            # ▲ moments stand out in the scrolling log.
            "learn": "bold bright_magenta",
            # PFC deliberation — green (matches thinking panel border)
            "thought": "bold green",
            "deliberation": "bold green",
            # Tool execution — cyan (visible action the agent is taking)
            "motor": "bold cyan",
            # Bio subsystems — colored tags (message dimmed via _BIO_SUBSYSTEMS)
            "hippo": "cyan",
            "hippocampus": "cyan",
            "nac": "magenta",
            "fear": "red",
            "pain": "red",
            "exec": "green",
            "cerebellum": "blue",
            "atl": "magenta",
            "sensory": "cyan",
            "body": "dim cyan",
            "percept": "cyan",
            "reaction": "magenta",
            "scn": "yellow",
            "ec": "yellow",
            "pipeline": "dim white",
            # New bio-system subsystems (Track 1+2)
            "enrichment": "magenta",
            "imagination": "yellow",
            "discovery": "blue",
            "gate": "white",
            "sensor": "cyan",
            "drive": "yellow",
            "reflex": "bold red",
        }
        sub_lower = subsystem.lower()
        is_bio = sub_lower in self._BIO_SUBSYSTEMS
        with self._lock:
            color = tag_colors.get(sub_lower, "white")
            tag = f"[{color}][{subsystem:>6}][/{color}]"
            # Agent nickname prefix — bright_cyan so it stands out without
            # clashing with subsystem colors
            agent_prefix = f"[bright_cyan][{agent:>8}][/bright_cyan] " if agent else ""
            # Bio messages in grey (readable but visually recessive)
            if is_bio:
                markup = f"{agent_prefix}{tag} [bright_black]{message}[/bright_black]"
            else:
                markup = f"{agent_prefix}{tag} {message}"
            self._log_lines.append(_LogEntry(agent=agent, markup=markup))
            # Keep absolute scroll position stable: when scrolled up,
            # each new line pushes the bottom further away, so bump
            # the offset to compensate.  Bump ALL view and the matching
            # agent view (if any).
            for key in (None, agent):
                if self._scroll_offsets.get(key, 0) > 0:
                    self._scroll_offsets[key] = self._scroll_offsets[key] + 1
            self._refresh()

    def set_status(self, **fields: str) -> None:
        """Update status bar fields (thread-safe)."""
        with self._lock:
            self._status.update(fields)
            self._refresh()

    def set_prompt(self, text: str) -> None:
        """Set the current prompt text in the input panel (thread-safe)."""
        with self._lock:
            self._prompt_text = text
            self._refresh()

    def clear_prompt(self) -> None:
        """Clear the input panel (thread-safe)."""
        with self._lock:
            self._prompt_text = ""
            self._refresh()

    @property
    def log_count(self) -> int:
        """Number of lines in the log panel. Thread-safe."""
        with self._lock:
            return len(self._log_lines)

    @property
    def page_height(self) -> int:
        """Approximate visible log lines for page-up scrolling. Thread-safe."""
        with self._lock:
            raw = max(5, (self._console.height if self._console else 40) - 14)
            log_r, think_r = self._resize_presets[self._resize_index]
            return max(3, (raw * log_r) // (log_r + think_r))

    def set_urgent(self, urgent: bool) -> None:
        """Set prompt urgency (gold border when agent is asking a question). Thread-safe."""
        with self._lock:
            self._prompt_urgent = urgent
            self._refresh()

    def set_status_style(self, style: str) -> None:
        """Set status bar style: 'normal' (gold), 'error' (red), 'stalled' (grey). Thread-safe."""
        with self._lock:
            self._status_style = style
            self._refresh()

    def warn(self, message: str) -> None:
        """Add a persistent warning shown in a fixed panel below the status bar.

        Warnings stay visible (not scrolled away like log entries).
        Duplicate messages are ignored.  Thread-safe.
        """
        with self._lock:
            if message not in self._warnings:
                self._warnings.append(message)
                self._refresh()

    def clear_warnings(self) -> None:
        """Remove all persistent warnings. Thread-safe."""
        with self._lock:
            self._warnings.clear()
            self._refresh()

    def set_scene(self, title: str = "", description: str = "") -> None:
        """Set the scene header panel (thread-safe)."""
        with self._lock:
            self._scene_title = title
            self._scene_description = description
            self._refresh()

    def scroll(self, delta: int) -> None:
        """Scroll the log panel. Positive = up (older), negative = down (newer)."""
        with self._lock:
            filtered = self._filtered_log_lines()
            total = len(filtered)
            # Approximate visible lines accounting for thinking panel ratio
            raw_visible = max(5, (self._console.height if self._console else 40) - 14)
            log_r, think_r = self._resize_presets[self._resize_index]
            approx_visible = max(3, (raw_visible * log_r) // (log_r + think_r))
            max_offset = max(0, total - approx_visible)
            cur = self._scroll_offsets.get(self._focused_agent, 0)
            self._scroll_offsets[self._focused_agent] = max(0, min(max_offset, cur + delta))
            self._refresh()

    # Keymap-compatible scroll methods (called by name from _KEYMAP).
    # These delegate to log or thinking panel based on _scroll_target.
    def scroll_up(self) -> None:
        """Scroll active panel up 3 lines."""
        if self._scroll_target == "thinking":
            self.scroll_thinking_up()
        else:
            self.scroll(3)

    def scroll_down(self) -> None:
        """Scroll active panel down 3 lines."""
        if self._scroll_target == "thinking":
            self.scroll_thinking_down()
        else:
            self.scroll(-3)

    def scroll_bottom(self) -> None:
        """Right arrow: if thinking is active, collapse back to default.

        Otherwise jump to bottom of log panel.
        """
        if self._scroll_target == "thinking":
            with self._lock:
                self._resize_index = self._default_resize_index
                self._scroll_target = "log"
                # Reset to bottom so the panel auto-follows new thoughts
                self._thinking_scroll_offset = 0
                self._refresh()
        else:
            self.scroll(-999999)

    def scroll_page_up(self) -> None:
        """Page up in log panel."""
        self.scroll(self.page_height)

    def add_extension(self, ext: DisplayExtension) -> None:
        """Register a display extension (adds panels, thread-safe)."""
        with self._lock:
            self._extensions.append(ext)
            self._refresh()

    def set_thinking(
        self,
        reasoning: str,
        cycle: int,
        max_cycles: int,
        enrichment_tags: list[str] | None = None,
        *,
        agent: str | None = None,
        completed: bool = False,
        salience: float | None = None,
        enrichment_details: dict[str, str] | None = None,
    ) -> None:
        """Update the thinking panel with deliberation state (thread-safe).

        Args:
            reasoning: Full LLM reasoning text (or summary if completed).
            cycle: Current deliberation cycle number.
            max_cycles: Hard cap on cycles.
            enrichment_tags: Bio-system names that contributed enrichment.
            agent: Agent nickname producing this deliberation.
            completed: If True, show as a completed summary (dim).
            salience: Computed thought salience (0.0-1.0) for display.
            enrichment_details: Map of system name → summary of what it
                contributed.  Shown as expandable detail under enrichment tags.
        """
        tags = enrichment_tags or []
        details = enrichment_details or {}
        with self._lock:
            prev = self._thinking_state
            if completed and prev is not None:
                # On completion, preserve the accumulated history
                history = list(prev.history)
            elif prev is not None and not completed:
                # Always accumulate — full session thought stream, scrollable
                history = list(prev.history)
                history.append((cycle, reasoning, tags, salience))
            else:
                # No prior state — first thought
                history = [(cycle, reasoning, tags, salience)]

            self._thinking_state = _ThinkingState(
                reasoning=reasoning,
                cycle=cycle,
                max_cycles=max_cycles,
                enrichment_tags=tags,
                enrichment_details=details,
                started_at=prev.started_at if prev is not None and not completed else time.monotonic(),
                agent=agent,
                completed=completed,
                history=history,
            )
            # Auto-scroll to bottom on new content (not when completing)
            if not completed:
                self._thinking_scroll_offset = 0
            self._refresh()

    def clear_thinking(self) -> None:
        """Clear the thinking panel (thread-safe)."""
        with self._lock:
            self._thinking_state = None
            self._thinking_scroll_offset = 0
            self._refresh()

    def scroll_thinking_up(self) -> None:
        """Scroll the thinking panel up (older). Thread-safe."""
        with self._lock:
            self._thinking_scroll_offset = max(0, self._thinking_scroll_offset + 3)
            self._refresh()

    def scroll_thinking_down(self) -> None:
        """Scroll the thinking panel down (newer). Thread-safe."""
        with self._lock:
            self._thinking_scroll_offset = max(0, self._thinking_scroll_offset - 3)
            self._refresh()

    def register_agent(self, nickname: str) -> None:
        """Register an agent nickname for the roster (thread-safe).

        Appends only if not already present — preserves ordering stability.
        """
        with self._lock:
            if nickname not in self._agent_roster:
                self._agent_roster.append(nickname)

    # -- Agent focus (Stage 2) -----------------------------------------------

    def focus_next(self) -> None:
        """Cycle agent focus forward: ALL -> AUT -> ORCH -> NPC1 -> ... -> ALL. Thread-safe."""
        with self._lock:
            if not self._agent_roster:
                return
            if self._focused_agent is None:
                self._focused_agent = self._agent_roster[0]
            else:
                try:
                    idx = self._agent_roster.index(self._focused_agent)
                    if idx + 1 >= len(self._agent_roster):
                        self._focused_agent = None  # wrap to ALL
                    else:
                        self._focused_agent = self._agent_roster[idx + 1]
                except ValueError:
                    self._focused_agent = None
            self._refresh()

    def focus_prev(self) -> None:
        """Cycle agent focus backward. Thread-safe."""
        with self._lock:
            if not self._agent_roster:
                return
            if self._focused_agent is None:
                self._focused_agent = self._agent_roster[-1]
            else:
                try:
                    idx = self._agent_roster.index(self._focused_agent)
                    if idx == 0:
                        self._focused_agent = None  # wrap to ALL
                    else:
                        self._focused_agent = self._agent_roster[idx - 1]
                except ValueError:
                    self._focused_agent = None
            self._refresh()

    def focus_agent(self, name: str | None) -> bool:
        """Set focus to a specific agent by nickname (thread-safe).

        ``name=None`` or ``name in {"", "all", "ALL"}`` restores the
        ALL view (no filter).  Case-insensitive nickname match — returns
        ``True`` if focus was applied, ``False`` if the name didn't match
        any registered agent.

        Provides the portable alternative to Shift+arrow keybindings,
        which macOS Terminal.app strips before delivery.  See the
        ``/focus`` slash command in
        :mod:`maxim.simulation.orchestrator`.
        """
        with self._lock:
            if not name or name.lower() == "all":
                self._focused_agent = None
                self._refresh()
                return True
            target = name.strip()
            for agent in self._agent_roster:
                if agent.lower() == target.lower():
                    self._focused_agent = agent
                    self._refresh()
                    return True
            return False

    @property
    def agent_roster(self) -> list[str]:
        """Snapshot of registered agent nicknames (thread-safe)."""
        with self._lock:
            return list(self._agent_roster)

    @property
    def focused_agent(self) -> str | None:
        """Currently-focused agent nickname, or ``None`` for ALL."""
        with self._lock:
            return self._focused_agent

    def _filtered_log_lines(self) -> list[_LogEntry]:
        """Return log lines filtered by focused agent. Caller must hold _lock."""
        entries = list(self._log_lines)
        if self._focused_agent is None:
            return entries
        return [e for e in entries if e.agent is None or e.agent == self._focused_agent]

    # -- Layout resize (Stage 3) ---------------------------------------------

    def resize_thinking_more(self) -> None:
        """Increase thinking panel ratio (decrease log). Thread-safe.

        Expanding above default switches arrow scrolling to the thinking panel.
        """
        with self._lock:
            if self._resize_index < len(self._resize_presets) - 1:
                self._resize_index += 1
                if self._resize_index > self._default_resize_index:
                    self._scroll_target = "thinking"
                self._refresh()

    def resize_thinking_less(self) -> None:
        """Decrease thinking panel ratio (increase log). Thread-safe.

        Shrinking to default or below switches arrow scrolling back to log.
        """
        with self._lock:
            if self._resize_index > 0:
                self._resize_index -= 1
                if self._resize_index <= self._default_resize_index:
                    self._scroll_target = "log"
                self._refresh()

    def _refresh(self) -> None:
        """Rebuild and update the live display. Caller must hold _lock."""
        if self._live is not None:
            try:
                self._live.update(self._build_layout())
            except Exception:
                pass

    def _build_layout(self) -> Any:
        """Compose all panels into a rich Layout."""
        if not _RICH_AVAILABLE:
            return ""

        # Title bar (dark purple) — always shows "Maxim" with scene in body
        scene_parts = []
        if self._scene_title and self._scene_title != self._title:
            scene_parts.append(f"[bold]{self._scene_title}[/bold]")
        if self._scene_description:
            scene_parts.append(f"[italic]{self._scene_description}[/italic]")
        scene_content = " — ".join(scene_parts) if scene_parts else ""
        title_height = 3 if not scene_content else 4
        title_panel = Panel(
            Text.from_markup(scene_content) if scene_content else Text(""),
            title=f"[bold dark_violet]{self._title}[/bold dark_violet]",
            border_style="dark_violet",
            height=title_height,
        )

        # Status bar (dynamic color: gold=normal, red=error, grey=stalled)
        _status_colors = {
            "normal": "dark_goldenrod",
            "error": "red",
            "stalled": "grey50",
        }
        status_border = _status_colors.get(self._status_style, "dark_goldenrod")
        status_parts = [f"{k}: {v}" for k, v in self._status.items()]
        # Agent focus indicator
        focus_label = self._focused_agent or "ALL"
        status_parts.append(f"Agent: {focus_label}")
        # Layout preset indicator
        log_r, think_r = self._resize_presets[self._resize_index]
        status_parts.append(f"Layout: {log_r}:{think_r}")
        status_text = "  ".join(status_parts)
        status_panel = Panel(
            Text(status_text or "Ready", style="bold"),
            border_style=status_border,
            height=3,
        )

        # Warnings panel (fixed, between status and log)
        warnings_panel = None
        warnings_height = 0
        if self._warnings:
            warnings_text = "\n".join(f"[yellow]⚠ {w}[/yellow]" for w in self._warnings)
            warnings_height = len(self._warnings) + 2  # +2 for panel border
            warnings_panel = Panel(
                Text.from_markup(warnings_text),
                border_style="yellow",
                height=warnings_height,
            )

        # Compute how many lines fit: terminal height minus fixed panels.
        try:
            term_height = self._console.height if self._console else 40
        except Exception:
            term_height = 40
        prompt_lines = self._prompt_text.count("\n") + 1 if self._prompt_text else 1
        input_height = max(4, prompt_lines + 3)
        # Total flexible space for log + thinking body
        body_lines = max(8, term_height - title_height - 3 - warnings_height - input_height - 4)

        # Split body between log and thinking using resize presets
        log_ratio, thinking_ratio = self._resize_presets[self._resize_index]
        total_ratio = log_ratio + thinking_ratio
        # -4 accounts for the two panel borders (2 lines each)
        usable_body = max(4, body_lines - 4)
        visible_log_lines = max(3, (usable_body * log_ratio) // total_ratio)
        visible_thinking_lines = max(2, usable_body - visible_log_lines)

        # Log panel with scroll support + agent filtering
        all_entries = self._filtered_log_lines()
        total = len(all_entries)
        scroll_offset = self._scroll_offsets.get(self._focused_agent, 0)
        if scroll_offset > 0:
            end = total - scroll_offset
            start = max(0, end - visible_log_lines)
            visible = all_entries[start:end]
            scroll_indicator = f" [{start + 1}-{end}/{total}]"
        else:
            visible = all_entries[-visible_log_lines:]
            start = max(0, total - visible_log_lines)
            scroll_indicator = ""
        lines = [e.markup for e in visible]
        # Show top-of-log indicator when scrolled to the first entry
        if scroll_offset > 0 and start == 0:
            lines.insert(0, "[dim italic]-- top of log --[/dim italic]")
        log_text = "\n".join(lines) or "(no log entries)"
        log_title = f"Agent Log{scroll_indicator}"
        log_active = self._scroll_target == "log"
        log_panel = Panel(
            Text.from_markup(log_text),
            title=log_title,
            border_style="green" if log_active else "dim",
        )

        # Thinking panel
        thinking_panel = self._build_thinking_panel(visible_thinking_lines)

        # Input panel — dynamically sized to fit the prompt content
        prompt_display = self._prompt_text or "> _"
        # +2 for panel border top/bottom, +1 for breathing room
        prompt_lines = prompt_display.count("\n") + 1
        input_height = max(4, prompt_lines + 3)
        if self._prompt_urgent:
            input_border = "dark_goldenrod"
        elif self._prompt_text:
            input_border = "green"
        else:
            input_border = "dim"
        input_panel = Panel(
            Text(prompt_display),
            border_style=input_border,
            height=input_height,
        )

        # Extension panels (side column if any)
        if self._extensions:
            ext_renderables = []
            for ext in self._extensions:
                try:
                    rendered = ext.render()
                    ext_renderables.append(Panel(rendered, title=ext.panel_name(), border_style="cyan"))
                except Exception:
                    pass

            if ext_renderables:
                # Two-column layout: left column (log+thinking), extensions right
                layout = Layout()
                ext_panels = [
                    Layout(title_panel, size=title_height),
                    Layout(status_panel, size=3),
                ]
                if warnings_panel is not None:
                    ext_panels.append(Layout(warnings_panel, size=warnings_height))
                layout.split_column(
                    *ext_panels,
                    Layout(name="body"),
                    Layout(input_panel, size=input_height),
                )
                # Left column splits into log + thinking rows
                left_col = Layout(name="left_col")
                left_col.split_column(
                    Layout(log_panel, ratio=log_ratio),
                    Layout(thinking_panel, ratio=thinking_ratio),
                )
                layout["body"].split_row(
                    left_col,
                    Layout(*ext_renderables if len(ext_renderables) == 1 else ext_renderables[0], ratio=1),
                )
                return layout

        # Layout: scene + status + [warnings] + body (log + thinking) + input
        layout = Layout()
        body = Layout(name="body")
        body.split_column(
            Layout(log_panel, ratio=log_ratio),
            Layout(thinking_panel, ratio=thinking_ratio),
        )
        panels = [
            Layout(title_panel, size=title_height),
            Layout(status_panel, size=3),
        ]
        if warnings_panel is not None:
            panels.append(Layout(warnings_panel, size=warnings_height))
        panels.append(body)
        panels.append(Layout(input_panel, size=input_height))
        layout.split_column(*panels)
        return layout

    def _build_thinking_panel(self, visible_lines: int) -> Any:
        """Build the thinking panel renderable."""
        if not _RICH_AVAILABLE:
            return ""

        thinking_active = self._scroll_target == "thinking"

        state = self._thinking_state
        if state is None:
            border = "green" if thinking_active else "bright_black"
            return Panel(
                Text("No active deliberation", style="bright_black"),
                title="Thinking",
                border_style=border,
            )

        # Header: cycle indicator + elapsed + enrichment tags
        elapsed = time.monotonic() - state.started_at
        if state.completed:
            header = f"[dim]Deliberated {state.cycle} cycle(s) in {elapsed:.1f}s[/dim]"
        else:
            header = f"[bold green]Cycle {state.cycle}/{state.max_cycles}[/bold green] [dim]{elapsed:.1f}s[/dim]"
        if state.enrichment_tags:
            _TAG_ICONS = {
                "hippocampus": "💾",
                "nac": "🧠",
                "atl": "🏷️",
                "cerebellum": "🎯",
                "component_index": "🗺️",
                "working_memory": "📝",
                "fear": "😨",
                "sensory": "📡",
            }
            tag_parts = []
            for t in state.enrichment_tags:
                icon = _TAG_ICONS.get(t.lower(), "🔬")
                tag_parts.append(f"{icon}{t}")
            header += f"  [dim cyan]enriched: {', '.join(tag_parts)}[/dim cyan]"
        if state.agent:
            header = f"[bright_cyan][{state.agent}][/bright_cyan] {header}"

        # Build enrichment detail lines (Track 4: what each system contributed)
        enrichment_lines: list[str] = []
        if state.enrichment_details and not state.completed:
            _DETAIL_ICONS = {
                "hippocampus": "💾",
                "nac": "🧠",
                "atl": "🏷️",
                "cerebellum": "🎯",
                "component_index": "🗺️",
                "working_memory": "📝",
                "fear": "😨",
            }
            for sys_name, detail in state.enrichment_details.items():
                icon = _DETAIL_ICONS.get(sys_name.lower(), "🔬")
                enrichment_lines.append(f"[dim]{icon} {detail}[/dim]")

        # Build reasoning text as a continuous flowing stream.
        # Deduplication happens at the source (agent_loop novelty gate),
        # so we just concatenate here.
        reasoning_parts: list[str] = []
        for _cyc_num, cyc_reasoning, _cyc_tags, _cyc_salience in state.history:
            text = cyc_reasoning.strip()
            if text:
                reasoning_parts.append(text)
        reasoning_lines = "\n".join(reasoning_parts).split("\n") if reasoning_parts else []
        total_reasoning = len(reasoning_lines)
        # Reserve 1-2 lines for header (+ hint when active)
        hint_lines = 1 if thinking_active else 0
        content_lines = max(1, visible_lines - 1 - hint_lines)

        if self._thinking_scroll_offset > 0:
            end = max(0, total_reasoning - self._thinking_scroll_offset)
            start = max(0, end - content_lines)
            visible = reasoning_lines[start:end]
        else:
            visible = reasoning_lines[-content_lines:]

        reasoning_text = "\n".join(visible) if visible else ""
        style = "bright_black" if state.completed else "white"
        # Assemble: header + enrichment details + reasoning
        content_parts = [header]
        if enrichment_lines:
            content_parts.extend(enrichment_lines)
        if reasoning_text:
            content_parts.append(f"[{style}]{reasoning_text}[/{style}]")
        content = "\n".join(content_parts)

        # Hint when thinking panel is the active scroll target
        if thinking_active:
            content += "\n[dim italic]press right arrow to return to logs[/dim italic]"

        # Border: green when active scroll target, dim otherwise
        if thinking_active:
            border = "green"
        elif state.completed:
            border = "bright_black"
        else:
            border = "dim"
        title = "Thinking"
        if self._thinking_scroll_offset > 0 and total_reasoning > content_lines:
            title += f" [{total_reasoning - self._thinking_scroll_offset}/{total_reasoning}]"

        return Panel(
            Text.from_markup(content),
            title=title,
            border_style=border,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_display(mode: str = "auto") -> MaximDisplay | None:
    """Create display if conditions are met.

    Args:
        mode: "auto" (rich if available + TTY), "on" (force), "off" (disable).

    Returns:
        MaximDisplay instance, or None if display should be disabled.
    """
    if mode == "off":
        return None

    if mode == "on":
        return MaximDisplay()

    # Auto: require rich + TTY
    if _RICH_AVAILABLE and sys.stdout.isatty():
        return MaximDisplay()

    return None
