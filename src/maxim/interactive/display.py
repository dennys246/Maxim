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
from abc import ABC, abstractmethod
from collections import deque
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

    # Bio subsystems get dim styling; scene/dialogue stays bright
    _BIO_SUBSYSTEMS = frozenset(
        {
            "hippo",
            "hippocampus",
            "nac",
            "fear",
            "pain",
            "exec",
            "motor",
            "cerebellum",
            "atl",
            "sensory",
            "body",
            "percept",
            "reaction",
            "scn",
            "ec",
            "pipeline",
            "salience",
        }
    )

    def __init__(self, title: str = "Maxim", max_log_lines: int = 500) -> None:
        self._title = title
        self._log_lines: deque[str] = deque(maxlen=max_log_lines)
        self._status: dict[str, str] = {}
        self._prompt_text: str = ""
        self._extensions: list[DisplayExtension] = []
        self._live: Any = None  # rich.live.Live when active
        self._console: Any = None
        self._lock = threading.RLock()
        self._scroll_offset: int = 0  # 0 = bottom (newest), positive = scrolled up

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

    def log(self, subsystem: str, message: str, level: str = "info") -> None:
        """Add a line to the scrolling log panel (thread-safe)."""
        # Scene/dialogue: bold + bright color for tag, white message
        # Bio subsystems: colored tag, grey message (readable but subdued)
        tag_colors = {
            # Scene/dialogue — bold tags
            "scene": "bold yellow",
            "npc": "bold yellow",
            "choice": "bold white",
            "result": "bold green",
            "blocked": "bold red",
            "turn": "bold blue",
            "summary": "bold white",
            "response": "bold green",
            "action": "bold cyan",
            "info": "white",
            # Bio subsystems — colored tags (not dim)
            "hippo": "cyan",
            "hippocampus": "cyan",
            "nac": "magenta",
            "fear": "red",
            "pain": "red",
            "exec": "green",
            "motor": "blue",
            "cerebellum": "blue",
            "atl": "magenta",
            "sensory": "cyan",
            "body": "white",
            "percept": "cyan",
            "reaction": "magenta",
            "scn": "yellow",
        }
        sub_lower = subsystem.lower()
        is_bio = sub_lower in self._BIO_SUBSYSTEMS
        with self._lock:
            color = tag_colors.get(sub_lower, "white")
            tag = f"[{color}][{subsystem:>6}][/{color}]"
            # Bio messages in grey (readable but visually recessive)
            line = f"{tag} {message}" if not is_bio else f"{tag} [bright_black]{message}[/bright_black]"
            self._log_lines.append(line)
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

    def scroll(self, delta: int) -> None:
        """Scroll the log panel. Positive = up (older), negative = down (newer)."""
        with self._lock:
            total = len(self._log_lines)
            # Approximate visible lines (exact is computed in _build_layout)
            approx_visible = max(5, (self._console.height if self._console else 40) - 10)
            max_offset = max(0, total - approx_visible)
            self._scroll_offset = max(0, min(max_offset, self._scroll_offset + delta))
            self._refresh()

    def add_extension(self, ext: DisplayExtension) -> None:
        """Register a display extension (adds panels, thread-safe)."""
        with self._lock:
            self._extensions.append(ext)
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

        # Status bar
        status_text = "  ".join(f"{k}: {v}" for k, v in self._status.items())
        status_panel = Panel(
            Text(status_text or "Ready", style="bold"),
            title=f"[bold blue]{self._title}[/bold blue]",
            border_style="blue",
            height=3,
        )

        # Compute how many log lines fit: terminal height minus status (3)
        # minus input panel (dynamic) minus borders/padding (~4).
        try:
            term_height = self._console.height if self._console else 40
        except Exception:
            term_height = 40
        prompt_lines = self._prompt_text.count("\n") + 1 if self._prompt_text else 1
        input_height = max(4, prompt_lines + 3)
        visible_lines = max(5, term_height - 3 - input_height - 4)

        # Log panel with scroll support
        all_lines = list(self._log_lines)
        total = len(all_lines)
        if self._scroll_offset > 0:
            end = total - self._scroll_offset
            start = max(0, end - visible_lines)
            visible = all_lines[start:end]
            scroll_indicator = f" [{start + 1}-{end}/{total}]"
        else:
            visible = all_lines[-visible_lines:]
            scroll_indicator = ""
        log_text = "\n".join(visible) or "(no log entries)"
        log_title = f"Agent Log{scroll_indicator}"
        log_panel = Panel(
            Text.from_markup(log_text),
            title=log_title,
            border_style="dim",
        )

        # Input panel — dynamically sized to fit the prompt content
        prompt_display = self._prompt_text or "> _"
        # +2 for panel border top/bottom, +1 for breathing room
        prompt_lines = prompt_display.count("\n") + 1
        input_height = max(4, prompt_lines + 3)
        input_panel = Panel(
            Text(prompt_display),
            border_style="green" if self._prompt_text else "dim",
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
                # Two-column layout: log left, extensions right
                layout = Layout()
                layout.split_column(
                    Layout(status_panel, size=3),
                    Layout(name="body"),
                    Layout(input_panel, size=input_height),
                )
                layout["body"].split_row(
                    Layout(log_panel, ratio=2),
                    Layout(*ext_renderables if len(ext_renderables) == 1 else ext_renderables[0], ratio=1),
                )
                return layout

        # Simple layout: status + log + input
        layout = Layout()
        layout.split_column(
            Layout(status_panel, size=3),
            Layout(log_panel),
            Layout(input_panel, size=input_height),
        )
        return layout


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
