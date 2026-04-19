"""Universal prompt protocol — types, request/response, and handler ABC.

Every user interaction in Maxim flows through this protocol:
DM encounter choices, architect interviews, freeform agent chat,
confirmation dialogs, numeric inputs, and ratings.

The protocol separates *what* the system asks (PromptRequest) from
*how* it's rendered (PromptHandler), enabling the same interaction
logic to work across rich terminal UI, plain stdin, Python callbacks,
replay from JSONL, and non-interactive defaults.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class PromptType(Enum):
    """Prompt types for interactive use cases.

    Only types with production callers are included. Add new types
    WITH a production caller in the same commit — don't add types
    without callers (0.3.1 cleanup removed SHORT_TEXT, LONG_TEXT,
    NUMERIC, RATING for this reason).
    """

    SINGLE_CHOICE = "single_choice"  # Pick one from a list
    MULTI_CHOICE = "multi_choice"  # Pick N from a list
    CONFIRM = "confirm"  # Yes/No
    FREEFORM = "freeform"  # Unprompted user input (agent chat)


@dataclass(frozen=True)
class PromptRequest:
    """What the system wants from the user.

    Frozen for hashability/safety.
    """

    prompt_type: PromptType
    question: str
    options: tuple[str, ...] | None = None
    default: str | None = None
    timeout_sec: float = 300.0


@dataclass(frozen=True)
class PromptResponse:
    """What the user responded."""

    value: str | list[str]
    timed_out: bool = False
    was_default: bool = False
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Handler ABC
# ---------------------------------------------------------------------------


class PromptHandler(ABC):
    """How prompts are delivered and responses collected.

    Subclass this to implement a new rendering backend (rich, web, etc.).
    """

    @abstractmethod
    def prompt(self, request: PromptRequest) -> PromptResponse:
        """Present prompt to user and collect response."""


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------


class PlainPromptHandler(PromptHandler):
    """Bare print() + input() handler.  Works everywhere.

    This is the fallback when rich is not installed, stdout is not a TTY,
    or the user explicitly disables the display.
    """

    def prompt(self, request: PromptRequest) -> PromptResponse:
        start = time.time()

        # Display the question
        print(f"\n  {request.question}")

        # Display options for choice types
        if request.options and request.prompt_type in (
            PromptType.SINGLE_CHOICE,
            PromptType.MULTI_CHOICE,
        ):
            for i, opt in enumerate(request.options, 1):
                print(f"    {i}. {opt}")

        # Display default and timeout
        parts = []
        if request.default:
            parts.append(f"default: {request.default}")
        if request.timeout_sec < 300:
            parts.append(f"timeout: {request.timeout_sec:.0f}s")
        if parts:
            print(f"  [{', '.join(parts)}]")

        # Collect input
        print("  > ", end="", flush=True)
        response = None
        timed_out = False

        try:
            import select

            ready, _, _ = select.select([sys.stdin], [], [], request.timeout_sec)
            if ready:
                response = sys.stdin.readline().strip()
            else:
                timed_out = True
                print("(timed out)")
        except (ImportError, OSError):
            try:
                response = input()
            except EOFError:
                timed_out = True

        elapsed = time.time() - start

        if not response or timed_out:
            response = request.default or ""

        # Map numbered choice to option text
        if request.options and response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(request.options):
                response = request.options[idx]

        # Handle multi-choice (comma-separated)
        if request.prompt_type == PromptType.MULTI_CHOICE and isinstance(response, str):
            values = [v.strip() for v in response.split(",") if v.strip()]
            return PromptResponse(
                value=values,
                timed_out=timed_out,
                was_default=timed_out or not response,
                elapsed_s=elapsed,
            )

        return PromptResponse(
            value=response or "",
            timed_out=timed_out,
            was_default=timed_out or not response,
            elapsed_s=elapsed,
        )


class RichPromptHandler(PromptHandler):
    """Rich-based prompt handler with styled panels and tables.

    Falls back to PlainPromptHandler if rich is not available.
    """

    def __init__(self) -> None:
        self._fallback = PlainPromptHandler()
        try:
            from rich.console import Console

            self._console = Console()
            self._rich_available = True
        except ImportError:
            self._console = None
            self._rich_available = False

    def prompt(self, request: PromptRequest) -> PromptResponse:
        if not self._rich_available:
            return self._fallback.prompt(request)

        start = time.time()

        # Build display
        if request.prompt_type == PromptType.CONFIRM:
            return self._prompt_confirm(request, start)
        elif request.prompt_type in (PromptType.SINGLE_CHOICE, PromptType.MULTI_CHOICE):
            return self._prompt_choice(request, start)
        else:
            return self._prompt_text(request, start)

    def _prompt_confirm(self, request: PromptRequest, start: float) -> PromptResponse:
        from rich.prompt import Confirm

        try:
            default_bool = request.default and request.default.lower() in ("y", "yes", "true")
            result = Confirm.ask(f"  {request.question}", default=default_bool)
            elapsed = time.time() - start
            return PromptResponse(
                value="yes" if result else "no",
                elapsed_s=elapsed,
            )
        except (EOFError, KeyboardInterrupt):
            return PromptResponse(value=request.default or "no", timed_out=True)

    def _prompt_choice(self, request: PromptRequest, start: float) -> PromptResponse:
        from rich.panel import Panel
        from rich.table import Table

        # Show options as a rich table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("", style="bold cyan")
        table.add_column("")
        if request.options:
            for i, opt in enumerate(request.options, 1):
                table.add_row(f"  {i}.", opt)

        self._console.print(
            Panel(
                table,
                title=f"[bold]{request.question}[/bold]",
                border_style="blue",
            )
        )

        # Collect input
        try:
            from rich.prompt import Prompt

            choice_str = Prompt.ask(
                "  Choose",
                default=request.default or (request.options[0] if request.options else ""),
            )
        except (EOFError, KeyboardInterrupt):
            choice_str = request.default or ""

        elapsed = time.time() - start

        # Map number to option
        if request.options and choice_str.isdigit():
            idx = int(choice_str) - 1
            if 0 <= idx < len(request.options):
                choice_str = request.options[idx]

        if request.prompt_type == PromptType.MULTI_CHOICE:
            values = [v.strip() for v in choice_str.split(",") if v.strip()]
            return PromptResponse(value=values, elapsed_s=elapsed)

        return PromptResponse(value=choice_str, elapsed_s=elapsed)

    def _prompt_text(self, request: PromptRequest, start: float) -> PromptResponse:
        from rich.prompt import Prompt

        try:
            result = Prompt.ask(f"  {request.question}", default=request.default or "")
        except (EOFError, KeyboardInterrupt):
            result = request.default or ""

        elapsed = time.time() - start
        return PromptResponse(value=result, elapsed_s=elapsed)


class CallbackPromptHandler(PromptHandler):
    """Delegates prompts to a user-provided Python callback.

    Used by the Python API so programmatic users can handle prompts
    without terminal interaction.

    Example::

        def my_handler(request: PromptRequest) -> str:
            if request.prompt_type == PromptType.SINGLE_CHOICE:
                return request.options[0]
            return request.default or ""

        handler = CallbackPromptHandler(my_handler)
    """

    def __init__(self, callback: Callable[[PromptRequest], str | list[str] | None]) -> None:
        self._callback = callback

    def prompt(self, request: PromptRequest) -> PromptResponse:
        start = time.time()
        try:
            result = self._callback(request)
        except Exception as e:
            log.warning("Callback prompt handler failed: %s", e)
            result = None

        elapsed = time.time() - start

        if result is None:
            result = request.default or ""

        return PromptResponse(
            value=result,
            was_default=result == request.default,
            elapsed_s=elapsed,
        )


class NonInteractiveHandler(PromptHandler):
    """Returns defaults immediately.  For CI, headless, and automated runs."""

    def prompt(self, request: PromptRequest) -> PromptResponse:
        value = request.default or ""
        if request.prompt_type == PromptType.MULTI_CHOICE:
            # Return first option as list, or empty list
            if request.options:
                value = [request.options[0]]
            else:
                value = []
        elif request.prompt_type == PromptType.CONFIRM:
            value = request.default or "no"
        return PromptResponse(value=value, was_default=True)


class SimPromptHandler(PromptHandler):
    """Prompt handler that coordinates with the sim stdin reader thread.

    Instead of reading stdin directly (which would fight with the
    ``sim.stdin`` reader thread), this handler posts a pending prompt
    and blocks until the stdin reader forwards the user's response.

    The stdin reader checks ``has_pending_prompt`` each iteration.
    When a prompt is pending, the next line of user input is forwarded
    via ``deliver_response()`` instead of being processed as a command.
    """

    def __init__(self, stop_event: threading.Event | None = None) -> None:
        self._pending: PromptRequest | None = None
        self._response_queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._stop_event = stop_event

    @property
    def has_pending_prompt(self) -> bool:
        """True when a tool is waiting for user input."""
        with self._lock:
            return self._pending is not None

    @property
    def pending_display_text(self) -> str:
        """The prompt text to show in the display/terminal."""
        with self._lock:
            if self._pending is None:
                return ""
            lines = [self._pending.question]
            if self._pending.options:
                for i, opt in enumerate(self._pending.options, 1):
                    lines.append(f"  [{i}] {opt}")
            return "\n".join(lines)

    def deliver_response(self, text: str) -> None:
        """Called by the stdin reader to forward user input to the blocked tool."""
        self._response_queue.put(text)

    def prompt(self, request: PromptRequest) -> PromptResponse:
        """Pause the Live display, collect input directly, then resume.

        Rich Live and ``input()`` can't coexist — Live owns the terminal
        cursor, so keystrokes echo below the panel instead of inside it.
        The fix: stop Live while the user is typing (the AUT and
        orchestrator are both blocked anyway, so nothing updates), then
        restart it after.
        """
        start = time.time()

        from maxim.simulation.sim_logger import get_active_display

        display = get_active_display()
        was_active = display is not None and display.is_active

        # Stop Live so the terminal is clean for input
        if was_active:
            display.stop()

        # Mark prompt as pending (stall detector / bridge gate check this)
        with self._lock:
            self._pending = request

        response = request.default or ""
        try:
            # Show the question with Rich styling (panel is stopped, so
            # print goes to a clean terminal)
            print(f"\n  {request.question}")
            if request.options:
                for i, opt in enumerate(request.options, 1):
                    print(f"    [{i}] {opt}")
            if request.default:
                print(f"  [default: {request.default}]")
            print()

            # Read input directly — no cursor fighting, clean terminal
            try:
                import select

                print("  > ", end="", flush=True)
                ready, _, _ = select.select([sys.stdin], [], [], request.timeout_sec)
                if ready:
                    response = sys.stdin.readline().strip()
                else:
                    response = request.default or ""
                    print("(timed out)")
            except (ImportError, OSError):
                try:
                    response = input("  > ")
                except EOFError:
                    response = request.default or ""

            if not response:
                response = request.default or ""

            elapsed = time.time() - start

            # Resolve numbered choice to option text
            if request.options and response.isdigit():
                idx = int(response) - 1
                if 0 <= idx < len(request.options):
                    response = request.options[idx]

            return PromptResponse(value=response, elapsed_s=elapsed)
        finally:
            with self._lock:
                self._pending = None
            # Restart Live — display resumes with the response logged
            if was_active and display is not None:
                display.start()
                try:
                    from maxim.simulation.sim_logger import _emit

                    _emit(f"  You answered: {response}", "scene")
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_VALID_MODES = frozenset({"auto", "rich", "plain", "non-interactive", "callback"})


def create_handler(
    mode: str = "auto",
    callback: Callable | None = None,
) -> PromptHandler:
    """Create the appropriate PromptHandler based on mode.

    Args:
        mode: One of "auto", "rich", "plain", "non-interactive", "callback".
        callback: Required if mode is "callback".

    Returns:
        A PromptHandler instance.

    Raises:
        ValueError: If *mode* is not one of the recognised values.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown prompt handler mode {mode!r}. Valid modes: {', '.join(sorted(_VALID_MODES))}")

    if mode == "callback":
        if callback is None:
            raise ValueError("callback mode requires a callback function")
        handler = CallbackPromptHandler(callback)
        log.info("PromptHandler: CallbackPromptHandler (mode=callback)")
        return handler

    if mode == "non-interactive":
        log.info("PromptHandler: NonInteractiveHandler (mode=non-interactive)")
        return NonInteractiveHandler()

    if mode == "plain":
        log.info("PromptHandler: PlainPromptHandler (mode=plain)")
        return PlainPromptHandler()

    if mode == "rich":
        handler = RichPromptHandler()
        if not handler._rich_available:
            log.warning("Rich not available, falling back to plain prompt handler")
            return PlainPromptHandler()
        log.info("PromptHandler: RichPromptHandler (mode=rich)")
        return handler

    # Auto mode: rich if available + TTY, else plain
    is_tty = sys.stdout.isatty()
    if is_tty:
        try:
            handler = RichPromptHandler()
            if handler._rich_available:
                log.info("PromptHandler: RichPromptHandler (mode=auto, tty=True)")
                return handler
        except Exception:
            pass

    log.info("PromptHandler: PlainPromptHandler (mode=auto, tty=%s)", is_tty)
    return PlainPromptHandler()
