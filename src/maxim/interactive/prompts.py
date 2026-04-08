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
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class PromptType(Enum):
    """Universal prompt types covering every interactive use case."""

    SINGLE_CHOICE = "single_choice"  # Pick one from a list
    MULTI_CHOICE = "multi_choice"  # Pick N from a list
    CONFIRM = "confirm"  # Yes/No
    SHORT_TEXT = "short_text"  # One-line freeform (name, keyword)
    LONG_TEXT = "long_text"  # Multi-line freeform (backstory, notes)
    FREEFORM = "freeform"  # Unprompted user input (agent chat)
    NUMERIC = "numeric"  # Number input with range validation
    RATING = "rating"  # 1-5 or 1-10 scale


def freeze_context(**kwargs: Any) -> tuple[tuple[str, Any], ...]:
    """Helper to create frozen context for PromptRequest."""
    return tuple(kwargs.items())


@dataclass(frozen=True)
class PromptRequest:
    """What the system wants from the user.

    Frozen for hashability/safety.  Use :func:`freeze_context` to build
    the ``context`` field from keyword arguments.
    """

    prompt_type: PromptType
    question: str
    options: tuple[str, ...] | None = None
    default: str | None = None
    timeout_sec: float = 300.0
    min_selections: int = 1
    max_selections: int | None = None
    value_range: tuple[float, float] | None = None
    context: tuple[tuple[str, Any], ...] = ()


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

    def supports_freeform(self) -> bool:
        """Whether this handler supports unprompted user input."""
        return False

    def poll_freeform(self) -> str | None:
        """Check for unprompted user input (non-blocking)."""
        return None


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

        # Display range for numeric
        if request.value_range and request.prompt_type == PromptType.NUMERIC:
            lo, hi = request.value_range
            print(f"    (range: {lo} - {hi})")

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

        self._console.print(Panel(
            table,
            title=f"[bold]{request.question}[/bold]",
            border_style="blue",
        ))

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


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
    """
    if mode == "callback":
        if callback is None:
            raise ValueError("callback mode requires a callback function")
        return CallbackPromptHandler(callback)

    if mode == "non-interactive":
        return NonInteractiveHandler()

    if mode == "plain":
        return PlainPromptHandler()

    if mode == "rich":
        handler = RichPromptHandler()
        if not handler._rich_available:
            log.warning("Rich not available, falling back to plain prompt handler")
            return PlainPromptHandler()
        return handler

    # Auto mode: rich if available + TTY, else plain
    if mode == "auto":
        if sys.stdout.isatty():
            try:
                handler = RichPromptHandler()
                if handler._rich_available:
                    return handler
            except Exception:
                pass
        return PlainPromptHandler()

    return PlainPromptHandler()
