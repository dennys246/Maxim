"""Interactive runtime — universal prompt protocol + rich display.

Provides a clean abstraction between *what* the system asks the user
and *how* it's rendered.  Every interaction (DM choices, architect
interviews, agent questions, freeform input) goes through the
:class:`PromptHandler` protocol.

Key types:
    PromptType      — what kind of input (single choice, text, confirm, etc.)
    PromptRequest   — what the system wants from the user
    PromptResponse  — what the user responded
    PromptHandler   — ABC for rendering + collecting input

Built-in handlers:
    PlainPromptHandler      — print() + input() (works everywhere)
    RichPromptHandler       — rich panels + tables (requires rich)
    CallbackPromptHandler   — delegates to a Python callback (API use)
    NonInteractiveHandler   — returns defaults immediately (CI/headless)
"""

from maxim.interactive.prompts import (
    PromptType,
    PromptRequest,
    PromptResponse,
    PromptHandler,
)

__all__ = [
    "PromptType",
    "PromptRequest",
    "PromptResponse",
    "PromptHandler",
]
