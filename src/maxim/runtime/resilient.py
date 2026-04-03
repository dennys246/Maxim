"""Structured error handling for the agentic loop (Phase 7).

Replaces blanket ``except Exception: pass`` with a decorator that logs
at the appropriate severity level based on subsystem tier.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from maxim.utils.structured_logging import log_agentic

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Tier → log level mapping
_TIER_LEVELS = {
    "critical": logging.WARNING,
    "important": logging.WARNING,
    "optional": logging.DEBUG,
    "silent": logging.DEBUG,
}


def resilient(tier: str = "important", operation: str = "") -> Callable[[F], F]:
    """Decorator for error-resilient subsystem calls.

    Instead of inline ``try/except Exception: pass``, wrap the call with
    ``@resilient(tier, operation)`` to get consistent logging:

    - **critical**: Tool execution, LLM submission, autonomy check.
      Logs WARNING + agentic structured log.
    - **important**: Memory storage, hippocampus, context pool.
      Logs WARNING.
    - **optional**: DefaultNetwork, simulation logger, provenance.
      Logs DEBUG.
    - **silent**: Step callbacks, on_event callbacks.
      Logs DEBUG only.

    The decorated function returns ``None`` on failure.
    """
    level = _TIER_LEVELS.get(tier, logging.WARNING)

    def decorator(fn: F) -> F:
        op_name = operation or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.log(level, "%s failed: %s", op_name, e)
                if tier in ("critical",):
                    log_agentic(
                        "agent_loop", "error",
                        {"context": op_name, "error": str(e)[:200]},
                        level="WARNING",
                    )
                return None

        return wrapper  # type: ignore[return-value]

    return decorator
