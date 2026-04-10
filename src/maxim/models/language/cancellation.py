"""Shared cancellation primitive for in-flight LLM requests.

Retry loops in backend modules (anthropic, openai, etc.) can spend tens of
seconds in backoff sleeps after a transient failure. If the user interrupts
the process mid-retry, a plain ``time.sleep(backoff)`` is not interruptible
and the loop will complete its full retry chain before returning — burning
cloud credits and holding up shutdown.

This module provides a process-wide ``threading.Event`` that any LLM code path
can consult. When the orchestrator (or a SIGINT handler) calls
``request_shutdown()``, retry loops observe it via ``is_shutdown_requested()``
and exit immediately. ``shutdown_wait(seconds)`` is a drop-in replacement for
``time.sleep`` that returns early when shutdown is signalled.

The event is global by design: the LLM call stack crosses many module
boundaries (worker pool → router → backend → HTTP client → retry loop), and
threading a cancellation token through every signature would be invasive and
brittle. A single well-documented global is cleaner than partial propagation.

Call sites:
  - Orchestrator shutdown → ``request_shutdown()``
  - SIGINT handler (safety net) → ``request_shutdown()``
  - Backend retry loops → ``shutdown_wait(backoff)`` instead of ``time.sleep``
  - Backend retry loops → ``if is_shutdown_requested(): break`` at top of each iteration

The event is ``clear()``-able via ``reset_shutdown()`` for test isolation;
production code should never reset it.
"""

from __future__ import annotations

import threading

_SHUTDOWN_EVENT = threading.Event()


def request_shutdown() -> None:
    """Signal that in-flight LLM work should abort at the next checkpoint.

    Safe to call from any thread. Idempotent — multiple calls are harmless.
    """
    _SHUTDOWN_EVENT.set()


def is_shutdown_requested() -> bool:
    """Return True if shutdown has been requested.

    Backend retry loops check this at the top of each iteration so they can
    abandon remaining retries immediately.
    """
    return _SHUTDOWN_EVENT.is_set()


def shutdown_wait(seconds: float) -> bool:
    """Sleep up to ``seconds``, returning early if shutdown is requested.

    Drop-in replacement for ``time.sleep(seconds)`` in retry backoffs. The
    event is process-wide, so any caller (orchestrator, signal handler,
    worker) can wake all waiting retry loops at once.

    Returns:
        True if shutdown was requested during the wait, False if the full
        duration elapsed normally. Callers can use the return value to break
        out of retry loops explicitly, though ``is_shutdown_requested()`` at
        the next iteration top is usually sufficient.
    """
    return _SHUTDOWN_EVENT.wait(seconds)


def reset_shutdown() -> None:
    """Clear the shutdown event for a fresh sim/session.

    Safe to call at sim boundaries — ``start_simulation_mode`` resets the
    event on entry so successive sims in the same process (REPL, continuous
    runs, test suites) start with a clean slate. Do NOT call this from
    inside an active sim; it would unmask a Ctrl+C mid-run.
    """
    _SHUTDOWN_EVENT.clear()
