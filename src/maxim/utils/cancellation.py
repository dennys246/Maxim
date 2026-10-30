"""Cooperative cancellation for LLM calls.

**Module location:** lives in ``maxim/utils/`` (not ``maxim/agents/``)
because the consumers span the full LLM stack — ``agents/llm_worker.py``,
``models/language/router.py``, and ``models/language/maxim_peer_backend.py``
all import these primitives. Putting the module in ``agents/`` would
create a router → agents back-edge in the dependency graph; ``utils/``
is the neutral home that everyone can depend on without cycles. (Plan
3.5 R6 review caught this — the original location was ``agents/``.)

When ``LLMWorker._call_llm_with_timeout`` abandons a future, the orphaned
background thread is still executing inside the router's ``_inference_lock``.
``future.cancel()`` only sets a flag on the future object — it cannot stop
a thread that is already running. The running thread keeps the lock until
the underlying HTTP call returns (observed: Cloudflare 524 at ~125s),
blocking every subsequent LLM call and polluting ``_provider_states``
with a phantom failure.

This module provides a ``ContextVar`` holding a ``threading.Event`` that
the orphaned thread can check at natural checkpoints (before an HTTP post,
between streaming chunks, before recording a provider failure) and
short-circuit with a clean exception. The main thread sets the event when
its timeout fires; the background thread sees it and unwinds.

**Two load-bearing rules** — if either is violated, cross-thread
signalling breaks silently:

1. **The ContextVar value MUST be the Event itself (a mutable shared
   reference), NOT a bool.** If we stored a bool, the child thread
   would see whatever value was set at submit time with no way for the
   parent to update it. The Event is shared by reference, so both
   threads see ``event.set()`` calls on the same object.

2. **ThreadPoolExecutor.submit() does NOT automatically propagate
   ContextVars.** The child thread starts with an empty context. You
   MUST wrap the callable with ``contextvars.copy_context().run(...)``
   when submitting::

       import contextvars
       ctx = contextvars.copy_context()
       future = executor.submit(ctx.run, worker_fn, *args, **kwargs)

   Without this wrapper, ``current_cancel_event()`` returns ``None`` in
   the child thread and cancellation is a no-op.

Both rules are enforced by regression tests in
``tests/unit/test_cancellation.py``. If a future refactor breaks either,
those tests fail loudly.

Usage pattern::

    import contextvars
    from maxim.utils.cancellation import (
        current_cancel_event, set_cancel_event, reset_cancel_event,
    )

    # Main thread, before submitting work:
    event = threading.Event()
    token = set_cancel_event(event)
    try:
        ctx = contextvars.copy_context()    # Snapshot parent context
        future = executor.submit(ctx.run, worker_fn)
        try:
            return future.result(timeout=timeout_s)
        except TimeoutError:
            event.set()                     # Signal orphan to unwind
            raise
    finally:
        reset_cancel_event(token)

    # Background thread, at checkpoints:
    ev = current_cancel_event()
    if ev is not None and ev.is_set():
        raise SomeCleanException("cancelled")
"""

from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from maxim.utils.deadline import Deadline

logger = logging.getLogger(__name__)

# The ContextVar's value MUST be an Event (mutable shared reference),
# NOT a bool. See module docstring for why.
_cancel_event_var: ContextVar[threading.Event | None] = ContextVar("maxim_cancel_event", default=None)


def current_cancel_event() -> threading.Event | None:
    """Return the cancellation Event for the current context, or None.

    Called from the background thread at checkpoint boundaries. If the
    returned Event is set, the caller should raise a clean exception
    (e.g., ``BackendDown(fix_hint="cancelled")``) to unwind through
    exception handlers that properly release locks and clean up state.
    """
    return _cancel_event_var.get()


def set_cancel_event(event: threading.Event | None) -> Token[threading.Event | None]:
    """Bind ``event`` as the cancellation sentinel for the current context.

    Returns a token that must be passed to :func:`reset_cancel_event` in
    a ``finally`` block to avoid leaking the binding into other requests.
    """
    return _cancel_event_var.set(event)


def reset_cancel_event(token: Token[threading.Event | None]) -> None:
    """Restore the prior cancellation binding. Pair with :func:`set_cancel_event`."""
    _cancel_event_var.reset(token)


def is_cancelled() -> bool:
    """Shorthand: True iff a cancel Event is bound AND set."""
    event = _cancel_event_var.get()
    return event is not None and event.is_set()


# ── The process stop request (#1438; docs/plans/bounded_stop.md §5) ─────────────────────────────────────────────
# The THIRD cancellation primitive, beside the per-call ContextVar above and the LLM backend's process-wide
# `models/language/cancellation.request_shutdown()`. It lives here, the neutral home, so memory, runtime, console and
# the CLI can read it without depending on the LLM package. Sticky: once requested it stays set until
# `clear_stop_request()`: the owner that made it, after its stop work finished (never on overrun), or the autouse
# conftest scrub. PR 0 lands the request and its accessors, which bound
# a session's abnormal-exit persist; PR 1b registers the stop events it also sets (loops, LLM workers) and the
# backend's hook pair.


@dataclass(frozen=True)
class StopRequest:
    """A stop the process was asked to make: why, and by when. In-process only, never persisted.

    SHAPE-FROZEN at 1.0 (CC3): never persisted or sent; it carries a monotonic Deadline.
    """

    reason: str
    deadline: Deadline


_stop_lock = threading.Lock()
_stop_request: StopRequest | None = None
# Held from the snapshot through running the hooks, for a request AND a clear, so a clear's on-clear hook can never
# land after a newer request's on-request hook (separate from ``_stop_lock``, which guards only the fields).
_hook_lock = threading.Lock()
# Event -> how many holders registered it: one event can be shared by several loops (the orchestrator's AUT and orch
# loops and a sub-simulation tool share its stop event), so the first to end must not unregister it for the others.
_stop_events: weakref.WeakKeyDictionary[threading.Event, int] = weakref.WeakKeyDictionary()
_stop_hooks: list[tuple[Callable[[], None], Callable[[], None] | None]] = []


def register_stop_event(event: threading.Event) -> None:
    """Set ``event`` when the process stop is requested (weakly held: it dies with its owner). Already requested: it is
    set now, so a loop or LLM worker that starts after the request is born stopped (#1438 PR 1b)."""
    with _stop_lock:
        _stop_events[event] = _stop_events.get(event, 0) + 1
        requested = _stop_request is not None
    if requested:
        event.set()


def unregister_stop_event(event: threading.Event) -> None:
    """Drop one registration of ``event``; a later request stops setting it once every holder has ended."""
    with _stop_lock:
        count = _stop_events.get(event, 0) - 1
        if count > 0:
            _stop_events[event] = count
        else:
            _stop_events.pop(event, None)


@contextmanager
def stop_event_registered(event: Any) -> Iterator[None]:
    """Register ``event`` (a ``threading.Event``; anything else is ignored) for the block: a loop's own stop event."""
    if not isinstance(event, threading.Event):
        yield
        return
    register_stop_event(event)
    try:
        yield
    finally:
        unregister_stop_event(event)


def register_stop_hook(on_request: Callable[[], None], on_clear: Callable[[], None] | None = None) -> None:
    """Run ``on_request`` when the stop is requested and ``on_clear`` when that request is cleared. For a module
    ``utils`` must not import: the LLM backend registers its ``request_shutdown`` / ``reset_shutdown`` pair here, so a
    cleared stop re-arms the backend. Hooks must not block, and must not call ``request_stop``,
    ``clear_stop_request`` or ``register_stop_hook`` (they run under the non-reentrant hook lock); each is guarded and
    logged. Already requested:
    ``on_request`` runs now (a module imported after the request)."""
    with _hook_lock:
        with _stop_lock:
            _stop_hooks.append((on_request, on_clear))
            requested = _stop_request is not None
        if requested:
            _run_hooks([on_request])


def _run_hooks(hooks: list[Callable[[], None]]) -> None:
    for hook in hooks:
        try:
            hook()
        except Exception:
            logger.exception("Stop hook %r failed", hook)


def request_stop(reason: str, deadline: Deadline) -> StopRequest:
    """Ask the process to stop within ``deadline``. The FIRST request wins; later ones return it unchanged.

    The first request sets every registered stop event (loops, LLM workers) and runs every on-request hook, outside
    ``_stop_lock``. Only a process owner makes it (the console lifespan, its drain, the serve signal path; owner
    decision 2026-10-30), and the one that made it clears it with ``clear_stop_request(expected=...)``."""
    global _stop_request
    with _hook_lock:
        with _stop_lock:
            if _stop_request is not None:
                return _stop_request
            _stop_request = request = StopRequest(reason=reason, deadline=deadline)
            events = list(_stop_events)
            hooks = [on_request for on_request, _ in _stop_hooks]
        for event in events:
            event.set()
        _run_hooks(hooks)
    return request


def current_stop() -> StopRequest | None:
    """The process's stop request, or None."""
    return _stop_request


def yield_requested() -> bool:
    """True once a stop was requested: a cooperative loop yields at its next turn boundary."""
    return _stop_request is not None


def clear_stop_request(expected: StopRequest | None = None) -> bool:
    """Forget the stop request; True if one was cleared. With ``expected``, a compare-and-clear: only that request is
    cleared (another owner's later one stays). Without it, unconditional (the conftest scrub). A clear runs every
    on-clear hook; registered events stay set (their owners are stopping)."""
    global _stop_request
    with _hook_lock:
        with _stop_lock:
            if _stop_request is None or (expected is not None and _stop_request is not expected):
                return False
            _stop_request = None
            hooks = [on_clear for _, on_clear in _stop_hooks if on_clear is not None]
        _run_hooks(hooks)
    return True
