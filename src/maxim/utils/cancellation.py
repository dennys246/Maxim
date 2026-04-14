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

import threading
from contextvars import ContextVar, Token

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
