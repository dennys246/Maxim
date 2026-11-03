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
import os
import signal
import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from subprocess import Popen as _REAL_POPEN
from typing import Any, NamedTuple

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


# Set by a caller that abandons its own call when the process stop is requested: it binds the cancel event above AND
# registered its stop event with ``register_stop_event`` (``LLMWorker``). ``LLMRouter._complete_text`` then runs the
# call inline; every other call it abandons itself at a stop (#1438 PR 1c). A cancel event bound only for a timeout
# does not set it, so binding one never silently loses the stop's abandon.
_abandons_on_stop_var: ContextVar[bool] = ContextVar("maxim_abandons_on_stop", default=False)


def mark_abandons_on_stop() -> Token[bool]:
    """Declare that this context's caller abandons its call on the process stop; pair with ``reset_abandons_on_stop``."""
    return _abandons_on_stop_var.set(True)


def reset_abandons_on_stop(token: Token[bool]) -> None:
    _abandons_on_stop_var.reset(token)


def abandons_on_stop() -> bool:
    """Whether this context's caller declared it abandons its own call on the process stop."""
    return _abandons_on_stop_var.get()


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
    ``utils`` must not import: the LLM backend registers its shutdown event's set / clear pair here, so a cleared stop
    re-arms the backend. Hooks must not block, and must not call ``request_stop``, ``clear_stop_request``,
    ``register_stop_hook`` or ``reset_unless_stopped`` (they run under the non-reentrant hook lock); each is guarded and
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


# ── Human-input reads a signal may interrupt (#1344; bounded_stop.md §6) ─────────────────────────────────────────
# A blocked ``input()`` or Rich prompt on the main thread wakes only by a raise (PEP 475 retries the read after a
# handler returns), so a library marks such a read ``interruptible()`` and the CLI's handler (``maxim.cli_signals``,
# the only handler) raises KeyboardInterrupt there. Plain globals, read lock-free by that handler. A region wraps the
# blocking read ONLY, never a save; after catching the interrupt the caller checks ``stop_noted()``.
_interruptible_depth = 0
_signal_noted = False


def note_signal() -> None:
    """The CLI's handler saw a stop signal (lock-free; the request itself lands a moment later, on its stop thread)."""
    global _signal_noted
    _signal_noted = True


def forget_noted_signal() -> None:
    """A new install of the CLI's handling starts with no signal noted."""
    global _signal_noted
    _signal_noted = False


def stop_noted() -> bool:
    """A stop signal arrived or a stop was requested: what a caller checks right after catching an interrupt."""
    return _signal_noted or _stop_request is not None


def interruptible_depth() -> int:
    return _interruptible_depth


@contextmanager
def interruptible() -> Iterator[None]:
    """Mark the enclosed blocking human-input read as one a stop signal may interrupt. On the main thread only (a
    handler runs there); elsewhere a no-op. Entered after a stop was noted, it raises at once."""
    global _interruptible_depth
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    _interruptible_depth += 1  # first: a signal landing after the check below finds the region open, and raises
    try:
        if stop_noted():
            raise KeyboardInterrupt()
        yield
    finally:
        _interruptible_depth -= 1


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


def reset_unless_stopped(reset: Callable[[], None]) -> bool:
    """Run ``reset`` only while no stop is requested; whether it ran (#1438 PR 1c).

    A reset re-arms work a stop request stopped (the backend's ``reset_shutdown``), so one made under a live request
    would undo it. Checked and run under the hook lock, so no request can land between the check and the reset; the
    request's own hooks run under the same lock, after. ``reset`` must not take the hook lock."""
    with _hook_lock:
        stopped = _stop_request is not None
        if not stopped:
            reset()
    if stopped:
        logger.info("A reset was refused: a process stop is requested")
    return not stopped


# ── The stop module's kill list: the process groups this process spawned (#1344 PR 3a; bounded_stop.md §7) ─────────
# Not cooperative cancellation: the children a forced stop must take with it. An LLM server or a tunnel daemon is
# registered from the moment its Popen returns (so a kill during a server's readiness wait reaches it) until its
# owner's ``stop()`` has reaped it. ``cli_signals._die`` reads the tuple lock-free (it must not take locks) and signals
# each live group; every other stop goes through the owner's ``stop``, the one claimer that reaps. Nothing here ever
# signals a process this process did not spawn: only a real ``subprocess.Popen`` whose leader is not yet reaped.


class ChildGroup(NamedTuple):
    """A spawned child that leads its own process group (``start_new_session``), its kind, and its owner's stop."""

    proc: Any
    kind: str  # "llm_server" | "tunnel"
    stop: Callable[[], None]


_children_lock = threading.Lock()
_children: tuple[ChildGroup, ...] = ()
_children_closed = False


def spawn_registered(launch: Callable[[], Any], *, kind: str, stop: Callable[[], None]) -> Any | None:
    """Run ``launch`` (a spawner's ``subprocess.Popen(...)``) and register what it returns, as one step under the
    registry lock; None, without launching, once the registry is closed (a restart is about to re-exec). So no child
    can be started after ``close_child_registry`` and miss its sweep. ``launch``'s own exception propagates."""
    global _children
    with _children_lock:
        if _children_closed:
            return None
        proc = launch()
        _children = (*_children, ChildGroup(proc, kind, stop))
    return proc


def unregister_child(proc: Any) -> None:
    """Drop ``proc``'s entry (its owner reaped it). Works on a closed registry."""
    global _children
    with _children_lock:
        _children = tuple(child for child in _children if child.proc is not proc)


def child_groups(kind: str | None = None) -> tuple[ChildGroup, ...]:
    """The registered children (of ``kind``), read lock-free: the tuple is rebound, never mutated."""
    children = _children
    return children if kind is None else tuple(child for child in children if child.kind == kind)


def close_child_registry() -> None:
    """Refuse every later launch (``spawn_registered`` returns None). Waits for a launch in flight to register first,
    so a sweep after this sees it. Never reopened: the caller is about to re-exec."""
    global _children_closed
    with _children_lock:
        _children_closed = True


def stop_children(kind: str | None = None) -> None:
    """Stop every registered child (of ``kind``) through its owner's ``stop``. The entries are copied under the lock
    and stopped after it is released: a ``stop`` unregisters (taking the lock), and a ``start`` holding its spawner's
    lock may be waiting on this one. A failed stop is logged; the next child is still stopped."""
    with _children_lock:
        children = [child for child in _children if kind is None or child.kind == kind]
    for child in children:
        try:
            child.stop()
        except Exception:
            logger.warning(
                "Could not stop a spawned %s (pid %s)", child.kind, getattr(child.proc, "pid", "?"), exc_info=True
            )


def signal_child_group(proc: Any, sig: int) -> bool:
    """Send ``sig`` to ``proc``'s process group; whether it was sent. POSIX only. Lock-free and log-free (the stop
    watchdog calls it on its way to ``os._exit``). Only a real ``subprocess.Popen`` (bound at import, so a test's patch
    of ``subprocess.Popen`` cannot pass), a pid > 1 that is not our own group, and a leader not yet reaped: an unreaped
    leader holds its pgid, so the group cannot be a stranger's. (A reap in another thread sets ``returncode`` a few
    bytecodes after ``waitpid``: the one window left, which also needs the pid to be reused.)"""
    if type(proc) is not _REAL_POPEN or not hasattr(os, "killpg"):
        return False
    pid = proc.pid
    if not isinstance(pid, int) or pid <= 1 or proc.returncode is not None or pid == os.getpgrp():
        return False
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):  # the group is gone (or, after the window above, not ours)
        return False
    return True


def kill_child_groups() -> None:
    """SIGKILL every registered child's group, lock-free: the forced stop's last act before ``os._exit``. POSIX."""
    if not hasattr(signal, "SIGKILL"):
        return
    for child in _children:
        signal_child_group(child.proc, signal.SIGKILL)


def _reset_children_for_tests() -> None:
    global _children, _children_lock, _children_closed
    _children_lock = threading.Lock()
    _children = ()
    _children_closed = False


if hasattr(os, "register_at_fork"):
    # A forked child (a multiprocessing or DataLoader worker) spawned none of these; a fresh lock, as one held by
    # another thread at the fork would never be released in the child.
    os.register_at_fork(after_in_child=_reset_children_for_tests)
