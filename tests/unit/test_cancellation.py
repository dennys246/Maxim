"""Regression tests for maxim.utils.cancellation.

The load-bearing invariant: the ContextVar's value MUST be a mutable
reference (threading.Event), not a bool. If someone ever "simplifies"
it to ContextVar[bool], the cross-thread propagation breaks silently —
ThreadPoolExecutor captures the context at submit time, so a bool set
in the parent thread after submission would have no effect on the
child's snapshot.

These tests exist specifically to catch that regression. If they pass,
cross-thread cancellation signalling works. If they fail, the
cancellation hygiene fix (Plan 3.5) is broken.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time

from maxim.utils.cancellation import (
    current_cancel_event,
    is_cancelled,
    reset_cancel_event,
    set_cancel_event,
)


def test_default_is_none():
    """With no binding, current_cancel_event returns None and is_cancelled is False."""
    assert current_cancel_event() is None
    assert is_cancelled() is False


def test_set_and_reset():
    """Binding an Event exposes it via current_cancel_event; reset clears it."""
    event = threading.Event()
    token = set_cancel_event(event)
    try:
        assert current_cancel_event() is event
        assert is_cancelled() is False
        event.set()
        assert is_cancelled() is True
    finally:
        reset_cancel_event(token)
    assert current_cancel_event() is None


def test_naive_threadpoolexecutor_submit_does_NOT_propagate():
    """**Documented Python gotcha.** ``ThreadPoolExecutor.submit(worker)``
    WITHOUT explicit context propagation does NOT carry the parent's
    ContextVar bindings into the child thread. The child sees the
    default value (``None``).

    This test exists so that if a future refactor assumes naive submit
    "just works" with ContextVars, the breakage is loud and obvious.
    The CORRECT pattern is
    ``executor.submit(contextvars.copy_context().run, worker)`` — see
    :func:`test_contextvars_propagate_with_copy_context`.
    """
    import contextvars

    event = threading.Event()
    token = set_cancel_event(event)
    captured: list[threading.Event | None] = [None]

    def worker() -> None:
        # Intentionally naive — no copy_context wrapper
        captured[0] = current_cancel_event()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(worker).result(timeout=2.0)
        # Confirms the gotcha: child did NOT inherit the parent's binding.
        assert captured[0] is None, (
            "Python's ThreadPoolExecutor now propagates ContextVars automatically! "
            "Good news — update maxim.utils.cancellation to drop the copy_context() "
            "wrapper in _call_llm_with_timeout."
        )
    finally:
        reset_cancel_event(token)
    # Belt-and-suspenders: verify contextvars is importable for the
    # sibling test's recommended pattern.
    assert contextvars is not None


def test_contextvars_propagate_with_copy_context():
    """**The correct usage pattern.** ``contextvars.copy_context().run(worker)``
    snapshots the parent's context and invokes ``worker`` inside it.
    Wrapping the executor submit this way carries the cancellation
    binding through to the child thread.

    Pattern::

        import contextvars
        ctx = contextvars.copy_context()
        future = executor.submit(ctx.run, worker_fn, *args)

    The Event itself is a mutable shared reference, so once the child
    has it, ``event.set()`` in the parent is visible to the child.
    """
    import contextvars

    event = threading.Event()
    token = set_cancel_event(event)
    child_saw_cancellation = threading.Event()
    child_started = threading.Event()

    def worker() -> bool:
        child_started.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if is_cancelled():
                child_saw_cancellation.set()
                return True
            time.sleep(0.01)
        return False

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            ctx = contextvars.copy_context()
            future = executor.submit(ctx.run, worker)
            assert child_started.wait(timeout=2.0), "child thread never started"
            # Parent thread signals cancellation AFTER submit
            event.set()
            result = future.result(timeout=3.0)
            assert result is True, "child thread did not observe cancellation"
            assert child_saw_cancellation.is_set()
    finally:
        reset_cancel_event(token)


def test_event_is_shared_by_reference_with_copy_context():
    """Proves that ``copy_context()`` captures the Event by reference,
    not by value. Both threads hold the same object, so ``event.set()``
    in one is visible to the other.
    """
    import contextvars

    event = threading.Event()
    token = set_cancel_event(event)
    captured: list[threading.Event | None] = [None]

    def worker() -> None:
        captured[0] = current_cancel_event()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            ctx = contextvars.copy_context()
            executor.submit(ctx.run, worker).result(timeout=2.0)
        # Same object reference, not a copy
        assert captured[0] is event
    finally:
        reset_cancel_event(token)


def test_nested_contexts_isolate():
    """Nested set/reset must not leak bindings to sibling contexts.

    Tests that each call to set_cancel_event is properly scoped and
    the previous binding is restored on reset.
    """
    outer_event = threading.Event()
    outer_token = set_cancel_event(outer_event)
    try:
        assert current_cancel_event() is outer_event

        inner_event = threading.Event()
        inner_token = set_cancel_event(inner_event)
        try:
            assert current_cancel_event() is inner_event
        finally:
            reset_cancel_event(inner_token)

        # After inner reset, the outer binding is restored
        assert current_cancel_event() is outer_event
    finally:
        reset_cancel_event(outer_token)
    assert current_cancel_event() is None


def test_unset_event_is_not_cancelled():
    """An Event that exists but hasn't been set means 'cancellable but not
    yet cancelled'. is_cancelled must return False in this state.
    """
    event = threading.Event()
    token = set_cancel_event(event)
    try:
        assert current_cancel_event() is event
        assert is_cancelled() is False
        # Setting flips it
        event.set()
        assert is_cancelled() is True
        # Clearing flips it back
        event.clear()
        assert is_cancelled() is False
    finally:
        reset_cancel_event(token)
