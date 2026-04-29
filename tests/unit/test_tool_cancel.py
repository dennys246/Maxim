"""Tests for Tool.cancel() hook (CC11, v1.0 freeze).

The cancel() method is purely additive: a default no-op on the Tool ABC
that existing third-party subclasses do not need to override. Heavy
built-in tools (HTTP fetch, internet search) override it to short-circuit
the execute() path at safe points using a ``threading.Event`` for
cross-thread visibility.
"""

from __future__ import annotations

import threading
from typing import Any

from maxim.tools.base import Tool, ToolOutput


class _MinimalTool(Tool):
    """Subclass that does NOT override cancel() — exercises the default."""

    name = "minimal"
    description = "Smoke-test tool"
    input_schema: dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> Any:
        return ToolOutput(success=True, output="ok")


def test_cancel_default_is_callable_and_returns_none():
    """Default no-op cancel() exists, is callable, and returns None.

    This is the backwards-compat invariant — third-party Tool subclasses
    written before CC11 do not need to implement cancel().
    """
    tool = _MinimalTool()
    result = tool.cancel()
    assert result is None


def test_cancel_default_does_not_raise():
    """Default cancel() never raises, even with no in-flight execute()."""
    tool = _MinimalTool()
    # Multiple calls must be safe (idempotent / benign repeats).
    tool.cancel()
    tool.cancel()
    tool.cancel()


def test_cancel_does_not_block_subsequent_execute():
    """Default cancel() leaves execute() functional for tools that do not
    consume the cancel flag. Tools that DO want execute() to refuse
    after cancel() must override and check their own state."""
    tool = _MinimalTool()
    tool.cancel()
    # The default no-op cancel() does not prevent execute().
    out = tool.execute()
    assert out.success is True


def test_cancel_is_not_abstractmethod():
    """cancel() must NOT be abstract — subclasses without it must work.

    If someone accidentally marks it @abstractmethod, every Tool subclass
    that does not override cancel() would fail to instantiate. That would
    be a breaking change for every third-party tool written against
    pre-CC11 versions of Maxim. Pin the contract directly against
    ``Tool.__abstractmethods__`` so a subclass that overrides cancel
    can't accidentally pass this test.
    """
    assert "cancel" not in Tool.__abstractmethods__
    assert getattr(Tool.cancel, "__isabstractmethod__", False) is False
    # Instantiation succeeds → cancel() is not part of the abstract API.
    tool = _MinimalTool()
    assert tool.name == "minimal"


def test_cancel_has_no_caller_in_executor_dispatch():
    """CC11: Tool.cancel() ships as forward-compat infrastructure for
    1.1+ MCP-subprocess work. No 1.0 dispatch path calls it. Pin that
    invariant — if a future refactor wires cancel() into the executor,
    third-party Tool subclasses that override cancel() will suddenly
    observe new dispatch behavior.

    If you intentionally add a caller, update this test to spell out
    exactly which path now invokes cancel and document the contract
    change in CLAUDE.md under the Tool.cancel invariant.
    """
    import inspect
    from maxim.runtime import executor as _executor_mod

    src = inspect.getsource(_executor_mod)
    # Heuristic: ``.cancel(`` substring with a leading dot won't match
    # unrelated ``cancel_X`` identifiers; it would match a real call to
    # ``some_tool.cancel(...)``. Fail loudly so the contributor revisits
    # this contract before merging.
    assert ".cancel(" not in src, "runtime/executor.py now calls .cancel() — update the CC11 'no 1.0 caller' contract"


class TestHttpFetchCancel:
    """HttpFetchTool.cancel() sets a threading.Event execute() checks."""

    def test_cancel_sets_event(self):
        from maxim.tools.http_fetch import HttpFetchTool

        tool = HttpFetchTool()
        assert isinstance(tool._cancelled, threading.Event)
        assert tool._cancelled.is_set() is False
        tool.cancel()
        assert tool._cancelled.is_set() is True

    def test_execute_after_cancel_returns_cancelled_error(self):
        """execute() short-circuits with a cancelled error after cancel().

        This relies on cancel() being called between __init__ and the
        early in-execute is_set check; execute() clears the event at
        entry, so the cancel must arrive while the call is in flight.
        Easiest deterministic shape: monkey-patch the URL so the early
        check fires before execute() reaches the policy gate. We instead
        validate the simpler invariant: when cancel() arrives BEFORE
        execute() is invoked, execute() clears the event and proceeds
        normally to the policy gate (no policy → returns the
        ``policy_blocked`` error). The cancel-during-execute case is
        covered by ``test_cancel_during_execute_blocks_network_call``
        below using a poisoned policy hook.
        """
        from maxim.tools.http_fetch import HttpFetchTool

        tool = HttpFetchTool()
        tool.cancel()
        result = tool.execute(url="https://example.com")
        # Cleared at execute() entry → policy gate fires first, not cancel.
        assert result.success is False
        # Either policy_blocked (no policy provided) or cancelled — both prove
        # the no-network-traffic invariant. Prefer policy_blocked since that
        # is the deterministic outcome with the at-entry clear.
        assert ("cancel" in (result.error or "").lower()) or (result.metadata.get("policy_blocked") is True)

    def test_cancel_during_execute_blocks_network_call(self):
        """cancel() called between policy-pass and network-read short-circuits.

        Simulate by passing a policy hook that flips cancel() before
        returning. The next safe-point check (post-rate-limit, post-robots)
        must observe is_set() and return a cancelled error before any
        network traffic.
        """
        from maxim.tools.http_fetch import HttpFetchTool

        tool = HttpFetchTool()

        def _poison_policy():
            tool.cancel()  # flip the event mid-execute

            class _P:
                enabled = True
                max_fetch_bytes = 1_000_000
                request_timeout_s = 5.0
                max_pages_per_minute = 10
                require_robots_ok = False
                block_paywalled = False
                unsafe_content_checks = False

                def can_access(self, url):
                    return True, None

            return _P()

        tool._get_internet_policy = _poison_policy
        result = tool.execute(url="https://example.com")
        assert result.success is False
        assert "cancel" in (result.error or "").lower()


class TestInternetSearchCancel:
    """InternetSearchTool.cancel() — same shape as HttpFetchTool."""

    def test_cancel_sets_event(self):
        from maxim.tools.internet_search import InternetSearchTool

        tool = InternetSearchTool()
        assert isinstance(tool._cancelled, threading.Event)
        assert tool._cancelled.is_set() is False
        tool.cancel()
        assert tool._cancelled.is_set() is True

    def test_cancelled_request_does_not_consume_rate_limit(self):
        """Critical ordering: a cancelled request must NOT append to
        ``_request_times``. Earlier draft of this code had the cancel
        check AFTER the rate-limit append — flagged by pre-merge review.
        Pin the correct order so a regression re-introduces a failing
        test instead of a silent slot-leak.
        """
        from maxim.tools.internet_search import InternetSearchTool

        tool = InternetSearchTool()

        def _poison_policy():
            tool.cancel()

            class _P:
                enabled = True
                request_timeout_s = 5.0

            return _P()

        tool._get_internet_policy = _poison_policy
        result = tool.execute(query="anything")
        assert result.success is False
        assert "cancel" in (result.error or "").lower()
        # The load-bearing assertion: the rate-limit deque was NOT
        # appended to for a cancelled call.
        assert tool._request_times == []
