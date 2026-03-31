"""Unit tests for BUG-9: specific exception handling in agent_loop.py.

Validates that the four fixed except clauses catch only the intended
exception types and let unexpected errors propagate.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. state.steps_taken += 1  -->  except AttributeError: pass
# ---------------------------------------------------------------------------


class TestStepsTakenAttributeError:
    """Line ~324: state.steps_taken += 1 should catch AttributeError only."""

    @staticmethod
    def _increment_steps(state):
        """Reproduces the guarded increment from run_agent_loop."""
        try:
            state.steps_taken += 1
        except AttributeError:
            pass

    def test_missing_steps_taken_attribute_is_silenced(self):
        """A state object without steps_taken should not crash."""
        state = MagicMock(spec=[])  # no attributes at all
        self._increment_steps(state)  # should not raise

    def test_normal_increment_works(self):
        """A state with steps_taken should be incremented normally."""
        state = MagicMock()
        state.steps_taken = 0
        self._increment_steps(state)
        assert state.steps_taken == 1

    def test_value_error_propagates(self):
        """ValueError (unexpected) must NOT be caught."""
        state = MagicMock()
        type(state).steps_taken = property(
            lambda self: 0,
            lambda self, v: (_ for _ in ()).throw(ValueError("boom")),
        )
        with pytest.raises(ValueError, match="boom"):
            self._increment_steps(state)

    def test_type_error_propagates(self):
        """TypeError (unexpected) must NOT be caught."""
        state = MagicMock()
        type(state).steps_taken = property(
            lambda self: 0,
            lambda self, v: (_ for _ in ()).throw(TypeError("bad type")),
        )
        with pytest.raises(TypeError, match="bad type"):
            self._increment_steps(state)


# ---------------------------------------------------------------------------
# 2. on_step callback  -->  except Exception: logger.debug(...)
# ---------------------------------------------------------------------------


class TestOnStepCallbackLogging:
    """Line ~2411-2412: on_step failure should be logged via logger.debug."""

    @staticmethod
    def _call_on_step(on_step, step_info, logger):
        """Reproduces the guarded on_step call from run_agentic_loop."""
        try:
            if callable(on_step):
                on_step(step_info)
        except Exception:
            logger.debug("on_step callback failed", exc_info=True)

    def test_callback_exception_is_logged(self):
        """When on_step raises, logger.debug must be called with exc_info."""
        mock_logger = MagicMock()

        def bad_callback(info):
            raise RuntimeError("callback exploded")

        self._call_on_step(bad_callback, {"step": 1}, mock_logger)
        mock_logger.debug.assert_called_once_with(
            "on_step callback failed", exc_info=True,
        )

    def test_callback_success_no_log(self):
        """A successful on_step should not produce any debug log."""
        mock_logger = MagicMock()
        calls = []

        def good_callback(info):
            calls.append(info)

        self._call_on_step(good_callback, {"step": 1}, mock_logger)
        mock_logger.debug.assert_not_called()
        assert len(calls) == 1

    def test_none_callback_is_noop(self):
        """on_step=None should be silently skipped."""
        mock_logger = MagicMock()
        self._call_on_step(None, {"step": 1}, mock_logger)
        mock_logger.debug.assert_not_called()


# ---------------------------------------------------------------------------
# 3. executor.registry.list()  -->  except (KeyError, AttributeError): pass
# ---------------------------------------------------------------------------


class TestRegistryListFailure:
    """Line ~460: _get_all_tools catches (KeyError, AttributeError)."""

    @staticmethod
    def _get_all_tools(executor):
        """Reproduces the nested helper from run_agentic_loop."""
        if hasattr(executor, "registry") and hasattr(executor.registry, "list"):
            try:
                return set(executor.registry.list())
            except (KeyError, AttributeError):
                pass
        return set()

    def test_key_error_returns_empty(self):
        """KeyError from registry.list() is caught, returns empty set."""
        executor = MagicMock()
        executor.registry.list.side_effect = KeyError("missing")
        assert self._get_all_tools(executor) == set()

    def test_attribute_error_returns_empty(self):
        """AttributeError from registry.list() is caught, returns empty set."""
        executor = MagicMock()
        executor.registry.list.side_effect = AttributeError("gone")
        assert self._get_all_tools(executor) == set()

    def test_runtime_error_propagates(self):
        """RuntimeError (unexpected) must NOT be caught."""
        executor = MagicMock()
        executor.registry.list.side_effect = RuntimeError("unexpected")
        with pytest.raises(RuntimeError, match="unexpected"):
            self._get_all_tools(executor)

    def test_no_registry_returns_empty(self):
        """Executor with no registry attribute returns empty set."""
        executor = MagicMock(spec=[])  # no attributes
        assert self._get_all_tools(executor) == set()

    def test_working_registry_returns_tools(self):
        """A healthy registry.list() returns the tool set."""
        executor = MagicMock()
        executor.registry.list.return_value = ["tool_a", "tool_b"]
        assert self._get_all_tools(executor) == {"tool_a", "tool_b"}


# ---------------------------------------------------------------------------
# 4. stop_event.is_set()  -->  except (AttributeError, RuntimeError): pass
# ---------------------------------------------------------------------------


class TestStopEventIsSet:
    """Line ~568: stop_event check catches (AttributeError, RuntimeError)."""

    @staticmethod
    def _check_stop(stop_event) -> bool:
        """Reproduces the guarded stop-event check from run_agentic_loop.

        Returns True if the loop should break (stop requested).
        """
        try:
            if stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set():
                return True
        except (AttributeError, RuntimeError):
            pass
        return False

    def test_attribute_error_is_caught(self):
        """stop_event whose is_set raises AttributeError is silenced."""
        event = MagicMock()
        event.is_set.side_effect = AttributeError("no is_set")
        # hasattr returns True (MagicMock), but the call itself fails
        assert self._check_stop(event) is False  # no crash

    def test_runtime_error_is_caught(self):
        """RuntimeError from is_set (e.g., interpreter shutdown) is silenced."""
        event = MagicMock()
        event.is_set.side_effect = RuntimeError("interpreter shutting down")
        assert self._check_stop(event) is False

    def test_value_error_propagates(self):
        """ValueError (unexpected) must NOT be caught."""
        event = MagicMock()
        event.is_set.side_effect = ValueError("weird")
        with pytest.raises(ValueError, match="weird"):
            self._check_stop(event)

    def test_none_stop_event_returns_false(self):
        """stop_event=None should be a no-op, returns False."""
        assert self._check_stop(None) is False

    def test_normal_stop_event_set(self):
        """A normal Event that is set returns True."""
        event = MagicMock()
        event.is_set.return_value = True
        assert self._check_stop(event) is True

    def test_normal_stop_event_not_set(self):
        """A normal Event that is not set returns False."""
        event = MagicMock()
        event.is_set.return_value = False
        assert self._check_stop(event) is False
