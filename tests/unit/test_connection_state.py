"""Tests for ConnectionState enum, state callbacks, and capability degradation."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from maxim.conscience.connection import (
    ConnectionConfig,
    ConnectionState,
    ReachyConnection,
)


class TestConnectionStateEnum:
    def test_all_states_exist(self):
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"
        assert ConnectionState.ERROR.value == "error"

    def test_initial_state_is_disconnected(self):
        conn = ReachyConnection(ConnectionConfig())
        assert conn.state == ConnectionState.DISCONNECTED


class TestConnectionStateCallbacks:
    def test_callback_fires_on_state_change(self):
        conn = ReachyConnection(ConnectionConfig())
        transitions = []
        conn.add_state_callback(lambda old, new: transitions.append((old, new)))

        conn._set_state(ConnectionState.CONNECTING)
        conn._set_state(ConnectionState.CONNECTED)

        assert transitions == [
            (ConnectionState.DISCONNECTED, ConnectionState.CONNECTING),
            (ConnectionState.CONNECTING, ConnectionState.CONNECTED),
        ]

    def test_no_callback_on_same_state(self):
        conn = ReachyConnection(ConnectionConfig())
        transitions = []
        conn.add_state_callback(lambda old, new: transitions.append((old, new)))

        conn._set_state(ConnectionState.DISCONNECTED)  # Same as initial
        assert transitions == []

    def test_multiple_callbacks(self):
        conn = ReachyConnection(ConnectionConfig())
        calls_a = []
        calls_b = []
        conn.add_state_callback(lambda o, n: calls_a.append(n))
        conn.add_state_callback(lambda o, n: calls_b.append(n))

        conn._set_state(ConnectionState.CONNECTING)
        assert calls_a == [ConnectionState.CONNECTING]
        assert calls_b == [ConnectionState.CONNECTING]

    def test_callback_error_doesnt_break_others(self):
        conn = ReachyConnection(ConnectionConfig())
        calls = []

        def bad_callback(old, new):
            raise RuntimeError("oops")

        conn.add_state_callback(bad_callback)
        conn.add_state_callback(lambda o, n: calls.append(n))

        conn._set_state(ConnectionState.CONNECTING)
        # Second callback should still fire despite first one raising
        assert calls == [ConnectionState.CONNECTING]


class TestCapabilityDegradation:
    """ConnectionMixin._degrade_capabilities / _restore_capabilities."""

    def _make_mixin(self):
        from maxim.runtime.capabilities import RuntimeCapabilities
        from maxim.conscience.connection import ConnectionMixin

        class FakeSelf(ConnectionMixin):
            def __init__(self):
                self._capabilities = RuntimeCapabilities(
                    has_robot=True,
                    has_motor=True,
                    has_vision=True,
                    has_audio=True,
                )
                self.log = MagicMock()
                # ConnectionMixin internals
                self._connection_failures = {}
                self._reconnect_thresholds = {}
                self._reconnect_lock = MagicMock()

        return FakeSelf()

    def test_degrade_clears_robot_capabilities(self):
        mixin = self._make_mixin()
        mixin._degrade_capabilities()
        assert mixin._capabilities.has_robot is False
        assert mixin._capabilities.has_motor is False
        assert mixin._capabilities.has_vision is False
        assert mixin._capabilities.has_audio is False

    def test_restore_sets_robot_capabilities(self):
        mixin = self._make_mixin()
        mixin._degrade_capabilities()
        mixin._restore_capabilities()
        assert mixin._capabilities.has_robot is True
        assert mixin._capabilities.has_motor is True
        assert mixin._capabilities.has_vision is True
        assert mixin._capabilities.has_audio is True

    def test_degrade_without_capabilities_is_safe(self):
        from maxim.conscience.connection import ConnectionMixin

        class NoCapsMixin(ConnectionMixin):
            def __init__(self):
                self.log = MagicMock()
                self._connection_failures = {}
                self._reconnect_thresholds = {}
                self._reconnect_lock = MagicMock()

        mixin = NoCapsMixin()
        # Should not raise even without _capabilities
        mixin._degrade_capabilities()


class TestAdaptiveHz:
    """_compute_target_hz adapts to capabilities."""

    def test_motor_gets_30hz(self):
        from maxim.conscience.agentic_runtime import _compute_target_hz
        from maxim.runtime.capabilities import RuntimeCapabilities

        caps = RuntimeCapabilities(has_motor=True, has_vision=True)
        assert _compute_target_hz(caps) == 30.0

    def test_vision_only_gets_10hz(self):
        from maxim.conscience.agentic_runtime import _compute_target_hz
        from maxim.runtime.capabilities import RuntimeCapabilities

        caps = RuntimeCapabilities(has_motor=False, has_vision=True)
        assert _compute_target_hz(caps) == 10.0

    def test_headless_gets_2hz(self):
        from maxim.conscience.agentic_runtime import _compute_target_hz
        from maxim.runtime.capabilities import RuntimeCapabilities

        caps = RuntimeCapabilities(has_motor=False, has_vision=False)
        assert _compute_target_hz(caps) == 2.0
