"""Tests for ConnectionState enum and capability degradation."""

from __future__ import annotations

from unittest.mock import MagicMock


from maxim.embodied_runtime.connection import (
    ConnectionState,
)


class TestConnectionStateEnum:
    def test_all_states_exist(self):
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"
        assert ConnectionState.ERROR.value == "error"


class TestCapabilityDegradation:
    """ConnectionMixin._degrade_capabilities / _restore_capabilities."""

    def _make_mixin(self):
        from maxim.runtime.capabilities import RuntimeCapabilities
        from maxim.embodied_runtime.connection import ConnectionMixin

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
        from maxim.embodied_runtime.connection import ConnectionMixin

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
        from maxim.embodied_runtime.agentic_runtime import _compute_target_hz
        from maxim.runtime.capabilities import RuntimeCapabilities

        caps = RuntimeCapabilities(has_motor=True, has_vision=True)
        assert _compute_target_hz(caps) == 30.0

    def test_vision_only_gets_10hz(self):
        from maxim.embodied_runtime.agentic_runtime import _compute_target_hz
        from maxim.runtime.capabilities import RuntimeCapabilities

        caps = RuntimeCapabilities(has_motor=False, has_vision=True)
        assert _compute_target_hz(caps) == 10.0

    def test_headless_gets_2hz(self):
        from maxim.embodied_runtime.agentic_runtime import _compute_target_hz
        from maxim.runtime.capabilities import RuntimeCapabilities

        caps = RuntimeCapabilities(has_motor=False, has_vision=False)
        assert _compute_target_hz(caps) == 2.0
