"""Unit tests for TimedProtocolSkill."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from maxim.skills.base import SkillState
from maxim.skills.timed_protocol import TimedProtocolConfig, TimedProtocolSkill


@pytest.fixture
def config():
    return TimedProtocolConfig(duration_minutes=0.01)  # 0.6 seconds for fast tests


@pytest.fixture
def mock_maxim():
    m = Mock()
    m.log = Mock()
    m._protocol_registry = Mock()
    m._protocol_registry.deactivate = Mock(return_value="Protocol 'test' deactivated.")
    return m


@pytest.fixture
def skill(config):
    return TimedProtocolSkill(config)


class TestTimedProtocolConfig:
    def test_defaults(self):
        cfg = TimedProtocolConfig()
        assert cfg.duration_minutes == 60.0

    def test_custom(self, config):
        assert config.duration_minutes == 0.01


class TestTimedProtocolProperties:
    def test_name(self, skill):
        assert skill.name == "timed_protocol"

    def test_description_includes_duration(self, skill):
        assert "0.01" in skill.description

    def test_no_tools(self, skill):
        assert skill.tools() == []


class TestTimedProtocolPreconditions:
    def test_valid_duration(self, skill, mock_maxim):
        ok, reason = skill.can_activate(mock_maxim)
        assert ok is True

    def test_zero_duration_rejected(self, mock_maxim):
        skill = TimedProtocolSkill(TimedProtocolConfig(duration_minutes=0))
        ok, reason = skill.can_activate(mock_maxim)
        assert ok is False
        assert "positive" in reason

    def test_negative_duration_rejected(self, mock_maxim):
        skill = TimedProtocolSkill(TimedProtocolConfig(duration_minutes=-5))
        ok, reason = skill.can_activate(mock_maxim)
        assert ok is False


class TestTimedProtocolActivation:
    def test_activate_starts_timer(self, skill, mock_maxim):
        context = {"_protocol_name": "test_proto"}
        result = skill.activate(mock_maxim, context)

        assert result.ok
        assert result.state == SkillState.ACTIVE
        assert skill.state == SkillState.ACTIVE
        assert skill._timer is not None
        assert skill._timer.is_alive()
        assert "0.01" in result.message

        skill.deactivate()

    def test_activate_reads_protocol_name(self, skill, mock_maxim):
        context = {"_protocol_name": "my_proto"}
        skill.activate(mock_maxim, context)
        assert skill._protocol_name == "my_proto"
        skill.deactivate()

    def test_activate_handles_no_context(self, skill, mock_maxim):
        result = skill.activate(mock_maxim, None)
        assert result.ok
        assert skill._protocol_name is None
        skill.deactivate()

    def test_metadata_contains_duration(self, skill, mock_maxim):
        result = skill.activate(mock_maxim, {})
        assert result.metadata["duration_minutes"] == 0.01
        skill.deactivate()


class TestTimedProtocolDeactivation:
    def test_deactivate_cancels_timer(self, skill, mock_maxim):
        skill.activate(mock_maxim, {})
        result = skill.deactivate()

        assert result.state == SkillState.IDLE
        assert skill._timer is None
        assert skill._protocol_name is None

    def test_deactivate_before_activate(self, skill):
        result = skill.deactivate()
        assert result.state == SkillState.IDLE


class TestTimedProtocolTimerExpiry:
    def test_timer_calls_deactivate(self, mock_maxim):
        """Timer should call registry.deactivate() when it expires."""
        skill = TimedProtocolSkill(TimedProtocolConfig(duration_minutes=0.005))  # 0.3s
        context = {"_protocol_name": "test_proto"}
        skill.activate(mock_maxim, context)

        # Wait for timer to fire
        time.sleep(0.8)

        mock_maxim._protocol_registry.deactivate.assert_called_once_with("test_proto")

    def test_timer_no_registry_logs_warning(self, mock_maxim):
        """If registry is missing, timer should log warning, not crash."""
        mock_maxim._protocol_registry = None
        skill = TimedProtocolSkill(TimedProtocolConfig(duration_minutes=0.005))
        context = {"_protocol_name": "test_proto"}
        skill.activate(mock_maxim, context)

        time.sleep(0.8)
        # Should not raise — just logs warning

    def test_timer_no_protocol_name_logs_warning(self, mock_maxim):
        """If no protocol name in context, timer should log warning."""
        skill = TimedProtocolSkill(TimedProtocolConfig(duration_minutes=0.005))
        skill.activate(mock_maxim, {})  # No _protocol_name

        time.sleep(0.8)
        mock_maxim._protocol_registry.deactivate.assert_not_called()


class TestTimedProtocolContext:
    def test_context_for_llm_active(self, skill, mock_maxim):
        skill.activate(mock_maxim, {})
        ctx = skill.context_for_llm()
        assert "auto-stop" in ctx.lower() or "0.01" in ctx
        skill.deactivate()

    def test_context_for_llm_idle(self, skill):
        assert skill.context_for_llm() == ""


class TestTimedProtocolHealth:
    def test_health_includes_timer_info(self, skill, mock_maxim):
        skill.activate(mock_maxim, {})
        h = skill.health()
        assert h["name"] == "timed_protocol"
        assert h["state"] == "active"
        assert h["duration_minutes"] == 0.01
        assert h["timer_alive"] is True
        skill.deactivate()

    def test_health_idle(self, skill):
        h = skill.health()
        assert h["state"] == "idle"
        assert "timer_alive" not in h
