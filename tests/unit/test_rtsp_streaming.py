"""Unit tests for RTSPStreamingSkill."""

from __future__ import annotations

import shutil
from typing import Any
from unittest.mock import Mock, patch, MagicMock

import pytest

from maxim.skills.base import SkillResult, SkillState
from maxim.skills.rtsp_streaming import RTSPStreamingConfig, RTSPStreamingSkill


@pytest.fixture
def config():
    return RTSPStreamingConfig(rtsp_url="rtsp://localhost:8554/test", fps=15)


@pytest.fixture
def skill(config):
    return RTSPStreamingSkill(config)


@pytest.fixture
def mock_maxim():
    m = Mock()
    m.mini = Mock()
    m.mini.media = Mock()
    return m


# Patch target: the import happens inside activate() via
#   from maxim.tools.rtsp_bridge import RTSPBridge, RTSPBridgeConfig
# so we patch at the source module.
BRIDGE_PATH = "maxim.tools.rtsp_bridge.RTSPBridge"
BRIDGE_CONFIG_PATH = "maxim.tools.rtsp_bridge.RTSPBridgeConfig"


class TestRTSPStreamingConfig:
    def test_defaults(self):
        cfg = RTSPStreamingConfig()
        assert cfg.rtsp_url == "rtsp://localhost:8554/reachy"
        assert cfg.fps == 20
        assert cfg.preset == "ultrafast"
        assert cfg.tune == "zerolatency"

    def test_custom_config(self, config):
        assert config.rtsp_url == "rtsp://localhost:8554/test"
        assert config.fps == 15


class TestRTSPStreamingSkillProperties:
    def test_name(self, skill):
        assert skill.name == "rtsp_streaming"

    def test_description(self, skill):
        assert "RTSP" in skill.description

    def test_initial_state(self, skill):
        assert skill.state == SkillState.IDLE
        assert skill.is_active is False

    def test_tools_returns_two(self, skill):
        tools = skill.tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "start_rtsp_stream" in names
        assert "stop_rtsp_stream" in names


class TestRTSPStreamingPreconditions:
    def test_can_activate_with_ffmpeg_and_robot(self, skill, mock_maxim):
        with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            ok, reason = skill.can_activate(mock_maxim)
        assert ok is True

    def test_cannot_activate_without_ffmpeg(self, skill, mock_maxim):
        with patch.object(shutil, "which", return_value=None):
            ok, reason = skill.can_activate(mock_maxim)
        assert ok is False
        assert "ffmpeg" in reason

    def test_cannot_activate_without_robot(self, skill):
        maxim = Mock(spec=[])  # no .mini
        with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            ok, reason = skill.can_activate(maxim)
        assert ok is False
        assert "robot" in reason.lower() or "camera" in reason.lower()


class TestRTSPStreamingActivation:
    @patch(BRIDGE_PATH)
    @patch(BRIDGE_CONFIG_PATH)
    def test_activate_success(self, MockConfig, MockBridge, skill, mock_maxim):
        bridge_instance = MockBridge.return_value
        bridge_instance.is_running = True

        result = skill.activate(mock_maxim)

        assert result.ok
        assert skill.state == SkillState.ACTIVE
        assert "rtsp://localhost:8554/test" in result.message

    @patch(BRIDGE_PATH)
    @patch(BRIDGE_CONFIG_PATH)
    def test_activate_failure(self, MockConfig, MockBridge, skill, mock_maxim):
        bridge_instance = MockBridge.return_value
        bridge_instance.is_running = False

        result = skill.activate(mock_maxim)

        assert not result.ok
        assert skill.state == SkillState.FAILED

    @patch(BRIDGE_PATH)
    @patch(BRIDGE_CONFIG_PATH)
    def test_activate_writes_shared_context(self, MockConfig, MockBridge, skill, mock_maxim):
        bridge_instance = MockBridge.return_value
        bridge_instance.is_running = True

        ctx = {}
        skill.activate(mock_maxim, context=ctx)

        assert ctx["rtsp_url"] == "rtsp://localhost:8554/test"
        assert ctx["rtsp_fps"] == 15


class TestRTSPStreamingDeactivation:
    @patch(BRIDGE_PATH)
    @patch(BRIDGE_CONFIG_PATH)
    def test_deactivate(self, MockConfig, MockBridge, skill, mock_maxim):
        bridge_instance = MockBridge.return_value
        bridge_instance.is_running = True
        skill.activate(mock_maxim)

        result = skill.deactivate()

        assert result.ok
        assert skill.state == SkillState.IDLE
        bridge_instance.stop.assert_called_once()


class TestRTSPStreamingRuntimeFailure:
    @patch(BRIDGE_PATH)
    @patch(BRIDGE_CONFIG_PATH)
    def test_runtime_failure_detected(self, MockConfig, MockBridge, skill, mock_maxim):
        bridge_instance = MockBridge.return_value
        bridge_instance.is_running = True
        skill.activate(mock_maxim)
        assert skill.state == SkillState.ACTIVE

        # Bridge dies
        bridge_instance.is_running = False
        assert skill.state == SkillState.FAILED


class TestRTSPStreamingContext:
    @patch(BRIDGE_PATH)
    @patch(BRIDGE_CONFIG_PATH)
    def test_context_when_active(self, MockConfig, MockBridge, skill, mock_maxim):
        bridge_instance = MockBridge.return_value
        bridge_instance.is_running = True
        skill.activate(mock_maxim)

        ctx = skill.context_for_llm()
        assert "rtsp://localhost:8554/test" in ctx
        assert "15" in ctx

    def test_context_when_idle(self, skill):
        ctx = skill.context_for_llm()
        assert "not active" in ctx.lower()


class TestRTSPStreamingHealth:
    @patch(BRIDGE_PATH)
    @patch(BRIDGE_CONFIG_PATH)
    def test_health_includes_config(self, MockConfig, MockBridge, skill, mock_maxim):
        bridge_instance = MockBridge.return_value
        bridge_instance.is_running = True
        skill.activate(mock_maxim)

        h = skill.health()
        assert h["rtsp_url"] == "rtsp://localhost:8554/test"
        assert h["fps"] == 15
