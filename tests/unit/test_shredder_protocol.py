"""Unit tests for ShredderSegmenterProtocol."""

from __future__ import annotations

import shutil
from unittest.mock import Mock, patch

import pytest

from maxim.skills.base import SkillResult, SkillState
from maxim.skills.protocols.shredder_segmenter import ShredderSegmenterProtocol


BRIDGE_PATH = "maxim.tools.rtsp_bridge.RTSPBridge"
BRIDGE_CONFIG_PATH = "maxim.tools.rtsp_bridge.RTSPBridgeConfig"


@pytest.fixture
def protocol():
    return ShredderSegmenterProtocol(
        rtsp_url="rtsp://localhost:8554/test",
        fps=15,
        yaw_range=25.0,
        pitch_range=15.0,
    )


@pytest.fixture
def protocol_with_api():
    return ShredderSegmenterProtocol(
        shredder_api_url="http://localhost:8000",
        shredder_license_id="test-license",
        shredder_api_key="test-key",
        shredder_camera_name="test-cam",
        shredder_site_id="test-site",
    )


class TestShredderSegmenterProperties:
    def test_name(self, protocol):
        assert protocol.name == "shredder_segmenter"

    def test_description(self, protocol):
        assert "RTSP" in protocol.description
        assert "ShredderSegmenter" in protocol.description

    def test_skills_returns_rtsp(self, protocol):
        skills = protocol.skills()
        assert len(skills) >= 1
        assert skills[0].name == "rtsp_streaming"

    def test_workspace_bounds(self, protocol):
        bounds = protocol.workspace_bounds()
        assert bounds.yaw == 25.0
        assert bounds.pitch == 15.0


class TestShredderSegmenterPhrases:
    def test_activation_phrases(self, protocol):
        phrases = protocol.phrases()
        assert len(phrases) > 0
        assert any("shredder" in p for p in phrases)
        assert any("start" in p or "run" in p for p in phrases)

    def test_stop_phrases(self, protocol):
        phrases = protocol.stop_phrases()
        assert len(phrases) > 0
        assert any("stop" in p for p in phrases)


class TestShredderSegmenterFailureHandling:
    def test_rtsp_failure_aborts(self, protocol):
        skill = Mock()
        skill.name = "rtsp_streaming"
        result = SkillResult(state=SkillState.FAILED, error="ffmpeg crashed")
        assert protocol.on_skill_failed(skill, result) == "abort"

    def test_other_skill_failure_continues(self, protocol):
        """Non-RTSP skills (timer, health) are optional — continue on failure."""
        skill = Mock()
        skill.name = "something_else"
        result = SkillResult(state=SkillState.FAILED)
        assert protocol.on_skill_failed(skill, result) == "continue"


class TestShredderSegmenterAPIRegistration:
    def test_no_api_configured_skips_registration(self, protocol):
        """Without API config, on_activate should not attempt registration."""
        mock_maxim = Mock()
        mock_maxim.mini = Mock()

        with patch(BRIDGE_PATH) as MockBridge, \
             patch(BRIDGE_CONFIG_PATH), \
             patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            MockBridge.return_value.is_running = True
            protocol.on_activate(mock_maxim)

        # No API call should have been made
        assert protocol._registered_camera_id is None

    @patch("urllib.request.urlopen")
    def test_api_registration_success(self, mock_urlopen, protocol_with_api):
        """With API config, on_activate should register camera."""
        import json

        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"id": "cam-123"}).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        mock_maxim = Mock()
        mock_maxim.mini = Mock()

        with patch(BRIDGE_PATH) as MockBridge, \
             patch(BRIDGE_CONFIG_PATH), \
             patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            MockBridge.return_value.is_running = True
            protocol_with_api.on_activate(mock_maxim)

        assert protocol_with_api._registered_camera_id == "cam-123"

    @patch("urllib.request.urlopen")
    def test_api_unregistration_on_deactivate(self, mock_urlopen, protocol_with_api):
        """Deactivation should unregister camera from API."""
        import json

        # Setup: register first
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"id": "cam-123"}).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        mock_maxim = Mock()
        mock_maxim.mini = Mock()

        with patch(BRIDGE_PATH) as MockBridge, \
             patch(BRIDGE_CONFIG_PATH), \
             patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            MockBridge.return_value.is_running = True
            protocol_with_api.on_activate(mock_maxim)

        assert protocol_with_api._registered_camera_id == "cam-123"

        # Now deactivate
        mock_urlopen.reset_mock()
        protocol_with_api.on_deactivate()

        # Should have called DELETE
        assert protocol_with_api._registered_camera_id is None
        mock_urlopen.assert_called()
