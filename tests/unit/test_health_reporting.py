"""Unit tests for HealthReportingSkill."""

from __future__ import annotations

import json
import time
from unittest.mock import Mock, patch

import pytest

from maxim.skills.base import SkillState
from maxim.skills.health_reporting import HealthReportingConfig, HealthReportingSkill


@pytest.fixture
def config():
    return HealthReportingConfig(
        endpoint_url="http://localhost:9999/health",
        interval_seconds=0.1,  # Fast for tests
        timeout_seconds=2.0,
    )


@pytest.fixture
def mock_maxim():
    m = Mock()
    m.log = Mock()
    m._protocol_registry = Mock()
    return m


@pytest.fixture
def skill(config):
    return HealthReportingSkill(config)


class TestHealthReportingConfig:
    def test_defaults(self):
        cfg = HealthReportingConfig()
        assert cfg.endpoint_url == ""
        assert cfg.interval_seconds == 30.0
        assert cfg.timeout_seconds == 5.0
        assert cfg.headers == {}

    def test_custom(self, config):
        assert config.endpoint_url == "http://localhost:9999/health"
        assert config.interval_seconds == 0.1

    def test_custom_headers(self):
        cfg = HealthReportingConfig(
            endpoint_url="http://example.com",
            headers={"Authorization": "Bearer tok123"},
        )
        assert cfg.headers["Authorization"] == "Bearer tok123"


class TestHealthReportingProperties:
    def test_name(self, skill):
        assert skill.name == "health_reporting"

    def test_description_includes_url(self, skill):
        assert "localhost:9999" in skill.description

    def test_description_not_configured(self):
        s = HealthReportingSkill()
        assert "not configured" in s.description

    def test_no_tools(self, skill):
        assert skill.tools() == []


class TestHealthReportingPreconditions:
    def test_valid_config(self, skill, mock_maxim):
        ok, reason = skill.can_activate(mock_maxim)
        assert ok is True

    def test_empty_url_rejected(self, mock_maxim):
        skill = HealthReportingSkill(HealthReportingConfig(endpoint_url=""))
        ok, reason = skill.can_activate(mock_maxim)
        assert ok is False
        assert "endpoint_url" in reason


class TestHealthReportingActivation:
    def test_activate_starts_thread(self, skill, mock_maxim):
        with patch.object(skill, "_report_loop"):
            result = skill.activate(mock_maxim, {})

        assert result.ok
        assert result.state == SkillState.ACTIVE
        assert skill._thread is not None
        assert "localhost:9999" in result.message

        skill.deactivate()

    def test_metadata_contains_config(self, skill, mock_maxim):
        with patch.object(skill, "_report_loop"):
            result = skill.activate(mock_maxim, {})

        assert result.metadata["endpoint_url"] == "http://localhost:9999/health"
        assert result.metadata["interval_seconds"] == 0.1

        skill.deactivate()

    def test_activate_stores_context(self, skill, mock_maxim):
        ctx = {"rtsp_url": "rtsp://localhost:8554/reachy", "_protocol_name": "test"}
        with patch.object(skill, "_report_loop"):
            skill.activate(mock_maxim, ctx)

        assert skill._context is ctx
        skill.deactivate()


class TestHealthReportingDeactivation:
    def test_deactivate_stops_thread(self, skill, mock_maxim):
        with patch.object(skill, "_report_loop"):
            skill.activate(mock_maxim, {})
        result = skill.deactivate()

        assert result.state == SkillState.IDLE
        assert skill._thread is None
        assert skill._context is None

    def test_deactivate_before_activate(self, skill):
        result = skill.deactivate()
        assert result.state == SkillState.IDLE


class TestHealthReportingPayload:
    def test_basic_payload(self, skill, mock_maxim):
        skill._maxim = mock_maxim
        skill._context = {}
        payload = skill._build_payload()

        assert payload["status"] == "active"
        assert "timestamp" in payload

    def test_payload_includes_rtsp_context(self, skill, mock_maxim):
        mock_maxim._protocol_registry._active = {}  # No active protocol
        skill._maxim = mock_maxim
        skill._context = {
            "rtsp_url": "rtsp://localhost:8554/reachy",
            "rtsp_fps": 20,
            "_protocol_name": "shredder_segmenter",
        }
        payload = skill._build_payload()

        assert payload["rtsp_url"] == "rtsp://localhost:8554/reachy"
        assert payload["rtsp_fps"] == 20
        assert payload["protocol"] == "shredder_segmenter"

    def test_payload_includes_sibling_health(self, skill, mock_maxim):
        """Should collect health from sibling skills in the same protocol."""
        sibling = Mock()
        sibling.health.return_value = {"name": "rtsp_streaming", "state": "active"}

        protocol = Mock()
        protocol._active_skills = [sibling, skill]

        mock_maxim._protocol_registry._active = {"test": protocol}

        skill._maxim = mock_maxim
        skill._context = {"_protocol_name": "test"}
        payload = skill._build_payload()

        assert "skills" in payload
        assert len(payload["skills"]) == 1  # Only sibling, not self
        assert payload["skills"][0]["name"] == "rtsp_streaming"


class TestHealthReportingPost:
    @patch("maxim.skills.health_reporting.urllib.request.urlopen")
    def test_post_health_sends_json(self, mock_urlopen, skill):
        payload = {"status": "active", "timestamp": 1234}
        skill._post_health(payload)

        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.get_method() == "POST"
        assert req.full_url == "http://localhost:9999/health"
        assert req.get_header("Content-type") == "application/json"
        body = json.loads(req.data)
        assert body["status"] == "active"

    @patch("maxim.skills.health_reporting.urllib.request.urlopen")
    def test_post_health_includes_custom_headers(self, mock_urlopen):
        cfg = HealthReportingConfig(
            endpoint_url="http://example.com/health",
            headers={"Authorization": "Bearer secret"},
        )
        skill = HealthReportingSkill(cfg)
        skill._post_health({"status": "active"})

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer secret"


class TestHealthReportingLoop:
    @patch("maxim.skills.health_reporting.urllib.request.urlopen")
    def test_report_loop_posts_periodically(self, mock_urlopen, skill, mock_maxim):
        """The background thread should post health reports."""
        mock_maxim._protocol_registry._active = {}
        ctx = {"_protocol_name": "test"}
        skill.activate(mock_maxim, ctx)

        # Wait for a few reports
        time.sleep(0.5)
        skill.deactivate()

        # Should have posted at least 2 reports in 0.5s with 0.1s interval
        assert mock_urlopen.call_count >= 2

    @patch("maxim.skills.health_reporting.urllib.request.urlopen", side_effect=Exception("connection refused"))
    def test_report_loop_tracks_failures(self, mock_urlopen, skill, mock_maxim):
        """Failures should increment consecutive_failures counter."""
        skill.activate(mock_maxim, {})
        time.sleep(0.4)
        skill.deactivate()

        assert skill._consecutive_failures >= 2


class TestHealthReportingFailureDetection:
    @patch("maxim.skills.health_reporting.urllib.request.urlopen", side_effect=Exception("connection refused"))
    def test_transitions_to_failed_after_10(self, mock_urlopen, config, mock_maxim):
        """10 consecutive failures in the report loop should trigger FAILED state."""
        # Use very fast interval so we hit 10 failures quickly
        import dataclasses

        fast_config = dataclasses.replace(config, interval_seconds=0.02)
        skill = HealthReportingSkill(fast_config)
        mock_maxim._protocol_registry._active = {}
        skill.activate(mock_maxim, {})
        time.sleep(0.8)  # Enough time for 10+ iterations at 0.02s each
        assert skill.state == SkillState.FAILED
        skill.deactivate()


class TestHealthReportingContext:
    def test_context_for_llm_active_healthy(self, skill, mock_maxim):
        with patch.object(skill, "_report_loop"):
            skill.activate(mock_maxim, {})
        ctx = skill.context_for_llm()
        assert "healthy" in ctx
        assert "localhost:9999" in ctx
        skill.deactivate()

    def test_context_for_llm_active_with_failures(self, skill, mock_maxim):
        with patch.object(skill, "_report_loop"):
            skill.activate(mock_maxim, {})
        skill._consecutive_failures = 3
        ctx = skill.context_for_llm()
        assert "3 consecutive failures" in ctx
        skill.deactivate()

    def test_context_for_llm_idle(self, skill):
        assert skill.context_for_llm() == ""


class TestHealthReportingHealth:
    @patch("maxim.skills.health_reporting.urllib.request.urlopen")
    def test_health_includes_endpoint(self, mock_urlopen, skill, mock_maxim):
        mock_maxim._protocol_registry._active = {}
        skill.activate(mock_maxim, {})
        h = skill.health()
        assert h["endpoint_url"] == "http://localhost:9999/health"
        assert h["consecutive_failures"] == 0
        assert h["thread_alive"] is True
        skill.deactivate()

    def test_health_idle(self, skill):
        h = skill.health()
        assert h["state"] == "idle"
        assert h["endpoint_url"] == "http://localhost:9999/health"
