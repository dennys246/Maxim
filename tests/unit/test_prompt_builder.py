"""Unit tests for standalone functions in maxim.agents.prompt_builder."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from maxim.agents.autonomy import AutonomyLevel
from maxim.agents.prompt_builder import (
    build_planning_banner,
    build_modification_section,
    build_identity_section,
    build_datetime_section,
    build_tools_section,
    build_tool_guidance_core,
    build_tool_guidance_extended,
    build_observation_section,
    build_instructions_section,
    is_realtime_request,
    compute_budget_tier,
    format_usd,
    format_spend_rate,
)


# ── build_planning_banner ──────────────────────────────────────────────────


class TestBuildPlanningBanner:
    def test_planning_mode_returns_banner(self):
        result = build_planning_banner(AutonomyLevel.PLANNING)
        assert "PLANNING MODE" in result
        assert "<|action_json|>" in result
        assert "PLAIN ENGLISH" in result or "PLAIN TEXT" in result

    def test_non_planning_returns_empty(self):
        assert build_planning_banner(AutonomyLevel.SUPERVISED) == ""

    def test_autonomous_returns_empty(self):
        assert build_planning_banner(AutonomyLevel.AUTONOMOUS) == ""


# ── build_modification_section ─────────────────────────────────────────────


class TestBuildModificationSection:
    def test_basic_modification(self):
        mod = {
            "original_tool_name": "internet_search",
            "original_action": {"params": {"query": "weather"}},
            "original_reasoning": "user asked about weather",
            "user_modification": "search for Denver weather instead",
        }
        result = build_modification_section(mod)
        assert "internet_search" in result
        assert "weather" in result
        assert "Denver" in result
        assert "ACTION MODIFICATION REQUESTED" in result

    def test_missing_keys_use_defaults(self):
        result = build_modification_section({})
        assert "unknown" in result
        assert "not provided" in result

    def test_empty_user_modification(self):
        mod = {"user_modification": ""}
        result = build_modification_section(mod)
        assert '""' in result


# ── build_identity_section ─────────────────────────────────────────────────


class TestBuildIdentitySection:
    def _make_mode(self, name="passive", goal="assist the user"):
        mode = MagicMock()
        mode.name = name
        mode.goal = goal
        return mode

    def _make_request(self, strategy=None, autonomy=AutonomyLevel.SUPERVISED, sleeping=False):
        req = MagicMock()
        req.current_strategy = strategy
        req.autonomy_level = autonomy
        req.is_sleeping = sleeping
        return req

    def test_basic_identity(self):
        result = build_identity_section(
            self._make_mode(), self._make_request(), "Monday", "10:00 AM"
        )
        assert "Maxim" in result
        assert "PASSIVE" in result
        assert "AWAKE" in result

    def test_sleeping_state(self):
        result = build_identity_section(
            self._make_mode(), self._make_request(sleeping=True), "Monday", "10:00 AM"
        )
        assert "SLEEP" in result

    def test_mode_goal_included(self):
        result = build_identity_section(
            self._make_mode(goal="explore the environment"),
            self._make_request(),
            "Monday",
            "10:00 AM",
        )
        assert "explore the environment" in result


# ── build_datetime_section ─────────────────────────────────────────────────


class TestBuildDatetimeSection:
    def test_contains_date_and_time(self):
        result = build_datetime_section("Wednesday, March 12, 2025", "3:30 PM")
        assert "Wednesday, March 12, 2025" in result
        assert "3:30 PM" in result

    def test_search_advice(self):
        result = build_datetime_section("Tuesday, January 1, 2030", "12:00 AM")
        assert "search" in result.lower()
        assert "Tuesday, January 1, 2030" in result


# ── build_tools_section ───────────────────────────────────────────────────


class TestBuildToolsSection:
    def _make_request(self, tools=None, descriptions=None):
        req = MagicMock()
        req.available_tools = tools or {"respond", "internet_search"}
        req.tool_descriptions = descriptions or {}
        return req

    def test_lists_tools_sorted(self):
        result = build_tools_section(self._make_request())
        lines = result.split("\n")
        tool_lines = [l for l in lines if l.startswith("- ")]
        assert tool_lines[0].startswith("- internet_search")
        assert tool_lines[1].startswith("- respond")

    def test_dict_description(self):
        desc = {
            "write_file": {
                "description": "Write a file",
                "params": {"path": "str", "content": "str"},
            }
        }
        req = self._make_request(tools={"write_file"}, descriptions=desc)
        result = build_tools_section(req)
        assert "Write a file" in result
        assert "REQUIRED params" in result

    def test_string_description(self):
        desc = {"respond": "Send a message to the user"}
        req = self._make_request(tools={"respond"}, descriptions=desc)
        result = build_tools_section(req)
        assert "Send a message to the user" in result

    def test_singularity_strips_workspace_prefix(self):
        desc = {
            "write_file": {
                "description": "Write a file",
                "example": {"path": ".maxim_workspace/test.py"},
            }
        }
        req = self._make_request(tools={"write_file"}, descriptions=desc)
        result = build_tools_section(req, mode_name="singularity")
        assert ".maxim_workspace/" not in result

    def test_passive_keeps_workspace_prefix(self):
        desc = {
            "write_file": {
                "description": "Write a file",
                "example": {"path": ".maxim_workspace/test.py"},
            }
        }
        req = self._make_request(tools={"write_file"}, descriptions=desc)
        result = build_tools_section(req, mode_name="passive")
        assert ".maxim_workspace/" in result


# ── build_tool_guidance_core ──────────────────────────────────────────────


class TestBuildToolGuidanceCore:
    def test_passive_mode(self):
        result = build_tool_guidance_core("passive")
        assert ".maxim_workspace/" in result
        assert "proposed as plans" in result

    def test_active_mode(self):
        result = build_tool_guidance_core("active")
        assert ".maxim_workspace/" in result
        assert "suggest edits" in result

    def test_singularity_mode(self):
        result = build_tool_guidance_core("singularity")
        assert "ANY file" in result
        assert ".maxim_workspace/" not in result


# ── build_tool_guidance_extended ──────────────────────────────────────────


class TestBuildToolGuidanceExtended:
    def test_passive_mode(self):
        result = build_tool_guidance_extended("passive")
        assert "Tool Selection" in result
        assert "drafts/" in result
        assert "Project directory" not in result

    def test_singularity_mode(self):
        result = build_tool_guidance_extended("singularity")
        assert "Project directory" in result

    def test_contains_batched_calls(self):
        result = build_tool_guidance_extended("active")
        assert "BATCHED" in result


# ── build_observation_section ─────────────────────────────────────────────


class TestBuildObservationSection:
    def _make_percept(self, transcript=None, detections=None, cli_input=None):
        p = MagicMock()
        p.transcript_chunk = transcript
        p.detections = detections
        p.cli_input = cli_input
        return p

    def test_empty_percept(self):
        result = build_observation_section(self._make_percept())
        assert result == ""

    def test_transcript(self):
        result = build_observation_section(self._make_percept(transcript="hello world"))
        assert "hello world" in result
        assert "Heard" in result

    def test_detections(self):
        dets = [{"label": "person"}, {"label": "cup"}]
        result = build_observation_section(self._make_percept(detections=dets))
        assert "person" in result
        assert "cup" in result

    def test_cli_input(self):
        result = build_observation_section(self._make_percept(cli_input="do something"))
        assert "do something" in result

    def test_combined(self):
        result = build_observation_section(
            self._make_percept(transcript="hi", cli_input="test")
        )
        assert "hi" in result
        assert "test" in result


# ── build_instructions_section ────────────────────────────────────────────


class TestBuildInstructionsSection:
    def test_planning_mode(self):
        req = MagicMock()
        req.autonomy_level = AutonomyLevel.PLANNING
        result = build_instructions_section(req)
        assert "PLANNING MODE" in result
        assert "<|action_json|>" in result

    def test_normal_mode(self):
        req = MagicMock()
        req.autonomy_level = AutonomyLevel.SUPERVISED
        result = build_instructions_section(req)
        assert "compact JSON" in result
        assert "PLANNING" not in result

    def test_autonomous_mode(self):
        req = MagicMock()
        req.autonomy_level = AutonomyLevel.AUTONOMOUS
        result = build_instructions_section(req)
        assert "compact JSON" in result


# ── is_realtime_request ───────────────────────────────────────────────────


class TestIsRealtimeRequest:
    @pytest.mark.parametrize("text", [
        "What's the score of the Broncos game?",
        "current weather in Denver",
        "latest news about AI",
        "bitcoin price today",
        "Patriots vs Bills",
    ])
    def test_realtime_detected(self, text):
        assert is_realtime_request(text) is True

    @pytest.mark.parametrize("text", [
        "What is the capital of France?",
        "Explain quantum physics",
        "Write a Python function",
    ])
    def test_non_realtime(self, text):
        assert is_realtime_request(text) is False

    def test_empty_string(self):
        assert is_realtime_request("") is False

    def test_none_like_empty(self):
        # The function guards with `if question_text` before .lower()
        assert is_realtime_request("") is False

    def test_case_insensitive(self):
        assert is_realtime_request("WEATHER forecast") is True


# ── compute_budget_tier ───────────────────────────────────────────────────


class TestComputeBudgetTier:
    def _make_policy(
        self,
        per_hour=1.0,
        per_day=10.0,
        per_month=100.0,
        critical=0.9,
        warning=0.7,
    ):
        p = MagicMock()
        p.max_cost_per_hour = per_hour
        p.max_cost_per_day = per_day
        p.max_cost_per_month = per_month
        p.cost_critical_threshold = critical
        p.cost_warning_threshold = warning
        return p

    def test_normal(self):
        totals = {"hourly": 0.1, "daily": 1.0, "monthly": 10.0}
        assert compute_budget_tier(self._make_policy(), totals) == "normal"

    def test_warning(self):
        totals = {"hourly": 0.75, "daily": 1.0, "monthly": 10.0}
        assert compute_budget_tier(self._make_policy(), totals) == "warning"

    def test_critical(self):
        totals = {"hourly": 0.95, "daily": 1.0, "monthly": 10.0}
        assert compute_budget_tier(self._make_policy(), totals) == "critical"

    def test_blocked(self):
        totals = {"hourly": 1.5, "daily": 1.0, "monthly": 10.0}
        assert compute_budget_tier(self._make_policy(), totals) == "blocked"

    def test_no_limits_returns_normal(self):
        policy = self._make_policy(per_hour=0, per_day=0, per_month=0)
        totals = {"hourly": 999, "daily": 999, "monthly": 999}
        assert compute_budget_tier(policy, totals) == "normal"

    def test_daily_triggers_warning(self):
        totals = {"hourly": 0.0, "daily": 7.5, "monthly": 0.0}
        assert compute_budget_tier(self._make_policy(), totals) == "warning"

    def test_monthly_triggers_blocked(self):
        totals = {"hourly": 0.0, "daily": 0.0, "monthly": 150.0}
        assert compute_budget_tier(self._make_policy(), totals) == "blocked"


# ── format_usd ────────────────────────────────────────────────────────────


class TestFormatUsd:
    def test_none(self):
        assert format_usd(None) == "unlimited"

    def test_zero(self):
        assert format_usd(0) == "$0.00"

    def test_negative(self):
        assert format_usd(-5) == "$0.00"

    def test_positive(self):
        assert format_usd(3.456) == "$3.46"

    def test_large_value(self):
        assert format_usd(1234.5) == "$1234.50"


# ── format_spend_rate ─────────────────────────────────────────────────────


class TestFormatSpendRate:
    def _make_rate(self, value=0.5, samples=10):
        r = MagicMock()
        r.value = value
        r.samples = samples
        return r

    def test_mature_rate(self):
        result = format_spend_rate(self._make_rate(0.5, 10), min_samples=5)
        assert result == "$0.50/hr"

    def test_immature_low_samples(self):
        result = format_spend_rate(self._make_rate(0.5, 2), min_samples=5)
        assert "immature" in result
        assert "samples=2" in result

    def test_immature_zero_value(self):
        result = format_spend_rate(self._make_rate(0.0, 10), min_samples=5)
        assert "immature" in result

    def test_exact_min_samples(self):
        result = format_spend_rate(self._make_rate(1.0, 5), min_samples=5)
        assert result == "$1.00/hr"
