"""Unit tests for LLM type definitions."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from maxim.agents.autonomy import AutonomyLevel
from maxim.agents.llm_types import LLMProposal, LLMRequest, ModeInfo, StrategyInfo


# ─────────────────────────────────────────────────────────────────────────────
# ModeInfo
# ─────────────────────────────────────────────────────────────────────────────


class TestModeInfoDefaults:
    def test_required_fields(self):
        mode = ModeInfo(name="test", goal="do stuff", context_prompt="prompt")
        assert mode.name == "test"
        assert mode.goal == "do stuff"
        assert mode.context_prompt == "prompt"

    def test_default_sets_empty(self):
        mode = ModeInfo(name="m", goal="g", context_prompt="p")
        assert mode.forbidden_tools == set()
        assert mode.allowed_tools == set()

    def test_default_capabilities(self):
        mode = ModeInfo(name="m", goal="g", context_prompt="p")
        assert mode.max_initiative == 1.0
        assert mode.can_access_filesystem is True
        assert mode.can_access_network is True

    def test_default_response_config(self):
        mode = ModeInfo(name="m", goal="g", context_prompt="p")
        assert mode.max_response_tokens == 512
        assert mode.context_window_tokens == 2048
        assert mode.response_format == "conversational"


class TestModeInfoGetAvailableTools:
    def _mode(self, **kwargs):
        defaults = dict(name="m", goal="g", context_prompt="p")
        defaults.update(kwargs)
        return ModeInfo(**defaults)

    def test_no_filters_returns_all(self):
        all_tools = {"read_file", "write_file", "speak", "think"}
        mode = self._mode()
        assert mode.get_available_tools(all_tools) == all_tools

    def test_forbidden_tools_excluded(self):
        all_tools = {"read_file", "write_file", "speak"}
        mode = self._mode(forbidden_tools={"speak"})
        result = mode.get_available_tools(all_tools)
        assert result == {"read_file", "write_file"}

    def test_allowed_tools_restricts(self):
        all_tools = {"read_file", "write_file", "speak", "think"}
        mode = self._mode(allowed_tools={"read_file", "speak"})
        result = mode.get_available_tools(all_tools)
        assert result == {"read_file", "speak"}

    def test_allowed_and_forbidden_combined(self):
        all_tools = {"read_file", "write_file", "speak", "think"}
        mode = self._mode(allowed_tools={"read_file", "speak"}, forbidden_tools={"speak"})
        result = mode.get_available_tools(all_tools)
        assert result == {"read_file"}

    def test_filesystem_disabled_removes_fs_tools(self):
        all_tools = {"read_file", "write_file", "list_directory", "execute_file", "speak"}
        mode = self._mode(can_access_filesystem=False)
        result = mode.get_available_tools(all_tools)
        assert result == {"speak"}

    def test_network_disabled_removes_net_tools(self):
        all_tools = {"internet_search", "http_fetch", "speak"}
        mode = self._mode(can_access_network=False)
        result = mode.get_available_tools(all_tools)
        assert result == {"speak"}

    def test_both_capabilities_disabled(self):
        all_tools = {
            "read_file", "write_file", "list_directory", "execute_file",
            "internet_search", "http_fetch", "speak",
        }
        mode = self._mode(can_access_filesystem=False, can_access_network=False)
        result = mode.get_available_tools(all_tools)
        assert result == {"speak"}

    def test_forbidden_plus_disabled_capabilities(self):
        all_tools = {"read_file", "speak", "internet_search", "think"}
        mode = self._mode(forbidden_tools={"think"}, can_access_network=False)
        result = mode.get_available_tools(all_tools)
        assert result == {"read_file", "speak"}

    def test_allowed_plus_disabled_filesystem(self):
        all_tools = {"read_file", "speak", "think"}
        mode = self._mode(allowed_tools={"read_file", "speak"}, can_access_filesystem=False)
        result = mode.get_available_tools(all_tools)
        assert result == {"speak"}

    def test_empty_all_tools(self):
        mode = self._mode()
        assert mode.get_available_tools(set()) == set()

    def test_forbidden_not_in_all_tools_no_error(self):
        mode = self._mode(forbidden_tools={"nonexistent"})
        result = mode.get_available_tools({"speak"})
        assert result == {"speak"}


# ─────────────────────────────────────────────────────────────────────────────
# StrategyInfo
# ─────────────────────────────────────────────────────────────────────────────


class TestStrategyInfo:
    def test_construction(self):
        s = StrategyInfo(name="explore", description="look around", approach_prompt="be curious")
        assert s.name == "explore"
        assert s.description == "look around"
        assert s.approach_prompt == "be curious"


# ─────────────────────────────────────────────────────────────────────────────
# LLMRequest
# ─────────────────────────────────────────────────────────────────────────────


def _make_request(priority=0, timestamp=None, **kwargs):
    defaults = dict(
        request_id="req-1",
        context=MagicMock(),
        mode=ModeInfo(name="m", goal="g", context_prompt="p"),
        autonomy_level=AutonomyLevel.SUPERVISED,
        strategies=[],
        internet_access=False,
        internet_policy_summary="none",
    )
    defaults.update(kwargs)
    if timestamp is not None:
        defaults["timestamp"] = timestamp
    defaults["priority"] = priority
    return LLMRequest(**defaults)


class TestLLMRequestDefaults:
    def test_default_priority(self):
        req = _make_request()
        assert req.priority == 0

    def test_default_collections_empty(self):
        req = _make_request()
        assert req.available_tools == set()
        assert req.tool_descriptions == {}
        assert req.agent_states == []
        assert req.recent_outcomes == []

    def test_default_strings_empty(self):
        req = _make_request()
        assert req.context_pool_text == ""
        assert req.triggering_input == ""
        assert req.lane == ""
        assert req.conversation_history_text == ""
        assert req.current_strategy == ""
        assert req.prefetch_context == ""

    def test_default_bools(self):
        req = _make_request()
        assert req.use_tool_prompting is False
        assert req.is_sleeping is False
        assert req.skip_exploration is False

    def test_default_none_fields(self):
        req = _make_request()
        assert req.pending_modification is None

    def test_timestamp_auto_set(self):
        before = time.time()
        req = _make_request()
        after = time.time()
        assert before <= req.timestamp <= after


class TestLLMRequestOrdering:
    def test_higher_priority_sorts_first(self):
        low = _make_request(priority=1, timestamp=100.0)
        high = _make_request(priority=5, timestamp=100.0)
        assert high < low  # higher priority = lower sort_index

    def test_same_priority_earlier_timestamp_first(self):
        early = _make_request(priority=0, timestamp=100.0)
        late = _make_request(priority=0, timestamp=200.0)
        assert early < late

    def test_sort_index_computed(self):
        req = _make_request(priority=3, timestamp=42.0)
        assert req.sort_index == (-3, 42.0)

    def test_sorted_list(self):
        r1 = _make_request(priority=0, timestamp=1.0, request_id="low-late")
        r2 = _make_request(priority=5, timestamp=2.0, request_id="high-late")
        r3 = _make_request(priority=5, timestamp=0.5, request_id="high-early")
        r4 = _make_request(priority=2, timestamp=1.0, request_id="mid")
        ordered = sorted([r1, r2, r3, r4])
        assert [r.request_id for r in ordered] == [
            "high-early", "high-late", "mid", "low-late",
        ]

    def test_equal_sort_index(self):
        a = _make_request(priority=1, timestamp=10.0)
        b = _make_request(priority=1, timestamp=10.0)
        assert not (a < b)
        assert not (b < a)
        assert a <= b


# ─────────────────────────────────────────────────────────────────────────────
# LLMProposal
# ─────────────────────────────────────────────────────────────────────────────


def _make_proposal(**kwargs):
    defaults = dict(
        request_id="req-1",
        action=None,
        reasoning="because",
        strategy_used=None,
        confidence=0.8,
        mode_goal_achieved=False,
    )
    defaults.update(kwargs)
    return LLMProposal(**defaults)


class TestLLMProposalDefaults:
    def test_default_lists_empty(self):
        p = _make_proposal()
        assert p.citations == []
        assert p.next_actions == []
        assert p.parallel_actions == []

    def test_default_scalars(self):
        p = _make_proposal()
        assert p.latency_ms == 0.0
        assert p.error is None
        assert p.triggering_input == ""
        assert p.plan_text is None
        assert p.requires_approval is False

    def test_timestamp_auto_set(self):
        before = time.time()
        p = _make_proposal()
        after = time.time()
        assert before <= p.timestamp <= after


class TestLLMProposalGetAllActions:
    def test_no_action_no_next(self):
        p = _make_proposal(action=None, next_actions=[])
        assert p.get_all_actions() == []

    def test_primary_only(self):
        action = {"tool": "speak", "args": {}}
        p = _make_proposal(action=action)
        assert p.get_all_actions() == [action]

    def test_next_only(self):
        nexts = [{"tool": "a"}, {"tool": "b"}]
        p = _make_proposal(action=None, next_actions=nexts)
        assert p.get_all_actions() == nexts

    def test_primary_plus_next(self):
        action = {"tool": "speak"}
        nexts = [{"tool": "a"}, {"tool": "b"}]
        p = _make_proposal(action=action, next_actions=nexts)
        result = p.get_all_actions()
        assert result == [action, {"tool": "a"}, {"tool": "b"}]


class TestLLMProposalGetParallelActions:
    def test_no_action_no_parallel(self):
        p = _make_proposal(action=None, parallel_actions=[])
        assert p.get_parallel_actions() == []

    def test_primary_only(self):
        action = {"tool": "speak"}
        p = _make_proposal(action=action)
        assert p.get_parallel_actions() == [action]

    def test_parallel_only(self):
        pars = [{"tool": "glob"}, {"tool": "read_file"}]
        p = _make_proposal(action=None, parallel_actions=pars)
        assert p.get_parallel_actions() == pars

    def test_primary_plus_parallel(self):
        action = {"tool": "speak"}
        pars = [{"tool": "glob"}, {"tool": "read_file"}]
        p = _make_proposal(action=action, parallel_actions=pars)
        result = p.get_parallel_actions()
        assert result == [action, {"tool": "glob"}, {"tool": "read_file"}]
