"""Tests for frozen dataclass immutability and hashability."""

import dataclasses

import pytest

from maxim.agents.bus import ToolErrorKind
from maxim.agents.llm_types import LLMProposal
from maxim.planning.plan_document import LongHorizonConfig
from maxim.tools.base import ToolOutput


# ── ToolOutput ───────────────────────────────────────────────────────────────


class TestToolOutput:
    def test_immutable(self):
        obj = ToolOutput(success=True, output="ok")
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.success = False

    def test_creation_with_all_fields(self):
        obj = ToolOutput(
            success=False,
            output="data",
            error="boom",
            error_kind=ToolErrorKind.TIMEOUT,
            metadata={"key": "val"},
        )
        assert obj.success is False
        assert obj.output == "data"
        assert obj.error == "boom"
        assert obj.error_kind == ToolErrorKind.TIMEOUT
        assert obj.metadata == {"key": "val"}

    def test_not_hashable(self):
        obj = ToolOutput(success=True, metadata={"a": 1})
        with pytest.raises(TypeError):
            hash(obj)


# ── LLMProposal ──────────────────────────────────────────────────────────────


class TestLLMProposal:
    def test_immutable(self):
        obj = LLMProposal(
            request_id="r1",
            action=None,
            reasoning="because",
            strategy_used=None,
            confidence=0.9,
            mode_goal_achieved=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.confidence = 0.0

    def test_not_hashable(self):
        obj = LLMProposal(
            request_id="r1",
            action=None,
            reasoning="because",
            strategy_used=None,
            confidence=0.9,
            mode_goal_achieved=False,
        )
        with pytest.raises(TypeError):
            hash(obj)


# ── LongHorizonConfig ────────────────────────────────────────────────────────


class TestLongHorizonConfig:
    def test_immutable(self):
        cfg = LongHorizonConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.enable_plans = False

    def test_hashable(self):
        cfg = LongHorizonConfig()
        h = hash(cfg)
        assert isinstance(h, int)

    def test_in_set(self):
        cfg1 = LongHorizonConfig()
        cfg2 = LongHorizonConfig(enable_plans=False)
        s = {cfg1, cfg2}
        assert len(s) == 2
        assert cfg1 in s
        assert cfg2 in s

    def test_replace(self):
        original = LongHorizonConfig(enable_plans=True, default_depth_bound=2)
        modified = dataclasses.replace(original, enable_plans=False)
        assert modified.enable_plans is False
        assert modified.default_depth_bound == 2
        assert original.enable_plans is True
