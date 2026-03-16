"""Unit tests for Skill ABC, SkillState, SkillResult, SkillConfig."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from maxim.skills.base import Skill, SkillConfig, SkillResult, SkillState


# --- Concrete test implementation ---

class DummySkill(Skill):
    """Minimal concrete Skill for testing."""

    def __init__(self, name: str = "dummy", deps: list[str] | None = None):
        self._name = name
        self._state = SkillState.IDLE
        self._last_result: SkillResult | None = None
        self._deps = deps or []
        self._bio: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A dummy skill for testing"

    def tools(self) -> list:
        return []

    def activate(self, maxim: Any, context: dict[str, Any] | None = None) -> SkillResult:
        self._state = SkillState.ACTIVATING
        self._state = SkillState.ACTIVE
        self._last_result = SkillResult(
            state=SkillState.ACTIVE,
            message="Dummy activated",
            metadata={"activated": True},
        )
        if context is not None:
            context["dummy_activated"] = True
        return self._last_result

    def deactivate(self) -> SkillResult:
        self._state = SkillState.DEACTIVATING
        self._state = SkillState.IDLE
        self._last_result = SkillResult(state=SkillState.IDLE, message="Dummy stopped")
        return self._last_result

    def bio_dependencies(self) -> list[str]:
        return self._deps


class FailingSkill(Skill):
    """Skill that fails on activation."""

    @property
    def name(self) -> str:
        return "failing"

    @property
    def description(self) -> str:
        return "Always fails"

    def tools(self) -> list:
        return []

    def activate(self, maxim: Any, context: dict[str, Any] | None = None) -> SkillResult:
        self._state = SkillState.FAILED
        self._last_result = SkillResult(
            state=SkillState.FAILED,
            message="Activation failed",
            error="Something broke",
        )
        return self._last_result

    def deactivate(self) -> SkillResult:
        self._state = SkillState.IDLE
        return SkillResult(state=SkillState.IDLE)

    def can_activate(self, maxim: Any) -> tuple[bool, str]:
        return False, "ffmpeg not found"


class PreconditionSkill(DummySkill):
    """Skill with precondition check."""

    def can_activate(self, maxim: Any) -> tuple[bool, str]:
        if hasattr(maxim, "mini") and maxim.mini is not None:
            return True, ""
        return False, "No robot connection"


@pytest.fixture
def dummy_skill():
    return DummySkill()


@pytest.fixture
def mock_maxim():
    m = Mock()
    m.mini = Mock()
    return m


# --- SkillState tests ---

class TestSkillState:
    def test_enum_values(self):
        assert SkillState.IDLE.value == "idle"
        assert SkillState.ACTIVATING.value == "activating"
        assert SkillState.ACTIVE.value == "active"
        assert SkillState.FAILED.value == "failed"
        assert SkillState.DEACTIVATING.value == "deactivating"

    def test_all_states_exist(self):
        assert len(SkillState) == 5


# --- SkillResult tests ---

class TestSkillResult:
    def test_ok_when_active(self):
        r = SkillResult(state=SkillState.ACTIVE, message="running")
        assert r.ok is True

    def test_ok_when_idle(self):
        r = SkillResult(state=SkillState.IDLE)
        assert r.ok is True

    def test_not_ok_when_failed(self):
        r = SkillResult(state=SkillState.FAILED, error="broke")
        assert r.ok is False

    def test_not_ok_when_activating(self):
        r = SkillResult(state=SkillState.ACTIVATING)
        assert r.ok is False

    def test_defaults(self):
        r = SkillResult(state=SkillState.IDLE)
        assert r.message == ""
        assert r.error is None
        assert r.metadata == {}

    def test_metadata_isolation(self):
        """Each instance gets its own metadata dict."""
        r1 = SkillResult(state=SkillState.IDLE)
        r2 = SkillResult(state=SkillState.IDLE)
        r1.metadata["key"] = "val"
        assert "key" not in r2.metadata


# --- Skill ABC tests ---

class TestSkillLifecycle:
    def test_initial_state_is_idle(self, dummy_skill):
        assert dummy_skill.state == SkillState.IDLE
        assert dummy_skill.is_active is False

    def test_activate_transitions_to_active(self, dummy_skill, mock_maxim):
        result = dummy_skill.activate(mock_maxim)
        assert result.ok
        assert dummy_skill.state == SkillState.ACTIVE
        assert dummy_skill.is_active is True

    def test_deactivate_transitions_to_idle(self, dummy_skill, mock_maxim):
        dummy_skill.activate(mock_maxim)
        result = dummy_skill.deactivate()
        assert result.ok
        assert dummy_skill.state == SkillState.IDLE
        assert dummy_skill.is_active is False

    def test_shared_context_blackboard(self, dummy_skill, mock_maxim):
        ctx = {}
        dummy_skill.activate(mock_maxim, context=ctx)
        assert ctx["dummy_activated"] is True

    def test_activate_without_context(self, dummy_skill, mock_maxim):
        result = dummy_skill.activate(mock_maxim, context=None)
        assert result.ok


class TestSkillPreconditions:
    def test_default_can_activate_returns_true(self, dummy_skill, mock_maxim):
        ok, reason = dummy_skill.can_activate(mock_maxim)
        assert ok is True
        assert reason == ""

    def test_failing_skill_precondition(self, mock_maxim):
        skill = FailingSkill()
        ok, reason = skill.can_activate(mock_maxim)
        assert ok is False
        assert "ffmpeg" in reason

    def test_precondition_with_robot(self, mock_maxim):
        skill = PreconditionSkill()
        ok, reason = skill.can_activate(mock_maxim)
        assert ok is True

    def test_precondition_without_robot(self):
        skill = PreconditionSkill()
        maxim = Mock(spec=[])  # no attributes
        ok, reason = skill.can_activate(maxim)
        assert ok is False
        assert "robot" in reason.lower()


class TestSkillBioDependencies:
    def test_default_no_dependencies(self, dummy_skill):
        assert dummy_skill.bio_dependencies() == []

    def test_custom_dependencies(self):
        skill = DummySkill(deps=["hippocampus", "atl"])
        assert skill.bio_dependencies() == ["hippocampus", "atl"]


class TestSkillContextAndHealth:
    def test_context_for_llm_default_empty(self, dummy_skill):
        assert dummy_skill.context_for_llm() == ""

    def test_context_for_llm_on_failure(self):
        skill = FailingSkill()
        skill.activate(Mock())
        ctx = skill.context_for_llm()
        assert "FAILED" in ctx
        assert "Something broke" in ctx

    def test_health_basic(self, dummy_skill, mock_maxim):
        h = dummy_skill.health()
        assert h["name"] == "dummy"
        assert h["state"] == "idle"

    def test_health_after_activate(self, dummy_skill, mock_maxim):
        dummy_skill.activate(mock_maxim)
        h = dummy_skill.health()
        assert h["state"] == "active"
        assert h["message"] == "Dummy activated"
        assert h["metadata"] == {"activated": True}

    def test_health_after_failure(self):
        skill = FailingSkill()
        skill.activate(Mock())
        h = skill.health()
        assert h["state"] == "failed"
        assert "error" in h
