"""Unit tests for Protocol ABC and WorkspaceBounds."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from maxim.skills.base import Skill, SkillResult, SkillState
from maxim.skills.protocol import Protocol, WorkspaceBounds


# --- Test helpers ---


class StubSkill(Skill):
    """Controllable stub skill for protocol tests."""

    def __init__(
        self,
        name: str = "stub",
        can_activate_result: tuple[bool, str] = (True, ""),
        activate_ok: bool = True,
        deps: list[str] | None = None,
    ):
        self._name = name
        self._state = SkillState.IDLE
        self._last_result: SkillResult | None = None
        self._can_activate_result = can_activate_result
        self._activate_ok = activate_ok
        self._bio: dict[str, Any] = {}
        self._deps = deps or []
        self.activated_with_context: dict | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Stub: {self._name}"

    def tools(self) -> list:
        return []

    def can_activate(self, maxim: Any) -> tuple[bool, str]:
        return self._can_activate_result

    def activate(self, maxim: Any, context: dict[str, Any] | None = None) -> SkillResult:
        self.activated_with_context = context
        if self._activate_ok:
            self._state = SkillState.ACTIVE
            self._last_result = SkillResult(state=SkillState.ACTIVE, message=f"{self._name} active")
        else:
            self._state = SkillState.FAILED
            self._last_result = SkillResult(state=SkillState.FAILED, error=f"{self._name} failed")
        return self._last_result

    def deactivate(self) -> SkillResult:
        self._state = SkillState.IDLE
        self._last_result = SkillResult(state=SkillState.IDLE)
        return self._last_result

    def bio_dependencies(self) -> list[str]:
        return self._deps


class SimpleProtocol(Protocol):
    """Minimal concrete Protocol for testing."""

    def __init__(self, skills: list[Skill] | None = None, bounds: WorkspaceBounds | None = None):
        super().__init__()
        self._skills = skills or []
        self._bounds = bounds

    @property
    def name(self) -> str:
        return "test_protocol"

    @property
    def description(self) -> str:
        return "A test protocol"

    def skills(self) -> list[Skill]:
        return self._skills

    def workspace_bounds(self) -> WorkspaceBounds | None:
        return self._bounds


class AbortOnFailProtocol(SimpleProtocol):
    """Protocol that aborts on any skill failure."""

    def on_skill_failed(self, skill: Skill, result: SkillResult) -> str:
        return "abort"


class ContinueOnFailProtocol(SimpleProtocol):
    """Protocol that continues on skill failure."""

    def on_skill_failed(self, skill: Skill, result: SkillResult) -> str:
        return "continue"


@pytest.fixture
def mock_maxim():
    m = Mock()
    m.mini = Mock()
    m.log = Mock()
    m._nac = Mock()
    # MemoryHub is how bio systems are resolved now
    m._memory_hub = Mock()
    m._memory_hub.hippocampus = Mock()
    m._memory_hub.atl = Mock()
    m._memory_hub.angular_gyrus = Mock()
    m._memory_hub.scn = Mock()
    return m


# --- WorkspaceBounds tests ---


class TestWorkspaceBounds:
    def test_default_all_none(self):
        b = WorkspaceBounds()
        assert b.yaw is None
        assert b.pitch is None
        assert b.roll is None
        assert b.x is None
        assert b.y is None
        assert b.z is None

    def test_partial_bounds(self):
        b = WorkspaceBounds(yaw=30.0, pitch=20.0)
        assert b.yaw == 30.0
        assert b.pitch == 20.0
        assert b.roll is None


# --- Protocol lifecycle tests ---


class TestProtocolActivation:
    def test_activate_single_skill(self, mock_maxim):
        skill = StubSkill()
        proto = SimpleProtocol(skills=[skill])

        proto.on_activate(mock_maxim)

        assert skill.state == SkillState.ACTIVE
        assert proto._active_skills == [skill]
        assert proto._maxim is mock_maxim

    def test_activate_multiple_skills(self, mock_maxim):
        s1 = StubSkill("s1")
        s2 = StubSkill("s2")
        proto = SimpleProtocol(skills=[s1, s2])

        proto.on_activate(mock_maxim)

        assert s1.state == SkillState.ACTIVE
        assert s2.state == SkillState.ACTIVE
        assert len(proto._active_skills) == 2

    def test_shared_context_passed_to_skills(self, mock_maxim):
        s1 = StubSkill("s1")
        s2 = StubSkill("s2")
        proto = SimpleProtocol(skills=[s1, s2])

        proto.on_activate(mock_maxim)

        # Both skills should receive the same context dict
        assert s1.activated_with_context is s2.activated_with_context
        assert isinstance(s1.activated_with_context, dict)

    def test_activate_no_skills(self, mock_maxim):
        proto = SimpleProtocol(skills=[])
        proto.on_activate(mock_maxim)
        assert proto._active_skills == []


class TestProtocolDeactivation:
    def test_deactivate_stops_skills(self, mock_maxim):
        skill = StubSkill()
        proto = SimpleProtocol(skills=[skill])
        proto.on_activate(mock_maxim)

        proto.on_deactivate()

        assert skill.state == SkillState.IDLE
        assert proto._active_skills == []
        assert proto._maxim is None

    def test_deactivate_without_activate(self):
        proto = SimpleProtocol()
        proto.on_deactivate()  # Should not raise
        assert proto._active_skills == []


class TestPreconditionFailFast:
    def test_precondition_failure_prevents_activation(self, mock_maxim):
        ok_skill = StubSkill("ok")
        bad_skill = StubSkill("bad", can_activate_result=(False, "missing ffmpeg"))
        proto = SimpleProtocol(skills=[ok_skill, bad_skill])

        with pytest.raises(RuntimeError, match="missing ffmpeg"):
            proto.on_activate(mock_maxim)

        # Neither skill should be active
        assert ok_skill.state == SkillState.IDLE
        assert bad_skill.state == SkillState.IDLE


class TestSkillFailureRecovery:
    def test_abort_on_failure_rolls_back(self, mock_maxim):
        s1 = StubSkill("s1", activate_ok=True)
        s2 = StubSkill("s2", activate_ok=False)
        proto = AbortOnFailProtocol(skills=[s1, s2])

        with pytest.raises(RuntimeError, match="s2"):
            proto.on_activate(mock_maxim)

        # s1 was rolled back
        assert s1.state == SkillState.IDLE

    def test_continue_on_failure_keeps_going(self, mock_maxim):
        s1 = StubSkill("s1", activate_ok=True)
        s2 = StubSkill("s2", activate_ok=False)
        s3 = StubSkill("s3", activate_ok=True)
        proto = ContinueOnFailProtocol(skills=[s1, s2, s3])

        proto.on_activate(mock_maxim)

        assert s1.state == SkillState.ACTIVE
        assert s2.state == SkillState.FAILED
        assert s3.state == SkillState.ACTIVE
        # Only successful skills in _active_skills
        assert s2 not in proto._active_skills

    def test_default_on_skill_failed_aborts(self, mock_maxim):
        """Default Protocol.on_skill_failed returns 'abort'."""
        s1 = StubSkill("s1", activate_ok=False)
        s2 = StubSkill("s2", activate_ok=True)
        proto = SimpleProtocol(skills=[s1, s2])

        with pytest.raises(RuntimeError, match="s1"):
            proto.on_activate(mock_maxim)

        assert s2.state == SkillState.IDLE


class TestBioSystemResolution:
    def test_bio_systems_resolved(self, mock_maxim):
        skill = StubSkill("bio_skill", deps=["hippocampus", "nac"])
        proto = SimpleProtocol(skills=[skill])

        proto.on_activate(mock_maxim)

        assert skill._bio["hippocampus"] is mock_maxim._memory_hub.hippocampus
        assert skill._bio["nac"] is mock_maxim._nac

    def test_missing_bio_system_is_none(self, mock_maxim):
        del mock_maxim._memory_hub  # Remove hub
        skill = StubSkill("bio_skill", deps=["hippocampus"])
        proto = SimpleProtocol(skills=[skill])

        proto.on_activate(mock_maxim)

        assert skill._bio.get("hippocampus") is None

    def test_no_bio_deps_skips_resolution(self, mock_maxim):
        skill = StubSkill()
        proto = SimpleProtocol(skills=[skill])

        proto.on_activate(mock_maxim)

        assert skill._bio == {}


class TestProtocolContext:
    def test_context_for_llm_includes_skill_context(self, mock_maxim):
        skill = StubSkill()
        proto = SimpleProtocol(skills=[skill])
        proto.on_activate(mock_maxim)

        ctx = proto.context_for_llm()
        assert "test_protocol" in ctx

    def test_context_for_llm_always_includes_name(self):
        proto = SimpleProtocol()
        ctx = proto.context_for_llm()
        assert "test_protocol" in ctx

    def test_phrases_auto_generated(self):
        proto = SimpleProtocol()
        phrases = proto.phrases()
        assert len(phrases) > 0
        assert any("test protocol" in p for p in phrases)

    def test_stop_phrases_auto_generated(self):
        proto = SimpleProtocol()
        stop = proto.stop_phrases()
        assert len(stop) > 0
        assert any("stop" in p for p in stop)
