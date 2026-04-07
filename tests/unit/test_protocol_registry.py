"""Unit tests for ProtocolRegistry."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from maxim.skills.base import Skill, SkillResult, SkillState
from maxim.skills.protocol import Protocol, WorkspaceBounds
from maxim.skills.registry import ProtocolRegistry
from maxim.skills.tools import RunProtocolTool, StopProtocolTool, ListProtocolsTool


# --- Test helpers ---


class StubSkill(Skill):
    def __init__(self, name: str = "stub"):
        self._name = name
        self._state = SkillState.IDLE
        self._last_result: SkillResult | None = None
        self._bio: dict[str, Any] = {}
        self._tool = Mock()
        self._tool.name = f"{name}_tool"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Stub: {self._name}"

    def tools(self) -> list:
        return [self._tool]

    def activate(self, maxim: Any, context: dict[str, Any] | None = None) -> SkillResult:
        self._state = SkillState.ACTIVE
        self._last_result = SkillResult(state=SkillState.ACTIVE)
        return self._last_result

    def deactivate(self) -> SkillResult:
        self._state = SkillState.IDLE
        return SkillResult(state=SkillState.IDLE)


class StubProtocol(Protocol):
    def __init__(
        self,
        name: str = "stub_proto",
        skills: list[Skill] | None = None,
        bounds: WorkspaceBounds | None = None,
        phrases: list[str] | None = None,
        stop_phrases: list[str] | None = None,
    ):
        super().__init__()
        self._name = name
        self._skills_list = skills or [StubSkill()]
        self._bounds = bounds
        self._phrases = phrases or []
        self._stop_phrases = stop_phrases or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Stub protocol: {self._name}"

    def skills(self) -> list[Skill]:
        return self._skills_list

    def workspace_bounds(self) -> WorkspaceBounds | None:
        return self._bounds

    def phrases(self) -> list[str]:
        return self._phrases

    def stop_phrases(self) -> list[str]:
        return self._stop_phrases


@pytest.fixture
def mock_maxim():
    m = Mock()
    m.log = Mock()
    m._workspace_limit_override = None
    m._register_phrase_response = Mock()
    return m


@pytest.fixture
def tool_registry():
    reg = Mock()
    reg.register = Mock()
    reg.deregister = Mock(return_value=True)
    return reg


@pytest.fixture
def registry(mock_maxim, tool_registry):
    return ProtocolRegistry(maxim=mock_maxim, tool_registry=tool_registry)


# --- Registration tests ---


class TestProtocolRegistration:
    def test_register_protocol(self, registry):
        proto = StubProtocol()
        registry.register(proto)
        assert "stub_proto" in registry.get_available()

    def test_register_multiple(self, registry):
        registry.register(StubProtocol("a"))
        registry.register(StubProtocol("b"))
        assert len(registry.get_available()) == 2

    def test_get_available_lists_names(self, registry):
        registry.register(StubProtocol("alpha"))
        assert "alpha" in registry.get_available()


# --- Activation tests ---


class TestProtocolActivation:
    def test_activate_success(self, registry):
        registry.register(StubProtocol())
        result = registry.activate("stub_proto")
        assert "activated" in result.lower()

    def test_activate_registers_tools(self, registry, tool_registry):
        skill = StubSkill("s1")
        registry.register(StubProtocol(skills=[skill]))
        registry.activate("stub_proto")
        tool_registry.register.assert_called()

    def test_activate_unknown_protocol(self, registry):
        result = registry.activate("nonexistent")
        assert "unknown" in result.lower()

    def test_activate_already_active(self, registry):
        registry.register(StubProtocol())
        registry.activate("stub_proto")
        result = registry.activate("stub_proto")
        assert "already active" in result.lower()

    def test_activate_applies_workspace_bounds(self, registry, mock_maxim):
        bounds = WorkspaceBounds(yaw=30.0, pitch=20.0)
        registry.register(StubProtocol(bounds=bounds))
        registry.activate("stub_proto")
        assert mock_maxim._workspace_limit_override is not None
        assert mock_maxim._workspace_limit_override["yaw"] == 30.0
        assert mock_maxim._workspace_limit_override["pitch"] == 20.0

    def test_activate_no_bounds(self, registry, mock_maxim):
        registry.register(StubProtocol(bounds=None))
        registry.activate("stub_proto")
        assert mock_maxim._workspace_limit_override is None

    def test_get_active(self, registry):
        registry.register(StubProtocol())
        assert len(registry.get_active()) == 0
        registry.activate("stub_proto")
        assert len(registry.get_active()) == 1


# --- Deactivation tests ---


class TestProtocolDeactivation:
    def test_deactivate_success(self, registry):
        registry.register(StubProtocol())
        registry.activate("stub_proto")
        result = registry.deactivate("stub_proto")
        assert "deactivated" in result.lower()

    def test_deactivate_not_active(self, registry):
        result = registry.deactivate("nonexistent")
        assert "not active" in result.lower()

    def test_deactivate_deregisters_tools(self, registry, tool_registry):
        skill = StubSkill("s1")
        registry.register(StubProtocol(skills=[skill]))
        registry.activate("stub_proto")
        registry.deactivate("stub_proto")
        tool_registry.deregister.assert_called()

    def test_deactivate_restores_workspace(self, registry, mock_maxim):
        bounds = WorkspaceBounds(yaw=30.0)
        registry.register(StubProtocol(bounds=bounds))
        registry.activate("stub_proto")
        assert mock_maxim._workspace_limit_override is not None

        registry.deactivate("stub_proto")
        assert mock_maxim._workspace_limit_override is None

    def test_deactivate_removes_from_active(self, registry):
        registry.register(StubProtocol())
        registry.activate("stub_proto")
        registry.deactivate("stub_proto")
        assert len(registry.get_active()) == 0


# --- Multi-protocol workspace composition ---


class TestWorkspaceComposition:
    def test_tightest_bounds_win(self, registry, mock_maxim):
        registry.register(StubProtocol("p1", bounds=WorkspaceBounds(yaw=30.0)))
        registry.register(StubProtocol("p2", bounds=WorkspaceBounds(yaw=20.0)))
        registry.activate("p1")
        registry.activate("p2")
        assert mock_maxim._workspace_limit_override["yaw"] == 20.0

    def test_removing_protocol_recomputes(self, registry, mock_maxim):
        registry.register(StubProtocol("p1", bounds=WorkspaceBounds(yaw=30.0)))
        registry.register(StubProtocol("p2", bounds=WorkspaceBounds(yaw=20.0)))
        registry.activate("p1")
        registry.activate("p2")

        registry.deactivate("p2")
        assert mock_maxim._workspace_limit_override["yaw"] == 30.0

    def test_removing_all_clears_override(self, registry, mock_maxim):
        registry.register(StubProtocol("p1", bounds=WorkspaceBounds(yaw=30.0)))
        registry.activate("p1")
        registry.deactivate("p1")
        assert mock_maxim._workspace_limit_override is None


# --- LLM context ---


class TestLLMContext:
    def test_context_empty_when_nothing_active(self, registry):
        assert registry.get_context_for_llm() == ""

    def test_context_includes_active_protocols(self, registry):
        registry.register(StubProtocol())
        registry.activate("stub_proto")
        ctx = registry.get_context_for_llm()
        assert "stub_proto" in ctx


# --- Phrase registration ---


class TestPhraseRegistration:
    def test_registers_activation_phrases(self, registry, mock_maxim):
        proto = StubProtocol(
            phrases=["start test", "run test"],
            stop_phrases=["stop test"],
        )
        registry.register(proto)
        registry._register_phrases("stub_proto", proto)

        calls = mock_maxim._register_phrase_response.call_args_list
        assert len(calls) == 3  # 2 activation + 1 stop

    def test_no_crash_without_register_method(self, tool_registry):
        maxim = Mock(spec=[])  # no _register_phrase_response
        reg = ProtocolRegistry(maxim=maxim, tool_registry=tool_registry)
        proto = StubProtocol(phrases=["test"])
        reg.register(proto)
        reg._register_phrases("stub_proto", proto)  # should not raise


# --- Protocol management tools ---


class TestProtocolTools:
    def test_run_protocol_tool(self, registry):
        registry.register(StubProtocol())
        tool = RunProtocolTool(registry)
        result = tool.execute(name="stub_proto")
        assert result.success is True

    def test_run_protocol_unknown(self, registry):
        tool = RunProtocolTool(registry)
        result = tool.execute(name="nope")
        assert result.success is False

    def test_stop_protocol_tool(self, registry):
        registry.register(StubProtocol())
        registry.activate("stub_proto")
        tool = StopProtocolTool(registry)
        result = tool.execute(name="stub_proto")
        assert result.success is True

    def test_stop_protocol_not_active(self, registry):
        tool = StopProtocolTool(registry)
        result = tool.execute(name="nope")
        assert result.success is True  # "not active" is treated as success

    def test_list_protocols_tool(self, registry):
        registry.register(StubProtocol())
        tool = ListProtocolsTool(registry)
        result = tool.execute()
        assert result.success is True
        assert "stub_proto" in result.output

    def test_list_shows_active_status(self, registry):
        registry.register(StubProtocol())
        registry.activate("stub_proto")
        tool = ListProtocolsTool(registry)
        result = tool.execute()
        assert "ACTIVE" in result.output
