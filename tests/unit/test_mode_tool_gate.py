"""#826: a mode's tool list is enforced at DISPATCH, not only in the prompt.

The mode's allow-list (and its forbidden list) shaped only the prompt roster: a tool outside it still
executed when the model named it (reproduced: ``search_code`` ran in passive mode at PLANNING, and a
mode-forbidden ``maxim_command`` reached its tool at AUTONOMOUS). The executor now reads the live mode
at every dispatch, and the roster asks the same executor, so what is advertised is what may run.
"""

from __future__ import annotations

from typing import Any

import pytest

from maxim.runtime.executor import Executor
from maxim.tools.base import Tool, ToolResult
from maxim.tools.registry import ToolRegistry


class _Recorder(Tool):
    input_schema: dict[str, Any] = {}

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name
        super().__init__()
        self.calls = 0

    def execute(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(success=True, output="ran")


def _executor(mode: dict[str, str], *names: str) -> tuple[Executor, dict[str, _Recorder]]:
    registry = ToolRegistry()
    tools = {name: _Recorder(name) for name in names}
    for tool in tools.values():
        registry.register(tool)
    executor = Executor(registry)
    executor.set_mode_source(lambda: mode["name"])
    return executor, tools


def test_a_tool_outside_the_mode_is_refused_at_dispatch() -> None:
    executor, tools = _executor({"name": "passive"}, "respond", "bash")
    refused = executor.execute({"tool_name": "bash", "params": {}})
    assert refused.success is False and "not available in passive mode" in (refused.error or "")
    assert tools["bash"].calls == 0  # the side effect, not only the status
    assert executor.execute({"tool_name": "respond", "params": {}}).success is True


def test_a_mode_forbidden_tool_is_refused_too() -> None:
    executor, tools = _executor({"name": "passive"}, "maxim_command")
    assert executor.execute({"tool_name": "maxim_command", "params": {}}).success is False
    assert tools["maxim_command"].calls == 0


def test_the_gate_reads_the_live_mode() -> None:
    mode = {"name": "passive"}
    executor, tools = _executor(mode, "bash")
    assert executor.execute({"tool_name": "bash", "params": {}}).success is False
    mode["name"] = "active"  # active allows everything it does not forbid
    assert executor.execute({"tool_name": "bash", "params": {}}).success is True
    assert tools["bash"].calls == 1


def test_what_is_advertised_is_what_may_run() -> None:
    """The prompt roster filters on ``permits``; dispatch refuses exactly the same set."""
    names = ("respond", "read_file", "bash", "edit_file", "git_commit", "search_code", "request_interaction")
    executor, tools = _executor({"name": "passive"}, *names)
    for name in names:
        ran = executor.execute({"tool_name": name, "params": {}}).success
        assert ran is executor.permits(name), name
    assert not executor.permits("bash") and executor.permits("request_interaction")


def test_no_mode_source_means_no_mode_restriction() -> None:
    executor, _ = _executor({"name": "passive"}, "bash")
    executor.set_mode_source(None)
    assert executor.execute({"tool_name": "bash", "params": {}}).success is True


def test_the_real_passive_registry_refuses_its_acting_tools() -> None:
    """The issue's reproduction path: build_tool_registry(passive) + build_executor, as the CLI does."""
    from maxim.runtime.bootstrap import build_executor, build_tool_registry

    registry = build_tool_registry(operational_mode="passive")
    executor = build_executor(registry, pain_bus=None, permissions=None)
    executor.set_mode_source(lambda: "passive")
    registered = set(registry._tools)
    for acting in {"bash", "edit_file", "git_commit", "run_tests", "execute_file", "maxim_command"} & registered:
        assert not executor.permits(acting), acting
    for allowed in {"respond", "search_code", "request_interaction"} & registered:
        assert executor.permits(allowed), allowed


def test_the_bodys_always_active_tools_pass_the_mode(monkeypatch) -> None:
    """SEM always-active tools join past the mode filter by design (sem_motor_binding Phase 1)."""
    import maxim.embodiment.tool_bridge as tool_bridge

    executor, tools = _executor({"name": "passive"}, "reachy_mini_turn_left")
    executor.embodiment = object()
    monkeypatch.setattr(tool_bridge, "always_active_sem_tools", lambda registry: [tools["reachy_mini_turn_left"]])
    assert executor.execute({"tool_name": "reachy_mini_turn_left", "params": {}}).success is True


def test_the_agent_loop_wires_the_gate_to_its_own_mode() -> None:
    """The real consumer: run_agentic_loop hands the executor a source reading ITS state's mode."""
    from maxim.runtime.agent_loop import run_agentic_loop

    class _Wired(Exception):
        pass

    class _SpyExecutor:
        source = None

        def set_mode_source(self, source):
            _SpyExecutor.source = source
            raise _Wired

    class _State:
        data = {"mode": "passive"}

    with pytest.raises(_Wired):
        run_agentic_loop(None, None, _State(), None, None, _SpyExecutor())
    assert _SpyExecutor.source is not None and _SpyExecutor.source() == "passive"
    _State.data["mode"] = "active"
    assert _SpyExecutor.source() == "active"
