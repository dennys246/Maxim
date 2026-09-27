"""#826: a mode's limits are enforced at DISPATCH, not only in the prompt.

A mode's tool lists shaped only the prompt roster: a tool the mode excludes still executed when the
model named it (reproduced: a mode-forbidden ``maxim_command`` reached its tool at AUTONOMOUS in passive
mode). The executor now reads the live mode at every dispatch and refuses by CAPABILITY (owner decision
2026-09-26): the mode's forbidden tools and what its capabilities exclude -- for passive, the tools that
act on the host. The hand-written allow-list shapes the prompt only, so memory, introspection, protocol
and user-registered tools stay usable; the roster filters on the same executor, so it never advertises
a tool dispatch would refuse.
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


@pytest.mark.parametrize("tool", ["bash", "edit_file", "git_commit", "run_tests", "execute_sandbox_script"])
def test_passive_refuses_a_tool_that_acts_on_the_host(tool) -> None:
    executor, tools = _executor({"name": "passive"}, "respond", tool)
    refused = executor.execute({"tool_name": tool, "params": {}})
    assert refused.success is False and "passive mode does not allow" in (refused.error or "")
    assert tools[tool].calls == 0  # the side effect, not only the status
    assert executor.execute({"tool_name": "respond", "params": {}}).success is True


def test_a_mode_forbidden_tool_is_refused() -> None:
    """The reproduced case: maxim_command, forbidden in passive, reached its tool before #826."""
    executor, tools = _executor({"name": "passive"}, "maxim_command")
    refused = executor.execute({"tool_name": "maxim_command", "params": {}})
    assert refused.success is False and "forbidden in passive mode" in (refused.error or "")
    assert tools["maxim_command"].calls == 0


@pytest.mark.parametrize("tool", ["memory_recall", "predict_outcome", "search_code", "my_user_tool"])
def test_passive_still_runs_what_a_runtime_or_user_registered(tool) -> None:
    """The allow-list shapes the prompt only: memory, introspection, read-only and user tools run."""
    executor, tools = _executor({"name": "passive"}, tool)
    assert executor.execute({"tool_name": tool, "params": {}}).success is True
    assert tools[tool].calls == 1


def test_the_gate_reads_the_live_mode() -> None:
    mode = {"name": "passive"}
    executor, tools = _executor(mode, "bash")
    assert executor.execute({"tool_name": "bash", "params": {}}).success is False
    mode["name"] = "active"  # active may act on the host
    assert executor.execute({"tool_name": "bash", "params": {}}).success is True
    assert tools["bash"].calls == 1


def test_active_mode_is_unchanged() -> None:
    """Active has can_execute_code=False, which only filters its prompt roster (as before #826): every
    tool that ran in active when named still runs -- execute_file included (the sim harnesses allow it)."""
    executor, tools = _executor({"name": "active"}, "execute_sandbox_script", "bash", "edit_file", "execute_file")
    for name in tools:
        assert executor.execute({"tool_name": name, "params": {}}).success is True, name


def test_permits_and_dispatch_agree() -> None:
    """The prompt roster filters on ``permits``; dispatch refuses exactly what it refuses."""
    names = ("respond", "read_file", "bash", "edit_file", "maxim_command", "search_code", "memory_recall")
    executor, _ = _executor({"name": "passive"}, *names)
    for name in names:
        assert executor.execute({"tool_name": name, "params": {}}).success is executor.permits(name), name


def test_a_hallucinated_name_gets_suggestions_it_could_run() -> None:
    """An unknown name still takes the not-registered path, and nothing the mode refuses is suggested."""
    executor, _ = _executor({"name": "passive"}, "bash", "bash_history", "respond")
    executor._consecutive_failures = 5  # past the threshold where the full list is shown
    result = executor.execute({"tool_name": "bashh", "params": {}})
    assert result.success is False and "not registered" in (result.error or "")
    assert "bash," not in (result.error or "") and "bash_history" in (result.error or "")


def test_no_mode_source_means_no_mode_restriction() -> None:
    executor, _ = _executor({"name": "passive"}, "bash")
    executor.set_mode_source(None)
    assert executor.execute({"tool_name": "bash", "params": {}}).success is True


def test_the_real_passive_registry() -> None:
    """The reproduction path: build_tool_registry(passive) + build_executor, as the CLI does."""
    from maxim.runtime.bootstrap import build_executor, build_tool_registry

    registry = build_tool_registry(operational_mode="passive")
    executor = build_executor(registry, pain_bus=None, permissions=None)
    executor.set_mode_source(lambda: "passive")
    registered = set(registry._tools)
    for acting in {"bash", "edit_file", "git_commit", "run_tests", "execute_file", "maxim_command"} & registered:
        assert not executor.permits(acting), acting
    for usable in {"respond", "search_code", "request_interaction", "memory_recall", "think"} & registered:
        assert executor.permits(usable), usable


@pytest.mark.parametrize(("run_mode", "bash_runs"), [("live", True), ("passive", False), ("agentic", True)])
def test_the_robot_runtimes_run_mode_reaches_the_gate(run_mode, bash_runs) -> None:
    """The embodied runtime seeds state.data["mode"] from its run mode; before, every robot run but
    exploration fell back to "observe" (passive). "agentic" names no mode table entry: unrestricted,
    as the roster treats it."""
    from maxim.embodied_runtime.agentic_runtime import seed_run_mode

    class _State:
        data: dict = {}

    state = _State()
    state.data = {}
    seed_run_mode(state, run_mode)
    executor, _ = _executor({"name": "unused"}, "bash")
    executor.set_mode_source(lambda: state.data.get("mode", "observe"))
    assert executor.execute({"tool_name": "bash", "params": {}}).success is bash_runs


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
