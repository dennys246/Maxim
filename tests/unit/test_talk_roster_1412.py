"""#1412: the prompt roster never advertises a tool the autonomy controller forbids.

Found by maxim-pulse on the 1.3.3 candidate: Console Talk forbids every non-conversational tool through
``SafetyConstraints.forbidden_tools`` (``console/handle.py::_launch_talk_loop``), but the agent loop filtered the
prompt's roster only through ``executor.permits`` (the mode and ``AgentPermissions``, bugs ledger D82). The model was
shown ``bash``, chose it to fetch news, and was refused. D82's rule ("advertise only what will be let through") now
covers autonomy forbids too: the static ``SafetyConstraints.forbidden_tools``, never the moment's pause or rate.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace


def _controller(forbidden: set[str]):
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SafetyConstraints

    return AutonomyController(
        initial_level=AutonomyLevel.AUTONOMOUS,
        safety_constraints=SafetyConstraints(forbidden_tools=frozenset(forbidden)),
    )


def test_an_autonomy_forbidden_tool_is_not_advertised() -> None:
    from maxim.runtime import agent_loop

    executor = SimpleNamespace(permits=lambda _t: True)
    tools = ["respond", "bash", "read_file", "write_file"]
    shown = agent_loop._advertisable_tools(tools, executor, _controller({"bash", "write_file"}))
    assert shown == ["respond", "read_file"]


def test_the_forbid_check_is_static_not_the_moment() -> None:
    """A pause blocks execution for now; it must not reshape the roster (the tool comes back when unpaused)."""
    ctrl = _controller({"bash"})
    ctrl.emergency_halt("test")  # pauses execution
    assert ctrl.forbids_tool("bash") is True
    assert ctrl.forbids_tool("respond") is False


def test_both_roster_sites_in_the_loop_use_the_one_filter() -> None:
    """The prompt roster and the loop-cap message's "available tools" list: both advertise tools to the model."""
    from maxim.runtime import agent_loop

    src = inspect.getsource(agent_loop.run_agentic_loop)
    assert src.count("_advertisable_tools(") == 2
    assert "executor.permits(t)" not in src


# ── review-round folds: one gate (the executor), the real Talk controller ──────────────────────────────


def _talk_executor():
    from maxim.console.handle import talk_autonomy
    from maxim.runtime.executor import Executor
    from maxim.runtime.loop_setup import _prepare_executor
    from maxim.runtime.state import RuntimeState
    from maxim.tools.registry import ToolRegistry
    from maxim.tools.filesystem import BashTool, ReadFileTool

    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(ReadFileTool())
    autonomy = talk_autonomy(registry)
    state = RuntimeState()
    state.data["mode"] = "active"  # Talk's own run mode (console/handle.py): the mode alone lets bash run
    executor = _prepare_executor(Executor(registry), None, state, autonomy_controller=autonomy)
    return executor, autonomy, registry


def test_the_real_talk_controller_hides_bash_from_the_roster_and_the_executor() -> None:
    """The composition pulse hit: console/handle.py's Talk controller, wired by the loop's own setup."""
    from maxim.runtime import agent_loop

    executor, autonomy, registry = _talk_executor()
    assert "bash" in registry.list_all()  # the tool exists; Talk must still never be shown it
    from maxim.runtime.executor import Executor
    from maxim.runtime.loop_setup import _prepare_executor
    from maxim.runtime.state import RuntimeState

    control_state = RuntimeState()
    control_state.data["mode"] = "active"
    assert _prepare_executor(Executor(registry), None, control_state).permits("bash")  # control: no forbid wired
    assert not executor.permits("bash")
    shown = agent_loop._advertisable_tools(registry.list_all(), executor, autonomy)
    assert "bash" not in shown and "read_file" in shown


def test_no_executor_error_suggests_a_forbidden_tool() -> None:
    """The second channel both review lenses found: an unknown tool's 'Did you mean' and 'Available tools' lists."""
    executor, _autonomy, _registry = _talk_executor()
    assert "bash" not in executor._runnable_tools()
    assert all(
        not executor.permits(t) or t in {"respond", "speak", "read_file", "glob", "sense_tools"}
        for t in executor._runnable_tools()
    )
