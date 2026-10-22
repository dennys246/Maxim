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

import pytest

RED = pytest.mark.xfail(strict=True, reason="#1412 red gate: the roster ignores the autonomy forbid list")


def _controller(forbidden: set[str]):
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SafetyConstraints

    return AutonomyController(
        initial_level=AutonomyLevel.AUTONOMOUS,
        safety_constraints=SafetyConstraints(forbidden_tools=frozenset(forbidden)),
    )


@RED
def test_an_autonomy_forbidden_tool_is_not_advertised() -> None:
    from maxim.runtime import agent_loop

    executor = SimpleNamespace(permits=lambda _t: True)
    tools = ["respond", "bash", "read_file", "write_file"]
    shown = agent_loop._advertisable_tools(tools, executor, _controller({"bash", "write_file"}))
    assert shown == ["respond", "read_file"]


@RED
def test_the_forbid_check_is_static_not_the_moment() -> None:
    """A pause blocks execution for now; it must not reshape the roster (the tool comes back when unpaused)."""
    ctrl = _controller({"bash"})
    ctrl.emergency_halt("test")  # pauses execution
    assert ctrl.forbids_tool("bash") is True
    assert ctrl.forbids_tool("respond") is False


@RED
def test_both_roster_sites_in_the_loop_use_the_one_filter() -> None:
    """The prompt roster and the loop-cap message's "available tools" list: both advertise tools to the model."""
    from maxim.runtime import agent_loop

    src = inspect.getsource(agent_loop.run_agentic_loop)
    assert src.count("_advertisable_tools(") == 2
    assert "executor.permits(t)" not in src
