"""A sub-simulation whose AUT loop dies says so (#1843 review, #863's fail-loud loop).

``SpawnSubSimulationTool`` runs the sub-AUT's ``run_agentic_loop`` on its own thread. Its handler used to append the
exception to a list nothing read, with no log, so a dead sub-AUT looked like a slow one: ``send_and_wait`` returned
``timed_out``. Now the failure is logged at ERROR with its traceback and named in the sub-report as ``aut_error``,
reduced and cut because it reaches the LLM (#1801 F3); ``extend_simulation`` on a dead sub-AUT says so too, at once,
instead of waiting out the response timeout (#1843 delta review).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class _Router:
    n_ctx = 4096

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(profile="aut")
        self.session_cost = 0.0

    def get_token_counter(self) -> Any:
        return None


_SECRET = "hunter2-sub-aut"
_FAULT = f"perception trace fault at https://user:{_SECRET}@10.0.0.5:8100/v1"


class _Worker:
    def __init__(self, *a: Any, **kw: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self, *a: Any, **k: Any) -> None:
        pass


def _spawn_a_dying_sub_aut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    """Spawn a sub-simulation whose AUT loop raises at once, dispatched the shipped way (Executor + ToolPolicy)."""
    import maxim.agents.llm_worker as LW
    import maxim.runtime.agent_loop as agent_loop
    import maxim.simulation.bridge as bridge_mod
    from maxim.simulation.tools import SpawnSubSimulationTool

    monkeypatch.chdir(tmp_path)

    def _dies(*a: Any, **k: Any) -> None:
        raise RuntimeError(_FAULT)

    def _wait_for_the_sub_aut(self: Any, goal: str) -> dict[str, Any]:
        for t in threading.enumerate():
            if t.name == "sim.sub_aut":
                t.join(timeout=10)
        return {"turn": 0, "response": None, "actions": [], "blocked": [], "timed_out": True}

    monkeypatch.setattr(LW, "LLMWorker", _Worker)
    monkeypatch.setattr(agent_loop, "run_agentic_loop", _dies)
    monkeypatch.setattr(bridge_mod.SimulationBridge, "send_and_wait", _wait_for_the_sub_aut)
    from maxim.runtime.executor import Executor
    from maxim.runtime.tool_policy import ToolPolicy
    from maxim.tools.base import Effect
    from maxim.tools.registry import ToolRegistry

    tool = SpawnSubSimulationTool(llm_router=_Router(), sim_tmpdir=str(tmp_path), sandbox_dirs=[str(tmp_path)])
    policy = ToolPolicy.only(["spawn_sub_simulation"], ceiling=Effect.ALL, reason="#1843 sub-AUT failure probe")
    out = Executor(ToolRegistry([tool]), policy=policy).execute(
        {"tool_name": "spawn_sub_simulation", "params": {"goal": "hello"}}
    )
    return tool, (out.output if hasattr(out, "output") else out)


def test_a_sub_aut_whose_loop_dies_is_logged_and_named_in_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger="maxim.simulation.tools"):
        _tool, report = _spawn_a_dying_sub_aut(tmp_path, monkeypatch)
    err = report["aut_error"]
    assert err.startswith("RuntimeError: perception trace fault at "), report
    assert _SECRET not in err and len(err) <= 300  # reaches the LLM: reduced, then cut (#1801 F3)
    [rec] = [r for r in caplog.records if r.getMessage().startswith("Sub-simulation AUT loop failed")]
    assert rec.levelno == logging.ERROR and rec.exc_info is not None


def test_extending_a_dead_sub_aut_says_so_without_waiting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.simulation.tools import ExtendSimulationTool

    tool, report = _spawn_a_dying_sub_aut(tmp_path, monkeypatch)
    assert tool.active_sub_bridge is not None  # stored for follow-ups, as before

    def _must_not_wait(self: Any, goal: str) -> dict[str, Any]:
        raise AssertionError("waited on a dead sub-AUT")

    monkeypatch.setattr(type(tool.active_sub_bridge), "send_and_wait", _must_not_wait)
    out = ExtendSimulationTool(main_bridge=None, spawn_tool=tool).execute(goal="more")
    assert out.success and out.output["extended_sub_simulation"] is True
    assert out.output["aut_error"] == report["aut_error"] and out.output["timed_out"] is False
    tool._teardown_sub()
    assert tool.sub_aut_error() is None  # the next sub-AUT starts clean


def test_a_sub_aut_dying_during_an_extension_is_named(tmp_path: Path) -> None:
    from maxim.simulation.tools import ExtendSimulationTool, SpawnSubSimulationTool

    tool = SpawnSubSimulationTool(llm_router=None, sim_tmpdir=str(tmp_path))

    class _Bridge:
        def send_and_wait(self, goal: str) -> dict[str, Any]:
            tool._sub_error.append(RuntimeError(_FAULT))  # the sub-AUT thread's handler, mid-objective
            return {"turn": 3, "response": None, "actions": [], "blocked": [], "timed_out": True, "duration_ms": 5.0}

    tool.active_sub_bridge = _Bridge()
    out = ExtendSimulationTool(main_bridge=None, spawn_tool=tool).execute(goal="more")
    assert out.output["timed_out"] is True and out.output["aut_error"].startswith("RuntimeError: ")
    assert _SECRET not in out.output["aut_error"]
