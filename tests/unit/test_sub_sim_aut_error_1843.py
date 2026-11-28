"""A sub-simulation whose AUT loop dies says so (#1843 review, #863's fail-loud loop).

``SpawnSubSimulationTool`` runs the sub-AUT's ``run_agentic_loop`` on its own thread. Its handler used to append the
exception to a list nothing read, with no log, so a dead sub-AUT looked like a slow one: ``send_and_wait`` returned
``timed_out``. Now the failure is logged at ERROR with its traceback and named in the sub-report as ``aut_error``.
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


class _Worker:
    def __init__(self, *a: Any, **kw: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self, *a: Any, **k: Any) -> None:
        pass


def test_a_sub_aut_whose_loop_dies_is_logged_and_named_in_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import maxim.agents.llm_worker as LW
    import maxim.runtime.agent_loop as agent_loop
    import maxim.simulation.bridge as bridge_mod
    from maxim.simulation.tools import SpawnSubSimulationTool

    monkeypatch.chdir(tmp_path)

    def _dies(*a: Any, **k: Any) -> None:
        raise RuntimeError("perception trace fault")

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
    with caplog.at_level(logging.ERROR, logger="maxim.simulation.tools"):
        out = Executor(ToolRegistry([tool]), policy=policy).execute(
            {"tool_name": "spawn_sub_simulation", "params": {"goal": "hello"}}
        )
    report = out.output if hasattr(out, "output") else out
    assert report["aut_error"] == "RuntimeError: perception trace fault", report
    [rec] = [r for r in caplog.records if r.getMessage().startswith("Sub-simulation AUT loop failed")]
    assert rec.levelno == logging.ERROR and rec.exc_info is not None
