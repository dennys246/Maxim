"""#1228 fold batch: who hands an agent its router, and what a run that cannot build the AUT's router does.

- ``MaximAgent(shared_router=)`` is a REQUIRED keyword (a forgotten router is a TypeError, never a silent private
  build), and ``exec_agent.NO_ROUTER`` means "no LLM, build none".
- The console's talk agent and a spawned sub-simulation's agent are handed their runtime's router (owner decision A:
  the sub-AUT is an agent under test, so all of it runs on the router the orchestrator hands it).
- ``ExecAgent.adopt_router`` is the robot runtime's late hand-off; an ``agent_profiles.exec_agent`` override a handed
  router bypasses is warned about once.
- ``--aut-model`` with an AUT router that cannot be built refuses the run, typed (owner decision B).
- The report's cost totals include the AUT's own router, split by role (owner decision C).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


class _Router:
    """A router stand-in that says whose it is."""

    n_ctx = 4096

    def __init__(self, profile: str, *, cost: float = 0.0, tokens: tuple[int, int] = (0, 0)) -> None:
        self.cfg = SimpleNamespace(profile=profile)
        self.session_cost = cost
        self._cost_tracker = SimpleNamespace(
            get_session_tokens=lambda: {"input_tokens": tokens[0], "output_tokens": tokens[1]}
        )

    def get_token_counter(self) -> Any:
        return None

    def __repr__(self) -> str:
        return f"_Router({self.cfg.profile!r})"


# ── MaximAgent / ExecAgent: the router is a stated choice ────────────────────


def test_maxim_agent_requires_its_router_choice() -> None:
    from maxim.agents.maxim_agent import MaximAgent

    with pytest.raises(TypeError, match="shared_router"):
        MaximAgent()  # type: ignore[call-arg]


def test_no_router_builds_no_router_and_no_private_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """``NO_ROUTER`` (a run without an LLM): ExecAgent never reaches ``load_llm_config`` or a ChatLLMAgent."""
    import maxim.agents.exec_agent as EA
    from maxim.agents.maxim_agent import MaximAgent

    built: list[str] = []  # recorded, not raised: _ensure_router catches a config error and returns None

    def _must_not_build(name: str) -> Any:
        def _build(*_a: Any, **_k: Any) -> Any:
            built.append(name)
            raise AssertionError(f"an agent handed NO_ROUTER reached {name}")

        return _build

    monkeypatch.setattr(EA, "load_llm_config", _must_not_build("load_llm_config"))
    monkeypatch.setattr(EA, "consented_router", _must_not_build("consented_router"))
    monkeypatch.setattr(EA, "ChatLLMAgent", _must_not_build("ChatLLMAgent"))
    agent = MaximAgent(shared_router=EA.NO_ROUTER)
    assert agent.exec_agent._ensure_router() is None
    assert agent.exec_agent._ensure_router(consent="ask") is None
    assert agent.exec_agent._ensure_llm() is None
    assert built == []
    assert EA.agent_router(None) is EA.NO_ROUTER


def test_a_handed_router_is_the_exec_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.agents.exec_agent as EA
    from maxim.agents.maxim_agent import MaximAgent

    monkeypatch.setattr(EA, "consented_router", lambda *a, **k: (_ for _ in ()).throw(AssertionError("built")))
    router = _Router("mine")
    agent = MaximAgent(shared_router=router)
    assert agent.exec_agent._ensure_router() is router
    assert EA.agent_router(router) is router


def test_adopt_router_hands_a_late_router_and_worker() -> None:
    """The robot runtime builds its LLM after its agent (``agentic_runtime``): adopt it, and the worker is not the
    agent's to stop. A router the agent already holds is kept."""
    import maxim.agents.exec_agent as EA
    from maxim.agents.maxim_agent import MaximAgent

    agent = MaximAgent(shared_router=EA.NO_ROUTER)
    late, worker = _Router("late"), MagicMock(name="worker")
    agent.exec_agent.adopt_router(late, worker)
    assert agent.exec_agent._ensure_router() is late
    assert agent.exec_agent._llm_worker is worker and agent.exec_agent._owns_llm_worker is False
    agent.exec_agent.adopt_router(_Router("other"))
    assert agent.exec_agent._ensure_router() is late


def test_a_bypassed_exec_agent_override_is_warned_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import maxim.agents.exec_agent as EA
    from maxim.agents.maxim_agent import MaximAgent

    monkeypatch.setattr(EA, "_BYPASSED_OVERRIDES_WARNED", set())
    monkeypatch.setattr(EA, "load_llm_config", lambda *a, **k: SimpleNamespace(agent_profiles={"exec_agent": "big"}))
    with caplog.at_level(logging.WARNING, logger="maxim.agents.exec_agent"):
        MaximAgent(shared_router=_Router("small"))
        MaximAgent(shared_router=_Router("small"))
        MaximAgent(shared_router=_Router("big"))  # the override's own profile: nothing bypassed
    warned = [r for r in caplog.records if "agent_profiles.exec_agent" in r.getMessage()]
    assert len(warned) == 1, [r.getMessage() for r in warned]


# ── the console's talk agent ─────────────────────────────────────────────────


def test_the_console_talk_agent_is_handed_the_talk_router(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.agents.maxim_agent as MA
    from maxim.runtime import agent_loop
    from tests.unit.test_bounded_stop_handle_1536 import _handle

    held: list[Any] = []
    real = MA.MaximAgent

    def _spy(*a: Any, **kw: Any) -> Any:
        held.append(kw.get("shared_router", "<missing>"))
        return real(*a, **kw)

    monkeypatch.setattr(MA, "MaximAgent", _spy)
    monkeypatch.setattr(agent_loop, "run_agentic_loop", lambda *a, **k: None)
    h = _handle()
    talk = _Router("talk")
    worker = MagicMock(name="worker")
    worker.router = talk
    with h._talk_lock:
        h._launch_talk_loop(worker, tmp_path / "ws")
    if h._talk_thread is not None:
        h._talk_thread.join(timeout=10)
    assert held == [talk], held


# ── a spawned sub-simulation's agent (owner decision A) ───────────────────────


def test_a_sub_simulation_agent_runs_on_the_router_it_is_handed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every LLM slot of the sub-AUT (its ExecAgent, FearAgent and LLMWorker) holds the tool's router, which the
    orchestrator sets to the run's AUT router (``test_aut_llm_slots_1228.py`` pins that hand-off)."""
    import maxim.agents.fear_agent as FA
    import maxim.agents.llm_worker as LW
    import maxim.agents.maxim_agent as MA
    import maxim.runtime.agent_loop as agent_loop
    import maxim.simulation.bridge as bridge_mod
    from maxim.runtime.executor import Executor
    from maxim.runtime.tool_policy import ToolPolicy
    from maxim.simulation.tools import SpawnSubSimulationTool
    from maxim.tools.base import Effect
    from maxim.tools.registry import ToolRegistry

    monkeypatch.chdir(tmp_path)
    held: dict[str, list[Any]] = {"MaximAgent": [], "FearAgent": [], "LLMWorker": []}
    real_agent, real_fear_init = MA.MaximAgent, FA.FearAgent.__init__

    def _agent(*a: Any, **kw: Any) -> Any:
        held["MaximAgent"].append(kw.get("shared_router", "<missing>"))
        return real_agent(*a, **kw)

    def _fear_init(self: Any, *a: Any, **kw: Any) -> None:
        held["FearAgent"].append(kw.get("llm", "<missing>"))
        real_fear_init(self, *a, **kw)

    class _Worker:
        def __init__(self, *a: Any, llm: Any = None, **kw: Any) -> None:
            held["LLMWorker"].append(llm)

        def start(self) -> None:
            pass

        def stop(self, *a: Any, **k: Any) -> None:
            pass

    monkeypatch.setattr(MA, "MaximAgent", _agent)
    monkeypatch.setattr(FA.FearAgent, "__init__", _fear_init)
    monkeypatch.setattr(LW, "LLMWorker", _Worker)
    monkeypatch.setattr(agent_loop, "run_agentic_loop", lambda *a, **k: None)
    monkeypatch.setattr(
        bridge_mod.SimulationBridge,
        "send_and_wait",
        lambda self, goal: {"turn": 0, "response": None, "actions": [], "blocked": [], "timed_out": False},
    )
    aut = _Router("aut")
    tool = SpawnSubSimulationTool(llm_router=aut, sim_tmpdir=str(tmp_path), sandbox_dirs=[str(tmp_path)])
    policy = ToolPolicy.only(["spawn_sub_simulation"], ceiling=Effect.ALL, reason="#1228 slot probe")
    Executor(ToolRegistry([tool]), policy=policy).execute(
        {"tool_name": "spawn_sub_simulation", "params": {"goal": "hello"}}
    )
    if tool._sub_thread is not None:
        tool._sub_thread.join(timeout=10)
    assert held == {"MaximAgent": [aut], "FearAgent": [aut], "LLMWorker": [aut]}, held


# ── owner decision B: a named AUT model whose router cannot be built refuses ───


@pytest.mark.parametrize("build", ["returns_none", "raises"])
def test_an_aut_router_that_cannot_be_built_refuses_the_run(build: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.runtime.lane_backends as LB
    from maxim.models.language.types import AutRouterUnavailable
    from maxim.simulation import orchestrator as orch

    def _build(**_k: Any) -> Any:
        if build == "raises":
            raise RuntimeError("peer unreachable")
        return None, None

    monkeypatch.setattr(LB, "build_run_router", _build)
    shared = _Router("shared")
    with pytest.raises(AutRouterUnavailable) as exc:
        orch._build_aut_router("aut-model", shared, warm=False)
    assert exc.value.profile == "aut-model"
    with pytest.raises(AutRouterUnavailable):  # also with no shared LLM: the named model is still not served
        orch._build_aut_router("aut-model", None, warm=False)


def test_the_aut_router_build_passes_a_missing_model_through_and_shares_without_aut_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maxim.runtime.lane_backends as LB
    from maxim.models.language.types import LocalModelUnavailable
    from maxim.simulation import orchestrator as orch

    def _missing(**_k: Any) -> Any:
        raise LocalModelUnavailable("aut-model")

    monkeypatch.setattr(LB, "build_run_router", _missing)
    shared = _Router("shared")
    with pytest.raises(LocalModelUnavailable):
        orch._build_aut_router("aut-model", shared, warm=False)
    assert orch._build_aut_router(None, shared, warm=False) is shared  # one model: no AUT build at all
    built = _Router("aut")
    monkeypatch.setattr(LB, "build_run_router", lambda **_k: (built, None))
    assert orch._build_aut_router("aut-model", shared, warm=False) is built


def test_the_run_releases_its_serving_registration_when_the_aut_build_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maxim.models.language.types import AutRouterUnavailable
    from maxim.simulation import orchestrator as orch
    from maxim.simulation import report as report_mod

    released: list[Any] = []
    monkeypatch.setattr(orch, "_build_shared_router", lambda *a, **k: _Router("shared"))

    def _refuse(*_a: Any, **_k: Any) -> Any:
        raise AutRouterUnavailable("aut-model")

    monkeypatch.setattr(orch, "_build_aut_router", _refuse)
    monkeypatch.setattr(report_mod, "release_start_provenance", lambda start: released.append(start))
    start: dict[str, Any] = {"token": 1}
    with pytest.raises(AutRouterUnavailable):
        orch._build_run_llms(None, aut_model="aut-model", start_provenance=start)
    assert released == [start]


# ── owner decision C: the AUT's own calls are in the report's cost ────────────


def test_the_report_cost_includes_the_aut_router_split_by_role() -> None:
    from maxim.simulation.report import _run_costs

    narrator = _Router("narrator", cost=0.25, tokens=(100, 20))
    aut = _Router("aut", cost=0.5, tokens=(300, 40))
    two = _run_costs(narrator, aut)
    assert (two["cost_usd"], two["total_input_tokens"], two["total_output_tokens"]) == (0.75, 400, 60)
    assert (two["narrator_cost_usd"], two["narrator_input_tokens"], two["narrator_output_tokens"]) == (0.25, 100, 20)
    assert (two["aut_cost_usd"], two["aut_input_tokens"], two["aut_output_tokens"]) == (0.5, 300, 40)
    one = _run_costs(narrator, narrator)  # one model: the AUT shares the narrator's router, counted once
    assert (one["cost_usd"], one["total_input_tokens"], one["aut_cost_usd"], one["aut_input_tokens"]) == (
        0.25,
        100,
        0.0,
        0,
    )
    assert _run_costs(None, None)["cost_usd"] == 0.0


def test_the_report_fields_carry_the_split() -> None:
    from dataclasses import fields

    from maxim.simulation.report import SimulationReport

    names = {f.name for f in fields(SimulationReport)}
    for role in ("narrator", "aut"):
        assert {f"{role}_cost_usd", f"{role}_input_tokens", f"{role}_output_tokens"} <= names
