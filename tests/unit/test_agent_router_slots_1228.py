"""#1228 fold batch: who hands an agent its router, and what a run that cannot build the AUT's router does.

- ``MaximAgent(shared_router=)`` and ``ExecAgent(shared_router=)`` are REQUIRED keywords (a forgotten router is a
  TypeError, never a silent private build). ``None`` means "no LLM, build none"; only ``exec_agent.BUILD_OWN`` (the
  CLI's consent-asking path) builds one lazily, so a router variable that happens to be None builds nothing.
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


def test_exec_agent_requires_its_router_choice() -> None:
    from maxim.agents.bus import AgentBus
    from maxim.agents.exec_agent import ExecAgent

    with pytest.raises(TypeError, match="shared_router"):
        ExecAgent(AgentBus(), MagicMock(name="memory"))  # type: ignore[call-arg]


def _record_builds(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Spy every way ExecAgent could build an LLM of its own; recorded, not raised (``_ensure_router`` catches a
    config error and returns None, which would hide a raise)."""
    import maxim.agents.exec_agent as EA
    import maxim.agents.exec_agent_router as R

    built: list[str] = []

    def _must_not_build(name: str) -> Any:
        def _build(*_a: Any, **_k: Any) -> Any:
            built.append(name)
            raise AssertionError(f"an agent handed None reached {name}")

        return _build

    for name in ("load_llm_config", "consented_router", "ChatLLMAgent"):
        monkeypatch.setattr(EA, name, _must_not_build(name))
    monkeypatch.setattr(R, "configured_override", lambda: None)
    return built


def test_none_builds_no_router_and_no_private_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """``shared_router=None`` (a run without an LLM, or a router variable that happens to be None): ExecAgent never
    reaches ``load_llm_config``, ``consented_router`` or a ChatLLMAgent, also after ``adopt_router(None, ...)``."""
    from maxim.agents.maxim_agent import MaximAgent

    built = _record_builds(monkeypatch)
    agent = MaximAgent(shared_router=None)
    assert agent.exec_agent._ensure_router() is None
    assert agent.exec_agent._ensure_router(consent="ask") is None
    assert agent.exec_agent._ensure_llm() is None
    agent.exec_agent.adopt_router(None, MagicMock(name="worker"))  # adopting None never re-enables a lazy build
    assert agent.exec_agent._ensure_router() is None
    assert agent.exec_agent._ensure_llm() is None
    assert built == []


def test_only_build_own_builds_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.agents.exec_agent as EA
    from maxim.agents.maxim_agent import MaximAgent

    built = _record_builds(monkeypatch)
    agent = MaximAgent(shared_router=EA.BUILD_OWN)
    assert agent.exec_agent._ensure_router() is None  # the build was attempted (and refused by the spy)
    assert built == ["load_llm_config"]


def test_a_handed_router_is_the_exec_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.agents.exec_agent as EA
    from maxim.agents.maxim_agent import MaximAgent

    monkeypatch.setattr(EA, "consented_router", lambda *a, **k: (_ for _ in ()).throw(AssertionError("built")))
    router = _Router("mine")
    agent = MaximAgent(shared_router=router)
    assert agent.exec_agent._ensure_router() is router


def test_adopt_router_hands_a_late_router_and_worker() -> None:
    """The robot runtime builds its LLM after its agent (``agentic_runtime``): adopt it, and the worker is not the
    agent's to stop. A router the agent already holds is kept."""
    from maxim.agents.maxim_agent import MaximAgent

    agent = MaximAgent(shared_router=None)
    late, worker = _Router("late"), MagicMock(name="worker")
    agent.exec_agent.adopt_router(late, worker)
    assert agent.exec_agent._ensure_router() is late
    assert agent.exec_agent._llm_worker is worker and agent.exec_agent._owns_llm_worker is False
    agent.exec_agent.adopt_router(_Router("other"))
    assert agent.exec_agent._ensure_router() is late


def test_a_bypassed_exec_agent_override_is_warned_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import maxim.agents.exec_agent_router as R
    from maxim.agents.maxim_agent import MaximAgent

    monkeypatch.setattr(R, "_BYPASSED_OVERRIDES_WARNED", set())
    reads: list[int] = []

    def _cfg(*_a: Any, **_k: Any) -> Any:
        reads.append(1)
        return SimpleNamespace(agent_profiles={"exec_agent": "big"})

    monkeypatch.setattr(R, "load_llm_config", _cfg)
    monkeypatch.setattr(R, "_OVERRIDE_READ", None)
    with caplog.at_level(logging.WARNING, logger="maxim.agents.exec_agent_router"):
        MaximAgent(shared_router=_Router("small"))
        MaximAgent(shared_router=_Router("small"))
        MaximAgent(shared_router=_Router("big"))  # the override's own profile: nothing bypassed
    warned = [r for r in caplog.records if "agent_profiles.exec_agent" in r.getMessage()]
    assert len(warned) == 1, [r.getMessage() for r in warned]
    assert len(reads) == 1  # read once per llm.json identity, not per constructor


def test_an_llm_json_edit_reaches_the_override_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``agent_profiles`` lives in llm.json, which no writer invalidates (the downloader's ``atomic_write_json``, a hand
    edit): rewriting the file between two agents, with no manual invalidate, gives the next constructor the new
    override (a long-lived ``maxim serve``)."""
    import json
    import os

    import maxim.agents.exec_agent_router as R
    from maxim.agents.maxim_agent import MaximAgent

    llm_json = tmp_path / "llm.json"
    monkeypatch.setenv("MAXIM_LLM_CONFIG", str(llm_json))  # the first discovery candidate
    monkeypatch.setattr(R, "_BYPASSED_OVERRIDES_WARNED", set())
    monkeypatch.setattr(R, "_OVERRIDE_READ", None)

    def _write(override: str, mtime_s: int) -> None:
        llm_json.write_text(json.dumps({"agent_profiles": {"exec_agent": override}}), encoding="utf-8")
        os.utime(llm_json, (mtime_s, mtime_s))  # a distinct mtime, however fast the two writes land

    with caplog.at_level(logging.WARNING, logger="maxim.agents.exec_agent_router"):
        _write("small", 1_700_000_000)
        MaximAgent(shared_router=_Router("small"))  # the override's own profile: nothing bypassed
        _write("big", 1_700_000_100)
        MaximAgent(shared_router=_Router("small"))
    warned = [r.getMessage() for r in caplog.records if "agent_profiles.exec_agent" in r.getMessage()]
    assert len(warned) == 1 and "'big'" in warned[0], warned


def test_starting_an_agent_handed_none_warns_once_naming_build_own(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``shared_router=None`` meant the lazy build on 1.3.3; now it means no LLM. ``ExecAgent.on_start`` (the only
    production starter of goal proposals) says so once per process, naming ``BUILD_OWN``."""
    import maxim.agents.exec_agent_router as R
    from maxim.agents.maxim_agent import MaximAgent

    monkeypatch.setattr(R, "_NO_ROUTER_START_WARNED", False)
    agents = [MaximAgent(shared_router=None), MaximAgent(shared_router=None), MaximAgent(shared_router=_Router("r"))]
    try:
        with caplog.at_level(logging.WARNING, logger="maxim.agents.exec_agent_router"):
            for agent in agents:
                agent.exec_agent.on_start()
    finally:
        for agent in agents:
            agent.exec_agent.on_stop()
    warned = [r.getMessage() for r in caplog.records if "shared_router=None" in r.getMessage()]
    assert len(warned) == 1 and "BUILD_OWN" in warned[0], warned


def test_a_nested_backend_errors_fix_hint_is_not_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inner ``BackendError`` enters the refusal as its ``problem`` (no fix hint), so ``str()`` carries one hint."""
    import maxim.runtime.lane_backends as LB
    from maxim.models.language.types import AutRouterUnavailable, BackendDown
    from maxim.simulation import orchestrator as orch

    inner = BackendDown("peer-lane")

    def _down(**_k: Any) -> Any:
        raise inner

    monkeypatch.setattr(LB, "build_run_router", _down)
    with pytest.raises(AutRouterUnavailable) as exc:
        orch._build_aut_router("aut-model", _Router("shared"), warm=False)
    assert inner.problem in str(exc.value)
    assert inner.fix_hint not in str(exc.value)


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
    reached: list[Any] = []
    monkeypatch.setattr(agent_loop, "run_agentic_loop", lambda *a, **k: reached.append(k.get("llm_worker")))
    h = _handle()
    talk = _Router("talk")
    worker = MagicMock(name="worker")
    worker.router = talk
    worker.acting_coach, worker.is_embodied = None, False  # LLMWorker's defaults: run_loop_stack checks both
    with h._talk_lock:
        h._launch_talk_loop(worker, tmp_path / "ws")
    if h._talk_thread is not None:
        h._talk_thread.join(timeout=10)
    assert held == [talk], held
    # the loop was really reached: a raise inside the talk thread is swallowed into logger.exception, so without
    # this the router assertion above would pass with the loop never run (loop_stack_builder.md, delta 2 F3)
    assert reached == [worker], reached


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
    # str() carries the build's cause (benchmarks record str(e) as RunResult.error), and the fix hint once.
    cause = "peer unreachable" if build == "raises" else "returned no router"
    assert cause in str(exc.value) and cause in exc.value.problem
    assert exc.value.fix_hint not in exc.value.problem and str(exc.value).count(exc.value.fix_hint) == 1
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
    # No narrator LLM: the named AUT model is still built (it used to be skipped, leaving the AUT with none).
    assert orch._build_aut_router("aut-model", None, warm=False) is built


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


@pytest.mark.parametrize("which", ["aut", "local"])
def test_the_cli_prints_a_backend_errors_fix_hint_once(which: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """``str(e)`` ends with the fix hint, and the CLI's ``fix=`` line prints it: the message is ``e.problem``, so the
    hint appears once (it appeared twice, for ``LocalModelUnavailable`` too)."""
    from maxim import cli
    from maxim.models.language.types import AutRouterUnavailable, LocalModelUnavailable

    err = AutRouterUnavailable("aut-model", reason="peer down") if which == "aut" else LocalModelUnavailable("m")

    def _raise(_argv: Any) -> int:
        raise err

    warned: list[tuple[str, Any]] = []
    monkeypatch.setattr(cli, "_main_impl", _raise)
    monkeypatch.setattr("maxim.utils.logging.user_warn", lambda msg, **kw: warned.append((msg, kw.get("fix"))))
    assert cli._main_guarded([]) == 2
    [(msg, fix)] = warned
    assert fix == err.fix_hint and err.fix_hint not in msg
    assert type(err).__name__ in msg
    if which == "aut":
        assert "peer down" in msg


def test_build_own_names_itself() -> None:
    """The sentinel's repr is its public name, so a log line or a test failure that shows it reads ``BUILD_OWN``."""
    import maxim.agents.exec_agent_router as R

    assert repr(R.BUILD_OWN) == "BUILD_OWN"


def test_override_in_reads_only_a_named_string_profile() -> None:
    """``agent_profiles`` that is not a dict names no override; ``executive`` is the fallback key; blank is none."""
    import maxim.agents.exec_agent_router as R

    assert R.override_in(SimpleNamespace(agent_profiles=["exec_agent"])) is None
    assert R.override_in(SimpleNamespace(agent_profiles={"executive": " big "})) == "big"
    assert R.override_in(SimpleNamespace(agent_profiles={"exec_agent": "  "})) is None


def test_no_llm_json_has_no_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no discovery candidate on disk the cache key is None (no file to watch), not an error."""
    import maxim.agents.exec_agent_router as R
    import maxim.models.language.config as C

    monkeypatch.setattr(C, "_llm_config_candidates", lambda: [str(tmp_path / "absent.json"), ""])
    assert R._llm_json_identity() is None


def test_an_llm_json_that_vanishes_before_stat_has_no_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The file passes the isfile check but is gone by the stat (removed or replaced in between): no identity, not a
    raise, and the later candidates are not consulted, since the first existing one is the file load_llm_config
    reads."""
    import maxim.agents.exec_agent_router as R
    import maxim.models.language.config as C

    vanished, later = tmp_path / "llm.json", tmp_path / "later.json"
    later.write_text("{}")
    monkeypatch.setattr(C, "_llm_config_candidates", lambda: [str(vanished), str(later)])
    monkeypatch.setattr(R.os.path, "isfile", lambda path: True)
    assert R._llm_json_identity() is None


def test_an_unreadable_config_names_no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config that cannot be read gives no override (the build path reports it), and the result is cached for that
    llm.json identity, so a constructor does not re-read a broken file."""
    import maxim.agents.exec_agent_router as R

    calls: list[int] = []

    def _broken(*_a: Any, **_k: Any) -> Any:
        calls.append(1)
        raise ValueError("bad llm.json")

    monkeypatch.setattr(R, "load_llm_config", _broken)
    monkeypatch.setattr(R, "_llm_json_identity", lambda: ("llm.json", 1, 2))
    monkeypatch.setattr(R, "_OVERRIDE_READ", None)
    assert R.configured_override() is None
    assert R.configured_override() is None
    assert calls == [1]


def test_an_aut_fear_gate_that_cannot_be_wired_logs_and_leaves_the_executor_unwrapped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``_fear_gate_aut_executor`` (moved out of ``start_simulation_mode``, rule T): a FearAgent that cannot be built
    on the AUT's router is a logged WARNING and the AUT keeps its unwrapped executor, as before the move."""
    import maxim.agents.fear_agent as FA
    from maxim.simulation import orchestrator

    def _refuse(self: Any, *_a: Any, **_k: Any) -> None:
        raise RuntimeError("no fear agent today")

    monkeypatch.setattr(FA.FearAgent, "__init__", _refuse)
    executor = object()
    with caplog.at_level(logging.WARNING, logger=orchestrator.logger.name):
        assert orchestrator._fear_gate_aut_executor(executor, _Router("aut")) is executor
    assert any(
        "Failed to wire FearGatedExecutor for AUT" in r.getMessage() and "no fear agent today" in r.getMessage()
        for r in caplog.records
    ), caplog.text
