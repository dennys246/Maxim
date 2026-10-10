"""Characterization of ``run_agentic_loop``'s SETUP block (1.3.2 decomposition, slice 1).

Pins, through the PUBLIC entry ``agent_loop.run_agentic_loop``, what the loop builds and does before
its first tick: the simulation adapter, the state path and its first persist, the defaulted autonomy
controller and evaluators, the context pool and the prefetcher, the ``LoopController``'s pass-through
wiring, the Default Network start, the bio handles (NAc, agent id, sensor encoder, situation cue),
``drive_relief_only`` per ``aut_mode``, the bio session start, the planning-liveness gate and its
"inactive" log, and the ORDER of the side effects. Written BEFORE slice 1 moved the block into
``runtime/loop_setup.py`` and kept green unchanged by that move (``docs/plans/roadmap_1_3_x.md``
§"The decomposition", coverage-first rule).

Most runs pass a pre-set ``stop_event``: the loop then builds everything, breaks at its first stop
check (step 0, before any section that thinks) and tears down -- the setup and the teardown that
consumes it, nothing else. The two ``drive_relief_only`` arms run one real tick each.

Observation is through class-level spies (``LoopController.__init__``, ``ContextPool.__init__``,
``SimulationAdapter.__init__``) and fakes passed as arguments, never through the setup's own module,
so the pins hold wherever the setup code lives.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

LOGGER = "maxim.runtime.agent_loop"
INACTIVE = "planning liveness requested but inactive"


# ── harness ──────────────────────────────────────────────────────────────────


class _Agent:
    """A plain agent: a fixed name (so the state path is predictable) and, optionally, one intent."""

    def __init__(self, name: str = "setup-probe", intent: dict[str, Any] | None = None) -> None:
        self.name = name
        self._intent = intent

    def propose_intent(self, state: Any, memory: Any) -> dict[str, Any] | None:
        intent, self._intent = self._intent, None
        return intent


class _Hub:
    """A MemoryHub stand-in that records, in order, which of the setup's attributes are read."""

    _RECORDED = ("agent_id", "nac", "hippocampus", "ec", "atl", "situation_cue")

    def __init__(
        self,
        events: list[str],
        *,
        agent_id: Any = "hub-agent-7",
        nac: Any = None,
        ec: Any = None,
        cue: Any = None,
        start_fails: bool = False,
    ) -> None:
        object.__setattr__(self, "_events", events)
        self._values = {"agent_id": agent_id, "nac": nac, "hippocampus": None, "ec": ec, "atl": None}
        self._cue = cue
        self._start_fails = start_fails
        self.ended: list[str] = []

    def __getattribute__(self, name: str) -> Any:
        if name in _Hub._RECORDED:
            object.__getattribute__(self, "_events").append(f"hub.{name}")
            if name == "situation_cue":
                cue = object.__getattribute__(self, "_cue")
                if isinstance(cue, Exception):
                    raise cue
                return cue
            return object.__getattribute__(self, "_values")[name]
        return object.__getattribute__(self, name)

    def on_session_start(self) -> dict[str, Any]:
        state_files = list(Path("data").glob("agents/*/runtime/state_*.json"))
        self._events.append(f"hub.on_session_start(persisted={bool(state_files)})")
        if self._start_fails:
            raise RuntimeError("hub start failed")
        return {}

    def on_session_end(self) -> dict[str, Any]:
        self.ended.append("full")
        return {}

    def on_session_end_lightweight(self) -> dict[str, Any]:
        self.ended.append("lightweight")
        return {}

    def record_plan_outcome(self, **_kw: Any) -> None:
        return None


class _DN:
    def __init__(self, events: list[str], *, fails: bool = False) -> None:
        self.events = events
        self.fails = fails
        self.stops = 0

    def start(self) -> None:
        self.events.append("dn.start")
        if self.fails:
            raise RuntimeError("dn start failed")

    def stop(self) -> None:
        self.stops += 1


class _Run(SimpleNamespace):
    ctrl: Any
    pools: list[Any]
    events: list[str]
    executor: Any
    state: Any


def _spy_init(monkeypatch: pytest.MonkeyPatch, cls: type, sink: list[Any], events: list[str], tag: str) -> None:
    real = cls.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real(self, *a, **k)
        sink.append(self)
        events.append(tag)

    monkeypatch.setattr(cls, "__init__", _init)


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    stop: bool = True,
    events: list[str] | None = None,
    agent: Any = None,
    overrides: Any = None,
    **kwargs: Any,
) -> _Run:
    """Run the real loop in ``tmp_path`` (the state file is CWD-relative); return what setup built."""
    from maxim.agents.context_pool import ContextPool
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime import loop_setup
    from maxim.runtime.bootstrap import build_decision_engine, build_executor, build_memory
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.state import RuntimeState
    from maxim.tools.registry import ToolRegistry

    monkeypatch.chdir(tmp_path)
    events = [] if events is None else events
    ctrls: list[Any] = []
    pools: list[Any] = []
    _spy_init(monkeypatch, LoopController, ctrls, events, "ctrl")
    _spy_init(monkeypatch, ContextPool, pools, events, "context_pool")
    real_overrides = loop_setup.resolve_llm_loop_overrides

    def _overrides() -> Any:
        events.append("overrides")
        return real_overrides()

    monkeypatch.setattr(loop_setup, "resolve_llm_loop_overrides", overrides or _overrides)

    class _Events(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if INACTIVE in record.getMessage():
                events.append("liveness_inactive_log")

    handler = _Events(level=logging.INFO)
    log = logging.getLogger(LOGGER)
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    executor = kwargs.pop("executor", None) or build_executor(ToolRegistry(), pain_bus=None, permissions=None)
    state = RuntimeState()
    state.data["mode"] = "active"
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    stop_event = threading.Event()
    if stop:
        stop_event.set()
    kwargs.setdefault("max_steps", 1)
    kwargs.setdefault("target_hz", 200.0)
    kwargs.setdefault("idle_sleep_s", 0.0)
    try:
        AL.run_agentic_loop(
            agent if agent is not None else _Agent(),
            FileSystemEnv(str(workspace)),
            state,
            build_memory(),
            build_decision_engine(),
            executor,
            stop_event=stop_event,
            **kwargs,
        )
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)
    return _Run(ctrl=ctrls[0] if ctrls else None, pools=pools, events=events, executor=executor, state=state)


def _state_files(tmp_path: Path) -> list[Path]:
    return sorted(tmp_path.glob("data/agents/*/runtime/state_*.json"))


# ── the simulation adapter ───────────────────────────────────────────────────


@pytest.mark.parametrize("adapter", [SimpleNamespace(is_sim_mode=True), SimpleNamespace()], ids=["sim", "no-flag"])
def test_a_sim_mode_sim_adapter_is_refused_before_any_side_effect(monkeypatch, tmp_path, adapter):
    events: list[str] = []
    hub = _Hub(events)
    with pytest.raises(ValueError, match=r"sim_adapter= must be a non-sim adapter"):
        _run(monkeypatch, tmp_path, events=events, sim_adapter=adapter, memory_hub=hub)
    assert _state_files(tmp_path) == []  # refused before the first persist
    assert events == []  # no context pool, controller, DN or bio session


def test_a_non_sim_adapter_is_the_loops_adapter(monkeypatch, tmp_path):
    from maxim.runtime.sim_adapter import NullSimulationAdapter

    reads: list[str] = []

    class _Adapter(NullSimulationAdapter):
        @property
        def is_sim_mode(self) -> bool:
            reads.append("is_sim_mode")
            return False

    events: list[str] = []
    hub = _Hub(events)
    _run(monkeypatch, tmp_path, events=events, sim_adapter=_Adapter(), memory_hub=hub)
    # Beyond the refusal check's one read, the loop itself (step 0, teardown) consulted the caller's adapter.
    assert len(reads) >= 2, reads
    assert hub.ended == ["full"]  # a non-sim adapter: full consolidation


def test_a_percept_source_builds_a_sim_adapter_and_ignores_sim_adapter(monkeypatch, tmp_path):
    from maxim.runtime.sim_adapter import SimulationAdapter

    adapters: list[Any] = []
    events: list[str] = []
    _spy_init(monkeypatch, SimulationAdapter, adapters, [], "sim")
    hub = _Hub(events)
    percepts = SimpleNamespace()
    run = _run(
        monkeypatch,
        tmp_path,
        events=events,
        percept_source=percepts,
        sim_adapter=SimpleNamespace(is_sim_mode=True),  # ignored, not refused
        memory_hub=hub,
    )
    [sim] = adapters
    assert sim.percept_source is percepts
    assert sim._tool_registry is run.executor.registry
    assert hub.ended == ["lightweight"]  # a sim run: lightweight consolidation


def test_no_adapter_and_no_percept_source_is_a_null_adapter_run(monkeypatch, tmp_path):
    events: list[str] = []
    hub = _Hub(events)
    _run(monkeypatch, tmp_path, events=events, memory_hub=hub)
    assert hub.ended == ["full"]


# ── state path, first persist, run id ────────────────────────────────────────


def test_the_state_is_persisted_under_the_given_run_id(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, agent=_Agent("Probe Agent!"), run_id="r-42")
    [path] = _state_files(tmp_path)
    assert path.relative_to(tmp_path) == Path("data/agents/Probe_Agent/runtime/state_r-42.json")
    payload = json.loads(path.read_text())
    assert payload["run_id"] == "r-42" and payload["agent_name"] == "Probe_Agent"
    assert run.ctrl.run_id == "r-42"
    assert run.ctrl.state_path == os.path.join("data", "agents", "Probe_Agent", "runtime", "state_r-42.json")


def test_a_missing_run_id_is_a_timestamp(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, run_id=None)
    [path] = _state_files(tmp_path)
    assert re.fullmatch(r"state_\d{4}-\d{2}-\d{2}_\d{6}\.json", path.name)
    assert path.name == f"state_{run.ctrl.run_id}.json"
    assert json.loads(path.read_text())["run_id"] == run.ctrl.run_id


# ── defaults and the controller's wiring ─────────────────────────────────────


def test_defaults_autonomy_controller_and_evaluators(monkeypatch, tmp_path):
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel

    run = _run(monkeypatch, tmp_path, autonomy_controller=None, evaluators=None)
    assert isinstance(run.ctrl.autonomy_controller, AutonomyController)
    assert run.ctrl.autonomy_controller.current_level == AutonomyLevel.PLANNING
    assert run.ctrl.evaluators == []


def test_the_controller_receives_the_loops_arguments(monkeypatch, tmp_path):
    from maxim.agents.autonomy import AutonomyController

    controller = AutonomyController()
    evaluators = [SimpleNamespace(name="ev")]
    hippocampus = SimpleNamespace(
        start_capture_worker=lambda: None, flush=lambda timeout: None, stop_capture_worker=lambda: None
    )
    hub = _Hub([])
    worker = SimpleNamespace()
    registry = SimpleNamespace()
    on_step = SimpleNamespace()
    on_event = SimpleNamespace()
    run = _run(
        monkeypatch,
        tmp_path,
        autonomy_controller=controller,
        evaluators=evaluators,
        llm_worker=worker,
        hippocampus=hippocampus,
        memory_hub=hub,
        max_steps=7,
        on_step=on_step,
        on_event=on_event,
        idle_sleep_s=0.25,
        persist_every_n_steps=3,
        target_hz=40.0,
        use_tool_prompting=False,
        protocol_registry=registry,
    )
    c = run.ctrl
    assert c.autonomy_controller is controller and c.evaluators is evaluators
    assert c.llm_worker is worker and c.hippocampus is hippocampus and c.memory_hub is hub
    assert c.max_steps == 7 and c.on_step is on_step and c.on_event is on_event
    assert c.idle_sleep_s == 0.25 and c.persist_every_n_steps == 3 and c.target_period == 1.0 / 40.0
    assert c.use_tool_prompting is False and c.protocol_registry is registry
    assert c.executor is run.executor and c.percept_source is None and c.action_sink is None
    assert c.stop_event is not None and c.stop_event.is_set()


def test_an_action_sink_wraps_the_executor_and_the_mode_source_reads_state(monkeypatch, tmp_path):
    from maxim.simulation.instrumented_executor import InstrumentedExecutor

    sink = SimpleNamespace(record=lambda *a, **k: None)
    run = _run(monkeypatch, tmp_path, action_sink=sink)
    assert isinstance(run.ctrl.executor, InstrumentedExecutor)
    assert run.ctrl.action_sink is sink
    # The executor's dispatch mode is the loop's live state mode (#826).
    source = run.executor._mode_source
    run.state.data["mode"] = "passive"
    assert source() == "passive"


# ── the context pool and the prefetcher ──────────────────────────────────────


def test_no_context_pool_config_is_the_default_config(monkeypatch, tmp_path):
    from maxim.agents.context_pool import ContextPoolConfig

    run = _run(monkeypatch, tmp_path, context_pool_config=None)
    [pool] = run.pools
    assert pool.config == ContextPoolConfig()
    assert run.ctrl.context_pool is pool


def test_context_pool_config_keys_map_onto_the_pool_config(monkeypatch, tmp_path):
    """Every key maps onto its own field: distinct values (strings for the flags), so a swap shows."""
    from maxim.agents.context_pool import ContextPoolConfig

    cfg = {
        "max_tokens": 1111,
        "summary_target_tokens": 222,
        "max_entries": 33,
        "keep_recent": 4,
        "include_agent_states": "agents",
        "include_outcomes": "outcomes",
        "include_abstractions": "abstractions",
        "persistence_path": str(tmp_path / "pool.json"),
        "not_a_key": "ignored",
    }
    run = _run(monkeypatch, tmp_path, context_pool_config=cfg)
    [pool] = run.pools
    assert pool.config == ContextPoolConfig(
        max_tokens=1111,
        summary_target_tokens=222,
        max_entries=33,
        keep_recent=4,
        include_agent_states="agents",
        include_outcomes="outcomes",
        include_abstractions="abstractions",
        persistence_path=str(tmp_path / "pool.json"),
    )


def test_a_partial_context_pool_config_keeps_the_mapped_defaults(monkeypatch, tmp_path):
    from maxim.agents.context_pool import ContextPoolConfig

    run = _run(monkeypatch, tmp_path, context_pool_config={"max_tokens": 1234})
    [pool] = run.pools
    assert pool.config == ContextPoolConfig(max_tokens=1234)


def test_the_prefetcher_is_the_global_one_on_the_loops_executor(monkeypatch, tmp_path):
    from maxim.runtime.prefetch import get_prefetcher

    run = _run(monkeypatch, tmp_path)
    assert run.ctrl.prefetcher is get_prefetcher()
    assert run.ctrl.prefetcher._executor is run.ctrl.executor
    assert run.ctrl.prefetcher._base_path == str(tmp_path)


def test_the_teardown_saves_the_setup_pool(monkeypatch, tmp_path):
    from maxim.agents.context_pool import ContextPool

    saved: list[Any] = []
    monkeypatch.setattr(ContextPool, "save", lambda self: saved.append(self))
    run = _run(monkeypatch, tmp_path)
    assert saved == run.pools


# ── the Default Network ──────────────────────────────────────────────────────


def test_a_started_default_network_is_stopped_at_teardown(monkeypatch, tmp_path):
    events: list[str] = []
    dn = _DN(events)
    run = _run(monkeypatch, tmp_path, events=events, default_network=dn)
    assert events.count("dn.start") == 1
    assert run.ctrl.dn_enabled is True
    assert dn.stops == 1


def test_a_default_network_that_fails_to_start_is_disabled(monkeypatch, tmp_path):
    events: list[str] = []
    dn = _DN(events, fails=True)
    run = _run(monkeypatch, tmp_path, events=events, default_network=dn)
    assert events.count("dn.start") == 1
    assert run.ctrl.dn_enabled is False
    assert dn.stops == 0  # disabled: never stopped


def test_a_sim_run_starts_the_default_network_but_never_stops_it(monkeypatch, tmp_path):
    events: list[str] = []
    dn = _DN(events)
    run = _run(monkeypatch, tmp_path, events=events, default_network=dn, percept_source=SimpleNamespace())
    assert events.count("dn.start") == 1 and run.ctrl.dn_enabled is True
    assert dn.stops == 0


# ── the bio handles: NAc, agent id, sensor encoder, situation cue ────────────


def _spy_substrate(monkeypatch: pytest.MonkeyPatch, proposal: Any = None) -> list[dict[str, Any]]:
    from maxim.runtime import substrate_proposal

    calls: list[dict[str, Any]] = []

    def _propose(**kw: Any) -> Any:
        calls.append(kw)
        return proposal if len(calls) == 1 else None

    monkeypatch.setattr(substrate_proposal, "propose_via_substrate", _propose)
    return calls


def test_the_substrate_tick_receives_the_hubs_bio_handles(monkeypatch, tmp_path):
    from maxim.decisions.nac import NAc
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.similarity.encoder import SensorEncoder

    calls = _spy_substrate(monkeypatch)
    nac, ec, cue = NAc(), EntorhinalCortex(), object()
    hub = _Hub([], nac=nac, ec=ec, cue=cue)
    _run(monkeypatch, tmp_path, stop=False, memory_hub=hub, aut_mode="substrate-primary")
    [kw] = calls
    assert kw["nac"] is nac
    assert kw["agent_id"] == "hub-agent-7"
    assert kw["situation_cue"] is cue
    enc = kw["sensor_encoder"]
    assert isinstance(enc, SensorEncoder) and enc.ec is ec and enc.atl is None and enc._nac is nac


def test_with_no_hub_the_substrate_tick_gets_the_agent_name_and_no_cue(monkeypatch, tmp_path):
    from maxim.runtime.substrate_proposal import NO_SITUATION_CUE

    calls = _spy_substrate(monkeypatch)
    _run(monkeypatch, tmp_path, stop=False, agent=_Agent("Name Only"), aut_mode="substrate-primary")
    [kw] = calls
    assert kw["nac"] is None and kw["sensor_encoder"] is None
    assert kw["agent_id"] == "Name_Only"
    assert kw["situation_cue"] is NO_SITUATION_CUE


def test_a_hub_without_an_agent_id_or_a_cue_falls_back(monkeypatch, tmp_path, caplog):
    from maxim.runtime.substrate_proposal import NO_SITUATION_CUE

    calls = _spy_substrate(monkeypatch)
    hub = _Hub([], agent_id=None, cue=RuntimeError("no ATL"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _run(monkeypatch, tmp_path, stop=False, agent=_Agent("Fallback"), memory_hub=hub, aut_mode="substrate-primary")
    [kw] = calls
    assert kw["agent_id"] == "Fallback"
    assert kw["situation_cue"] is NO_SITUATION_CUE
    assert any("no situation cue this run" in r.getMessage() for r in caplog.records)


# ── drive_relief_only per aut_mode ───────────────────────────────────────────


def _spy_outcomes(monkeypatch: pytest.MonkeyPatch, tool: str) -> list[dict[str, Any]]:
    from maxim.runtime import tool_dispatch

    seen: list[dict[str, Any]] = []
    real = tool_dispatch.record_outcome

    def _spy(**kw: Any) -> Any:
        if kw.get("tool_name") == tool:
            seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(tool_dispatch, "record_outcome", _spy)
    return seen


def _probe_executor(tool: str, *, raises: bool) -> Any:
    from maxim.runtime.bootstrap import build_executor
    from maxim.tools.base import Tool, ToolOutput
    from maxim.tools.registry import ToolRegistry

    class _Probe(Tool):
        name = tool
        description = "stub"
        input_schema: dict = {}

        def execute(self, **kwargs: Any) -> Any:
            return ToolOutput(success=True, output="done")

    registry = ToolRegistry()
    registry.register(_Probe())
    executor = build_executor(registry, pain_bus=None, permissions=None)
    if raises:

        def _boom(action: Any, *a: Any, **k: Any) -> Any:
            raise RuntimeError("probe failure")

        executor.execute = _boom
    return executor


def test_substrate_primary_credits_tool_success_too(monkeypatch, tmp_path):
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.agents.llm_types import LLMProposal

    tool = "probe_relief_substrate"
    _spy_substrate(
        monkeypatch,
        LLMProposal(
            request_id="p-1",
            action={"tool_name": tool, "params": {}},
            reasoning="probe",
            strategy_used="substrate-primary",
            confidence=0.9,
            mode_goal_achieved=False,
            triggering_input="",
        ),
    )
    seen = _spy_outcomes(monkeypatch, tool)
    _run(
        monkeypatch,
        tmp_path,
        stop=False,
        max_steps=4,
        executor=_probe_executor(tool, raises=False),
        autonomy_controller=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS),
        aut_mode="substrate-primary",
    )
    assert seen, "the substrate proposal's outcome was never recorded"
    assert all(kw["drive_relief_only"] is False for kw in seen)


def test_llm_primary_credits_drive_relief_only(monkeypatch, tmp_path):
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel

    tool = "probe_relief_llm"
    seen = _spy_outcomes(monkeypatch, tool)
    _run(
        monkeypatch,
        tmp_path,
        stop=False,
        max_steps=1,
        agent=_Agent(intent={"goal": {"tool_name": tool, "params": {}}, "confidence": 0.9}),
        executor=_probe_executor(tool, raises=True),
        autonomy_controller=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS),
    )
    assert seen, "the agent-fallback outcome was never recorded"
    assert all(kw["drive_relief_only"] is True for kw in seen)


# ── the bio session ──────────────────────────────────────────────────────────


def test_a_hub_that_fails_to_start_is_not_ended(monkeypatch, tmp_path):
    hub = _Hub([], start_fails=True)
    _run(monkeypatch, tmp_path, memory_hub=hub)
    assert hub.ended == []


def test_the_hippocampus_capture_worker_starts(monkeypatch, tmp_path):
    started: list[int] = []
    hippocampus = SimpleNamespace(
        start_capture_worker=lambda: started.append(1), flush=lambda timeout: None, stop_capture_worker=lambda: None
    )
    _run(monkeypatch, tmp_path, hippocampus=hippocampus)
    assert started == [1]


# ── planning liveness: the gate's inactive log ───────────────────────────────


def _inactive(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and INACTIVE in r.getMessage()]


def test_liveness_requested_in_substrate_primary_logs_inactive(monkeypatch, tmp_path, caplog):
    _spy_substrate(monkeypatch)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _run(monkeypatch, tmp_path, planning_liveness=True, aut_mode="substrate-primary")
    assert _inactive(caplog) == [
        "planning liveness requested but inactive (aut_mode=substrate-primary, llm_worker=no, env_opt_out=no)"
    ]


def test_liveness_with_a_worker_in_substrate_primary_logs_inactive(monkeypatch, tmp_path, caplog):
    """The substrate-primary exclusion is in the gate itself, not incidental to a missing worker (S7)."""
    _spy_substrate(monkeypatch)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _run(monkeypatch, tmp_path, planning_liveness=True, llm_worker=SimpleNamespace(), aut_mode="substrate-primary")
    assert _inactive(caplog) == [
        "planning liveness requested but inactive (aut_mode=substrate-primary, llm_worker=yes, env_opt_out=no)"
    ]


def test_liveness_without_a_worker_logs_inactive(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _run(monkeypatch, tmp_path, planning_liveness=True)
    assert _inactive(caplog) == [
        "planning liveness requested but inactive (aut_mode=llm-primary, llm_worker=no, env_opt_out=no)"
    ]


def test_liveness_with_the_env_opt_out_logs_inactive(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("MAXIM_SIM_PLANNING_LIVENESS", "0")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _run(monkeypatch, tmp_path, planning_liveness=True, llm_worker=SimpleNamespace())
    assert _inactive(caplog) == [
        "planning liveness requested but inactive (aut_mode=llm-primary, llm_worker=yes, env_opt_out=yes)"
    ]


def test_active_liveness_logs_nothing(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv("MAXIM_SIM_PLANNING_LIVENESS", raising=False)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _run(monkeypatch, tmp_path, planning_liveness=True, llm_worker=SimpleNamespace())
    assert _inactive(caplog) == []


def test_liveness_not_requested_logs_nothing(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _run(monkeypatch, tmp_path, planning_liveness=False, aut_mode="substrate-primary")
    assert _inactive(caplog) == []


# ── the order of the side effects ────────────────────────────────────────────


def test_the_setup_side_effects_run_in_order(monkeypatch, tmp_path):
    """Persist, pool, controller, overrides, DN start, bio handles, agent id, encoder, cue, session start,
    liveness log -- today's order, which the extraction must keep."""
    from maxim.decisions.nac import NAc
    from maxim.similarity.ec import EntorhinalCortex

    events: list[str] = []
    hub = _Hub(events, nac=NAc(), ec=EntorhinalCortex(), cue=object())
    _spy_substrate(monkeypatch)
    _run(
        monkeypatch,
        tmp_path,
        events=events,
        memory_hub=hub,
        default_network=_DN(events),
        planning_liveness=True,
        aut_mode="substrate-primary",
    )
    end = events.index("liveness_inactive_log") + 1
    assert events[:end] == [
        "context_pool",
        "ctrl",
        "overrides",
        "dn.start",
        "hub.nac",
        "hub.hippocampus",
        "hub.agent_id",
        "hub.ec",
        "hub.atl",
        "hub.situation_cue",
        "hub.on_session_start(persisted=True)",
        "liveness_inactive_log",
    ]


def test_an_override_config_error_stops_the_run_before_the_bio_session(monkeypatch, tmp_path):
    class _ConfigBroken(Exception):
        pass

    def _broken() -> Any:
        raise _ConfigBroken("bad llm.max_response_tokens")

    events: list[str] = []
    hub = _Hub(events)
    with pytest.raises(_ConfigBroken):
        _run(monkeypatch, tmp_path, events=events, overrides=_broken, memory_hub=hub, default_network=_DN(events))
    assert len(_state_files(tmp_path)) == 1  # persisted before the overrides resolve
    assert events == ["context_pool", "ctrl"]  # no DN start, no bio handles, no session start
