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
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.policy_helpers import TEST_POLICY

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

    def on_session_start(self, *, owner: Any, live: bool = False, deadline_s: float | None = None) -> Any:
        from maxim.utils.paths import data_home

        state_files = list(data_home().glob("runtime_state/*/state_*.json"))
        self._events.append(f"hub.on_session_start(persisted={bool(state_files)})")
        if self._start_fails:
            raise RuntimeError("hub start failed")
        self.ticket = SimpleNamespace(owner=owner, live=live, generation=1, start_stats={})
        return self.ticket

    def on_session_end(self, ticket: Any, *, deadline: Any = None, consolidation_bound: Any = None) -> dict[str, Any]:
        assert ticket is self.ticket, "ended a session this fake never opened"  # #1438 PR 0: every fake checks
        self.ended.append("full")
        return {}

    def on_session_end_lightweight(self, ticket: Any, *, deadline: Any = None) -> dict[str, Any]:
        assert ticket is self.ticket, "ended a session this fake never opened"
        self.ended.append("lightweight")
        return {}

    def release_session_claim(self, ticket: Any) -> None:
        return None

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
    """Run the real loop in ``tmp_path``, with the data home (where the state file lands, #1264) at
    ``tmp_path / "home"``; return what setup built."""
    from maxim.agents.context_pool import ContextPool
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime import loop_setup
    from maxim.runtime.bootstrap import build_decision_engine, build_executor, build_memory
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.state import RuntimeState
    from maxim.tools.registry import ToolRegistry

    from maxim.utils.paths import _reset_caches

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MAXIM_DATA_HOME", str(_data_home(tmp_path)))
    _reset_caches()
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

    executor = kwargs.pop("executor", None) or build_executor(ToolRegistry(), pain_bus=None, policy=TEST_POLICY)
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


def _data_home(tmp_path: Path) -> Path:
    return tmp_path / "home"


def _state_files(tmp_path: Path) -> list[Path]:
    return sorted(_data_home(tmp_path).glob("runtime_state/*/state_*.json"))


# ── the simulation adapter ───────────────────────────────────────────────────


@pytest.mark.parametrize("adapter", [SimpleNamespace(is_sim_mode=True), SimpleNamespace()], ids=["sim", "no-flag"])
def test_a_sim_mode_sim_adapter_is_refused_before_any_side_effect(monkeypatch, tmp_path, adapter):
    events: list[str] = []
    hub = _Hub(events)
    with pytest.raises(ValueError, match=r"sim_adapter= must be a non-sim adapter"):
        _run(monkeypatch, tmp_path, events=events, sim_adapter=adapter, memory_hub=hub)
    assert _state_files(tmp_path) == []  # refused before the first persist
    # #1438 PR 0: the session is claimed BEFORE setup (bio_session), so the refusal ends it on the abnormal path;
    # still no context pool, controller or DN.
    assert events == ["hub.on_session_start(persisted=False)"]
    assert hub.ended == ["lightweight"]


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
        response_policy=None,  # a test sim with no response policy answers no prompt (#1185)
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
    rel = path.relative_to(_data_home(tmp_path))
    assert rel.parent == Path("runtime_state/Probe_Agent")
    assert re.fullmatch(r"state_r-42_[0-9a-f]{12}\.json", rel.name), rel.name
    payload = json.loads(path.read_text())
    assert payload["run_id"] == "r-42" and payload["agent_name"] == "Probe_Agent"
    assert run.ctrl.run_id == "r-42"
    assert run.ctrl.state_path == str(path)  # the controller holds the run's claimed path, not a second one


def test_a_missing_run_id_is_a_timestamp(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, run_id=None)
    [path] = _state_files(tmp_path)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{6}", run.ctrl.run_id)  # run ids are unchanged (#1264)
    assert re.fullmatch(rf"state_{run.ctrl.run_id}_[0-9a-f]{{12}}\.json", path.name)
    assert json.loads(path.read_text())["run_id"] == run.ctrl.run_id


# ── #1264: one state file per loop run, under the data home ──────────────────


def test_two_loops_with_one_run_id_keep_two_state_files_under_the_data_home_1264(monkeypatch, tmp_path):
    """RED on the pre-fix loop: the state file was ``data/agents/<name>/runtime/state_<run_id>.json`` under the
    CWD, so two loops of one agent class started in the same second (one default run id) wrote ONE file, the
    later overwriting the earlier (the AUT and narrator of a sim). Each run now keeps its own file in the data
    home, and nothing lands under the working directory."""
    first = _run(monkeypatch, tmp_path, agent=_Agent("MaximAgent"), run_id="2026-11-16_120000")
    second = _run(monkeypatch, tmp_path, agent=_Agent("MaximAgent"), run_id="2026-11-16_120000")

    assert not (tmp_path / "data").exists(), "the state file must not be CWD-relative (D15)"
    files = _state_files(tmp_path)
    assert len(files) == 2, files  # neither run's state was overwritten
    assert {str(p) for p in files} == {first.ctrl.state_path, second.ctrl.state_path}
    for path in files:
        payload = json.loads(path.read_text())
        assert payload["run_id"] == "2026-11-16_120000" and payload["agent_name"] == "MaximAgent"
        assert payload["_format_version"] == "1.0"


def test_a_state_path_two_runs_still_share_fails_loudly_1264(monkeypatch, tmp_path):
    """If two runs still resolve to one file (here: the unique token forced equal), the second run fails at its
    claim with a typed ``StatePathCollision`` before it writes, and the first run's state is left intact."""
    from maxim.runtime import loop_state
    from maxim.runtime.loop_state import StatePathCollision

    monkeypatch.setattr(loop_state, "_state_token", lambda: "0" * 12)
    _run(monkeypatch, tmp_path, agent=_Agent("MaximAgent"), run_id="r-1264")
    [path] = _state_files(tmp_path)
    before = path.read_bytes()
    with pytest.raises(StatePathCollision, match="already belongs to another loop run"):
        _run(monkeypatch, tmp_path, agent=_Agent("MaximAgent"), run_id="r-1264")
    assert _state_files(tmp_path) == [path]
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("run_id", "segment"),
    [
        ("../escape", "___escape"),
        ("a/b", "a_b"),
        ("a\\b", "a_b"),
        ("demo 1", "demo_1"),
        ("2026-11-16T12:00:00", "2026-11-16T12_00_00"),
        (".hidden", "_hidden"),
        ("..", "__"),
        ("a..b", "a__b"),
    ],
)
def test_a_run_id_is_sanitised_in_the_filename_only_1264(monkeypatch, tmp_path, run_id, segment):
    """``../``, a separator, a space or a ``:`` in the run id never puts the state file outside its runtime directory:
    the FILENAME segment maps them to ``_`` (as ``safe_agent_name`` does), and the raw run id is the one in the meta."""
    run = _run(monkeypatch, tmp_path, agent=_Agent("MaximAgent"), run_id=run_id)
    [path] = _state_files(tmp_path)
    assert path.parent == _data_home(tmp_path) / "runtime_state" / "MaximAgent"
    assert re.fullmatch(rf"state_{re.escape(segment)}_[0-9a-f]{{12}}\.json", path.name), path.name
    payload = json.loads(path.read_text())
    assert payload["run_id"] == run_id  # the raw id, unchanged
    assert run.ctrl.run_id == run_id
    assert [p for p in tmp_path.rglob("state_*") if p != path] == []


@pytest.mark.parametrize(
    "run_id", ["2026-11-16_120000", "20261116_120000", "sim_turn_007", "talk_3f2a9c", "r-42", "a.b", "0123abcdef"]
)
def test_the_run_ids_maxim_produces_name_a_state_file_1264(monkeypatch, tmp_path, run_id):
    from maxim.runtime.loop_state import loop_state_path
    from maxim.utils.paths import _reset_caches

    monkeypatch.setenv("MAXIM_DATA_HOME", str(_data_home(tmp_path)))
    _reset_caches()
    path = Path(loop_state_path("MaximAgent", run_id))
    assert path.parent == _data_home(tmp_path) / "runtime_state" / "MaximAgent"
    assert path.name.startswith(f"state_{run_id}_")


def test_state_home_puts_the_state_file_in_the_given_root_only_1264(monkeypatch, tmp_path):
    """``run_agentic_loop(state_home=...)`` (a sim's sandbox, a harness's per-agent home): the state file goes under
    that data-home-shaped root and nowhere else; the data home is untouched, and nothing outlives the call."""
    from maxim.runtime.loop_state import loop_state_path

    home = tmp_path / "aut_home"
    run = _run(monkeypatch, tmp_path, agent=_Agent("MaximAgent"), run_id="r-1264", state_home=home)
    [path] = sorted(home.glob("runtime_state/*/state_*.json"))
    assert run.ctrl.state_path == str(path)
    assert json.loads(path.read_text())["run_id"] == "r-1264"
    assert _state_files(tmp_path) == []  # the data home holds none
    assert Path(loop_state_path("MaximAgent", "after")).is_relative_to(_data_home(tmp_path))  # no root: data home


def _state_home_args(rel: str) -> list[str | None]:
    """Per ``run_agentic_loop(...)`` call in ``src/maxim/<rel>``, in source order: its ``state_home=`` argument as
    source text, or None when it passes none (the data home)."""
    import ast

    src = (Path(__file__).resolve().parents[2] / "src" / "maxim" / rel).read_text(encoding="utf-8")
    found: list[tuple[int, str | None]] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "run_agentic_loop":
                kw = {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
                found.append((node.lineno, kw.get("state_home")))
    return [arg for _, arg in sorted(found)]


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        # sandboxed callers: the state file goes with the sandbox (owner decision 2026-11-16)
        ("cli.py", ["getattr(args, '_sim_tmpdir', None)"]),
        ("simulation/orchestrator.py", ["sim_tmpdir", "sim_tmpdir"]),  # the AUT, then the narrator
        ("simulation/tools.py", ["self._sim_tmpdir"]),  # the sub-AUT: its parent sim's sandbox
        ("simulation/interactive.py", ["run_tmpdir"]),
        # the data home (minecraft_harness passes it through _loop_kwargs, pinned in test_minecraft_harness)
        ("console/handle.py", [None]),
        # home_dir, else the data home (_api_home): D15's home_dir half, owner decision 2026-11-18
        ("api.py", ["effective_home"]),
        ("embodied_runtime/agentic_runtime.py", [None]),
    ],
)
def test_each_loop_caller_passes_the_state_home_it_owns_1264(rel, expected):
    assert _state_home_args(rel) == expected


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


def test_a_sim_run_starts_the_default_network_and_stops_it(monkeypatch, tmp_path):
    """#1263/#1281: the loop that starts the DN stops it in a sim too (it used to skip the stop there, which left
    the sim AUT's DN ticking after the run)."""
    events: list[str] = []
    dn = _DN(events)
    run = _run(
        monkeypatch, tmp_path, events=events, default_network=dn, percept_source=SimpleNamespace(), response_policy=None
    )
    assert events.count("dn.start") == 1 and run.ctrl.dn_enabled is True
    assert dn.stops == 1


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
    executor = build_executor(registry, pain_bus=None, policy=TEST_POLICY)
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


def test_a_hub_that_fails_to_start_aborts_the_loop_and_is_not_ended(monkeypatch, tmp_path):
    """#1438 PR 0: only "no memory hub configured" stays a soft skip; a hub whose session cannot be claimed
    aborts the loop (it used to run on silently, without memory)."""
    hub = _Hub([], start_fails=True)
    with pytest.raises(RuntimeError, match="hub start failed"):
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
        "hub.on_session_start(persisted=False)",  # #1438 PR 0: claimed BEFORE setup starts anything
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
    # #1438 PR 0: the session was claimed first, so the error ends it on the abnormal path; no DN start, no bio handles
    assert events == ["hub.on_session_start(persisted=False)", "context_pool", "ctrl"]
    assert hub.ended == ["lightweight"]


# ── #1264: only a collision aborts; any other claim failure is reported and the run goes on ──


def test_a_claim_that_fails_with_another_oserror_is_reported_and_the_run_goes_on_1264(monkeypatch, tmp_path, caplog):
    """A data home the loop cannot write (here: ``runtime_state`` is a FILE, so its directory cannot be made) is NOT
    a collision: the claim is reported in the Stage-1 form and the run starts, as before #1264; the persists that
    follow fail and are reported too."""
    home = _data_home(tmp_path)
    home.mkdir(parents=True)
    (home / "runtime_state").write_text("not a directory")
    with caplog.at_level(logging.WARNING):
        run = _run(monkeypatch, tmp_path, agent=_Agent("MaximAgent"), run_id="r-1264")
    assert run.ctrl is not None and run.ctrl.run_id == "r-1264"  # the run started
    assert "loop_state.py:claim_state_path" in caplog.text
    assert (home / "runtime_state").read_text() == "not a directory"


def test_an_empty_placeholder_is_removed_when_the_first_persist_fails_1264(monkeypatch, tmp_path, caplog):
    from maxim.runtime import loop_state
    from maxim.utils.paths import _reset_caches

    monkeypatch.setenv("MAXIM_DATA_HOME", str(_data_home(tmp_path)))
    _reset_caches()

    class _Unsavable:
        def save_json(self, path: str, *, meta: Any = None) -> None:
            raise OSError("disk full")

    with caplog.at_level(logging.WARNING):
        run_id, path = loop_state.start_run_state(_Unsavable(), "MaximAgent", "r-1264")
    assert run_id == "r-1264"
    assert not Path(path).exists()  # no empty, unversioned state file left behind
    assert "removed the empty state-file placeholder" in caplog.text


def test_state_path_collision_is_a_file_exists_error_naming_the_path_1264(monkeypatch, tmp_path):
    import errno

    from maxim.runtime import loop_state
    from maxim.utils.paths import _reset_caches

    monkeypatch.setenv("MAXIM_DATA_HOME", str(_data_home(tmp_path)))
    _reset_caches()
    monkeypatch.setattr(loop_state, "_state_token", lambda: "0" * 12)
    first = loop_state.claim_state_path("MaximAgent", "r")
    with pytest.raises(loop_state.StatePathCollision) as info:
        loop_state.claim_state_path("MaximAgent", "r")
    assert isinstance(info.value, FileExistsError)
    assert info.value.errno == errno.EEXIST and info.value.filename == first


def test_a_controller_built_on_its_own_has_no_state_path_1264():
    from maxim.agents.autonomy import AutonomyController
    from maxim.runtime.loop_controller import LoopController

    ctrl = LoopController(None, None, None, None, None, None, autonomy_controller=AutonomyController())
    assert ctrl.state_path is None
