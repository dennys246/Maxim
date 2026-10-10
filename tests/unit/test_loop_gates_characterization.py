"""Characterization of ``run_agentic_loop``'s PRE-TICK GATE, §0-§0.6 (1.3.2 decomposition, slice 2).

Pins, through the PUBLIC entry ``agent_loop.run_agentic_loop``, what every pass of the loop does before
it perceives: the §0 stop checks (``stop_event``, a ``"shutdown"`` mode, the Default Network's mode with
the operator's grant winning, the pause sleep), §0.45 the live tick (body drift + experience clock) and
the display auto-revert, §0.5 the percept-source exhaustion check, and §0.6 the idle gate: which wake
source runs a full pass, which pass sleeps ``idle_sleep_s`` and continues, and the D13 planning-liveness
backstop that requeues a planning turn which ended with nothing executable and, once its budget is
spent, breaks through normal teardown and raises ``PlanningLivenessExhausted``. Written BEFORE slice 2
moved the block into ``runtime/loop_gates.py`` and kept green unchanged by that move
(``docs/plans/roadmap_1_3_x.md`` §"The decomposition", coverage-first rule). The idle gate is an Exp 60
trigger path (the substrate wake source).

Observation is location-independent, so the pins hold wherever the gate's code lives: a pass is seen
through the loop's own ``loop_iteration`` event (the abstraction buffer, patched at its module), a FULL
pass through the adapter's ``next_observation`` (§1, the first thing after the gate), sleeps through the
global ``time.sleep``, and the rest through class-level spies (``LoopController``,
``ExperienceClockDriver``), the display module's ``maybe_auto_revert_display`` and fakes passed as
arguments. Nothing patches ``agent_loop`` or the gate's module.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from maxim.runtime.sim_adapter import NullSimulationAdapter

LOGGER = "maxim.runtime.agent_loop"
IDLE = 0.0123  # a distinctive idle_sleep_s: the §0 pause and §0.6 idle sleeps are exactly this
# A pending-work marker. Every arm that sets one clears it at §1 (``next_observation``), so it only has to
# wake the gate; nothing after the gate ever acts on it.
_MARK = SimpleNamespace(action=None, error=None, tool="t", followup_type="x", mode="m", result="", original_query="")


# ── harness ──────────────────────────────────────────────────────────────────


class _Agent:
    name = "gate-probe"

    def propose_intent(self, state: Any, memory: Any) -> None:
        return None


class _Adapter(NullSimulationAdapter):
    """A non-sim adapter (``sim_adapter=``) that records the full pass and the exhaustion check."""

    def __init__(self, ev: list[tuple], exhaust_at: int | None = None) -> None:
        super().__init__()
        self.ev = ev
        self.exhaust_at = exhaust_at
        self.on_observe: Callable[[], None] | None = None

    def next_observation(self, environment: Any, default_network: Any | None = None) -> dict:
        self.ev.append(("run",))
        if self.on_observe is not None:
            self.on_observe()
        return super().next_observation(environment, default_network)

    def check_exhaustion(self, pending_proposal: Any | None) -> bool:
        self.ev.append(("exhaustion?", pending_proposal))
        return self.exhaust_at is not None and _step_of(self.ev) >= self.exhaust_at


class _Body:
    """``executor.embodiment``: the llm-primary drift tick calls ``evaluate_failures``."""

    def __init__(self, ev: list[tuple], raises: bool = False) -> None:
        self.ev, self.raises = ev, raises

    def evaluate_failures(self) -> None:
        self.ev.append(("drift",))
        if self.raises:
            raise RuntimeError("body tick failed")


class _Worker:
    """The ``llm_worker`` surface the loop and the D13 handlers read. ``states[k]`` is
    ``latest_attempt_state()`` during pass k (the last entry repeats; an Exception instance is raised).
    Keyed by PASS, not by call: a full pass reads it again after the gate (the #1048 submit hold)."""

    def __init__(self, ev: list[tuple], states: list[Any], requeue_ok: bool = True) -> None:
        self.ev, self.states, self.requeue_ok = ev, list(states), requeue_ok
        self.reads = 0

    def latest_attempt_state(self) -> Any:
        self.reads += 1
        self.ev.append(("state?",))
        s = self.states[min(_step_of(self.ev), len(self.states) - 1)]
        if isinstance(s, Exception):
            raise s
        return s

    def get_latest_proposal(self) -> None:
        return None

    def submit_context(self, *_a: Any, **_k: Any) -> bool:
        self.ev.append(("submit",))
        return True

    def requeue_request(self, request: Any, **kw: Any) -> bool:
        self.ev.append(("requeue_request", kw))
        return self.requeue_ok

    def requeue_last_request(self, **kw: Any) -> bool:
        self.ev.append(("requeue_last", kw))
        return self.requeue_ok


def _step_of(ev: list[tuple]) -> int:
    return max((e[1] for e in ev if e[0] == "iter"), default=-1)


class _Run(SimpleNamespace):
    ev: list[tuple]
    ctrl: Any
    state: Any
    error: BaseException | None


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    steps: int = 4,
    hooks: dict[int, Callable[[Any], None]] | None = None,
    adapter: Any = None,
    ev: list[tuple] | None = None,
    mode: str | None = "active",
    submit_interval: float | None = 1e12,
    body: Any = None,
    executor: Any = None,
    stop_event: Any = None,
    **kwargs: Any,
) -> _Run:
    """Run the real loop for ``steps`` passes; ``hooks[k](run)`` runs at pass k's ``loop_iteration`` event
    (before §0). ``submit_interval`` replaces the controller's LLM submit cadence before the loop reads it
    (``1e12`` keeps a worker-bearing run from submitting on its own; None keeps the default)."""
    import maxim.simulation.sim_logger as sim_logger
    import maxim.utils.structured_logging as sl
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime.bootstrap import build_decision_engine, build_executor, build_memory
    from maxim.runtime.experience_time import ExperienceClockDriver
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.state import RuntimeState
    from maxim.tools.registry import ToolRegistry

    monkeypatch.chdir(tmp_path)
    ev = [] if ev is None else ev
    hooks = hooks or {}
    run = _Run(ev=ev, ctrl=None, state=None, error=None)

    real_init = LoopController.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        if submit_interval is not None:
            self.llm_submit_interval = submit_interval
        run.ctrl = self

    monkeypatch.setattr(LoopController, "__init__", _init)
    monkeypatch.setattr(LoopController, "configure_dn_for_mode", lambda self, m: ev.append(("dn", m)))
    real_live = ExperienceClockDriver.on_live_pass

    def _live(self: Any) -> int:
        ev.append(("xclock",))
        return real_live(self)

    monkeypatch.setattr(ExperienceClockDriver, "on_live_pass", _live)
    monkeypatch.setattr(sim_logger, "maybe_auto_revert_display", lambda: ev.append(("display",)) or False)

    real_buffer = sl.get_abstraction_buffer()

    class _Buffer:
        def append(self, record: Any) -> None:
            if record.source == "agent_loop" and record.event == "loop_iteration":
                step = record.data["step"]
                ev.append(("iter", step))
                if step in hooks:
                    hooks[step](run)
            elif record.source == "agent_loop" and record.event in ("shutdown", "planning_liveness_exhausted"):
                ev.append((record.event, dict(record.data)))
            real_buffer.append(record)

        def __getattr__(self, name: str) -> Any:
            return getattr(real_buffer, name)

    monkeypatch.setattr(sl, "get_abstraction_buffer", lambda: _Buffer())

    loop_thread = threading.get_ident()
    real_sleep = time.sleep

    def _sleep(dt: float) -> None:
        if threading.get_ident() == loop_thread and dt == IDLE:
            ev.append(("sleep",))
            return
        real_sleep(min(dt, 0.001))

    monkeypatch.setattr(time, "sleep", _sleep)

    if executor is None:
        executor = build_executor(ToolRegistry(), pain_bus=None, permissions=None)
    if body is not None:
        executor.embodiment = body
    state = RuntimeState()
    if mode is not None:  # None: the state carries no mode at all (the CLI loop, ``maxim.run()``)
        state.data["mode"] = mode
    run.state = state
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    if adapter is None and "percept_source" not in kwargs:
        adapter = _Adapter(ev)
    kwargs.setdefault("target_hz", 1e6)
    try:
        AL.run_agentic_loop(
            _Agent(),
            FileSystemEnv(str(workspace)),
            state,
            build_memory(),
            build_decision_engine(),
            executor,
            sim_adapter=adapter,
            stop_event=threading.Event() if stop_event is None else stop_event,
            max_steps=steps,
            idle_sleep_s=IDLE,
            **kwargs,
        )
    except Exception as exc:  # the PlanningLivenessExhausted arms inspect it
        run.error = exc
    return run


def _passes(ev: list[tuple]) -> dict[int, list[tuple]]:
    """Events grouped by the pass they happened in (the ``iter`` that preceded them)."""
    out: dict[int, list[tuple]] = {}
    step = -1
    for e in ev:
        if e[0] == "iter":
            step = e[1]
            out[step] = []
        else:
            out.setdefault(step, []).append(e)
    return out


def _kinds(events: list[tuple]) -> list[str]:
    return [e[0] for e in events]


def _ran(ev: list[tuple]) -> list[int]:
    """The passes that ran a FULL tick (reached §1's ``next_observation``)."""
    return [s for s, es in _passes(ev).items() if ("run",) in es]


def _slept(ev: list[tuple]) -> list[int]:
    return [s for s, es in _passes(ev).items() if ("sleep",) in es]


# ── an ordinary pass: the order of §0-§0.6 ───────────────────────────────────


def test_an_idle_pass_and_a_full_pass_in_order(monkeypatch, tmp_path):
    ev: list[tuple] = []
    run = _run(monkeypatch, tmp_path, steps=3, ev=ev, body=_Body(ev))
    passes = _passes(run.ev)
    # Pass 0 (the first step) runs a full tick; the order up to §1 is: DN mode, live tick (body drift then
    # the experience clock), display revert, exhaustion check, then perception.
    assert _kinds(passes[0])[:6] == ["dn", "drift", "xclock", "display", "exhaustion?", "run"]
    assert passes[0][0] == ("dn", "active")
    assert passes[0][4] == ("exhaustion?", None)  # check_exhaustion(ctrl.pending_proposal)
    # Later passes have nothing to react to: the same pre-tick sequence, then one idle sleep and continue.
    for step in (1, 2):
        assert _kinds(passes[step]) == ["dn", "drift", "xclock", "display", "exhaustion?", "sleep"]
    assert _ran(run.ev) == [0] and _slept(run.ev) == [1, 2]


def test_the_live_tick_advances_the_experience_clock_once_per_live_pass(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, steps=5)
    assert [s for s, es in _passes(run.ev).items() if es.count(("xclock",)) == 1] == [0, 1, 2, 3, 4]


def test_the_body_drifts_only_on_llm_primary(monkeypatch, tmp_path):
    ev: list[tuple] = []
    run = _run(monkeypatch, tmp_path, steps=2, ev=ev, body=_Body(ev), aut_mode="substrate-primary")
    assert ("drift",) not in run.ev  # substrate-primary ticks its own body
    assert sum(1 for e in run.ev if e == ("xclock",)) == 2  # the clock still advances


def test_a_failing_body_tick_does_not_stop_the_pass(monkeypatch, tmp_path):
    ev: list[tuple] = []
    run = _run(monkeypatch, tmp_path, steps=2, ev=ev, body=_Body(ev, raises=True))
    assert _kinds(_passes(run.ev)[1]) == ["dn", "drift", "xclock", "display", "exhaustion?", "sleep"]


def test_a_failing_display_revert_does_not_stop_the_pass(monkeypatch, tmp_path):
    import maxim.simulation.sim_logger as sim_logger

    ev: list[tuple] = []

    def _boom() -> bool:
        ev.append(("display",))
        raise RuntimeError("revert failed")

    run = _run(
        monkeypatch,
        tmp_path,
        steps=2,
        ev=ev,
        hooks={0: lambda r: monkeypatch.setattr(sim_logger, "maybe_auto_revert_display", _boom)},
    )
    assert _kinds(_passes(run.ev)[1]) == ["dn", "xclock", "display", "exhaustion?", "sleep"]


# ── §0 stop checks ───────────────────────────────────────────────────────────


def test_stop_event_breaks_before_anything_else(monkeypatch, tmp_path):
    stop = threading.Event()
    run = _run(monkeypatch, tmp_path, steps=6, stop_event=stop, hooks={2: lambda r: stop.set()})
    passes = _passes(run.ev)
    assert max(passes) == 2  # no pass after the break
    assert passes[2] == [("shutdown", {"reason": "stop_event"})]  # nothing of §0-§1 ran in that pass


@pytest.mark.parametrize("error", [AttributeError, RuntimeError])
def test_a_stop_event_that_raises_is_ignored(monkeypatch, tmp_path, error):
    class _Raising:
        def is_set(self) -> bool:
            raise error("broken event")

    run = _run(monkeypatch, tmp_path, steps=3, stop_event=_Raising())
    assert sorted(_passes(run.ev)) == [0, 1, 2]
    assert not any(e[0] == "shutdown" for e in run.ev)


def test_a_stop_event_without_is_set_is_ignored(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, steps=3, stop_event=SimpleNamespace())
    assert sorted(_passes(run.ev)) == [0, 1, 2]


def test_shutdown_mode_breaks_before_the_default_network_is_configured(monkeypatch, tmp_path):
    def _shutdown(r: Any) -> None:
        r.state.data["mode"] = "shutdown"

    run = _run(monkeypatch, tmp_path, steps=6, hooks={3: _shutdown})
    passes = _passes(run.ev)
    assert max(passes) == 3
    assert passes[3] == [("shutdown", {"reason": "shutdown_mode"})]


def test_shutdown_mode_from_the_start_runs_no_pass(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, steps=3, mode="shutdown")
    assert _passes(run.ev) == {0: [("shutdown", {"reason": "shutdown_mode"})]}


def test_the_default_network_follows_the_mode_each_pass(monkeypatch, tmp_path):
    def _observe(r: Any) -> None:
        r.state.data["mode"] = "observe"

    run = _run(monkeypatch, tmp_path, steps=3, hooks={1: _observe})
    assert [e for e in run.ev if e[0] == "dn"] == [("dn", "active"), ("dn", "observe"), ("dn", "observe")]


def test_the_operators_grant_wins_for_the_default_network(monkeypatch, tmp_path):
    from maxim.runtime.bootstrap import build_executor
    from maxim.tools.registry import ToolRegistry

    executor = build_executor(ToolRegistry(), pain_bus=None, permissions=None)
    executor.set_operational_override("observe")
    run = _run(monkeypatch, tmp_path, steps=2, executor=executor)
    assert [e for e in run.ev if e[0] == "dn"] == [("dn", "observe"), ("dn", "observe")]


def test_an_empty_mode_configures_the_default_network_as_observe(monkeypatch, tmp_path):
    """#963 (owner decision Q4) changed this pin: an empty run mode left the Default Network unconfigured; the one
    operational-mode reader gives the one default, ``observe``."""
    run = _run(monkeypatch, tmp_path, steps=2, mode="")
    assert [e for e in run.ev if e[0] == "dn"] == [("dn", "observe"), ("dn", "observe")]
    assert _ran(run.ev) == [0]  # the rest of the pass is unchanged


# ── the Default Network's mode, by run mode and grant (#963 characterization) ─────────────────────────


def _granted(grant: str | None) -> Any:
    from maxim.runtime.bootstrap import build_executor
    from maxim.tools.registry import ToolRegistry

    executor = build_executor(ToolRegistry(), pain_bus=None, permissions=None)
    if grant is not None:
        executor.set_operational_override(grant)
    return executor


@pytest.mark.parametrize(
    ("mode", "grant", "dn"),
    [
        ("live", None, "live"),
        ("observe", None, "observe"),
        ("active", None, "active"),
        ("live", "passive", "passive"),  # the grant wins (#829)
        ("", "passive", "passive"),
    ],
)
def test_the_default_networks_mode_by_run_mode_and_grant(monkeypatch, tmp_path, mode, grant, dn):
    run = _run(monkeypatch, tmp_path, steps=2, mode=mode, executor=_granted(grant))
    assert [e for e in run.ev if e[0] == "dn"] == [("dn", dn), ("dn", dn)]


def test_a_state_with_no_mode_configures_the_default_network_as_observe(monkeypatch, tmp_path):
    """The CLI loop and ``maxim.run()`` seed no mode. #963 (owner decision Q4) changed this pin: the Default Network
    was left unconfigured; it is now ``observe``, the one default."""
    run = _run(monkeypatch, tmp_path, steps=2, mode=None)
    assert [e for e in run.ev if e[0] == "dn"] == [("dn", "observe"), ("dn", "observe")]


def test_the_default_network_follows_a_grant_set_between_ticks(monkeypatch, tmp_path):
    """Read every pass, never cached: a grant set between pass 0 and pass 1 reaches the Default Network at pass 1
    (owner decision Q6, #963). Green before #963 too; it guards the accessor against caching."""
    executor = _granted(None)
    run = _run(
        monkeypatch,
        tmp_path,
        steps=3,
        mode="live",
        executor=executor,
        hooks={1: lambda r: executor.set_operational_override("passive")},
    )
    assert [e for e in run.ev if e[0] == "dn"] == [("dn", "live"), ("dn", "passive"), ("dn", "passive")]


def test_a_paused_controller_sleeps_and_continues_before_the_live_tick(monkeypatch, tmp_path):
    from maxim.agents.autonomy import AutonomyController

    controller = AutonomyController()
    hooks = {
        1: lambda r: controller.emergency_halt("probe"),
        3: lambda r: controller.resume(),
    }
    run = _run(monkeypatch, tmp_path, steps=5, autonomy_controller=controller, hooks=hooks)
    passes = _passes(run.ev)
    for step in (1, 2):  # paused: the DN mode is still configured, then one sleep and nothing else
        assert passes[step] == [("dn", "active"), ("sleep",)]
    assert _kinds(passes[3]) == ["dn", "xclock", "display", "exhaustion?", "sleep"]  # resumed, idle


# ── §0.5 exhaustion ──────────────────────────────────────────────────────────


def test_exhaustion_breaks_the_loop_after_the_live_tick(monkeypatch, tmp_path):
    ev: list[tuple] = []
    run = _run(monkeypatch, tmp_path, steps=6, ev=ev, adapter=_Adapter(ev, exhaust_at=2))
    passes = _passes(run.ev)
    assert max(passes) == 2
    assert _kinds(passes[2]) == ["dn", "xclock", "display", "exhaustion?"]  # broke: no sleep, no run


def test_exhaustion_is_asked_with_the_pending_proposal(monkeypatch, tmp_path):
    ev: list[tuple] = []
    adapter = _Adapter(ev)
    holder: list[Any] = []

    def _hook(r: Any) -> None:
        holder.append(r)
        r.ctrl.pending_proposal = _MARK

    adapter.on_observe = lambda: holder and setattr(holder[0].ctrl, "pending_proposal", None)
    run = _run(monkeypatch, tmp_path, steps=3, ev=ev, adapter=adapter, hooks={1: _hook})
    passes = _passes(run.ev)
    assert ("exhaustion?", _MARK) in passes[1]
    assert ("exhaustion?", None) in passes[2]


# ── §0.6 the idle gate: every wake source ────────────────────────────────────


def _set_data(key: str) -> Callable[[Any], None]:
    def hook(r: Any) -> None:
        r.state.data[key] = "hello"

    return hook


def _clear_data(state: Any, key: str) -> None:
    state.data.pop(key, None)


def _ctrl_set(attr: str, value: Any) -> Callable[[Any], None]:
    def hook(r: Any) -> None:
        setattr(r.ctrl, attr, value)

    return hook


@pytest.mark.parametrize(
    ("name", "wake", "clear"),
    [
        ("pending_cli_input", _set_data("pending_cli_input"), lambda r: _clear_data(r.state, "pending_cli_input")),
        (
            "pending_voice_input",
            _set_data("pending_voice_input"),
            lambda r: _clear_data(r.state, "pending_voice_input"),
        ),
        ("pending_proposal", _ctrl_set("pending_proposal", _MARK), lambda r: setattr(r.ctrl, "pending_proposal", None)),
        (
            "pending_action_followup",
            _ctrl_set("pending_action_followup", _MARK),
            lambda r: setattr(r.ctrl, "pending_action_followup", None),
        ),
        (
            "pending_plan_proposal",
            _ctrl_set("pending_plan_proposal", _MARK),
            lambda r: setattr(r.ctrl, "pending_plan_proposal", None),
        ),
        (
            "pending_next_actions",
            lambda r: r.ctrl.pending_next_actions.append({"tool_name": "noop"}),
            lambda r: r.ctrl.pending_next_actions.clear(),
        ),
        ("carried_percept", "carry", None),  # the adapter's mailbox; next_observation consumes it
    ],
)
def test_each_wake_source_runs_exactly_the_pass_it_is_present_in(monkeypatch, tmp_path, name, wake, clear):
    ev: list[tuple] = []
    adapter = _Adapter(ev)
    holder: list[Any] = []

    def hook(r: Any) -> None:
        holder.append(r)
        if wake == "carry":
            adapter.carry_percept(SimpleNamespace(modality=None, sensory=None))
        else:
            wake(r)

    if clear is not None:
        adapter.on_observe = lambda: holder and clear(holder[0])  # cleared at §1, before anything acts on it
    run = _run(monkeypatch, tmp_path, steps=5, ev=ev, adapter=adapter, hooks={2: hook})
    assert _ran(run.ev) == [0, 2], name
    assert _slept(run.ev) == [1, 3, 4], name


def test_no_wake_source_idles_every_pass_after_the_first(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, steps=5)
    assert _ran(run.ev) == [0] and _slept(run.ev) == [1, 2, 3, 4]


class _Percepts:
    """A percept source for a SIM run: no percepts, never exhausted, ``has_pending`` scripted per pass."""

    def __init__(self, ev: list[tuple], pending_at: set[int]) -> None:
        self.ev, self.pending_at = ev, pending_at

    def has_pending(self) -> bool:
        return _step_of(self.ev) in self.pending_at

    def next_percept(self) -> None:
        return None

    def is_exhausted(self) -> bool:
        return False


class _PerceptsNoPending:
    """A source without ``has_pending``: the gate's default counts it as always pending."""

    def next_percept(self) -> None:
        return None

    def is_exhausted(self) -> bool:
        return False


def _spy_sim_adapter(monkeypatch: pytest.MonkeyPatch, ev: list[tuple]) -> None:
    from maxim.runtime.sim_adapter import SimulationAdapter

    real = SimulationAdapter.next_observation

    def _next(self: Any, environment: Any, default_network: Any = None) -> dict:
        ev.append(("run",))
        return real(self, environment, default_network)

    monkeypatch.setattr(SimulationAdapter, "next_observation", _next)


def test_a_sim_percept_wakes_the_pass_it_is_pending_in(monkeypatch, tmp_path):
    ev: list[tuple] = []
    _spy_sim_adapter(monkeypatch, ev)
    run = _run(monkeypatch, tmp_path, steps=5, ev=ev, adapter=None, percept_source=_Percepts(ev, {2}))
    assert _ran(run.ev) == [0, 2]
    assert _slept(run.ev) == [1, 3, 4]


def test_a_sim_percept_source_without_has_pending_always_wakes(monkeypatch, tmp_path):
    ev: list[tuple] = []
    _spy_sim_adapter(monkeypatch, ev)
    run = _run(monkeypatch, tmp_path, steps=4, ev=ev, adapter=None, percept_source=_PerceptsNoPending())
    assert _ran(run.ev) == [0, 1, 2, 3] and _slept(run.ev) == []


def test_a_recent_submit_holds_a_non_liveness_loop_awake_for_120_s(monkeypatch, tmp_path):
    from maxim.agents.llm_worker import LLMAttemptState

    ev: list[tuple] = []

    def _recent(r: Any) -> None:
        r.ctrl.last_llm_submit_time = time.time() - 119.0

    def _old(r: Any) -> None:
        r.ctrl.last_llm_submit_time = time.time() - 121.0

    worker = _Worker(ev, [LLMAttemptState.NONE])
    run = _run(monkeypatch, tmp_path, steps=5, ev=ev, llm_worker=worker, hooks={1: _recent, 3: _old})
    assert _ran(run.ev) == [0, 1, 2]
    assert _slept(run.ev) == [3, 4]
    assert worker.reads == 0  # without planning liveness the worker's job state is never read


def test_the_120_s_window_needs_a_worker(monkeypatch, tmp_path):
    def _recent(r: Any) -> None:
        r.ctrl.last_llm_submit_time = time.time()

    run = _run(monkeypatch, tmp_path, steps=3, hooks={1: _recent})
    assert _ran(run.ev) == [0]


def test_the_substrate_cadence_wakes_a_substrate_primary_loop(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, steps=4, aut_mode="substrate-primary", submit_interval=0.0)
    assert _ran(run.ev) == [0, 1, 2, 3]


def test_a_substrate_tick_not_yet_due_idles(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, steps=4, aut_mode="substrate-primary", submit_interval=1e12)
    assert _ran(run.ev) == [0] and _slept(run.ev) == [1, 2, 3]


def test_the_substrate_cadence_is_not_a_wake_source_for_llm_primary(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, steps=3, submit_interval=0.0)
    assert _ran(run.ev) == [0]


def test_a_pending_proposal_suspends_the_substrate_cadence(monkeypatch, tmp_path):
    """Substrate due needs no pending proposal: with one pending the pass still runs (the proposal is
    itself a wake source), and the cadence alone does not wake the pass after it is cleared at §1."""
    ev: list[tuple] = []
    adapter = _Adapter(ev)
    holder: list[Any] = []

    def _hook(r: Any) -> None:
        holder.append(r)
        r.ctrl.pending_proposal = _MARK
        r.ctrl.last_llm_submit_time = time.time()

    adapter.on_observe = lambda: holder and setattr(holder[0].ctrl, "pending_proposal", None)
    run = _run(monkeypatch, tmp_path, steps=4, ev=ev, adapter=adapter, aut_mode="substrate-primary", hooks={1: _hook})
    assert 1 in _ran(run.ev)


# ── §0.6 planning liveness (D13): the exact worker job state ─────────────────


def _liveness_run(monkeypatch, tmp_path, states, *, steps=6, hooks=None, requeue_ok=True, **kw) -> tuple[_Run, _Worker]:
    ev: list[tuple] = []
    worker = _Worker(ev, states, requeue_ok=requeue_ok)
    run = _run(monkeypatch, tmp_path, steps=steps, ev=ev, llm_worker=worker, planning_liveness=True, hooks=hooks, **kw)
    return run, worker


def _submitted(r: Any) -> None:
    """A planning submit happened and nothing came back since."""
    r.ctrl.last_llm_submit_time = time.time()
    r.ctrl.last_proposal_time = 0.0


@pytest.mark.parametrize("state_name", ["PENDING", "RUNNING", "COMPLETED"])
def test_an_active_job_keeps_the_loop_awake_and_is_never_a_failure(monkeypatch, tmp_path, state_name):
    from maxim.agents.llm_worker import LLMAttemptState

    run, worker = _liveness_run(monkeypatch, tmp_path, [LLMAttemptState[state_name]], steps=4, hooks={1: _submitted})
    assert _ran(run.ev) == [0, 1, 2, 3]
    assert not any(e[0].startswith("requeue") for e in run.ev)
    for es in _passes(run.ev).values():  # read at the gate, after the exhaustion check, every pass
        assert _kinds(es).index("state?") == _kinds(es).index("exhaustion?") + 1 < _kinds(es).index("run")
    assert run.error is None


def test_liveness_does_not_use_the_120_s_window(monkeypatch, tmp_path):
    from maxim.agents.llm_worker import LLMAttemptState

    def _recent(r: Any) -> None:  # a recent submit, but a proposal arrived after it: no backstop either
        r.ctrl.last_llm_submit_time = time.time()
        r.ctrl.last_proposal_time = time.time() + 1.0

    run, _ = _liveness_run(monkeypatch, tmp_path, [LLMAttemptState.NONE], steps=4, hooks={1: _recent})
    assert _ran(run.ev) == [0]
    assert not any(e[0].startswith("requeue") for e in run.ev)


@pytest.mark.parametrize("state_name", ["FAILED", "CANCELLED", "MISSING"])
def test_a_failed_job_spends_the_transport_budget_then_aborts_after_teardown(monkeypatch, tmp_path, state_name):
    from maxim.agents.llm_worker import LLMAttemptState
    from maxim.runtime.loop_controller import PlanningLivenessExhausted
    from tests.unit.test_loop_setup_characterization import _Hub

    hub = _Hub([])
    stamped: list[float] = []

    def _stamp(r: Any) -> None:
        _submitted(r)
        r.ctrl.last_llm_submit_time -= 1.0  # strictly earlier than any re-stamp, whatever the clock resolution
        stamped.append(r.ctrl.last_llm_submit_time)

    run, worker = _liveness_run(
        monkeypatch,
        tmp_path,
        [LLMAttemptState.NONE, LLMAttemptState[state_name]],
        steps=12,
        hooks={0: _stamp},
        memory_hub=hub,
    )
    reason = f"worker_job_{LLMAttemptState[state_name].value}"
    # Pass 0 runs (first step); passes 1-3 each requeue the last request and idle; pass 4 exhausts and breaks.
    assert _ran(run.ev) == [0]
    passes = _passes(run.ev)
    for step in (1, 2, 3):
        assert passes[step][-2:] == [("requeue_last", {}), ("sleep",)]
    assert max(passes) == 4
    assert passes[4][-1] == (
        "planning_liveness_exhausted",
        {"planning_streak": 0, "transport_streak": 4, "reason": reason, "status": "worker_unavailable"},
    )
    assert isinstance(run.error, PlanningLivenessExhausted)
    assert run.error.finish_status == "worker_unavailable"
    assert run.ctrl.planning_exhausted_reason == reason
    assert hub.ended == ["full"]  # teardown ran before the raise
    # Each retry re-stamps the submit time (the handler's pacing), so it moved past pass 0's stamp.
    assert run.ctrl.last_llm_submit_time > stamped[0]


@pytest.mark.parametrize("state_name", ["NONE", "CONSUMED"])
def test_a_job_that_completed_without_a_proposal_is_a_planning_failure(monkeypatch, tmp_path, state_name):
    from maxim.agents.llm_worker import LLMAttemptState
    from maxim.runtime.loop_controller import PlanningLivenessExhausted
    from tests.unit.test_loop_setup_characterization import _Hub

    hub = _Hub([])
    run, _ = _liveness_run(
        monkeypatch, tmp_path, [LLMAttemptState[state_name]], steps=12, hooks={0: _submitted}, memory_hub=hub
    )
    passes = _passes(run.ev)
    want = {"failed_tool": None, "reason": "planning_job_completed_without_proposal"}
    for step in (1, 2, 3):
        assert passes[step][-2:] == [("requeue_last", want), ("sleep",)]
    assert max(passes) == 4
    assert passes[4][-1][0] == "planning_liveness_exhausted"
    assert passes[4][-1][1]["planning_streak"] == 4 and passes[4][-1][1]["status"] == "planning_failed"
    assert isinstance(run.error, PlanningLivenessExhausted)
    assert run.error.finish_status == "planning_failed"
    assert hub.ended == ["full"]


def test_an_unreadable_job_state_is_missing_and_logged(monkeypatch, tmp_path, caplog):
    from maxim.agents.llm_worker import LLMAttemptState

    caplog.set_level(logging.WARNING, logger=LOGGER)
    run, _ = _liveness_run(
        monkeypatch,
        tmp_path,
        [LLMAttemptState.RUNNING, RuntimeError("pool gone"), LLMAttemptState.RUNNING],
        steps=3,
        hooks={0: _submitted},
    )
    passes = _passes(run.ev)
    assert passes[1][-2:] == [("requeue_last", {}), ("sleep",)]  # MISSING -> transport failure
    assert run.ctrl.planning_transport_failure_streak == 1
    records = [r for r in caplog.records if "planning worker state unavailable" in r.getMessage()]
    assert len(records) == 1 and records[0].name == LOGGER
    assert "pool gone" in records[0].getMessage()


def test_a_completed_job_resets_the_transport_streak(monkeypatch, tmp_path):
    from maxim.agents.llm_worker import LLMAttemptState

    run, _ = _liveness_run(
        monkeypatch,
        tmp_path,
        [LLMAttemptState.NONE, LLMAttemptState.FAILED, LLMAttemptState.FAILED, LLMAttemptState.COMPLETED],
        steps=5,
        hooks={0: _submitted},
    )
    assert run.ctrl.planning_transport_failure_streak == 0
    assert sum(1 for e in run.ev if e[0] == "requeue_last") == 2


def test_no_backstop_without_a_submit(monkeypatch, tmp_path):
    from maxim.agents.llm_worker import LLMAttemptState

    run, _ = _liveness_run(monkeypatch, tmp_path, [LLMAttemptState.FAILED], steps=4)
    assert not any(e[0].startswith("requeue") for e in run.ev)
    assert _slept(run.ev) == [1, 2, 3]


def test_no_backstop_once_a_proposal_arrived_after_the_submit(monkeypatch, tmp_path):
    from maxim.agents.llm_worker import LLMAttemptState

    def _answered(r: Any) -> None:
        r.ctrl.last_llm_submit_time = time.time()
        r.ctrl.last_proposal_time = r.ctrl.last_llm_submit_time  # equal counts as answered

    run, _ = _liveness_run(monkeypatch, tmp_path, [LLMAttemptState.FAILED], steps=4, hooks={0: _answered})
    assert not any(e[0].startswith("requeue") for e in run.ev)


def test_no_backstop_once_planning_is_exhausted(monkeypatch, tmp_path):
    from maxim.agents.llm_worker import LLMAttemptState

    def _exhausted(r: Any) -> None:
        _submitted(r)
        r.ctrl.planning_exhausted = True

    run, _ = _liveness_run(monkeypatch, tmp_path, [LLMAttemptState.FAILED], steps=4, hooks={0: _exhausted})
    assert not any(e[0].startswith("requeue") for e in run.ev)
    assert _slept(run.ev) == [1, 2, 3] and run.error is None


def test_an_active_job_after_a_submit_is_not_failed_by_the_backstop(monkeypatch, tmp_path):
    from maxim.agents.llm_worker import LLMAttemptState

    run, _ = _liveness_run(
        monkeypatch,
        tmp_path,
        [LLMAttemptState.NONE, LLMAttemptState.RUNNING, LLMAttemptState.COMPLETED, LLMAttemptState.CONSUMED],
        steps=4,
        hooks={0: _submitted},
    )
    passes = _passes(run.ev)
    assert _ran(run.ev) == [0, 1, 2]
    assert passes[3][-2:] == [
        ("requeue_last", {"failed_tool": None, "reason": "planning_job_completed_without_proposal"}),
        ("sleep",),
    ]


def test_a_rejected_requeue_still_idles_the_pass(monkeypatch, tmp_path):
    from maxim.agents.llm_worker import LLMAttemptState

    run, _ = _liveness_run(
        monkeypatch,
        tmp_path,
        [LLMAttemptState.NONE, LLMAttemptState.FAILED],
        steps=3,
        hooks={0: _submitted},
        requeue_ok=False,
    )
    assert _passes(run.ev)[1][-2:] == [("requeue_last", {}), ("sleep",)]
    assert run.ctrl.planning_transport_failure_streak == 2


def test_liveness_off_never_reads_the_job_state_or_requeues(monkeypatch, tmp_path):
    from maxim.agents.llm_worker import LLMAttemptState

    ev: list[tuple] = []
    worker = _Worker(ev, [LLMAttemptState.FAILED])

    def _old_submit(r: Any) -> None:
        r.ctrl.last_llm_submit_time = time.time() - 500.0

    run = _run(monkeypatch, tmp_path, steps=4, ev=ev, llm_worker=worker, hooks={0: _old_submit})
    assert worker.reads == 0
    assert not any(e[0].startswith("requeue") for e in run.ev)
    assert _slept(run.ev) == [1, 2, 3]
