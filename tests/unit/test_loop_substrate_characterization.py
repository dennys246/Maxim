"""Characterization of ``run_agentic_loop``'s SUBSTRATE TICK, §6b (1.3.2 decomposition, slice 3).

Pins, through the PUBLIC entry ``agent_loop.run_agentic_loop`` in ``aut_mode="substrate-primary"``, what
the substrate-primary cadence branch does on a pass where it is due: the turn-scoped action gate is
consulted BEFORE the proposer (a denied tick never proposes, still advances the submit clock and still
reports ``proposal=None, gated=True``), an idle proposer (``None``) leaves nothing pending, a proposal
becomes ``ctrl.pending_proposal`` and the sim's "EXEC" line names its tool, a telemetry writer that raises
never stops the loop, a pending proposal suspends the cadence (the ``ctrl.pending_proposal is None`` term
of ``loop_state._substrate_tick_due``, unpinned until now: the slice-2 review), telemetry gets the hub's
EC (or None without a hub), and the proposer gets the run's NAc, the hub's agent id, the situation cue and
the loop's sensor encoder by identity. Written BEFORE slice 3 moved the block out of ``agent_loop.py``
(``docs/plans/roadmap_1_3_x.md`` §"The decomposition", coverage-first rule). §6b is on the Exp 60/61/62
path (substrate-primary through ``propose_via_substrate``).

Observation is location-independent wherever it can be: a pass is seen through the loop's own
``loop_iteration`` event (the abstraction buffer, patched at its module), sleeps through the global
``time.sleep``, the controller through a class-level ``LoopController.__init__`` spy, telemetry and the
gate through fakes passed as arguments, and the sim's log line through ``sim_logger.sim_log``. The ONE
seam patched by name is the proposer, ``propose_via_substrate``, on the module the substrate tick reads
it from (``_seam()`` below): the slice that moves the tick retargets that one line, and that retarget is the
evidence the move kept the seam.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from maxim.agents.llm_worker import LLMProposal
from maxim.runtime.sim_adapter import NullSimulationAdapter
from tests.unit.test_loop_setup_characterization import _Hub

LOGGER = "maxim.runtime.agent_loop"
IDLE = 0.0123  # a distinctive idle_sleep_s, so the gate's idle sleeps are recognisable
HOUR = 3600.0  # a submit interval no run outlasts: the cadence is due on the first pass only


def _seam() -> Any:
    """The module ``run_agentic_loop``'s substrate tick reads ``propose_via_substrate`` from."""
    from maxim.runtime import substrate_proposal

    return substrate_proposal


# ── harness ──────────────────────────────────────────────────────────────────


class _Agent:
    name = "substrate-probe"

    def propose_intent(self, state: Any, memory: Any) -> None:
        return None


class _Telemetry:
    """``substrate_telemetry``: records every snapshot's keyword arguments; optionally raises."""

    def __init__(self, raises: bool = False) -> None:
        self.snaps: list[dict[str, Any]] = []
        self.raises = raises

    def snapshot(self, **kw: Any) -> None:
        self.snaps.append(kw)
        if self.raises:
            raise RuntimeError("telemetry writer failed")


class _Percepts:
    """A percept source that never has anything: it only makes the run a sim run (``is_sim_mode``)."""

    def next_percept(self) -> None:
        return None

    def has_pending(self) -> bool:
        return False

    def is_exhausted(self) -> bool:
        return False


def _proposal(tool: str | None = "probe_tool") -> LLMProposal:
    return LLMProposal(
        request_id="substrate-probe",
        action={"tool_name": tool, "params": {}} if tool is not None else None,
        reasoning="probe reasoning",
        strategy_used="substrate-primary",
        confidence=0.75,
        mode_goal_achieved=False,
        triggering_input="",
    )


class _Run(SimpleNamespace):
    ev: list[tuple]
    ctrl: Any
    calls: list[dict[str, Any]]
    executor: Any


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    steps: int = 4,
    submit_interval: float = 0.0,
    propose: Callable[[int], Any] = lambda n: None,
    **kwargs: Any,
) -> _Run:
    """Run the real substrate-primary loop for ``steps`` passes. ``propose(n)`` is what the proposer returns
    on its n-th call (from 1); every call's keyword arguments are recorded in ``run.calls``. The
    controller's submit cadence is ``submit_interval`` (0.0 = due on every pass with nothing pending)."""
    import maxim.utils.structured_logging as sl
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime.bootstrap import build_decision_engine, build_executor, build_memory
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.state import RuntimeState
    from maxim.tools.registry import ToolRegistry

    monkeypatch.chdir(tmp_path)
    ev: list[tuple] = []
    run = _Run(ev=ev, ctrl=None, calls=[], executor=None)

    real_init = LoopController.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        self.llm_submit_interval = submit_interval
        run.ctrl = self

    monkeypatch.setattr(LoopController, "__init__", _init)

    def _propose(**kw: Any) -> Any:
        run.calls.append(kw)
        ev.append(("propose",))
        return propose(len(run.calls))

    monkeypatch.setattr(_seam(), "propose_via_substrate", _propose)

    real_buffer = sl.get_abstraction_buffer()

    class _Buffer:
        def append(self, record: Any) -> None:
            if record.source == "agent_loop" and record.event == "loop_iteration":
                ev.append(("iter", record.data["step"]))
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

    executor = kwargs.pop("executor", None) or build_executor(ToolRegistry(), pain_bus=None, permissions=None)
    run.executor = executor
    state = RuntimeState()
    state.data["mode"] = "active"
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    if "percept_source" not in kwargs:
        kwargs.setdefault("sim_adapter", NullSimulationAdapter())
    kwargs.setdefault("target_hz", 1e6)
    AL.run_agentic_loop(
        _Agent(),
        FileSystemEnv(str(workspace)),
        state,
        build_memory(),
        build_decision_engine(),
        executor,
        aut_mode="substrate-primary",
        stop_event=threading.Event(),
        max_steps=steps,
        idle_sleep_s=IDLE,
        **kwargs,
    )
    return run


def _passes_with(ev: list[tuple], kind: str) -> list[int]:
    """The passes (by step) in which an event of ``kind`` happened."""
    out: list[int] = []
    step = -1
    for e in ev:
        if e[0] == "iter":
            step = e[1]
        elif e[0] == kind and step not in out:
            out.append(step)
    return out


# ── (a) the turn-scoped action gate ──────────────────────────────────────────


def test_a_denied_tick_never_proposes_but_advances_the_clock_and_reports_gated(monkeypatch, tmp_path):
    asked: list[int] = []
    tel = _Telemetry()

    def _gate() -> bool:
        asked.append(1)
        return False

    before = time.time()
    run = _run(
        monkeypatch, tmp_path, steps=4, submit_interval=HOUR, substrate_action_gate=_gate, substrate_telemetry=tel
    )
    assert run.calls == [], "a denied tick must not reach the proposer"
    assert len(asked) == 1  # due on the first pass only: the denied tick still restarted the cadence
    assert run.ctrl.last_llm_submit_time >= before
    assert _passes_with(run.ev, "sleep") == [1, 2, 3]  # nothing else is due after it
    [snap] = tel.snaps
    assert snap["proposal"] is None and snap["gated"] is True
    assert run.ctrl.pending_proposal is None


def test_an_allowing_gate_is_asked_on_every_due_tick_before_the_proposer(monkeypatch, tmp_path):
    order: list[str] = []
    tel = _Telemetry()

    def _gate() -> bool:
        order.append("gate")
        return True

    def _propose(n: int) -> None:
        order.append("propose")
        return None

    run = _run(monkeypatch, tmp_path, steps=3, propose=_propose, substrate_action_gate=_gate, substrate_telemetry=tel)
    assert order == ["gate", "propose"] * 3
    assert len(run.calls) == 3
    assert [s["gated"] for s in tel.snaps] == [False, False, False]


def test_no_gate_means_every_due_tick_proposes(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, steps=3)
    assert len(run.calls) == 3
    assert _passes_with(run.ev, "propose") == [0, 1, 2]


# ── (b) an idle proposer ─────────────────────────────────────────────────────


def test_an_idle_proposer_leaves_nothing_pending_and_reports_an_ungated_none(monkeypatch, tmp_path):
    tel = _Telemetry()
    before = time.time()
    run = _run(monkeypatch, tmp_path, steps=4, submit_interval=HOUR, substrate_telemetry=tel)
    assert len(run.calls) == 1
    assert run.ctrl.pending_proposal is None
    assert run.ctrl.last_llm_submit_time >= before
    [snap] = tel.snaps
    assert snap["proposal"] is None and snap["gated"] is False


def test_telemetry_fires_on_every_due_tick_with_the_step_and_executor(monkeypatch, tmp_path):
    tel = _Telemetry()
    run = _run(monkeypatch, tmp_path, steps=3, substrate_telemetry=tel)
    assert [s["step"] for s in tel.snaps] == [0, 1, 2]
    assert all(s["executor"] is run.executor for s in tel.snaps)


# ── (c) a proposal ───────────────────────────────────────────────────────────


def test_a_proposal_becomes_the_pending_proposal_and_its_telemetry(monkeypatch, tmp_path):
    p = _proposal()
    tel = _Telemetry()
    run = _run(monkeypatch, tmp_path, steps=1, propose=lambda n: p, substrate_telemetry=tel)
    assert run.ctrl.pending_proposal is p
    [snap] = tel.snaps
    assert snap["proposal"] is p and snap["gated"] is False


def test_a_sim_run_logs_the_proposal_as_an_exec_line_with_its_tool(monkeypatch, tmp_path):
    import maxim.simulation.sim_logger as sim_logger

    lines: list[tuple[str, str]] = []
    monkeypatch.setattr(sim_logger, "sim_log", lambda cat, msg, data=None: lines.append((cat, msg)))
    p = _proposal("probe_tool")
    run = _run(monkeypatch, tmp_path, steps=1, propose=lambda n: p, percept_source=_Percepts())
    assert run.ctrl.pending_proposal is p
    execs = [m for c, m in lines if c == "EXEC" and "substrate-primary proposal" in m]
    assert execs == ["substrate-primary proposal: tool=probe_tool confidence=0.75 reasoning=probe reasoning"]


def test_a_non_sim_run_logs_no_exec_line(monkeypatch, tmp_path):
    import maxim.simulation.sim_logger as sim_logger

    lines: list[tuple[str, str]] = []
    monkeypatch.setattr(sim_logger, "sim_log", lambda cat, msg, data=None: lines.append((cat, msg)))
    _run(monkeypatch, tmp_path, steps=1, propose=lambda n: _proposal())
    assert not [m for c, m in lines if "substrate-primary proposal" in m]


# ── (d) a failing telemetry writer ───────────────────────────────────────────


def test_a_raising_telemetry_writer_never_stops_the_loop(monkeypatch, tmp_path, caplog):
    tel = _Telemetry(raises=True)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        run = _run(monkeypatch, tmp_path, steps=3, substrate_telemetry=tel)
    assert len(tel.snaps) == 3 and len(run.calls) == 3
    raised = [r for r in caplog.records if r.getMessage() == "substrate telemetry callback raised"]
    assert len(raised) == 3
    assert {r.name for r in raised} == {LOGGER} and all(r.exc_info for r in raised)


# ── (e) a pending proposal suspends the cadence ──────────────────────────────


def test_a_pending_proposal_suspends_the_substrate_cadence(monkeypatch, tmp_path):
    """The ``ctrl.pending_proposal is None`` term of ``_substrate_tick_due``. A proposal with no action is
    never executed (§4 needs one), so it stays pending: it wakes every later pass (pending work), but the
    cadence (due on EVERY pass at interval 0) must not propose again while it is there."""
    held = _proposal(tool=None)
    run = _run(monkeypatch, tmp_path, steps=5, propose=lambda n: held)
    assert len(run.calls) == 1, "the substrate proposer ran again while a proposal was pending"
    assert run.ctrl.pending_proposal is held
    assert _passes_with(run.ev, "sleep") == []  # every pass ran: the held proposal is itself a wake source


# ── (f) the EC telemetry is handed ───────────────────────────────────────────


def test_telemetry_gets_the_hubs_ec(monkeypatch, tmp_path):
    from maxim.decisions.nac import NAc
    from maxim.similarity.ec import EntorhinalCortex

    ec = EntorhinalCortex()
    hub = _Hub([], nac=NAc(), ec=ec, cue=lambda *a, **k: None)
    tel = _Telemetry()
    _run(monkeypatch, tmp_path, steps=2, memory_hub=hub, substrate_telemetry=tel)
    assert len(tel.snaps) == 2 and all(s["ec"] is ec for s in tel.snaps)


def test_without_a_hub_telemetry_gets_no_ec(monkeypatch, tmp_path):
    tel = _Telemetry()
    _run(monkeypatch, tmp_path, steps=2, substrate_telemetry=tel)
    assert len(tel.snaps) == 2 and all(s["ec"] is None for s in tel.snaps)


# ── (g) the proposer's arguments ─────────────────────────────────────────────


def test_the_proposer_gets_the_runs_handles_by_identity(monkeypatch, tmp_path):
    from maxim.decisions.nac import NAc
    from maxim.runtime import loop_setup

    nac, cue, encoder = NAc(), (lambda *a, **k: None), object()
    monkeypatch.setattr(loop_setup, "_build_loop_sensor_encoder", lambda hub, n: encoder)
    hub = _Hub([], nac=nac, cue=cue)
    tel = _Telemetry()
    run = _run(monkeypatch, tmp_path, steps=2, memory_hub=hub, substrate_telemetry=tel)
    assert len(run.calls) == 2
    for kw in run.calls:
        assert set(kw) == {"nac", "agent_id", "executor", "situation_cue", "sensor_encoder"}
        assert kw["nac"] is nac
        assert kw["agent_id"] == "hub-agent-7"
        assert kw["executor"] is run.executor
        assert kw["situation_cue"] is cue
        assert kw["sensor_encoder"] is encoder
    assert all(s["nac"] is nac for s in tel.snaps)


def test_without_a_hub_the_proposer_gets_no_nac_and_the_explicit_no_cue(monkeypatch, tmp_path):
    run = _run(monkeypatch, tmp_path, steps=1)
    [kw] = run.calls
    assert kw["nac"] is None and kw["sensor_encoder"] is None
    assert kw["agent_id"] == "substrate-probe"
    assert repr(kw["situation_cue"]) == "NO_SITUATION_CUE"


def test_both_moved_modules_keep_the_loop_logger_name():
    """Slice 3 moved code out of agent_loop without renaming its log records: both the tick and the leaf
    proposer (whose readers the Executor also calls) still log as ``maxim.runtime.agent_loop``, as the
    slice-1/2 modules do, so a filter or handler keyed on that name sees the same records."""
    from maxim.runtime import loop_substrate, substrate_proposal

    assert loop_substrate.logger.name == LOGGER
    assert substrate_proposal.logger.name == LOGGER
