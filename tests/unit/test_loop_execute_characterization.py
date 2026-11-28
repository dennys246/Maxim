"""Characterization of ``run_agentic_loop``'s §4, the pending-action execution (1.3.6 loop phase 2, slice 8).

Pins, through the PUBLIC entry ``agent_loop.run_agentic_loop``, what §4 does with the proposal §2 installed: the
#1451 re-plan rule runs first (a run-ending answer with an input held behind it is set aside once); the proposal is
re-stamped with the live situation (``substrate_proposal._attach_live_situation``); the consecutive same-tool cap
blocks the sixth identical call in a row (a SYSTEM outcome names it and the counters reset; varied params or another
tool reset the count, and the count is carried across passes); ``parallel_actions`` run together as one
``batched_exploration`` follow-up; a plan that needs approval goes through the plan gate; an allowed action runs and
its follow-up is queued; at PLANNING a refused action is queued for approval; a confirmation request is parked;
and a hard rejection is booked, its follow-up queued. A ``continue`` in §4 skips the rest of the pass, so the pass is
not counted in ``state.steps_taken`` (§8). Written BEFORE slice 8 moved the block out of ``_run_agentic_loop_body``
and kept green unchanged by that move (``docs/plans/roadmap_1_3_x.md`` §1.3.6; coverage-first rule).

Observation is location-independent, so the pins hold wherever §4's code lives: registered probe tools record what
ran (the main probe asks for a ``process`` follow-up, patched at ``tool_dispatch``, which builds it); the controller is captured by a class-level ``LoopController.__init__`` spy, and its ``recent_outcomes``,
``pending_action_followup`` and ``pending_proposal`` read at the end; the controller's ``park_confirmation`` and
``queue_for_approval`` are spied at the class; the autonomy check is the passed controller's own; the live-situation
stamp is patched at its home module, the binding §4 calls through. Nothing patches ``agent_loop`` or the module §4
moves to.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.policy_helpers import TEST_POLICY

PROBE = "probe_execute"  # the main registered tool
OTHER = "probe_execute_other"  # a second one, for "another tool resets the count"
FINISH = "probe_execute_finish"  # a tool that ends the run (``Tool.ends_run``), for #1451's re-plan rule
CAP = 5  # agent_loop's ``_MAX_CONSECUTIVE_SAME_TOOL``


def _proposal(
    tool: str = PROBE,
    *,
    n: int = 1,
    parallel: list[dict[str, Any]] | None = None,
    plan_text: str = "",
    requires_approval: bool = False,
    confidence: float = 0.9,
    rid: str = "p",
) -> Any:
    from maxim.agents.llm_types import LLMProposal

    return LLMProposal(
        request_id=rid,
        action={"tool_name": tool, "params": {"n": n}},
        reasoning="because",
        strategy_used="llm",
        confidence=confidence,
        mode_goal_achieved=False,
        timestamp=time.time(),
        parallel_actions=list(parallel or []),
        plan_text=plan_text,
        requires_approval=requires_approval,
        triggering_input="hello",
    )


class _Source:
    """A sim percept source with no ``has_pending``: the idle gate wakes every pass, and §1 calls
    ``next_percept`` once per full pass, so ``passes`` is the pass index."""

    def __init__(self) -> None:
        self.passes = -1

    def next_percept(self) -> None:
        self.passes += 1
        return None

    def is_exhausted(self) -> bool:
        return False


class _Worker:
    """``script[k]`` is what ``get_latest_proposal`` returns on pass k (missing = None)."""

    def __init__(self, source: _Source, script: dict[int, Any]) -> None:
        self.source, self.script = source, dict(script)

    def latest_attempt_state(self) -> Any:
        from maxim.agents.llm_types import LLMAttemptState

        return LLMAttemptState.NONE

    def get_latest_proposal(self) -> Any:
        return self.script.pop(self.source.passes, None)

    def submit_context(self, *_a: Any, **_k: Any) -> bool:
        raise AssertionError("the submit cadence is held off in this driver")

    def requeue_request(self, *_a: Any, **_k: Any) -> bool:
        return True

    def requeue_last_request(self, **_k: Any) -> bool:
        return True

    def record_outcome(self, *_a: Any, **_k: Any) -> None:
        return None


class _Agent:
    name = "execute-probe"

    def propose_intent(self, state: Any, memory: Any) -> None:
        return None


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script: dict[int, Any],
    *,
    steps: int = 3,
    level: str = "AUTONOMOUS",
    verdict: Any = None,
    liveness: bool = False,
    heard: list[str] | None = None,
    situation: Any = None,
) -> SimpleNamespace:
    import maxim.simulation.sim_logger as sim_logger
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime.bootstrap import build_decision_engine, build_executor, build_memory
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.state import RuntimeState
    from maxim.tools.base import Tool, ToolOutput
    from maxim.tools.registry import ToolRegistry

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_logger, "_interactive_mode", sim_logger.InteractiveMode.OFF)
    obs = SimpleNamespace(executed=[], ctrl=None, error=None, state=None, parked=[], queued=[], situations=[])

    real_init = LoopController.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        self.llm_submit_interval = 1e12  # §6 never submits: §4 sees only the scripted proposals
        obs.ctrl = self

    monkeypatch.setattr(LoopController, "__init__", _init)
    real_park, real_queue = LoopController.park_confirmation, LoopController.queue_for_approval

    def _park(self: Any, action: dict[str, Any], *, sim: Any) -> None:
        obs.parked.append(dict(action))
        real_park(self, action, sim=sim)

    def _queue(self: Any, action: dict[str, Any], *, confidence: float, sim: Any) -> None:
        obs.queued.append((dict(action), confidence))
        real_queue(self, action, confidence=confidence, sim=sim)

    from maxim.runtime import tool_dispatch

    real_followup_type = tool_dispatch.get_tool_followup_type
    monkeypatch.setattr(  # the probe asks for a follow-up, so §4's install of it is observable
        tool_dispatch,
        "get_tool_followup_type",
        lambda name, mode, overlay=None: "process"
        if name == PROBE
        else real_followup_type(name, mode, overlay=overlay),
    )
    monkeypatch.setattr(LoopController, "park_confirmation", _park)
    monkeypatch.setattr(LoopController, "queue_for_approval", _queue)

    if situation is not None:
        from maxim.runtime import substrate_proposal

        real_attach = substrate_proposal._attach_live_situation

        def _attach(proposal: Any, **kw: Any) -> Any:
            obs.situations.append(dict(kw))
            return situation(real_attach(proposal, **kw))

        monkeypatch.setattr(substrate_proposal, "_attach_live_situation", _attach)

    def _tool(tool_name: str, ends_run: bool = False) -> Any:
        class _Probe(Tool):
            name = tool_name
            description = "stub"
            input_schema: dict = {}

            def execute(self, **kwargs: Any) -> Any:
                obs.executed.append((tool_name, dict(kwargs)))
                return ToolOutput(success=True, output="ok")

        _Probe.ends_run = ends_run  # type: ignore[attr-defined]
        return _Probe()

    registry = ToolRegistry()
    for t in (_tool(PROBE), _tool(OTHER), _tool(FINISH, ends_run=True)):
        registry.register(t)
    executor = build_executor(registry, pain_bus=None, policy=TEST_POLICY)

    class _Autonomy(AutonomyController):
        def action_verdict(self, action: dict[str, Any], confidence: float | None = None) -> Any:
            return verdict if verdict is not None else super().action_verdict(action, confidence=confidence)

    state = RuntimeState()
    state.data["mode"] = "active"
    obs.state = state
    memory = build_memory()
    if heard is not None:  # what the memory heard and the loop has not submitted (#1451's unread inputs)
        memory.recent_cli_inputs = lambda: list(heard)  # type: ignore[attr-defined]
    source = _Source()
    (tmp_path / "ws").mkdir(exist_ok=True)
    try:
        AL.run_agentic_loop(
            _Agent(),
            FileSystemEnv(str(tmp_path / "ws")),
            state,
            memory,
            build_decision_engine(),
            executor,
            llm_worker=_Worker(source, script),
            autonomy_controller=_Autonomy(initial_level=getattr(AutonomyLevel, level)),
            max_steps=steps,
            target_hz=1e6,
            idle_sleep_s=0.0,
            stop_event=threading.Event(),
            planning_liveness=liveness,
            percept_source=source,
            response_policy=None,  # no bridge, so no ResponsePolicy: no gating prompt is answered (#1185)
        )
    except Exception as exc:
        obs.error = exc
    return obs


def _outcomes(obs: Any) -> list[Any]:
    return list(obs.ctrl.recent_outcomes)


def _cap_outcomes(obs: Any) -> list[Any]:
    return [o for o in _outcomes(obs) if "blocked to prevent a loop" in str(o)]


# ── an allowed action runs ────────────────────────────────────────────────────


def test_an_allowed_action_runs_its_followup_is_queued_and_the_proposal_cleared(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, {0: _proposal()}, steps=2)
    assert obs.error is None
    assert obs.executed == [(PROBE, {"n": 1})]
    assert obs.ctrl.pending_proposal is None
    followup = obs.ctrl.pending_action_followup
    assert followup is not None and followup.tool == PROBE
    assert obs.state.steps_taken == 2  # nothing in §4 skipped a pass


def test_the_proposal_is_restamped_with_the_live_situation_before_it_runs(monkeypatch, tmp_path):
    """The stamp's return replaces the pending proposal; the action §4 runs is the one read BEFORE the stamp."""
    stamped: list[Any] = []

    def _mark(p: Any) -> Any:
        import dataclasses

        marked = dataclasses.replace(p, triggering_input="stamped")
        stamped.append(marked)
        return marked

    obs = _run(monkeypatch, tmp_path, {0: _proposal()}, steps=2, situation=_mark)
    assert obs.executed == [(PROBE, {"n": 1})]
    [kw] = obs.situations
    assert kw["aut_mode"] == "llm-primary"
    assert {"sensor_encoder", "agent_id", "executor"} <= set(kw)
    assert len(stamped) == 1
    assert obs.ctrl.pending_action_followup.original_query == "stamped"  # the stamped proposal is the one run


# ── the consecutive same-tool cap ─────────────────────────────────────────────


def test_the_sixth_identical_call_in_a_row_is_capped_and_the_pass_skipped(monkeypatch, tmp_path):
    script = {k: _proposal(rid=f"p{k}") for k in range(CAP + 1)}
    obs = _run(monkeypatch, tmp_path, script, steps=CAP + 2)
    assert obs.executed == [(PROBE, {"n": 1})] * CAP  # the sixth is blocked
    [capped] = _cap_outcomes(obs)
    text = str(capped)
    assert f"'{PROBE}' was called {CAP + 1} times with identical parameters" in text
    assert "Try a DIFFERENT tool or different parameters." in text
    assert OTHER in text and FINISH in text  # the other tools the model may use instead
    assert obs.ctrl.pending_proposal is None
    assert obs.state.steps_taken == CAP + 1  # the capped pass is skipped (§8 not reached)


def test_after_the_cap_the_count_restarts(monkeypatch, tmp_path):
    script = {k: _proposal(rid=f"p{k}") for k in range(2 * CAP + 2)}
    obs = _run(monkeypatch, tmp_path, script, steps=2 * CAP + 3)
    assert len(obs.executed) == 2 * CAP  # capped on the 6th and the 12th: the counters were reset after the cap
    assert len(_cap_outcomes(obs)) == 2


def test_varied_params_are_never_capped(monkeypatch, tmp_path):
    script = {k: _proposal(n=k, rid=f"p{k}") for k in range(CAP + 3)}
    obs = _run(monkeypatch, tmp_path, script, steps=CAP + 4)
    assert [p["n"] for _, p in obs.executed] == list(range(CAP + 3))
    assert _cap_outcomes(obs) == []


def test_a_repeat_after_varied_params_counts_from_one(monkeypatch, tmp_path):
    """Same tool, other params: the count restarts at 1 and the new params become the ones compared."""
    script = {0: _proposal(n=0, rid="a")}
    script.update({k: _proposal(n=1, rid=f"p{k}") for k in range(1, CAP + 2)})  # 1 + CAP + ... identical n=1
    obs = _run(monkeypatch, tmp_path, script, steps=CAP + 3)
    assert [p["n"] for _, p in obs.executed] == [0] + [1] * CAP  # the (CAP+1)-th identical n=1 is capped
    assert len(_cap_outcomes(obs)) == 1


def test_another_tool_resets_the_count(monkeypatch, tmp_path):
    script = {k: _proposal(rid=f"p{k}") for k in range(CAP)}
    script[CAP] = _proposal(OTHER, rid="o")
    script.update({k: _proposal(rid=f"q{k}") for k in range(CAP + 1, 2 * CAP + 1)})
    obs = _run(monkeypatch, tmp_path, script, steps=2 * CAP + 2)
    assert len(obs.executed) == 2 * CAP + 1  # nothing capped: the other tool broke the run
    assert _cap_outcomes(obs) == []


def test_the_count_is_carried_across_passes_with_no_proposal(monkeypatch, tmp_path):
    """Passes with nothing pending leave the counters as they are: the cap counts calls, not passes."""
    script = {2 * k: _proposal(rid=f"p{k}") for k in range(CAP + 1)}  # one every other pass
    obs = _run(monkeypatch, tmp_path, script, steps=2 * CAP + 3)
    assert len(obs.executed) == CAP
    assert len(_cap_outcomes(obs)) == 1


# ── parallel actions ──────────────────────────────────────────────────────────


def test_parallel_actions_run_together_as_one_batched_followup_and_skip_the_pass(monkeypatch, tmp_path):
    proposal = _proposal(parallel=[{"tool_name": OTHER, "params": {"n": 2}}])
    obs = _run(monkeypatch, tmp_path, {0: proposal}, steps=2)
    assert sorted(obs.executed) == sorted([(PROBE, {"n": 1}), (OTHER, {"n": 2})])
    followup = obs.ctrl.pending_action_followup
    assert followup is not None and followup.tool == "batched_exploration"
    assert followup.followup_type == "process"
    assert followup.original_query == "hello"
    assert obs.ctrl.pending_proposal is None
    assert obs.state.steps_taken == 1  # the parallel pass is skipped


# ── the plan gate ─────────────────────────────────────────────────────────────


def test_a_plan_that_needs_approval_goes_through_the_plan_gate_and_skips_the_pass(monkeypatch, tmp_path):
    from maxim.runtime import loop_controller

    monkeypatch.setattr(loop_controller, "no_person_to_ask", lambda: True)  # the non-interactive machine approval
    proposal = _proposal(plan_text="1. run the probe", requires_approval=True)
    obs = _run(monkeypatch, tmp_path, {0: proposal}, steps=2)
    assert obs.executed == [(PROBE, {"n": 1})]  # run once, by the gate's approved path
    assert obs.ctrl.pending_proposal is None
    assert obs.state.steps_taken == 1  # the gate's pass is skipped


# ── an action the autonomy check does not allow ──────────────────────────────


def test_at_planning_a_refused_action_is_queued_for_approval(monkeypatch, tmp_path):
    from maxim.agents.autonomy import VERDICT_REFUSE, ActionVerdict

    obs = _run(
        monkeypatch, tmp_path, {0: _proposal()}, steps=1, level="PLANNING", verdict=ActionVerdict(VERDICT_REFUSE, "x")
    )
    assert obs.queued == [({"tool_name": PROBE, "params": {"n": 1}}, 0.9)]
    assert obs.parked == []
    assert obs.ctrl.pending_proposal is None
    assert obs.state.steps_taken == 1


def test_a_confirmation_request_is_parked_and_the_proposal_cleared(monkeypatch, tmp_path):
    from maxim.agents.autonomy import VERDICT_CONFIRM, ActionVerdict
    from maxim.utils.input_source import HUMAN_APPROVERS

    verdict = ActionVerdict(VERDICT_CONFIRM, "needs a yes", satisfied_by=HUMAN_APPROVERS)
    obs = _run(monkeypatch, tmp_path, {0: _proposal()}, steps=1, level="SUPERVISED", verdict=verdict)
    assert obs.executed == []  # parked: the answer arrives as the next pass's input
    assert obs.parked == [{"tool_name": PROBE, "params": {"n": 1}}]
    assert obs.queued == []
    assert obs.ctrl.pending_proposal is None


def test_a_hard_rejection_is_booked_and_its_followup_queued(monkeypatch, tmp_path):
    from maxim.agents.autonomy import VERDICT_REFUSE, ActionVerdict

    verdict = ActionVerdict(VERDICT_REFUSE, "forbidden here")
    obs = _run(monkeypatch, tmp_path, {0: _proposal()}, steps=2, level="SUPERVISED", verdict=verdict)
    assert obs.executed == [] and obs.parked == [] and obs.queued == []
    assert obs.ctrl.pending_proposal is None
    followup = obs.ctrl.pending_action_followup
    assert followup is not None and followup.tool == PROBE
    assert "forbidden here" in str(followup.result)


# ── #1451: a run-ending answer with an input held behind it ──────────────────


def test_a_run_ending_answer_with_a_held_input_is_set_aside_once(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, {0: _proposal(FINISH)}, steps=2, liveness=True, heard=["a line held behind it"])
    assert obs.executed == []  # set aside before it ran
    assert obs.ctrl.run_end_set_aside is True
    assert any("new input(s) to read before finishing" in str(o) for o in _outcomes(obs))


def test_without_liveness_a_run_ending_answer_runs(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, {0: _proposal(FINISH)}, steps=2, heard=["a line held behind it"])
    assert obs.executed == [(FINISH, {"n": 1})]
    assert not any("new input(s) to read before finishing" in str(o) for o in _outcomes(obs))
