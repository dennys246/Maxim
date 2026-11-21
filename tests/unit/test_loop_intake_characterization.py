"""Characterization of ``run_agentic_loop``'s §2, the LLM proposal intake (1.3.6 loop phase 2, slice 6).

Pins, through the PUBLIC entry ``agent_loop.run_agentic_loop``, what §2 does with what the worker hands back:
a fresh proposal stamps the D13 clock and is installed (its ``next_actions`` queued and run one per pass as
``multi_step`` continuations); a stale one is dropped; in sim mode a parse-failure fallback or a proposal naming
an unregistered tool is dropped, the bad name recorded in the executor's hallucinated-tool list, the pending
user input cleared and ``_llm_unavailable`` surfaced to the action sink; an error proposal and (under planning
liveness) a proposal with no action are dropped; and under planning liveness every drop reschedules the turn
through the worker and, with the budget spent, ends the run with ``PlanningLivenessExhausted`` and its status.
Written BEFORE slice 6 moved the block out of ``_run_agentic_loop_body`` and kept green unchanged by that move
(``docs/plans/roadmap_1_3_x.md`` §1.3.6; coverage-first rule). §2 is on the LLM-primary path.

Observation is location-independent, so the pins hold wherever §2's code lives: the worker (a fake passed as
``llm_worker``) records every requeue; the action sink is a real ``RecordingSink``; a registered probe tool
records what ran; the abstraction buffer (patched at its module) records the loop's ``log_agentic`` events;
``on_event`` records the stream; the controller is captured by a class-level ``LoopController.__init__`` spy.
Nothing patches ``agent_loop`` or the module §2 moves to.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.policy_helpers import TEST_POLICY

PROBE = "probe_intake"  # the one registered tool
UNKNOWN = "no_such_tool_intake"  # a name no registry carries
EVENTS = ("llm_response", "multi_step_queued", "error", "planning_retry", "planning_liveness_exhausted")
PENDING_KEYS = ("pending_user_input", "pending_user_input_time", "pending_user_input_source")


def _proposal(
    tool: str | None = PROBE,
    *,
    reasoning: str = "because",
    error: str | None = None,
    next_actions: list[dict[str, Any]] | None = None,
    age_s: float = 0.0,
    original_request: Any = None,
    ready_to_act: bool = True,
    mode_goal_achieved: bool = False,
    confidence: float = 0.9,
    rid: str = "p",
) -> Any:
    from maxim.agents.llm_types import LLMProposal

    return LLMProposal(
        request_id=rid,
        action={"tool_name": tool, "params": {"n": 1}} if tool else {},
        reasoning=reasoning,
        strategy_used="llm",
        confidence=confidence,
        mode_goal_achieved=mode_goal_achieved,
        timestamp=time.time() - age_s,
        error=error,
        next_actions=list(next_actions or []),
        original_request=original_request,
        ready_to_act=ready_to_act,
    )


class _Source:
    """A sim percept source with no ``has_pending``: the idle gate wakes every pass, and §1 calls
    ``next_percept`` once per full pass, so ``passes`` is the pass index §2 runs in."""

    def __init__(self) -> None:
        self.passes = -1

    def next_percept(self) -> None:
        self.passes += 1
        return None

    def is_exhausted(self) -> bool:
        return False


class _Worker:
    """``script[k]`` is what ``get_latest_proposal`` returns on pass k (missing = None)."""

    def __init__(self, source: _Source, script: dict[int, Any], requeue_ok: bool = True) -> None:
        self.source, self.script, self.requeue_ok = source, dict(script), requeue_ok
        self.requeues: list[tuple[str, Any, dict[str, Any]]] = []
        self.consumed: list[int] = []

    def latest_attempt_state(self) -> Any:
        from maxim.agents.llm_types import LLMAttemptState

        return LLMAttemptState.NONE

    def get_latest_proposal(self) -> Any:
        p = self.script.pop(self.source.passes, None)
        if p is not None:
            self.consumed.append(self.source.passes)
        return p

    def submit_context(self, *_a: Any, **_k: Any) -> bool:
        raise AssertionError("the submit cadence is held off in this driver")

    def requeue_request(self, request: Any, **kw: Any) -> bool:
        self.requeues.append(("request", request, kw))
        return self.requeue_ok

    def requeue_last_request(self, **kw: Any) -> bool:
        self.requeues.append(("last", None, kw))
        return self.requeue_ok

    def record_outcome(self, *_a: Any, **_k: Any) -> None:
        return None


class _ToolIndex:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], str]] = []

    def record_surfaced_but_unused(self, goal: str, surfaced: Any, chosen: str) -> None:
        self.calls.append((goal, list(surfaced), chosen))


class _Agent:
    name = "intake-probe"

    def propose_intent(self, state: Any, memory: Any) -> None:
        return None


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script: dict[int, Any],
    *,
    steps: int = 4,
    liveness: bool = False,
    sink: bool = True,
    sim: bool = True,
    pending_input: bool = True,
    surfaced: list[str] | None = None,
    goal: str = "",
    on_event: Any = None,
    transport_streak: int = 0,
) -> SimpleNamespace:
    import maxim.simulation.sim_logger as sim_logger
    import maxim.utils.structured_logging as sl
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime.bootstrap import build_decision_engine, build_executor, build_memory
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.sim_adapter import NullSimulationAdapter
    from maxim.runtime.state import RuntimeState
    from maxim.simulation.sinks import RecordingSink
    from maxim.tools.base import Tool, ToolOutput
    from maxim.tools.registry import ToolRegistry

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_logger, "_interactive_mode", sim_logger.InteractiveMode.OFF)
    obs = SimpleNamespace(executed=[], events=[], stream=[], ctrl=None, error=None, state=None, worker=None, sink=None)

    real_init = LoopController.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        self.llm_submit_interval = 1e12  # §6 never submits: §2 sees only the scripted proposals
        if surfaced is not None:
            self.last_surfaced_tools.extend(surfaced)
        self.planning_transport_failure_streak = transport_streak  # earlier worker-transport failures, if any
        obs.ctrl = self

    monkeypatch.setattr(LoopController, "__init__", _init)

    real_buffer = sl.get_abstraction_buffer()

    class _Buffer:
        def append(self, record: Any) -> None:
            if record.source == "agent_loop" and record.event in EVENTS:
                obs.events.append((record.event, dict(record.data)))
            real_buffer.append(record)

        def __getattr__(self, name: str) -> Any:
            return getattr(real_buffer, name)

    monkeypatch.setattr(sl, "get_abstraction_buffer", lambda: _Buffer())

    class _Probe(Tool):
        name = PROBE
        description = "stub"
        input_schema: dict = {}

        def execute(self, **kwargs: Any) -> Any:
            obs.executed.append(dict(kwargs))
            return ToolOutput(success=True, output="ok")

    registry = ToolRegistry()
    registry.register(_Probe())
    executor = build_executor(registry, pain_bus=None, policy=TEST_POLICY)
    index = _ToolIndex()
    if surfaced is not None:  # §2 reads only ``_tool_index`` off the bridge; dispatch then fails on the stand-in
        executor._tool_pain_bridge = SimpleNamespace(_tool_index=index)
    obs.index, obs.executor = index, executor

    state = RuntimeState()
    state.data["mode"] = "active"
    if goal:
        state.data["active_goal"] = goal
    if pending_input:  # an input the LLM is answering: §2 clears it on most outcomes
        state.data.update(
            {"pending_user_input": "hello", "pending_user_input_time": time.time(), "pending_user_input_source": "CLI"}
        )
    obs.state = state
    source = _Source()
    worker = _Worker(source, script)
    obs.worker = worker
    kwargs: dict[str, Any] = {}
    if sim:
        kwargs["percept_source"] = source
        if sink:
            obs.sink = RecordingSink()
            kwargs["action_sink"] = obs.sink
    else:

        class _Adapter(NullSimulationAdapter):
            def next_observation(self, environment: Any, default_network: Any | None = None) -> dict:
                source.next_percept()
                return super().next_observation(environment, default_network)

            def has_carried_percept(self) -> bool:
                return True  # wake every pass, as the sim source does

        kwargs["sim_adapter"] = _Adapter()
    (tmp_path / "ws").mkdir(exist_ok=True)
    try:
        AL.run_agentic_loop(
            _Agent(),
            FileSystemEnv(str(tmp_path / "ws")),
            state,
            build_memory(),
            build_decision_engine(),
            executor,
            llm_worker=worker,
            autonomy_controller=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS),
            max_steps=steps,
            target_hz=1e6,
            idle_sleep_s=0.0,
            stop_event=threading.Event(),
            planning_liveness=liveness,
            on_event=on_event or (lambda e: obs.stream.append((e.kind, dict(e.payload)))),
            **kwargs,
        )
    except Exception as exc:  # the liveness-exhaustion arms inspect it
        obs.error = exc
    return obs


def _sink(obs: Any) -> list[tuple[str, dict, bool, str | None]]:
    return [(a.tool_name, dict(a.tool_args), a.result_success, a.result_error) for a in obs.sink.actions]


def _pending_cleared(obs: Any) -> bool:
    return not any(k in obs.state.data for k in PENDING_KEYS)


UNAVAILABLE = ("_llm_unavailable", {}, False, "LLM unavailable after 1 consecutive failures")


def _retry(reason: str, attempt: int) -> tuple[str, dict[str, Any]]:
    return ("planning_retry", {"reason": reason, "attempt": attempt, "limit": 3, "requeued": True})


# ── a proposal that is installed ─────────────────────────────────────────────


def test_a_fresh_proposal_is_installed_and_its_next_actions_run_one_per_pass(monkeypatch, tmp_path):
    """Stamped (the D13 clock), streamed, logged, the pending input cleared, executed; its ``next_actions`` are
    queued and become ``multi_step`` continuations, one per pass, in order."""
    script = {
        0: _proposal(next_actions=[{"tool_name": PROBE, "params": {"n": 2}}, {"tool_name": PROBE, "params": {"n": 3}}])
    }
    obs = _run(monkeypatch, tmp_path, script, steps=5)
    assert obs.executed == [{"n": 1}, {"n": 2}, {"n": 3}]
    assert _sink(obs) == [(PROBE, {"n": k}, True, None) for k in (1, 2, 3)]
    assert obs.events == [
        (
            "llm_response",
            {"tool": PROBE, "confidence": 0.9, "reasoning": "because", "has_plan": False, "requires_approval": False},
        ),
        ("multi_step_queued", {"count": 2}),
    ]
    assert obs.stream == [("inference_end", {"has_proposal": True})]
    assert _pending_cleared(obs) and obs.ctrl.last_proposal_time > 0
    assert list(obs.ctrl.pending_next_actions) == [] and obs.ctrl.pending_proposal is None
    assert obs.worker.requeues == [] and obs.executor._tools_hallucinated == []


def test_the_continuations_are_multi_step_proposals(monkeypatch, tmp_path):
    """What §2 installs for a queued next action: ``multi_step_continuation`` at confidence 0.8 (read where §4
    logs each proposal it runs)."""
    monkeypatch.setattr(sys.modules[__name__], "EVENTS", EVENTS + ("goal_proposed",))
    script = {0: _proposal(next_actions=[{"tool_name": PROBE, "params": {"n": 2}}])}
    obs = _run(monkeypatch, tmp_path, script, steps=4)
    proposed = [d for e, d in obs.events if e == "goal_proposed"]
    assert [(d["reasoning"], d["confidence"]) for d in proposed] == [("because", 0.9), ("multi_step_continuation", 0.8)]


def test_an_installed_proposal_records_the_surfaced_tools_it_did_not_choose(monkeypatch, tmp_path):
    """With a learned tool index on the executor's pain bridge, the active goal and the last surfaced tools, the
    chosen tool is recorded as surfaced-but-unused against the rest; no goal, no record."""
    obs = _run(monkeypatch, tmp_path, {0: _proposal()}, steps=2, surfaced=["a", "b"], goal="g1")
    assert obs.index.calls == [("g1", ["a", "b"], PROBE)]
    obs = _run(monkeypatch, tmp_path / "nogoal", {0: _proposal()}, steps=2, surfaced=["a", "b"])
    assert obs.index.calls == []


def test_outside_sim_mode_a_parse_failure_fallback_is_executed(monkeypatch, tmp_path):
    """Only the sim adapter drops fallbacks: a non-sim loop installs and runs one like any proposal."""
    obs = _run(monkeypatch, tmp_path, {0: _proposal(reasoning="llm_fallback")}, steps=3, sim=False)
    assert obs.executed == [{"n": 1}]
    assert [e for e, _ in obs.events] == ["llm_response"]
    assert _pending_cleared(obs)


# ── proposals that are dropped (planning liveness off) ───────────────────────


def test_a_parse_failure_fallback_is_dropped_and_surfaces_llm_unavailable(monkeypatch, tmp_path):
    """Sim mode: nothing runs, the pending input is cleared, and each dropped fallback records ONE synthetic
    ``_llm_unavailable`` action (the stall threshold is 1, and the count resets after each record)."""
    script = {0: _proposal(reasoning="llm_fallback"), 2: _proposal(reasoning="llm_fallback")}
    obs = _run(monkeypatch, tmp_path, script, steps=4)
    assert obs.executed == [] and _sink(obs) == [UNAVAILABLE, UNAVAILABLE]
    assert obs.events == [] and obs.stream == [] and obs.worker.requeues == []
    assert _pending_cleared(obs) and obs.ctrl.last_proposal_time > 0
    assert obs.executor._tools_hallucinated == []  # a parse failure is not a bad tool name


def test_without_an_action_sink_a_dropped_fallback_records_nothing(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, {0: _proposal(reasoning="llm_fallback")}, steps=3, sink=False)
    assert obs.executed == [] and obs.events == [] and _pending_cleared(obs)


def test_an_unregistered_tool_is_dropped_and_its_name_recorded_as_hallucinated(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, {0: _proposal(UNKNOWN)}, steps=3)
    assert obs.executed == [] and _sink(obs) == [UNAVAILABLE]
    assert obs.executor._tools_hallucinated == [UNKNOWN]
    assert obs.events == [] and obs.stream == [] and _pending_cleared(obs)


def test_an_error_proposal_is_dropped_logged_and_clears_the_input(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, {0: _proposal(None, error="boom")}, steps=3)
    assert obs.executed == [] and _sink(obs) == []
    assert obs.events == [("error", {"context": "llm_proposal", "error": "boom"})]
    assert obs.stream == [("inference_end", {"has_proposal": True})]
    assert _pending_cleared(obs) and obs.worker.requeues == []


def test_without_liveness_a_proposal_with_no_action_is_ignored_and_keeps_the_input(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, {0: _proposal(None)}, steps=3)
    assert obs.executed == [] and obs.events == [] and obs.worker.requeues == []
    assert obs.stream == [("inference_end", {"has_proposal": True})]
    assert not _pending_cleared(obs)


def test_a_stale_proposal_is_dropped_and_keeps_the_input(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, {0: _proposal(age_s=100.0)}, steps=3)
    assert obs.executed == [] and obs.events == [] and obs.stream == [] and obs.worker.requeues == []
    assert not _pending_cleared(obs) and obs.ctrl.last_proposal_time > 0


# ── planning liveness (D13): every drop reschedules, a spent budget ends the run ─


def test_under_liveness_a_dropped_fallback_requeues_its_original_request(monkeypatch, tmp_path):
    script = {0: _proposal(reasoning="llm_fallback", original_request="REQ")}
    obs = _run(monkeypatch, tmp_path, script, steps=3, liveness=True)
    assert obs.worker.requeues == [("request", "REQ", {"failed_tool": None, "reason": "fallback_proposal_dropped"})]
    assert obs.events == [_retry("fallback_proposal_dropped", 1)]
    assert _sink(obs) == [UNAVAILABLE] and obs.error is None


def test_under_liveness_an_unregistered_tool_requeues_naming_the_tool(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, {0: _proposal(UNKNOWN)}, steps=3, liveness=True)
    assert obs.worker.requeues == [("last", None, {"failed_tool": UNKNOWN, "reason": "unregistered_tool_proposed"})]
    assert obs.events == [_retry("unregistered_tool_proposed", 1)]
    assert obs.executor._tools_hallucinated == [UNKNOWN]


def test_under_liveness_an_error_requeues_but_a_shutdown_does_not(monkeypatch, tmp_path):
    obs = _run(
        monkeypatch, tmp_path, {0: _proposal(None, error="boom", original_request="REQ")}, steps=3, liveness=True
    )
    assert obs.worker.requeues == [("request", "REQ", {"failed_tool": None, "reason": "proposal_error:boom"})]
    assert obs.events == [("error", {"context": "llm_proposal", "error": "boom"}), _retry("proposal_error:boom", 1)]
    obs = _run(monkeypatch, tmp_path / "shutdown", {0: _proposal(None, error="shutdown")}, steps=3, liveness=True)
    assert obs.worker.requeues == []
    assert obs.events == [("error", {"context": "llm_proposal", "error": "shutdown"})]


def test_under_liveness_a_proposal_with_no_action_requeues_with_its_reason(monkeypatch, tmp_path):
    script = {0: _proposal(None), 1: _proposal(None, ready_to_act=False), 2: _proposal(None, mode_goal_achieved=True)}
    obs = _run(monkeypatch, tmp_path, script, steps=4, liveness=True)
    reasons = ["proposal_without_action", "proposal_not_ready_to_act", "proposal_completed_without_action"]
    assert obs.worker.requeues == [("last", None, {"failed_tool": None, "reason": r}) for r in reasons]
    assert obs.events == [_retry(r, k) for k, r in enumerate(reasons, 1)]
    assert _pending_cleared(obs) and obs.error is None


def test_under_liveness_a_stale_proposal_requeues(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, {0: _proposal(age_s=100.0)}, steps=3, liveness=True)
    assert obs.worker.requeues == [("last", None, {"failed_tool": None, "reason": "stale_proposal_dropped"})]
    assert obs.events == [_retry("stale_proposal_dropped", 1)]
    assert not _pending_cleared(obs)


_EXHAUST = {  # drop kind -> (proposal kwargs, requeue reason, finish status, ``_llm_unavailable`` records)
    "fallback": ({"reasoning": "llm_fallback"}, "fallback_proposal_dropped", "llm_wedged", 4),
    "unregistered": ({"tool": UNKNOWN}, "unregistered_tool_proposed", "planning_failed", 4),
    "error": ({"tool": None, "error": "boom"}, "proposal_error:boom", "llm_wedged", 0),
    "no_action": ({"tool": None}, "proposal_without_action", "planning_failed", 0),
    "stale": ({"age_s": 100.0}, "stale_proposal_dropped", "planning_failed", 0),
}


@pytest.mark.parametrize("kind", sorted(_EXHAUST))
def test_a_spent_retry_budget_ends_the_run_in_that_pass_with_its_status(monkeypatch, tmp_path, kind):
    """Three retries, then the fourth drop exhausts the budget: the loop leaves in that same pass (no later pass
    consumes anything) and ``run_agentic_loop`` raises ``PlanningLivenessExhausted`` with the drop's status."""
    from maxim.runtime.loop_controller import PlanningLivenessExhausted

    kw, reason, status, unavailable = _EXHAUST[kind]
    script = {k: _proposal(**kw) for k in range(8)}
    obs = _run(monkeypatch, tmp_path, script, steps=10, liveness=True)
    assert obs.worker.consumed == [0, 1, 2, 3] and obs.worker.source.passes == 3
    assert len(obs.worker.requeues) == 3 and _sink(obs) == [UNAVAILABLE] * unavailable
    retries = [e for e in obs.events if e[0] in ("planning_retry", "planning_liveness_exhausted")]
    assert retries[:3] == [_retry(reason, k) for k in (1, 2, 3)]
    assert retries[3] == (
        "planning_liveness_exhausted",
        {"planning_streak": 4, "transport_streak": 0, "reason": reason, "status": status},
    )
    assert len(retries) == 4
    assert isinstance(obs.error, PlanningLivenessExhausted) and obs.error.finish_status == status
    assert obs.executor._tools_hallucinated == ([UNKNOWN] if kind == "unregistered" else [])  # recorded once


# ── the two reported failures inside §2 do not stop the intake ───────────────


def test_a_failing_hallucinated_tool_record_is_reported_and_the_drop_still_happens(monkeypatch, tmp_path, caplog):
    class _Raising(list):
        def append(self, item: Any) -> None:
            raise RuntimeError("record refused")

    from maxim.runtime.executor import Executor

    real_init = Executor.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        self._tools_hallucinated = _Raising()

    monkeypatch.setattr(Executor, "__init__", _init)
    caplog.set_level(logging.DEBUG, logger="maxim")  # the explicit swallow form reports at DEBUG
    obs = _run(monkeypatch, tmp_path, {0: _proposal(UNKNOWN)}, steps=3)
    assert obs.executed == [] and _sink(obs) == [UNAVAILABLE] and _pending_cleared(obs)
    assert any("record_hallucinated_tool" in r.getMessage() for r in caplog.records)


def test_a_raising_on_event_is_reported_and_the_proposal_still_runs(monkeypatch, tmp_path, caplog):
    def _boom(_e: Any) -> None:
        raise RuntimeError("stream down")

    caplog.set_level(logging.DEBUG, logger="maxim")
    obs = _run(monkeypatch, tmp_path, {0: _proposal()}, steps=3, on_event=_boom)
    assert obs.executed == [{"n": 1}]
    assert any("on_event:inference_end" in r.getMessage() for r in caplog.records)


# ── what §2 carries from pass to pass, and resets ────────────────────────────


def test_the_fallback_count_is_carried_across_passes_and_reset_by_a_real_proposal(monkeypatch, tmp_path, caplog):
    """Without an action sink nothing resets the count after a drop, so it climbs across passes (1, 2), a real
    proposal resets it, and the next drop counts from 1 again. Logged on the agent loop's own logger."""
    caplog.set_level(logging.INFO, logger="maxim.runtime.agent_loop")
    fallback = {"reasoning": "llm_fallback"}
    script = {0: _proposal(**fallback), 2: _proposal(**fallback), 3: _proposal(), 4: _proposal(**fallback)}
    obs = _run(monkeypatch, tmp_path, script, steps=6, sink=False)
    skips = [r for r in caplog.records if r.getMessage().startswith("Sim mode: skipping fallback proposal")]
    assert [r.getMessage() for r in skips] == [
        f"Sim mode: skipping fallback proposal ({n} consecutive)" for n in (1, 2, 1)
    ]
    assert {r.name for r in skips} == {"maxim.runtime.agent_loop"}
    assert obs.executed == [{"n": 1}]


def test_any_proposal_resets_the_worker_transport_streak(monkeypatch, tmp_path):
    """A proposal proves the worker answered, so §2 resets the transport-failure streak even for a proposal it
    then drops; with nothing returned, the streak stands."""
    obs = _run(monkeypatch, tmp_path, {}, steps=2, transport_streak=2)
    assert obs.ctrl.planning_transport_failure_streak == 2
    obs = _run(monkeypatch, tmp_path / "dropped", {0: _proposal(age_s=100.0)}, steps=2, transport_streak=2)
    assert obs.ctrl.planning_transport_failure_streak == 0
