"""#1414's second cause: the agent-fallback path (§3) never closes the goal agent's tool call.

``AgenticGoalAgent.propose_intent`` returns its pending tool call until it learns that call's fate, and nothing in the
loop told it. So a call autonomy refused was proposed and refused again on every pass, and a call that ran was
proposed again too. #1427 fixed #1414's other cause (one input re-captured every tick); this half was found still live
on main 043eab8f by a real-loop probe on 2026-11-21 (six passes: 6/6 proposals on main, 1 with the fix) and is ported
from ``fix/1414-consumed-input``.

Owner decisions 2026-11-21 (review round): §3 closes a call that ran or raised DIRECTLY
(``AgenticGoalAgent.close_tool_call``), not with a ``ToolResult`` on the bus (#1727); a RETRY sub-goal's attempts are
counted once per launch; a refused goal ends with ``GoalCompleted(refused=True)``, which ExecAgent's contemplation
learning skips.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest


def _goal_agent_with_pending_call(tool: str = "bash") -> Any:
    from maxim.agents.agentic_goal_agent import AgenticGoalAgent
    from maxim.agents.bus import AgentBus, GoalPriority, ProposedGoal

    goal_agent = AgenticGoalAgent(AgentBus())
    goal_agent._bus.publish(
        ProposedGoal(id="g1", description="get the news", priority=GoalPriority.HIGH, tool_name=tool)
    )
    return goal_agent


def _maxim_like(goal_agent: Any) -> Any:
    """The surface the loop reaches on ``MaximAgent``: ``close_tool_call`` and ``refuse_tool_call``
    (``test_maxim_agent_exposes_the_surface_the_loop_reaches`` pins that the real class has both, delegating)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        goal=goal_agent, close_tool_call=goal_agent.close_tool_call, refuse_tool_call=goal_agent.refuse_tool_call
    )


def _on_bus(goal_agent: Any, event_type: Any) -> list[Any]:
    seen: list[Any] = []
    goal_agent._bus.subscribe(event_type, seen.append)
    return seen


def test_a_refused_goal_tool_is_not_proposed_again() -> None:
    from maxim.runtime import agent_loop

    goal_agent = _goal_agent_with_pending_call()
    intent = goal_agent.propose_intent(state=None, memory=None)
    assert intent is not None and intent["goal"]["tool_name"] == "bash"  # control: the call is pending
    action = {"tool_name": "bash", "params": {}}
    agent_loop._fallback_rejected(_maxim_like(goal_agent), intent, action, "Tool 'bash' is forbidden", 1.0)
    assert goal_agent.propose_intent(state=None, memory=None) is None


def test_an_executed_goal_tool_is_closed_too() -> None:
    from types import SimpleNamespace

    from maxim.agents.bus import GoalCompleted
    from maxim.runtime import agent_loop

    goal_agent = _goal_agent_with_pending_call("read_file")
    done = _on_bus(goal_agent, GoalCompleted)
    intent = goal_agent.propose_intent(state=None, memory=None)
    output = SimpleNamespace(success=True, output="contents", error=None)
    agent_loop._close_fallback_call(_maxim_like(goal_agent), intent, {"tool_name": "read_file"}, output)
    assert goal_agent.propose_intent(state=None, memory=None) is None
    assert [(c.success, c.result, c.refused) for c in done] == [(True, "contents", False)]


def test_section_3_publishes_no_tool_result() -> None:
    """Owner decision: §3 closes calls directly. A ToolResult from §3 alone would wake ToolResult's dormant subscribers
    (memory, statistician, the API's tool_call event) for a biased slice of calls and book one failure up to three
    times; the bus producer comes to every dispatcher at once (#1727)."""
    from types import SimpleNamespace

    from maxim.agents.bus import ToolResult
    from maxim.runtime import agent_loop

    for kind in ("ran", "raised", "refused"):
        goal_agent = _goal_agent_with_pending_call()
        seen = _on_bus(goal_agent, ToolResult)
        intent = goal_agent.propose_intent(state=None, memory=None)
        agent = _maxim_like(goal_agent)
        if kind == "ran":
            agent_loop._close_fallback_call(agent, intent, {"tool_name": "bash"}, SimpleNamespace(success=False))
        elif kind == "raised":
            agent_loop._close_fallback_call(agent, intent, {"tool_name": "bash"}, error=RuntimeError("x"))
        else:
            agent_loop._fallback_rejected(agent, intent, {"tool_name": "bash"}, "forbidden", 1.0)
        assert seen == [] and goal_agent.propose_intent(state=None, memory=None) is None, kind


def test_a_raise_after_the_close_does_not_close_again() -> None:
    """One close per call id: §3 closes right after the execute, so its except closes only when the execute itself
    raised (a second close under a RETRY sub-goal's reused id would fail a call that succeeded)."""
    from types import SimpleNamespace

    from maxim.runtime import agent_loop

    calls: list[tuple] = []
    agent = SimpleNamespace(close_tool_call=lambda *a: calls.append(a))
    intent = {"tool_call_id": "tc-1"}
    output = SimpleNamespace(success=True, output="contents", error=None)
    agent_loop._close_fallback_call(agent, intent, {"tool_name": "read_file"}, output)
    agent_loop._close_fallback_call(agent, intent, {"tool_name": "read_file"}, output, error=RuntimeError("x"))
    assert calls == [("tc-1", True, "contents", None)]
    agent_loop._close_fallback_call(agent, {"tool_call_id": "tc-2"}, {"tool_name": "t"}, error=ValueError("e"))
    assert calls[-1] == ("tc-2", False, None, "e")  # the execute itself raised: closed as failed


def test_an_intent_without_a_call_id_closes_nothing_quietly(caplog: pytest.LogCaptureFixture) -> None:
    """Not every §3 intent comes from the goal agent: one with no ``tool_call_id`` has nothing to close."""
    from types import SimpleNamespace

    from maxim.runtime import agent_loop

    goal_agent = _goal_agent_with_pending_call()
    agent = _maxim_like(goal_agent)
    with caplog.at_level(logging.WARNING, logger="maxim.runtime.agent_loop"):
        agent_loop._close_fallback_call(agent, {}, {"tool_name": "bash"}, SimpleNamespace(success=True))
        agent_loop._fallback_rejected(agent, {}, {"tool_name": "bash"}, "forbidden", 1.0)
    assert goal_agent.propose_intent(state=None, memory=None) is not None
    assert not [r for r in caplog.records if "never closed" in r.getMessage()]


def test_a_goal_call_on_an_agent_without_the_hook_warns_once(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An intent WITH a call id on an agent that lacks the hook breaks the contract: the call is re-proposed on every
    pass. That is said once per agent type and hook, never silently."""
    from maxim.runtime import agent_loop

    class _NoHooks:
        pass

    monkeypatch.setattr(agent_loop, "_MISSING_CLOSE_HOOK_WARNED", set())
    with caplog.at_level(logging.WARNING, logger="maxim.runtime.agent_loop"):
        for _ in range(3):
            agent_loop._close_fallback_call(_NoHooks(), {"tool_call_id": "tc-1"}, {"tool_name": "bash"}, object())
            agent_loop._fallback_rejected(_NoHooks(), {"tool_call_id": "tc-1"}, {"tool_name": "bash"}, "no", 1.0)
    hooks = sorted(m for r in caplog.records if "never closed" in (m := r.getMessage()))
    assert len(hooks) == 2 and "close_tool_call" in hooks[0] and "refuse_tool_call" in hooks[1]


def test_maxim_agent_exposes_the_surface_the_loop_reaches() -> None:
    """The real-loop test drives a stand-in for ``MaximAgent``; this pins that the real class has the two hooks the
    helpers look up by name, each delegating to its goal agent with the arguments the loop passes."""
    from types import SimpleNamespace

    from maxim.agents.maxim_agent import MaximAgent

    goal_agent = _goal_agent_with_pending_call()
    intent = goal_agent.propose_intent(state=None, memory=None)
    assert MaximAgent.refuse_tool_call(SimpleNamespace(goal=goal_agent), intent["tool_call_id"], "forbidden") is True
    assert goal_agent.propose_intent(state=None, memory=None) is None

    goal_agent = _goal_agent_with_pending_call()
    intent = goal_agent.propose_intent(state=None, memory=None)
    me = SimpleNamespace(goal=goal_agent)
    assert MaximAgent.close_tool_call(me, intent["tool_call_id"], True, "out", None) is True
    assert goal_agent.propose_intent(state=None, memory=None) is None
    assert MaximAgent.close_tool_call(me, intent["tool_call_id"], True, "out", None) is False  # already closed


def _goal_with_sub_goals(on_failure: Any) -> Any:
    from maxim.agents.agentic_goal_agent import AgenticGoalAgent
    from maxim.agents.bus import AgentBus, GoalPriority, ProposedGoal, SubGoal

    goal_agent = AgenticGoalAgent(AgentBus())
    goal_agent._bus.publish(
        ProposedGoal(
            id="g2",
            description="look it up, then answer",
            priority=GoalPriority.HIGH,
            sub_goals=[
                SubGoal(id="s1", description="fetch", tool_name="bash", on_failure=on_failure),
                SubGoal(id="s2", description="answer", tool_name="respond", depends_on=[]),
            ],
        )
    )
    return goal_agent


def test_a_refused_skip_sub_goal_moves_on_to_the_next() -> None:
    """A refusal follows the sub-goal's policy with retries exhausted: SKIP skips it, and the goal goes on."""
    from unittest.mock import MagicMock

    from maxim.agents.bus import FailureStrategy

    goal_agent = _goal_with_sub_goals(FailureStrategy.SKIP)
    goal_agent._memory = MagicMock()
    first = goal_agent.propose_intent(state=None, memory=None)
    assert first["goal"]["tool_name"] == "bash"
    assert goal_agent.refuse_tool_call(first["tool_call_id"], "forbidden") is True
    after = goal_agent.propose_intent(state=None, memory=None)
    assert after is not None and after["goal"]["tool_name"] == "respond"
    goal_agent._memory.set_active_sub_goals.assert_called_with(["answer"])  # the skipped one is no longer active


_NON_SKIP = ("RETRY", "ABORT_PARENT", "ESCALATE", "REPLAN")


def test_every_strategy_but_skip_ends_a_refused_goal() -> None:
    """The parametrisation below covers every member: a strategy added later fails here until it is decided."""
    from maxim.agents.bus import FailureStrategy

    assert {s.name for s in FailureStrategy} == {"SKIP", *_NON_SKIP}


@pytest.mark.parametrize("strategy", _NON_SKIP)
def test_a_refused_non_skip_sub_goal_ends_the_goal_without_a_retry(strategy: str) -> None:
    """A refused call would be refused again, so RETRY, ESCALATE and REPLAN cannot help: only SKIP goes on. The goal
    ends as a failed GoalCompleted marked ``refused`` (REPLAN reaches the plan's own replanning that way)."""
    from maxim.agents.bus import FailureStrategy, GoalCompleted

    goal_agent = _goal_with_sub_goals(getattr(FailureStrategy, strategy))
    done = _on_bus(goal_agent, GoalCompleted)
    first = goal_agent.propose_intent(state=None, memory=None)
    assert goal_agent.refuse_tool_call(first["tool_call_id"], "forbidden") is True
    assert goal_agent.propose_intent(state=None, memory=None) is None
    assert goal_agent.refuse_tool_call(first["tool_call_id"], "forbidden") is False  # already closed
    assert [(c.success, c.refused) for c in done] == [(False, True)]


def test_a_retry_sub_goal_is_launched_max_retries_times() -> None:
    """Attempts count launches only (owner decision): with the default ``max_retries=2`` a failing RETRY sub-goal is
    launched twice, then the goal ends. Before, the failure counted too, so it was launched once and never retried."""
    from maxim.agents.bus import FailureStrategy, GoalCompleted

    goal_agent = _goal_with_sub_goals(FailureStrategy.RETRY)
    done = _on_bus(goal_agent, GoalCompleted)
    launches = 0
    while (intent := goal_agent.propose_intent(state=None, memory=None)) is not None:
        assert intent["goal"]["tool_name"] == "bash"
        launches += 1
        assert goal_agent.close_tool_call(intent["tool_call_id"], False, None, "boom") is True
        assert launches <= 5, "runaway retry"
    assert launches == 2
    assert [(c.success, c.refused) for c in done] == [(False, False)]


def test_exec_agent_contemplation_skips_a_refused_goal_but_staging_still_runs() -> None:
    """A refused goal says nothing about how it was drafted: ExecAgent's contemplation stats and its NAc link skip it,
    while acute staging (memory) still evaluates the failed goal. A goal that ran is booked as before."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from maxim.agents.bus import GoalCompleted
    from maxim.agents.exec_agent import ExecAgent

    def _me() -> Any:
        return SimpleNamespace(
            _contemplation_log={"g": {"contemplated": True, "refined": False, "timestamp": 0.0}},
            _contemplation_stats=dict.fromkeys(
                ("contemplated_total", "contemplated_success", "uncontemplated_total", "uncontemplated_success"), 0
            ),
            _nac=MagicMock(),
            log=logging.getLogger("test"),
            _memory=MagicMock(),
            _evaluate_staging=MagicMock(),
        )

    refused = _me()
    ExecAgent._on_goal_completed(refused, GoalCompleted(goal_id="g", success=False, error="Refused: x", refused=True))
    assert refused._nac.observe.call_count == 0 and refused._contemplation_stats["contemplated_total"] == 0
    assert refused._evaluate_staging.call_count == 1

    failed = _me()
    ExecAgent._on_goal_completed(failed, GoalCompleted(goal_id="g", success=False, error="boom"))
    assert failed._nac.observe.call_count == 1 and failed._contemplation_stats["contemplated_total"] == 1
    assert failed._evaluate_staging.call_count == 1


@pytest.mark.parametrize("branch", ["refused", "executed", "raised", "raised_after_execute"])
def test_the_real_loop_proposes_a_goal_call_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, branch: str) -> None:
    """Through the REAL run_agentic_loop (the shipped path, not a hand-composed sequence): a goal call is proposed
    once over six full passes, whichever §3 branch closes it: autonomy refuses it, §3 runs it, the execute raises, or
    a later step in §3's try raises after the execute (the review round's window: the close used to sit behind the
    audit log). The harness's executor has an empty registry, so the "executed" call runs and fails (an unregistered
    tool): what matters is that it is closed. Red on main 043eab8f: 6 of 6 passes proposed it in the refused and
    executed cases."""
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SafetyConstraints
    from maxim.agents.bus import GoalCompleted, ToolResult
    from maxim.runtime.executor import Executor

    from tests.unit import test_loop_gates_characterization as gates

    goal_agent = _goal_agent_with_pending_call()
    results = _on_bus(goal_agent, ToolResult)
    done = _on_bus(goal_agent, GoalCompleted)
    proposed: list[Any] = []
    face = _maxim_like(goal_agent)

    class _GoalFace:
        name = "goal-face"
        close_tool_call = staticmethod(face.close_tool_call)
        refuse_tool_call = staticmethod(face.refuse_tool_call)

        def propose_intent(self, state: Any, memory: Any) -> Any:
            intent = goal_agent.propose_intent(state, memory)
            if intent:
                proposed.append(intent)
            return intent

    monkeypatch.setattr(gates, "_Agent", _GoalFace)
    ev: list[tuple] = []
    adapter = gates._Adapter(ev)
    holder: list[Any] = []

    def wake(r: Any) -> None:  # a pending next action wakes the idle gate; cleared at §1 before anything acts on it
        holder.append(r)
        r.ctrl.pending_next_actions.append({"tool_name": "noop"})

    adapter.on_observe = lambda: holder and holder[-1].ctrl.pending_next_actions.clear()
    if branch == "raised":

        def _raise(self: Any, action: Any) -> Any:
            raise RuntimeError("the executor broke")

        monkeypatch.setattr(Executor, "execute", _raise)
    if branch == "raised_after_execute":

        def _audit_raises(self: Any, *a: Any, **k: Any) -> None:
            raise OSError("the audit log is full")

        monkeypatch.setattr(AutonomyController, "log_action", _audit_raises)
    autonomy = AutonomyController(
        initial_level=AutonomyLevel.AUTONOMOUS,
        safety_constraints=SafetyConstraints(
            forbidden_tools=frozenset({"bash"}) if branch == "refused" else frozenset()
        ),
    )
    run = gates._run(
        monkeypatch,
        tmp_path,
        steps=6,
        ev=ev,
        adapter=adapter,
        hooks=dict.fromkeys(range(6), wake),
        autonomy_controller=autonomy,
    )
    assert run.error is None
    assert len(gates._ran(run.ev)) == 6  # control: every pass reached §1, so §3 ran on each
    assert len(proposed) == 1
    assert goal_agent.propose_intent(state=None, memory=None) is None
    assert results == []  # closed directly, never by a ToolResult (owner decision, #1727)
    assert len(done) == 1 and done[0].success is False  # the empty registry fails "executed" too
    assert done[0].refused is (branch == "refused")
    if branch == "raised":
        assert done[0].error == "the executor broke"


@pytest.mark.parametrize("refused", [True, False], ids=["refused", "ran"])
def test_a_refused_goals_staging_never_borrows_another_events_surprise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, refused: bool
) -> None:
    """A refused goal made no contemplation observation, so NAc's ``last_rpe`` is some other event's surprise: staging
    scores the refused goal at 0.0, while a goal that ran keeps the real RPE (delta review, 2026-11-21)."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from maxim.agents import exec_agent
    from maxim.agents.bus import GoalCompleted

    seen: list[float] = []

    class _Stop(Exception):
        pass

    def _capture(**kw: Any) -> Any:
        seen.append(kw["rpe_raw"])
        raise _Stop

    monkeypatch.setattr(exec_agent, "CycleContext", _capture)
    me = SimpleNamespace(_weight_learner=MagicMock(), _staging_dir=str(tmp_path), _nac=SimpleNamespace(last_rpe=0.7))
    with pytest.raises(_Stop):
        exec_agent.ExecAgent._evaluate_staging(me, GoalCompleted(goal_id="g", success=False, refused=refused))
    assert seen == [0.0 if refused else 0.7]


def test_a_subscriber_contract_error_does_not_strand_the_goal() -> None:
    """A subscriber whose contract error re-raises out of ``publish`` (``EncodingContractError``) used to leave the goal
    set with no pending call: never completed, nothing proposed again. The goal is cleared regardless."""
    from maxim.agents.bus import GoalCompleted
    from maxim.memory.encoding import EncodingContractError

    goal_agent = _goal_agent_with_pending_call()

    def _breaks(_: Any) -> None:
        raise EncodingContractError("a broken contract")

    goal_agent._bus.subscribe(GoalCompleted, _breaks)
    intent = goal_agent.propose_intent(state=None, memory=None)
    with pytest.raises(EncodingContractError):
        goal_agent.close_tool_call(intent["tool_call_id"], True, "out", None)
    assert goal_agent._current_goal is None and goal_agent._pending_tool_call is None
