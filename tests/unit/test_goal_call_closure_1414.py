"""#1414's second cause: the agent-fallback path (§3) never closes the goal agent's tool call.

``AgenticGoalAgent.propose_intent`` returns its pending tool call until a ``ToolResult`` with that call's id arrives
on the bus. Nothing in the loop published one: ``MaximAgent.publish_tool_result`` had no caller. So a call autonomy
refused was proposed and refused again on every pass, and a call that ran was proposed again too. #1427 fixed
#1414's other cause (one input re-captured every tick); this half was found still live on main 043eab8f by a
real-loop probe on 2026-11-21 (six passes: 6/6 proposals on main, 1 with the fix) and is ported here from
``fix/1414-consumed-input``.
"""

from __future__ import annotations

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
    """The surface the loop reaches on ``MaximAgent``: ``publish_tool_result`` onto the goal agent's bus, and
    ``refuse_tool_call`` (``test_maxim_agent_exposes_the_surface_the_loop_reaches`` pins that the real class has both)."""
    from types import SimpleNamespace

    from maxim.agents.bus import ToolResult

    def publish_tool_result(tool_call_id, tool_name, success, result=None, error=None, params=None):  # noqa: ANN001
        goal_agent._bus.publish(
            ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                success=success,
                result=result,
                error=error,
                params=params or {},
            )
        )

    return SimpleNamespace(
        goal=goal_agent, publish_tool_result=publish_tool_result, refuse_tool_call=goal_agent.refuse_tool_call
    )


def _results_on(goal_agent: Any) -> list[Any]:
    from maxim.agents.bus import ToolResult

    seen: list[Any] = []
    goal_agent._bus.subscribe(ToolResult, seen.append)
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

    from maxim.runtime import agent_loop

    goal_agent = _goal_agent_with_pending_call("read_file")
    intent = goal_agent.propose_intent(state=None, memory=None)
    output = SimpleNamespace(success=True, output="contents", error=None)
    agent_loop._report_fallback_result(_maxim_like(goal_agent), intent, {"tool_name": "read_file"}, output)
    assert goal_agent.propose_intent(state=None, memory=None) is None


def test_a_refusal_is_not_published_as_a_tool_result() -> None:
    """The tool never ran: the bus's result subscribers (memory, statistician, the API's tool_call callbacks) must
    not book a refusal as a failed call."""
    from maxim.runtime import agent_loop

    goal_agent = _goal_agent_with_pending_call()
    seen = _results_on(goal_agent)
    intent = goal_agent.propose_intent(state=None, memory=None)
    agent_loop._fallback_rejected(_maxim_like(goal_agent), intent, {"tool_name": "bash"}, "forbidden", 1.0)
    assert seen == [] and goal_agent.propose_intent(state=None, memory=None) is None


def test_a_raise_after_the_report_does_not_report_again() -> None:
    """One result per call id: §3's except reports only when the execute itself raised."""
    from types import SimpleNamespace

    from maxim.runtime import agent_loop

    goal_agent = _goal_agent_with_pending_call("read_file")
    seen = _results_on(goal_agent)
    intent = goal_agent.propose_intent(state=None, memory=None)
    agent = _maxim_like(goal_agent)
    output = SimpleNamespace(success=True, output="contents", error=None)
    agent_loop._report_fallback_result(agent, intent, {"tool_name": "read_file"}, output)
    agent_loop._report_fallback_result(agent, intent, {"tool_name": "read_file"}, output, error=RuntimeError("x"))
    assert [r.success for r in seen] == [True]
    agent_loop._report_fallback_result(agent, {"tool_call_id": "tc-other"}, {"tool_name": "t"}, error=ValueError("e"))
    assert [r.success for r in seen] == [True, False]  # the execute itself raised: reported failed


def test_an_intent_without_a_call_id_or_an_agent_without_the_hooks_closes_nothing() -> None:
    """Not every §3 intent comes from the goal agent: one with no ``tool_call_id``, or an agent that exposes neither
    hook, is left alone rather than raising."""
    from types import SimpleNamespace

    from maxim.runtime import agent_loop

    goal_agent = _goal_agent_with_pending_call()
    seen = _results_on(goal_agent)
    agent = _maxim_like(goal_agent)
    agent_loop._report_fallback_result(agent, {}, {"tool_name": "bash"}, SimpleNamespace(success=True))
    agent_loop._fallback_rejected(agent, {}, {"tool_name": "bash"}, "forbidden", 1.0)
    bare = SimpleNamespace()
    agent_loop._report_fallback_result(bare, {"tool_call_id": "tc-1"}, {"tool_name": "bash"}, SimpleNamespace())
    agent_loop._fallback_rejected(bare, {"tool_call_id": "tc-1"}, {"tool_name": "bash"}, "forbidden", 1.0)
    assert seen == [] and goal_agent.propose_intent(state=None, memory=None) is not None


def test_maxim_agent_exposes_the_surface_the_loop_reaches() -> None:
    """The real-loop test drives a stand-in for ``MaximAgent``; this pins that the real class has the two hooks the
    helpers look up by name, and that ``refuse_tool_call`` delegates to its goal agent."""
    from types import SimpleNamespace

    from maxim.agents.maxim_agent import MaximAgent

    assert callable(getattr(MaximAgent, "publish_tool_result", None))
    goal_agent = _goal_agent_with_pending_call()
    intent = goal_agent.propose_intent(state=None, memory=None)
    assert MaximAgent.refuse_tool_call(SimpleNamespace(goal=goal_agent), intent["tool_call_id"], "forbidden") is True
    assert goal_agent.propose_intent(state=None, memory=None) is None


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


def test_every_strategy_but_skip_ends_a_refused_goal() -> None:
    """The parametrisation below covers every member: a strategy added later fails here until it is decided."""
    from maxim.agents.bus import FailureStrategy

    assert {s.name for s in FailureStrategy} == {"SKIP", *_NON_SKIP}


_NON_SKIP = ("RETRY", "ABORT_PARENT", "ESCALATE", "REPLAN")


@pytest.mark.parametrize("strategy", _NON_SKIP)
def test_a_refused_non_skip_sub_goal_ends_the_goal_without_a_retry(strategy: str) -> None:
    """A refused call would be refused again, so RETRY, ESCALATE and REPLAN cannot help: only SKIP goes on."""
    from maxim.agents.bus import FailureStrategy

    goal_agent = _goal_with_sub_goals(getattr(FailureStrategy, strategy))
    first = goal_agent.propose_intent(state=None, memory=None)
    assert goal_agent.refuse_tool_call(first["tool_call_id"], "forbidden") is True
    assert goal_agent.propose_intent(state=None, memory=None) is None
    assert goal_agent.refuse_tool_call(first["tool_call_id"], "forbidden") is False  # already closed


@pytest.mark.parametrize("branch", ["refused", "executed", "raised"])
def test_the_real_loop_proposes_a_goal_call_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, branch: str) -> None:
    """Through the REAL run_agentic_loop (the shipped path, not a hand-composed sequence): a goal call is proposed
    once over six full passes, whichever §3 branch closes it: autonomy refuses it, §3 runs it, or the execute raises.
    The harness's executor has an empty registry, so the "executed" call runs and fails (an unregistered tool): what
    matters is that it is closed. Red on main 043eab8f: 6 of 6 passes proposed it in the refused and executed cases.
    Each of §3's three closing calls was deleted in turn on 2026-11-21, and its case failed."""
    from maxim.runtime.executor import Executor
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SafetyConstraints

    from tests.unit import test_loop_gates_characterization as gates

    goal_agent = _goal_agent_with_pending_call()
    seen = _results_on(goal_agent)
    proposed: list[Any] = []
    face = _maxim_like(goal_agent)

    class _GoalFace:
        name = "goal-face"
        publish_tool_result = staticmethod(face.publish_tool_result)
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
    if branch == "refused":
        assert seen == []  # closed by the REFUSAL, not by a ToolResult: §3's refused branch ran
    elif branch == "raised":
        assert [(r.success, r.error) for r in seen] == [(False, "the executor broke")]  # §3's except reported it
    else:
        assert [r.tool_name for r in seen] == ["bash"]  # closed by the one ToolResult §3 published
