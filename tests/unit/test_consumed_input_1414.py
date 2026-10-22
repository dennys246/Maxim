"""#1414: one observation is re-captured every 0.5 s, and a refused goal tool is refused again every tick.

Found by maxim-pulse on the 1.3.3 candidate. After a rejected tool call the hippocampus "Captured: observation
(salience=0.90)" every 0.5 s for about 44 s, and the rejection was recorded about 87 times. Two causes:

1. ``RuntimeState.update`` merges every observation into ``data``, so the user's ``cli_input`` stays there after the
   tick that brought it, and ``PerceptionAgent.propose_intent`` reads ``data`` on every pass: one message was
   re-published as a salient Percept, and re-captured as new memory, every tick.
2. ``AgenticGoalAgent.propose_intent`` returns its pending tool call until a ``ToolResult`` with that call's id
   arrives, and nothing in the agent loop's fallback path (§3) published one: a call autonomy refused was proposed
   and refused again every tick.
"""

from __future__ import annotations

from typing import Any

import pytest

RED = pytest.mark.xfail(strict=True, reason="#1414 red gate: flips with the fix")


def _perception() -> tuple[Any, list[Any]]:
    from maxim.agents.bus import AgentBus, Percept
    from maxim.agents.perception_agent import PerceptionAgent

    bus = AgentBus()
    seen: list[Any] = []
    bus.subscribe(Percept, seen.append)
    return PerceptionAgent(bus), seen


@RED
def test_one_user_message_is_published_once_not_every_tick() -> None:
    from maxim.runtime.state import RuntimeState

    agent, seen = _perception()
    state = RuntimeState()
    state.update({"cli_input": "what's the news today?"})
    for _ in range(5):  # five loop passes, the sim adapter returning {} once its queue is drained
        state.update({})
        agent.propose_intent(state, memory=None)
    carrying = [p for p in seen if getattr(p, "cli_input", None) or getattr(p, "has_maxim_keyword", False)]
    assert len(carrying) == 1


@RED
def test_the_same_words_sent_again_are_a_new_message() -> None:
    from maxim.runtime.state import RuntimeState

    agent, seen = _perception()
    state = RuntimeState()
    for _ in range(2):
        state.update({"cli_input": "hi"})
        agent.propose_intent(state, memory=None)
        agent.propose_intent(state, memory=None)
    carrying = [p for p in seen if getattr(p, "has_maxim_keyword", False)]
    assert len(carrying) == 2


def _goal_agent_with_pending_call(tool: str = "bash") -> Any:
    from maxim.agents.agentic_goal_agent import AgenticGoalAgent
    from maxim.agents.bus import AgentBus, GoalPriority, ProposedGoal

    bus = AgentBus()
    goal_agent = AgenticGoalAgent(bus)
    bus.publish(ProposedGoal(id="g1", description="get the news", priority=GoalPriority.HIGH, tool_name=tool))
    return goal_agent


def _maxim_like(goal_agent: Any) -> Any:
    """The surface the loop reaches: ``agent.publish_tool_result`` (``MaximAgent``'s) onto the goal agent's bus."""
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

    return SimpleNamespace(goal=goal_agent, publish_tool_result=publish_tool_result)


@RED
def test_a_refused_goal_tool_is_not_proposed_again() -> None:
    from maxim.runtime import agent_loop

    goal_agent = _goal_agent_with_pending_call()
    intent = goal_agent.propose_intent(state=None, memory=None)
    assert intent is not None and intent["goal"]["tool_name"] == "bash"  # control: the call is pending
    action = {"tool_name": "bash", "params": {}}
    agent_loop._fallback_rejected(_maxim_like(goal_agent), intent, action, "Tool 'bash' is forbidden", 1.0)
    assert goal_agent.propose_intent(state=None, memory=None) is None


@RED
def test_an_executed_goal_tool_is_closed_too() -> None:
    from types import SimpleNamespace

    from maxim.runtime import agent_loop

    goal_agent = _goal_agent_with_pending_call("read_file")
    intent = goal_agent.propose_intent(state=None, memory=None)
    output = SimpleNamespace(success=True, output="contents", error=None)
    agent_loop._report_fallback_result(_maxim_like(goal_agent), intent, {"tool_name": "read_file"}, output)
    assert goal_agent.propose_intent(state=None, memory=None) is None


@RED
def test_the_loop_reports_every_fallback_outcome() -> None:
    """The composition: §3's executed, raised and refused branches each close the goal agent's call."""
    import inspect

    from maxim.runtime import agent_loop

    src = inspect.getsource(agent_loop.run_agentic_loop)
    assert src.count("_report_fallback_result(agent, intent, action") == 2  # executed, raised
    assert "_fallback_rejected(agent, intent, action, reason, confidence)" in src
