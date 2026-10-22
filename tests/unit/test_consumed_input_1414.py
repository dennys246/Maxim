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


def _perception() -> tuple[Any, list[Any]]:
    from maxim.agents.bus import AgentBus, Percept
    from maxim.agents.perception_agent import PerceptionAgent

    bus = AgentBus()
    seen: list[Any] = []
    bus.subscribe(Percept, seen.append)
    return PerceptionAgent(bus), seen


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

    return SimpleNamespace(
        goal=goal_agent, publish_tool_result=publish_tool_result, refuse_tool_call=goal_agent.refuse_tool_call
    )


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


def test_the_loop_reports_every_fallback_outcome() -> None:
    """The composition: §3's executed, raised and refused branches each close the goal agent's call."""
    import inspect

    from maxim.runtime import agent_loop

    src = inspect.getsource(agent_loop.run_agentic_loop)
    assert src.count("_report_fallback_result(agent, intent, action") == 2  # executed, raised
    assert "_fallback_rejected(agent, intent, action, reason, confidence)" in src


def _results_on(goal_agent: Any) -> list[Any]:
    from maxim.agents.bus import ToolResult

    seen: list[Any] = []
    goal_agent._bus.subscribe(ToolResult, seen.append)
    return seen


def test_a_refusal_is_not_published_as_a_tool_result() -> None:
    """The tool never ran: the bus's result subscribers (memory, statistician, the API's tool_call callbacks) must
    not book a refusal as a failed call (review fold)."""
    from maxim.runtime import agent_loop

    goal_agent = _goal_agent_with_pending_call()
    seen = _results_on(goal_agent)
    intent = goal_agent.propose_intent(state=None, memory=None)
    agent_loop._fallback_rejected(_maxim_like(goal_agent), intent, {"tool_name": "bash"}, "forbidden", 1.0)
    assert seen == [] and goal_agent.propose_intent(state=None, memory=None) is None


def test_a_raise_after_the_report_does_not_report_again() -> None:
    """One result per call id: §3's except reports only when the execute itself raised (review fold)."""
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


def test_a_fresh_state_s_first_message_is_not_dropped() -> None:
    from maxim.runtime.state import RuntimeState

    import gc

    agent, seen = _perception()
    for _ in range(3):  # one agent, three states in a row, each with its first message (seq 1 every time)
        state = RuntimeState()
        state.update({"cli_input": "hello"})
        agent.propose_intent(state, memory=None)
        del state  # freed before the next is built, so CPython hands the next one the same id
        gc.collect()
    assert len([p for p in seen if getattr(p, "has_maxim_keyword", False)]) == 3


def test_the_real_loop_proposes_a_refused_goal_call_once(monkeypatch, tmp_path) -> None:
    """Through the REAL run_agentic_loop (the shipped path, not a hand-composed sequence): a goal call autonomy
    forbids is proposed and refused once over six full passes, not once per pass."""
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
    autonomy = AutonomyController(
        initial_level=AutonomyLevel.AUTONOMOUS,
        safety_constraints=SafetyConstraints(forbidden_tools=frozenset({"bash"})),
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
    assert seen == []  # closed by the REFUSAL (an executed call would publish a ToolResult): §3's refused branch ran
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
    """A refusal follows the sub-goal's policy with retries exhausted: SKIP skips it, the goal goes on (review)."""
    from maxim.agents.bus import FailureStrategy

    goal_agent = _goal_with_sub_goals(FailureStrategy.SKIP)
    first = goal_agent.propose_intent(state=None, memory=None)
    assert first["goal"]["tool_name"] == "bash"
    assert goal_agent.refuse_tool_call(first["tool_call_id"], "forbidden") is True
    after = goal_agent.propose_intent(state=None, memory=None)
    assert after is not None and after["goal"]["tool_name"] == "respond"


def test_a_refused_retry_sub_goal_ends_the_goal_without_a_retry() -> None:
    from maxim.agents.bus import FailureStrategy

    goal_agent = _goal_with_sub_goals(FailureStrategy.RETRY)
    first = goal_agent.propose_intent(state=None, memory=None)
    assert goal_agent.refuse_tool_call(first["tool_call_id"], "forbidden") is True
    assert goal_agent.propose_intent(state=None, memory=None) is None
    assert goal_agent.refuse_tool_call(first["tool_call_id"], "forbidden") is False  # already closed
