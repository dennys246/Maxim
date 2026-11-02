"""#1451: an input that reaches the narrator while its finish is being planned is read before the run ends.

The narrator plans one request at a time (#1048): an input that arrives while a planning job is in flight waits for
the next submit. When the answer it waited behind is ``finish_simulation``, the finish ends the run (#1259) and the
narrator never reads that input. Owner decisions (2026-10-30, 2026-11-01): re-plan once. A finish that inputs were
held behind is set aside, the narrator plans again with them in view, and that answer runs. If inputs are held
behind the re-plan's finish too, "once" holds: that finish runs and the inputs it was not shown get one WARNING.

Driven through the real ``start_simulation_mode`` by the orchestrator harness's probe arms (``PROBE_ARMS`` in
``tests/unit/_orch_harness.py``): the operator line is fed through the real stdin reader (free text while running is
the narrator's ``USER GUIDANCE``) while the narrator's planning call is held open. Red gate: d53e369c.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from tests.unit._orch_harness import GOAL, HELD_GUIDANCE, RUN_SIDE, assert_this_checkout, canonical_json, run_arm


class _Warnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _run(arm: str, workdir: Path) -> tuple[dict[str, Any], list[str]]:
    assert_this_checkout()  # a worktree without PYTHONPATH would otherwise test main's code
    handler = _Warnings()
    logging.getLogger("maxim").addHandler(handler)
    try:
        trace = run_arm(arm, workdir)
    finally:
        logging.getLogger("maxim").removeHandler(handler)
    return json.loads(canonical_json(trace)), handler.messages


def _calls(trace: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [
        (c["kind"], c.get("executed") or c["request"][:40], c["answer"]["tool_name"]) for c in trace["narrator_calls"]
    ]


@pytest.mark.timeout(240)
def test_a_line_held_behind_the_finish_is_planned_before_the_run_ends(tmp_path: Path) -> None:
    t, _warnings = _run("held_finish", tmp_path)
    guidance = f"user guidance: {HELD_GUIDANCE[0]}"
    assert (
        _calls(t)
        == [
            ("plan", f"simulation goal: {GOAL}"[:40], "send_message"),
            ("followup", "send_message", "finish_simulation"),  # planned while the line arrived: set aside
            ("plan", guidance[:40], "finish_simulation"),  # the re-plan, with the line in view
        ]
    )
    # The re-plan tells the narrator why it is asked again, within the 50 characters the prompt shows of an outcome.
    assert (
        "finish_simulation: failed (SYSTEM: 1 new input(s) to read before finishing"
        in (RUN_SIDE["held_finish"]["prompts"]["narrator"][-1])
    )
    # Only the re-plan's finish ran: it is the run's finish, and it ended the run (no operator stop).
    assert [f.get("reason") for f in t["bridge_finish_calls"]] == ["planned with the guidance in view"] * 2
    assert t["report"]["llm_finish_reason"] == "planned with the guidance in view"
    assert [s["line"] for s in t["stdin_lines"]] == [HELD_GUIDANCE[0]]
    assert t["result"]["finish_reason"] == "completed"


@pytest.mark.timeout(240)
def test_a_line_held_behind_the_replans_finish_is_warned_and_the_finish_runs(tmp_path: Path) -> None:
    t, warnings = _run("held_finish_twice", tmp_path)
    assert _calls(t) == [
        ("plan", f"simulation goal: {GOAL}"[:40], "send_message"),
        ("followup", "send_message", "finish_simulation"),
        ("plan", f"user guidance: {HELD_GUIDANCE[0]}"[:40], "finish_simulation"),  # "once": this finish runs
    ]
    assert t["report"]["llm_finish_reason"] == "planned with the guidance in view"
    assert [s["line"] for s in t["stdin_lines"]] == list(HELD_GUIDANCE)
    unread = [w for w in warnings if "unread at finish" in w]
    assert len(unread) == 1 and HELD_GUIDANCE[1] in unread[0]


@pytest.mark.timeout(240)
def test_every_line_held_behind_the_finish_is_shown_to_the_re_plan(tmp_path: Path) -> None:
    t, warnings = _run("held_finish_two_lines", tmp_path)
    assert _calls(t)[-1] == ("plan", f"user guidance: {HELD_GUIDANCE[0]}"[:40], "finish_simulation")
    replan = RUN_SIDE["held_finish_two_lines"]["prompts"]["narrator"][-1]
    assert "2 new input(s) to read before finishing" in replan
    assert f"=== New Instructions ===\nUSER GUIDANCE: {HELD_GUIDANCE[1]}" in replan  # the second rides as held
    assert t["report"]["llm_finish_reason"] == "planned with the guidance in view"
    assert not [w for w in warnings if "unread at finish" in w]  # it was shown both: nothing unread


# ── the seam, unit by unit (loop_controller.set_aside_run_end) ────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fresh_warn_once() -> Any:
    """The once-per-type WARNINGs remember the types they warned about, process-wide: each test starts clean."""
    from maxim.runtime import loop_controller as lc

    lc._REPLAN_OFF_WARNED.clear()
    yield
    lc._REPLAN_OFF_WARNED.clear()


class _Registry:
    def __init__(self, tools: dict[str, Any]) -> None:
        self._tools = tools

    def find(self, name: str) -> Any:
        return self._tools.get(name)


def _seam(tool: str = "finish_simulation", *, heard: list[str] | None = None, processed: list[str] | None = None):
    import collections
    from types import SimpleNamespace

    from maxim.memory.base import InMemoryMemory
    from maxim.runtime.loop_controller import LoopController
    from maxim.simulation.tools import FinishSimulationTool, SendMessageTool

    ctrl = LoopController.__new__(LoopController)  # only the fields the seam reads
    ctrl._pending_proposal = SimpleNamespace(action={"tool_name": tool, "params": {}})
    ctrl.processed_cli_inputs = collections.deque(processed or [], maxlen=20)
    ctrl.recent_outcomes, ctrl.max_recent_outcomes, ctrl.run_end_set_aside = [], 10, False
    memory = InMemoryMemory()
    for line in heard or []:
        memory.record_command(line)
    executor = SimpleNamespace(
        registry=_Registry({"finish_simulation": FinishSimulationTool, "send_message": SendMessageTool}),
        resolve_tool_name=lambda name: name,
    )
    sim = SimpleNamespace(logged=[], log=lambda kind, text: sim.logged.append((kind, text)))
    return ctrl, memory, executor, sim


def test_the_tools_declare_whether_they_end_the_run() -> None:
    from maxim.simulation.tools import FinishSimulationTool, SendMessageTool
    from maxim.tools.base import Tool

    assert Tool.ends_run is False and SendMessageTool.ends_run is False
    assert FinishSimulationTool.ends_run is True


def test_a_finish_with_an_unread_input_is_set_aside_once_with_a_reason() -> None:
    from maxim.runtime.loop_controller import set_aside_run_end

    ctrl, memory, executor, sim = _seam(heard=["goal", "USER GUIDANCE: wait"], processed=["goal"])
    set_aside_run_end(ctrl, memory, executor, sim, True)
    assert ctrl.pending_proposal is None and ctrl.run_end_set_aside is True
    assert ctrl.recent_outcomes[-1]["error"] == "set_aside_for_held_inputs"
    assert ctrl.recent_outcomes[-1]["result"][:50] == "SYSTEM: 1 new input(s) to read before finishing. Y"
    assert sim.logged and sim.logged[-1][1].startswith("SET ASIDE: finish_simulation")


def test_the_second_finish_runs_and_names_the_unread_inputs(caplog: pytest.LogCaptureFixture) -> None:
    from maxim.runtime.loop_controller import set_aside_run_end

    ctrl, memory, executor, sim = _seam(heard=["goal", "USER GUIDANCE: wait"], processed=["goal"])
    ctrl.run_end_set_aside = True
    proposal = ctrl.pending_proposal
    with caplog.at_level(logging.WARNING, logger="maxim.runtime.agent_loop"):
        set_aside_run_end(ctrl, memory, executor, sim, True)
    assert ctrl.pending_proposal is proposal  # "once": it runs
    assert [r.getMessage() for r in caplog.records if "unread at finish" in r.getMessage()] == [
        "finish_simulation runs with 1 input(s) it was not shown, held behind its re-plan; unread at finish: "
        "['USER GUIDANCE: wait'] (#1451)"
    ]
    assert ctrl.run_end_set_aside is False  # let through: re-armed


@pytest.mark.parametrize(
    ("tool", "liveness_on", "heard"),
    [
        ("finish_simulation", True, ["goal"]),  # nothing unread
        ("finish_simulation", True, ["goal", "[ACTION_FOLLOWUP type=x tool=y query='z']: r"]),  # not an input
        ("finish_simulation", False, ["goal", "USER GUIDANCE: wait"]),  # no planning liveness (the AUT)
        ("send_message", True, ["goal", "USER GUIDANCE: wait"]),  # does not end the run
    ],
)
def test_nothing_is_set_aside_otherwise(tool: str, liveness_on: bool, heard: list[str]) -> None:
    from maxim.runtime.loop_controller import set_aside_run_end

    ctrl, memory, executor, sim = _seam(tool, heard=heard, processed=["goal"])
    proposal = ctrl.pending_proposal
    set_aside_run_end(ctrl, memory, executor, sim, liveness_on)
    assert ctrl.pending_proposal is proposal and ctrl.recent_outcomes == [] and sim.logged == []


def test_an_answer_that_does_not_end_the_run_re_arms_the_rule() -> None:
    from maxim.runtime.loop_controller import set_aside_run_end

    ctrl, memory, executor, sim = _seam("send_message", heard=["goal"], processed=["goal"])
    ctrl.run_end_set_aside = True
    set_aside_run_end(ctrl, memory, executor, sim, True)
    assert ctrl.run_end_set_aside is False


def test_a_finish_the_tool_refuses_leaves_the_rule_armed_for_the_next_one() -> None:
    """A finish below the run's turn floor is refused and the run goes on (``--sim-run-full-turns``): the finish
    that was let through re-armed the rule, so the next finish with an input held behind it is set aside again."""
    from maxim.runtime.loop_controller import set_aside_run_end

    ctrl, memory, executor, sim = _seam(heard=["goal"], processed=["goal"])
    ctrl.run_end_set_aside = True  # a re-plan's finish, nothing held behind it
    set_aside_run_end(ctrl, memory, executor, sim, True)
    assert ctrl.run_end_set_aside is False and ctrl.pending_proposal is not None  # it runs (and is refused)
    memory.record_command("USER GUIDANCE: one more")
    set_aside_run_end(ctrl, memory, executor, sim, True)
    assert ctrl.pending_proposal is None and ctrl.run_end_set_aside is True


def test_the_re_plan_takes_every_held_input_once() -> None:
    from maxim.runtime.loop_controller import take_replan_inputs

    ctrl, memory, _executor, _sim = _seam(heard=["goal", "a", "b"], processed=["goal", "a"])
    assert take_replan_inputs(ctrl, memory, True) == []  # nothing was set aside
    ctrl.run_end_set_aside = True
    assert take_replan_inputs(ctrl, memory, False) == []  # no planning liveness
    assert take_replan_inputs(ctrl, memory, True) == ["b"]  # "a" is the request, already marked by the scan
    assert list(ctrl.processed_cli_inputs) == ["goal", "a", "b"] and take_replan_inputs(ctrl, memory, True) == []


def test_the_follow_up_fold_and_the_re_plan_share_one_unread_test() -> None:
    from types import SimpleNamespace

    from maxim.runtime.agent_loop import _take_deferred_inputs
    from maxim.runtime.loop_controller import _unread_inputs, is_unread_input

    lines = ["", "seen", "[ACTION_FOLLOWUP type=x]: r", "new"]
    assert [c for c in lines if is_unread_input(c, ["seen"])] == ["new"]
    ctrl, memory, _e, _s = _seam(heard=lines[1:], processed=["seen"])
    assert _unread_inputs(memory, ctrl.processed_cli_inputs) == ["new"]
    assert _take_deferred_inputs(SimpleNamespace(cli_inputs=list(lines)), ["seen"], True) == ["new"]


def test_the_tool_a_call_would_run_is_resolved_through_its_alias() -> None:
    from types import SimpleNamespace

    from maxim.runtime.tool_dispatch import tool_to_run
    from maxim.simulation.tools import FinishSimulationTool

    executor = SimpleNamespace(
        registry=_Registry({"finish_simulation": FinishSimulationTool}),
        resolve_tool_name=lambda name: {"finish": "finish_simulation"}.get(name, name),
    )
    assert tool_to_run(executor, "finish") is FinishSimulationTool
    assert tool_to_run(SimpleNamespace(), "finish_simulation") is None


def test_a_mock_tools_attribute_is_not_a_declaration() -> None:
    from unittest.mock import MagicMock

    from maxim.runtime.loop_controller import set_aside_run_end

    ctrl, memory, _executor, sim = _seam(heard=["goal", "USER GUIDANCE: wait"], processed=["goal"])
    proposal = ctrl.pending_proposal
    set_aside_run_end(ctrl, memory, MagicMock(), sim, True)
    assert ctrl.pending_proposal is proposal


def test_the_system_outcome_list_keeps_its_newest() -> None:
    from maxim.runtime.loop_controller import note_system_outcome

    outcomes: list[dict[str, Any]] = []
    for i in range(3):
        note_system_outcome(outcomes, 2, f"t{i}", f"SYSTEM: {i}", "consecutive_tool_cap")
    assert [o["tool_name"] for o in outcomes] == ["t1", "t2"]
    assert outcomes[-1] | {"timestamp": 0} == {
        "tool": "t2",
        "tool_name": "t2",
        "success": False,
        "result": "SYSTEM: 2",
        "error": "consecutive_tool_cap",
        "timestamp": 0,
    }


def test_a_tool_that_sets_the_runs_stop_declares_that_it_ends_the_run() -> None:
    """Nothing else ties ``ends_run`` to the behaviour: a narrator tool that sets the run's ``stop_event`` without
    declaring it would end the run with no re-plan. Every Tool in the narrator's tool module is checked."""
    import inspect

    from maxim.simulation import tools as sim_tools
    from maxim.tools.base import Tool

    for _name, cls in inspect.getmembers(sim_tools, inspect.isclass):
        if issubclass(cls, Tool) and cls is not Tool and cls.__module__ == sim_tools.__name__:
            sets_stop = ".set()" in inspect.getsource(cls) and "stop_event" in inspect.getsource(cls)
            assert cls.ends_run is sets_stop, cls.__name__


def test_a_memory_that_cannot_list_its_inputs_is_warned_once(caplog: pytest.LogCaptureFixture) -> None:
    from types import SimpleNamespace

    from maxim.runtime import loop_controller as lc

    class _Opaque:
        pass

    ctrl, _memory, executor, sim = _seam()
    with caplog.at_level(logging.WARNING, logger="maxim.runtime.agent_loop"):
        lc.set_aside_run_end(ctrl, _Opaque(), executor, sim, True)
        lc.set_aside_run_end(ctrl, _Opaque(), executor, SimpleNamespace(log=lambda *a: None), True)
    assert len([r for r in caplog.records if "has no recent_cli_inputs()" in r.getMessage()]) == 1
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="maxim.runtime.agent_loop"):
        lc.set_aside_run_end(ctrl, _memory, SimpleNamespace(), sim, True)  # an executor with no registry
    assert len([r for r in caplog.records if "has no registry.find()" in r.getMessage()]) == 1
    assert ctrl.pending_proposal is not None  # the re-plan is off, the finish runs


def test_the_agent_memory_lists_the_inputs_it_heard() -> None:
    from maxim.agents.memory_agent import MemoryAgent

    agent = MemoryAgent.__new__(MemoryAgent)
    import collections
    import threading

    agent._lock, agent._cli_inputs = threading.Lock(), collections.deque(["a", "b"], maxlen=20)
    assert agent.recent_cli_inputs() == ["a", "b"]


# ── the re-plan's prompt (agents/prompt_builder.py) ────────────────────────────────────────────────────────────────


def _plan_prompt(triggering_input: str, held: list[str], *, tools: bool = True) -> str:
    """The prompt a real worker builds for a plan request carrying held inputs (no follow-up line in context)."""
    import time

    from maxim.agents.autonomy import AutonomyLevel
    from maxim.agents.bus import StructuredContext
    from maxim.agents.llm_worker import LLMWorker
    from tests.unit.test_planning_liveness import NoneLLM, _make_mode_info, _wait_for_proposal

    llm = NoneLLM()
    worker = LLMWorker(llm=llm, stale_threshold_s=10.0)
    worker.start()
    try:
        ctx = StructuredContext(timestamp=time.time())
        ctx.cli_inputs = [triggering_input]
        assert worker.submit_context(
            context=ctx,
            mode=_make_mode_info(),
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            internet_access=False,
            internet_policy_summary="",
            use_tool_prompting=tools,
            available_tools={"send_message", "finish_simulation"} if tools else set(),
            deliberation_available=False,
            triggering_input=triggering_input,
            deferred_inputs=held,
        )
        _wait_for_proposal(worker)
        return llm.prompts[0] if llm.prompts else ""
    finally:
        worker.stop()


def test_a_plan_request_shows_its_held_inputs_inside_the_budget() -> None:
    prompt = _plan_prompt("USER GUIDANCE: ask about tools", ["USER GUIDANCE: and its memory"])
    assert "=== New Instructions ===\nUSER GUIDANCE: and its memory" in prompt


def test_a_canned_answer_never_swallows_held_inputs() -> None:
    """ "what time is it" has a canned reply that skips the model; with held inputs the request goes to the model."""
    prompt = _plan_prompt("what time is it", ["USER GUIDANCE: and its memory"])
    assert prompt.startswith("TOOL_PROMPT|") and "USER GUIDANCE: and its memory" in prompt


def test_a_request_with_no_tool_prompt_warns_about_the_held_inputs_it_drops(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="maxim.agents.prompt_builder"):
        _plan_prompt("tell me a story", ["USER GUIDANCE: and its memory"], tools=False)
    assert any("drops 1 held input(s) unshown" in r.getMessage() for r in caplog.records)


def test_held_inputs_the_budget_drops_are_warned_by_name(caplog: pytest.LogCaptureFixture) -> None:
    """CRITICAL is not a guarantee: a held paste larger than the budget is dropped whole, and it is already marked
    read, so its loss is a WARNING naming it."""
    paste = "USER GUIDANCE: " + "word " * 60_000
    with caplog.at_level(logging.WARNING, logger="maxim.agents.prompt_builder"):
        prompt = _plan_prompt("USER GUIDANCE: ask about tools", [paste])
    assert "held_inputs" in prompt  # the budget notice names the dropped section
    assert "word word word" not in prompt  # dropped whole, never cut
    assert any("the prompt budget dropped 1 held input(s) unshown" in r.getMessage() for r in caplog.records)
