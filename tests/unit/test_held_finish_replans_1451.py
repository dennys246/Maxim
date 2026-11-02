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
    # The re-plan tells the narrator why it is asked again.
    assert "finish_simulation was set aside" in RUN_SIDE["held_finish"]["prompts"]["narrator"][-1]
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


# ── the seam, unit by unit (loop_controller.set_aside_run_end) ────────────────────────────────────────────────────


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
    assert "finish_simulation was set aside: 1 input(s)" in ctrl.recent_outcomes[-1]["result"]
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
        "finish_simulation runs although 1 input(s) arrived while it was being re-planned; unread at finish: "
        "['USER GUIDANCE: wait'] (#1451)"
    ]


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


def test_a_memory_that_cannot_list_its_inputs_is_warned_once(caplog: pytest.LogCaptureFixture) -> None:
    from types import SimpleNamespace

    from maxim.runtime import loop_controller as lc

    class _Opaque:
        pass

    ctrl, _memory, executor, sim = _seam()
    lc._NO_RECENT_INPUTS_WARNED.discard("test_a_memory_that_cannot_list_its_inputs_is_warned_once.<locals>._Opaque")
    with caplog.at_level(logging.WARNING, logger="maxim.runtime.agent_loop"):
        lc.set_aside_run_end(ctrl, _Opaque(), executor, sim, True)
        lc.set_aside_run_end(ctrl, _Opaque(), executor, SimpleNamespace(log=lambda *a: None), True)
    assert len([r for r in caplog.records if "has no recent_cli_inputs()" in r.getMessage()]) == 1
    assert ctrl.pending_proposal is not None  # the re-plan is off, the finish runs


def test_the_agent_memory_lists_the_inputs_it_heard() -> None:
    from maxim.agents.memory_agent import MemoryAgent

    agent = MemoryAgent.__new__(MemoryAgent)
    import collections
    import threading

    agent._lock, agent._cli_inputs = threading.Lock(), collections.deque(["a", "b"], maxlen=20)
    assert agent.recent_cli_inputs() == ["a", "b"]
