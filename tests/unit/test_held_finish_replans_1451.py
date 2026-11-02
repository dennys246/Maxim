"""#1451: an input that reaches the narrator while its finish is being planned is read before the run ends.

The narrator plans one request at a time (#1048): an input that arrives while a planning job is in flight waits for
the next submit. When the answer it waited behind is ``finish_simulation``, the finish ends the run (#1259) and the
narrator never reads that input. Owner decisions (2026-10-30, 2026-11-01): re-plan once. A finish that inputs were
held behind is set aside, the narrator plans again with them in view, and that answer runs. If inputs are held
behind the re-plan's finish too, "once" holds: that finish runs and the inputs it was not shown get one WARNING.

Driven through the real ``start_simulation_mode`` by the orchestrator harness's probe arms (``PROBE_ARMS`` in
``tests/unit/_orch_harness.py``): the operator line is fed through the real stdin reader (free text while running is
the narrator's ``USER GUIDANCE``) while the narrator's planning call is held open.
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


@pytest.mark.xfail(
    strict=True,
    reason="#1451: the held line's finish runs and ends the run; the line is then submitted after the stop, its answer"
    " never runs",
)
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


@pytest.mark.xfail(strict=True, reason="#1451: no re-plan, so there is no second finish to hold a line behind")
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
