"""#1541: the run's ``sim.stdin`` reader stops when the run ends, however it ends.

Before the fix the line reader (``orchestrator.py``'s ``_stdin_reader_line``, the non-TTY reader and the TTY reader's
fallback) blocked in ``input()`` and exited only on ``/cancel``, EOF or an interrupt. A run that ended any other way
(the narrator's ``finish_simulation`` since #1259, a campaign finish, ``max_turns``, the D12 hard abort, the grace)
returned with its reader alive, and that reader took the next stdin line meant for a later run in the same process.
Both tests drive the REAL ``start_simulation_mode`` offline (``tests/unit/_orch_harness.py``):

  - the narrator arm ends on the narrator's finish, with no operator line after it: no reader is left;
  - the new_goal arm runs twice (``/new`` re-enters the run) and the stdin arm then runs in the same process: its
    operator ``/cancel`` after turn 2 reaches its own run, not a reader the earlier runs left behind.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.unit._orch_harness import run_arm  # noqa: E402

_RED = pytest.mark.xfail(strict=True, reason="#1541: the sim.stdin reader outlives a run that does not end on /cancel")


@_RED
@pytest.mark.timeout(240)
def test_a_finished_run_leaves_no_stdin_reader_1541(tmp_path: Path) -> None:
    trace = run_arm("narrator", tmp_path)
    assert "sim.stdin" not in trace["threads_left_alive"]["added"], trace["threads_left_alive"]


@_RED
@pytest.mark.timeout(480)
def test_a_later_runs_operator_lines_reach_that_run_1541(tmp_path: Path) -> None:
    (tmp_path / "new_goal").mkdir()
    (tmp_path / "stdin").mkdir()
    run_arm("new_goal", tmp_path / "new_goal")
    trace = run_arm("stdin", tmp_path / "stdin")
    assert {"after_turn": 2, "line": "/cancel"} in trace["stdin_lines"], trace["stdin_lines"]
    turns = [call.get("turn") for call in trace["narrator_calls"] if call["kind"] == "followup"]
    assert turns == [1, 2], f"a follow-up lost its turn (the send failed under the harness): {turns}"
