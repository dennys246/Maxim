"""#1541: the run's ``sim.stdin`` reader stops when the run ends, however it ends.

Before the fix the line reader (``orchestrator.py``'s ``_stdin_reader_line``, the non-TTY reader and the TTY reader's
fallback) blocked in ``input()`` and exited only on ``/cancel``, EOF or an interrupt. A run that ended any other way
(the narrator's ``finish_simulation`` since #1259, a campaign finish, ``max_turns``, the D12 hard abort, the grace)
returned with its reader alive, and that reader took the next stdin line meant for a later run in the same process.
Both tests drive the REAL ``start_simulation_mode`` offline (``tests/unit/_orch_harness.py``):

  - the narrator arm ends on the narrator's finish, with no operator line after it: no reader is left;
  - the teardown joins the reader after the run's stop and before the post-sim prompt reads stdin. That join is an
    ordering guarantee no run can observe (the reader's poll ends it within ``POLL_S`` anyway, so a deleted join
    leaves every run green): in an interactive session the raw reader's ``finally`` restores the terminal, and it
    must do so before ``read_post_sim_goal_raw`` puts the terminal into cbreak mode on the same fd. So it is pinned
    by line order in the function.

The cross-run defect on ONE shared stdin (the first run's stopped reader must leave the next line to the second, as
an in-process caller's runs share the process's stdin) is guarded by
``test_stdin_lines.py::test_a_stopped_runs_reader_leaves_the_next_line_to_the_next_run``. The red gate's second test
(the new_goal arm, then the stdin arm, in one process: how the leak showed in the golden's regeneration) was removed
after it flipped: with the harness's stdin now one pipe per arm, whose end of input reaches every reader on it, a
deletion probe of the fix left it green, so it no longer guarded #1541.

Red gate: commit 87cecca6 (xfail strict).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.unit._orch_harness import run_arm  # noqa: E402
from tests.unit._orchestrator_source import start_simulation_mode_def  # noqa: E402


@pytest.mark.timeout(240)
def test_a_finished_run_leaves_no_stdin_reader_1541(tmp_path: Path) -> None:
    trace = run_arm("narrator", tmp_path)
    assert "sim.stdin" not in trace["threads_left_alive"]["added"], trace["threads_left_alive"]


def test_teardown_joins_the_reader_after_the_stop_and_before_the_post_sim_prompt() -> None:
    fn = start_simulation_mode_def()
    blocks = [getattr(n, f) for n in ast.walk(fn) for f in ("body", "orelse", "finalbody") if hasattr(n, f)]
    ((block, at),) = [
        (b, i)
        for b in blocks
        if isinstance(b, list)
        for i, stmt in enumerate(b)
        if isinstance(stmt, ast.Expr) and ast.unparse(stmt.value) == "stop_reader(stdin_thread)"
    ]
    shutdown = block[at - 1]
    assert isinstance(shutdown, ast.Try), "the join directly follows the shutdown signals' try"
    assert "stop_event.set()" in [ast.unparse(s) for s in shutdown.body], "...which sets the run's stop"
    (prompt,) = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "_show_report_and_post_sim_prompt"
    ]
    assert block[at].lineno < prompt, "the reader is joined before the post-sim prompt reads stdin"
