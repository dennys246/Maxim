"""#1541: the run's ``sim.stdin`` reader stops when the run ends, however it ends.

Before the fix the line reader (``orchestrator.py``'s ``_stdin_reader_line``, the non-TTY reader and the TTY reader's
fallback) blocked in ``input()`` and exited only on ``/cancel``, EOF or an interrupt. A run that ended any other way
(the narrator's ``finish_simulation`` since #1259, a campaign finish, ``max_turns``, the D12 hard abort, the grace)
returned with its reader alive, and that reader took the next stdin line meant for a later run in the same process.
  - the narrator arm (the REAL ``start_simulation_mode`` offline, ``tests/unit/_orch_harness.py``) ends on the
    narrator's finish, with no operator line after it: no reader is left;
  - the run's end joins the reader before it returns, so before the post-sim prompt reads stdin. That join is an
    ordering guarantee no run can observe (the reader's poll ends it within ``POLL_S`` anyway, so a deleted join
    leaves every run green): in an interactive session the raw reader's ``finally`` restores the terminal, and it
    must do so before ``read_post_sim_goal_raw`` puts the terminal into cbreak mode on the same fd. Since the tail
    slice (#1438 PR 1c) the join is ``sim_teardown.py::_wait_for_run``'s first wait, which both of ``end_run``'s
    paths take after ``_signal_shutdown`` sets the run's stop. So it is pinned in two halves: ``_wait_for_run``
    joins the reader first, within the stop's bound (driven, over stand-ins), and ``start_simulation_mode`` hands
    the reader to the tail and calls ``end_run`` before the post-sim prompt (by AST).

The cross-run defect on ONE shared stdin (the first run's stopped reader must leave the next line to the second, as
an in-process caller's runs share the process's stdin) is guarded by
``test_stdin_lines.py::test_a_stopped_runs_reader_leaves_the_next_line_to_the_next_run``. The red gate's second test
(the new_goal arm, then the stdin arm, in one process: how the leak showed in the golden's regeneration) was removed
after it flipped: with the harness's stdin now one pipe per arm, whose end of input reaches every reader on it, a
deletion probe of the fix left it green, so it no longer guarded #1541.

Red gate: commit 98229045 (xfail strict).
"""

from __future__ import annotations

import ast
import sys
from typing import Any
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


class _Joinable:
    """A thread stand-in that records each join, in order, into a shared log."""

    def __init__(self, name: str, log: list[tuple[str, float | None]]) -> None:
        self.name, self._log = name, log

    def join(self, timeout: float | None = None) -> None:
        self._log.append((self.name, timeout))

    def is_alive(self) -> bool:
        return False


@pytest.mark.parametrize("bounded", [False, True])
def test_the_run_end_joins_the_reader_first_within_the_stops_bound(
    monkeypatch: pytest.MonkeyPatch, bounded: bool
) -> None:
    from unittest.mock import MagicMock

    import maxim.simulation.sim_logger as sim_logger
    import maxim.simulation.sim_teardown as teardown
    from maxim.simulation.stdin_lines import STOP_JOIN_S
    from maxim.utils.deadline import Deadline

    monkeypatch.setattr(sim_logger, "display_status", lambda *a, **k: None)
    log: list[tuple[str, float | None]] = []
    fields: dict[str, Any] = {name: None for name in teardown.SimTail.__dataclass_fields__}
    fields.update(
        aut_thread=_Joinable("aut", log),
        stdin_thread=_Joinable("sim.stdin", log),
        dm_error=[],
        orch_instance=MagicMock(),
    )
    bound = Deadline.after(0.25) if bounded else None
    teardown._wait_for_run(teardown.SimTail(**fields), bound=bound)
    assert [name for name, _ in log] == ["sim.stdin", "aut"], "the reader is joined before any other wait"
    timeout = log[0][1]
    assert timeout is not None and (timeout <= 0.25 if bounded else timeout == STOP_JOIN_S), timeout


def test_the_run_hands_its_reader_to_the_tail_and_ends_before_the_post_sim_prompt() -> None:
    fn = start_simulation_mode_def()
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    (tail,) = [c for c in calls if ast.unparse(c.func) == "SimTail"]
    assert "stdin_thread=stdin_thread" in [ast.unparse(k) for k in tail.keywords], "the tail joins THIS run's reader"
    (end,) = [c.lineno for c in calls if ast.unparse(c.func) == "end_run"]
    (prompt,) = [c.lineno for c in calls if ast.unparse(c.func) == "_show_report_and_post_sim_prompt"]
    assert end < prompt, "the run ends (and joins its reader) before the post-sim prompt reads stdin"
