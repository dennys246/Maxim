"""#1226: a resumed run can restore the agent's memory while the NARRATOR starts fresh.

On ``--resume-sim`` the sim narrator was seeded with the previous session's summary, findings and recommendations, a
cross-session memory channel beside the agent's own memory that no Exp 37 arm ablated. The T1-2 re-fire (owner
decision 2026-10-14: a narrow re-fire of Exp 37's A/B/C arms) needs a non-resumed narrator on every cross-session arm,
so only the agent's memory differs. ``start_simulation_mode(narrator_resume=False)`` / ``maxim --sim ... --resume-sim
ID --fresh-narrator``: the stores still restore (``_restore_aut_from_session`` is untouched, #932's site), the
narrator gets the fresh kickoff, and the run's ``resume_record`` says ``narrator_resumed``. Default True: every
existing caller is unchanged.
"""

from __future__ import annotations

import inspect

import pytest

_GATE = pytest.mark.xfail(strict=True, reason="#1226 not built yet")
_RESUME = {
    "turns": 7,
    "total_actions": 12,
    "summary": "PRIOR-SESSION-SUMMARY",
    "issues_found": [],
    "recommendations": [],
}


@_GATE
def test_a_fresh_narrator_gets_the_kickoff_not_the_previous_session():
    from maxim.simulation import orchestrator as orch

    fresh = orch._narrator_opening(_RESUME, "goal-x", "generative", resumed=False, observe_only=False, tools_block="")
    kickoff = orch._build_kickoff_prompt("goal-x", tools_block="", observe_only=False)
    assert fresh == kickoff and "PRIOR-SESSION-SUMMARY" not in fresh


@_GATE
def test_a_resumed_narrator_still_gets_the_previous_session():
    from maxim.simulation import orchestrator as orch

    resumed = orch._narrator_opening(_RESUME, "goal-x", "generative", resumed=True, observe_only=False, tools_block="")
    assert resumed == orch._build_resume_prompt(_RESUME, "goal-x", "generative", observe_only=False, tools_block="")


@_GATE
def test_start_simulation_mode_takes_narrator_resume_defaulting_true():
    from maxim.simulation.orchestrator import start_simulation_mode

    param = inspect.signature(start_simulation_mode).parameters["narrator_resume"]
    assert param.default is True


@_GATE
def test_the_cli_flag_maps_to_narrator_resume():
    from maxim.cli_parser import _build_parser

    parser = _build_parser()
    assert parser.parse_args(["--sim", "g", "--resume-sim", "S", "--fresh-narrator"]).fresh_narrator is True
    assert parser.parse_args(["--sim", "g", "--resume-sim", "S"]).fresh_narrator is False
