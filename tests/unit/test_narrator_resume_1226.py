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


_RESUME = {
    "turns": 7,
    "total_actions": 12,
    "summary": "PRIOR-SESSION-SUMMARY",
    "issues_found": [],
    "recommendations": [],
}


def test_a_fresh_narrator_gets_the_kickoff_not_the_previous_session():
    from maxim.simulation import orchestrator as orch

    fresh = orch._narrator_opening(_RESUME, "goal-x", "generative", resumed=False, observe_only=False, tools_block="")
    kickoff = orch._build_kickoff_prompt("goal-x", tools_block="", observe_only=False)
    assert fresh == kickoff and "PRIOR-SESSION-SUMMARY" not in fresh


def test_a_resumed_narrator_still_gets_the_previous_session():
    from maxim.simulation import orchestrator as orch

    resumed = orch._narrator_opening(_RESUME, "goal-x", "generative", resumed=True, observe_only=False, tools_block="")
    assert resumed == orch._build_resume_prompt(_RESUME, "goal-x", "generative", observe_only=False, tools_block="")


def test_start_simulation_mode_takes_narrator_resume_defaulting_true():
    from maxim.simulation.orchestrator import start_simulation_mode

    param = inspect.signature(start_simulation_mode).parameters["narrator_resume"]
    assert param.default is True


def test_the_cli_flag_maps_to_narrator_resume():
    from maxim.cli_parser import _build_parser

    parser = _build_parser()
    assert parser.parse_args(["--sim", "g", "--resume-sim", "S", "--fresh-narrator"]).fresh_narrator is True
    assert parser.parse_args(["--sim", "g", "--resume-sim", "S"]).fresh_narrator is False


def test_every_cli_call_of_start_simulation_mode_forwards_the_flag():
    """The flag is only real if every CLI call site forwards it: a call passing resume_session but not
    narrator_resume would silently resume the narrator (a first cut missed both sites and every other test passed)."""
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).resolve().parents[2] / "src" / "maxim" / "cli.py").read_text(encoding="utf-8"))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "start_simulation_mode"
    ]
    resuming = [c for c in calls if any(k.arg == "resume_session" for k in c.keywords)]
    assert resuming, "no CLI call passes resume_session"
    assert all(any(k.arg == "narrator_resume" for k in c.keywords) for c in resuming)


@pytest.mark.timeout(300)
def test_the_real_simulation_opens_a_fresh_narrator_on_a_resume(monkeypatch, tmp_path):
    """The composition: the real ``start_simulation_mode`` (offline harness), resuming a prior session on disk with
    narrator_resume=False, restores the run but does NOT hand the narrator the previous session, and records it.
    (The resumed default is today's behaviour; the unit tests pin its prompt. One arm keeps this to ~3 min.)"""
    narrator_resume = False
    import json

    from maxim.simulation import orchestrator as orch
    from tests.unit import _orch_harness as H

    prior = tmp_path / "data" / "sim_reports" / "prior_1226"
    prior.mkdir(parents=True)
    (prior / "report.json").write_text(
        json.dumps({"_format_version": "1.0", "llm_summary": "PRIOR-SESSION-SUMMARY", "turns": 3, "total_actions": 2})
    )
    seen: list[tuple[bool, str]] = []
    real = orch._narrator_opening

    def recording(*args, **kwargs):
        opening = real(*args, **kwargs)
        seen.append((kwargs["resumed"], opening))
        return opening

    monkeypatch.setattr(orch, "_narrator_opening", recording)
    arm = f"resume_{narrator_resume}"
    monkeypatch.setitem(
        H.EXTRA_ARMS,
        arm,
        {"max_turns": 3, "extra_kwargs": {"resume_session": "prior_1226", "narrator_resume": narrator_resume}},
    )
    trace = H.run_arm(arm, tmp_path)
    assert len(seen) == 1
    resumed, opening = seen[0]
    assert resumed is narrator_resume
    assert ("PRIOR-SESSION-SUMMARY" in opening) is narrator_resume
    assert trace["report"]["provenance"]["resume"]["narrator_resumed"] is narrator_resume
