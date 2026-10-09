"""Characterization of the PLANNING approved path (§5) and of the ``write_file`` overwrite retry (#1085, slice 4).

Written BEFORE §5 moves into ``runtime/loop_planning.py``. At PLANNING, ``run_agentic_loop`` §4 queues every
proposal in ``AutonomyController.proposal_queue``; §5 ("CHECK FOR APPROVED PROPOSALS") executes the entries an
embedder approved, through its OWN copy of the dispatch. Driven through the REAL loop with an embedder that
approves at submit (``tests/unit/_execute_learn_driver.py``, ``level="planning"``).

These pins record CURRENT behaviour, defects included, so the change #1085 makes is visible here; the fix
commit updates each pin it changes, per test:

- §5 books the outcome WITHOUT the tool's params and side effects (a harmful approved action books POSITIVE);
- its follow-up carries ``original_query=""`` (the proposal's triggering input is lost);
- no capture, no plan outcome and no ``environment.step`` run for an approved action;
- ``get_approved`` takes EVERY approved entry out of the queue before the first one executes;
- the deliberation reset runs for a non-``think`` tool and is skipped for ``think``.

The autonomous path's ``write_file`` overwrite retry (``tool_dispatch.execute_and_learn``) is pinned too: a
failed ``write_file`` on an existing file is re-executed with ``overwrite=True`` and the RETRY's result is what
gets credited. It must stay exactly so for the autonomous path; #1085 removes it only where a human approved.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit._execute_learn_driver import (
    CLUSTERS,
    HUB_AGENT_ID,
    REASONING,
    audit_view,
    credit_view,
    run_once,
)

TOOL = "probe_learn_1133"
pytestmark = pytest.mark.timeout(60)


def _approved_expected(**overrides: Any) -> dict[str, Any]:
    """What §5 hands the recorder today: no ``tool_params``, no side-effect fields."""
    base = {
        "active_goal": None,
        "agent_id": HUB_AGENT_ID,
        "cluster_id": CLUSTERS["interoception"],
        "clusters": CLUSTERS,
        "drive_relief_only": False,
        "error": None,
        "llm_worker": None,
        "max_recent": 10,
        "nac_is_hub_nac": True,
        "reasoning": REASONING,
        "result_summary": "done",
        "success": True,
        "tool_name": TOOL,
    }
    base.update(overrides)
    return base


def _valences(obs: Any, tool: str = TOOL) -> list[str]:
    return [o["outcome_valence"].value for o in obs.nac_observations if o["event_signature"] == f"tool:{tool}"]


def test_current_behaviour_the_approved_path_books_without_params_or_side_effects(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, level="planning", params={"x": 1})
    assert [e["_tool"] for e in obs.executed] == [TOOL]
    assert [credit_view(kw, obs) for kw in obs.outcomes] == [_approved_expected()]
    assert _valences(obs) == ["positive"]


def test_current_behaviour_an_approved_action_that_harms_books_positive(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, level="planning", side_effects={"embodiment_failures": ["arms.thermal"]})
    assert [credit_view(kw, obs) for kw in obs.outcomes] == [_approved_expected()]
    assert _valences(obs) == ["positive"]


def test_current_behaviour_no_capture_plan_outcome_or_step_for_an_approved_action(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, level="planning")
    assert [e["_tool"] for e in obs.executed] == [TOOL]
    assert obs.captures == []
    assert obs.plan_outcomes == []
    assert obs.env_steps == []


def test_current_behaviour_the_approved_follow_up_loses_the_triggering_input(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, tool="run_tests", level="planning")
    fu = obs.ctrl.pending_action_followup
    assert fu is not None
    assert (fu.tool, fu.result, fu.original_query, fu.followup_type) == ("run_tests", "done", "", "process")


def test_current_behaviour_the_approved_audit_entry(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, level="planning")
    [entry] = audit_view(obs)
    assert entry["human_involved"] is True
    assert entry["outcome"] == "success"
    assert entry["action"] == {"tool_name": TOOL, "params": {}}
    assert (entry["reasoning"], entry["confidence"], entry["citations"], entry["error"]) == (REASONING, 0.9, [], None)
    [proposed] = audit_view(obs, "proposed")
    assert proposed["human_involved"] is False


@pytest.mark.parametrize(("tool", "resets"), [(TOOL, 1), ("think", 0)])
def test_current_behaviour_the_deliberation_reset_skips_think(monkeypatch, tmp_path, tool, resets):
    obs = run_once(monkeypatch, tmp_path, tool=tool, level="planning", think_probe=tool != "think")
    assert [e["_tool"] for e in obs.executed] == [tool]
    assert len(obs.deliberation_resets) == resets


def test_current_behaviour_get_approved_takes_every_approved_entry_at_once(monkeypatch, tmp_path):
    """Two entries approved before one drain: both leave the queue in the one ``get_approved`` call."""
    from maxim.agents.autonomy import ProposalQueue

    calls: list[list[str]] = []
    real_get = ProposalQueue.get_approved

    def _spy(self: Any) -> Any:
        out = real_get(self)
        if out:
            calls.append([(p.action or {}).get("tool_name") for p in out])
        return out

    monkeypatch.setattr(ProposalQueue, "get_approved", _spy)
    obs = run_once(
        monkeypatch,
        tmp_path,
        level="planning",
        then_propose="second_1085",
        approve_batch=2,
        submit_interval=0.0,
        max_steps=12,
    )
    assert calls == [[TOOL, "second_1085"]]
    assert [e["_tool"] for e in obs.executed] == [TOOL, "second_1085"]
    assert [p.status for p in obs.submitted] == ["approved", "approved"]


def test_the_autonomous_write_file_overwrite_retry(monkeypatch, tmp_path):
    """The autonomous path re-executes a ``write_file`` that failed on an existing file with ``overwrite=True``,
    and credits the RETRY's result. #1085 keeps this exactly so for the autonomous path."""
    obs = run_once(
        monkeypatch,
        tmp_path,
        tool="write_file",
        params={"path": "a.txt", "content": "x"},
        fail_unless_overwrite=True,
    )
    assert obs.executed == [
        {"_tool": "write_file", "path": "a.txt", "content": "x"},
        {"_tool": "write_file", "path": "a.txt", "content": "x", "overwrite": True},
    ]
    assert [(kw["success"], kw["error"], kw["tool_params"]) for kw in obs.outcomes] == [
        (True, None, {"path": "a.txt", "content": "x"})
    ]
    assert obs.invalidations == [{"path": "a.txt"}]
