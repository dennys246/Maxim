"""Characterization of the PLANNING approved path (§5) and of the ``write_file`` overwrite retry (#1085, slice 4).

Written BEFORE §5 moves into ``runtime/loop_planning.py``. At PLANNING, ``run_agentic_loop`` §4 queues every
proposal in ``AutonomyController.proposal_queue``; §5 ("CHECK FOR APPROVED PROPOSALS") executes the entries an
embedder approved, through its OWN copy of the dispatch. Driven through the REAL loop with an embedder that
approves at submit (``tests/unit/_execute_learn_driver.py``, ``level="planning"``).

These pins recorded the behaviour on ``main``, defects included, so the change #1085 makes is visible here. The
fix commit (§5 → ``loop_planning.drain_approved`` → ``tool_dispatch.execute_and_learn(human_involved=True)``)
changed these pins, each saying so in its docstring:

- §5 booked the outcome WITHOUT the tool's params and side effects (a harmful approved action booked POSITIVE)
  → it books the autonomous path's full kwargs (harm NEGATIVE);
- its follow-up carried ``original_query=""`` → the proposal's triggering input;
- no capture, no plan outcome and no ``environment.step`` ran for an approved action → all three run;
- ``get_approved`` took EVERY approved entry out of the queue before the first one executed → the drain takes
  ONE at a time (``ProposalQueue.pop_approved``).

Unchanged: the audit entry ON SUCCESS (``human_involved=True``; on a failure the entry now also carries the tool's
``error``, which §5 did not pass) and the deliberation reset (skipped for ``think``).

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
    TRIGGER,
    audit_view,
    credit_view,
    run_once,
)
from tests.unit.test_execute_and_learn_characterization import _expected as _autonomous_expected

TOOL = "probe_learn_1133"
pytestmark = pytest.mark.timeout(60)


def _approved_expected(**overrides: Any) -> dict[str, Any]:
    """What the approved path hands the recorder: the autonomous path's full kwargs (changed by #1085's fix; on
    ``main`` §5 passed no ``tool_params`` and no side-effect fields)."""
    expected = _autonomous_expected(**overrides)
    assert (expected["agent_id"], expected["clusters"]) == (HUB_AGENT_ID, CLUSTERS)
    return expected


def _valences(obs: Any, tool: str = TOOL) -> list[str]:
    return [o["outcome_valence"].value for o in obs.nac_observations if o["event_signature"] == f"tool:{tool}"]


def test_the_approved_path_books_with_its_params_and_side_effects(monkeypatch, tmp_path):
    """Changed by #1085's fix: ``main`` booked no ``tool_params`` and no side-effect fields."""
    obs = run_once(monkeypatch, tmp_path, level="planning", params={"x": 1})
    assert [e["_tool"] for e in obs.executed] == [TOOL]
    assert [credit_view(kw, obs) for kw in obs.outcomes] == [_approved_expected(tool_params={"x": 1})]
    assert _valences(obs) == ["positive"]


def test_an_approved_action_that_harms_books_negative(monkeypatch, tmp_path):
    """Changed by #1085's fix: ``main`` booked an approved harmful action POSITIVE (side effects unread)."""
    obs = run_once(monkeypatch, tmp_path, level="planning", side_effects={"embodiment_failures": ["arms.thermal"]})
    assert [credit_view(kw, obs) for kw in obs.outcomes] == [_approved_expected(embodiment_failed=True)]
    assert _valences(obs) == ["negative"]


def test_an_approved_action_is_captured_planned_and_stepped(monkeypatch, tmp_path):
    """Changed by #1085's fix: on ``main`` no capture, plan outcome or ``environment.step`` ran."""
    obs = run_once(monkeypatch, tmp_path, level="planning")
    assert [e["_tool"] for e in obs.executed] == [TOOL]
    assert [c["situation"] for c in obs.captures] == [CLUSTERS]
    assert obs.plan_outcomes == [{"goal": REASONING, "tool_sequence": [TOOL], "success": True}]
    assert len(obs.env_steps) == 1


def test_the_approved_follow_up_carries_the_triggering_input(monkeypatch, tmp_path):
    """Changed by #1085's fix: ``main`` queued ``original_query=""``."""
    obs = run_once(monkeypatch, tmp_path, tool="run_tests", level="planning")
    fu = obs.ctrl.pending_action_followup
    assert fu is not None
    assert (fu.tool, fu.result, fu.original_query, fu.followup_type) == ("run_tests", "done", TRIGGER, "process")


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


def test_the_drain_takes_one_approved_entry_at_a_time(monkeypatch, tmp_path):
    """Two entries approved before one drain. Changed by #1085's fix: on ``main`` both left the queue in ONE
    ``get_approved`` call before the first executed; the drain now pops one, executes it, then pops the next."""
    from maxim.agents.autonomy import ProposalQueue
    from maxim.runtime import loop_setup

    calls: list[str] = []
    real_pop = ProposalQueue.pop_approved
    real_eal = loop_setup.execute_and_learn

    def _eal(**kw: Any) -> Any:
        calls.append(f"exec:{kw['action']['tool_name']}")
        return real_eal(**kw)

    monkeypatch.setattr(loop_setup, "execute_and_learn", _eal)

    def _spy(self: Any) -> Any:
        out = real_pop(self)
        if out is not None:
            calls.append(f"pop:{(out.action or {}).get('tool_name')}")
        return out

    monkeypatch.setattr(ProposalQueue, "pop_approved", _spy)
    obs = run_once(
        monkeypatch,
        tmp_path,
        level="planning",
        then_propose="second_1085",
        approve_batch=2,
        submit_interval=0.0,
        max_steps=12,
    )
    assert [e["_tool"] for e in obs.executed] == [TOOL, "second_1085"]
    assert calls == [f"pop:{TOOL}", f"exec:{TOOL}", "pop:second_1085", "exec:second_1085"]
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
