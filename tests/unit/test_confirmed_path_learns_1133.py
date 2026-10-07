"""#1133 — a CONFIRMED action learns exactly as the same action does when it runs autonomously.

At SUPERVISED autonomy a tool in ``requires_confirmation`` is parked for a yes/no; on "yes",
``LoopController.handle_confirmation`` ran it through its own copy of the dispatch. That copy crashed
after a successful run (``PendingConfirmation`` has no ``params``) and, had it not, it would still have
learned the wrong thing: it never read the tool's side effects (harm booked POSITIVE), credited under
the loop's agent name rather than the hub's ``agent_id``, credited no situation (the proposal was
cleared when the confirmation was asked), skipped the capture, the plan outcome and
``environment.step``, and queued no follow-up for a failed "process" tool. Owner decision 2026-10-06:
both paths call ONE function, ``tool_dispatch.execute_and_learn``, and the confirmed path keys its
credit to the proposal (``PendingConfirmation.source``).

Driven through the REAL loop: the proposal's tool requires confirmation and interactive mode is OFF,
so the loop answers "yes" itself on the next tick (``tests/unit/_execute_learn_driver.py``).
"""

from __future__ import annotations

import logging

import pytest

from tests.unit._execute_learn_driver import CLUSTERS, HUB_AGENT_ID, REASONING, TRIGGER, credit_view, run_once

TOOL = "probe_learn_1133"
pytestmark = pytest.mark.timeout(60)
RED = pytest.mark.xfail(strict=True, reason="#1133: the confirmed path does not execute-and-learn")


def _valences(obs) -> list[str]:
    return [o["outcome_valence"].value for o in obs.nac_observations if o["event_signature"] == f"tool:{TOOL}"]


@RED
def test_a_confirmed_action_that_harms_books_negative(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, level="supervised", side_effects={"embodiment_failures": ["arms.thermal"]})
    assert [kw["embodiment_failed"] for kw in obs.outcomes] == [True]
    assert _valences(obs) == ["negative"]


@RED
def test_a_confirmed_action_is_credited_to_the_hub_agent_and_its_proposal_time_situation(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, level="supervised")
    assert len(obs.outcomes) == 1, obs.outcomes
    kw = obs.outcomes[0]
    assert (kw["agent_id"], kw["cluster_id"], kw["clusters"], kw["success"]) == (
        HUB_AGENT_ID,
        CLUSTERS["interoception"],
        CLUSTERS,
        True,
    )


@RED
def test_a_confirmed_action_is_captured_planned_and_stepped(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, level="supervised")
    assert [c["situation"] for c in obs.captures] == [CLUSTERS]
    assert obs.plan_outcomes == [{"goal": REASONING, "tool_sequence": [TOOL], "success": True}]
    assert len(obs.env_steps) == 1


@RED
def test_a_confirmed_failed_process_tool_queues_its_follow_up(monkeypatch, tmp_path):
    obs = run_once(
        monkeypatch, tmp_path, tool="run_tests", level="supervised", success=False, output=None, error="nope"
    )
    fu = obs.ctrl.pending_action_followup
    assert fu is not None
    assert (fu.tool, fu.result, fu.original_query, fu.followup_type) == (
        "run_tests",  # a "process" tool outside ALWAYS_ALLOWED_TOOLS, so it does need confirmation
        "[ERROR: nope]",
        TRIGGER,
        "process",
    )


@RED
def test_a_display_that_raises_leaves_the_success_recorded(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="maxim.runtime.loop_controller"):
        obs = run_once(monkeypatch, tmp_path, level="supervised", display_raises=True)
    assert [(kw["success"], kw["error"]) for kw in obs.outcomes] == [(True, None)]
    assert not [r for r in caplog.records if "Confirmed action failed" in r.getMessage()]
    assert obs.ctrl.pending_confirmation is None


@RED
def test_the_same_action_autonomous_and_confirmed_credits_identically(monkeypatch, tmp_path):
    effects = {"drive_potential_diff": 0.4, "drive_relief_channel": "audio"}
    auto = run_once(monkeypatch, tmp_path / "auto", side_effects=effects)
    monkeypatch.undo()
    confirmed = run_once(monkeypatch, tmp_path / "confirmed", side_effects=effects, level="supervised")
    assert len(auto.outcomes) == 1
    assert [credit_view(kw, confirmed) for kw in confirmed.outcomes] == [credit_view(kw, auto) for kw in auto.outcomes]
