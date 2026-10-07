"""Characterization of the main path's execute-and-learn block (#1133, written BEFORE the move).

``run_agentic_loop`` §4 executes an autonomous proposal and books everything the agent learns from it.
#1133 moves that block into ``tool_dispatch.execute_and_learn`` (a pure move) so the SUPERVISED
confirmation path can call the same thing. These pins hold on ``main`` and must hold, unchanged,
across the move: the FULL kwargs the outcome recorder receives in each learning case, the valence the
NAc books, the plan outcome, the capture's situation, ``environment.step``, the write cache
invalidation, the ``tool_called`` event and the follow-up.

Two pins record CURRENT behaviour that is a known defect, so a fix that changes them is visible here:

- #1145: a step after the credit raising (here the capture's ``EncodingContractError``) makes the
  ``except`` branch book a SECOND, negative outcome for the same action.
- The confirmation handler clears ``ctrl.pending_proposal`` when it handles the answer, so a proposal
  made while the confirmation was pending is dropped unexecuted (the substrate proposer runs later in
  the same tick that asked for confirmation).

Driver: ``tests/unit/_execute_learn_driver.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from maxim.decisions.causal_link import Valence
from tests.unit._execute_learn_driver import (
    CLUSTERS,
    HUB_AGENT_ID,
    REASONING,
    TRIGGER,
    credit_view,
    run_once,
    tool_events,
)

TOOL = "probe_learn_1133"
pytestmark = pytest.mark.timeout(60)


def _expected(**overrides: Any) -> dict[str, Any]:
    base = {
        "active_goal": None,
        "agent_id": HUB_AGENT_ID,
        "cluster_id": CLUSTERS["interoception"],
        "clusters": CLUSTERS,
        "drive_credit_withheld": False,
        "drive_potential_diff": None,
        "drive_relief_channel": None,
        "drive_relief_only": False,
        "embodiment_failed": False,
        "error": None,
        "llm_worker": None,
        "max_recent": 10,
        "nac_is_hub_nac": True,
        "outcome_valence": None,
        "reasoning": REASONING,
        "result_summary": "done",
        "success": True,
        "tool_name": TOOL,
        "tool_params": {},
    }
    base.update(overrides)
    return base


def _valences(obs: Any) -> list[str]:
    return [o["outcome_valence"].value for o in obs.nac_observations if o["event_signature"] == f"tool:{TOOL}"]


def test_success_full_credit_kwargs(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path)
    assert [credit_view(kw, obs) for kw in obs.outcomes] == [_expected()]
    assert _valences(obs) == ["positive"]
    assert obs.plan_outcomes == [{"goal": REASONING, "tool_sequence": [TOOL], "success": True}]


def test_harm_books_negative_and_a_failed_plan(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, side_effects={"embodiment_failures": ["arms.thermal"]})
    assert [credit_view(kw, obs) for kw in obs.outcomes] == [_expected(embodiment_failed=True)]
    assert _valences(obs) == ["negative"]
    assert obs.plan_outcomes == [{"goal": REASONING, "tool_sequence": [TOOL], "success": False}]


def test_clamp_withholds_drive_credit(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, side_effects={"drive_credit_withheld": True, "outcome_valence": "neutral"})
    assert [credit_view(kw, obs) for kw in obs.outcomes] == [
        _expected(drive_credit_withheld=True, outcome_valence=Valence.NEUTRAL)
    ]
    assert _valences(obs) == ["neutral"]


def test_drive_relief_carries_its_potential_diff_and_channel(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, side_effects={"drive_potential_diff": 0.4, "drive_relief_channel": "audio"})
    assert [credit_view(kw, obs) for kw in obs.outcomes] == [
        _expected(drive_potential_diff=0.4, drive_relief_channel="audio")
    ]


def test_capture_situation_step_and_event(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path)
    assert [c["situation"] for c in obs.captures] == [CLUSTERS]
    assert obs.captures[0]["intent"] == {"goal": REASONING, "source": "llm_worker"}
    assert len(obs.env_steps) == 1 and obs.env_steps[0].output == "done"
    assert tool_events(obs, "tool_called", TOOL) == [{"tool": TOOL, "success": True, "source": "llm_worker"}]
    assert obs.invalidations == []


def test_a_successful_write_invalidates_its_cached_path(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, tool="write_file", params={"path": "notes/x.txt"})
    assert obs.invalidations == [{"path": "notes/x.txt"}]


def test_a_failed_process_tool_still_queues_its_follow_up(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, tool="read_file", success=False, output=None, error="nope")
    fu = obs.ctrl.pending_action_followup
    assert fu is not None
    assert (fu.tool, fu.result, fu.original_query, fu.followup_type, fu.mode) == (
        "read_file",
        "[ERROR: nope]",
        TRIGGER,
        "process",
        "active",
    )


def test_current_behaviour_1145_a_post_credit_raise_books_a_second_negative(monkeypatch, tmp_path):
    """CURRENT behaviour, a defect (#1145): the capture raising after the credit re-credits as a failure."""
    obs = run_once(monkeypatch, tmp_path, capture_raises=True)
    assert [(kw["success"], kw["error"]) for kw in obs.outcomes] == [
        (True, None),
        (False, "probe: capture contract broken"),
    ]
    assert _valences(obs) == ["positive", "negative"]


def test_current_behaviour_the_confirmation_answer_drops_a_newer_proposal(monkeypatch, tmp_path):
    """CURRENT behaviour: handling the answer clears ``ctrl.pending_proposal``, so the proposal the
    substrate made while the confirmation was pending never runs. The proposer runs at the END of a
    tick and the answer is handled near its START, before §4: with the cadence at zero, the tick that
    asks for confirmation also proposes again, and the next tick's answer clears that proposal."""
    obs = run_once(
        monkeypatch, tmp_path, level="supervised", then_propose="probe_second_1133", submit_interval=0.0, max_steps=8
    )
    assert obs.proposed == [TOOL, "probe_second_1133"], "the second proposal was never made"
    assert [e["_tool"] for e in obs.executed] == [TOOL]
