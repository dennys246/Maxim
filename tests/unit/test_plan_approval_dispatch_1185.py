"""#1185 Surface (owner decision 2026-11-21): an approved plan executes exactly once, through ``execute_and_learn``.

Before: ``handle_plan_approval``'s "yes" moved the parked plan back into ``ctrl.pending_proposal`` with
``requires_approval`` and ``plan_text`` still set, so the loop's plan gate parked it again and prompted again: in
production a "yes" never executed anything. Now the answer executes the plan it was given for, once, on the
PLANNING approved path's own step (``loop_planning._drain_one``: ``approved_action_blocker``, then
``execute_and_learn(human_involved=True)``); a "no" or an ambiguous reply never executes it, and a second "yes"
finds no plan to approve.

Driven through the REAL loop (``tests/unit/_execute_learn_driver.py``): the person's replies arrive in the
interactive mailbox once the plan is shown.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit._execute_learn_driver import run_once

TOOL = "probe_learn_1133"
PLAN = "delete the build directory"
pytestmark = pytest.mark.timeout(60)


def _run(monkeypatch, tmp_path, answers: list[str], **kw: Any):
    return run_once(monkeypatch, tmp_path, plan_text=PLAN, answers=answers, max_steps=10, **kw)


def _events(obs, name: str) -> list[dict[str, Any]]:
    return [d for e, d in obs.events if e == name]


@pytest.mark.parametrize("level", ["autonomous", "planning"])
def test_yes_dispatches_the_plan_exactly_once(monkeypatch, tmp_path, level):
    obs = _run(monkeypatch, tmp_path, ["yes"], level=level)
    assert [e["_tool"] for e in obs.executed] == [TOOL]
    assert len(obs.outcomes) == 1
    shown = [d for d in _events(obs, "plan_awaiting_approval") if not d.get("reask")]
    assert len(shown) == 1, "the approved plan was parked again"
    [entry] = [a for a in obs.autonomy.get_audit_log() if a.action_type == "executed"]
    assert entry.human_involved is True


def test_no_dispatches_nothing_and_books_the_refusal(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, ["No."])
    assert obs.executed == []
    assert len(_events(obs, "plan_rejected")) == 1
    [booked] = obs.outcomes
    assert booked["success"] is False and booked["error"] == "User rejected the proposed plan"


@pytest.mark.parametrize("reply", ["not ok", "ok", "yes?", "\U0001f44e"])
def test_an_ambiguous_reply_dispatches_nothing_and_reasks(monkeypatch, tmp_path, reply):
    obs = _run(monkeypatch, tmp_path, [reply])
    assert obs.executed == []
    assert obs.outcomes == []
    assert [d.get("reask") for d in _events(obs, "plan_awaiting_approval")] == [None, True]
    # The re-asked plan is still the one waiting; nothing else took its place.
    assert obs.ctrl.pending_plan_proposal is not None
    assert obs.ctrl.pending_plan_proposal.action["tool_name"] == TOOL


def test_an_ambiguous_reply_then_yes_dispatches_once(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, ["not ok", "yes"])
    assert [e["_tool"] for e in obs.executed] == [TOOL]


def test_yes_twice_dispatches_once(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, ["yes", "yes"])
    assert [e["_tool"] for e in obs.executed] == [TOOL]
    assert len(_events(obs, "plan_approved")) == 1


def test_a_consumed_answer_never_reaches_the_model_as_a_turn(monkeypatch, tmp_path):
    """The one-line ``agent_loop`` change (#1185): once ``handle_plan_approval`` consumed the reply, ``cli_input``
    is dropped, so the reply never enters the context pool as the person's input (the percept the model's next
    submit reads, ``percept.cli_input``)."""
    from maxim.agents.context_pool import ContextPool

    seen: list[Any] = []
    real_add = ContextPool.add_percept

    def _add(self, percept):
        seen.append(getattr(percept, "cli_input", None))
        return real_add(self, percept)

    monkeypatch.setattr(ContextPool, "add_percept", _add)
    _run(monkeypatch, tmp_path, ["not ok"])
    assert "not ok" not in seen


def test_an_approved_plan_that_is_forbidden_is_refused_not_run(monkeypatch, tmp_path):
    """Approval lifts only the level (S7): a hard-forbidden tool is refused at execution time, typed, as on the
    drain, and booked as a machine refusal."""
    obs = _run(monkeypatch, tmp_path, ["yes"], level="planning", forbid=True)
    assert obs.executed == []
    assert [d["reason"] for d in _events(obs, "plan_refused")] == ["blocked"]
