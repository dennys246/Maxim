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
    kw.setdefault("interactive", True)  # a person answers the prompt
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


# ── S3: the slot holds the plan that was shown ───────────────────────────────────────────────────────


SECOND = "second_1085"


def test_a_plan_proposed_while_one_waits_is_refused_and_the_yes_runs_the_waiting_one(monkeypatch, tmp_path):
    """A parked, then B proposed: B is refused ``awaiting_human`` (a machine refusal: no NAc) and A keeps the
    slot; the person's "yes" (typed after B was refused) runs A once and B never."""
    obs = run_once(
        monkeypatch,
        tmp_path,
        plan_text=PLAN,
        plan_every=True,
        then_propose=SECOND,
        submit_interval=0.0,
        answers=["yes"],
        answer_on="plan_refused",
        max_steps=14,
        interactive=True,
    )
    assert [e["_tool"] for e in obs.executed] == [TOOL]
    assert [(d["reason"], d["tool"]) for d in _events(obs, "plan_refused")] == [("awaiting_human", SECOND)]
    assert [o for o in obs.nac_observations if SECOND in str(o.get("event_signature"))] == []  # B booked no NAc
    [refusal] = [kw for kw in obs.all_outcomes if kw["tool_name"] == SECOND]  # B: the machine refusal only
    assert refusal["success"] is False and refusal.get("nac") is None
    assert "awaiting_human" in refusal["error"]
    [shown] = [d for d in _events(obs, "plan_awaiting_approval") if not d.get("reask")]
    assert (shown["tool"], shown["params"]) == (TOOL, "{}")  # bounded JSON (a write_file body must not flood)


# ── Decision A: the approval is the SUPERVISED confirmation; the policy's other checks still apply ──────


def test_at_supervised_an_approved_plan_needs_no_second_confirmation(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, ["yes"], level="supervised")  # the probe tool is in requires_confirmation
    assert [e["_tool"] for e in obs.executed] == [TOOL]
    assert _events(obs, "user_confirmed") == []


def test_at_supervised_a_forbidden_tool_is_refused(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, ["yes"], level="supervised", policy={"forbidden_tools": {TOOL}})
    assert obs.executed == []
    assert [d["reason"] for d in _events(obs, "plan_refused")] == ["blocked"]


CONFIRMABLE = [
    ({"allowed_tools": {"some_other_tool"}}, TOOL, {}),
    ({}, "write_file", {"path": "/definitely/outside/the/workspace.txt"}),
]
CONFIRMABLE_IDS = ["allowed_tools", "cwd_write"]


@pytest.mark.parametrize(("policy", "tool", "params"), CONFIRMABLE, ids=CONFIRMABLE_IDS)
def test_a_persons_approval_satisfies_every_confirmable_reason(monkeypatch, tmp_path, policy, tool, params):
    """Corrected decision A (2026-11-21): outside ``allowed_tools`` and a CWD write each "require approval", which
    is exactly what the person just gave, so the approved plan runs, once, as a SUPERVISED confirm-yes would."""
    obs = _run(monkeypatch, tmp_path, ["yes"], tool=tool, params=params, level="planning", policy=policy or None)
    assert [e["_tool"] for e in obs.executed] == [tool]
    assert _events(obs, "plan_refused") == []


def test_the_confidence_floor_still_refuses_an_approved_plan(monkeypatch, tmp_path):
    obs = _run(monkeypatch, tmp_path, ["yes"], level="planning", policy={"min_confidence_autonomous": 0.95})
    assert obs.executed == []
    [refused] = _events(obs, "plan_refused")
    assert refused["reason"] == "not_permitted" and "requires approval" not in refused["detail"]
    assert obs.nac_observations == []


@pytest.mark.parametrize(("policy", "tool", "params"), CONFIRMABLE, ids=CONFIRMABLE_IDS)
def test_a_machine_approval_refuses_what_needs_a_person(monkeypatch, tmp_path, policy, tool, params):
    """The non-interactive run's machine approval lifts the PLANNING gate only: a reason left for a person is
    refused ``needs_person`` (a machine refusal, no NAc), told plainly, never "requires approval"."""
    from maxim.simulation import sim_logger

    shown: list[str] = []
    monkeypatch.setattr(sim_logger, "display_status", lambda text: shown.append(str(text)))
    obs = run_once(
        monkeypatch,
        tmp_path,
        tool=tool,
        params=params,
        plan_text=PLAN,
        max_steps=10,
        level="planning",
        policy=policy,
        interactive=False,
    )
    assert obs.executed == []
    [refused] = _events(obs, "plan_refused")
    assert refused["reason"] == "needs_person"
    assert any("needs a person's approval; this run answered its own prompt" in t for t in shown), shown
    assert not any("requires approval" in t for t in shown), shown
    assert obs.nac_observations == []


def test_a_machine_approval_runs_what_needs_no_person(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, plan_text=PLAN, max_steps=10, level="planning", interactive=False)
    assert [e["_tool"] for e in obs.executed] == [TOOL]
    [entry] = [a for a in obs.autonomy.get_audit_log() if a.action_type == "executed"]
    assert entry.human_involved is True  # the approved path's flag: approved, by a person or a policy


def test_three_non_interactive_plans_make_three_decisions(monkeypatch, tmp_path):
    """Wire 1: with nobody to answer, a parked plan used to wait forever and every later plan was refused
    ``awaiting_human``. Now each plan is decided at once (machine approval): three plans, three dispatches."""
    obs = run_once(
        monkeypatch,
        tmp_path,
        plan_text=PLAN,
        plan_every=True,
        then_propose=(SECOND, "third_1185"),
        submit_interval=0.0,
        max_steps=18,
        interactive=False,  # the default `maxim --mode agentic --interactive false` shape
    )
    assert [e["_tool"] for e in obs.executed] == [TOOL, SECOND, "third_1185"]
    assert [d for d in _events(obs, "plan_refused") if d["reason"] == "awaiting_human"] == []
    assert _events(obs, "plan_awaiting_approval") == []


# ── Decision B: the sim's "yes" takes the same path ──────────────────────────────────────────────────


@pytest.mark.parametrize("level", ["autonomous", "planning"])
def test_a_sim_yes_dispatches_once_through_the_approved_path(monkeypatch, tmp_path, level):
    obs = run_once(monkeypatch, tmp_path, plan_text=PLAN, sim_plan_answer="Yes.", max_steps=10, level=level)
    assert [e["_tool"] for e in obs.executed] == [TOOL]
    [entry] = [a for a in obs.autonomy.get_audit_log() if a.action_type == "executed"]
    assert entry.human_involved is True  # a sim AUTO_APPROVE counts as involved (execute_and_learn's docstring)
    assert _events(obs, "plan_awaiting_approval") == []  # the policy answered: nothing parked


def test_a_sim_no_dispatches_nothing(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, plan_text=PLAN, sim_plan_answer="no", max_steps=10)
    assert obs.executed == []


# ── Wire item 1: the loop's own "yes" is a machine's ─────────────────────────────────────────────────


def test_a_non_interactive_supervised_confirmation_is_refused_not_auto_confirmed(monkeypatch, tmp_path):
    """The non-interactive run answers its own SUPERVISED confirmation "yes" (``InputSource.LOOP``). That machine
    "yes" satisfies no reason that needs a person (corrected decision A), so the action is refused typed
    ``needs_person`` (a machine refusal, no NAc) instead of being auto-confirmed."""
    obs = run_once(monkeypatch, tmp_path, level="supervised", person_confirms=False, max_steps=10)
    assert obs.executed == []
    assert [d["reason"] for d in _events(obs, "plan_refused")] == ["needs_person"]
    assert obs.nac_observations == []


def test_a_person_confirming_at_supervised_still_runs(monkeypatch, tmp_path):
    obs = run_once(monkeypatch, tmp_path, level="supervised", max_steps=10)
    assert [e["_tool"] for e in obs.executed] == [TOOL]
