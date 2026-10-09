"""#1085 (decomposition slice 4, PR-a) -- an APPROVED action learns exactly as the same action does autonomously.

At PLANNING autonomy the loop queues every proposal; an embedder's ``proposal_queue.approve`` is the only route
that approves one (this PR opens no other: the approval surface is #1185). The approved path, §5 of
``run_agentic_loop``, hand-rolled its own dispatch, so an approved action learned less than the same action run
autonomously (no params, no side effects: harm booked POSITIVE; no capture, plan outcome or
``environment.step``; a follow-up without its triggering input). It also drained the queue all at once (one
raise escaping its per-entry handler, e.g. from the recorder, silently lost every other approved entry) and
checked nothing before executing, so an approved entry ran even when its tool was hard-forbidden (by the safety
constraints or the supervision policy) or when a halt landed between the queueing and the drain.

Owner decisions (2026-10-08; 2026-10-09 after the review):

- the approved path runs through ``tool_dispatch.execute_and_learn(human_involved=True)``, the one dispatch the
  autonomous and the SUPERVISED-confirmed paths already share (#1133), keyed to the proposal-time situation
  (#1083); approval lifts only the PLANNING level (S7), so ``AutonomyController.approved_action_blocker`` (pause,
  hard safety constraints, the supervision policy's forbidden tools/prefixes/categories) is checked at drain time;
  the policy's ``allowed_tools`` list is a "needs approval" check, which the approval satisfies (2026-10-09). The
  pause check is narrow: a paused loop's gate idles, so it catches only a pause landing after this tick's gate
  (later in the tick, or during the drain);
- the drain takes ONE entry at a time; a raise (any ``BaseException``) refuses the rest, typed (even if reporting
  them raises too), and the ORIGINAL exception propagates;
- the drain's MACHINE refusals (``blocked``, ``drain_aborted``) are booked to ``recent_outcomes`` and the LLM
  carryover only: no NAc, no situation, no goal credit (#1185 decides the rest);
- with ``human_involved=True`` (confirmed or approved, by a person OR a policy) there is no ``write_file``
  overwrite retry: what was approved runs exactly, and the retry is a second action nobody approved (also on
  #1133's confirmed path; see #1146).

Driven through the REAL loop (``tests/unit/_execute_learn_driver.py``, ``level="planning"``: an embedder approves
at submit). Each ``xfail(strict=True)`` gate fails on ``main`` for the reason it states.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit._execute_learn_driver import TRIGGER, audit_view, credit_view, run_once

TOOL = "probe_learn_1133"
SECOND = "second_1085"
pytestmark = pytest.mark.timeout(60)

_LEVEL_ONLY = ("human_involved", "autonomy_level")


def _refusals(obs: Any) -> list[dict[str, Any]]:
    return [d for e, d in obs.events if e == "plan_refused"]


# ── (a) credit parity ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(
    strict=True,
    reason="#1085: §5 hand-rolls the dispatch -- no tool_params, no side effects, no capture, plan outcome or step",
)
@pytest.mark.parametrize(
    "effects",
    [
        {"drive_potential_diff": 0.4, "drive_relief_channel": "audio"},
        {"embodiment_failures": ["arms.thermal"]},
    ],
    ids=["drive_relief", "harm"],
)
def test_the_same_action_autonomous_and_approved_learns_identically(monkeypatch, tmp_path, effects):
    auto = run_once(monkeypatch, tmp_path / "auto", params={"x": 1}, side_effects=effects)
    monkeypatch.undo()
    approved = run_once(monkeypatch, tmp_path / "approved", params={"x": 1}, side_effects=effects, level="planning")
    assert len(auto.outcomes) == 1
    assert [credit_view(kw, approved) for kw in approved.outcomes] == [credit_view(kw, auto) for kw in auto.outcomes]
    assert [o["outcome_valence"] for o in approved.nac_observations] == [
        o["outcome_valence"] for o in auto.nac_observations
    ]
    assert [c["situation"] for c in approved.captures] == [c["situation"] for c in auto.captures]
    assert approved.plan_outcomes == auto.plan_outcomes
    assert len(approved.env_steps) == len(auto.env_steps) == 1
    # The audit entries differ ONLY in who was involved (and the level each ran at).
    [a_entry], [p_entry] = audit_view(auto), audit_view(approved)
    assert (a_entry["human_involved"], p_entry["human_involved"]) == (False, True)
    assert {k: v for k, v in p_entry.items() if k not in _LEVEL_ONLY} == {
        k: v for k, v in a_entry.items() if k not in _LEVEL_ONLY
    }


@pytest.mark.xfail(
    strict=True, reason="#1085: §5's follow-up drops the error text (result=None) and the trigger (original_query='')"
)
def test_the_approved_follow_up_follows_the_main_path_rule(monkeypatch, tmp_path):
    auto = run_once(monkeypatch, tmp_path / "auto", tool="run_tests", success=False, output=None, error="nope")
    monkeypatch.undo()
    approved = run_once(
        monkeypatch, tmp_path / "approved", tool="run_tests", success=False, output=None, error="nope", level="planning"
    )
    a_fu, p_fu = auto.ctrl.pending_action_followup, approved.ctrl.pending_action_followup
    assert a_fu is not None and p_fu is not None
    view = lambda fu: (fu.tool, fu.result, fu.original_query, fu.followup_type, fu.mode)  # noqa: E731
    assert view(p_fu) == view(a_fu) == ("run_tests", "[ERROR: nope]", TRIGGER, "process", "active")


# ── (b) one raising entry does not lose the others ───────────────────────────────────────────────────


def _two_approved(monkeypatch, tmp_path, **kw: Any) -> Any:
    return run_once(
        monkeypatch,
        tmp_path,
        level="planning",
        then_propose=SECOND,
        approve_batch=2,
        submit_interval=0.0,
        max_steps=12,
        **kw,
    )


@pytest.mark.xfail(
    strict=True,
    reason="#1085: get_approved takes every approved entry before the first executes; a raise loses the rest silently",
)
def test_a_raising_approved_entry_refuses_the_rest_typed_and_propagates(monkeypatch, tmp_path):
    obs = _two_approved(monkeypatch, tmp_path, record_raises_for=TOOL, catch=True)
    assert isinstance(obs.raised, RuntimeError), obs.raised
    assert [e["_tool"] for e in obs.executed] == [TOOL]
    first, second = obs.submitted
    assert (second.status, second.rejected_reason) == ("rejected", "drain_aborted")
    assert [(d["reason"], d["tool"]) for d in _refusals(obs)] == [("drain_aborted", SECOND)]


@pytest.mark.xfail(strict=True, reason="#1085: a raise loses the rest; nothing guards a failing refusal")
def test_an_aborted_drain_rejects_every_entry_even_when_booking_them_raises(monkeypatch, tmp_path):
    """Executor S1: ``execute_and_learn`` raises E1 (the first entry's recorder) and booking the refusal raises E2
    (the second entry's recorder). Every remaining entry still ends rejected/drain_aborted, E1 (never E2)
    propagates, and a later drain finds nothing approved to run."""
    obs = _two_approved(monkeypatch, tmp_path, record_raises_for=(TOOL, SECOND), catch=True)
    assert isinstance(obs.raised, RuntimeError) and TOOL in str(obs.raised), obs.raised
    assert [e["_tool"] for e in obs.executed] == [TOOL]
    _first, second = obs.submitted
    assert (second.status, second.rejected_reason) == ("rejected", "drain_aborted")
    from maxim.runtime.loop_planning import drain_approved  # after the behavioural checks (absent on main)

    ran: list[Any] = []
    again = drain_approved(
        autonomy_controller=obs.autonomy,
        execute_and_learn=lambda **kw: ran.append(kw),
        book_machine_refusal=lambda **kw: ran.append(kw),
        observation={},
        state=obs.ctrl.state,
        sim=_NullSim(),
    )
    assert again is None and ran == []


class _NullSim:
    def log(self, *a: Any, **k: Any) -> None:
        return None


@pytest.mark.xfail(
    strict=True,
    reason="#1085: a KeyboardInterrupt in the first approved entry leaves the rest approved, for a later drain to run",
)
def test_a_keyboard_interrupt_mid_drain_leaves_nothing_approved(monkeypatch, tmp_path):
    """Delta review S2: a ``BaseException`` (here ``KeyboardInterrupt`` out of the first entry's tool, which
    ``execute_and_learn``'s own handler does not catch) still refuses every remaining entry, propagates, and
    leaves a later drain nothing to run."""
    obs = _two_approved(monkeypatch, tmp_path, tool_raises=KeyboardInterrupt, catch=True)
    assert isinstance(obs.raised, KeyboardInterrupt), obs.raised
    assert [e["_tool"] for e in obs.executed] == [TOOL]
    _first, second = obs.submitted
    assert (second.status, second.rejected_reason) == ("rejected", "drain_aborted")
    from maxim.runtime.loop_planning import drain_approved  # after the behavioural checks (absent on main)

    ran: list[Any] = []
    again = drain_approved(
        autonomy_controller=obs.autonomy,
        execute_and_learn=lambda **kw: ran.append(kw),
        book_machine_refusal=lambda **kw: ran.append(kw),
        observation={},
        state=obs.ctrl.state,
        sim=_NullSim(),
    )
    assert again is None and ran == []


# ── (c) approval lifts only the PLANNING level ───────────────────────────────────────────────────────

# How each blocked entry is made, and what its refusal detail says. Approval lifts only the level (owner decision
# S7): a hard SafetyConstraints forbid, the supervision policy's hard denials (forbidden tools, prefixes) and a
# pause landing after this tick's gate (``pause_on_approve`` halts inside the submit, later in the tick) all still
# refuse.
_BLOCKED = {
    "forbidden": ({"forbid": True}, "is forbidden"),
    "paused": ({"pause_on_approve": True}, "paused"),
    "policy_forbidden": ({"policy": {"forbidden_tools": {TOOL}}}, "is forbidden"),
    "policy_prefix": ({"policy": {"forbidden_prefixes": ("probe_",)}}, "prefix rule"),
}


@pytest.mark.xfail(
    strict=True,
    reason="#1085: §5 executes an approved entry without checking the pause, the hard safety constraints or the policy's hard denials",
)
@pytest.mark.parametrize("how", list(_BLOCKED))
def test_an_approved_blocked_action_is_refused_typed_and_not_executed(monkeypatch, tmp_path, how):
    knobs, detail = _BLOCKED[how]
    obs = run_once(monkeypatch, tmp_path, level="planning", **knobs)
    assert obs.executed == []
    [proposal] = obs.submitted
    assert (proposal.status, proposal.rejected_reason) == ("rejected", "blocked")
    [refusal] = _refusals(obs)
    assert (refusal["reason"], refusal["tool"]) == ("blocked", TOOL)
    assert detail in refusal["detail"].lower()


@pytest.mark.xfail(
    strict=True,
    reason="#1085: §5 hand-rolls the dispatch -- an approved tool outside allowed_tools runs without parity",
)
def test_an_approved_tool_outside_allowed_tools_executes_with_parity(monkeypatch, tmp_path):
    """Owner decision 2026-10-09: ``allowed_tools`` is a "needs approval" check, not a hard denial, so an approved
    tool outside a non-empty list RUNS, and learns as it would autonomously."""
    auto = run_once(monkeypatch, tmp_path / "auto", params={"x": 1})
    monkeypatch.undo()
    approved = run_once(
        monkeypatch, tmp_path / "approved", params={"x": 1}, level="planning", policy={"allowed_tools": {"other"}}
    )
    assert [e["_tool"] for e in approved.executed] == [TOOL]
    assert _refusals(approved) == []
    assert [credit_view(kw, approved) for kw in approved.outcomes] == [credit_view(kw, auto) for kw in auto.outcomes]
    assert approved.plan_outcomes == auto.plan_outcomes


# ── (d) a machine refusal books no NAc ───────────────────────────────────────────────────────────────


@pytest.mark.xfail(
    strict=True,
    reason="#1085: no machine refusal exists on main -- a blocked entry runs (and books NAc), a drained-away one is lost",
)
@pytest.mark.parametrize("case", ["forbidden", "policy_forbidden", "policy_prefix", "drain_aborted"])
def test_a_machine_refusal_books_recent_outcomes_only(monkeypatch, tmp_path, case):
    reason = "drain_aborted" if case == "drain_aborted" else "blocked"
    if case != "drain_aborted":
        obs = run_once(monkeypatch, tmp_path, level="planning", active_goal="probe goal", **_BLOCKED[case][0])
        refused = TOOL
    else:
        obs = _two_approved(monkeypatch, tmp_path, record_raises_for=TOOL, catch=True, active_goal="probe goal")
        refused = SECOND
    booked = [kw for kw in obs.all_outcomes if kw["tool_name"] == refused]
    assert len(booked) == 1, booked
    [kw] = booked
    assert kw["success"] is False and reason in kw["error"]
    # No NAc and no situation: nothing to credit, by the signature of the call.
    assert kw["nac"] is None
    assert kw.get("cluster_id") is None and kw.get("clusters") is None
    # It reaches the outcome window the LLM's carryover reads.
    assert [o["tool"] for o in obs.ctrl.recent_outcomes if not o["success"]][-1] == refused
    # Nothing touched the NAc for it: no causal observation, no cluster reward, no goal credit.
    assert not [o for o in obs.nac_observations if o["event_signature"] == f"tool:{refused}"]
    assert not [name for name, _ in obs.nac_calls if name in ("update_cluster_reward", "credit_goal")]


# ── (e) no overwrite retry for an action approved by a person or a policy ────────────────────────────

_WRITE = {"path": "a.txt", "content": "x"}


@pytest.mark.xfail(
    strict=True,
    reason="#1085/#1146: execute_and_learn re-executes a CONFIRMED write_file with overwrite=True",
)
def test_a_policy_confirmed_write_file_is_not_retried_with_overwrite(monkeypatch, tmp_path):
    """The confirmation here is a MACHINE "yes" (interactive mode OFF: the non-interactive SUPERVISED auto-yes).
    ``human_involved=True`` means confirmed or approved by a person OR a policy, and either way the action
    confirmed runs exactly: no second, ``overwrite=True`` execution."""
    obs = run_once(
        monkeypatch, tmp_path, tool="write_file", params=dict(_WRITE), fail_unless_overwrite=True, level="supervised"
    )
    assert obs.executed == [{"_tool": "write_file", **_WRITE}]
    assert [(kw["success"], kw["error"]) for kw in obs.outcomes] == [(False, "File already exists: a.txt")]


def test_an_approved_write_file_is_not_retried_with_overwrite(monkeypatch, tmp_path):
    """A GUARD, green on ``main`` (§5 hand-rolls ONE ``executor.execute``): routing §5 through
    ``execute_and_learn`` must not bring the autonomous path's overwrite retry with it."""
    obs = run_once(
        monkeypatch, tmp_path, tool="write_file", params=dict(_WRITE), fail_unless_overwrite=True, level="planning"
    )
    assert obs.executed == [{"_tool": "write_file", **_WRITE}]
    assert [(kw["success"], kw["error"]) for kw in obs.outcomes] == [(False, "File already exists: a.txt")]
