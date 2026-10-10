"""#963 -- what the model is shown and what dispatch enforces read ONE operational mode.

The operator's launch grant (``--operational-mode``, #829) decides dispatch, but the agent loop's other mode
readers each read the loop state's run mode with their own default. So under a passive grant over a ``live`` run a
search still queued an ``engage`` follow-up (the "offer follow-ups" template) labelled ``live``, the audit recorded
``live`` for an action dispatch ran as passive, an explicit empty run mode restricted nothing at dispatch, and a
state with no mode left the Default Network unconfigured.

Owner decisions (#963, 2026-10-09): one precedence, owned by ``Executor.effective_operational_mode()`` (the grant,
else the loop's mode source, normalised; the loop reads it through ``loop_state.operational_mode``) with ONE
default, ``observe``, and never an empty value (Q1, Q4); the follow-up's mode, the audit's mode and the minimal
context's mode record the OPERATIONAL mode (Q5); behavioural tests for the Default Network and the follow-up, each
flipping the grant between two ticks so a cached value cannot pass (Q6).

Driven through the REAL loop (``tests/unit/_execute_learn_driver.py``; the Default Network through the pre-tick
gate characterization's driver). Each ``xfail(strict=True)`` gate fails on ``main`` for the reason it states. The
by-class follow-up downgrade (Q3) has its own gates: ``tests/unit/test_followup_type_by_class_963.py``.
"""

from __future__ import annotations

import pytest

from tests.unit._execute_learn_driver import run_once

pytestmark = pytest.mark.timeout(60)

ENGAGE_TOOL = "internet_search"


@pytest.mark.xfail(
    strict=True, reason="#963: the follow-up reads state.data['mode'] (default 'live'), not the passive grant"
)
@pytest.mark.parametrize("level", ["autonomous", "supervised", "planning"])
def test_a_passive_grant_is_the_follow_ups_type_and_mode_on_every_path(monkeypatch, tmp_path, level):
    """All three paths share ``execute_and_learn``: autonomous, policy-confirmed (SUPERVISED, non-interactive
    auto-yes) and approved (PLANNING, an embedder's approval)."""
    obs = run_once(monkeypatch, tmp_path, tool=ENGAGE_TOOL, level=level, state_mode="live", grant="passive")
    assert [e["_tool"] for e in obs.executed] == [ENGAGE_TOOL]
    [fu] = obs.followups
    assert (fu.followup_type, fu.mode) == ("respond", "passive")


# Every audit entry a run logs, by path: executed (autonomous, confirmed, approved), proposed (PLANNING), and the
# refusals: a SUPERVISED hard rejection (the policy forbids the tool) and a drained PLANNING entry refused (blocked).
_AUDITED = {
    "autonomous": ({"level": "autonomous"}, {"executed"}),
    "supervised": ({"level": "supervised"}, {"executed"}),
    "planning": ({"level": "planning"}, {"proposed", "executed"}),
    "supervised_hard_rejection": ({"level": "supervised", "policy": {"forbidden_tools": {ENGAGE_TOOL}}}, {"rejected"}),
    "planning_refused": ({"level": "planning", "forbid": True}, {"proposed", "rejected"}),
}


@pytest.mark.xfail(strict=True, reason="#963: the audit records the run mode ('live'), not the mode dispatch enforced")
@pytest.mark.parametrize("path", list(_AUDITED))
def test_the_audit_records_the_operational_mode(monkeypatch, tmp_path, path):
    knobs, kinds = _AUDITED[path]
    obs = run_once(monkeypatch, tmp_path, tool=ENGAGE_TOOL, state_mode="live", grant="passive", **knobs)
    entries = obs.autonomy.get_audit_log()
    assert {e.action_type for e in entries} == kinds
    assert {e.mode for e in entries} == {"passive"}


@pytest.mark.xfail(strict=True, reason="#963: a refused confirmation's audit entry records the run mode ('live')")
def test_a_refused_confirmation_is_audited_with_the_operational_mode():
    """The confirmation "no" branch (``LoopController.handle_confirmation``), with the run's real executor."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from maxim.agents.llm_types import LLMProposal
    from maxim.runtime.bootstrap import build_executor
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.loop_setup import _prepare_executor
    from maxim.runtime.loop_types import PendingConfirmation
    from maxim.tools.registry import ToolRegistry

    state = SimpleNamespace(data={"mode": "live"})
    executor = _prepare_executor(build_executor(ToolRegistry(), pain_bus=None, permissions=None), None, state)
    executor.set_operational_override("passive")
    autonomy = MagicMock()
    ctrl = LoopController(
        agent=MagicMock(),
        environment=MagicMock(),
        state=state,
        memory=MagicMock(),
        decision_engine=MagicMock(),
        executor=executor,
        autonomy_controller=autonomy,
    )
    ctrl.pending_confirmation = PendingConfirmation.from_proposal(
        LLMProposal(
            request_id="r-963",
            action={"tool_name": ENGAGE_TOOL, "params": {}},
            reasoning="probe",
            strategy_used=None,
            confidence=0.9,
            mode_goal_achieved=False,
        )
    )
    assert ctrl.handle_confirmation("no", execute_and_learn=MagicMock(), book_refusal=MagicMock(), observation={})
    [call] = autonomy.log_action.call_args_list
    assert (call.kwargs["action_type"], call.kwargs["mode"]) == ("rejected", "passive")


@pytest.mark.xfail(
    strict=True, reason="#963: the follow-up never sees the grant, so a grant set mid-run changes nothing"
)
def test_a_grant_set_between_two_ticks_reaches_the_next_follow_up_and_audit(monkeypatch, tmp_path):
    """Owner decision Q6: read per use, never cached. The first search runs with no grant, the grant is set before
    the second is proposed, and the second's follow-up and audit entry carry it."""
    obs = run_once(
        monkeypatch,
        tmp_path,
        tool=ENGAGE_TOOL,
        then_propose="web_search",
        submit_interval=0.0,
        max_steps=12,
        state_mode="live",
        grant_at={1: "passive"},
    )
    assert [e["_tool"] for e in obs.executed] == [ENGAGE_TOOL, "web_search"]
    assert [(f.tool, f.followup_type, f.mode) for f in obs.followups] == [
        (ENGAGE_TOOL, "engage", "live"),
        ("web_search", "respond", "passive"),
    ]
    assert [(e.action["tool_name"], e.mode) for e in obs.autonomy.get_audit_log()] == [
        (ENGAGE_TOOL, "live"),
        ("web_search", "passive"),
    ]


@pytest.mark.xfail(strict=True, reason="#963: an empty mode from the loop's mode source restricts nothing at dispatch")
def test_an_explicit_empty_run_mode_is_passive_at_dispatch():
    """Owner decision Q4: the one default is ``observe`` (passive), never an empty value. The prompt roster already
    showed passive for it; dispatch ran a host-acting tool. Composed as the loop composes it."""
    from maxim.runtime.bootstrap import build_executor
    from maxim.runtime.loop_setup import _prepare_executor
    from maxim.runtime.state import RuntimeState
    from maxim.tools.registry import ToolRegistry

    state = RuntimeState()
    state.data["mode"] = ""
    executor = _prepare_executor(build_executor(ToolRegistry(), pain_bus=None, permissions=None), None, state)
    denial = executor._mode_denial("bash")
    assert denial is not None and "passive mode does not allow" in denial


@pytest.mark.xfail(strict=True, reason="#963: a state with no mode leaves the Default Network unconfigured")
def test_the_default_network_is_observe_when_the_state_has_no_mode(monkeypatch, tmp_path):
    from tests.unit.test_loop_gates_characterization import _run

    run = _run(monkeypatch, tmp_path, steps=2, mode=None)
    assert [e for e in run.ev if e[0] == "dn"] == [("dn", "observe"), ("dn", "observe")]
