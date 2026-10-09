"""Unit pins for the PLANNING approved path's pieces (#1085, decomposition slice 4).

The loop-level behaviour is pinned through the real loop in ``test_approved_path_learns_1085.py`` and
``test_approved_path_characterization.py``; these pin the pieces directly:

- ``AutonomyController.approval_blocker`` is THE non-level half of ``can_execute_action`` (one function, called
  by it, so the two cannot drift), and approval lifts only the PLANNING level;
- ``ProposalQueue.pop_approved`` takes the oldest approved entry, one at a time;
- ``SupervisionPolicy.hard_deny`` is the policy's one hard-denial check (forbidden prefixes, categories, tools; NOT
  the ``allowed_tools`` list, owner decision 2026-10-09), ``can_execute``'s decisions are unchanged, and
  ``approved_action_blocker`` refuses an approved action on it (owner decision S7);
- ``loop_planning.drain_approved``'s edges: an empty queue, an entry with no action, a Proposal with no source,
  and the follow-up it returns.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from maxim.agents.autonomy import (
    AutonomyController,
    AutonomyLevel,
    Proposal,
    ProposalQueue,
    SafetyConstraints,
    SupervisionPolicy,
)
from maxim.runtime import loop_planning
from maxim.runtime.loop_types import ActionFollowup


def _proposal(pid: str, tool: str | None = "probe", **kw: Any) -> Proposal:
    return Proposal(
        id=pid,
        action={"tool_name": tool, "params": {}} if tool is not None else None,
        reasoning="r",
        confidence=0.7,
        **kw,
    )


# ── approval_blocker ─────────────────────────────────────────────────────────────────────────────────


def test_can_execute_action_takes_its_non_level_refusal_from_approval_blocker(monkeypatch):
    """Structural: ``can_execute_action`` returns whatever ``approval_blocker`` says, so a non-level check that
    lived only in ``can_execute_action`` would not be seen at drain time. Deleting the call fails this."""
    ctrl = AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS)
    seen: list[Any] = []

    def _blocker(action: dict[str, Any], *, level: AutonomyLevel | None = None) -> str | None:
        seen.append(level)
        return "probe: blocked"

    monkeypatch.setattr(ctrl, "approval_blocker", _blocker)
    assert ctrl.can_execute_action({"tool_name": "respond"}) == (False, "probe: blocked")
    assert seen == [AutonomyLevel.AUTONOMOUS]  # the level it read, passed through


def test_a_halted_controller_blocks_until_resumed():
    """``emergency_halt`` (PLANNING + paused) is the in-tree pause: an approval does not run through it."""
    ctrl = AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS)
    ctrl.emergency_halt("probe")
    assert ctrl.approval_blocker({"tool_name": "respond"}) == "Execution is paused"
    assert ctrl.can_execute_action({"tool_name": "respond"}) == (False, "Execution is paused")
    ctrl.resume()
    assert ctrl.approval_blocker({"tool_name": "respond"}) is None


def test_a_forbidden_tool_is_blocked_even_when_always_allowed():
    ctrl = AutonomyController(
        initial_level=AutonomyLevel.PLANNING,
        safety_constraints=SafetyConstraints(forbidden_tools=frozenset({"search_code"})),
    )
    assert "search_code" in AutonomyController.ALWAYS_ALLOWED_TOOLS
    assert ctrl.approval_blocker({"tool_name": "search_code"}) == "Tool 'search_code' is forbidden"
    assert ctrl.can_execute_action({"tool_name": "search_code"}) == (False, "Tool 'search_code' is forbidden")


def test_approval_lifts_only_the_planning_level():
    """At PLANNING an unforbidden tool is refused for its LEVEL only: ``approval_blocker`` has nothing against it."""
    ctrl = AutonomyController(initial_level=AutonomyLevel.PLANNING)
    can, reason = ctrl.can_execute_action({"tool_name": "probe"})
    assert (can, reason) == (False, "PLANNING mode requires human approval for all actions")
    assert ctrl.approval_blocker({"tool_name": "probe"}) is None


# ── ProposalQueue.pop_approved ───────────────────────────────────────────────────────────────────────


def test_pop_approved_takes_the_oldest_approved_one_at_a_time():
    q = ProposalQueue()
    for pid in ("a", "b", "c"):
        q.submit(_proposal(pid))
    assert q.pop_approved() is None
    assert q.approve("c") and q.approve("a")
    first = q.pop_approved()
    assert first is not None and first.id == "a" and first.status == "approved"
    second = q.pop_approved()
    assert second is not None and second.id == "c"
    assert q.pop_approved() is None
    assert [p.id for p in q.get_pending()] == ["b"]


# ── drain_approved ───────────────────────────────────────────────────────────────────────────────────


class _Sim:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, category: str, msg: str, data: dict | None = None) -> None:
        self.lines.append(f"{category}:{msg}")


def _drain(ctrl: AutonomyController, eal: Any, booked: list[dict[str, Any]]) -> Any:
    return loop_planning.drain_approved(
        autonomy_controller=ctrl,
        execute_and_learn=eal,
        book_machine_refusal=lambda **kw: booked.append(kw),
        observation={"tick": 1},
        state=SimpleNamespace(data={"mode": "active"}),
        sim=_Sim(),
    )


def _fu(tool: str) -> ActionFollowup:
    return ActionFollowup(
        tool=tool, result="r", original_query="q", followup_type="process", mode="active", timestamp=0
    )


def test_an_empty_queue_drains_nothing():
    calls: list[Any] = []
    assert _drain(AutonomyController(), lambda **kw: calls.append(kw), []) is None
    assert calls == []


def test_an_approved_entry_with_no_action_is_refused_typed_and_not_booked():
    ctrl = AutonomyController()
    p = _proposal("x", tool=None)
    ctrl.proposal_queue.submit(p)
    ctrl.proposal_queue.approve("x")
    calls: list[Any] = []
    booked: list[dict[str, Any]] = []
    assert _drain(ctrl, lambda **kw: calls.append(kw), booked) is None
    assert calls == [] and booked == []
    assert (p.status, p.rejected_reason) == ("rejected", "no_action")
    [audit] = [e for e in ctrl.get_audit_log() if e.action_type == "rejected"]
    assert audit.human_involved is False


def test_a_proposal_with_no_source_is_its_own_record():
    """Built outside the loop (no ``LLMProposal`` behind it): it executes, keyed to no situation."""
    ctrl = AutonomyController()
    p = _proposal("x")
    ctrl.proposal_queue.submit(p)
    ctrl.proposal_queue.approve("x")
    calls: list[dict[str, Any]] = []

    def _eal(**kw: Any) -> Any:
        calls.append(kw)
        return SimpleNamespace(followup=None)

    _drain(ctrl, _eal, [])
    [kw] = calls
    assert kw["proposal"] is p and kw["human_involved"] is True and kw["confidence"] == 0.7
    assert kw["observation"] == {"tick": 1}


def test_the_drain_returns_the_last_follow_up_and_keys_each_to_its_source():
    ctrl = AutonomyController()
    sources = {pid: SimpleNamespace(name=f"src-{pid}") for pid in ("a", "b", "c")}
    for pid in ("a", "b", "c"):
        ctrl.proposal_queue.submit(_proposal(pid, tool=f"t_{pid}", source=sources[pid]))
        ctrl.proposal_queue.approve(pid)
    fus = {"t_a": _fu("t_a"), "t_b": None, "t_c": None}
    seen: list[Any] = []

    def _eal(**kw: Any) -> Any:
        seen.append(kw["proposal"])
        return SimpleNamespace(followup=fus[kw["action"]["tool_name"]])

    assert _drain(ctrl, _eal, []) is fus["t_a"]  # a later None does not erase an earlier follow-up
    assert seen == [sources["a"], sources["b"], sources["c"]]


# ── SupervisionPolicy.hard_deny / AutonomyController.approved_action_blocker (owner decision S7) ─────────

_POLICY = dict(
    allowed_tools={"probe", "write_file", "execute_file", "slow_tool", "confirm_me", "pre_x", "cat_tool"},
    forbidden_tools={"banned"},
    forbidden_prefixes=("pre_",),
    forbidden_categories=frozenset({"net"}),
    requires_confirmation={"confirm_me"},
)

# (action, confidence) -> can_execute's (decision, message), as on ``main`` before hard_deny was extracted (the
# same table passes against the pre-#1085 code). Order matters: a prefix beats a category beats a forbid beats
# the allowed list beats confidence beats confirmation beats the write/execute rules.
_MATRIX = [
    ({"tool_name": "probe"}, 0.9, (True, None)),
    ({"tool_name": "pre_x"}, 0.9, (False, "Tool 'pre_x' blocked by prefix rule: pre_")),
    ({"tool_name": "cat_tool", "category": "net"}, 0.9, (False, "Tool 'cat_tool' blocked by category: net")),
    ({"tool_name": "banned"}, 0.9, (False, "Tool 'banned' is forbidden")),
    ({"tool_name": "elsewhere"}, 0.9, (False, "Tool 'elsewhere' requires approval")),
    ({"tool_name": "slow_tool"}, 0.1, (False, "Confidence 0.10 below threshold")),
    ({"tool_name": "confirm_me"}, 0.9, (False, "Tool 'confirm_me' requires confirmation")),
    (
        {"tool_name": "write_file", "params": {"path": "/etc/x"}},
        0.9,
        (False, "CWD write requires approval in supervised mode"),
    ),
    ({"tool_name": "execute_file", "params": {}}, 0.9, (False, "Sandbox execution requires approval")),
]


def test_supervision_policy_decisions_are_unchanged_by_the_hard_deny_extraction():
    policy = SupervisionPolicy(**_POLICY)
    assert [policy.can_execute(a, confidence=c) for a, c, _ in _MATRIX] == [want for _, _, want in _MATRIX]


def test_can_execute_takes_its_hard_denials_from_hard_deny(monkeypatch):
    """Structural: deleting ``can_execute``'s ``hard_deny`` call (or re-copying the checks) fails this."""
    policy = SupervisionPolicy()
    monkeypatch.setattr(policy, "hard_deny", lambda action: "probe: denied")
    assert policy.can_execute({"tool_name": "probe"}, confidence=1.0) == (False, "probe: denied")


def test_an_approved_action_is_blocked_by_hard_denials_but_not_by_needs_approval_checks():
    ctrl = AutonomyController(initial_level=AutonomyLevel.PLANNING, supervision_policy=SupervisionPolicy(**_POLICY))
    blocked = {a["tool_name"]: ctrl.approved_action_blocker(a) for a, _, _ in _MATRIX}
    assert blocked == {
        "probe": None,
        "pre_x": "Tool 'pre_x' blocked by prefix rule: pre_",
        "cat_tool": "Tool 'cat_tool' blocked by category: net",
        "banned": "Tool 'banned' is forbidden",
        # "needs approval" checks: the approval is what satisfies them (allowed_tools included, 2026-10-09)
        "elsewhere": None,
        "slow_tool": None,
        "confirm_me": None,
        "write_file": None,
        "execute_file": "Tool 'execute_file' is forbidden",  # the DEFAULT SafetyConstraints forbid, not the policy
    }
    ctrl.emergency_halt("probe")  # approval_blocker still comes first
    assert ctrl.approved_action_blocker({"tool_name": "banned"}) == "Execution is paused"


def test_can_execute_action_ignores_the_policy_outside_supervised():
    """The S7 fold does not move ``can_execute_action``: at AUTONOMOUS the policy's denials do not apply."""
    ctrl = AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS, supervision_policy=SupervisionPolicy(**_POLICY))
    assert ctrl.can_execute_action({"tool_name": "banned"}) == (True, None)


def test_a_raise_anywhere_in_an_entry_aborts_the_drain_and_keeps_the_original(monkeypatch):
    """Executor S1: the blocker itself raising (not only the execution) aborts the drain: the rest are rejected
    ``drain_aborted`` even when reporting them raises, and the ORIGINAL exception propagates."""
    ctrl = AutonomyController()
    entries = [_proposal(pid, tool=f"t_{pid}") for pid in ("a", "b", "c")]
    for p in entries:
        ctrl.proposal_queue.submit(p)
        ctrl.proposal_queue.approve(p.id)

    def _boom(action: dict[str, Any]) -> str | None:
        raise LookupError("probe: blocker broke")

    monkeypatch.setattr(ctrl, "approved_action_blocker", _boom)

    def _book(**kw: Any) -> None:
        raise ValueError("probe: booking broke")

    try:
        loop_planning.drain_approved(
            autonomy_controller=ctrl,
            execute_and_learn=lambda **kw: None,
            book_machine_refusal=_book,
            observation={},
            state=SimpleNamespace(data={}),
            sim=_Sim(),
        )
    except LookupError as exc:
        assert "blocker broke" in str(exc)
    else:
        raise AssertionError("the drain swallowed the raise")
    assert [(p.status, p.rejected_reason) for p in entries[1:]] == [("rejected", "drain_aborted")] * 2
    assert ctrl.proposal_queue.pop_approved() is None
