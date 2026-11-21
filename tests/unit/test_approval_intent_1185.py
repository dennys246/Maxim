"""#1185: ``detect_approval_intent`` read negated and rejecting replies as APPROVAL on live plan-approval input.

The old parser tested YES words before NO words with a word-set intersection, so "stop, don't run it" (the
token "run"), "not ok", "that is not fine" and "no good" all came out ``approve``; "No." came out ``unknown``
(punctuation kept the token from matching), and mixed replies flipped with ``PYTHONHASHSEED`` (``set.pop()``).

The gate must fail closed: approval needs the WHOLE reply (punctuation stripped) to be an approval phrase; a
rejection or negation word anywhere means it is never approval; contradictory replies are ``unknown`` (the
caller re-asks).
"""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from maxim.agents.llm_types import LLMProposal
from maxim.runtime.approval import detect_approval_intent

_CASES = [
    # The false approvals #1185 reported (and its probe list).
    ("stop, don't run it", "reject"),
    ("not ok", "unknown"),
    ("that is not fine", "unknown"),
    ("no good", "unknown"),
    ("not good", "unknown"),
    ("don't run", "reject"),
    ("not sure", "unknown"),
    ("no, go back", "unknown"),
    # Plain approvals.
    ("yes", "approve"),
    ("Yes!", "approve"),
    ("ok", "approve"),
    ("OK.", "approve"),
    ("go ahead", "approve"),
    ("Go ahead.", "approve"),
    ("approve", "approve"),
    ("lgtm", "approve"),
    ("  y  ", "approve"),
    # Plain rejections.
    ("no", "reject"),
    ("No.", "reject"),
    ("no!", "reject"),
    ("stop", "reject"),
    ("don't", "reject"),
    ("Don’t", "reject"),
    ("cancel", "reject"),
    # Mixed or negated.
    ("ok but don't run it", "unknown"),
    ("not yet", "reject"),
    ("yes no", "unknown"),
    ("no yes", "unknown"),
    ("yes, but don't", "unknown"),
    ("yes the answer is surely no", "unknown"),
    ("i can't approve that", "unknown"),
    ("never", "reject"),
    # Approval words inside a longer reply are never approval.
    ("yes please", "approve"),
    ("yes sir", "unknown"),
    ("yes but write it to /tmp instead", "modify"),
    ("run", "unknown"),
    ("fine", "unknown"),
    # A long negated reply is a modification request (the plan is dropped, the LLM sees the text), never approval.
    ("please don't delete the logs, write them to a temp dir instead", "modify"),
    # No polarity at all.
    ("write it to the docs folder instead", "modify"),
    ("hmm", "unknown"),
    ("", "unknown"),
    ("   ", "unknown"),
    ("?!", "unknown"),
]


@pytest.mark.parametrize(("text", "expected"), _CASES)
def test_detect_approval_intent_table(text: str, expected: str) -> None:
    intent, modification = detect_approval_intent(text)
    assert intent == expected, f"{text!r} -> {intent!r}, expected {expected!r}"
    if intent == "modify":
        assert modification == text
    else:
        assert modification is None


def test_only_whole_approval_phrases_approve() -> None:
    """Fail-closed invariant over the whole table: every input that approves is, punctuation stripped, an
    exact approval phrase, and none of them carries a negation."""
    from maxim.runtime.approval import _APPROVE_PHRASES, _normalize

    for text, _ in _CASES:
        if detect_approval_intent(text)[0] == "approve":
            assert " ".join(_normalize(text)) in _APPROVE_PHRASES


def test_mixed_replies_do_not_depend_on_the_hash_seed() -> None:
    """The old parser popped an arbitrary element of a word set, so a mixed reply's intent varied by process."""
    src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    code = (
        "from maxim.runtime.approval import detect_approval_intent as d;"
        "print([d(t)[0] for t in ('yes the answer is surely no', 'no the answer is surely yes', 'yes no', 'hmm stop')])"
    )
    outs = set()
    for seed in ("0", "1", "2", "3", "4", "5", "6", "7"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": os.path.abspath(src)}
        outs.add(
            subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
        )
    assert len(outs) == 1, outs


def _plan_controller():
    from maxim.runtime.loop_controller import LoopController

    state = MagicMock()
    state.data = {"mode": "live"}
    ctrl = LoopController(
        agent=MagicMock(),
        environment=MagicMock(),
        state=state,
        memory=MagicMock(),
        decision_engine=MagicMock(),
        executor=MagicMock(),
        autonomy_controller=MagicMock(),
    )
    ctrl.pending_plan_proposal = LLMProposal(
        request_id="r-1185",
        action={"tool_name": "bash", "params": {"command": "rm -rf build"}},
        reasoning="plan",
        strategy_used=None,
        confidence=0.5,
        mode_goal_achieved=False,
        plan_text="delete the build dir",
        requires_approval=True,
    )
    return ctrl


@pytest.mark.parametrize("text", ["stop, don't run it", "not ok", "that is not fine", "no good", "No.", "yes no"])
def test_the_live_caller_never_executes_a_negated_reply(text: str) -> None:
    """Through the real ``LoopController.handle_plan_approval``: a negated reply never promotes the plan to
    ``pending_proposal`` (the slot the loop executes)."""
    ctrl = _plan_controller()
    book = MagicMock()
    ctrl.handle_plan_approval(text, book_refusal=book)
    assert ctrl.pending_proposal is None


def test_the_live_caller_reasks_on_unknown() -> None:
    ctrl = _plan_controller()
    book = MagicMock()
    assert ctrl.handle_plan_approval("not ok", book_refusal=book) is True
    assert ctrl.pending_proposal is None
    assert ctrl.pending_plan_proposal is not None  # still awaiting an answer
    assert "Awaiting approval" in ctrl.state.data["pending_plan_text"]
    book.assert_not_called()


def test_the_live_caller_executes_a_plain_yes() -> None:
    ctrl = _plan_controller()
    plan = ctrl.pending_plan_proposal
    assert ctrl.handle_plan_approval("Yes.", book_refusal=MagicMock()) is True
    assert ctrl.pending_proposal is plan
