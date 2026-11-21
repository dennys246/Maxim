"""#1185: the gating prompts read refusals as approval, and three handlers kept three vocabularies.

``detect_approval_intent`` tested YES words before NO words with a word-set intersection, so "stop, don't run
it" (the token "run"), "not ok", "that is not fine" and "no good" came out ``approve`` on live plan-approval
input; "No." came out ``unknown``; and mixed replies flipped with ``PYTHONHASHSEED`` (``set.pop()``).
``handle_confirmation`` and ``handle_timeout_retry`` kept their own yes/no sets, so "No." at a confirmation became
a modification (the refusal never booked) and "Yes." at a timeout prompt fell through.

Owner decision (2026-11-21), the issue's narrow vocabulary: only a whole reply of {yes, y, approve} is yes, only
{no, n, reject, cancel, stop} is no; anything else carrying an approval word, a negation, a '?' or a symbol
re-asks. One reader, ``approval.classify_yes_no``, serves every gating prompt.
"""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from maxim.agents.llm_types import LLMProposal
from maxim.runtime.approval import classify_yes_no, detect_approval_intent

_CASES = [
    # The false approvals #1185 reported, and its probe list: all re-ask.
    ("stop, don't run it", "unknown"),
    ("not ok", "unknown"),
    ("that is not fine", "unknown"),
    ("no good", "unknown"),
    ("not good", "unknown"),
    ("don't run", "unknown"),
    ("not sure", "unknown"),
    ("no, go back", "unknown"),
    # Plain approvals: only {yes, y, approve}.
    ("yes", "approve"),
    ("Yes!", "approve"),
    ("YES.", "approve"),
    ("  y  ", "approve"),
    ("approve", "approve"),
    ('"yes"', "approve"),
    # Former approvals that now re-ask.
    ("ok", "unknown"),
    ("OK.", "unknown"),
    ("sure", "unknown"),
    ("go ahead", "unknown"),
    ("lgtm", "unknown"),
    ("yes please", "unknown"),
    ("fine", "unknown"),
    # Plain rejections: only {no, n, reject, cancel, stop}.
    ("no", "reject"),
    ("No.", "reject"),
    ("no!", "reject"),
    ("stop", "reject"),
    ("Cancel", "reject"),
    ("reject", "reject"),
    ("n", "reject"),
    # Negations that are not a plain no re-ask (a refusal books NAc; only a human "no" may).
    ("don't", "unknown"),
    ("Don’t", "unknown"),
    ("never", "unknown"),
    ("not yet", "unknown"),
    ("why not?", "unknown"),
    ("no problem", "unknown"),
    ("not a problem", "unknown"),
    ("don't stop", "unknown"),
    ("wait", "unknown"),
    ("hold on", "unknown"),
    # Mixed.
    ("ok but don't run it", "unknown"),
    ("yes no", "unknown"),
    ("no yes", "unknown"),
    ("yes, but don't", "unknown"),
    ("yes the answer is surely no", "unknown"),
    ("i can't approve that", "unknown"),
    ("No, no.", "unknown"),
    # Questions never approve and never reject.
    ("yes?", "unknown"),
    ("sure?", "unknown"),
    ("ok?", "unknown"),
    ("no?", "unknown"),
    ("what does it do?", "unknown"),
    # Symbols and emoticons are not a whole-phrase match.
    ("\U0001f44e", "unknown"),
    ("❌", "unknown"),
    ("\U0001f6d1", "unknown"),
    ("⛔", "unknown"),
    ("\U0001f44d", "unknown"),
    ("yes \U0001f44d", "unknown"),
    ("no \U0001f44e", "unknown"),
    (":(", "unknown"),
    ("yes :(", "unknown"),
    ("no :(", "unknown"),
    # Free text with no approval word, no negation, no '?' and no symbol is a modification (the plan is
    # dropped and the LLM sees the text; it never executes the plan).
    ("write the file to the other folder", "modify"),
    ("write it to /tmp/out.txt instead", "modify"),
    ("add a header", "modify"),
    # No signal, too short to be a request.
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


@pytest.mark.parametrize(("text", "expected"), _CASES)
def test_classify_yes_no_is_the_yes_no_half(text: str, expected: str) -> None:
    """``classify_yes_no`` is the one yes/no reading the three handlers share; ``detect_approval_intent``
    approves and rejects exactly where it says yes and no."""
    assert classify_yes_no(text) == {"approve": "yes", "reject": "no"}.get(expected)


def test_classify_yes_no_handles_none() -> None:
    assert classify_yes_no(None) is None
    assert detect_approval_intent(None) == ("unknown", None)


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


# ── The live callers ──────────────────────────────────────────────────────────


def _controller():
    from maxim.runtime.loop_controller import LoopController

    from maxim.runtime.executor import Executor

    state = MagicMock()
    state.data = {"mode": "live"}
    executor = MagicMock(spec=Executor)
    executor.effective_operational_mode.return_value = "live"
    return LoopController(
        agent=MagicMock(),
        environment=MagicMock(),
        state=state,
        memory=MagicMock(),
        decision_engine=MagicMock(),
        executor=executor,
        autonomy_controller=MagicMock(),
    )


def _proposal() -> LLMProposal:
    return LLMProposal(
        request_id="r-1185",
        action={"tool_name": "bash", "params": {"command": "rm -rf build"}},
        reasoning="clean the build",
        strategy_used=None,
        confidence=0.5,
        mode_goal_achieved=False,
        plan_text="delete the build dir",
        requires_approval=True,
    )


def _plan_controller():
    ctrl = _controller()
    ctrl.pending_plan_proposal = _proposal()
    return ctrl


@pytest.mark.parametrize(
    "text", ["stop, don't run it", "not ok", "that is not fine", "no good", "No.", "yes no", "yes?", "ok", "\U0001f44e"]
)
def test_the_plan_handler_never_executes_a_non_yes(text: str) -> None:
    """Through the real ``LoopController.handle_plan_approval``: anything but a plain yes never promotes the plan
    to ``pending_proposal`` (the slot the loop executes)."""
    ctrl = _plan_controller()
    with patch("maxim.simulation.sim_logger.display_status"):
        ctrl.handle_plan_approval(text, book_refusal=MagicMock())
    assert ctrl.pending_proposal is None


def test_the_plan_handler_reasks_on_unknown() -> None:
    """An unknown reply is consumed (True, the pending input slot cleared), the plan stays pending, no refusal is
    booked, and the question is shown again on the abstraction stream and the status line."""
    ctrl = _plan_controller()
    ctrl.state.data["pending_user_input"] = "not ok"
    book = MagicMock()
    with (
        patch("maxim.simulation.sim_logger.display_status") as status,
        patch("maxim.runtime.loop_controller.log_agentic") as log,
    ):
        assert ctrl.handle_plan_approval("not ok", book_refusal=book) is True
    assert ctrl.pending_proposal is None
    assert ctrl.pending_plan_proposal is not None
    assert "pending_user_input" not in ctrl.state.data
    book.assert_not_called()
    assert "Awaiting approval" in status.call_args.args[0]
    assert any(c.args[1] == "plan_awaiting_approval" and c.args[2].get("reask") is True for c in log.call_args_list), (
        log.call_args_list
    )


def test_the_plan_handler_executes_a_plain_yes() -> None:
    ctrl = _plan_controller()
    plan = ctrl.pending_plan_proposal
    assert ctrl.handle_plan_approval("Yes.", book_refusal=MagicMock()) is True
    assert ctrl.pending_proposal is plan


def test_the_plan_handler_books_a_plain_no() -> None:
    ctrl = _plan_controller()
    book = MagicMock()
    assert ctrl.handle_plan_approval("No.", book_refusal=book) is True
    assert ctrl.pending_plan_proposal is None
    assert book.call_args.kwargs["tool_name"] == "bash"


def test_a_plan_modification_reaches_the_next_submit() -> None:
    """The modify branch wrote ``plan_modification_context``, which nothing read. It now writes
    ``pending_modification``, the shape ``handle_confirmation`` writes and the §6 submit pops into the prompt."""
    ctrl = _plan_controller()
    assert ctrl.handle_plan_approval("write the file to the other folder", book_refusal=MagicMock()) is False
    mod = ctrl.state.data["pending_modification"]
    assert mod["user_modification"] == "write the file to the other folder"
    assert mod["original_tool_name"] == "bash"
    assert mod["original_action"]["params"] == {"command": "rm -rf build"}
    assert "plan_modification_context" not in ctrl.state.data
    assert ctrl.pending_plan_proposal is None


def _confirm_controller():
    from maxim.runtime.loop_types import PendingConfirmation

    ctrl = _controller()
    ctrl.pending_confirmation = PendingConfirmation.from_proposal(_proposal())
    return ctrl


def _confirm(ctrl, text: str):
    book, run = MagicMock(), MagicMock()
    with patch("maxim.simulation.sim_logger.display_status"):
        consumed = ctrl.handle_confirmation(text, execute_and_learn=run, book_refusal=book, observation={})
    return consumed, run, book


def test_a_plain_no_at_a_confirmation_books_the_refusal() -> None:
    """Before: "No." was not in the confirmation's own set, so it became a modification and no refusal booked."""
    ctrl = _confirm_controller()
    consumed, run, book = _confirm(ctrl, "No.")
    assert consumed is True
    run.assert_not_called()
    book.assert_called_once()
    assert "pending_modification" not in ctrl.state.data


@pytest.mark.parametrize("text", ["not ok", "ok", "sure?", "yes no", "❌"])
def test_an_ambiguous_confirmation_reasks(text: str) -> None:
    ctrl = _confirm_controller()
    consumed, run, book = _confirm(ctrl, text)
    assert consumed is True
    run.assert_not_called()
    book.assert_not_called()
    assert ctrl.pending_confirmation is not None
    assert "pending_modification" not in ctrl.state.data


def test_a_confirmation_yes_executes() -> None:
    ctrl = _confirm_controller()
    consumed, run, _ = _confirm(ctrl, "Yes!")
    assert consumed is True
    run.assert_called_once()


def test_a_confirmation_modification_is_kept() -> None:
    ctrl = _confirm_controller()
    consumed, run, book = _confirm(ctrl, "write the file to the other folder")
    assert consumed is True
    run.assert_not_called()
    book.assert_not_called()
    assert ctrl.state.data["pending_modification"]["user_modification"] == "write the file to the other folder"


def _timeout_controller():
    from maxim.runtime.loop_types import TimeoutRetry

    ctrl = _controller()
    ctrl.llm_worker = MagicMock()
    ctrl.set_timeout_retry(TimeoutRetry(original_request=object(), timeout_s=60.0))
    return ctrl


def test_a_punctuated_yes_at_a_timeout_prompt_retries() -> None:
    """Before: the timeout handler matched the raw lowercased text, so "Yes." fell through to normal processing."""
    ctrl = _timeout_controller()
    with patch("maxim.simulation.sim_logger.display_status"):
        assert ctrl.handle_timeout_retry("Yes.") is True
    assert ctrl.llm_worker.retry_with_timeout.call_args.args[1] == 120.0


def test_a_punctuated_no_at_a_timeout_prompt_declines() -> None:
    ctrl = _timeout_controller()
    with patch("maxim.simulation.sim_logger.display_status"):
        assert ctrl.handle_timeout_retry("No!") is True
    ctrl.llm_worker.retry_with_timeout.assert_not_called()


@pytest.mark.parametrize("text", ["ok", "sure?", "not now"])
def test_an_ambiguous_timeout_answer_does_not_retry(text: str) -> None:
    ctrl = _timeout_controller()
    ctrl.handle_timeout_retry(text)
    ctrl.llm_worker.retry_with_timeout.assert_not_called()


# ── #1723 (narrow half): a failing sim response policy never approves ─────────


def _bridge_adapter(policy):
    from maxim.runtime.sim_adapter import SimulationAdapter
    from maxim.simulation.bridge import SimulationBridge

    bridge = SimulationBridge.__new__(SimulationBridge)
    bridge.response_policy = policy
    return SimulationAdapter(percept_source=bridge)


@pytest.mark.parametrize(
    ("method", "arg"),
    [("resolve_plan_approval", "plan"), ("resolve_timeout_retry", 60.0), ("resolve_confirmation", {})],
)
def test_a_raising_policy_never_yields_yes(method: str, arg) -> None:
    """``SimulationAdapter.resolve_*`` wrapped the policy call in ``except Exception: pass`` and then returned
    "yes", so any failure in the bridge's response policy approved. It now raises."""
    policy = MagicMock()
    getattr(policy, method).side_effect = RuntimeError("policy broke")
    adapter = _bridge_adapter(policy)
    with pytest.raises(RuntimeError, match="policy broke"):
        getattr(adapter, method)(arg)


def test_a_working_policy_still_answers() -> None:
    policy = MagicMock()
    policy.resolve_plan_approval.return_value = "no"
    assert _bridge_adapter(policy).resolve_plan_approval("plan") == "no"
