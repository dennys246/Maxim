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
from maxim.utils.input_source import InputSource

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
        autonomy_controller=MagicMock(**{"approved_action_denial.return_value": None}),
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


def _answer(ctrl, text: str, book=None, run=None) -> bool:
    """``handle_plan_approval`` with the run's callables faked: ``run`` is ``execute_and_learn``."""
    return ctrl.handle_plan_approval(
        text,
        book_refusal=book if book is not None else MagicMock(),
        execute_and_learn=run if run is not None else MagicMock(return_value=MagicMock(followup=None)),
        book_machine_refusal=MagicMock(),
        observation={},
        sim=MagicMock(),
        answer_source=InputSource.OPERATOR,
    )


@pytest.mark.parametrize(
    "text", ["stop, don't run it", "not ok", "that is not fine", "no good", "No.", "yes no", "yes?", "ok", "\U0001f44e"]
)
def test_the_plan_handler_never_executes_a_non_yes(text: str) -> None:
    """Through the real ``LoopController.handle_plan_approval``: anything but a plain yes never executes the plan
    nor promotes it to ``pending_proposal`` (the slot the loop executes)."""
    ctrl = _plan_controller()
    run = MagicMock()
    with patch("maxim.simulation.sim_logger.display_status"):
        _answer(ctrl, text, run=run)
    run.assert_not_called()
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
        assert _answer(ctrl, "not ok", book=book) is True
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
    run = MagicMock(return_value=MagicMock(followup=None))
    assert _answer(ctrl, "Yes.", run=run) is True
    run.assert_called_once()
    assert run.call_args.kwargs["proposal"] is plan and run.call_args.kwargs["human_involved"] is True
    assert ctrl.pending_proposal is None and ctrl.pending_plan_proposal is None


def test_the_plan_handler_books_a_plain_no() -> None:
    ctrl = _plan_controller()
    book = MagicMock()
    assert _answer(ctrl, "No.", book=book) is True
    assert ctrl.pending_plan_proposal is None
    assert book.call_args.kwargs["tool_name"] == "bash"


def test_a_plan_modification_reaches_the_next_submit() -> None:
    """The modify branch wrote ``plan_modification_context``, which nothing read. It now writes
    ``pending_modification``, the shape ``handle_confirmation`` writes and the §6 submit pops into the prompt."""
    ctrl = _plan_controller()
    assert _answer(ctrl, "write the file to the other folder") is True  # consumed: the LLM reads it once
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
        consumed = ctrl.handle_confirmation(
            text,
            execute_and_learn=run,
            book_refusal=book,
            observation={},
            sim=MagicMock(),
            book_machine_refusal=MagicMock(),
            answer_source=InputSource.OPERATOR,
        )
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
        assert ctrl.handle_timeout_retry("Yes.", sim=MagicMock()) is True
    assert ctrl.llm_worker.retry_with_timeout.call_args.args[1] == 120.0


def test_a_punctuated_no_at_a_timeout_prompt_declines() -> None:
    ctrl = _timeout_controller()
    with patch("maxim.simulation.sim_logger.display_status"):
        assert ctrl.handle_timeout_retry("No!", sim=MagicMock()) is True
    ctrl.llm_worker.retry_with_timeout.assert_not_called()


@pytest.mark.parametrize("text", ["ok", "sure?", "not now"])
def test_an_ambiguous_timeout_answer_does_not_retry(text: str) -> None:
    ctrl = _timeout_controller()
    with patch("maxim.simulation.sim_logger.display_status"):
        assert ctrl.handle_timeout_retry(text, sim=MagicMock()) is True  # consumed: re-asked (decision C)
    assert ctrl.get_timeout_retry() is not None  # still pending
    ctrl.llm_worker.retry_with_timeout.assert_not_called()


# ── #1723 (narrow half): a failing sim response policy never approves ─────────


def _bridge_adapter(policy):
    """The adapter exactly as production builds it (#1185 owner decision A): ``loop_setup._build_sim_adapter`` from
    the bridge's PERCEPT SOURCE (never the bridge) and its response policy, passed explicitly."""
    from maxim.runtime.loop_setup import _build_sim_adapter
    from maxim.simulation.bridge import SimulationBridge

    bridge = SimulationBridge(response_policy=policy)
    return _build_sim_adapter(
        percept_source=bridge.percept_source,
        response_policy=bridge.response_policy,
        action_sink=bridge.action_sink,
        pain_bus=None,
        sim_adapter=None,
        executor=None,
    )


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


# ── Delta-round folds ─────────────────────────────────────────────────────────


def test_a_tool_follow_up_is_never_an_answer() -> None:
    """Fold 3: an ``[ACTION_FOLLOWUP`` arriving while a plan waits is a tool result, not the person's answer."""
    ctrl = _plan_controller()
    run = MagicMock()
    text = "[ACTION_FOLLOWUP type=process tool=read_file mode=active query='x']: yes"
    assert (
        ctrl.answer_pending_prompt(
            text,
            execute_and_learn=run,
            book_refusal=MagicMock(),
            book_machine_refusal=MagicMock(),
            observation={},
            sim=MagicMock(),
            answer_source=InputSource.OPERATOR,
        )
        is False
    )
    run.assert_not_called()
    assert ctrl.pending_plan_proposal is not None


def test_the_loop_reply_answers_first_and_the_observation_waits() -> None:
    """Fold 2: a sim confirmation answered "yes" (the loop's own reply) while the narrator's sentence arrives in the
    same observation. The reply used to be dropped, so the sentence came out "unknown", was consumed by the re-ask,
    and the confirmation stayed parked forever. Now the reply answers first and the sentence is the next tick's."""
    from maxim.runtime.loop_perception import take_user_input

    sentence = "Great, can you also look at the logs?"
    state = MagicMock()
    state.data = {"pending_loop_reply": "yes"}
    agent = MagicMock()
    agent.publish_arrival.return_value = "cap-1"
    first = take_user_input({"cli_input": sentence, "source": "cli"}, state, agent, sim_mode=True)
    assert first.cli_input == "yes"
    assert first.addressed == "" and first.source is InputSource.LOOP  # the loop's own reply: no one said it
    second = take_user_input({}, state, agent, sim_mode=True)
    # Wire item 3: the deferred turn keeps what it had on arrival, so it enriches and recalls like an undeferred one.
    assert (second.cli_input, second.addressed, second.own_capture_id) == (sentence, sentence, "cap-1")
    assert second.source is InputSource.SIM
    assert agent.publish_arrival.call_count == 1  # memorized once, as it arrived
    assert "pending_deferred_input" not in state.data


def test_the_sentence_after_the_reply_wedges_nothing() -> None:
    """Fold 2 end to end at the controller: the reply confirms, then the sentence is an ordinary turn."""
    from maxim.runtime.loop_perception import take_user_input

    ctrl = _confirm_controller()
    ctrl.state.data["pending_loop_reply"] = "yes"
    first = take_user_input(
        {"cli_input": "Great, can you also look at the logs?"}, ctrl.state, MagicMock(), sim_mode=False
    )
    consumed, run, _ = _confirm(ctrl, first.cli_input)
    assert consumed is True and ctrl.pending_confirmation is None
    run.assert_called_once()
    second = take_user_input({}, ctrl.state, MagicMock(), sim_mode=False)
    assert second.cli_input == "Great, can you also look at the logs?"


def test_a_reask_reaches_the_sim_channel() -> None:
    ctrl = _confirm_controller()
    sim = MagicMock()
    with patch("maxim.simulation.sim_logger.display_status"):
        ctrl.handle_confirmation(
            "not ok",
            execute_and_learn=MagicMock(),
            book_refusal=MagicMock(),
            observation={},
            sim=sim,
            book_machine_refusal=MagicMock(),
            answer_source=InputSource.OPERATOR,
        )
    [call] = sim.emit_prompt.call_args_list
    assert "Type 'yes' or 'no'" in call.args[0]


def test_the_sim_adapter_puts_a_prompt_in_its_action_sink() -> None:
    from maxim.runtime.sim_adapter import NullSimulationAdapter, SimulationAdapter
    from maxim.simulation.sinks import RecordingSink

    sink = RecordingSink()
    SimulationAdapter(percept_source=None, action_sink=sink, response_policy=None).emit_prompt(
        "Confirm bash? Type 'yes' or 'no'."
    )
    [record] = sink.actions
    assert record.utterance == "Confirm bash? Type 'yes' or 'no'."
    assert NullSimulationAdapter().emit_prompt("x") is None


def test_the_reask_events_are_shown_at_the_prompt_level() -> None:
    from maxim.utils.structured_logging import EVENT_VERBOSITY

    level = EVENT_VERBOSITY["plan_awaiting_approval"]
    assert EVENT_VERBOSITY["confirmation_reask"] == EVENT_VERBOSITY["timeout_retry_reask"] == level == 0


def test_a_plan_prompt_shows_the_tool_and_params_that_will_run() -> None:
    from maxim.runtime.loop_controller import LoopController

    text = LoopController._plan_prompt(_proposal())
    assert "Run bash with" in text and '"command": "rm -rf build"' in text
    assert "The model's description: delete the build dir" in text


def test_the_timeout_prompt_offers_only_the_answers_it_reads() -> None:
    import inspect

    from maxim.agents import llm_worker

    src = inspect.getsource(llm_worker)
    assert "'no' to skip" not in src
    assert "or 'no'." in src


# ── Delta 3: synthetic records, input order, voice, wake, sources ─────────────────────────────────────


def _bridge_with(actions: int, *, reasks: int):
    import time as _time

    from maxim.runtime.sim_adapter import SimulationAdapter
    from maxim.simulation.bridge import SimulationBridge
    from maxim.simulation.sinks import ActionRecord

    bridge = SimulationBridge(
        response_timeout=0.5, settle_s=0.1, aut_mode="substrate-primary", substrate_actions_per_turn=3
    )
    adapter = SimulationAdapter(
        percept_source=bridge.percept_source, action_sink=bridge.action_sink, response_policy=bridge.response_policy
    )
    for i in range(actions):
        bridge.action_sink.record(ActionRecord(timestamp=_time.time(), tool_name="probe", result_success=i % 2 == 0))
        for _ in range(reasks):
            adapter.emit_prompt("Confirm probe? Type 'yes' or 'no'.")
    return bridge


def test_re_asks_do_not_change_a_runs_action_counts() -> None:
    """Fold 1: a re-asked prompt is recorded so the simulated person sees it, but it is not an action: the report's
    totals and tool counts, and the substrate budget, are the same with re-asks as without."""
    from maxim.simulation.report import build_report

    def _report(bridge):
        return build_report(
            goal="g", mode="m", bridge=bridge, duration_s=1.0, finish_reason="completed", started_at=0.0, provenance={}
        )

    plain, noisy = _bridge_with(2, reasks=0), _bridge_with(2, reasks=3)
    assert len(noisy.action_sink.actions) == 8  # the markers are recorded (the person reads them)
    a, b = _report(plain), _report(noisy)
    view = lambda r: (r.total_actions, r.blocked_actions, r.tool_usage, r.tool_success_rates)  # noqa: E731
    assert view(a) == view(b) == (2, 0, {"probe": 2}, {"probe": 0.5})
    assert noisy.substrate_action_allowed() is plain.substrate_action_allowed() is True  # 2 < 3 either way
    from maxim.simulation.sinks import ActionRecord

    assert ActionRecord(timestamp=0.0, tool_name="_gating_prompt").is_synthetic
    assert not ActionRecord(timestamp=0.0, tool_name="respond").is_synthetic
    assert not ActionRecord(timestamp=0.0, tool_name="_mcp_tool").is_synthetic  # by name, not by prefix (fold 7)


def test_deferred_inputs_are_taken_oldest_first() -> None:
    """Fold 2's probe: yes, yes, obs3, obs1, obs2 arrive as (reply+obs1), (reply+obs2), (obs3), (), (); the words
    come out in arrival order, never overtaken by newer input."""
    from maxim.runtime.loop_perception import take_user_input

    state = MagicMock()
    state.data = {}
    agent = MagicMock(**{"publish_arrival.return_value": None})
    taken = []
    for obs, reply in ((("obs1"), "yes"), ("obs2", "yes"), ("obs3", None), (None, None), (None, None)):
        if reply:
            state.data["pending_loop_reply"] = reply
        got = take_user_input({"cli_input": obs} if obs else {}, state, agent, sim_mode=False)
        taken.append(got.cli_input)
    assert taken == ["yes", "yes", "obs1", "obs2", "obs3"]


def test_a_voice_turn_that_arrives_with_typed_input_is_not_dropped() -> None:
    """Fold 3: voice behind typed input (or behind the loop's reply) used to be popped and dropped."""
    from maxim.runtime.loop_perception import take_user_input

    state = MagicMock()
    state.data = {"pending_voice_input": "v1", "pending_loop_reply": "yes"}
    agent = MagicMock(**{"publish_arrival.return_value": None})
    first = take_user_input({"cli_input": "typed"}, state, agent, sim_mode=False)
    second = take_user_input({}, state, agent, sim_mode=False)
    third = take_user_input({}, state, agent, sim_mode=False)
    assert [first.cli_input, second.cli_input, third.cli_input] == ["yes", "typed", "v1"]
    assert third.is_agentic_voice_input is True and third.source is InputSource.OPERATOR


def test_a_deferred_turn_recalls_like_an_undeferred_one() -> None:
    """Wire item 3: the tick that sends deferred words carries their ``addressed`` and ``own_capture_id`` (the
    enrichment context's cue and self-exclusion, #1447/#1573), exactly as if they had not waited."""
    from maxim.runtime.loop_perception import percept_enrichment_context, take_user_input

    def _ctx(user_input):
        state = MagicMock()
        state.data = {}
        return percept_enrichment_context(state, [], user_input)

    agent = MagicMock(**{"publish_arrival.return_value": "cap-9"})
    direct_state = MagicMock()
    direct_state.data = {}
    direct = take_user_input({"cli_input": "where is my cat"}, direct_state, agent, sim_mode=False)
    waited_state = MagicMock()
    waited_state.data = {"pending_loop_reply": "yes"}
    take_user_input({"cli_input": "where is my cat"}, waited_state, agent, sim_mode=False)
    waited = take_user_input({}, waited_state, agent, sim_mode=False)
    a, b = _ctx(direct), _ctx(waited)
    assert (a.user_input, a.own_capture_id) == (b.user_input, b.own_capture_id) == ("where is my cat", "cap-9")


def test_a_change_the_submit_cannot_read_is_told_not_kept_awake() -> None:
    """Fold 4: where the section 6 submit cannot read the change (``llm_reads_it`` False: substrate-primary, no
    worker, asleep) it is told on the prompts' surfaces, the sim channel included, and cleared."""
    ctrl = _plan_controller()
    ctrl.state.data["pending_modification"] = {"original_tool_name": "bash", "user_modification": "x"}
    sim = MagicMock()
    with patch("maxim.simulation.sim_logger.display_status"):
        assert ctrl.settle_pending_modification(llm_reads_it=False, sim=sim) is False
    assert "pending_modification" not in ctrl.state.data
    assert "can't apply a change" in sim.emit_prompt.call_args.args[0]


def test_asleep_the_loop_does_not_wake_for_a_change() -> None:
    """Fold 4: ``loop_gates`` passes ``llm_reads_it`` False while asleep (no wake keyword reaches the submit)."""
    import inspect

    from maxim.runtime import loop_gates

    src = inspect.getsource(loop_gates.pre_tick_gate)
    assert 'state.data.get("processing_state", "awake") != "sleep"' in src


def test_one_follow_up_predicate() -> None:
    """Fold 5: one test for a tool's ``[ACTION_FOLLOWUP`` line, at every site this branch touches."""
    import inspect

    from maxim.runtime import agent_loop, loop_controller, loop_perception
    from maxim.runtime.loop_perception import is_tool_followup

    assert (
        is_tool_followup("[ACTION_FOLLOWUP type=x]: y") and not is_tool_followup("yes") and not is_tool_followup(None)
    )
    for module in (loop_controller, loop_perception):
        body = inspect.getsource(module).replace('return str(text or "").startswith("[ACTION_FOLLOWUP")', "")
        assert 'startswith("[ACTION_FOLLOWUP")' not in body, module.__name__
    assert 'startswith("[ACTION_FOLLOWUP")' not in inspect.getsource(agent_loop)


# ── Delta 4 decision A: the sim's policy answers, passed explicitly ──────────────────────────────────


@pytest.mark.parametrize("method", ["resolve_confirmation", "resolve_plan_approval"])
def test_a_production_built_adapter_consults_the_sims_policy(method: str) -> None:
    """In production the adapter is handed ``bridge.percept_source`` (a ``ConversationalSource``), so the old
    ``isinstance(percept_source, SimulationBridge)`` lookup never found the policy and its own default "yes"
    answered. Built as production builds it, the policy is consulted and the approver is the sim's."""
    policy = MagicMock()
    getattr(policy, method).return_value = "no"
    adapter = _bridge_adapter(policy)
    assert getattr(adapter, method)({} if method == "resolve_confirmation" else "plan") == "no"
    getattr(policy, method).assert_called_once()
    assert adapter.answer_source() is InputSource.SIM


def test_an_adapter_with_no_policy_answers_nothing() -> None:
    from maxim.runtime.sim_adapter import SimulationAdapter

    adapter = SimulationAdapter(percept_source=object(), response_policy=None)
    assert (
        adapter.resolve_confirmation({}),
        adapter.resolve_plan_approval("p"),
        adapter.resolve_timeout_retry(1.0),
    ) == (
        None,
        None,
        None,
    )
    assert adapter.answer_source() is InputSource.LOOP  # a prompt it cannot answer is the loop's to settle


def test_the_adapter_requires_the_policy_argument() -> None:
    """Pushed into the type: a sim cannot forget it into the old silent default "yes"."""
    from maxim.runtime.sim_adapter import SimulationAdapter

    with pytest.raises(TypeError):
        SimulationAdapter(object())  # type: ignore[call-arg]


def test_a_sim_confirmation_yes_is_the_sims_and_runs() -> None:
    """End to end at the controller: a SUPERVISED confirmation parked under a sim with an auto-approve policy is
    answered by the policy (approver SIM, ``pending_loop_reply_source``), and that answer satisfies the confirmable
    reason, so the action runs once."""
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SupervisionPolicy
    from maxim.runtime.loop_perception import take_user_input
    from maxim.simulation.response_policy import auto_approve

    ctrl = _controller()
    ctrl.autonomy_controller = AutonomyController(
        initial_level=AutonomyLevel.SUPERVISED,
        supervision_policy=SupervisionPolicy(requires_confirmation={"bash"}, min_confidence_autonomous=0.0),
    )
    ctrl.pending_proposal = _proposal()
    sim = _bridge_adapter(auto_approve())
    ctrl.park_confirmation(ctrl.pending_proposal.action, sim=sim)
    assert ctrl.state.data["pending_loop_reply_source"] == "sim"
    taken = take_user_input({}, ctrl.state, MagicMock(), sim_mode=True)
    assert (taken.cli_input, taken.source) == ("yes", InputSource.SIM)
    run = MagicMock(return_value=MagicMock(followup=None))
    with patch("maxim.simulation.sim_logger.display_status"), patch("maxim.simulation.sim_logger.display_action"):
        assert ctrl.handle_confirmation(
            taken.cli_input,
            execute_and_learn=run,
            book_refusal=MagicMock(),
            book_machine_refusal=MagicMock(),
            observation={},
            sim=sim,
            answer_source=taken.source,
        )
    run.assert_called_once()


def test_the_verdict_names_who_may_satisfy_it() -> None:
    """Fold 1: a ``confirm`` must name its approvers, an ``allow`` or a ``refuse`` none; the PLANNING gate's set is
    listed, never the whole enum (fold 2)."""
    from maxim.agents.autonomy import ActionVerdict, VerdictKind
    from maxim.utils.input_source import HUMAN_APPROVERS, PLANNING_GATE_APPROVERS

    with pytest.raises(ValueError):
        ActionVerdict(VerdictKind.CONFIRM, "x")
    with pytest.raises(ValueError):
        ActionVerdict(VerdictKind.ALLOW, None, HUMAN_APPROVERS)
    # VOICE is not a human approver yet: a room microphone's "yes" must not confirm a write (#1729 decides the set)
    assert HUMAN_APPROVERS == {InputSource.OPERATOR, InputSource.SIM}
    assert InputSource.VOICE not in PLANNING_GATE_APPROVERS
    assert InputSource.SELF not in PLANNING_GATE_APPROVERS and InputSource.UNKNOWN not in PLANNING_GATE_APPROVERS


def test_a_needs_person_refusal_is_worded_from_its_cause_and_source() -> None:
    """Fold 3: true for the loop's own answer and for a future voice-only reason alike."""
    from maxim.utils.input_source import needs_person_detail

    loop = needs_person_detail("A tool outside the allow-list", InputSource.LOOP)
    assert (
        loop
        == "A tool outside the allow-list needs a person's approval; this run answered its own prompt, and no person did"
    )
    assert "stays parked until a person confirms" in needs_person_detail("Unlocking the door", InputSource.VOICE)
    assert "non-interactive" not in loop and "requires approval" not in loop


def test_off_mode_is_machine_approval_and_on_mode_waits_for_the_approver() -> None:
    """Owner decision 2026-11-21: in OFF mode (``interactive="off"``) the loop's machine auto-approve approves a
    queued proposal (``InputSource.LOOP``); with ``interactive`` on, as an embedder that approves plans itself runs,
    the proposal waits for that approver."""
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.simulation import sim_logger

    for mode, expect in ((sim_logger.InteractiveMode.OFF, "approved"), (sim_logger.InteractiveMode.ON, "pending")):
        ctrl = _controller()
        ctrl.autonomy_controller = AutonomyController(initial_level=AutonomyLevel.PLANNING)
        ctrl.pending_proposal = _proposal()
        with patch.object(sim_logger, "_interactive_mode", mode):
            ctrl.queue_for_approval(ctrl.pending_proposal.action, confidence=0.9, sim=MagicMock())
        [entry] = list(ctrl.autonomy_controller.proposal_queue._pending)
        assert entry.status == expect, mode
        if mode is sim_logger.InteractiveMode.OFF:
            assert entry.approver is InputSource.LOOP


# ── Delta 5: a timeout retry is no approval; the policy wiring ────────────────────────────────────────


def _timeout_tick(sim, mode):
    """An LLM timeout under ``sim`` in interactive ``mode``, then the loop's next two ticks: the timeout prompt's
    answer (if the loop gives one) and a person's next words."""
    from maxim.runtime.loop_perception import take_user_input
    from maxim.runtime.loop_types import TimeoutRetry  # noqa: F401  (the stashed shape)
    from maxim.runtime.tool_dispatch import _stash_timeout_retry
    from maxim.simulation import sim_logger

    ctrl = _controller()
    ctrl.llm_worker = MagicMock()
    with patch.object(sim_logger, "_interactive_mode", mode), patch("maxim.simulation.sim_logger.display_status"):
        _stash_timeout_retry({"_timeout_s": 30.0, "_original_request": object()}, sim, ctrl.state)
        answer = take_user_input({}, ctrl.state, MagicMock(), sim_mode=bool(sim.is_sim_mode))
        handled = []
        for got in (answer, take_user_input({"cli_input": "hello there"}, ctrl.state, MagicMock(), sim_mode=False)):
            if got.cli_input is None:
                handled.append(None)
                continue
            handled.append(
                ctrl.answer_pending_prompt(
                    str(got.cli_input),
                    execute_and_learn=MagicMock(),
                    book_refusal=MagicMock(),
                    book_machine_refusal=MagicMock(),
                    observation={},
                    sim=sim,
                    answer_source=got.source,
                )
            )
    return ctrl, answer, handled


@pytest.mark.parametrize("off", [True, False])
def test_a_sim_with_no_policy_survives_a_timeout(off: bool) -> None:
    """Delta-5 BLOCKER: the narrator, the CLI scenario, interactive turns and minecraft run with no policy. At base
    their adapter answered a timeout "yes"; after #1185 it answered nothing, the prompt parked, and every later
    input was re-asked. A timeout retry is no approval: with nobody to answer, the loop retries itself."""
    from maxim.runtime.sim_adapter import SimulationAdapter
    from maxim.simulation import sim_logger

    sim = SimulationAdapter(percept_source=object(), response_policy=None)
    mode = sim_logger.InteractiveMode.OFF if off else sim_logger.InteractiveMode.ON
    ctrl, answer, handled = _timeout_tick(sim, mode)
    assert (answer.cli_input, answer.source) == ("yes", InputSource.LOOP)
    ctrl.llm_worker.retry_with_timeout.assert_called_once()
    assert handled == [True, False]  # the retry consumed its answer; the next words are an ordinary turn


def test_production_headless_survives_a_timeout_and_a_person_is_asked() -> None:
    """The null adapter: in OFF mode the loop retries itself (mirroring the OFF-mode auto-yes); with interactive on
    a person is asked (nothing is answered for them)."""
    from maxim.runtime.sim_adapter import NullSimulationAdapter
    from maxim.simulation import sim_logger

    ctrl, answer, handled = _timeout_tick(NullSimulationAdapter(), sim_logger.InteractiveMode.OFF)
    assert answer.cli_input == "yes" and answer.source is InputSource.LOOP
    ctrl.llm_worker.retry_with_timeout.assert_called_once()
    assert handled == [True, False]
    ctrl, answer, _ = _timeout_tick(NullSimulationAdapter(), sim_logger.InteractiveMode.ON)
    assert answer.cli_input is None  # parked for the person
    assert ctrl.get_timeout_retry() is not None


@pytest.mark.parametrize(
    ("rel", "bridge"),
    [
        ("simulation/orchestrator.py", "bridge"),
        ("simulation/tools.py", "sub_bridge"),
        ("console/handle.py", "bridge"),
    ],
)
def test_each_bridge_sim_passes_its_bridges_policy(rel: str, bridge: str) -> None:
    """The sentinel catches a sim that forgets ``response_policy=``; this catches one that passes the wrong value:
    each sim with a bridge hands its loop ``<bridge>.response_policy``."""
    import ast
    from pathlib import Path

    import maxim

    tree = ast.parse((Path(maxim.__file__).parent / rel).read_text())
    passed = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", "")) == "run_agentic_loop"
        ):
            kw = {k.arg: k.value for k in node.keywords}
            value = kw.get("response_policy")
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "response_policy"
                and isinstance(value.value, ast.Name)
                and value.value.id == bridge
            ):
                passed.append(node.lineno)
    assert len(passed) == 1, f"{rel}: run_agentic_loop must take response_policy={bridge}.response_policy"


def test_a_sim_that_does_not_say_its_policy_is_refused() -> None:
    """Delta-5 item 2: ``run_agentic_loop`` / ``build_loop_run`` default to a sentinel, and a ``percept_source``
    without an explicit ``response_policy=`` is a ``TypeError``, so a sim cannot lose its policy silently."""
    from maxim.runtime.loop_setup import RESPONSE_POLICY_UNSET, _build_sim_adapter

    kwargs = dict(action_sink=None, pain_bus=None, sim_adapter=None, executor=None)
    with pytest.raises(TypeError, match="response_policy"):
        _build_sim_adapter(percept_source=object(), response_policy=RESPONSE_POLICY_UNSET, **kwargs)
    assert (
        _build_sim_adapter(percept_source=object(), response_policy=None, **kwargs).answer_source() is InputSource.LOOP
    )
    assert _build_sim_adapter(percept_source=None, response_policy=RESPONSE_POLICY_UNSET, **kwargs).is_sim_mode is False
