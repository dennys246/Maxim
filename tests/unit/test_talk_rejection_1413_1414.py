"""A rejected tool call in Console Talk (#1413) and a consumed input that is never cleared (#1414).

Both were found by maxim-pulse live on the 1.3.3 candidate: Talk chose `bash`, autonomy refused it, and then:
- #1413: nothing scheduled a second LLM call, so the turn idled to its 180 s timeout and the chat said only
  "(no reply — the turn timed out before finishing)";
- #1414: the user's `cli_input` stayed in `state.data`, so `propose_intent` re-published it every 0.5 s tick and the
  Hippocampus re-captured it as new memory (about 87 times in 44 s).

Owner decisions 2026-10-22: a rejection becomes a capped follow-up (at most 2 per user turn, then a blocked record
that ends the turn with a reply naming the tool), in every loop; an input is consumed once, with no content dedupe
(a user saying the same thing twice is two observations).
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

# ── #1413: a hard rejection is fed back to the model ───────────────────────────────────────────────────────────


def test_in_substrate_primary_a_hard_rejection_is_booked_and_leaves_no_follow_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
):
    """The REAL loop, substrate-primary (the driver's mode), one hard-forbidden proposal: there is no model to
    re-plan (its LLM submission is off), so the refusal is booked as before and NO follow-up is left, which would
    never be consumed and would keep the idle gate awake (review). The LLM-primary behaviour is pinned below."""
    from tests.unit._execute_learn_driver import run_once

    tool = "probe_forbidden_1413"
    obs = run_once(monkeypatch, tmp_path, tool=tool, level="supervised", policy={"forbidden_tools": {tool}})
    assert obs.executed == []  # refused: the tool never ran
    [kw] = obs.outcomes
    assert kw["error"].startswith("Rejected by autonomy") and kw["success"] is False  # booked, as before
    obs.note_followup()
    assert obs.followups == []


# ── The REAL loop, LLM-primary (the mode Talk runs in), with a stub worker ───────────────────────────────────


def _run_llm_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    proposal_tool: str | None = None,
    answered_at_tick: int = 3,
    ticks: int = 10,
    forbidden: frozenset[str] = frozenset(),
) -> tuple[list[Any], list[dict[str, Any]]]:
    """One user input on tick 0, then drained observations. The stub worker answers on ``answered_at_tick`` (the
    loop's ``pending_user_input`` cleared, as a real proposal clears it; until then §3 is gated off), proposing
    ``proposal_tool`` once if given. The agent is the REAL ``PerceptionAgent`` on a recording bus. Returns the
    published Percepts and every ``submit_context`` call's keywords."""
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel, SafetyConstraints
    from maxim.agents.llm_types import LLMProposal
    from maxim.agents.perception_agent import PerceptionAgent
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime.bootstrap import build_decision_engine, build_executor, build_memory
    from maxim.runtime.state import RuntimeState
    from maxim.simulation import sim_logger
    from maxim.tools.registry import ToolRegistry

    from maxim.runtime.loop_controller import LoopController

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_logger, "_interactive_mode", sim_logger.InteractiveMode.OFF)
    real_init = LoopController.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        self.llm_submit_interval = 0.0  # 200 Hz ticks last ms: the 0.5 s wall-clock cadence would never resubmit

    monkeypatch.setattr(LoopController, "__init__", _init)
    (tmp_path / "ws").mkdir(exist_ok=True)
    state = RuntimeState()
    state.data["mode"] = "active"
    clock = {"tick": -1}
    submits: list[dict[str, Any]] = []
    proposals = [proposal_tool] if proposal_tool else []

    class _Env(FileSystemEnv):
        def observe(self) -> dict[str, Any]:
            clock["tick"] += 1
            if clock["tick"] == 0:
                return {"cli_input": "what is the news today"}
            if clock["tick"] == answered_at_tick:  # the LLM answered: the loop's pending input is cleared
                for key in ("pending_user_input", "pending_user_input_time", "pending_user_input_source"):
                    state.data.pop(key, None)
            return {}

    class _Worker:
        def latest_attempt_state(self) -> None:
            return None

        def get_latest_proposal(self) -> Any:
            if clock["tick"] < answered_at_tick or not proposals:
                return None
            tool = proposals.pop(0)
            return LLMProposal(
                request_id=f"p-1413-{tool}",
                action={"tool_name": tool, "params": {}},
                reasoning="fetch the news",
                strategy_used="llm",
                confidence=0.9,
                mode_goal_achieved=False,
                triggering_input="what is the news today",
            )

        def submit_context(self, **kw: Any) -> bool:
            submits.append(kw)
            return True

        def requeue_request(self, *_a: Any, **_k: Any) -> bool:
            return True

        def requeue_last_request(self, **_k: Any) -> bool:
            return True

        def record_outcome(self, *_a: Any, **_k: Any) -> None:
            return None

    class _Bus:
        def __init__(self) -> None:
            self.published: list[Any] = []

        def publish(self, item: Any) -> None:
            self.published.append(item)

        def subscribe(self, *_a: Any, **_k: Any) -> None:
            return None

    bus = _Bus()
    autonomy = AutonomyController(
        initial_level=AutonomyLevel.AUTONOMOUS, safety_constraints=SafetyConstraints(forbidden_tools=set(forbidden))
    )
    AL.run_agentic_loop(
        PerceptionAgent(bus=bus),
        _Env(str(tmp_path / "ws")),
        state,
        build_memory(),
        build_decision_engine(),
        build_executor(ToolRegistry(), pain_bus=None, permissions=None),
        autonomy_controller=autonomy,
        llm_worker=_Worker(),
        max_steps=ticks,
        target_hz=200.0,
        idle_sleep_s=0.0,
        stop_event=threading.Event(),
    )
    return [p for p in bus.published if getattr(p, "cli_input", None)], submits


def test_one_input_is_published_once_while_an_llm_answer_is_pending(monkeypatch: pytest.MonkeyPatch, tmp_path: Any):
    """#1414, in the mode Talk runs in: §3 (``propose_intent``) is gated off on the delivering tick while the LLM is
    pending, so the input must outlive that tick (dropping it per tick published it ZERO times, review W1) and be
    consumed by the read that publishes it (keeping it published it once per tick, #1414)."""
    published, _ = _run_llm_loop(monkeypatch, tmp_path)
    assert [p.cli_input for p in published] == ["what is the news today"], published


@pytest.mark.parametrize("pending_until", [1, 4])
def test_one_input_is_published_once_however_long_the_answer_takes(monkeypatch, tmp_path, pending_until):
    published, _ = _run_llm_loop(monkeypatch, tmp_path, answered_at_tick=pending_until, ticks=pending_until + 6)
    assert len(published) == 1, published


def test_a_hard_rejection_makes_the_next_llm_call_a_refused_follow_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Any):
    """#1413, in the mode Talk runs in: after the model proposes a forbidden tool, the loop's next submission carries
    the refusal as a ``refused`` follow-up, so the model re-plans or replies instead of the turn idling."""
    _, submits = _run_llm_loop(monkeypatch, tmp_path, proposal_tool="bash", forbidden=frozenset({"bash"}))
    followups = [
        c
        for kw in submits
        for c in (getattr(kw.get("context"), "cli_inputs", None) or [])
        if str(c).startswith("[ACTION_FOLLOWUP type=refused tool=bash")
    ]
    assert followups, [getattr(kw.get("context"), "cli_inputs", None) for kw in submits]
    assert "Rejected by autonomy" in followups[0]


# ── The chat names a refused tool ──────────────────────────────────────────────────────────────────────────────


def test_a_turn_ended_by_a_refused_tool_says_the_tool_is_not_allowed():
    from types import SimpleNamespace

    from maxim.console.handle import _describe_silent_turn

    blocked = SimpleNamespace(
        tool_name="bash",
        blocked=True,
        block_reason="Rejected by autonomy: Tool 'bash' is forbidden",
        result_success=False,
        utterance=None,
    )
    text = _describe_silent_turn({"timed_out": False, "actions": [blocked]})
    assert "bash" in text and "not allowed" in text, text


# ── #1413: the cap — two re-plans per user turn, then the turn ends with the refusal ──────────────────────────


def _reject(
    ctrl: Any, sink: Any, booked: list[dict[str, Any]], *, turn: str, tool: str = "bash", aut_mode: str = "llm-primary"
) -> Any:
    from types import SimpleNamespace

    from maxim.runtime.tool_dispatch import book_hard_rejection

    audit = SimpleNamespace(log_action=lambda **kw: booked.append({"audit": kw}))

    def _record(**kw: Any) -> None:  # like tool_dispatch.record_outcome: the outcome lands in the window
        booked.append(kw)
        ctrl.recent_outcomes.append({"tool": kw["tool_name"], "success": kw["success"], "error": kw["error"]})

    proposal = SimpleNamespace(reasoning="get the news", cluster_id=None, clusters=None, triggering_input=turn)
    return book_hard_rejection(
        ctrl=ctrl,
        autonomy_controller=audit,
        rec_outcome=_record,
        agent_id="talk",
        nac=None,
        aut_mode=aut_mode,
        proposal=proposal,
        action={"tool_name": tool, "params": {"command": "curl news"}},
        reason=f"Tool '{tool}' is forbidden",
        confidence=0.9,
        mode="active",
        active_goal=None,
    )


def _ctrl(sink: Any = None) -> Any:
    """A controller stand-in whose executor records blocks onto ``sink`` (InstrumentedExecutor.record_block's
    shape), or has no ``record_block`` at all (no action sink: no record)."""
    from types import SimpleNamespace

    from maxim.simulation.sinks import ActionRecord

    executor = SimpleNamespace()
    if sink is not None:
        executor.record_block = lambda tool, reason, params=None: sink.record(
            ActionRecord(timestamp=0.0, tool_name=tool, tool_args=params or {}, blocked=True, block_reason=reason)
        )
    return SimpleNamespace(
        recent_outcomes=[], max_recent_outcomes=10, llm_worker=None, context_pool=None,
        rejection_turn=None, rejection_followups=0, executor=executor,
    )  # fmt: skip


def _new_turn(ctrl: Any, text: str) -> None:
    from maxim.runtime.loop_controller import LoopController

    LoopController.note_user_input(ctrl, text)


def test_a_turn_gets_two_rejection_follow_ups_then_ends_with_a_blocked_record():
    from maxim.runtime.tool_dispatch import MAX_REJECTION_FOLLOWUPS
    from maxim.simulation.sinks import RecordingSink

    assert MAX_REJECTION_FOLLOWUPS == 2  # owner decision 2026-10-22
    sink, booked = RecordingSink(), []
    ctrl = _ctrl(sink)
    _new_turn(ctrl, "news?")
    first = _reject(ctrl, sink, booked, turn="news?")
    second = _reject(ctrl, sink, booked, turn="news?")
    assert first is not None and second is not None and sink.actions == []  # no record while the model re-plans
    assert first.followup_type == "refused" and first.original_query == "news?"
    third = _reject(ctrl, sink, booked, turn="news?")
    assert third is None  # the cap: the turn ends
    [record] = sink.actions
    assert record.blocked and record.tool_name == "bash"
    assert (record.block_reason or "").startswith("Rejected by autonomy")  # reports name the reason, not "unknown"
    assert record.utterance is None  # a refusal says nothing; the chat names the tool instead
    # Every rejection is still booked for the model and audited, follow-up or not.
    assert sum(1 for b in booked if "audit" in b) == 3 and sum(1 for b in booked if "error" in b) == 3
    # A new user input starts the count again, even when it repeats the last one word for word (review: a turn is an
    # input, not a text).
    _new_turn(ctrl, "news?")
    assert _reject(ctrl, sink, booked, turn="news?") is not None and len(sink.actions) == 1


def test_a_deliberated_proposal_keeps_the_turns_question_and_its_count():
    """Deliberated proposals carry no triggering input: they share the turn's count and its question (review)."""
    from maxim.simulation.sinks import RecordingSink

    sink, booked = RecordingSink(), []
    ctrl = _ctrl(sink)
    _new_turn(ctrl, "news?")
    first = _reject(ctrl, sink, booked, turn="")
    assert first is not None and first.original_query == "news?"
    assert _reject(ctrl, sink, booked, turn="") is not None
    assert _reject(ctrl, sink, booked, turn="") is None and len(sink.actions) == 1  # the same turn: capped


def test_substrate_primary_keeps_the_booking_only():
    """No model to re-plan: no follow-up (it would never be consumed and would keep the idle gate awake) and no
    record (it would shift that arm's action counts), however many refusals (review)."""
    from maxim.simulation.sinks import RecordingSink

    sink, booked = RecordingSink(), []
    ctrl = _ctrl(sink)
    for _ in range(4):
        assert _reject(ctrl, sink, booked, turn="", aut_mode="substrate-primary") is None
    assert sink.actions == [] and sum(1 for b in booked if "error" in b) == 4  # still booked every time


def test_the_refused_prompt_says_the_tool_did_not_run_outside_the_frame():
    from maxim.agents.prompt_builder import PromptBuilder
    from maxim.runtime.agent_loop import _followup_synthetic_input
    from maxim.runtime.loop_types import ActionFollowup

    followup = ActionFollowup(
        tool="bash",
        result="Rejected by autonomy: Tool 'bash' is forbidden",
        original_query="news?",
        followup_type="refused",
        mode="active",
        timestamp=0.0,
    )
    prompt = PromptBuilder(
        llm=None, reasoning_carryover=None, n_ctx=32000, token_counter=len, tool_index=None
    )._build_followup_prompt(_followup_synthetic_input(followup), tools={"respond", "read_file"})
    outside = prompt.split("<<TOOL_OUTPUT", 1)[0] + prompt.rsplit(">>", 1)[-1]
    assert "REFUSED" in outside and "did not run" in outside and "Do not propose 'bash' again" in outside
    assert "You just executed" not in prompt


def test_the_count_is_consecutive_an_action_in_between_starts_it_again():
    """A long autonomous run has no user turns: an action that ran between two refusals ends the chain (review W2),
    so the run is not capped once for its whole life."""
    from maxim.simulation.sinks import RecordingSink

    sink, booked = RecordingSink(), []
    ctrl = _ctrl(sink)
    assert _reject(ctrl, sink, booked, turn="") is not None
    assert _reject(ctrl, sink, booked, turn="") is not None
    ctrl.recent_outcomes.append({"tool": "read_file", "success": True, "error": None})  # an action ran
    assert _reject(ctrl, sink, booked, turn="") is not None and sink.actions == []


def test_without_an_action_sink_the_cap_records_nothing():
    from maxim.simulation.sinks import RecordingSink

    sink, booked = RecordingSink(), []
    ctrl = _ctrl()  # the executor has no record_block: no sink was given to the run
    for _ in range(3):
        _reject(ctrl, sink, booked, turn="news?")
    assert sink.actions == [] and ctrl.rejection_followups == 2
