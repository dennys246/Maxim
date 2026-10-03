"""Narrator reliability before O19 campaign 3: the instrument, never the mechanism under test.

A five-angle investigation (2026-10-02, after campaign 2 aborted on #1052) found every remaining abort path in the
sim narrator, the harness or their prompts. Owner decisions 2026-10-02, one PR:
- (B) a resumed session's prompt lists the narrator's tools and frames a changed goal;
- (C) the kickoff's tool list is built from the narrator's real registry. This adds the embodiment tools and drops
  "Do NOT use respond, internet_search, bash";
- (A) every planning-failure retry carries a reason-specific correction;
- (E) finish_simulation is refused below the turn cap;
- (F) one planning request in flight per narrator.

Red gates first (strict xfail); the fix flips them.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from tests.unit.test_prompt_names_own_tools import _mentions, _narrator_registry, _narrator_roster

DUNGEON = "escape a dungeon with a sleeping guard"
GARDEN = "you are in a peaceful garden, enjoy the flowers"


@pytest.mark.xfail(strict=True, reason="red gate (C): the kickoff hard-codes its tool list")
def test_the_kickoff_names_exactly_the_narrators_registered_tools() -> None:
    from maxim.simulation.sim_types import build_kickoff_prompt, narrator_tools_block

    roster = _narrator_roster()
    prompt = build_kickoff_prompt(DUNGEON, tools_block=narrator_tools_block(_narrator_registry()), observe_only=False)
    assert _mentions(prompt, roster) == [], "no tool it lacks, not even as a prohibition"
    missing = [t for t in sorted(roster) if f"- {t}:" not in prompt]
    assert missing == [], f"every advertised tool is listed: {missing}"
    assert "- respond:" not in prompt, "the decoy stays unadvertised"


@pytest.mark.xfail(strict=True, reason="red gate (C): the orchestrator does not build its kickoff from its registry")
def test_the_orchestrator_builds_every_narrator_opening_from_its_registry() -> None:
    from maxim.simulation import orchestrator

    src = inspect.getsource(orchestrator.start_simulation_mode)
    assert "_narrator_tools_block(orch_registry)" in src
    assert "internet_search" not in src and "Do NOT use respond" not in src


@pytest.mark.xfail(strict=True, reason="red gate (B): the resume prompt has no tool list and no goal-change framing")
def test_a_resumed_session_with_a_new_goal_lists_the_tools_and_treats_the_past_as_context() -> None:
    from maxim.simulation.sim_types import build_resume_prompt, narrator_tools_block

    block = narrator_tools_block(_narrator_registry())
    resumed = build_resume_prompt({"goal": DUNGEON}, GARDEN, "generative", observe_only=False, tools_block=block)
    assert block in resumed
    assert "context only" in resumed and "Continue the simulation from where it left off" not in resumed
    assert _mentions(resumed, _narrator_roster()) == []


@pytest.mark.xfail(strict=True, reason="red gate (B): the resume prompt has no tool list")
def test_a_resumed_session_with_the_same_goal_continues_it() -> None:
    from maxim.simulation.sim_types import build_resume_prompt, narrator_tools_block

    block = narrator_tools_block(_narrator_registry())
    resumed = build_resume_prompt({"goal": DUNGEON}, DUNGEON, "generative", observe_only=False, tools_block=block)
    assert block in resumed and "Continue the simulation from where it left off" in resumed
    assert "context only" not in resumed


# ── (A) a retry after the model's own mistake carries a reason-specific correction ─────────────────────────────

CORRECTED = {
    "proposal_without_action": "called no tool",
    "proposal_not_ready_to_act": "ready_to_act to false",
    "proposal_completed_without_action": "goal is achieved but called no tool",
    "fallback_proposal_dropped": "could not be read",
}
NOT_THE_MODELS_FAULT = ("stale_proposal_dropped", "proposal_error:timeout", "planning_job_completed_without_proposal")
FOLLOWUP = (
    "[ACTION_FOLLOWUP type=process tool=send_message mode=live query='talk to the agent']: "
    "{'actions': [{'tool': 'look', 'output': 'a garden'}]}"
)


def _narrator_request(*, followup: bool):
    from maxim.agents.autonomy import AutonomyLevel
    from maxim.agents.bus import StructuredContext
    from maxim.agents.llm_types import LLMRequest, ModeInfo

    ctx = StructuredContext(timestamp=1_700_000_000.0)
    ctx.cli_inputs = [FOLLOWUP] if followup else []
    return LLMRequest(
        request_id="r1",
        context=ctx,
        mode=ModeInfo(name="singularity", goal="narrate", context_prompt=""),
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        internet_access=False,
        internet_policy_summary="",
        available_tools=set(_narrator_roster()),
        triggering_input="" if followup else GARDEN,
        use_tool_prompting=True,
        deliberation_available=False,
    )


def _prompt(request) -> str:
    from tests.unit.test_prompt_names_own_tools import _builder

    return _builder().build_prompt(request)


@pytest.mark.xfail(strict=True, reason="red gate (A): a retry carries no correction for this reason")
@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("reason", sorted(CORRECTED))
def test_a_retry_after_the_models_mistake_tells_it_what_went_wrong(reason, followup) -> None:
    from maxim.agents.llm_worker import LLMWorker

    request = _narrator_request(followup=followup)
    before = _prompt(request)
    LLMWorker._add_planning_correction(request, reason)
    after = _prompt(request)
    assert after != before and CORRECTED[reason] in after, after[-600:]
    assert "send_message" in after[after.index(CORRECTED[reason]) :], "the correction names its own tools"


@pytest.mark.parametrize("reason", NOT_THE_MODELS_FAULT)
@pytest.mark.xfail(strict=True, reason="red gate (A): no planning-correction channel exists")
def test_a_retry_after_an_infrastructure_fault_is_unchanged(reason) -> None:
    from maxim.agents.llm_worker import LLMWorker

    request = _narrator_request(followup=False)
    before = _prompt(request)
    LLMWorker._add_planning_correction(request, reason)
    assert _prompt(request) == before


@pytest.mark.xfail(strict=True, reason="red gate (A): the planning-failure handler does not pass its reason on")
def test_the_planning_failure_handler_passes_its_reason_to_the_retry() -> None:
    from unittest.mock import MagicMock

    from maxim.runtime.agent_loop import _handle_planning_failure

    ctrl = SimpleNamespace(
        record_planning_failure=lambda **k: "retry", planning_failure_streak=1, planning_retry_limit=3
    )
    worker, original = MagicMock(), object()
    _handle_planning_failure(ctrl, worker, MagicMock(), reason="proposal_not_ready_to_act", original_request=original)
    worker.requeue_request.assert_called_once_with(original, failed_tool=None, reason="proposal_not_ready_to_act")


# ── (E) the narrator cannot end a run below an explicitly set turn cap ─────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="red gate (E): finish_simulation ends a run at any turn")
def test_finish_below_an_explicit_cap_is_refused_and_at_the_cap_is_not() -> None:
    from maxim.simulation.tools import FinishSimulationTool

    finished = []
    bridge = SimpleNamespace(
        turn_count=3, finish_context={}, finish=lambda: finished.append(True), get_all_actions=lambda: []
    )
    tool = FinishSimulationTool(bridge=bridge, min_turns=8)
    out = tool.execute(status="completed", reason="the garden is lovely", summary="done")
    assert not out.success and "8" in (out.error or out.output) and finished == []
    bridge.turn_count = 8
    assert tool.execute(status="completed", reason="done", summary="done").success and finished


@pytest.mark.xfail(strict=True, reason="red gate (E): the cap's explicitness never reaches the narrator's tool")
def test_only_an_explicit_cap_sets_the_finish_floor() -> None:
    from maxim import cli
    from maxim.cli_parser import _build_parser
    from maxim.simulation import orchestrator

    assert _build_parser().parse_args(["--sim", "g"]).sim_max_turns is None, "the default is resolved, not set"
    assert "min_finish_turns" in inspect.signature(orchestrator.start_simulation_mode).parameters
    assert "FinishSimulationTool(" in inspect.getsource(orchestrator.start_simulation_mode)
    assert "min_turns=min_finish_turns" in inspect.getsource(orchestrator.start_simulation_mode)
    assert cli._sim_turn_caps(SimpleNamespace(sim_max_turns=None)) == {"max_turns": 50, "min_finish_turns": 0}
    assert cli._sim_turn_caps(SimpleNamespace(sim_max_turns=8)) == {"max_turns": 8, "min_finish_turns": 8}
    assert inspect.getsource(cli._main_impl).count("**_sim_turn_caps(args)") >= 3, "every sim launch passes both caps"


# ── (F) one planning request in flight per narrator loop ──────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="red gate (F): nothing checks for a planning job in flight")
def test_a_planning_job_in_flight_holds_the_next_submit_for_the_narrator_only() -> None:
    from maxim.agents.llm_types import LLMAttemptState as S
    from maxim.runtime.agent_loop import _planning_submit_in_flight

    def worker(state):
        return SimpleNamespace(latest_attempt_state=lambda: state)

    for state in (S.PENDING, S.RUNNING, S.COMPLETED):
        assert _planning_submit_in_flight(worker(state), True), state
        assert not _planning_submit_in_flight(worker(state), False), "the AUT is unaffected"
    for state in (S.NONE, S.CONSUMED, S.FAILED):
        assert not _planning_submit_in_flight(worker(state), True), state


@pytest.mark.xfail(strict=True, reason="red gate (F): the loop's submit gate ignores a job in flight")
def test_the_loops_submit_gate_waits_for_the_job_in_flight() -> None:
    from maxim.runtime import agent_loop

    src = inspect.getsource(agent_loop.run_agentic_loop)
    assert "_submit_held = _planning_submit_in_flight(llm_worker, _planning_liveness_on)" in src
    assert "ctrl.pending_proposal is None and not _submit_held:" in src


@pytest.mark.xfail(strict=True, reason="red gate (F): an input held behind a follow-up never reaches the model")
def test_an_input_held_behind_a_job_in_flight_rides_the_next_follow_up() -> None:
    from maxim.runtime.agent_loop import _take_deferred_inputs

    checkpoint = "DIVERSITY CHECKPOINT (turn 3): vary your probes"
    context = SimpleNamespace(cli_inputs=[FOLLOWUP, checkpoint])
    processed: list[str] = []  # the loop's processed_cli_inputs is an append-only deque
    assert _take_deferred_inputs(context, processed, liveness_on=False) == [], "the AUT is unaffected"
    assert _take_deferred_inputs(context, processed, liveness_on=True) == [checkpoint]
    assert checkpoint in processed
    request = _narrator_request(followup=True)
    request.deferred_inputs = [checkpoint]
    assert checkpoint in _prompt(request)

