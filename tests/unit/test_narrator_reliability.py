"""Narrator reliability before O19 campaign 3: the instrument, never the mechanism under test.

A five-angle investigation (2026-10-02, after campaign 2 aborted on #1052) found every remaining abort path in the
sim narrator, the harness or their prompts. Owner decisions 2026-10-02, one PR:
- (B) a resumed session's prompt lists the narrator's tools and frames a changed goal;
- (C) the kickoff's tool list is built from the narrator's real registry. This adds the embodiment tools and drops
  "Do NOT use respond, internet_search, bash";
- (A) every planning-failure retry carries a reason-specific correction;
- (E) finish_simulation is refused below the turn cap;
- (F) one planning request in flight per narrator.

The red-gate commit marked these strict xfail; the fix flips them.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from tests.unit.test_prompt_names_own_tools import _mentions, _narrator_registry, _narrator_roster

DUNGEON = "escape a dungeon with a sleeping guard"
GARDEN = "you are in a peaceful garden, enjoy the flowers"


def test_the_kickoff_names_exactly_the_narrators_registered_tools() -> None:
    from maxim.simulation.sim_types import build_kickoff_prompt, narrator_tools_block

    roster = _narrator_roster()
    prompt = build_kickoff_prompt(DUNGEON, tools_block=narrator_tools_block(_narrator_registry()), observe_only=False)
    assert _mentions(prompt, roster) == [], "no tool it lacks, not even as a prohibition"
    missing = [t for t in sorted(roster) if f"- {t}:" not in prompt]
    assert missing == [], f"every advertised tool is listed: {missing}"
    assert "- respond:" not in prompt, "the decoy stays unadvertised"


def test_the_orchestrator_builds_every_narrator_opening_from_its_registry() -> None:
    from maxim.simulation import orchestrator

    src = inspect.getsource(orchestrator.start_simulation_mode)
    assert "_narrator_tools_block(orch_registry)" in src
    assert "internet_search" not in src and "Do NOT use respond" not in src


def test_a_resumed_session_with_a_new_goal_lists_the_tools_and_treats_the_past_as_context() -> None:
    from maxim.simulation.sim_types import build_resume_prompt, narrator_tools_block

    block = narrator_tools_block(_narrator_registry())
    resumed = build_resume_prompt({"goal": DUNGEON}, GARDEN, "generative", observe_only=False, tools_block=block)
    assert block in resumed
    assert "context only" in resumed and "Continue the simulation from where it left off" not in resumed
    assert _mentions(resumed, _narrator_roster()) == []


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
def test_a_retry_after_an_infrastructure_fault_is_unchanged(reason) -> None:
    from maxim.agents.llm_worker import LLMWorker

    request = _narrator_request(followup=False)
    before = _prompt(request)
    LLMWorker._add_planning_correction(request, reason)
    assert _prompt(request) == before


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


def test_only_the_run_full_turns_flag_sets_the_finish_floor_and_an_observer_is_exempt() -> None:
    """Owner decisions 2026-10-02: opt-in (other harnesses pass --sim-max-turns and keep early finishing)."""
    from maxim import cli
    from maxim.cli_parser import _build_parser
    from maxim.simulation import orchestrator

    plain = _build_parser().parse_args(["--sim", "g", "--sim-max-turns", "8"])
    full = _build_parser().parse_args(["--sim", "g", "--sim-max-turns", "8", "--sim-run-full-turns"])
    assert cli._sim_turn_caps(plain) == {"max_turns": 8, "min_finish_turns": 0}
    assert cli._sim_turn_caps(full) == {"max_turns": 8, "min_finish_turns": 8}
    assert inspect.getsource(cli._main_impl).count("**_sim_turn_caps(args)") >= 3, "every sim launch passes both caps"
    src = inspect.getsource(orchestrator.start_simulation_mode)
    assert "min_turns=0 if _is_observe_only else min_finish_turns" in src
    assert src.index("_is_observe_only = ") < src.index("FinishSimulationTool("), "known before the tool is built"


# ── (F) one planning request in flight per narrator loop ──────────────────────────────────────────────────────


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


def test_the_loops_submit_gate_waits_for_the_job_in_flight() -> None:
    from maxim.runtime import agent_loop

    src = inspect.getsource(agent_loop.run_agentic_loop)
    assert "_submit_held = _planning_submit_in_flight(llm_worker, _planning_liveness_on)" in src
    assert "ctrl.pending_proposal is None and not _submit_held:" in src


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


def test_the_worker_records_the_reason_on_the_retry_it_sends(monkeypatch) -> None:
    from maxim.agents.llm_worker import LLMWorker
    from tests.unit.test_planning_liveness import NoneLLM

    worker = LLMWorker(llm=NoneLLM(), stale_threshold_s=10.0)
    monkeypatch.setattr(worker, "_resubmit", lambda request, job_suffix: True)
    request = _narrator_request(followup=False)
    assert worker.requeue_request(request, reason="proposal_not_ready_to_act")
    assert request.planning_corrections == ["proposal_not_ready_to_act"]
    assert worker.requeue_request(request, reason="stale_proposal_dropped"), "an infrastructure fault"
    assert request.planning_corrections == ["proposal_not_ready_to_act"], "records nothing for a fault"


def test_the_loop_folds_held_inputs_into_the_follow_up_it_submits() -> None:
    from maxim.runtime import agent_loop

    src = "".join(inspect.getsource(agent_loop.run_agentic_loop).split())  # whitespace-insensitive (ruff wraps)
    for line in (
        "_deferred_inputs=_take_deferred_inputs(context,processed_cli_inputs,_planning_liveness_on)",
        "new_cli_input=Noneif_deferred_inputselsenew_cli_input",  # the follow-up keeps its original query
        "_release_unsent_deferred_inputs(processed_cli_inputs,_deferred_inputs,submitted)",
        "deferred_inputs=_deferred_inputs,",
    ):
        assert line in src, line


def test_the_worker_carries_held_inputs_into_the_follow_up_prompt() -> None:
    import time

    from maxim.agents.autonomy import AutonomyLevel
    from maxim.agents.bus import StructuredContext
    from maxim.agents.llm_worker import LLMWorker
    from tests.unit.test_planning_liveness import NoneLLM, _make_mode_info, _wait_for_proposal

    llm = NoneLLM()
    worker = LLMWorker(llm=llm, stale_threshold_s=10.0)
    worker.start()
    try:
        ctx = StructuredContext(timestamp=time.time())
        ctx.cli_inputs = [FOLLOWUP]
        assert worker.submit_context(
            context=ctx,
            mode=_make_mode_info(),
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            internet_access=False,
            internet_policy_summary="",
            use_tool_prompting=True,
            available_tools=set(_narrator_roster()),
            deliberation_available=False,
            deferred_inputs=["DIVERSITY CHECKPOINT (turn 3): vary your probes"],
        )
        _wait_for_proposal(worker)
        assert llm.prompts and "DIVERSITY CHECKPOINT (turn 3)" in llm.prompts[0]
    finally:
        worker.stop()


def test_a_failed_submit_releases_the_inputs_it_carried() -> None:
    from collections import deque

    from maxim.runtime.agent_loop import _release_unsent_deferred_inputs

    processed = deque(["a", "checkpoint"], maxlen=20)
    _release_unsent_deferred_inputs(processed, ["checkpoint"], submitted=True)
    assert "checkpoint" in processed
    _release_unsent_deferred_inputs(processed, ["checkpoint", "evicted"], submitted=False)
    assert list(processed) == ["a"], "unprocessed again, so they ride the next submit"


def test_a_tool_description_naming_a_tool_the_narrator_lacks_is_not_shown() -> None:
    from maxim.simulation.sim_types import narrator_tools_block

    class _Tool:
        description = "Like internet_search but local. More text."

    class _Registry:
        def advertised(self):
            return ["send_message", "lookup"]

        def get(self, name):
            return _Tool()

    block = narrator_tools_block(_Registry())
    assert "internet_search" not in block and "- lookup:" in block


def test_the_release_runs_even_when_the_submit_raises() -> None:
    import ast
    import inspect

    from maxim.runtime import agent_loop

    tree = ast.parse(inspect.getsource(agent_loop.run_agentic_loop).lstrip())
    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and "submit_context" in ast.unparse(ast.Module(body=node.body, type_ignores=[]))
        and "_release_unsent_deferred_inputs" in ast.unparse(ast.Module(body=node.finalbody, type_ignores=[]))
    ]
    assert len(guarded) == 1, "the narrator's submit releases its carried inputs in a finally"


def test_every_open_o19_campaign_runs_every_turn_it_asks_for() -> None:
    """The flag's caller (#1058). It replaced a red gate keyed on an Exp 10 campaign 3, which the owner dropped on
    2026-10-03 (strict: no successor on changed subject code; T1-1 goes to #1060): every O19 campaign but the two closed before the flag existed runs
    each phase to its cap, so C1 is not lost to a narrator that finishes early."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "o19_verdict.py"
    spec = importlib.util.spec_from_file_location("o19_verdict_for_caller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pre_flag = {"10", "10c2"}  # closed before #1058 shipped the flag; every other campaign, future ones too, needs it
    keys = [k for k in module.PROTOCOL if k not in pre_flag]
    assert "09" in keys, "Exp 09's campaign is the flag's caller"
    for key in keys:
        assert all("--sim-run-full-turns" in phase[4] for phase in module.PROTOCOL[key]["phases"]), key
