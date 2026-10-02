"""#1052: an agent is offered "think first" (``ready_to_act: false``) only if its loop can deliberate.

O19 Exp 10 campaign 2 attempt 1 aborted in phase 3 (the resumed garden session). On its first turn the simulation
narrator answered ``ready_to_act: false`` four times, as its prompt invites. Since April the instructions have said
"false to keep thinking" (499151e5), and the PFC preamble reaches every sim agent through ``_sim_active``
(0b595938). But the narrator's loop runs with no bio-enrichment pipeline, so no deliberation cycle continues the
thought, and each answer counts as a planning failure (D13, #523) until the budget runs out.

Owner decisions 2026-10-02:
- the fix is decided PER LOOP: deliberation is available when the loop has a pipeline, so the AUT's prompt is
  byte-identical;
- one keyword in the fenced ``agent_loop.py`` (an exception recorded in ``roadmap_1_3_x.md``);
- the resumed session's first-action instruction matches the fresh one's.

Red gates first (strict xfail); the fix commit removes the markers.
"""

from __future__ import annotations

import inspect
import time

import pytest

from maxim.agents.autonomy import AutonomyLevel
from maxim.agents.bus import StructuredContext
from maxim.agents.llm_types import LLMRequest, ModeInfo

THINK_OFFERS = ("false to keep thinking", "PRIVATE INNER THOUGHT", 'set "ready_to_act" to false')


def _builder():
    from maxim.agents.llm_fallback import ReasoningCarryover
    from maxim.agents.prompt_builder import PromptBuilder
    from maxim.models.language.token_counter import CharEstimateCounter

    return PromptBuilder(
        llm=None,
        reasoning_carryover=ReasoningCarryover(),
        n_ctx=32000,
        token_counter=CharEstimateCounter(),
        tool_index=None,
    )


def _request(**extra) -> LLMRequest:
    return LLMRequest(
        request_id="r1",
        context=StructuredContext(timestamp=1_700_000_000.0),
        mode=ModeInfo(name="singularity", goal="narrate", context_prompt=""),
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        internet_access=False,
        internet_policy_summary="",
        available_tools={"send_message", "observe_actions"},
        triggering_input="you are in a peaceful garden, enjoy the flowers",
        use_tool_prompting=True,
        **extra,
    )


@pytest.fixture
def in_sim(monkeypatch):
    from maxim.simulation import sim_logger

    monkeypatch.setattr(sim_logger, "_sim_active", True)


@pytest.mark.xfail(strict=True, reason="red gate #1052: a loop that cannot deliberate is still offered it")
def test_a_loop_that_cannot_deliberate_is_not_offered_it(in_sim) -> None:
    prompt = _builder().build_prompt(_request(deliberation_available=False))
    offered = [t for t in THINK_OFFERS if t in prompt]
    assert offered == [], offered
    assert '"action": {"tool_name"' in prompt, "it is still told how to act"


@pytest.mark.xfail(strict=True, reason="red gate #1052: the request carries no deliberation fact")
def test_a_loop_that_deliberates_sees_the_prompt_unchanged(in_sim) -> None:
    today = _builder().build_prompt(_request())
    assert _builder().build_prompt(_request(deliberation_available=True)) == today
    assert all(t in today for t in THINK_OFFERS), "today's prompt offers deliberation (the AUT keeps it)"


@pytest.mark.xfail(strict=True, reason="red gate #1052: submit_context cannot say whether the loop deliberates")
def test_the_worker_carries_the_loops_answer_into_the_prompt(in_sim) -> None:
    from maxim.agents.llm_worker import LLMWorker
    from tests.unit.test_planning_liveness import NoneLLM, _make_mode_info, _wait_for_proposal

    llm = NoneLLM()
    worker = LLMWorker(llm=llm, stale_threshold_s=10.0)
    worker.start()
    try:
        assert worker.submit_context(
            context=StructuredContext(timestamp=time.time()),
            mode=_make_mode_info(),
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            internet_access=False,
            internet_policy_summary="",
            use_tool_prompting=True,
            triggering_input="go",
            deliberation_available=False,
        )
        _wait_for_proposal(worker)
        assert llm.prompts and not any(t in llm.prompts[0] for t in THINK_OFFERS)
    finally:
        worker.stop()


@pytest.mark.xfail(strict=True, reason="red gate #1052: the planning submit does not say whether the loop deliberates")
def test_the_loop_says_it_deliberates_only_when_it_has_a_pipeline() -> None:
    from maxim.runtime import agent_loop

    src = inspect.getsource(agent_loop.run_agentic_loop)
    planning_submit = src[src.index("submitted = llm_worker.submit_context(") :][:2500]
    assert "deliberation_available=bio_enrichment_pipeline is not None" in planning_submit


@pytest.mark.xfail(strict=True, reason="red gate #1052: a resumed session lacks the fresh one's first-action line")
def test_a_resumed_session_is_told_to_act_first_as_a_fresh_one_is() -> None:
    from maxim.simulation import orchestrator
    from maxim.simulation.sim_types import FIRST_ACTION_INSTRUCTION, build_resume_prompt

    resumed = build_resume_prompt({"goal": "escape a dungeon"}, "you are in a peaceful garden", "generative")
    assert FIRST_ACTION_INSTRUCTION in resumed
    assert FIRST_ACTION_INSTRUCTION.split(".")[0] in inspect.getsource(orchestrator), "the same line a fresh one gets"
