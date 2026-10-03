"""#935 -- a follow-up retry after an unregistered tool carries its correction (D13's retry rule).

`LLMWorker.requeue_request` records the rejected name in `request.failed_tools`, but `PromptBuilder.build_prompt`
returned the follow-up prompt early and never read it, so every retry of a follow-up was byte-identical and a weak
model kept echoing the same unavailable name until the planning-liveness budget aborted the run (`planning_failed`
on every Sim-Short re-run since August). The narrator, whose follow-up prompt never lists its own tools, echoed the
AUT's `sense_tools` from the result it had just read.

Owner decision (2026-09-30): the correction is retry-only; only a retry after a rejected tool carries it. (Since
#1042, 2026-10-01, a "choose your next tool" follow-up lists the agent's own tools on its first attempt too.) And the
"call 'sense_tools'" hint appears only for an agent that has `sense_tools`.
"""

from __future__ import annotations

from maxim.agents.autonomy import AutonomyLevel
from maxim.agents.bus import StructuredContext
from maxim.agents.llm_types import LLMRequest, ModeInfo
from maxim.agents.llm_worker import LLMWorker
from maxim.agents.prompt_builder import PromptBuilder, build_failed_tools_section

NARRATOR_TOOLS = {"send_message", "observe_actions", "check_completion"}
# The follow-up the narrator received in session 20260927_110748 (#935): its send_message result reports the AUT's
# sense_tools listing, so the AUT's tool names are in the narrator's prompt and its own are not.
FOLLOWUP = (
    "[ACTION_FOLLOWUP type=process tool=send_message mode=live query='talk to the agent']: "
    "{'actions': [{'tool': 'sense_tools', 'output': 'Your capabilities: look, listen'}]}"
)


def _builder() -> PromptBuilder:
    return PromptBuilder(llm=None, reasoning_carryover=None, n_ctx=32000, token_counter=len, tool_index=None)


def _request(tools: set[str]) -> LLMRequest:
    ctx = StructuredContext(timestamp=1_700_000_000.0)
    ctx.cli_inputs = [FOLLOWUP]
    return LLMRequest(
        request_id="r1",
        context=ctx,
        mode=ModeInfo(name="live", goal="narrate", context_prompt=""),
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        internet_access=False,
        internet_policy_summary="",
        available_tools=set(tools),
    )


def test_the_first_follow_up_attempt_lists_the_agents_own_tools() -> None:
    """Inverted 2026-10-01 (owner decision, #1042): every follow-up lists the agent's own tools, not only a retry.
    With no rejected tool the prompt is the follow-up template plus that list, and no correction."""
    first = _builder().build_prompt(_request(NARRATOR_TOOLS))
    own = first[first.index("=== Your Tools ===") :]
    assert all(repr(tool) in own for tool in NARRATOR_TOOLS)
    assert "=== Correction ===" not in first


def test_a_follow_up_retry_names_the_rejected_tool_and_the_agents_own_tools() -> None:
    builder = _builder()
    request = _request(NARRATOR_TOOLS)
    first = builder.build_prompt(request)
    LLMWorker._add_failed_tool_feedback(request, "sense_tools")
    retry = builder.build_prompt(request)
    assert retry != first, "a retry after a rejected tool is not a byte-identical resend"
    assert "'sense_tools'" in retry[len(first) :], "the correction names the rejected tool"
    assert all(tool in retry[len(first) :] for tool in NARRATOR_TOOLS), "and lists the agent's own tools"


def test_the_sense_tools_hint_is_only_for_an_agent_that_has_it() -> None:
    narrator = _request(NARRATOR_TOOLS)
    LLMWorker._add_failed_tool_feedback(narrator, "sense_tools")
    assert "call 'sense_tools'" not in build_failed_tools_section(narrator)
    aut = _request({"sense_tools", "look", "listen"})
    LLMWorker._add_failed_tool_feedback(aut, "fly")
    assert "call 'sense_tools'" in build_failed_tools_section(aut)


def test_the_worker_retry_carries_the_correction_into_the_next_prompt() -> None:
    """The shipped path: requeue_request -> _resubmit -> _process_request -> build_prompt, on the same request."""
    import time

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
            autonomy_level=AutonomyLevel.SUPERVISED,
            internet_access=False,
            internet_policy_summary="",
            use_tool_prompting=True,
            available_tools=set(NARRATOR_TOOLS),
            deliberation_available=False,  # the narrator's loop (#1052)
            deferred_inputs=[],
        )
        first = _wait_for_proposal(worker)
        assert first is not None and first.original_request is not None
        assert worker.requeue_request(first.original_request, failed_tool="sense_tools") is True
        assert _wait_for_proposal(worker) is not None
        assert len(llm.prompts) >= 2
        assert "=== Correction ===" not in llm.prompts[0]
        assert "You called 'sense_tools'" in llm.prompts[1] and "'observe_actions'" in llm.prompts[1]
    finally:
        worker.stop()


def test_a_correction_for_an_agent_with_no_listed_tools_still_names_the_rejected_one() -> None:
    request = _request(set())
    LLMWorker._add_failed_tool_feedback(request, "x" * 500)
    from maxim.agents.prompt_builder import build_followup_retry_correction

    correction = build_followup_retry_correction(request)
    assert "Your tools are" not in correction and repr("x" * 64) in correction and "x" * 65 not in correction
