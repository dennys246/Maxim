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

The red-gate commit marked these strict xfail; the fix flips them.
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


def test_a_loop_that_cannot_deliberate_is_not_offered_it(in_sim) -> None:
    prompt = _builder().build_prompt(_request(deliberation_available=False))
    offered = [t for t in THINK_OFFERS if t in prompt]
    assert offered == [], offered
    assert '"action": {"tool_name"' in prompt, "it is still told how to act"


def test_a_loop_that_deliberates_sees_the_prompt_unchanged(in_sim) -> None:
    today = _builder().build_prompt(_request())
    assert _builder().build_prompt(_request(deliberation_available=True)) == today
    assert all(t in today for t in THINK_OFFERS), "today's prompt offers deliberation (the AUT keeps it)"


@pytest.mark.parametrize("deliberation_available, offered", [(False, False), (None, True)])
def test_the_worker_carries_the_loops_answer_into_the_prompt(in_sim, deliberation_available, offered) -> None:
    """Through the real worker: False removes the offer; None (today's callers) keeps it (the positive control)."""
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
            available_tools={"send_message", "observe_actions"},
            use_tool_prompting=True,
            triggering_input="go",
            deliberation_available=deliberation_available,
            deferred_inputs=[],
        )
        _wait_for_proposal(worker)
        assert llm.prompts
        assert any(t in llm.prompts[0] for t in THINK_OFFERS) is offered
    finally:
        worker.stop()


def test_the_loop_says_it_deliberates_only_when_it_has_a_pipeline() -> None:
    from maxim.runtime import agent_loop

    src = inspect.getsource(agent_loop.run_agentic_loop)
    planning_submit = src[src.index("submitted = llm_worker.submit_context(") :][:2500]
    assert "deliberation_available=bio_enrichment_pipeline is not None" in planning_submit


@pytest.mark.parametrize(
    "goal, observe_only, expect",
    [
        ("you are in a peaceful garden", False, "Your FIRST action MUST be send_message"),
        ("CAMPAIGN PROTOCOL: the heist", False, "send the NEXT campaign turn verbatim"),
        ("interactive", True, "OBSERVE ONLY"),
    ],
)
def test_a_resumed_session_gets_the_same_first_action_line_as_a_fresh_one(goal, observe_only, expect) -> None:
    """One chooser for both: the fresh kickoff calls it, and so does the resume prompt (every mode)."""
    from maxim.simulation import orchestrator
    from maxim.simulation.sim_types import build_resume_prompt, kickoff_instruction

    line = kickoff_instruction(goal, observe_only=observe_only, resumed=True)
    assert expect in line
    if "CAMPAIGN" in goal:  # a fresh campaign starts at its first turn
        assert "FIRST campaign turn" in kickoff_instruction(goal, observe_only=False)
    resumed = build_resume_prompt(
        {"goal": "escape a dungeon"}, goal, "generative", observe_only=observe_only, tools_block=""
    )
    assert resumed.endswith(line)
    if observe_only:
        assert "send_message NOW" not in resumed and "continue probing" not in resumed
    src = inspect.getsource(orchestrator.start_simulation_mode)
    # Every opening goes through sim_types (the fresh kickoff and the not-found fallback share build_kickoff_prompt,
    # which calls kickoff_instruction; the resume prompt calls it with resumed=True).
    assert "_build_kickoff_prompt(goal, tools_block=_tools_block, observe_only=_is_observe_only)" in src
    assert "observe_only=_is_observe_only, tools_block=_tools_block" in src
    assert "OBSERVE ONLY" not in src, "the kickoff text lives in one place"


def test_submit_context_requires_the_deliberation_fact() -> None:
    """Forgetting it is a TypeError, not a silent return of #1052 (owner decision 2026-10-02)."""
    from maxim.agents.llm_worker import LLMWorker

    param = inspect.signature(LLMWorker.submit_context).parameters["deliberation_available"]
    assert param.default is inspect.Parameter.empty and param.kind is inspect.Parameter.KEYWORD_ONLY


def test_every_submit_in_the_loop_states_the_deliberation_fact() -> None:
    """A required keyword is a TypeError only when the call runs: the deliberation-cycle submit runs only for an AUT
    whose PFC gate passed, so an omission there would crash the AUT mid-deliberation, unseen by most tests. Every
    submit_context call in agent_loop states it, directly or in the kwargs dict it unpacks."""
    import ast

    from maxim.runtime import agent_loop

    tree = ast.parse(inspect.getsource(agent_loop))
    dicts = {
        t.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "dict"
        for t in node.targets
        if isinstance(t, ast.Name)
    }  # fmt: skip
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "submit_context"
    ]  # fmt: skip
    assert len(calls) >= 2
    # ``_submit_fn(_ctx, _kw=_submit_kwargs)`` unpacks its default: the dict literal the loop built.
    unpacked = {"_kw": "_submit_kwargs"}
    for call in calls:
        named = {k.arg for k in call.keywords if k.arg}
        for k in call.keywords:
            if k.arg is None and isinstance(k.value, ast.Name):
                source = dicts[unpacked.get(k.value.id, k.value.id)]
                named |= {kw.arg for kw in source.keywords}
        assert "deliberation_available" in named, ast.unparse(call)[:120]
