"""#1042 PR B: an agent's prompt names only tools it has.

O19 Exp 10 attempt 1 (#1041) aborted because the narrator (tools: send_message, observe_actions, check_completion)
was shown other tools' names and proposed them: the full prompt's hard-coded tool guidance (``internet_search``
three times, ``write_file``/``read_file``/``speak``/``respond``), a "REAL-TIME DATA" hint fired by the substring
"now" in "Call send_message NOW", and a first follow-up attempt that never listed its own tools. Owner decisions
2026-10-01: every agent's prompt is gated on its own roster (unchanged text for an agent that has the tools), and a
follow-up lists the agent's own tools on its first attempt too.
"""

from __future__ import annotations

import inspect

import pytest

from maxim.agents.autonomy import AutonomyLevel
from maxim.agents.bus import StructuredContext
from maxim.agents.llm_types import LLMRequest, ModeInfo
from maxim.agents.prompt_builder import (
    PromptBuilder,
    build_tool_guidance_core,
    build_tool_guidance_extended,
    is_realtime_request,
)

NARRATOR_TOOLS = {"send_message", "observe_actions", "check_completion"}
FOREIGN = ("internet_search", "write_file", "read_file", "speak", "respond", "math", "focus_interests", "track_target")
# The stall nudge the narrator received in attempt 1 (orchestrator.py's _stall_detector).
STALL = (
    "SYSTEM: Stall detected (30s idle, 1 AUT actions so far). Last AUT action was 'sense_tools' (blocked=False). "
    "Call send_message NOW with a different probe."
)
FOLLOWUP = (
    "[ACTION_FOLLOWUP type=process tool=send_message mode=live query='talk to the agent']: "
    "{'actions': [{'tool': 'sense_tools', 'output': 'Your capabilities: look, listen'}]}"
)


def _builder() -> PromptBuilder:
    from maxim.agents.llm_fallback import ReasoningCarryover
    from maxim.models.language.token_counter import CharEstimateCounter

    return PromptBuilder(
        llm=None,
        reasoning_carryover=ReasoningCarryover(),
        n_ctx=32000,
        token_counter=CharEstimateCounter(),
        tool_index=None,
    )


def _request(tools: set[str], *, triggering_input: str = "", cli_inputs: list[str] | None = None) -> LLMRequest:
    ctx = StructuredContext(timestamp=1_700_000_000.0)
    ctx.cli_inputs = list(cli_inputs or [])
    return LLMRequest(
        request_id="r1",
        context=ctx,
        mode=ModeInfo(name="live", goal="narrate", context_prompt=""),
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        internet_access=False,
        internet_policy_summary="",
        available_tools=set(tools),
        triggering_input=triggering_input,
        use_tool_prompting=True,  # the narrator's planning calls use the full tool-aware prompt
    )


def _named(prompt: str, tool: str) -> bool:
    return any(f"{q}{tool}{q}" in prompt for q in ("'", '"')) or f"- {tool}:" in prompt


@pytest.mark.xfail(strict=True, reason="red gate #1042 PR B: the full prompt hard-codes tools the narrator lacks")
def test_the_narrators_full_prompt_names_no_tool_it_lacks() -> None:
    prompt = _builder().build_prompt(_request(NARRATOR_TOOLS, triggering_input=STALL))
    leaked = [t for t in FOREIGN if _named(prompt, t)]
    assert leaked == [], leaked


@pytest.mark.xfail(strict=True, reason="red gate #1042 PR B: 'now' matched as a substring of 'NOW'")
def test_a_stall_nudge_is_not_a_real_time_request() -> None:
    assert is_realtime_request(STALL) is False
    assert is_realtime_request("what's the Broncos score now?") is True  # a real one still is


@pytest.mark.xfail(strict=True, reason="red gate #1042 PR B: the first follow-up attempt lists no own tools")
def test_a_first_follow_up_lists_the_agents_own_tools() -> None:
    prompt = _builder().build_prompt(_request(NARRATOR_TOOLS, cli_inputs=[FOLLOWUP]))
    assert all(f"'{t}'" in prompt for t in NARRATOR_TOOLS), prompt[-400:]
    assert not _named(prompt, "respond"), "the narrator has no 'respond'"


@pytest.mark.xfail(strict=True, reason="red gate #1042 PR B: the guidance takes no roster")
def test_an_agent_with_every_tool_sees_unchanged_guidance() -> None:
    every = set(FOREIGN) | {"send_message"}
    for mode in ("passive", "active", "singularity"):
        assert build_tool_guidance_core(mode_name=mode, tools=every) == build_tool_guidance_core(mode_name=mode)
        assert build_tool_guidance_extended(mode_name=mode, tools=every) == build_tool_guidance_extended(mode_name=mode)


@pytest.mark.xfail(strict=True, reason="red gate #1042 PR B: the narrator's inputs quote the AUT's tools unlabelled")
def test_the_narrators_stall_and_diversity_inputs_label_the_agents_tools() -> None:
    from maxim.simulation import orchestrator

    src = inspect.getsource(orchestrator)
    assert "the agent under test's last action was" in src
    assert "the agent under test's tools so far (not yours)" in src
