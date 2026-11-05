"""#1592: the diversity checkpoint must not hand the narrator the agent's tool names, and a retry after the narrator
proposed an agent tool must say whose tool it was and name the narrator's own.

The Exp 63b smoke (big-mac-mini, b1120188, 2026-11-03, mistral-7b) stopped at phase 0 with ``planning_failed``: the
stall detector's DIVERSITY CHECKPOINT listed the agent under test's tools by name (``sense_tools (1x), ...``), the
narrator proposed one of them 3 s later, and each of its three retries picked another. The checkpoint is a CLI input,
so it is data and the #1042 roster gate never sees it. Owner decision 2026-11-05: the checkpoint gives counts only and
points at the narrator's own ``observe_actions``.

The composed test drives the real ``start_simulation_mode`` through the orchestrator harness's
``agent_tool_after_checkpoint`` probe arm (``tests/unit/_orch_harness.py``): the real stall detector builds the
checkpoint over the AUT's real actions, the real loop drops the narrator's agent-tool proposal and requeues it
(``agent_loop`` -> ``loop_controller._handle_planning_failure`` -> ``LLMWorker.requeue_request``), and the real prompt
builder renders the retry. Only the LLM is faked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.unit._orch_harness import AGENT_TOOL, RUN_SIDE, assert_this_checkout, canonical_json, run_arm
from tests.unit.test_prompt_names_own_tools import _mentions, _narrator_roster

ARM = "agent_tool_after_checkpoint"
# The correction's sentence that says whose tool it was (``prompt_builder.PLANNING_CORRECTIONS``).
WHOSE = "the agent under test's"


def _run(workdir: Path) -> dict[str, Any]:
    assert_this_checkout()  # a worktree without PYTHONPATH would otherwise test main's code
    return json.loads(canonical_json(run_arm(ARM, workdir)))


def _checkpoints(trace: dict[str, Any]) -> list[str]:
    return [e["text"] for e in trace["events"] if e.get("text", "").startswith("DIVERSITY CHECKPOINT")]


@pytest.fixture(scope="module")
def composed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _run(tmp_path_factory.mktemp("orch_1592"))


@pytest.mark.timeout(240)
def test_the_run_reaches_the_retry_through_the_shipped_drop(composed: dict[str, Any]) -> None:
    """The arm's premise, not the fix: without it the two tests below would pass vacuously."""
    calls = [(c["kind"], c.get("executed"), (c["answer"] or {}).get("tool_name")) for c in composed["narrator_calls"]]
    assert ("followup", "observe_actions", AGENT_TOOL) in calls, calls
    assert calls[-1] == ("followup", "observe_actions", "finish_simulation"), calls
    aut_tools = [(c["answer"] or {}).get("tool_name") for c in composed["aut_calls"]]
    assert AGENT_TOOL in aut_tools, f"the AUT used the tool the narrator lacks: {aut_tools}"
    assert len(_checkpoints(composed)) == 1


@pytest.mark.timeout(240)
def test_the_checkpoint_names_no_tool_of_the_agent(composed: dict[str, Any]) -> None:
    (checkpoint,) = _checkpoints(composed)
    roster = _narrator_roster()
    assert _mentions(checkpoint, roster) == [], checkpoint
    assert AGENT_TOOL not in checkpoint, checkpoint
    assert "observe_actions" in checkpoint, "it points at the narrator's own tool for the details"


@pytest.mark.timeout(240)
def test_the_retry_says_the_tool_was_the_agents_and_names_the_narrators_own(composed: dict[str, Any]) -> None:
    retry = RUN_SIDE[ARM]["prompts"]["narrator"][-1]
    correction = retry[retry.index("=== Correction ===") :]
    assert WHOSE in correction, correction
    assert AGENT_TOOL in correction and "'send_message'" in correction, correction
    # The checkpoint rode the failed follow-up and rides its retry, as a held input: it must not re-prime the retry.
    # (The observe_actions RESULT above it lists the agent's actions: data the narrator asked for, not a prime.)
    (held,) = re.findall(r"=== New Instructions ===\n(.*?)(?:\n=== |\Z)", retry, re.S)
    assert held.startswith("DIVERSITY CHECKPOINT") and _mentions(held, _narrator_roster()) == [], held


def test_a_plan_retry_after_an_agent_tool_says_whose_it_was_and_names_the_narrators_own() -> None:
    """The plan path: the checkpoint arrived with no follow-up in flight, so it is the plan's request."""
    from maxim.agents.llm_worker import LLMWorker
    from tests.unit.test_narrator_reliability import _narrator_request, _prompt

    request = _narrator_request(followup=False)
    LLMWorker._add_failed_tool_feedback(request, AGENT_TOOL)
    LLMWorker._add_planning_correction(request, "unregistered_tool_proposed")
    prompt = _prompt(request)
    tail = prompt[prompt.index(AGENT_TOOL) :]
    assert WHOSE in prompt, prompt[-800:]
    assert "'send_message'" in tail, "the retry names the narrator's own tools after the rejected one"
