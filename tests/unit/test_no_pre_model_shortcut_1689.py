"""#1689: no canned answer before the model.

``PromptBuilder.build_prompt`` used to answer short time, date, identity, greeting and wellbeing questions itself and
hand the worker a JSON action disguised as the prompt (``'{"action":...'``), which the worker returned as a
``direct_answer`` proposal without calling the model. Keyword matching dropped real requests ("hi, can you look to
your left" got "Hello! How can I help you?") and the canned reply was credited as a decision. Owner decision
2026-11-17: delete it. The canned answers now live only in the model-unavailable fallback.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

# Each one got a canned reply, with no model call, before #1689 (measured on a7825aa1 against
# ``llm_fallback.generate_simple_answer``).
SHORTCUT_INPUTS = [
    "what time is it in tokyo",
    "hi, can you look to your left",
    "hey, what's on the table",
    "who are you talking to",
    "what day did we meet",
    "remind me what time the meeting is",
    "maxim, what time is it?",
    "hello",
]


class _RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_json(self, prompt: str, *_a: Any, **_k: Any) -> dict[str, Any] | None:
        self.prompts.append(prompt)
        return {
            "action": {"tool_name": "respond", "params": {"message": "from the model"}},
            "reasoning": "reply",
            "strategy": "assist",
            "confidence": 0.9,
            "mode_goal_achieved": False,
        }


def _request(text: str) -> Any:
    from maxim.agents.autonomy import AutonomyLevel
    from maxim.agents.bus import StructuredContext
    from maxim.agents.llm_types import LLMRequest, ModeInfo

    return LLMRequest(
        request_id="r1689",
        context=StructuredContext(timestamp=time.time()),
        mode=ModeInfo(name="active", goal="assist", context_prompt=""),
        autonomy_level=AutonomyLevel.SUPERVISED,
        internet_access=False,
        internet_policy_summary="",
        available_tools={"respond"},
        tool_descriptions={"respond": "reply"},
        use_tool_prompting=True,
        triggering_input=text,
        deliberation_available=False,
    )


def _worker(llm: _RecordingLLM) -> Any:
    from maxim.agents.llm_worker import LLMWorker

    return LLMWorker(llm=llm, n_ctx=8192)


@pytest.mark.parametrize("text", SHORTCUT_INPUTS)
def test_a_short_question_builds_a_real_prompt_that_carries_it(text: str) -> None:
    llm = _RecordingLLM()
    prompt = _worker(llm)._build_prompt(_request(text))
    assert not prompt.lstrip().startswith("{"), f"build_prompt answered {text!r} itself: {prompt[:120]!r}"
    core = text.lower().replace("maxim", "").strip().lstrip(",").strip().rstrip("?").strip()
    assert core in prompt.lower(), f"the request {core!r} is not in the prompt"


@pytest.mark.parametrize("text", SHORTCUT_INPUTS)
def test_the_model_answers_a_short_question(text: str) -> None:
    """The shipped path end to end: one model call, and the model's answer is the proposal."""
    llm = _RecordingLLM()
    worker = _worker(llm)
    worker.start()  # the model call runs on the worker's pool
    try:
        proposal = worker._process_request(_request(text))
    finally:
        worker.stop()
    assert len(llm.prompts) == 1, "the model was not called"
    assert proposal is not None and proposal.strategy_used != "direct_answer"
    assert proposal.action == {"tool_name": "respond", "params": {"message": "from the model"}}


def test_the_unavailable_model_fallback_still_tells_the_time() -> None:
    from maxim.agents.llm_fallback import generate_llm_fallback

    action = generate_llm_fallback(_request("maxim, what time is it?"))
    assert action is not None and action["tool_name"] == "respond"
    assert action["params"]["message"].startswith("The current time is ")


def test_the_unavailable_model_fallback_still_tells_the_date() -> None:
    from maxim.agents.llm_fallback import generate_llm_fallback

    action = generate_llm_fallback(_request("what's the date"))
    assert action is not None and action["params"]["message"].startswith("Today is ")
