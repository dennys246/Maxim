"""#1689: no canned answer before the model.

``PromptBuilder.build_prompt`` used to answer short time, date, identity, greeting and wellbeing questions itself and
hand the worker a JSON action disguised as the prompt (``'{"action":...'``), which the worker returned as a
``direct_answer`` proposal without calling the model. Keyword matching dropped real requests ("hi, can you look to
your left" got "Hello! How can I help you?") and the canned reply was credited as a decision. Owner decision
2026-11-17: delete it, matcher and all (the review showed the fallback, spoken outside a sim on any unparseable
reply, would have brought the misfires back).
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


@pytest.mark.parametrize("text", SHORTCUT_INPUTS)
def test_the_fallback_never_keyword_answers(text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback runs on any missing or unparseable model reply, and outside a sim it is spoken as the turn's
    answer. So it must not bring the deleted keyword answers back: it says it is having trouble (review fold, #1689)."""
    from maxim.agents import llm_fallback

    monkeypatch.setattr(llm_fallback, "get_fallback_responses", lambda: {})  # the shipped default templates
    action = llm_fallback.generate_llm_fallback(_request(text))
    assert action is not None and action["tool_name"] == "respond"
    message = action["params"]["message"]
    for canned in ("The current time is", "Today is", "I'm Maxim, a robot assistant", "How can I help you"):
        assert canned not in message, f"the fallback keyword-answered {text!r}: {message!r}"


def test_the_matcher_is_gone() -> None:
    from maxim.agents import llm_fallback, llm_worker

    for name in ("generate_simple_answer", "matches_phrase", "normalize_phrases"):
        assert not hasattr(llm_fallback, name) and not hasattr(llm_worker, name), name


# ── the no-tool (ANSWER_ONLY) path carries the clock too (review fold) ──────────────────────────────────────────────


def test_the_answer_only_prompt_carries_the_date_and_time() -> None:
    """With no tool prompt, ``build_prompt`` returns ``ANSWER_ONLY|``. That path once relied on the canned answer for
    "what time is it"; its model call now gets the builder's clock line, before a blank line and the question."""
    from datetime import datetime

    req = _request("maxim, what time is it?")
    req.use_tool_prompting = False
    before = datetime.now().strftime("%A, %B %d, %Y")
    prompt = _worker(_RecordingLLM())._build_prompt(req)
    after = datetime.now().strftime("%A, %B %d, %Y")
    assert prompt.startswith("ANSWER_ONLY|Today is ")
    clock, sep, question = prompt[len("ANSWER_ONLY|") :].partition("\n\n")
    assert sep and question == "what time is it"
    assert (before in clock or after in clock) and "The time is " in clock


def test_the_router_puts_the_clock_before_the_question() -> None:
    from maxim.models.language.router import LLMRouter

    router = LLMRouter.__new__(LLMRouter)  # only generate_json's ANSWER_ONLY dispatch is exercised
    seen: dict[str, Any] = {}

    def fake_answer_only(backend: Any, question: str, temperature: float, max_tokens: int, **kw: Any) -> dict:
        seen.update(question=question, **kw)
        return {"action": None}

    router.enabled = lambda: True  # type: ignore[method-assign]
    router._get_backend = lambda: object()  # type: ignore[method-assign]
    router._generate_answer_only = fake_answer_only  # type: ignore[method-assign]
    router.generate_json("ANSWER_ONLY|Today is Monday. The time is 09:00 AM.\n\nwhat time is it")
    assert seen["question"] == "what time is it" and seen["context"] == "Today is Monday. The time is 09:00 AM."
    seen.clear()
    router.generate_json("ANSWER_ONLY|what time is it")  # a bare payload still works
    assert seen["question"] == "what time is it" and seen["context"] == ""


def test_the_answer_only_model_call_puts_the_clock_line_before_the_question() -> None:
    """The real ``_generate_answer_only`` sends the model the clock line, a blank line, then the question."""
    from maxim.models.language.router import LLMRouter

    router = LLMRouter.__new__(LLMRouter)  # only _generate_answer_only runs; the model call is faked
    users: list[str] = []

    def fake_complete_text(system: str, user: str, **kw: Any) -> tuple[str, dict]:
        users.append(user)
        return "Answer: nine o'clock", {}

    router._complete_text = fake_complete_text  # type: ignore[method-assign]
    out = router._generate_answer_only(object(), "what time is it", 0.0, 64, context="Today is Monday.")
    assert users[-1] == "Today is Monday.\n\nQuestion: what time is it\nAnswer:"
    assert out is not None and out["action"]["params"]["message"] == "nine o'clock"
    router._generate_answer_only(object(), "what time is it", 0.0, 64)
    assert users[-1] == "Question: what time is it\nAnswer:"  # no context, no leading blank line
