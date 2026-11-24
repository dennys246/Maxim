"""#1775: a model's reply reaches a Python log record only as ``cloud_redaction.text_shape``.

A reply can quote what the model was shown, including a PRIVATE or SECRET tool's output fed back in a follow-up.
Log records persist (``MAXIM_LOG_FILE`` records DEBUG, plus the sim JSONL and the leader's ``/v1/debug/logs``), so
every site that copied a slice of a reply into a record now logs its length and a stable digest instead.

Every router test drives a REAL ``LLMRouter`` over a stub backend, as ``test_router_generate_1338`` does, and scans
every captured record, both its message and its structured ``data`` extra. Each test drives one path, including the
parse-failure branches. The agentic stream's sites (``llm_response``, ``goal_proposed``, ``plan_awaiting_approval``)
are pinned by ``test_loop_intake_characterization.py`` and ``test_plan_approval_dispatch_1185.py``, which expect
``text_shape`` values. A site on a path no test drives is outside these guards: the repo-wide AST sweep is
mechanization backlog M98.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

import pytest

from maxim.models.language.config import LLMConfig
from maxim.models.language.router import LLMRouter
from maxim.models.language.types import LLMResponse
from maxim.utils.cloud_redaction import text_shape

CANARY = "SECRET-CANARY-1775-do-not-log"


class _StubBackend:
    requires_prompt_formatting = False

    def __init__(self, content: str) -> None:
        self.content = content

    def complete_with_usage(self, **_kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self.content, input_tokens=3, output_tokens=2, model="fake-model", provider="p")


def _router(content: str) -> LLMRouter:
    cfg = dataclasses.replace(
        LLMConfig(),
        enabled=True,
        providers={
            "fake-peer": {
                "type": "maxim_peer",
                "base_url": "http://127.0.0.1:9999/v1",
                "api_key_env": "FAKE_KEY",
                "model": "fake-model",
                "allow_local_endpoints": True,
                "pricing_required": False,
            }
        },
    )
    router = LLMRouter(cfg)
    router._backends["fake-peer"] = _StubBackend(content)
    return router


def _record_text(r: logging.LogRecord) -> str:
    return r.getMessage() + " " + json.dumps(getattr(r, "data", None), default=str)


def _assert_no_canary(caplog: pytest.LogCaptureFixture) -> list[str]:
    texts = [_record_text(r) for r in caplog.records]
    leaked = [t for t in texts if CANARY in t]
    assert not leaked, f"a reply's text reached a log record: {leaked}"
    return texts


def _assert_shape_logged(texts: list[str], marker: str, value: str) -> None:
    assert any(marker in t and text_shape(value) in t for t in texts), (
        f"no {marker!r} record carried the shape of {value!r}; records: {texts}"
    )


# ── the seam: every reply, every path ─────────────────────────────────────


@pytest.mark.parametrize(
    "prompt, reply",
    [
        ("What next?", json.dumps({"action": "wait", "reasoning": f"the file said {CANARY}"})),
        ("ANSWER_ONLY|What does the note say?", f"The note says {CANARY}."),
        ("TOOL_PROMPT|Tools: respond\n\nSay what you read.", json.dumps({"tool": "respond", "params": {"m": CANARY}})),
    ],
    ids=["json", "answer_only", "tool_prompt"],
)
def test_every_reply_is_logged_once_at_the_seam_by_shape(
    caplog: pytest.LogCaptureFixture, prompt: str, reply: str
) -> None:
    with caplog.at_level(logging.DEBUG):
        out = _router(reply).generate_json(prompt)
    assert out is not None and CANARY in json.dumps(out), "the reply did not come back: the path was not exercised"
    texts = _assert_no_canary(caplog)
    _assert_shape_logged(texts, "LLM reply", reply)


def test_generate_reply_is_logged_at_the_seam_by_shape(caplog: pytest.LogCaptureFixture) -> None:
    reply = f"Halt. The ledger reads {CANARY}."
    with caplog.at_level(logging.DEBUG):
        assert _router(reply).generate("Who goes there?") == reply
    _assert_shape_logged(_assert_no_canary(caplog), "LLM reply", reply)


# ── the parse-failure branches (found by the review round) ────────────────


def test_a_non_json_reply_on_the_json_path_is_warned_by_shape(caplog: pytest.LogCaptureFixture) -> None:
    reply = f"I cannot comply; the file said {CANARY}"
    with caplog.at_level(logging.DEBUG):
        assert _router(reply).generate_json("What next?") is None
    _assert_shape_logged(_assert_no_canary(caplog), "LLM returned non-dict", reply)


def test_a_salvaged_message_is_logged_by_shape(caplog: pytest.LogCaptureFixture) -> None:
    message = f"hello there {CANARY}"
    reply = f'Sure: "message": "{message}"'
    with caplog.at_level(logging.DEBUG):
        out = _router(reply).generate_json("TOOL_PROMPT|Tools: respond\n\nSay hello.")
    assert out is not None and CANARY in json.dumps(out), "the salvage branch did not run"
    _assert_shape_logged(_assert_no_canary(caplog), "Salvaged message", message)


def test_a_synthetic_planning_proposal_is_logged_by_shape(caplog: pytest.LogCaptureFixture) -> None:
    reply = json.dumps({"action": {"tool_name": "internet_search", "params": {"query": CANARY}}})
    with caplog.at_level(logging.DEBUG):
        out = _router(reply).generate_json("TOOL_PROMPT|PLANNING MODE: ask for APPROVAL first.\n\nFind it.")
    assert out is not None and CANARY in out.get("_plan_text", ""), "the synthetic-proposal branch did not run"
    _assert_shape_logged(_assert_no_canary(caplog), "Generated synthetic proposal", out["_plan_text"])


# ── the other sites outside the router ────────────────────────────────────


def test_npc_reply_is_logged_by_shape_only(caplog: pytest.LogCaptureFixture) -> None:
    from maxim.runtime.agent_pool import AgentPool

    reply = f"I read {CANARY} on the wall."
    pool = AgentPool(bus=object(), llm_router=_router(reply))
    with caplog.at_level(logging.DEBUG):
        out = pool._generate_llm_response("npc-1", "gruff", "What did you see?")
    assert out == reply
    _assert_shape_logged(_assert_no_canary(caplog), "NPC 'npc-1' LLM response", reply)


def test_a_scenario_reply_that_fails_to_parse_is_logged_by_shape(caplog: pytest.LogCaptureFixture) -> None:
    from maxim.simulation.simulation_generator import _extract_json

    broken = f'{{"scenario": "{CANARY}", }}'
    with caplog.at_level(logging.DEBUG):
        assert _extract_json(f"Here it is: {broken}") is None
    _assert_shape_logged(_assert_no_canary(caplog), "JSON extraction failed", broken)


def test_a_critique_s_issues_are_logged_as_a_count() -> None:
    from maxim.agents.exec_agent import _issue_count

    assert _issue_count({"issues": [CANARY, "two"]}) == 2
    assert _issue_count({"issues": CANARY}) == 0
    assert _issue_count(None) == 0


# ── the sim stream's RESPONSE record (owner, 2026-11-24: stop mirroring its text) ──


def test_a_sim_response_reaches_the_log_file_by_shape_but_the_session_keeps_it(tmp_path: Any) -> None:
    from maxim.simulation import sim_logger

    reply = f"The note says {CANARY}."
    handler = logging.FileHandler(tmp_path / "maxim.jsonl")
    handler._maxim_jsonl = True  # type: ignore[attr-defined]
    captured: list[logging.LogRecord] = []
    handler.emit = captured.append  # type: ignore[method-assign]
    root = logging.getLogger()
    root.addHandler(handler)
    sim_logger.enable_sim_logging(use_color=False)
    try:
        sim_logger.sim_log("RESPONSE", reply, {"text": reply, "actions": ["respond"]}, _terminal=False)
        sim_logger.sim_log("NARRATOR", "the scene opens", _terminal=False)
        session = sim_logger.get_sim_records()
    finally:
        sim_logger.disable_sim_logging()
        root.removeHandler(handler)
        handler.close()

    responses = [r for r in session if r["subsystem"] == "RESPONSE"]
    assert responses and responses[0]["message"] == reply, "the session trail (the Talk view's source) lost the reply"
    texts = [_record_text(r) for r in captured]
    assert any("sim_response" == getattr(r, "event", None) for r in captured), "RESPONSE was not mirrored at all"
    assert not [t for t in texts if CANARY in t], f"a sim reply's text reached MAXIM_LOG_FILE: {texts}"
    _assert_shape_logged(texts, text_shape(reply), reply)
    assert any("the scene opens" in t for t in texts), "a non-reply subsystem lost its text"
