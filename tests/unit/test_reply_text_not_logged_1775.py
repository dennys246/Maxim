"""#1775: no log record carries the text of a model's reply, at any level.

A reply can quote what the model was shown, including a PRIVATE or SECRET tool's output fed back in a follow-up.
Log records persist (``MAXIM_LOG_FILE`` records DEBUG, plus the sim JSONL and the leader's ``/v1/debug/logs``), so
each path that used to log the reply's first N characters now logs ``cloud_redaction.text_shape`` (length and a
stable digest). Every test drives a REAL ``LLMRouter`` over a stub backend, as ``test_router_generate_1338`` does, and
scans every record captured at DEBUG, not only the line that was changed: a new site that logs the reply fails here.
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


def _assert_reply_not_logged(caplog: pytest.LogCaptureFixture, reply: str, *, shape_logged: str) -> None:
    messages = [r.getMessage() for r in caplog.records]
    leaked = [m for m in messages if CANARY in m]
    assert not leaked, f"the reply's text reached a log record: {leaked}"
    assert any(shape_logged in m and text_shape(reply) in m for m in messages), (
        f"no {shape_logged!r} record carried the reply's shape; records: {messages}"
    )


def test_json_path_logs_the_reply_shape_only(caplog: pytest.LogCaptureFixture) -> None:
    reply = json.dumps({"action": "wait", "reasoning": f"the file said {CANARY}"})
    with caplog.at_level(logging.DEBUG):
        out = _router(reply).generate_json("What next?")
    assert out is not None and CANARY in json.dumps(out), "the reply did not come back: the path was not exercised"
    _assert_reply_not_logged(caplog, reply, shape_logged="LLM raw response")


def test_answer_only_path_logs_the_answer_shape_only(caplog: pytest.LogCaptureFixture) -> None:
    answer = f"The note says {CANARY}."
    with caplog.at_level(logging.DEBUG):
        out = _router(answer).generate_json("ANSWER_ONLY|What does the note say?")
    assert out is not None and CANARY in json.dumps(out), "the answer did not come back: the path was not exercised"
    _assert_reply_not_logged(caplog, answer, shape_logged="LLM answer_only response")


def test_tool_response_path_logs_the_reply_shape_only(caplog: pytest.LogCaptureFixture) -> None:
    reply = json.dumps({"tool": "respond", "params": {"message": CANARY}})
    with caplog.at_level(logging.DEBUG):
        _router(reply).generate_json("TOOL_PROMPT|Tools: respond\n\nSay what you read.")
    _assert_reply_not_logged(caplog, reply, shape_logged="LLM tool_response raw")


def test_npc_reply_is_logged_by_shape_only(caplog: pytest.LogCaptureFixture) -> None:
    from maxim.runtime.agent_pool import AgentPool

    reply = f"I read {CANARY} on the wall."
    pool = AgentPool(bus=object(), llm_router=_router(reply))
    with caplog.at_level(logging.DEBUG):
        out = pool._generate_llm_response("npc-1", "gruff", "What did you see?")
    assert out == reply
    _assert_reply_not_logged(caplog, reply, shape_logged="LLM response")
