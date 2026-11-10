"""Typed failures the router can act on (#1370, #1369).

#1370: an error chunk in the middle of a stream (``data: {"error": ...}``) was ignored and the partial text came back
as a successful completion; a chunk that parses to JSON but is not an object raised a raw ``AttributeError`` out of the
typed contract. Both must raise a typed :class:`BackendError` and surface no partial text.

#1369: a request whose prompt overflows the model's context window (the leader proxy's 413, llama-cpp-python's 400,
vLLM's 400, the llama.cpp server's 400) became a generic ``BackendError`` and the router cooled down a healthy provider,
so the next, smaller prompt was refused too. It must become :class:`BackendContextOverflow` carrying the requested and
window token counts, and the router must not cool the provider down.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import logging
import os
import textwrap
import time
from unittest.mock import MagicMock, patch

import pytest

from maxim.models.language import types as _types
from maxim.models.language.config import LLMConfig
from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
from maxim.models.language.router import LLMRouter
from maxim.models.language.types import BackendDown, BackendError, ProviderState
from maxim.utils import http as _http
from maxim.utils.http import HTTPClientError, Response

BackendContextOverflow = getattr(_types, "BackendContextOverflow", None)

_RED = pytest.mark.xfail(strict=True, reason="red gate: #1370/#1369 not fixed yet")


# ─── Fixtures ──────────────────────────────────────────────────────────


def _make_backend() -> _MaximPeerBackend:
    cfg = dataclasses.replace(
        LLMConfig(),
        providers={
            "test-peer": {
                "type": "maxim_peer",
                "base_url": "http://127.0.0.1:9999/v1",
                "api_key_env": "TEST_MAXIM_PEER_KEY_1369",
                "model": "test-model",
                "allow_local_endpoints": True,
                "pricing_required": False,
            }
        },
    )
    os.environ["TEST_MAXIM_PEER_KEY_1369"] = "test-key"
    return _MaximPeerBackend(cfg, provider_key="test-peer")


class _FakeStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def iter_lines(self):
        yield from self._lines


def _sse(data: object) -> str:
    return f"data: {json.dumps(data)}"


def _content(text: str) -> str:
    return _sse({"choices": [{"delta": {"content": text}, "finish_reason": None}]})


def _client_error(status: int, body: bytes) -> HTTPClientError:
    err = HTTPClientError("peer-test-peer", status=status, fix_hint=f"Client error ({status})")
    err.response = Response(
        status=status, headers={}, content=body, elapsed_ms=1.0, endpoint="peer-test-peer", request_id="r"
    )
    return err


def _call(backend: _MaximPeerBackend, *, stream: bool):
    return backend.complete_with_usage(system="", user="hi", max_tokens=512, temperature=0.0, stream=stream)


# ─── #1370: a mid-stream error chunk ───────────────────────────────────


@_RED
@pytest.mark.parametrize(
    "error_payload",
    [
        {"message": "upstream failed mid-generation", "type": "server_error"},
        "upstream failed mid-generation",
        {"code": 500},
    ],
    ids=["object", "string", "code-only"],
)
def test_an_error_chunk_after_content_raises_and_returns_no_partial_text(error_payload) -> None:
    backend = _make_backend()
    lines = [_content("partial "), _content("answer"), _sse({"error": error_payload}), "data: [DONE]"]
    with patch.object(_http, "stream_post", return_value=_FakeStream(lines)):
        with pytest.raises(BackendDown) as excinfo:
            _call(backend, stream=True)
    assert "partial answer" not in str(excinfo.value)
    assert "answer" not in (excinfo.value.fix_hint or "")


@_RED
@pytest.mark.parametrize("chunk", ["[1, 2]", '"text"', "null", "42"], ids=["list", "string", "null", "number"])
def test_a_json_chunk_that_is_not_an_object_raises_backend_down(chunk: str) -> None:
    backend = _make_backend()
    lines = [_content("partial"), f"data: {chunk}"]
    with patch.object(_http, "stream_post", return_value=_FakeStream(lines)):
        with pytest.raises(BackendDown):
            _call(backend, stream=True)


# ─── #1369: context overflow, every server shape ──────────────────────

_OVERFLOW_SHAPES = {
    # The leader proxy's admission gate (runtime/leader_proxy.py::_check_context_admission).
    "leader-proxy-413": (
        413,
        {
            "error": {
                "code": "context_overflow",
                "type": "invalid_request_error",
                "message": "Estimated prompt tokens (9000) + max_tokens (512) + safety overhead (256) = 9768 exceeds",
                "estimated_prompt_tokens": 9000,
                "max_tokens": 512,
                "context_window": 8192,
                "safety_overhead_tokens": 256,
            }
        },
        9768,  # prompt + max_tokens + the proxy's safety overhead
        8192,
    ),
    # llama-cpp-python 0.3.x, chat completion (llama_cpp/server/errors.py::context_length_exceeded).
    "llama-cpp-python-chat": (
        400,
        {
            "error": {
                "message": (
                    "This model's maximum context length is 8192 tokens. However, you requested 9512 tokens "
                    "(9000 in the messages, 512 in the completion). Please reduce the length of the messages or "
                    "completion."
                ),
                "type": "invalid_request_error",
                "param": "messages",
                "code": "context_length_exceeded",
            }
        },
        9512,
        8192,
    ),
    # llama-cpp-python 0.3.x, text completion.
    "llama-cpp-python-text": (
        400,
        {
            "error": {
                "message": (
                    "This model's maximum context length is 8192 tokens, however you requested 9512 tokens "
                    "(9000 in your prompt; 512 for the completion). Please reduce your prompt; or completion length."
                ),
                "type": "invalid_request_error",
                "param": "messages",
                "code": "context_length_exceeded",
            }
        },
        9512,
        8192,
    ),
    # vLLM's OpenAI server: a top-level error object with no structured code.
    "vllm": (
        400,
        {
            "object": "error",
            "message": (
                "This model's maximum context length is 8192 tokens. However, you requested 9512 tokens "
                "(9000 in the messages, 512 in the completion). Please reduce the length of the messages or "
                "completion."
            ),
            "type": "BadRequestError",
            "param": None,
            "code": 400,
        },
        9512,
        8192,
    ),
    # The llama.cpp C++ server.
    "llama-cpp-server": (
        400,
        {
            "error": {
                "code": 400,
                "message": "the request exceeds the available context size, try increasing it",
                "type": "exceed_context_size_error",
                "n_prompt_tokens": 9000,
                "n_ctx": 8192,
            }
        },
        9000,
        8192,
    ),
}


@_RED
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
@pytest.mark.parametrize("shape", sorted(_OVERFLOW_SHAPES))
def test_a_context_overflow_becomes_a_typed_overflow_with_its_numbers(shape: str, stream: bool) -> None:
    status, body, requested, window = _OVERFLOW_SHAPES[shape]
    err = _client_error(status, json.dumps(body).encode())
    backend = _make_backend()
    with patch.object(_http, "stream_post" if stream else "post", side_effect=err):
        with pytest.raises(BackendContextOverflow) as excinfo:
            _call(backend, stream=stream)
    assert excinfo.value.status == status
    assert excinfo.value.requested_tokens == requested
    assert excinfo.value.context_window == window
    assert excinfo.value.provider_key == "test-peer"


@_RED
def test_an_overflow_shaped_error_chunk_mid_stream_is_a_typed_overflow() -> None:
    _status, body, requested, window = _OVERFLOW_SHAPES["llama-cpp-server"]
    lines = [_content("partial"), _sse(body)]
    with patch.object(_http, "stream_post", return_value=_FakeStream(lines)):
        with pytest.raises(BackendContextOverflow) as excinfo:
            _call(_make_backend(), stream=True)
    assert (excinfo.value.requested_tokens, excinfo.value.context_window) == (requested, window)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
@pytest.mark.parametrize(
    "body",
    [
        json.dumps({"error": {"message": "temperature must be >= 0", "type": "invalid_request_error"}}).encode(),
        b"<html>bad request</html>",
        b"",
        json.dumps({"error": {"message": "maximum context length is lovely", "code": "something_else"}}).encode(),
    ],
    ids=["other-400", "html", "empty", "near-miss-wording"],
)
def test_control_a_400_that_is_not_an_overflow_stays_a_plain_backend_error(body: bytes, stream: bool) -> None:
    err = _client_error(400, body)
    with patch.object(_http, "stream_post" if stream else "post", side_effect=err):
        with pytest.raises(BackendError) as excinfo:
            _call(_make_backend(), stream=stream)
    assert type(excinfo.value) is BackendError


# ─── #1369: the router ─────────────────────────────────────────────────


def _make_router() -> LLMRouter:
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
    router._provider_states.setdefault("fake-peer", ProviderState())
    router._backends["fake-peer"] = MagicMock()
    return router


def _try(router: LLMRouter, side_effect):
    with router._inference_lock:
        with patch.object(router, "_invoke_backend", side_effect=side_effect):
            return router._try_provider(
                provider_key="fake-peer",
                system="",
                user="hi",
                temperature=0.0,
                max_tokens=512,
                prompt_tokens=1,
                budget_tier="local_only",
                tools=None,
                thinking=None,
                stream=False,
                request_context=None,
                now=0.0,
            )


def _overflow() -> BackendError:
    return BackendContextOverflow("fake-peer", status=413, requested_tokens=9512, context_window=8192)


@_RED
def test_the_router_does_not_cool_down_a_provider_for_a_context_overflow(caplog) -> None:
    router = _make_router()
    with patch.object(router, "_maybe_schedule_auto_drain") as drain:
        with caplog.at_level(logging.WARNING):
            result = _try(router, _overflow())
    assert result == ("", None, "failed")
    state = router._provider_states["fake-peer"]
    assert state.backoff_until == 0.0
    assert state.consecutive_errors == 0
    drain.assert_not_called()
    # The next, smaller prompt still finds the provider.
    providers, _tier, _totals = router._candidate_providers(100, 64, time.time())
    assert "fake-peer" in providers
    # Loud, with the numbers.
    loud = [r for r in caplog.records if r.levelno >= logging.WARNING and "9512" in r.getMessage()]
    assert loud and "8192" in loud[0].getMessage()
    attempt = router._dispatch_attempts[-1]
    assert attempt["outcome"] == "context_overflow"
    assert (attempt["requested_tokens"], attempt["context_window"]) == (9512, 8192)


@_RED
def test_the_router_catches_the_overflow_before_the_generic_backend_error() -> None:
    src = textwrap.dedent(inspect.getsource(LLMRouter._try_provider))
    order: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name):
            order.append(node.type.id)
    assert "BackendContextOverflow" in order
    assert order.index("BackendContextOverflow") < order.index("BackendError")


def test_control_a_generic_backend_error_still_cools_the_provider_down() -> None:
    router = _make_router()
    _try(router, BackendError("fake-peer", status=400))
    assert router._provider_states["fake-peer"].backoff_until > 0.0
