"""Unit tests for :class:`_MaximPeerBackend` (Plan 3 R2.5).

Covers:

- Typed exception mapping (HTTPError subclasses → BackendError subclasses),
  in specific-before-general order per the 2026-04-12 stage-2 probe
  review fix (auth must NOT map to BackendInferenceBroken).
- Streaming happy path + **mid-stream failure raises BackendDown** (the
  intentional contract difference vs ``_OpenAIBackend._stream_response``
  which collects partial output).
- Shutdown check before the HTTP call.
- ``LLMResponse`` parsing from OpenAI-compatible JSON.
- ``_parse_queue_depth`` header extraction.
- ``warmup()`` gating on api_key + base_url validation.
- ``complete()`` delegates to ``complete_with_usage()``.

The backend is intentionally thin: one HTTP call, one json.loads, one
LLMResponse. Anything more complex is either the router's responsibility
(failover, backoff) or the upstream's (retry, cost tracking, redaction).
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from maxim.models.language.config import LLMConfig
from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
from maxim.models.language.types import (
    BackendAuthFailed,
    BackendDown,
    BackendError,
    BackendModelMissing,
    BackendOverloaded,
    BackendTimeout,
    LLMResponse,
)
from maxim.utils import http as _http
from maxim.utils.http import (
    HTTPAuthError,
    HTTPClientError,
    HTTPConnectionError,
    HTTPError,
    HTTPRateLimited,
    HTTPServerError,
    HTTPTimeout,
    Response,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


def _make_backend(
    *,
    base_url: str = "http://127.0.0.1:9999/v1",
    api_key: str = "test-key",
    model: str = "test-model",
) -> _MaximPeerBackend:
    """Construct a backend with a minimal in-memory LLMConfig."""
    import dataclasses
    import os

    base = LLMConfig()
    cfg = dataclasses.replace(
        base,
        providers={
            "test-peer": {
                "type": "maxim_peer",
                "base_url": base_url,
                "api_key_env": "TEST_MAXIM_PEER_KEY",
                "model": model,
                "allow_local_endpoints": True,
                "pricing_required": False,
            }
        },
    )
    os.environ["TEST_MAXIM_PEER_KEY"] = api_key
    return _MaximPeerBackend(cfg, provider_key="test-peer")


def _make_response(status: int = 200, body: dict | None = None) -> Response:
    """Build a fake ``utils.http.Response``."""
    content = json.dumps(body or {}).encode("utf-8")
    return Response(
        status=status,
        headers={},
        content=content,
        elapsed_ms=10.0,
        endpoint="peer-test-peer",
        request_id="req-test",
    )


@pytest.fixture
def ok_body() -> dict[str, Any]:
    return {
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    }


# ─── Happy path ────────────────────────────────────────────────────────


class TestHappyPath:
    def test_complete_with_usage_returns_populated_llmresponse(self, ok_body):
        backend = _make_backend()
        mock_response = _make_response(200, ok_body)
        with patch.object(_http, "post", return_value=mock_response):
            resp = backend.complete_with_usage(
                system="you are helpful",
                user="hi",
                max_tokens=16,
                temperature=0.0,
            )
        assert isinstance(resp, LLMResponse)
        assert resp.content == "hello"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.cached_input_tokens == 2
        assert resp.uncached_input_tokens == 8
        assert resp.model == "test-model"
        assert resp.provider == "test-peer"
        assert resp.stop_reason == "stop"

    def test_complete_delegates_to_complete_with_usage(self, ok_body):
        backend = _make_backend()
        mock_response = _make_response(200, ok_body)
        with patch.object(_http, "post", return_value=mock_response):
            text = backend.complete(
                "hi",
                max_tokens=16,
                temperature=0.0,
                stop=(),
                system="you are helpful",
            )
        assert text == "hello"

    def test_warmup_passes_with_valid_config(self):
        backend = _make_backend()
        assert backend.warmup() is True

    def test_warmup_fails_without_api_key(self):
        backend = _make_backend(api_key="")
        assert backend.warmup() is False

    def test_warmup_fails_with_invalid_base_url(self):
        backend = _make_backend(base_url="not-a-valid-url")
        assert backend.warmup() is False


# ─── Typed exception mapping ───────────────────────────────────────────
#
# The most important tests in this file. These assert the
# specific-before-general ordering that was the root cause of the
# 2026-04-12 stage-2 probe bug (auth mis-classified as inference_broken
# because the generic handler came first).


class TestExceptionMapping:
    def test_401_maps_to_backend_auth_failed(self):
        backend = _make_backend()
        err = HTTPAuthError("peer-test-peer", status=401, fix_hint="bad key")
        with patch.object(_http, "post", side_effect=err):
            with pytest.raises(BackendAuthFailed) as excinfo:
                backend.complete_with_usage(system="", user="hi", max_tokens=1, temperature=0.0)
        assert excinfo.value.status == 401
        assert excinfo.value.provider_key == "test-peer"

    def test_403_maps_to_backend_auth_failed_not_inference_broken(self):
        """Regression guard for the 2026-04-12 stage-2 probe bug.

        Before the R2 review round, the stage-2 probe mapped 403 to
        ``inference_broken`` because ``except HTTPError`` came before
        ``except HTTPAuthError``. This test locks in the correct ordering
        for the backend's own exception handler.
        """
        backend = _make_backend()
        err = HTTPAuthError("peer-test-peer", status=403, fix_hint="forbidden")
        with patch.object(_http, "post", side_effect=err):
            with pytest.raises(BackendAuthFailed):
                backend.complete_with_usage(system="", user="hi", max_tokens=1, temperature=0.0)

    def test_429_maps_to_backend_overloaded_with_retry_after(self):
        backend = _make_backend()
        err = HTTPRateLimited(
            "peer-test-peer",
            status=429,
            retry_after_s=7.5,
            suggested_peer="rtx-leader",
        )
        err.response = Response(
            status=429,
            headers={"X-Maxim-Queue-Depth": "12"},
            content=b"",
            elapsed_ms=1.0,
            endpoint="peer-test-peer",
            request_id="r",
        )
        with patch.object(_http, "post", side_effect=err):
            with pytest.raises(BackendOverloaded) as excinfo:
                backend.complete_with_usage(system="", user="hi", max_tokens=1, temperature=0.0)
        assert excinfo.value.retry_after_s == 7.5
        assert excinfo.value.suggested_peer == "rtx-leader"
        assert excinfo.value.queue_depth == 12

    def test_429_without_response_headers_returns_zero_queue_depth(self):
        backend = _make_backend()
        err = HTTPRateLimited("peer-test-peer", status=429, retry_after_s=2.0)
        # No .response set — exercises the None branch in _parse_queue_depth
        with patch.object(_http, "post", side_effect=err):
            with pytest.raises(BackendOverloaded) as excinfo:
                backend.complete_with_usage(system="", user="hi", max_tokens=1, temperature=0.0)
        assert excinfo.value.queue_depth == 0

    def test_404_maps_to_backend_model_missing(self):
        backend = _make_backend()
        err = HTTPClientError("peer-test-peer", status=404, fix_hint="unknown model")
        with patch.object(_http, "post", side_effect=err):
            with pytest.raises(BackendModelMissing) as excinfo:
                backend.complete_with_usage(system="", user="hi", max_tokens=1, temperature=0.0)
        assert excinfo.value.requested_model == "test-model"

    def test_400_maps_to_generic_backend_error(self):
        backend = _make_backend()
        err = HTTPClientError("peer-test-peer", status=400, fix_hint="bad request")
        with patch.object(_http, "post", side_effect=err):
            with pytest.raises(BackendError) as excinfo:
                backend.complete_with_usage(system="", user="hi", max_tokens=1, temperature=0.0)
        # Not BackendModelMissing (404-specific)
        assert not isinstance(excinfo.value, BackendModelMissing)
        assert excinfo.value.status == 400

    def test_502_maps_to_backend_down(self):
        backend = _make_backend()
        err = HTTPServerError("peer-test-peer", status=502, fix_hint="upstream bad")
        with patch.object(_http, "post", side_effect=err):
            with pytest.raises(BackendDown) as excinfo:
                backend.complete_with_usage(system="", user="hi", max_tokens=1, temperature=0.0)
        assert excinfo.value.status == 502

    def test_timeout_maps_to_backend_timeout_with_elapsed_s(self):
        backend = _make_backend()
        err = HTTPTimeout("peer-test-peer", fix_hint="read timeout")
        with patch.object(_http, "post", side_effect=err):
            with pytest.raises(BackendTimeout) as excinfo:
                backend.complete_with_usage(system="", user="hi", max_tokens=1, temperature=0.0)
        assert excinfo.value.elapsed_s >= 0.0

    def test_connection_error_maps_to_backend_down(self):
        backend = _make_backend()
        err = HTTPConnectionError("peer-test-peer", fix_hint="refused")
        with patch.object(_http, "post", side_effect=err):
            with pytest.raises(BackendDown):
                backend.complete_with_usage(system="", user="hi", max_tokens=1, temperature=0.0)

    def test_unhandled_http_error_maps_to_generic_backend_error(self):
        backend = _make_backend()
        err = HTTPError("peer-test-peer", status=418, fix_hint="teapot")
        with patch.object(_http, "post", side_effect=err):
            with pytest.raises(BackendError) as excinfo:
                backend.complete_with_usage(system="", user="hi", max_tokens=1, temperature=0.0)
        assert excinfo.value.status == 418


# ─── Shutdown responsiveness ───────────────────────────────────────────


class TestShutdown:
    def test_shutdown_check_raises_backend_down_before_http_call(self):
        """The shutdown check runs BEFORE registering the endpoint or
        making the HTTP call so Ctrl+C fires immediately."""
        backend = _make_backend()
        with patch(
            "maxim.models.language.maxim_peer_backend.is_shutdown_requested",
            return_value=True,
        ):
            mock_post = MagicMock()
            with patch.object(_http, "post", mock_post):
                with pytest.raises(BackendDown) as excinfo:
                    backend.complete_with_usage(system="", user="hi", max_tokens=1, temperature=0.0)
            mock_post.assert_not_called()
            assert "shutdown" in (excinfo.value.fix_hint or "").lower()


# ─── Streaming ─────────────────────────────────────────────────────────


class _FakeStreamingResponse:
    """In-memory stand-in for :class:`utils.http.StreamingResponse`."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __enter__(self) -> "_FakeStreamingResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def iter_lines(self):
        yield from self._lines


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}"


class TestStreaming:
    def test_streaming_happy_path_collects_chunks(self):
        backend = _make_backend()
        lines = [
            _sse({"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}),
            _sse({"choices": [{"delta": {"content": " world"}, "finish_reason": None}]}),
            _sse(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                }
            ),
            "data: [DONE]",
        ]
        with patch.object(_http, "stream_post", return_value=_FakeStreamingResponse(lines)):
            resp = backend.complete_with_usage(
                system="",
                user="hi",
                max_tokens=16,
                temperature=0.0,
                stream=True,
            )
        assert isinstance(resp, LLMResponse)
        assert resp.content == "hello world"
        assert resp.stop_reason == "stop"
        assert resp.input_tokens == 7
        assert resp.output_tokens == 2

    def test_streaming_mid_stream_malformed_chunk_raises_backend_down(self):
        """Intentional contract difference vs _OpenAIBackend._stream_response.

        Where the cloud backend silently collects whatever content arrived
        before the error, the peer backend raises BackendDown so the
        router's fallback loop can dispatch to a different provider. See
        docs/architecture/llm_routing.md "Behaviors not obvious".
        """
        backend = _make_backend()
        lines = [
            _sse({"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]}),
            "data: {not-valid-json",  # malformed chunk mid-stream
        ]
        with patch.object(_http, "stream_post", return_value=_FakeStreamingResponse(lines)):
            with pytest.raises(BackendDown) as excinfo:
                backend.complete_with_usage(system="", user="hi", max_tokens=16, temperature=0.0, stream=True)
        assert "non-JSON" in (excinfo.value.fix_hint or "")

    def test_streaming_connection_error_mid_stream_raises_backend_down(self):
        """Mid-stream httpx read failure surfaces as BackendDown."""
        backend = _make_backend()

        class _ExplodingStream:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return None

            def iter_lines(self):
                yield _sse({"choices": [{"delta": {"content": "p"}, "finish_reason": None}]})
                raise HTTPConnectionError("peer-test-peer", fix_hint="socket closed")

        with patch.object(_http, "stream_post", return_value=_ExplodingStream()):
            with pytest.raises(BackendDown):
                backend.complete_with_usage(system="", user="hi", max_tokens=16, temperature=0.0, stream=True)

    def test_streaming_empty_content_raises_backend_down(self):
        """Zero-content stream is treated as a broken peer under the strict
        contract (unlike the cloud backend which returns empty LLMResponse)."""
        backend = _make_backend()
        lines = ["data: [DONE]"]
        with patch.object(_http, "stream_post", return_value=_FakeStreamingResponse(lines)):
            with pytest.raises(BackendDown):
                backend.complete_with_usage(system="", user="hi", max_tokens=16, temperature=0.0, stream=True)

    def test_streaming_auth_error_on_open_raises_backend_auth_failed(self):
        """Stream-open auth rejection surfaces as BackendAuthFailed, not
        BackendInferenceBroken — the same specific-before-general ordering
        that the non-streaming path enforces."""
        backend = _make_backend()
        err = HTTPAuthError("peer-test-peer", status=401, fix_hint="bad key")
        with patch.object(_http, "stream_post", side_effect=err):
            with pytest.raises(BackendAuthFailed):
                backend.complete_with_usage(system="", user="hi", max_tokens=1, temperature=0.0, stream=True)


# ─── Multi-agent context propagation ───────────────────────────────────


class TestRequestContext:
    def test_request_context_dict_is_normalized_and_threaded(self):
        """The backend delegates to
        ``agents.llm_worker._normalize_request_context`` — the canonical
        Plan 2 R2b shim. Verify the typed RequestContext lands on the
        http.post call."""
        backend = _make_backend()
        captured: dict[str, Any] = {}

        def _capture(*args, **kwargs):
            captured["context"] = kwargs.get("context")
            return _make_response(
                200,
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        with patch.object(_http, "post", side_effect=_capture):
            backend.complete_with_usage(
                system="",
                user="hi",
                max_tokens=1,
                temperature=0.0,
                request_context={
                    "agent_id": "npc-mother",
                    "session_id": "sim-42",
                    "request_id": "r-abc",
                    "lane": "large",
                },
            )
        ctx = captured["context"]
        assert ctx is not None
        assert ctx.agent_id == "npc-mother"
        assert ctx.session_id == "sim-42"
        assert ctx.request_id == "r-abc"
        assert ctx.lane == "large"

    def test_legacy_agent_key_is_normalized_to_agent_id(self):
        """The Plan 2 R2b shim bridges ``ctx["agent"]`` → ``agent_id``. This
        test locks in the compat behavior so a future rename of the
        legacy field gets caught."""
        backend = _make_backend()
        captured: dict[str, Any] = {}

        def _capture(*args, **kwargs):
            captured["context"] = kwargs.get("context")
            return _make_response(
                200,
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        with patch.object(_http, "post", side_effect=_capture):
            backend.complete_with_usage(
                system="",
                user="hi",
                max_tokens=1,
                temperature=0.0,
                request_context={"agent": "legacy-agent"},
            )
        assert captured["context"].agent_id == "legacy-agent"

    # ─── Plan 4 A.2: contextvar fallback when dict is None ────────────

    def test_contextvar_fallback_populates_context_when_dict_is_none(self):
        """Plan 4 A.2 regression guard: when complete_with_usage is called
        with request_context=None, it must fall through to the
        ``utils.http._current_context`` ContextVar rather than manufacturing
        an empty RequestContext with agent_id=None. This was the Phase D
        observability gap: the router was dropping the dict on the floor,
        so every peer_backend_call event logged agent_id=null. With the
        boundary set_context() wired in llm_worker, None-dict calls must
        see the bound ContextVar.
        """
        from maxim.utils.http import RequestContext, reset_context, set_context

        backend = _make_backend()
        captured: dict[str, Any] = {}

        def _capture(*args, **kwargs):
            captured["context"] = kwargs.get("context")
            return _make_response(
                200,
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        bound = RequestContext(
            request_id="r-from-contextvar",
            agent_id="npc-from-contextvar",
            session_id="sim-from-contextvar",
            lane="large",
        )
        token = set_context(bound)
        try:
            with patch.object(_http, "post", side_effect=_capture):
                backend.complete_with_usage(
                    system="",
                    user="hi",
                    max_tokens=1,
                    temperature=0.0,
                    # NOTE: request_context=None — this is the gap path
                )
        finally:
            reset_context(token)

        ctx = captured["context"]
        assert ctx is not None
        # Must be the bound ContextVar value, not a freshly-generated empty
        assert ctx.agent_id == "npc-from-contextvar"
        assert ctx.session_id == "sim-from-contextvar"
        assert ctx.request_id == "r-from-contextvar"
        assert ctx.lane == "large"

    def test_explicit_dict_still_wins_over_contextvar(self):
        """Precedence: explicit non-None request_context dict beats the
        bound ContextVar. The fallback only kicks in when the dict is
        None."""
        from maxim.utils.http import RequestContext, reset_context, set_context

        backend = _make_backend()
        captured: dict[str, Any] = {}

        def _capture(*args, **kwargs):
            captured["context"] = kwargs.get("context")
            return _make_response(
                200,
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        bound = RequestContext(
            request_id="r-contextvar",
            agent_id="from-contextvar",
        )
        token = set_context(bound)
        try:
            with patch.object(_http, "post", side_effect=_capture):
                backend.complete_with_usage(
                    system="",
                    user="hi",
                    max_tokens=1,
                    temperature=0.0,
                    request_context={
                        "agent_id": "from-explicit-dict",
                        "request_id": "r-explicit",
                    },
                )
        finally:
            reset_context(token)

        ctx = captured["context"]
        # Explicit dict wins — precedence locked in
        assert ctx.agent_id == "from-explicit-dict"
        assert ctx.request_id == "r-explicit"

    def test_supports_request_context_capability_flag_is_declared(self):
        """The router uses this flag to decide whether to forward the
        kwarg. Cloud backends omit it to avoid 'unexpected keyword
        argument' crashes. Regression guard: removing the flag would
        silently drop agent_id from peer_backend_call logs again."""
        backend = _make_backend()
        assert getattr(backend, "supports_request_context", False) is True


# ─── Parse helpers ─────────────────────────────────────────────────────


class TestParseHelpers:
    def test_parse_queue_depth_handles_missing_headers(self):
        backend = _make_backend()
        assert backend._parse_queue_depth(None) == 0
        assert backend._parse_queue_depth({}) == 0

    def test_parse_queue_depth_is_case_insensitive(self):
        backend = _make_backend()
        assert backend._parse_queue_depth({"X-Maxim-Queue-Depth": "3"}) == 3
        assert backend._parse_queue_depth({"x-maxim-queue-depth": "5"}) == 5

    def test_parse_queue_depth_handles_non_integer(self):
        backend = _make_backend()
        assert backend._parse_queue_depth({"X-Maxim-Queue-Depth": "not-a-number"}) == 0

    def test_parse_llm_response_with_no_usage(self):
        backend = _make_backend()
        raw = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
        resp = backend._parse_llm_response(raw, model="m", start=time.monotonic())
        assert resp.content == "x"
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0

    def test_non_json_body_raises_backend_inference_broken(self):
        """A 200 OK with malformed body means the listener is alive but the
        chat endpoint is broken — classify as inference_broken so the
        router applies the short (15s) cooldown."""
        from maxim.models.language.types import BackendInferenceBroken

        backend = _make_backend()
        # Construct a Response with invalid JSON content
        bad = Response(
            status=200,
            headers={},
            content=b"<html>oops</html>",
            elapsed_ms=10.0,
            endpoint="peer-test-peer",
            request_id="r",
        )
        with patch.object(_http, "post", return_value=bad):
            with pytest.raises(BackendInferenceBroken):
                backend.complete_with_usage(system="", user="hi", max_tokens=1, temperature=0.0)


# ─── Health check (R2.6) ──────────────────────────────────────────────
#
# R3 review fix: the executor-lens reviewer flagged that ``health_check``
# had no direct unit tests — only indirect coverage via
# ``test_two_stage_probe.py::_probe`` which routes through ``for_url``.
# These tests lock in the two-stage probe contract: missing-base-url,
# two-attempt fires on unreachable, stage-2 runs only when stage-1
# returns ``ok``, and the ``auth_rejected`` short-circuit.


class TestHealthCheck:
    def test_missing_base_url_returns_other(self):
        """An unset base_url must return ``outcome="other"`` with no
        HTTP side effects (and no SSRF validation path)."""
        import dataclasses

        cfg = dataclasses.replace(
            LLMConfig(),
            providers={"empty": {"type": "maxim_peer", "model": "m"}},
        )
        backend = _MaximPeerBackend(cfg, provider_key="empty")
        result = backend.health_check()
        assert result.outcome == "other"
        assert result.url == ""

    def test_two_attempt_fires_on_unreachable_first_attempt(self):
        """When the first liveness attempt is unreachable (not ``ok``
        and not ``auth_rejected``), a second attempt runs with the
        retry_timeout budget. R3 review fix: this was previously only
        exercised indirectly via the ``_probe()`` helper in
        test_two_stage_probe — add a direct call-count assertion."""
        backend = _MaximPeerBackend.for_url("http://127.0.0.1:9/v1")
        call_count = [0]

        def fail_then_succeed(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise HTTPConnectionError("probe", fix_hint="refused")
            # Second attempt returns ok
            return Response(
                status=200,
                headers={},
                content=b'{"data": []}',
                elapsed_ms=1.0,
                endpoint="_external",
                request_id="r",
            )

        with patch.object(_http, "fetch_url", side_effect=fail_then_succeed):
            result = backend.health_check(enable_stage2=False)
        assert call_count[0] == 2, "two-attempt liveness did not fire the retry"
        assert result.outcome == "ok"

    def test_stage2_skipped_when_stage1_auth_rejected(self):
        """``auth_rejected`` short-circuits — both stage-1 attempts
        should NOT fire, and stage-2 MUST NOT run even with
        ``enable_stage2=True``. Regression guard: if stage-2 ran after
        an auth rejection it would mis-classify as
        ``inference_broken``."""
        backend = _MaximPeerBackend.for_url("http://127.0.0.1:9/v1")
        call_count = [0]

        def always_auth_reject(*args, **kwargs):
            call_count[0] += 1
            raise HTTPAuthError("probe", status=401, fix_hint="bad key")

        with patch.object(_http, "fetch_url", side_effect=always_auth_reject):
            result = backend.health_check(enable_stage2=True)
        # Only stage-1 fires (one attempt; auth_rejected short-circuits
        # the two-attempt retry because is_reachable returns True for it).
        assert call_count[0] == 1, f"auth_rejected did not short-circuit: fetch_url called {call_count[0]}x"
        assert result.outcome == "auth_rejected"
        assert result.outcome != "inference_broken"

    def test_stage2_runs_only_when_stage1_ok(self):
        """Stage-2 runs only when stage-1 returned exactly ``ok`` AND
        the caller passes ``enable_stage2=True``. Regression guard for
        the two-stage probe contract."""
        backend = _MaximPeerBackend.for_url("http://127.0.0.1:9/v1", model="m")
        call_count = [0]

        def stage1_ok_stage2_500(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return Response(
                    status=200,
                    headers={},
                    content=b'{"data": []}',
                    elapsed_ms=1.0,
                    endpoint="_external",
                    request_id="r",
                )
            # Second call is stage-2 readiness; raise 500
            raise HTTPServerError("probe", status=500, fix_hint="chat broken")

        with patch.object(_http, "fetch_url", side_effect=stage1_ok_stage2_500):
            result = backend.health_check(enable_stage2=True)
        assert call_count[0] == 2
        assert result.outcome == "inference_broken"


# ─── for_url factory safety (R3 review fix) ────────────────────────────


class TestForUrlFactory:
    def test_for_url_does_not_mutate_os_environ(self):
        """R3 review critical finding #1: ``for_url`` previously stored
        the probe API key in ``os.environ["MAXIM_PEER_PROBE_KEY"]``
        which races across concurrent probes. The instance-override
        replacement must leave the env untouched."""
        import os

        # Capture the pre-call state
        had = "MAXIM_PEER_PROBE_KEY" in os.environ
        prior = os.environ.get("MAXIM_PEER_PROBE_KEY")
        try:
            backend = _MaximPeerBackend.for_url(
                "http://127.0.0.1:9999/v1",
                api_key="my-secret-key",
                model="m",
            )
            # The env var must NOT have been set
            assert "MAXIM_PEER_PROBE_KEY" not in os.environ or os.environ["MAXIM_PEER_PROBE_KEY"] == prior
            # The key must reach the instance via _get_api_key
            assert backend._get_api_key() == "my-secret-key"
        finally:
            if had:
                os.environ["MAXIM_PEER_PROBE_KEY"] = prior or ""
            else:
                os.environ.pop("MAXIM_PEER_PROBE_KEY", None)

    def test_for_url_concurrent_instances_have_distinct_keys(self):
        """Two backends built via ``for_url`` with different keys must
        NOT observe each other's keys. This is the regression guard
        for the env-var race the reviewer flagged."""
        a = _MaximPeerBackend.for_url("http://127.0.0.1:9998/v1", api_key="key-A")
        b = _MaximPeerBackend.for_url("http://127.0.0.1:9997/v1", api_key="key-B")
        assert a._get_api_key() == "key-A"
        assert b._get_api_key() == "key-B"
        # Swap order — neither observation should have leaked
        assert a._get_api_key() == "key-A"
        assert b._get_api_key() == "key-B"

    def test_for_url_none_key_returns_empty_string(self):
        """``for_url(api_key=None)`` still builds a usable backend
        whose ``_get_api_key`` returns an empty string (not a stale
        value from the environment)."""
        import os

        os.environ.pop("MAXIM_PEER_PROBE_KEY", None)
        os.environ.pop("MAXIM_LANE_LARGE_REMOTE_API_KEY", None)
        os.environ.pop("MAXIM_PEER_API_KEY", None)
        backend = _MaximPeerBackend.for_url("http://127.0.0.1:9996/v1", api_key=None)
        # _api_key_override is None, so _get_api_key falls through to
        # the env path with the empty api_key_env fallback
        assert backend._api_key_override is None
        # Falls through to cfg["api_key_env"] which is "" in for_url's
        # built cfg; os.getenv("", "") returns "".
        assert backend._get_api_key() == ""
