"""Unit tests for maxim.utils.http — the unified HTTP client (Plan 1 R1).

Focuses on behavior observable without a live network:

- Header sanitization (control chars, CR/LF, non-ASCII, length cap)
- RequestContext + contextvars round-trip + automatic propagation
- Endpoint registry lifecycle (register, lookup, dedupe)
- Typed HTTPError classification for status + transport failures
- Metrics snapshot shape + counter increments
- MAX_HEADER_VALUE_LEN / DEFAULT_POOL_PER_ENDPOINT / PROTOCOL_VERSION are
  named constants, not inline values

Network-level tests (live GET/POST against httpbin etc.) are deferred to
the migration-specific test files shipped alongside each sub-phase.
"""

from __future__ import annotations

import pytest

from maxim.utils import http


# ─── Named constants ──────────────────────────────────────────────────────


def test_named_constants_exist() -> None:
    assert http.MAX_HEADER_VALUE_LEN == 256
    assert http.DEFAULT_POOL_PER_ENDPOINT == 10
    assert http.PROTOCOL_VERSION == "1"
    assert http.DEFAULT_USER_AGENT == "maxim-peer/1.0"


# ─── Header sanitization ──────────────────────────────────────────────────


def test_sanitize_accepts_ascii_printable() -> None:
    assert http._sanitize_header_value("hello world!", "test") == "hello world!"
    assert http._sanitize_header_value("abc123-_.,~", "test") == "abc123-_.,~"


def test_sanitize_rejects_cr_lf() -> None:
    for bad in ("has\nnewline", "has\rcarriage", "both\r\n"):
        with pytest.raises(http.HTTPClientError) as exc_info:
            http._sanitize_header_value(bad, "X-Test")
        assert "CR/LF" in exc_info.value.fix_hint


def test_sanitize_rejects_control_chars() -> None:
    with pytest.raises(http.HTTPClientError) as exc_info:
        http._sanitize_header_value("has\x00null", "X-Test")
    assert "control" in exc_info.value.fix_hint.lower()


def test_sanitize_rejects_non_ascii() -> None:
    with pytest.raises(http.HTTPClientError) as exc_info:
        http._sanitize_header_value("café", "X-Test")
    assert "ascii" in exc_info.value.fix_hint.lower()


def test_sanitize_rejects_overlong() -> None:
    too_long = "x" * (http.MAX_HEADER_VALUE_LEN + 1)
    with pytest.raises(http.HTTPClientError) as exc_info:
        http._sanitize_header_value(too_long, "X-Test")
    assert "too long" in exc_info.value.fix_hint.lower()


def test_sanitize_exact_length_ok() -> None:
    exact = "x" * http.MAX_HEADER_VALUE_LEN
    assert http._sanitize_header_value(exact, "X-Test") == exact


def test_sanitize_rejection_increments_metric() -> None:
    before = http.metrics_snapshot()["header_rejected_total"].get("X-Count-Test", 0)
    with pytest.raises(http.HTTPClientError):
        http._sanitize_header_value("bad\nvalue", "X-Count-Test")
    after = http.metrics_snapshot()["header_rejected_total"].get("X-Count-Test", 0)
    assert after == before + 1


# ─── RequestContext + contextvars ─────────────────────────────────────────


def test_request_context_auto_generates_request_id() -> None:
    ctx = http.new_request_context()
    assert ctx.request_id
    assert len(ctx.request_id) >= 16  # mesh_trace uses UUID4 hex


def test_request_context_defaults_are_none() -> None:
    ctx = http.new_request_context()
    assert ctx.agent_id is None
    assert ctx.session_id is None
    assert ctx.lane is None
    assert ctx.parent_request_id is None


def test_contextvar_round_trip() -> None:
    assert http.current_context() is None
    ctx = http.new_request_context(agent_id="npc-alpha", session_id="s1", lane="large")
    token = http.set_context(ctx)
    try:
        cur = http.current_context()
        assert cur is not None
        assert cur.agent_id == "npc-alpha"
        assert cur.lane == "large"
    finally:
        http.reset_context(token)
    assert http.current_context() is None


def test_context_scope_manager() -> None:
    ctx = http.new_request_context(agent_id="npc-bravo")
    assert http.current_context() is None
    with http.context_scope(ctx):
        assert http.current_context().agent_id == "npc-bravo"
    assert http.current_context() is None


# ─── Endpoint registry ────────────────────────────────────────────────────


def test_register_endpoint_roundtrip() -> None:
    ep = http.HTTPEndpoint(
        name="_test_ep_1",
        base_url="http://127.0.0.1:1",
        default_headers={"User-Agent": http.DEFAULT_USER_AGENT},
    )
    http.register_endpoint(ep)
    assert http.is_registered("_test_ep_1")
    fetched = http.get_endpoint("_test_ep_1")
    assert fetched.name == "_test_ep_1"
    assert fetched.internal is False  # safe-by-default; opt in with internal=True


def test_register_endpoint_idempotent_replace() -> None:
    http.register_endpoint(http.HTTPEndpoint(name="_test_ep_2", base_url="http://a"))
    http.register_endpoint(http.HTTPEndpoint(name="_test_ep_2", base_url="http://b"))
    assert http.get_endpoint("_test_ep_2").base_url == "http://b"


def test_unknown_endpoint_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        http.get_endpoint("__does_not_exist__")


# ─── Header building ──────────────────────────────────────────────────────


def test_build_headers_propagates_x_maxim_for_internal() -> None:
    ep = http.HTTPEndpoint(
        name="_test_internal",
        base_url="http://example",
        default_headers={"User-Agent": "maxim-peer/1.0"},
        internal=True,
    )
    ctx = http.new_request_context(agent_id="agent-x", session_id="sess-y", lane="large")
    h = http._build_headers(ep, ctx, None)
    assert h["User-Agent"] == "maxim-peer/1.0"
    assert h["X-Maxim-Protocol-Version"] == "1"
    assert h["X-Maxim-Request-Id"] == ctx.request_id
    assert h["X-Maxim-Agent-Id"] == "agent-x"
    assert h["X-Maxim-Session-Id"] == "sess-y"
    assert h["X-Maxim-Lane"] == "large"


def test_build_headers_skips_x_maxim_for_external() -> None:
    ep = http.HTTPEndpoint(
        name="_test_external",
        base_url=None,
        default_headers={"User-Agent": "maxim-peer/1.0"},
        internal=False,
    )
    ctx = http.new_request_context(agent_id="agent-x")
    h = http._build_headers(ep, ctx, None)
    assert "X-Maxim-Request-Id" not in h
    assert "X-Maxim-Agent-Id" not in h
    assert "X-Maxim-Protocol-Version" not in h
    assert h["User-Agent"] == "maxim-peer/1.0"


def test_build_headers_auth_provider_sets_bearer() -> None:
    ep = http.HTTPEndpoint(
        name="_test_auth",
        base_url="http://example",
        default_headers={},
        auth_provider=lambda: "secret-token",
    )
    h = http._build_headers(ep, http.new_request_context(), None)
    assert h["Authorization"] == "Bearer secret-token"


def test_build_headers_auth_provider_none_omits_header() -> None:
    ep = http.HTTPEndpoint(
        name="_test_noauth",
        base_url="http://example",
        default_headers={},
        auth_provider=lambda: None,
    )
    h = http._build_headers(ep, http.new_request_context(), None)
    assert "Authorization" not in h


def test_build_headers_caller_extras_override() -> None:
    ep = http.HTTPEndpoint(
        name="_test_override",
        base_url="http://example",
        default_headers={"User-Agent": "original/1.0"},
    )
    h = http._build_headers(ep, http.new_request_context(), {"User-Agent": "override/2.0"})
    assert h["User-Agent"] == "override/2.0"


# ─── Status classification ────────────────────────────────────────────────


def test_classify_2xx_returns_none() -> None:
    assert http._classify_status("ep", 200, {}) is None
    assert http._classify_status("ep", 204, {}) is None
    assert http._classify_status("ep", 299, {}) is None


def test_classify_401_auth() -> None:
    err = http._classify_status("ep", 401, {})
    assert isinstance(err, http.HTTPAuthError)
    assert err.status == 401
    assert err.fix_hint


def test_classify_403_auth() -> None:
    err = http._classify_status("ep", 403, {})
    assert isinstance(err, http.HTTPAuthError)


def test_classify_429_rate_limited_with_retry_after() -> None:
    err = http._classify_status("ep", 429, {"Retry-After": "5.0"})
    assert isinstance(err, http.HTTPRateLimited)
    assert err.retry_after_s == 5.0


def test_classify_429_no_retry_after_defaults_to_zero() -> None:
    err = http._classify_status("ep", 429, {})
    assert isinstance(err, http.HTTPRateLimited)
    assert err.retry_after_s == 0.0


def test_classify_4xx_client_error() -> None:
    err = http._classify_status("ep", 400, {})
    assert isinstance(err, http.HTTPClientError)
    err = http._classify_status("ep", 404, {})
    assert isinstance(err, http.HTTPClientError)


def test_classify_5xx_server_error() -> None:
    for code in (500, 502, 503, 504):
        err = http._classify_status("ep", code, {})
        assert isinstance(err, http.HTTPServerError)
        assert err.status == code


# ─── Exception taxonomy ───────────────────────────────────────────────────


def test_exception_hierarchy_inherits_from_http_error() -> None:
    for cls in (
        http.HTTPTimeout,
        http.HTTPConnectionError,
        http.HTTPAuthError,
        http.HTTPClientError,
        http.HTTPServerError,
        http.HTTPRateLimited,
    ):
        assert issubclass(cls, http.HTTPError)


def test_http_error_carries_endpoint_and_fix_hint() -> None:
    err = http.HTTPError("my-endpoint", status=503, fix_hint="boom")
    assert err.endpoint == "my-endpoint"
    assert err.status == 503
    assert err.fix_hint == "boom"
    assert "boom" in str(err)


# ─── Metrics snapshot ─────────────────────────────────────────────────────


def test_metrics_snapshot_shape() -> None:
    snap = http.metrics_snapshot()
    assert "requests_total" in snap
    assert "latency" in snap
    assert "pool_exhausted_total" in snap
    assert "header_rejected_total" in snap
    assert "startup_phases" in snap


def test_record_startup_phase() -> None:
    http.record_startup_phase("test_phase", 42.0)
    snap = http.metrics_snapshot()
    assert snap["startup_phases"]["test_phase"] == 42.0


# ─── Streaming response lifetime ──────────────────────────────────────────
#
# Regression guards for the "httpx stream contexts must outlive their
# consumers" lesson in CLAUDE.md. Both ``stream_post`` and
# ``raw_proxy_forward_streaming`` enter an httpx stream context manager
# manually via ``stream_ctx.__enter__()``. If the caller does not hold a
# reference to ``stream_ctx`` beyond the function return, Python GC calls
# ``stream_ctx.__exit__()`` and closes the underlying stream before the
# consumer can call ``iter_bytes``/``iter_lines``. Storing the context on
# ``StreamingResponse._stream_ctx`` keeps it alive until ``close()`` runs.
#
# The bug was originally caught in ``raw_proxy_forward_streaming`` by the
# Cloudflare 524 incident (commit 627727e) but no regression test was
# added at the time — ``stream_post`` had the same bug for the Mac peer's
# streaming inference path and was only caught during the 2026-04-13
# 125s-latency investigation.


def test_stream_post_stores_stream_ctx() -> None:
    """``stream_post`` must set ``_stream_ctx`` so GC does not close the stream."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"data: hi\n\n")

    ep_name = "_test_stream_post_ctx"
    http.register_endpoint(http.HTTPEndpoint(name=ep_name, base_url="http://test.local", internal=False))
    http._registry._clients[ep_name] = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://test.local",
    )
    try:
        response = http.stream_post(ep_name, path="/v1/chat", json={"x": 1})
        try:
            # Tight structural guard: the field is the fix. If it's None
            # here, GC will close the stream before the consumer iterates
            # (see CLAUDE.md "httpx stream contexts must outlive their
            # consumers").
            assert response._stream_ctx is not None, (
                "stream_post must store stream_ctx on StreamingResponse — "
                "without it, GC closes the httpx stream before the consumer "
                "can iterate"
            )
        finally:
            response.close()
    finally:
        http._registry._clients.pop(ep_name, None)
        http._registry._endpoints.pop(ep_name, None)


def test_raw_proxy_forward_streaming_stores_stream_ctx() -> None:
    """``raw_proxy_forward_streaming`` must set ``_stream_ctx`` — same lesson.

    Regression guard for commit 627727e ("fix(http): prevent GC from
    closing httpx stream before iter_bytes()"). The fix landed without a
    test; this locks it in.
    """
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"chunk")

    # raw_proxy_forward_streaming uses the reserved _external endpoint.
    # Override its client with a MockTransport-backed one for this test.
    http._ensure_external_endpoint()
    original_client = http._registry._clients.get(http._EXTERNAL_ENDPOINT)
    http._registry._clients[http._EXTERNAL_ENDPOINT] = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="",
    )
    try:
        response = http.raw_proxy_forward_streaming(
            "http://test.local/v1/chat",
            "POST",
            headers={"X-Maxim-Request-Id": "test-req"},
            body=b"{}",
        )
        try:
            assert response._stream_ctx is not None, (
                "raw_proxy_forward_streaming must store stream_ctx on "
                "StreamingResponse — GC closes the httpx stream otherwise"
            )
        finally:
            response.close()
    finally:
        if original_client is not None:
            http._registry._clients[http._EXTERNAL_ENDPOINT] = original_client
        else:
            http._registry._clients.pop(http._EXTERNAL_ENDPOINT, None)


class TestLoggableUrl:
    """2026-08-12 privacy audit: external-URL log events must not carry
    query strings (LLM-generated search queries) into MAXIM_LOG_FILE or
    the remote-readable proxy log buffer."""

    def test_query_string_stripped(self):
        from maxim.utils.http import _loggable_url

        assert (
            _loggable_url("https://api.duckduckgo.com/?q=secret+llm+query&format=json")
            == "https://api.duckduckgo.com/?…"
        )

    def test_fragment_stripped_and_plain_url_kept(self):
        from maxim.utils.http import _loggable_url

        assert _loggable_url("https://example.com/page#token=abc") == "https://example.com/page?…"
        assert _loggable_url("https://example.com/docs/page") == "https://example.com/docs/page"
