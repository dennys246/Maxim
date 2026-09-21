"""#825 — the fetch byte cap bounds the DOWNLOAD, not just what is kept afterwards.

Before the fix `http_fetch` applied `max_fetch_bytes` after `response.content` had already read
the whole body, so a multi-GB response was fully downloaded first. These tests serve a large
streamed body and count how many chunks the client actually pulled.
"""

from __future__ import annotations

import httpx
import pytest

from maxim.utils import http as _http

CHUNK = 1 << 16  # 64 KiB
TOTAL_CHUNKS = 800  # a 50 MiB body


class _CountingBody(httpx.SyncByteStream):
    def __init__(self, chunks: int = TOTAL_CHUNKS) -> None:
        self.chunks = chunks
        self.pulled = 0

    def __iter__(self):
        for _ in range(self.chunks):
            self.pulled += 1
            yield b"a" * CHUNK


@pytest.fixture
def served(monkeypatch):
    """Route the shared `_external` endpoint to a counting in-memory server."""
    state: dict = {"body": None, "content_type": "text/html"}

    def handler(request: httpx.Request) -> httpx.Response:
        state["body"] = _CountingBody()
        return httpx.Response(200, headers={"Content-Type": state["content_type"]}, stream=state["body"])

    _http._ensure_external_endpoint()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(_http._registry, "_clients", {**_http._registry._clients, _http._EXTERNAL_ENDPOINT: client})
    yield state
    client.close()


def test_fetch_url_stops_downloading_at_the_cap(served) -> None:
    resp = _http.fetch_url("https://big.example/file", max_bytes=1_000_000)
    assert len(resp.content) == 1_000_000
    assert resp.truncated is True
    # At most one chunk past the cap was pulled -- not the other ~49 MiB.
    assert served["body"].pulled <= (1_000_000 // CHUNK) + 2


def test_fetch_url_without_a_cap_is_unchanged(served) -> None:
    resp = _http.fetch_url("https://big.example/file")
    assert len(resp.content) == CHUNK * TOTAL_CHUNKS
    assert resp.truncated is False


def test_a_body_under_the_cap_is_not_marked_truncated(served) -> None:
    resp = _http.fetch_url("https://big.example/file", max_bytes=CHUNK * TOTAL_CHUNKS)
    assert len(resp.content) == CHUNK * TOTAL_CHUNKS and resp.truncated is False


def _fetch_tool():
    from maxim.tools.http_fetch import HttpFetchTool
    from maxim.utils.internet_access import InternetAccessPolicy

    policy = InternetAccessPolicy(enabled=True, max_fetch_bytes=1_000_000, require_robots_ok=False)
    policy._is_private_ip = lambda host: False  # the domain is fictional; skip DNS
    return HttpFetchTool(get_internet_policy=lambda: policy)


def test_http_fetch_downloads_at_most_the_policy_cap(served) -> None:
    result = _fetch_tool().execute(url="https://big.example/page")
    assert result.success is True
    assert result.metadata["truncated"] is True
    assert served["body"].pulled <= (1_000_000 // CHUNK) + 2


def test_http_fetch_refuses_a_binary_content_type(served) -> None:
    served["content_type"] = "application/octet-stream"
    result = _fetch_tool().execute(url="https://big.example/blob")
    assert result.success is False
    assert result.metadata.get("unsupported_content_type") is True


# -- review folds ----------------------------------------------------------------------------


@pytest.fixture
def serve(monkeypatch):
    """Serve one fixed response through the `_external` endpoint."""

    def _install(status: int, body: bytes, headers: dict) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, headers=headers, stream=httpx.ByteStream(body))

        _http._ensure_external_endpoint()
        client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(_http._registry, "_clients", {**_http._registry._clients, _http._EXTERNAL_ENDPOINT: client})

    return _install


def test_a_gzip_bomb_is_inflated_only_up_to_the_cap(serve) -> None:
    import gzip
    import tracemalloc

    bomb = gzip.compress(b"\0" * 200_000_000)  # ~200 MB inflates from ~200 KB
    serve(200, bomb, {"Content-Type": "text/html", "Content-Encoding": "gzip"})
    tracemalloc.start()
    try:
        resp = _http.fetch_url("https://bomb.example/", max_bytes=1000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(resp.content) == 1000 and resp.truncated is True
    assert peak < 20_000_000, f"inflation was not bounded: peak {peak:,} bytes"


def test_an_encoding_it_cannot_bound_is_refused(serve) -> None:
    serve(200, b"\x0b\x02\x80", {"Content-Type": "text/html", "Content-Encoding": "br"})
    with pytest.raises(_http.HTTPError):
        _http.fetch_url("https://br.example/", max_bytes=1000)


def test_a_4xx_on_the_capped_path_is_still_classified(serve) -> None:
    serve(404, b"x" * 5_000_000, {"Content-Type": "text/html"})
    with pytest.raises(_http.HTTPClientError) as exc:
        _http.fetch_url("https://missing.example/", max_bytes=10)
    assert exc.value.response.truncated is True and len(exc.value.response.content) == 10


def test_the_model_cannot_raise_the_cap_above_the_policy(served) -> None:
    result = _fetch_tool().execute(url="https://big.example/page", max_bytes=10**10)
    assert result.success is True
    assert served["body"].pulled <= (1_000_000 // CHUNK) + 2  # still the policy's 1 MB


def test_javascript_and_yaml_are_textual() -> None:
    from maxim.tools.http_fetch import _is_textual

    for ct in ("application/javascript", "application/yaml; charset=utf-8", "application/ld+json", "text/plain"):
        assert _is_textual(ct), ct
    assert not _is_textual("application/pdf")
