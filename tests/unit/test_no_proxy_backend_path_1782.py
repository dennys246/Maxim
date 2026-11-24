"""#1782 (1.3.5 line 6 item 6, owner decision D16; Overdrive's OD-I10): NO_PROXY on the backend path, and strict.

D16: ``NO_PROXY`` is honoured on the backend call path, where an embedder makes the call, and the 2026-09-28 rule
("the proxy wins") keeps holding for Maxim's own operator commands. These tests pin both on each backend's REAL
client, the one its call goes through, for a cleartext tailnet URL (``"private"``: the address-checked client with
httpx's proxy mounts) and an ``https://`` one (``"any"``: a plain client). A provider entry's
``strict_address_check`` refuses every env proxy, and on ``"any"`` it used to be ignored.
"""

from __future__ import annotations

import dataclasses
import socket

import pytest

from maxim.utils import http as _http

PROXY = "http://proxy.corp:3128"
TAILNET_IP = "100.64.0.7"  # CGNAT: non-public, so a cleartext URL to it is allowed with allow_local_endpoints
TAILNET_NAME = "mini.tailnet.test"
KEY_ENV = "T1782_PEER_KEY"


@pytest.fixture(autouse=True)
def _no_env_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setattr(_http, "_proxy_warned", False)
    monkeypatch.setenv(KEY_ENV, "k" * 32)
    real = socket.getaddrinfo

    def fake(host, port, *args, **kwargs):  # the tailnet name resolves to the tailnet address, offline
        if host == TAILNET_NAME:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (TAILNET_IP, port or 0))]
        return real(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake)


def _route(client, url: str) -> str:
    """Where httpx sends ``url`` from ``client``: "proxy" (an HTTP or SOCKS proxy pool) or "direct"."""
    pool = client._transport_for_url(_http.httpx.URL(url))._pool
    return "proxy" if type(pool).__name__ in ("HTTPProxy", "SOCKSProxy") else "direct"


@pytest.fixture
def peer_client():
    """The real client ``_MaximPeerBackend``'s calls go through, for a base URL and provider extras."""
    from maxim.models.language.config import LLMConfig
    from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

    made: list[str] = []

    def build(base_url: str, **extra):
        cfg = dataclasses.replace(
            LLMConfig(),
            providers={
                "t1782": {
                    "type": "maxim_peer",
                    "base_url": base_url,
                    "api_key_env": KEY_ENV,
                    "model": "m",
                    "allow_local_endpoints": True,
                    "pricing_required": False,
                    **extra,
                }
            },
        )
        backend = _MaximPeerBackend(cfg, provider_key="t1782")
        assert backend._ensure_endpoint_registered()
        made.append(backend._endpoint_name)
        return _http._registry.get_client(backend._endpoint_name)

    yield build
    with _http._registry._lock:
        for name in made:
            _http._registry._endpoints.pop(name, None)
            client = _http._registry._clients.pop(name, None)
            if client is not None:
                client.close()


TAILNET_URLS = [f"http://{TAILNET_IP}:8099/v1", f"https://{TAILNET_NAME}/v1"]


@pytest.mark.parametrize("base_url", TAILNET_URLS, ids=["http-private", "https-any"])
def test_the_peer_backend_honours_no_proxy(peer_client, monkeypatch, base_url) -> None:
    monkeypatch.setenv("HTTP_PROXY", PROXY)
    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    monkeypatch.setenv("NO_PROXY", f"{TAILNET_IP},{TAILNET_NAME}")
    assert _route(peer_client(base_url), base_url + "/chat/completions") == "direct"


@pytest.mark.parametrize("base_url", TAILNET_URLS, ids=["http-private", "https-any"])
def test_without_no_proxy_a_configured_proxy_carries_the_backend_call(peer_client, monkeypatch, base_url) -> None:
    """The control for the test above: the proxy wins unless NO_PROXY or strict says otherwise (2026-09-28)."""
    monkeypatch.setenv("HTTP_PROXY", PROXY)
    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    assert _route(peer_client(base_url), base_url + "/chat/completions") == "proxy"


@pytest.mark.parametrize("base_url", TAILNET_URLS, ids=["http-private", "https-any"])
def test_a_strict_provider_entry_never_uses_an_env_proxy(peer_client, monkeypatch, base_url) -> None:
    """strict_address_check on the provider entry reaches the endpoint; on "any" it used to be ignored."""
    monkeypatch.setenv("HTTP_PROXY", PROXY)
    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    assert _route(peer_client(base_url, strict_address_check=True), base_url + "/chat/completions") == "direct"


def test_the_operator_leader_endpoint_still_goes_through_a_proxy(monkeypatch) -> None:
    """D16's other half: Maxim's own operator commands (``peer update``, ``peer logs``: the ``leader`` endpoint)."""
    from maxim.peer.config import register_leader_endpoint

    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    url = f"https://{TAILNET_NAME}/v1"
    saved = _http._registry._endpoints.get("leader")
    register_leader_endpoint(url)
    try:
        client = _http._registry.get_client("leader")
        assert _route(client, url + "/admin/version") == "proxy"
    finally:
        with _http._registry._lock:
            _http._registry._endpoints.pop("leader", None)
            stale = _http._registry._clients.pop("leader", None)
            if stale is not None:
                stale.close()
            if saved is not None:
                _http._registry._endpoints["leader"] = saved


def _openai_http_client(monkeypatch, base_url: str, **extra):
    pytest.importorskip("openai", reason="the OpenAI backend is an optional extra")
    from maxim.models.language.config import LLMConfig
    from maxim.models.language.openai_backend import _OpenAIBackend

    cfg = LLMConfig(
        enabled=True,
        providers={
            "openai_compatible": {
                "type": "openai",
                "api_key_env": KEY_ENV,
                "model": "m",
                "base_url": base_url,
                "allow_local_endpoints": True,
                **extra,
            }
        },
    )
    backend = _OpenAIBackend(cfg, provider_key="openai_compatible")
    client = backend._ensure_client()
    assert client is not None
    return backend, client._client


@pytest.mark.parametrize("base_url", TAILNET_URLS, ids=["http-private", "https-any"])
def test_the_openai_backend_honours_no_proxy(monkeypatch, base_url) -> None:
    monkeypatch.setenv("HTTP_PROXY", PROXY)
    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    monkeypatch.setenv("NO_PROXY", f"{TAILNET_IP},{TAILNET_NAME}")
    backend, httpx_client = _openai_http_client(monkeypatch, base_url)
    try:
        assert _route(httpx_client, base_url + "/chat/completions") == "direct"
    finally:
        backend.unload()


@pytest.mark.parametrize("base_url", TAILNET_URLS, ids=["http-private", "https-any"])
def test_a_strict_openai_provider_entry_never_uses_an_env_proxy(monkeypatch, base_url) -> None:
    monkeypatch.setenv("HTTP_PROXY", PROXY)
    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    backend, httpx_client = _openai_http_client(monkeypatch, base_url, strict_address_check=True)
    try:
        assert _route(httpx_client, base_url + "/chat/completions") == "direct"
        assert httpx_client.follow_redirects is True  # the SDK's own defaults are kept
    finally:
        backend.unload()
    assert httpx_client.is_closed  # a strict "any" client is ours, so unload closes it


def test_address_checked_client_strict_on_any_reads_no_env_proxy(monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    assert _http.address_checked_client("any", timeout=5.0, base_url=None) is None  # unchanged: the SDK's own
    client = _http.address_checked_client("any", timeout=5.0, base_url=None, strict=True)
    try:
        assert client is not None
        assert _route(client, "https://api.example.com/v1") == "direct"
    finally:
        client.close()
