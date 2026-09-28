"""#921: the two check-then-reconnect paths #824 left open.

- `download_to_file` followed redirects with no per-hop address check. Owner decision 2026-09-28:
  a download stays in the address class of its FIRST DIAL -- a public start is public on every hop (a
  public registry can never redirect into the LAN or 169.254.169.254); a LAN/loopback start the
  operator configured (a local Oasis, a mirror) is unrestricted. Classified on the dial itself, so no
  separate lookup can be steered.
- `validate_base_url` checked a backend URL on its own lookup, then the client looked the host up again
  to connect. Its rules -- public-only without `allow_local`, cleartext `http://` only to a private
  network -- are enforced on the address actually dialled, by one shared classifier; the peer's
  probes (which carry the bearer key) included.
- Owner decision 2026-09-28: an env HTTP(S)_PROXY that applies wins for these operator paths, with one
  warning (the proxy dials, so no check can apply); the model-chosen fetch (#824) never uses a proxy.

Offline throughout: local servers on 127.0.0.1 play the services, a stubbed resolver plays DNS, and
`_is_public_address` is narrowed so 127.0.0.1 can play a public registry where a test needs one.
"""

from __future__ import annotations

import http.server
import logging
import socket
import threading

import pytest

from maxim.utils import http as _http

HOST = "rebind.test"
PUBLIC = "93.184.216.34"


@pytest.fixture(autouse=True)
def _no_env_proxy_and_a_fresh_warning(monkeypatch):
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_http, "_proxy_warned", False)


@pytest.fixture
def endpoints():
    """Register throwaway endpoints and remove them afterwards (the registry is process-global)."""
    names: list[str] = []

    def register(ep):
        _http.register_endpoint(ep)
        names.append(ep.name)

    yield register
    with _http._registry._lock:
        for name in names:
            _http._registry._endpoints.pop(name, None)
            client = _http._registry._clients.pop(name, None)
            if client is not None:
                client.close()


REDIRECTS: dict[str, str] = {}
"""Redirect table for the local server: request path -> Location. Tests fill it; the server never
echoes request input into a header (code scanning: py/http-response-splitting)."""


@pytest.fixture
def server():
    """A local server: /file serves bytes, a path in ``REDIRECTS`` redirects, every path is recorded."""
    hits: list[str] = []
    REDIRECTS.clear()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            location = REDIRECTS.get(self.path)
            if location is not None:
                self.send_response(302)
                self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = b"PAYLOAD"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], hits
    srv.shutdown()


def _resolver(monkeypatch, table):
    """``table``: host -> list of answers (address lists), one per lookup; the last one repeats."""
    real = socket.getaddrinfo
    calls: dict[str, int] = {}

    def fake(host, port, *args, **kwargs):
        if host not in table:
            return real(host, port, *args, **kwargs)
        answers = table[host]
        addresses = answers[min(calls.get(host, 0), len(answers) - 1)]
        calls[host] = calls.get(host, 0) + 1
        family = lambda a: socket.AF_INET6 if ":" in a else socket.AF_INET  # noqa: E731
        return [(family(a), socket.SOCK_STREAM, 6, "", (a, port or 0)) for a in addresses]

    monkeypatch.setattr(socket, "getaddrinfo", fake)


def _redirect_url(origin: str, target: str) -> str:
    """``origin``/redirect answers 302 to ``target`` (registered, never read from the request)."""
    REDIRECTS["/redirect"] = target
    return f"{origin}/redirect"


def _loopback_plays_public(monkeypatch):
    monkeypatch.setattr(_http, "_is_public_address", lambda a: a == "127.0.0.1")


# ── downloads: the class of the first dial ───────────────────────────────────────────────────────


def test_a_public_download_cannot_be_redirected_into_the_lan(monkeypatch, server, tmp_path) -> None:
    port, hits = server
    _loopback_plays_public(monkeypatch)
    _resolver(monkeypatch, {"registry.test": [["127.0.0.1"]], "internal.test": [["127.0.0.2"]]})
    with pytest.raises(_http.HTTPConnectionError, match="non-public"):
        _http.download_to_file(
            _redirect_url(f"http://registry.test:{port}", f"http://internal.test:{port}/file"), tmp_path / "out"
        )
    assert hits == ["/redirect"]  # the first hop only


def test_a_public_start_stays_public_even_when_the_same_name_rebinds(monkeypatch, server, tmp_path) -> None:
    """The reviewers' attack, reversed: the start dials public, then the SAME host's next lookup (a
    new connection -- another port) answers private. The class of the first dial holds."""
    port, hits = server
    _loopback_plays_public(monkeypatch)
    _resolver(monkeypatch, {"registry.test": [["127.0.0.1"], ["127.0.0.2"]]})
    with pytest.raises(_http.HTTPConnectionError, match="non-public"):
        _http.download_to_file(
            _redirect_url(f"http://registry.test:{port}", f"http://registry.test:{port + 1}/file"), tmp_path / "out"
        )


def test_the_class_is_the_first_dial_not_a_separate_lookup(monkeypatch) -> None:
    """No classification lookup exists to be steered: the backend's class comes from the connection's
    own resolution, and a public first dial pins public."""
    _loopback_plays_public(monkeypatch)
    backend = _http._StartClassBackend()
    assert backend.start_class is None
    backend.admit("registry.test", ["127.0.0.1"])
    assert backend.start_class == "public"
    with pytest.raises(_http.NonPublicAddressRefused):
        backend.admit("internal.test", ["127.0.0.2"])
    lan = _http._StartClassBackend()
    lan.admit("oasis.lan", ["10.0.0.5"])
    assert lan.start_class == "any"
    lan.admit("anything", [PUBLIC])  # a LAN start is unrestricted


def test_a_public_download_may_redirect_to_another_public_host(monkeypatch, server, tmp_path) -> None:
    port, hits = server
    _loopback_plays_public(monkeypatch)
    _resolver(monkeypatch, {"registry.test": [["127.0.0.1"]], "cdn.test": [["127.0.0.1"]]})
    written = _http.download_to_file(
        _redirect_url(f"http://registry.test:{port}", f"http://cdn.test:{port}/file"), tmp_path / "out"
    )
    assert written == len(b"PAYLOAD") and (tmp_path / "out").read_bytes() == b"PAYLOAD"


def test_a_lan_download_the_operator_configured_still_works(server, tmp_path) -> None:
    """The control: a local Oasis or mirror on the LAN/loopback is unrestricted."""
    port, hits = server
    written = _http.download_to_file(f"http://127.0.0.1:{port}/file", tmp_path / "out")
    assert written == len(b"PAYLOAD") and hits == ["/file"]


# ── proxies (owner decision: the proxy wins for operator paths, loudly) ──────────────────────────


def _routes(client, url: str) -> str:
    """Where httpx sends ``url`` from this client: "checked" (our address-checking default transport)
    or "proxy"."""
    transport = client._transport_for_url(_http.httpx.URL(url))
    backend = getattr(getattr(transport, "_pool", None), "_network_backend", None)
    return "checked" if isinstance(backend, _http._AddressCheckingBackend) else "proxy"


def test_the_httpx_proxy_helper_this_relies_on_exists() -> None:
    """``_proxy_mounts`` reuses httpx's own proxy rules instead of re-deriving them; fail loudly if
    httpx moves the helper."""
    from httpx._utils import get_environment_proxies

    assert callable(get_environment_proxies)


def test_with_a_proxy_proxied_urls_use_it_and_direct_ones_are_checked(monkeypatch, caplog) -> None:
    """httpx decides per URL with its OWN NO_PROXY semantics -- including the port form the first
    draft misread -- so a NO_PROXY host is dialled directly AND checked, and a proxied one goes through
    the proxy, with one warning."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.corp:3128")
    monkeypatch.setenv("NO_PROXY", "internal.example,10.0.0.5:8000,169.254.169.254")
    with caplog.at_level(logging.WARNING, logger="maxim.utils.http"):
        client = _http._download_client(_http._registry.get(_http._EXTERNAL_ENDPOINT))
        client2 = _http._download_client(_http._registry.get(_http._EXTERNAL_ENDPOINT))
    try:
        assert _routes(client, "https://huggingface.co/x") == "proxy"
        assert _routes(client, "https://internal.example/x") == "checked"
        assert _routes(client, "http://10.0.0.5:8000/x") == "checked"
        # a redirect hop to the metadata address, NO_PROXY'd as AWS advises: direct, so CHECKED
        assert _routes(client, "http://169.254.169.254/latest/meta-data") == "checked"
        assert sum("proxy is configured" in r.getMessage() for r in caplog.records) == 1  # once
    finally:
        client.close()
        client2.close()


def test_without_a_proxy_everything_is_checked() -> None:
    client = _http._download_client(_http._registry.get(_http._EXTERNAL_ENDPOINT))
    try:
        assert _routes(client, "https://huggingface.co/x") == "checked"
    finally:
        client.close()


def test_the_model_chosen_fetch_never_uses_a_proxy(monkeypatch) -> None:
    """#824 stays strict: the public-only endpoint has no proxy mounts, even with a proxy set."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp:3128")
    client = _http._checked_client(
        _http._PublicOnlyBackend(), limits=_http.httpx.Limits(max_connections=1), timeout=5.0, strict=True
    )
    try:
        assert _routes(client, "https://example.com/x") == "checked"
    finally:
        client.close()


# ── backend base URLs ────────────────────────────────────────────────────────────────────────────


def test_base_url_address_policy_matches_validate_base_url() -> None:
    from maxim.utils.net import base_url_address_policy

    assert base_url_address_policy("https://api.example.com/v1", allow_local=False) == "public"
    assert base_url_address_policy("http://192.168.1.5:8100/v1", allow_local=True) == "private"
    assert base_url_address_policy("https://peer.lan/v1", allow_local=True) == "any"


def test_one_classifier_validate_and_connect_agree_on_cgnat(monkeypatch) -> None:
    """validate_base_url and the connect check used to disagree on 100.64.0.0/10 (CGNAT, Tailscale):
    public to one, non-public to the other. One classifier now."""
    from maxim.utils import net

    assert _http._is_public_address is net.is_public_address
    monkeypatch.setattr(net.socket, "getaddrinfo", lambda h, p, **k: [(socket.AF_INET, 1, 6, "", ("100.64.1.2", p))])
    assert net.validate_base_url("https://ts-peer.test/v1", allow_local=False) is None  # not public
    assert net.validate_base_url("http://ts-peer.test/v1", allow_local=True) == "http://ts-peer.test/v1"  # LAN-class


def test_a_cleartext_backend_never_reaches_a_public_host(monkeypatch, server, endpoints) -> None:
    port, hits = server
    _resolver(monkeypatch, {HOST: [[PUBLIC]]})
    endpoints(_http.HTTPEndpoint(name="t921-private", base_url=f"http://{HOST}:{port}", address_policy="private"))
    with pytest.raises(_http.HTTPConnectionError, match="public address"):
        _http.get("t921-private", "/file")
    assert hits == []


def test_a_cleartext_backend_still_reaches_its_lan_server(monkeypatch, server, endpoints) -> None:
    port, hits = server
    _resolver(monkeypatch, {HOST: [["127.0.0.1"]]})
    endpoints(_http.HTTPEndpoint(name="t921-private-ok", base_url=f"http://{HOST}:{port}", address_policy="private"))
    assert _http.get("t921-private-ok", "/file").content == b"PAYLOAD"


def test_an_unknown_policy_is_refused_not_silently_unrestricted() -> None:
    with pytest.raises(ValueError, match="unknown address_policy"):
        _http.HTTPEndpoint(name="typo", base_url=None, address_policy="pubilc")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _http.fetch_url("http://127.0.0.1:9/x", address_policy="pubilc")  # type: ignore[arg-type]


def test_the_peer_probe_carrying_the_key_never_reaches_a_public_host_in_cleartext(monkeypatch, server) -> None:
    """The liveness probe sends `Authorization: Bearer <key>` to the base URL; a cleartext URL whose
    host resolves public is refused before any byte is sent. The local server plays that PUBLIC host
    (127.0.0.1 counts as public here), so an unchecked probe WOULD reach it -- `hits` proves it did not."""
    from maxim.models.language.maxim_peer_backend import _probe_once

    port, hits = server
    _resolver(monkeypatch, {HOST: [["127.0.0.1"]]})
    with monkeypatch.context() as m:
        _loopback_plays_public(m)
        result = _probe_once(f"http://{HOST}:{port}/v1", "secret-key", 2.0)
    assert result.outcome != "ok" and hits == []
    # the LAN control: the same URL resolving to a private address (127.0.0.1, really private) is probed
    assert _probe_once(f"http://{HOST}:{port}/v1", "secret-key", 2.0).outcome == "ok"
    assert hits


def test_the_peer_backend_registers_its_endpoint_with_the_url_policy(monkeypatch) -> None:
    import dataclasses
    import os

    from maxim.models.language.config import LLMConfig
    from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

    _resolver(monkeypatch, {"peer.example": [[PUBLIC]]})  # validate_base_url requires a resolvable host
    for url, policy in (("http://127.0.0.1:9999/v1", "private"), ("https://peer.example/v1", "any")):
        cfg = dataclasses.replace(
            LLMConfig(),
            providers={
                "t921-peer": {
                    "type": "maxim_peer",
                    "base_url": url,
                    "api_key_env": "T921_PEER_KEY",
                    "model": "m",
                    "allow_local_endpoints": True,
                    "pricing_required": False,
                }
            },
        )
        os.environ["T921_PEER_KEY"] = "k"
        try:
            backend = _MaximPeerBackend(cfg, provider_key="t921-peer")
            assert backend._ensure_endpoint_registered()
            assert _http._registry.get(backend._endpoint_name).address_policy == policy, url
        finally:
            os.environ.pop("T921_PEER_KEY", None)
            with _http._registry._lock:
                _http._registry._endpoints.pop("peer-t921-peer", None)
                client = _http._registry._clients.pop("peer-t921-peer", None)
                if client is not None:
                    client.close()


def _openai_backend(monkeypatch, base_url: str, *, allow_local: bool, answer: str):
    pytest.importorskip("openai", reason="the OpenAI backend is an optional extra")
    from maxim.models.language.config import LLMConfig
    from maxim.models.language.openai_backend import _OpenAIBackend

    monkeypatch.setenv("T921_OPENAI_KEY", "k")
    monkeypatch.setattr(socket, "getaddrinfo", lambda h, p, *a, **k: [(socket.AF_INET, 1, 6, "", (answer, p))])
    cfg = LLMConfig(
        enabled=True,
        providers={
            "openai_compatible": {
                "type": "openai",
                "api_key_env": "T921_OPENAI_KEY",
                "model": "m",
                "base_url": base_url,
                "allow_local_endpoints": allow_local,
            }
        },
    )
    return _OpenAIBackend(cfg, provider_key="openai_compatible")


@pytest.mark.parametrize(
    ("base_url", "allow_local", "answer", "backend_cls"),
    [
        ("http://lan-llm.test:8000/v1", True, "10.0.0.5", "_PrivateOnlyBackend"),
        ("https://cloud-llm.test/v1", False, PUBLIC, "_PublicOnlyBackend"),
    ],
)
def test_the_openai_backend_hands_the_sdk_a_checked_client(
    monkeypatch, base_url, allow_local, answer, backend_cls
) -> None:
    backend = _openai_backend(monkeypatch, base_url, allow_local=allow_local, answer=answer)
    client = backend._ensure_client()
    assert client is not None
    httpx_client = client._client
    assert isinstance(httpx_client._transport._pool._network_backend, getattr(_http, backend_cls))
    # the SDK's own defaults kept: redirects followed (every hop is checked) and its connection limits
    assert httpx_client.follow_redirects is True
    assert httpx_client._transport._pool._max_connections == 1000
    backend.unload()
    assert httpx_client.is_closed


def test_the_openai_backend_names_a_refusal_instead_of_connection_error() -> None:
    from maxim.models.language.openai_backend import _address_refusal

    refusal = _http.PublicAddressRefused("refused: 'x' resolves to a public address")
    wrapped = RuntimeError("Connection error.")
    wrapped.__cause__ = ConnectionError("pool")
    wrapped.__cause__.__cause__ = refusal
    assert _address_refusal(wrapped) == str(refusal)
    assert _address_refusal(RuntimeError("other")) is None


def test_a_refused_openai_call_takes_the_normal_failure_path(monkeypatch) -> None:
    """A policy refusal returns the backend's usual empty response (as every other failure does), not a
    raw SDK exception the router would count as an unclassified bug."""
    # validation passes (a LAN answer); the refusal a rebinding connect would raise is injected below
    backend = _openai_backend(monkeypatch, "http://lan-llm.test:8000/v1", allow_local=True, answer="10.0.0.5")
    client = backend._ensure_client()
    assert client is not None
    refusal = _http.PublicAddressRefused("refused: 'lan-llm.test' resolves to a public address")

    def refuse(*args, **kwargs):
        err = ConnectionError("Connection error.")
        err.__cause__ = refusal
        raise err

    monkeypatch.setattr(client.chat.completions.with_raw_response, "create", refuse)
    response = backend.complete_with_usage(system="s", user="u", max_tokens=4, temperature=0.0)
    assert response.content == ""
    backend.unload()


def test_a_client_replaced_after_an_auth_reset_is_closed_at_unload(monkeypatch) -> None:
    backend = _openai_backend(monkeypatch, "http://lan-llm.test:8000/v1", allow_local=True, answer="10.0.0.5")
    first = backend._ensure_client()._client
    backend._client = None  # what an auth error does: the next call builds a fresh client
    second = backend._ensure_client()._client
    assert first is not second and not first.is_closed  # retired, not closed under an in-flight call
    backend._client = None  # a persistent 401: another reset
    third = backend._ensure_client()._client
    assert first.is_closed and not second.is_closed  # only ONE retired client is kept
    assert len(backend._retired_http_clients) == 1
    backend.unload()
    assert second.is_closed and third.is_closed
