"""#824: a public-only fetch checks the address it actually dials.

The policy's private-IP check resolved the host separately (through a 5-minute cache), and httpx then
resolved it again to connect -- so a rebinding host answered public to the check and private to the
connect. Everything here is offline: a local server on 127.0.0.1 plays the internal service, and a
stubbed resolver plays the rebinding DNS.
"""

from __future__ import annotations

import http.server
import socket
import threading

import httpcore
import pytest

from maxim.utils import http as _http

HOST = "rebind.test"
PUBLIC = "93.184.216.34"


@pytest.fixture
def internal_server():
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            body = b"INTERNAL SECRET"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], hits
    server.shutdown()


def _resolver(monkeypatch, answers):
    """``answers``: a list of address lists for HOST, one per lookup (the last one repeats)."""
    real = socket.getaddrinfo
    calls = []

    def fake(host, port, *args, **kwargs):
        if host != HOST:
            return real(host, port, *args, **kwargs)
        addresses = answers[min(len(calls), len(answers) - 1)]
        calls.append(addresses)
        family = lambda a: socket.AF_INET6 if ":" in a else socket.AF_INET  # noqa: E731
        return [(family(a), socket.SOCK_STREAM, 6, "", (a, port or 0)) for a in addresses]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    return calls


def test_a_rebinding_host_passes_the_policy_but_never_reaches_the_internal_service(monkeypatch, internal_server):
    """The issue's scenario, end to end through the tool: public to the policy, private to the connect."""
    from maxim.tools.http_fetch import HttpFetchTool
    from maxim.utils import internet_access
    from maxim.utils.internet_access import InternetAccessPolicy

    port, hits = internal_server
    monkeypatch.setattr(internet_access, "_dns_cache", {})
    _resolver(monkeypatch, [[PUBLIC], ["127.0.0.1"]])
    policy = InternetAccessPolicy(enabled=True, require_robots_ok=False, unsafe_content_checks=False)
    assert policy.can_access(f"http://{HOST}:{port}/secret") == (True, None)  # the pre-check is fooled

    result = HttpFetchTool(get_internet_policy=lambda: policy).execute(url=f"http://{HOST}:{port}/secret")

    assert result.success is False
    assert "INTERNAL SECRET" not in str(result.output or "")
    assert hits == []


@pytest.mark.parametrize(
    "addresses",
    [["127.0.0.1"], [PUBLIC, "10.0.0.5"], ["::ffff:127.0.0.1"], ["169.254.169.254"]],
    ids=["loopback", "mixed-public-and-private", "ipv4-mapped-ipv6", "cloud-metadata"],
)
def test_a_public_only_fetch_refuses_any_non_public_answer(monkeypatch, internal_server, addresses):
    port, hits = internal_server
    _resolver(monkeypatch, [addresses])
    with pytest.raises(_http.HTTPConnectionError, match="non-public"):
        _http.fetch_url(f"http://{HOST}:{port}/secret", public_only=True)
    assert hits == []


def test_the_connection_dials_the_address_that_was_checked(monkeypatch):
    """The vetted IP is what the connection dials -- not the name, which would be resolved again."""
    _resolver(monkeypatch, [[PUBLIC], ["127.0.0.1"]])
    dialled = []

    def record(self, host, port, *args, **kwargs):
        dialled.append(host)
        raise httpcore.ConnectError("stop here")

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", record)
    with pytest.raises(_http.HTTPConnectionError):
        _http.fetch_url(f"http://{HOST}:8080/x", public_only=True)
    assert dialled == [PUBLIC]


def test_a_fetch_that_is_not_public_only_still_reaches_lan_hosts(monkeypatch, internal_server):
    """The control: the shared _external client (the leader proxy, `for_url` admin calls) is unrestricted;
    downloads and operator URLs are checked separately since #921."""
    port, hits = internal_server
    _resolver(monkeypatch, [["127.0.0.1"]])
    response = _http.fetch_url(f"http://{HOST}:{port}/ok")
    assert response.content == b"INTERNAL SECRET"
    assert hits == ["/ok"]


def test_the_public_only_client_really_uses_the_vetting_backend():
    """httpx has no backend parameter; the transport sets its pool's. Fails if httpx stops honouring it."""
    _http.fetch_url  # noqa: B018 -- the module is imported
    _http._ensure_external_endpoint()
    client = _http._registry.get_client(_http._EXTERNAL_PUBLIC_ENDPOINT)
    assert isinstance(client._transport._pool._network_backend, _http._PublicOnlyBackend)


# ── review folds ──


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("93.184.216.34", True),
        ("2606:4700::1111", True),
        ("64:ff9b::5db8:d822", True),  # NAT64 around a public IPv4
        ("2002:5db8:d822::", True),  # 6to4 around a public IPv4
        ("64:ff9b::a00:1", False),  # NAT64 around 10.0.0.1 -- is_global says True
        ("64:ff9b::a9fe:a9fe", False),  # NAT64 around the cloud metadata address
        ("::ffff:0:7f00:1", False),  # IPv4-translated loopback -- is_global says True
        ("::127.0.0.1", False),  # IPv4-compatible loopback -- is_global says True
        ("2002:a00:1::", False),  # 6to4 around 10.0.0.1
        ("fec0::1", False),  # site-local -- is_global says True
        ("224.0.0.1", False),  # multicast -- is_global says True
        ("ff02::1", False),
        ("::", False),
    ],
)
def test_public_means_globally_routable_whatever_the_form(address, public) -> None:
    assert _http._is_public_address(address) is public


def test_every_vetted_address_is_tried_in_turn(monkeypatch):
    """A dual-stack answer whose first address has no route still reaches the second, as before."""
    _resolver(monkeypatch, [["2606:4700::1111", PUBLIC]])
    dialled = []

    def connect(self, host, port, *args, **kwargs):
        dialled.append(host)
        raise httpcore.ConnectError(f"no route to {host}")

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", connect)
    with pytest.raises(_http.HTTPConnectionError):
        _http.fetch_url(f"http://{HOST}:8080/x", public_only=True)
    assert dialled == ["2606:4700::1111", PUBLIC]


def test_the_refusal_is_classified_by_its_type_not_its_wording(monkeypatch, internal_server):
    port, _ = internal_server
    _resolver(monkeypatch, [["127.0.0.1"]])
    with pytest.raises(_http.HTTPConnectionError) as info:
        _http.fetch_url(f"http://{HOST}:{port}/secret", public_only=True)
    assert "never connects to a private or internal address" in str(info.value.fix_hint)
