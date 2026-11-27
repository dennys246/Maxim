"""#1806 PR 0: ``fetch_url(address_policy="private", strict=True)`` dials a private endpoint DIRECTLY.

Under #921 an env proxy that applies to an operator URL wins (the proxy dials, so no address check can apply). The
Reachy daemon probe must never be answered by a proxy for a robot it never reached, so the strict private mode has no
proxy mounts and no implicit env/system proxy, keeps the #1715 whole-request deadline, and still refuses a public
address on the address actually dialled.

Offline: local servers on 127.0.0.1 play the daemon; the "proxy" is a dead port, so a proxied request cannot succeed.
"""

from __future__ import annotations

import http.server
import socket
import threading
import time

import pytest

from maxim.utils import http as _http

DEAD_PROXY = "http://127.0.0.1:9"
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@pytest.fixture(autouse=True)
def _fresh_clients(monkeypatch):
    """The external clients are pooled process-wide and built from the env at first use: rebuild them per test."""
    for name in (*_PROXY_VARS, "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_http, "_proxy_warned", False)
    _http._ensure_external_endpoint()
    names = list(_http._DEADLINE_ENDPOINTS)
    saved = {n: _http._registry._clients.pop(n, None) for n in names}
    yield
    for name, client in saved.items():
        fresh = _http._registry._clients.pop(name, None)
        if fresh is not None:
            fresh.close()
        if client is not None:
            _http._registry._clients[name] = client


def _set_dead_proxy(monkeypatch) -> None:
    for name in _PROXY_VARS:
        monkeypatch.setenv(name, DEAD_PROXY)


@pytest.fixture
def server():
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            body = b'{"state": "running"}'
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


def test_a_configured_proxy_is_bypassed_and_the_daemon_is_dialled_directly(monkeypatch, server) -> None:
    from maxim import _offline

    if _offline.is_offline():
        pytest.skip("offline: every client is strict already")
    port, hits = server
    _set_dead_proxy(monkeypatch)
    resp = _http.fetch_url(
        f"http://127.0.0.1:{port}/api/daemon/status", address_policy="private", strict=True, deadline_s=3.0
    )
    assert resp.json() == {"state": "running"} and hits == ["/api/daemon/status"]
    assert resp.endpoint == "_external_policy_private_strict"


def test_control_the_default_mode_lets_the_same_proxy_carry_it(monkeypatch, server) -> None:
    """Without ``strict=True`` the dead proxy carries the URL (#921) and the daemon is never reached: this is what
    makes the test above non-vacuous."""
    from maxim import _offline

    if _offline.is_offline():
        pytest.skip("offline: every client is strict, with no proxy mounts")
    port, hits = server
    _set_dead_proxy(monkeypatch)
    with pytest.raises(_http.HTTPError):
        _http.fetch_url(f"http://127.0.0.1:{port}/api/daemon/status", address_policy="private", deadline_s=3.0)
    assert hits == []


def test_the_strict_client_has_no_proxy_mounts_and_does_not_trust_env(monkeypatch) -> None:
    _set_dead_proxy(monkeypatch)
    client = _http._registry.get_client(_http._EXTERNAL_STRICT_ENDPOINTS["private"])
    assert not any(t is not None for t in client._mounts.values())
    assert client._trust_env is False


def test_a_public_address_is_still_refused_with_a_proxy_set(monkeypatch, server) -> None:
    """The check runs on the address dialled: a name that resolves public is refused before any byte is sent, and the
    proxy (which would dial it unchecked under #921) is never used."""
    port, hits = server
    _set_dead_proxy(monkeypatch)
    real = socket.getaddrinfo

    def fake(host, p, *args, **kwargs):
        if host == "robot.test":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", p or 0))]
        return real(host, p, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    monkeypatch.setattr(_http, "_is_public_address", lambda a: a == "127.0.0.1")  # loopback plays a public host
    with pytest.raises(_http.HTTPConnectionError, match="public address"):
        _http.fetch_url(f"http://robot.test:{port}/api/daemon/status", address_policy="private", strict=True)
    assert hits == []


def test_the_deadline_bounds_the_whole_strict_request() -> None:
    """#1715: a daemon that drips its body is refused at about ``deadline_s``, not after each read's timeout."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    stop = threading.Event()

    def drip() -> None:
        conn, _ = srv.accept()
        with conn:
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 40\r\n\r\n")
            for _ in range(40):
                if stop.wait(0.1):
                    return
                try:
                    conn.sendall(b"x")
                except OSError:
                    return

    threading.Thread(target=drip, daemon=True).start()
    try:
        started = time.monotonic()
        with pytest.raises(_http.HTTPTimeout):
            _http.fetch_url(
                f"http://127.0.0.1:{srv.getsockname()[1]}/api/daemon/status",
                address_policy="private",
                strict=True,
                timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=2.0, pool_s=2.0),
                deadline_s=0.6,
            )
        assert time.monotonic() - started < 1.5
    finally:
        stop.set()
        srv.close()


def test_every_fetch_url_endpoint_carries_the_deadline_backend() -> None:
    """Iterates ``_DEADLINE_ENDPOINTS`` itself, so the strict endpoint (and any later one) is covered."""
    assert _http._EXTERNAL_STRICT_ENDPOINTS["private"] in _http._DEADLINE_ENDPOINTS
    for name in _http._DEADLINE_ENDPOINTS:
        client = _http._registry.get_client(name)
        transports = [client._transport, *(t for t in client._mounts.values() if t is not None)]
        for t in transports:
            assert isinstance(t._pool._network_backend, _http._DeadlineBackend), name
