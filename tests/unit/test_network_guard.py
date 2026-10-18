"""No test reaches the network (roadmap 1.3.1): tests/network_guard.py, installed by tests/conftest.py.

Each case uses an address that would really leave the machine (a public IP, a real hostname) -- and
proves it cannot. Nothing here needs the network to pass, and none of it can reach it."""

from __future__ import annotations

import http.server
import socket
import threading

import pytest

from tests import network_guard


def test_the_guard_is_installed_for_the_session() -> None:
    assert socket.socket.connect is network_guard._connect
    assert socket.getaddrinfo is network_guard._getaddrinfo


def test_an_outbound_tcp_connect_raises() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        with pytest.raises(network_guard.NetworkBlocked):
            sock.connect(("192.0.2.1", 80))


def test_connect_ex_reports_unreachable() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        assert sock.connect_ex(("192.0.2.1", 80)) != 0


def test_a_dns_lookup_of_a_real_name_raises() -> None:
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("api.anthropic.com", 443)
    with pytest.raises(socket.gaierror):
        socket.gethostbyname("example.com")


def test_an_http_client_cannot_reach_the_internet() -> None:
    """Through the stack real code uses: urllib/httpx resolve with getaddrinfo, then connect."""
    import httpx

    with pytest.raises(httpx.ConnectError):
        httpx.get("https://example.com", timeout=1.0)


def test_loopback_still_works() -> None:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import httpx

        assert httpx.get(f"http://localhost:{server.server_address[1]}/", timeout=2.0).status_code == 204
    finally:
        server.shutdown()


def test_udp_connect_sends_nothing_and_is_allowed() -> None:
    """The "find my LAN address" idiom connects a UDP socket to a public IP; no packet leaves, and the in-process
    guard allows it. Inside CI's loopback-only network namespace (MAXIM_EXPECT_NETNS=1, #940) the KERNEL refuses
    the route instead (ENETUNREACH): the guard still did not raise NetworkBlocked, which is what this pins."""
    import errno
    import os

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        if os.environ.get("MAXIM_EXPECT_NETNS") == "1":
            with pytest.raises(OSError) as exc:
                sock.connect(("8.8.8.8", 80))
            assert not isinstance(exc.value, network_guard.NetworkBlocked)
            assert exc.value.errno == errno.ENETUNREACH
            return
        sock.connect(("8.8.8.8", 80))
        assert sock.getsockname()[0]


@pytest.mark.allow_network
def test_the_marker_lifts_the_guard_for_its_test() -> None:
    assert network_guard.enabled is False  # nothing is contacted: the flag is the contract


def test_the_guard_is_back_after_a_marked_test() -> None:
    assert network_guard.enabled is True
