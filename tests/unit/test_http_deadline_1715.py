"""#1715: ``fetch_url(deadline_s=)`` bounds the whole request, not each socket read.

httpx times the pool wait, the connect and each socket read and write, never the whole request. So a server that
drips its body, drips its headers, or streams ``100 Continue`` responses forever can hold a call far past every
timeout the caller set. Design: docs/plans/http_deadline_1715.md (v1's chunk-loop check could not see a drip, which is
why these tests use real sockets and drip byte by byte).

Every server here is a raw TCP socket on 127.0.0.1. The deadline is 0.6 s against a per-read timeout of 2 s, and a
drip that would take several seconds to finish, so a loaded box cannot move a result across a boundary.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator

import pytest

from maxim.utils import http

DEADLINE_S = 0.6
READ_S = 2.0
# A refused call must end well before the drip would have finished (several seconds) and well before READ_S.
REFUSED_WITHIN_S = DEADLINE_S + 0.9

_BODY = b"x" * 40


def _policy() -> http.TimeoutPolicy:
    return http.TimeoutPolicy(connect_s=READ_S, read_s=READ_S, pool_s=READ_S)


def _drip(conn: socket.socket, data: bytes, gap_s: float, stop: threading.Event) -> None:
    for i in range(len(data)):
        if stop.wait(gap_s):
            return
        try:
            conn.sendall(data[i : i + 1])
        except OSError:
            return


def _body_drip(conn: socket.socket, stop: threading.Event) -> None:
    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n" % len(_BODY))
    _drip(conn, _BODY, 0.1, stop)  # 40 bytes, 4 s in total, each byte far inside READ_S


def _header_drip(conn: socket.socket, stop: threading.Event) -> None:
    head = b"HTTP/1.1 200 OK\r\nX-Slow: " + b"y" * 40 + b"\r\nContent-Length: 2\r\n\r\nok"
    _drip(conn, head, 0.1, stop)


def _continue_flood(conn: socket.socket, stop: threading.Event) -> None:
    while not stop.wait(0.05):  # complete 1xx responses: httpcore loops over them, h11 never limits them
        try:
            conn.sendall(b"HTTP/1.1 100 Continue\r\n\r\n")
        except OSError:
            return


def _fast(conn: socket.socket, stop: threading.Event) -> None:
    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")


@pytest.fixture
def serve() -> Iterator[Callable[[Callable[[socket.socket, threading.Event], None]], str]]:
    """``serve(behaviour)`` starts a one-behaviour HTTP server and returns its URL."""
    stop = threading.Event()
    socks: list[socket.socket] = []

    def start(behaviour: Callable[[socket.socket, threading.Event], None]) -> str:
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        socks.append(srv)

        def loop() -> None:
            while not stop.is_set():
                try:
                    conn, _ = srv.accept()
                except OSError:
                    return
                socks.append(conn)

                def handle(c: socket.socket = conn) -> None:
                    try:
                        c.recv(65536)  # the request
                        behaviour(c, stop)
                        if behaviour is _fast:  # keep-alive: answer one more request on this connection
                            if c.recv(65536):
                                behaviour(c, stop)
                    except OSError:
                        pass

                threading.Thread(target=handle, daemon=True).start()

        threading.Thread(target=loop, daemon=True).start()
        return f"http://127.0.0.1:{srv.getsockname()[1]}/"

    yield start
    stop.set()
    for s in socks:
        try:
            s.close()
        except OSError:
            pass


def _timed_refusal(url: str, **kw) -> tuple[float, BaseException]:
    t0 = time.monotonic()
    with pytest.raises(http.HTTPTimeout) as exc:
        http.fetch_url(url, timeout=_policy(), deadline_s=DEADLINE_S, **kw)
    return time.monotonic() - t0, exc.value


# Without the deadline the 100-Continue flood never ends: the timeout turns that into a bounded FAIL.
@pytest.mark.timeout(20)
@pytest.mark.parametrize("behaviour", [_body_drip, _header_drip, _continue_flood], ids=["body", "headers", "100"])
@pytest.mark.parametrize("policy", ["any", "private"])
def test_a_drip_is_refused_at_the_deadline(serve, behaviour, policy):
    elapsed, exc = _timed_refusal(serve(behaviour), address_policy=policy)
    assert DEADLINE_S * 0.9 <= elapsed <= REFUSED_WITHIN_S, elapsed
    assert "deadline" in (exc.fix_hint or "").lower()


def test_without_a_deadline_the_body_drip_still_completes(serve):
    """No ``deadline_s``: unchanged. Each byte is inside read_s, so httpx reads the whole drip."""
    resp = http.fetch_url(serve(_body_drip), timeout=_policy())
    assert resp.status == 200 and resp.content == _BODY


def test_a_fast_response_inside_the_deadline_is_unchanged(serve):
    resp = http.fetch_url(serve(_fast), timeout=_policy(), deadline_s=DEADLINE_S)
    assert resp.status == 200 and resp.content == b"ok"


def test_the_deadline_does_not_leak_past_the_call(serve):
    with pytest.raises(http.HTTPTimeout):
        http.fetch_url(serve(_body_drip), timeout=_policy(), deadline_s=DEADLINE_S)
    assert http._REQUEST_DEADLINE.get() is None
    http.fetch_url(serve(_fast), timeout=_policy(), deadline_s=DEADLINE_S)
    assert http._REQUEST_DEADLINE.get() is None


def test_an_ordinary_read_timeout_does_not_claim_the_deadline(serve):
    t = http.TimeoutPolicy(connect_s=READ_S, read_s=0.2, pool_s=READ_S)
    with pytest.raises(http.HTTPTimeout) as exc:
        http.fetch_url(serve(_stall), timeout=t, deadline_s=5.0)
    assert "deadline" not in (exc.value.fix_hint or "").lower()


def _stall(conn: socket.socket, stop: threading.Event) -> None:
    stop.wait(5.0)


@pytest.mark.parametrize("value", [0, -1.0, float("nan"), float("inf")])
def test_a_meaningless_deadline_is_refused(value):
    with pytest.raises(ValueError):
        http.fetch_url("http://127.0.0.1:9/", deadline_s=value)


def test_every_external_client_transport_carries_the_deadline_backend():
    """A future ``_external*`` client or mount without the wrapper would silently ignore ``deadline_s``."""
    http._ensure_external_endpoint()
    names = [http._EXTERNAL_ENDPOINT, http._EXTERNAL_PUBLIC_ENDPOINT, *http._EXTERNAL_POLICY_ENDPOINTS.values()]
    for name in names:
        client = http._registry.get_client(name)
        transports = [client._transport, *(t for t in client._mounts.values() if t is not None)]
        for t in transports:
            backend = t._pool._network_backend
            assert isinstance(backend, http._DeadlineBackend), (name, type(backend))


# ── HttpFetchTool: one budget for the robots.txt hops and the page ──────────────────────────────────────────────


def test_robots_hops_take_what_is_left_of_one_budget(monkeypatch):
    """Six robots.txt redirects cannot multiply the deadline: each hop gets the remainder, then the budget is spent."""
    from maxim.tools import http_fetch

    clock = {"t": 100.0}
    monkeypatch.setattr(http_fetch.time, "monotonic", lambda: clock["t"])
    given: list[float] = []

    def fetch_url(url, **kw):
        given.append(kw["deadline_s"])
        clock["t"] += 1.5  # each slow hop spends 1.5 s of the budget
        return http.Response(
            status=301, headers={"Location": "/robots.txt"}, content=b"", elapsed_ms=1.0, endpoint="t", request_id="t"
        )

    monkeypatch.setattr(http, "fetch_url", fetch_url)
    deadline_at = clock["t"] + 4.0
    assert http_fetch._fetch_robots_txt("https://example.com", 2.0, deadline_at=deadline_at) is None
    assert given == pytest.approx([4.0, 2.5, 1.0])  # then the budget is spent: unreadable, the fetch refused


def test_the_page_fetch_gets_the_remainder_and_a_spent_budget_never_connects(monkeypatch):
    from maxim.tools import http_fetch

    clock = {"t": 50.0}
    monkeypatch.setattr(http_fetch.time, "monotonic", lambda: clock["t"])
    given: list[float] = []
    monkeypatch.setattr(http, "fetch_url", lambda url, **kw: given.append(kw["deadline_s"]) or "page")
    assert (
        http_fetch._fetch_page("https://e.x/", timeout_s=3.0, max_bytes=10, public_only=True, deadline_at=53.5)
        == "page"
    )
    assert given == pytest.approx([3.5])
    with pytest.raises(http.HTTPTimeout):
        http_fetch._fetch_page("https://e.x/", timeout_s=3.0, max_bytes=10, public_only=True, deadline_at=50.0)
    assert len(given) == 1
