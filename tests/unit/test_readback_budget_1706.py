"""#1706: the fresh readback's budget is the deadline, not a third of it.

``_get_state_full`` used to split ``FRESH_READ_DEADLINE_S`` into equal thirds and pass them as
``TimeoutPolicy(connect_s=, read_s=, total_s=)``. ``total_s`` (renamed ``pool_s`` by #1715) maps to httpx's POOL wait, which a fresh
connection to the daemon never spends, so the connect and the read each got a third of the deadline: on the
robot (2026-11-19) a 35-43 ms idle read from the Mac timed out under a "100 ms" deadline.

These tests drive the controller's real ``maxim.utils.http.fetch_url`` (so httpx's own timeouts apply) against a
local HTTP server that answers after a set delay. Only the port is rewritten: the controller always dials
``:8000``. The deadlines are scaled up (0.5 s) so a loaded box cannot move a result across a boundary.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.unit._reachy_fakes import FakeDaemon, controller, install

DEADLINE_S = 0.5


def _target(**kw):
    from maxim.hardware.controller import MotionTarget

    return MotionTarget(**kw)


def _outcome(result):
    from maxim.hardware.controller import as_outcome

    return as_outcome(result)


@pytest.fixture
def slow_daemon(monkeypatch):
    """A FakeDaemon whose ``GET /api/state/full`` is served over a real socket after ``delay_s``."""
    import maxim.hardware.reachy.controller as reachy_controller
    import maxim.utils.http as maxim_http

    real_fetch_url = maxim_http.fetch_url
    daemon = FakeDaemon()
    install(monkeypatch, daemon)
    state = {"delay_s": 0.0, "timeouts": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the http.server hook name
            threading.Event().wait(state["delay_s"])  # the answer is late, before any header
            body = json.dumps(daemon.state_full()).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client gave up first: the over-budget case

        def log_message(self, *_a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def fetch_url(url, **kw):
        if "/api/state/full" in url:
            daemon.rest_calls += 1
            state["timeouts"].append(kw.get("timeout"))
            return real_fetch_url(url.replace(":8000/", f":{port}/"), **kw)
        return daemon.fetch_url(url, **kw)

    monkeypatch.setattr(maxim_http, "fetch_url", fetch_url)
    monkeypatch.setattr(reachy_controller, "FRESH_READ_DEADLINE_S", DEADLINE_S)
    daemon.slow = state
    try:
        yield daemon
    finally:
        server.shutdown()
        server.server_close()


def test_a_read_inside_the_deadline_but_past_a_third_of_it_succeeds(slow_daemon):
    """Red before #1706: 0.25 s is past the old 0.167 s per-phase read timeout and inside the 0.5 s deadline."""
    slow_daemon.slow["delay_s"] = DEADLINE_S / 2
    ctl = controller(slow_daemon)
    out = ctl.goto_target(_target(head_yaw=0.1, duration=1.0))
    assert out, _outcome(out).reason
    assert len(slow_daemon.gotos()) == 1


def test_a_read_past_the_deadline_refuses_reference_and_nothing_moves(slow_daemon):
    slow_daemon.slow["delay_s"] = DEADLINE_S * 1.5
    ctl = controller(slow_daemon)
    out = ctl.goto_target(_target(head_yaw=0.1, duration=1.0))
    assert not out and _outcome(out).kind == "reference", _outcome(out).reason
    assert slow_daemon.gotos() == []


def test_no_httpx_phase_is_shorter_than_the_deadline(slow_daemon):
    """The timeouts actually passed: connect, read and the pool wait may each use the whole deadline.

    httpx has no whole-request timeout; the controller's elapsed-time check after the read is what bounds the
    reference's age at the deadline. That check is pinned by
    ``test_motion_path_1499.py::test_a_slow_fresh_read_past_the_deadline_refuses`` (a fake fetch); the over-budget
    read here is refused by httpx's own read timeout instead.
    """
    ctl = controller(slow_daemon)
    assert ctl.goto_target(_target(head_yaw=0.1, duration=1.0))
    policy = slow_daemon.slow["timeouts"][0]
    httpx_timeout = policy.to_httpx()
    assert httpx_timeout.connect == pytest.approx(DEADLINE_S)
    assert httpx_timeout.read == pytest.approx(DEADLINE_S)
    assert httpx_timeout.write == pytest.approx(DEADLINE_S)
    assert httpx_timeout.pool == pytest.approx(DEADLINE_S)
