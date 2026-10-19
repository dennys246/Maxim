"""The ``console`` extra serves ``/ws`` on a REAL uvicorn (#1347; standard R2, #1335).

``pymaxim[console]`` installed plain ``uvicorn``, which has no WebSocket implementation, and pymaxim's core
dependencies bring none, so after a clean install ``maxim serve`` answered every ``/ws`` upgrade with 404 and logged
"No supported WebSocket library detected": the Console had no live stream. The ``/ws`` tests in
``test_console_auth.py`` use Starlette's ``TestClient``, which never goes through uvicorn, so none could see it.

This test runs the shipped entry point, ``python -m maxim serve``, on a free loopback port and makes the upgrade with
a raw socket: a websocket client library here would install exactly what the extra must provide. In the fast suite
it passes over whatever else is installed; the R2 extras matrix's console cells (``scripts/extras_matrix.py``) run
it where ``pymaxim[console]`` and pytest are ALL that is installed, which is where it proves the extra.
"""

from __future__ import annotations

import base64
import os
import socket
import subprocess
import sys
import time

import pytest

pytestmark = (
    pytest.mark.extras_console
)  # the R2 extras matrix's console cells (scripts/lane_rosters/extras-console.json)


@pytest.fixture(scope="module")
def serve(tmp_path_factory: pytest.TempPathFactory):
    """A real ``maxim serve`` on a free port, its HOME, data and config dirs a temp dir. Yields (port, token, proc)."""
    tmp_path = tmp_path_factory.mktemp("serve")
    pytest.importorskip("uvicorn", reason="requires the `console` extra (pip install 'pymaxim[console]')")
    pytest.importorskip("fastapi", reason="requires the `console` extra (pip install 'pymaxim[console]')")
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "MAXIM_DATA_HOME": str(tmp_path / ".maxim"),
        "XDG_CONFIG_HOME": str(tmp_path / ".config"),  # the console token lives here: not the suite's shared root
    }
    env.pop("MAXIM_CONSOLE_UI_DIST", None)
    cmd = [sys.executable, "-m", "maxim", "serve"]
    shown = subprocess.run([*cmd, "--show-token"], env=env, capture_output=True, text=True, timeout=60, check=True)
    token = shown.stdout.strip().splitlines()[-1]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [*cmd, "--port", str(port)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"maxim serve exited {proc.returncode}: {proc.stdout.read()[-2000:]}")
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("maxim serve did not listen within 60 s")
        yield port, token, proc
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()


def _upgrade(port: int, token: str) -> str:
    """The raw RFC 6455 handshake on /ws; returns the response's status line."""
    key = base64.b64encode(os.urandom(16)).decode()
    request = (
        f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nAuthorization: Bearer {token}\r\n\r\n"
    )
    with socket.create_connection(("127.0.0.1", port), timeout=10) as conn:
        conn.sendall(request.encode())
        response = b""
        while b"\r\n" not in response:
            chunk = conn.recv(4096)
            if not chunk:
                break
            response += chunk
    return response.split(b"\r\n", 1)[0].decode(errors="replace")


def test_maxim_serve_upgrades_ws_on_a_real_uvicorn(serve):
    port, token, proc = serve
    assert _upgrade(port, token).startswith("HTTP/1.1 101"), "the console extra cannot serve /ws"
    assert proc.poll() is None, "the 101 came from something other than our server (it is not running)"


def test_an_unauthenticated_upgrade_is_refused_on_a_real_uvicorn(serve):
    """The control: the auth guard runs on the real WebSocket path, so the 101 above is the extra working, not a
    missing guard. A bad token is closed before accept, which a real WebSocket implementation answers with 403; it is
    a second red against #1347, where no implementation means the upgrade never reaches the guard (404)."""
    port, _token, _proc = serve
    status = _upgrade(port, "not-the-token")
    assert status.startswith("HTTP/1.1 403"), status
