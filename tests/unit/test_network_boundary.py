"""CI's network boundary at the process-tree level (roadmap 1.3.2 item 7, #940; owner decision 2026-10-04).

The in-process guard (tests/network_guard.py) cannot see a SUBPROCESS a test spawns (git, pip, curl, another
python). In CI the fast suite therefore runs inside a loopback-only Linux network namespace, and sets
``MAXIM_EXPECT_NETNS=1`` inside it. This is the positive control: with the variable set, a child process (which the
in-process guard never touched) cannot reach the internet, while loopback still works. Without the variable it
skips (macOS has no namespace); ``tests/unit/test_ci_workflow_shape.py`` pins that CI sets it, so a CI run that
dropped the namespace but kept the variable fails here, and one that dropped the variable fails there.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MAXIM_EXPECT_NETNS") != "1",
    reason="only meaningful inside CI's loopback-only network namespace (MAXIM_EXPECT_NETNS=1)",
)


def _child(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False)


def test_a_subprocess_cannot_open_a_tcp_connection_to_the_internet():
    r = _child("import socket; socket.create_connection(('1.1.1.1', 443), timeout=5); print('CONNECTED')")
    assert r.returncode != 0 and "CONNECTED" not in r.stdout, r.stdout + r.stderr


def test_a_subprocess_cannot_resolve_a_public_name():
    r = _child("import socket; print(socket.gethostbyname('example.com'))")
    assert r.returncode != 0, r.stdout


def test_loopback_still_works():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        r = _child(f"import socket; socket.create_connection(('127.0.0.1', {port}), timeout=5); print('OK')")
        assert r.returncode == 0 and "OK" in r.stdout, r.stderr
