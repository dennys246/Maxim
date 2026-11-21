"""CI's network boundary at the process-tree level (roadmap 1.3.2 item 7, #940; owner decision 2026-10-04).

The in-process guard (tests/network_guard.py) cannot see a SUBPROCESS a test spawns (git, pip, curl, another
python). In CI the fast suite therefore runs inside a loopback-only Linux network namespace, and sets
``MAXIM_EXPECT_NETNS=1`` inside it. This is the positive control: with the variable set, a child process (which the
in-process guard never touched) has no network interface but loopback, while loopback still works. A network
namespace does not cover filesystem sockets, so (#1426) the namespace also drops the docker group, covers the
runner's session sockets and runs with no_new_privs; ``MAXIM_EXPECT_NO_DOCKER=1`` arms the second control below
(every lane but slow-tests, which keeps Docker on purpose). scripts/ci_netns_probe.py checks the same from
inside every lane before its suite; this file is the second reader, in the unit-tests lane. Without the variable it
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


@pytest.mark.skipif(
    os.environ.get("MAXIM_EXPECT_NO_DOCKER") != "1",
    reason="only meaningful in a lane that closes Docker (MAXIM_EXPECT_NO_DOCKER=1)",
)
def test_a_subprocess_cannot_reach_the_docker_daemon_or_regain_root():
    """#1426: the docker group is dropped and no_new_privs is set. A refused or absent socket passes; one that answers
    fails. On CI at least one docker socket must exist, or the check proved nothing."""
    code = (
        "import os, socket\n"
        "for p in ('/var/run/docker.sock', '/run/docker.sock'):\n"
        "    s = socket.socket(socket.AF_UNIX)\n"
        "    try:\n"
        "        s.connect(p); print('REACHED', p)\n"
        "    except (PermissionError, FileNotFoundError):\n"
        "        print('CLOSED' if os.path.exists(p) else 'ABSENT', p)\n"
    )
    r = _child(code)
    assert r.returncode == 0 and "REACHED" not in r.stdout, r.stdout + r.stderr
    if os.environ.get("GITHUB_ACTIONS") == "true":
        assert "CLOSED" in r.stdout, r.stdout
    try:
        sudo = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=30, check=False)
    except FileNotFoundError:
        return
    assert sudo.returncode != 0, "sudo -n succeeded inside the namespace: no_new_privs is not in force"
