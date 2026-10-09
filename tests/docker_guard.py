"""The fast suite's Docker guard (#1103; owner decision 2026-10-08: the fast suite is hermetic, real-Docker tests
run in the nightly SLOW lane, like real-network tests).

Why: two CI runs of ONE commit reported different coverage because the fast suite touched the runner's real Docker
daemon. ``create_sandbox(backend="auto")`` and ``check_docker_available()`` took the Docker path when the daemon
answered and the tmpdir path when it did not, so which lines ran depended on the machine, not the code.

Installed once for the session by ``tests/conftest.py`` (so a module-level probe at COLLECTION time is caught too):
spawning the ``docker`` CLI (``subprocess.Popen`` with ``docker`` as the program, or ``executable="docker"``, or a
``shell=True`` command string whose first word is ``docker``) raises ``DockerBlocked`` -- deliberately NOT an
``OSError``, so ``check_docker_available``'s ``except OSError`` cannot turn it into a quiet "unavailable" -- and is
recorded in ``attempts``; the conftest fixture also FAILS the test at teardown, so a handler that swallows the
exception still cannot hide the touch.

``@pytest.mark.slow`` lifts the guard for that one test: the slow lane (nightly, ``-m slow``, held to
``scripts/lane_rosters/slow.json``) is where a real daemon may be used. A fast test that needs a sandbox passes
``backend="tmpdir"``; one that needs the Docker DECISION monkeypatches ``check_docker_available`` and fakes the
runner.

Scope: the docker CLI only (the one way ``src/maxim`` reaches the daemon, ``container_runner.LocalDockerRunner`` and
``check_docker_available``); a docker invocation buried inside a script a test runs (``bash -c "x; docker ..."``) or a
direct ``/var/run/docker.sock`` connection is not seen. The guard stays on through interpreter exit, so in a LOCAL ``-m slow`` run a container the
runner's atexit reaper would remove is left behind (the blocked call is swallowed there); CI runners are ephemeral.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any

_real_popen_init = subprocess.Popen.__init__

enabled = True
attempts: list[str] = []


class DockerBlocked(RuntimeError):
    """A fast-suite test tried to reach the real Docker daemon."""


def _program(args: Any, executable: Any, shell: bool) -> str:
    if executable:
        return os.path.basename(os.fsdecode(executable))
    if isinstance(args, (str, bytes, os.PathLike)):
        text = os.fsdecode(args)
        if shell:
            try:
                words = shlex.split(text)
            except ValueError:
                words = text.split()
            return os.path.basename(words[0]) if words else ""
        return os.path.basename(text)
    try:
        first = next(iter(args))
    except (TypeError, StopIteration):
        return ""
    return os.path.basename(os.fsdecode(first))


def is_docker_command(args: Any, executable: Any = None, shell: bool = False) -> bool:
    return _program(args, executable, shell) in ("docker", "docker.exe")


def _guarded_popen_init(self: subprocess.Popen, args: Any, *a: Any, **kw: Any) -> None:  # type: ignore[type-arg]
    executable = a[1] if len(a) > 1 else kw.get("executable")  # Popen(args, bufsize, executable, ...)
    if enabled and is_docker_command(args, executable, bool(kw.get("shell", False))):
        shown = args if isinstance(args, (str, bytes)) else " ".join(map(str, list(args)[:3]))
        attempts.append(str(shown))
        raise DockerBlocked(
            f"docker blocked in the fast suite: {shown!r} -- force backend='tmpdir', monkeypatch "
            "check_docker_available + fake the runner, or mark the test slow and add it to "
            "scripts/lane_rosters/slow.json (#1103)"
        )
    _real_popen_init(self, args, *a, **kw)


def install() -> None:
    subprocess.Popen.__init__ = _guarded_popen_init  # type: ignore[method-assign]
