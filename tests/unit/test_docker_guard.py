"""The fast suite's Docker guard (tests/docker_guard.py, #1103) is installed and recognises the docker CLI."""

from __future__ import annotations

import subprocess

import pytest

from tests import docker_guard


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        (["docker", "info"], {}, True),
        (["/usr/local/bin/docker", "run", "x"], {}, True),
        # Popen("docker info") without shell runs a program literally named "docker info"; with shell, the first word
        ("docker info", {}, False),
        ("docker info", {"shell": True}, True),
        (["anything"], {"executable": "/usr/bin/docker"}, True),
        (["dockerd"], {}, False),
        (["echo", "docker"], {}, False),
        ("echo docker", {"shell": True}, False),
        ([], {}, False),
    ],
)
def test_is_docker_command(args, kwargs, expected):
    assert docker_guard.is_docker_command(args, kwargs.get("executable"), kwargs.get("shell", False)) is expected


def test_a_fast_test_cannot_spawn_docker():
    """The guard is live in the fast suite: spawning docker raises (not an OSError, so a probe's ``except OSError``
    cannot read it as "unavailable") and is recorded for the conftest teardown check."""
    before = len(docker_guard.attempts)
    try:
        with pytest.raises(docker_guard.DockerBlocked):
            subprocess.run(["docker", "info"], capture_output=True, check=False)
        assert not issubclass(docker_guard.DockerBlocked, OSError)
        assert docker_guard.attempts[before:] == ["docker info"]
    finally:
        del docker_guard.attempts[before:]  # this test's own touch is deliberate; don't fail it at teardown


def test_other_programs_still_run():
    assert subprocess.run(["true"], check=False).returncode == 0
