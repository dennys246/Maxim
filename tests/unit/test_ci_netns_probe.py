"""The rules of scripts/ci_netns_probe.py, the in-namespace check scripts/ci_netns.sh runs before every lane's suite
(#1426). Each rule is driven with real temporary sockets: one that answers, one that is a plain file (refused, as a
/dev/null-covered socket is), one the test cannot open (denied) and one that does not exist."""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="AF_UNIX path semantics are POSIX")


def _probe():
    spec = importlib.util.spec_from_file_location("ci_netns_probe_under_test", ROOT / "scripts" / "ci_netns_probe.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def socks(tmp_path):
    short = Path("/tmp") / f"nsp{os.getpid()}"  # AF_UNIX paths are capped near 108 bytes
    short.mkdir(exist_ok=True)
    listening = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    answered = short / "answers.sock"
    listening.bind(str(answered))
    listening.listen(1)
    denied = short / "denied.sock"
    blocked = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    blocked.bind(str(denied))
    blocked.listen(1)
    os.chmod(denied, 0)
    covered = short / "covered.sock"
    covered.write_text("")  # a plain file stands in for /dev/null over a socket: connect is refused
    paths = {"answered": str(answered), "denied": str(denied), "covered": str(covered), "absent": str(short / "x")}
    yield paths
    listening.close()
    blocked.close()
    for p in (answered, denied, covered):
        p.unlink(missing_ok=True)
    short.rmdir()


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root opens a mode-0 socket")
def test_connect_result_names_each_outcome(socks):
    p = _probe()
    assert p.connect_result(socks["answered"]) == "answered"
    assert p.connect_result(socks["denied"]) == "denied"
    assert p.connect_result(socks["covered"]) == "refused"
    assert p.connect_result(socks["absent"]) == "absent"


def _breaches(p, socks, **over):
    args = dict(
        mode="closed",
        on_ci=True,
        docker=[socks["denied"]],
        containerd=socks["absent"],
        runtime=[],
        links=["lo"],
        sudo_is_refused=True,
    )
    args.update(over)
    return p.breaches(**args)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root opens a mode-0 socket")
def test_a_closed_namespace_holds(socks):
    assert _breaches(_probe(), socks) == []


@pytest.mark.parametrize(
    "over, expect",
    [
        ({"docker": ["@answered"]}, "the docker socket must be refused"),
        ({"docker": ["@absent"]}, "probe vacuous"),
        ({"links": ["eth0", "lo"]}, "only lo may exist"),
        ({"sudo_is_refused": False}, "no_new_privs is not in force"),
        ({"containerd": "@answered"}, "answered"),
        ({"runtime": ["@answered"]}, "should have covered"),
        ({"mode": "keep", "docker": ["@absent"]}, "no docker socket answered"),
        ({"mode": "sideways"}, "unknown mode"),
    ],
    ids=[
        "docker-answers",
        "vacuous-on-ci",
        "an-interface",
        "sudo-works",
        "containerd",
        "runtime-socket",
        "keep-dead",
        "mode",
    ],
)
def test_each_breach_is_reported(socks, over, expect):
    resolved = {
        k: (
            [socks[v[1:]] if v.startswith("@") else v for v in val]
            if isinstance(val, list)
            else (socks[val[1:]] if isinstance(val, str) and val.startswith("@") else val)
        )
        for k, val in over.items()
    }
    found = _breaches(_probe(), socks, **resolved)
    assert any(expect in line for line in found), found


def test_off_ci_an_absent_docker_socket_is_not_vacuous_and_keep_is_a_skip(socks):
    p = _probe()
    assert _breaches(p, socks, on_ci=False, docker=[socks["absent"]]) == []
    assert _breaches(p, socks, on_ci=False, mode="keep", docker=[socks["absent"]]) == []


def test_keep_mode_on_ci_passes_when_docker_answers(socks):
    assert _breaches(_probe(), socks, mode="keep", docker=[socks["answered"]]) == []


def test_a_covered_runtime_socket_is_not_a_socket_and_pulse_is_left_alone(tmp_path):
    p = _probe()
    short = Path("/tmp") / f"nsr{os.getpid()}"
    (short / "pulse").mkdir(parents=True, exist_ok=True)
    (short / "systemd").mkdir(exist_ok=True)
    keep = []
    try:
        for rel in ("bus", "pulse/native", "systemd/private"):
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.bind(str(short / rel))
            keep.append(s)
        (short / "covered").write_text("")
        found = sorted(p.runtime_sockets(short))
        assert found == [str(short / "bus"), str(short / "systemd" / "private")]
    finally:
        for s in keep:
            s.close()
        for rel in ("bus", "pulse/native", "systemd/private", "covered"):
            (short / rel).unlink(missing_ok=True)
        for d in ("pulse", "systemd", ""):
            (short / d).rmdir()


def test_docker_host_adds_its_unix_path():
    p = _probe()
    assert p.docker_paths({"DOCKER_HOST": "unix:///run/user/1001/docker.sock"})[-1] == "/run/user/1001/docker.sock"
    assert p.docker_paths({"DOCKER_HOST": "tcp://127.0.0.1:2375"}) == list(p.DOCKER_PATHS)


def test_interfaces_reads_this_namespace_from_the_kernel_not_sysfs():
    """sysfs shows the namespace that MOUNTED it, so inside ``unshare --net`` /sys/class/net still lists the host's
    interfaces; the probe must ask the kernel (if_nameindex) for its own."""
    p = _probe()
    assert p.interfaces() == sorted(name for _i, name in socket.if_nameindex())


@pytest.mark.skipif(not __import__("shutil").which("unshare"), reason="needs util-linux unshare")
def test_inside_a_fresh_network_namespace_interfaces_is_only_lo():
    """Unprivileged (a user namespace mapping this user to root), no sudo: where the kernel allows it."""
    import subprocess

    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('p', {str(ROOT / 'scripts' / 'ci_netns_probe.py')!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(m.interfaces())\n"
    )
    r = subprocess.run(
        ["unshare", "--user", "--map-root-user", "--net", sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if r.returncode != 0 and "Operation not permitted" in r.stderr:
        pytest.skip("unprivileged user namespaces are disabled here")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "['lo']"


def test_a_searchable_but_unlistable_subtree_is_a_breach_and_an_unsearchable_one_is_not(tmp_path):
    """A socket in a directory you can search (x) but not list (r) is reachable by name, so that subtree is a breach;
    a mode-000 directory (systemd's systemd/inaccessible/dir on every runner) hides nothing reachable."""
    p = _probe()
    short = Path("/tmp") / f"nse{os.getpid()}"
    (short / "searchable").mkdir(parents=True, exist_ok=True)
    (short / "sealed").mkdir(exist_ok=True)
    os.chmod(short / "searchable", 0o100)
    os.chmod(short / "sealed", 0)
    try:
        if os.access(short / "sealed", os.R_OK):
            pytest.skip("running as root: nothing is unreadable")
        errors: list[str] = []
        p.runtime_sockets(short, errors)
        assert len(errors) == 1 and "searchable" in errors[0] and "cannot scan" in errors[0], errors
        found = p.breaches(
            mode="closed",
            on_ci=False,
            docker=[],
            containerd=str(short / "x"),
            runtime=[],
            links=["lo"],
            sudo_is_refused=True,
            scan_errors=errors,
        )
        assert found == errors
    finally:
        os.chmod(short / "searchable", 0o700)
        os.chmod(short / "sealed", 0o700)
        (short / "searchable").rmdir()
        (short / "sealed").rmdir()
        short.rmdir()
