"""scripts/rig_suite.sh end to end, over a fake ``ssh`` that runs each remote command locally.

The rig's SSH server (Tailscale SSH, 2026-11-03) reports exit status 0 for every remote command while stdout arrives
intact, so rig_suite.sh must not read any decision from ssh's own exit status. The fake ssh models three transports:
``tailscale`` (always exits 0), ``openssh`` (passes the command's status through) and ``drop`` (loses the last line
of output, then exits 0). The remote side runs zsh where it is installed (the rig's login shell), else bash.

Each test builds an upstream repo whose ``scripts/suite_at_commit.sh`` is a stub, a worktree cloned from it, and the
rig's clone at ``$HOME/maxim-suites/repo`` under a temporary HOME; ``git push ssh://…`` reaches that clone through
the same fake ssh. No maxim import, no network.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "rig_suite.sh"
REMOTE_SHELL = shutil.which("zsh") or shutil.which("bash") or "/bin/bash"
TRANSPORTS = ("tailscale", "openssh")

# Red on main (rig_suite.sh trusts ssh's exit status and polls a fixed 60 s): (test, transport) pairs. Emptied by the
# fix; each must fail on main for its own reason.
RED_ON_MAIN = {
    ("g1", "tailscale"),
    ("g1", "openssh"),
    ("g2", "tailscale"),
    ("g2", "openssh"),
    ("g3", "openssh"),
    ("g4", "tailscale"),
    ("g6", "tailscale"),
    ("g6", "openssh"),
}

_FAKE_SSH = textwrap.dedent(
    """\
    #!/bin/bash
    # A fake ssh: options are skipped (-G answers git's probe), the host is ignored, the command runs locally.
    while [ $# -gt 0 ]; do
        case "$1" in
            -G) printf 'user fake\\nhostname fake\\n'; exit 0 ;;
            -o|-p|-i|-l|-F) shift 2 ;;
            -*) shift ;;
            *) break ;;
        esac
    done
    shift  # the host
    cd "$HOME" || exit 255
    case "${FAKE_SSH_MODE:-tailscale}" in
        tailscale) "$FAKE_REMOTE_SHELL" -c "$*"; exit 0 ;;
        openssh) "$FAKE_REMOTE_SHELL" -c "$*"; exit $? ;;
        drop) out="$("$FAKE_REMOTE_SHELL" -c "$*")"; printf '%s\\n' "$out" | sed '$d'; exit 0 ;;
    esac
    """
)

_STUB_SUITE = textwrap.dedent(
    """\
    #!/bin/bash
    echo "suite at $1"
    for s in fast-suite release-gates memory-hub ratchet; do echo "suite stage $s: exit=0"; done
    sleep "${STUB_SLEEP:-0}"
    exit "${STUB_EXIT:-0}"
    """
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


class _Rig:
    def __init__(self, tmp: Path, transport: str) -> None:
        self.transport = transport
        self.home = tmp / "home"
        self.logs = self.home / "maxim-suites" / "logs"
        bindir = tmp / "bin"
        bindir.mkdir()
        (bindir / "ssh").write_text(_FAKE_SSH)
        (bindir / "ssh").chmod(0o755)
        upstream = tmp / "upstream"
        upstream.mkdir()
        _git(upstream, "init", "-q", "-b", "main")
        _git(upstream, "config", "user.email", "t@example.invalid")
        _git(upstream, "config", "user.name", "t")
        _git(upstream, "config", "commit.gpgsign", "false")
        (upstream / "scripts").mkdir()
        (upstream / "scripts" / "suite_at_commit.sh").write_text(_STUB_SUITE)
        (upstream / "scripts" / "suite_at_commit.sh").chmod(0o755)
        _git(upstream, "add", "-A")
        _git(upstream, "commit", "-q", "-m", "stub")
        self.wt = tmp / "wt"
        subprocess.run(["git", "clone", "-q", str(upstream), str(self.wt)], check=True)
        self.sha = _git(self.wt, "rev-parse", "HEAD")
        self.home.mkdir()
        subprocess.run(["git", "clone", "-q", str(upstream), str(self.home / "maxim-suites" / "repo")], check=True)
        self.logs.mkdir(parents=True)
        self.env = {
            "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(self.home),
            "RIG_SUITE_HOST": "rig@fake",
            "RIG_SUITE_POLL_S": "1",
            "FAKE_SSH_MODE": transport,
            "FAKE_REMOTE_SHELL": REMOTE_SHELL,
            "GIT_SSH_VARIANT": "ssh",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "LC_ALL": "C",
        }

    def run(self, *args: str, timeout: float = 60, **extra: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["bash", str(SCRIPT), *args],
                env={**self.env, **extra},
                capture_output=True,
                text=True,
                timeout=timeout,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise AssertionError(f"rig_suite.sh {' '.join(args)} did not finish in {timeout} s") from exc


@pytest.fixture(params=TRANSPORTS)
def rig(request: pytest.FixtureRequest, tmp_path: Path) -> _Rig:
    gate = request.node.originalname.split("_")[1]
    if (gate, request.param) in RED_ON_MAIN:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True, raises=AssertionError, reason="red gate: rig_suite.sh trusts ssh's exit status"
            )
        )
    return _Rig(tmp_path, request.param)


def test_g1_a_fresh_label_launches_and_a_passing_suite_passes(rig: _Rig) -> None:
    r = rig.run(str(rig.wt), "fresh")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "rig suite: PASS" in r.stdout
    assert _git(rig.home / "maxim-suites" / "repo", "rev-parse", "refs/heads/suite/fresh") == rig.sha  # it landed


def test_g2_a_failing_suite_fails_with_its_own_exit(rig: _Rig) -> None:
    r = rig.run(str(rig.wt), "failing", STUB_EXIT="1")
    assert r.returncode == 1, r.stdout + r.stderr


def test_g3_a_stale_log_from_a_launch_that_never_wrote_is_never_read_as_this_runs_pass(rig: _Rig) -> None:
    log = rig.logs / "stale.log"
    log.write_text("suite at 0000000\nsuite stage fast-suite: exit=0\nsuite-exit=0\n")
    log.chmod(0o444)  # this launch cannot write it, though the remote launch block may still report success
    try:
        r = rig.run(str(rig.wt), "stale")
    finally:
        log.chmod(0o644)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "rig suite: PASS" not in r.stdout


def test_g4_an_attach_to_a_run_that_died_without_finishing_exits_2(rig: _Rig) -> None:
    gone = subprocess.Popen(["true"])
    gone.wait()
    (rig.logs / "dead.log").write_text("rig-suite-run=deadbeef0000-1-1-1\nsuite at x\n")
    (rig.logs / "dead.pid").write_text(f"{gone.pid}\n")
    r = rig.run("--attach", "dead", timeout=30)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "died without finishing" in r.stderr


def test_g5_a_label_still_running_refuses_a_second_launch(rig: _Rig) -> None:
    nonce = "cafe00000000-1-1-1"
    live = subprocess.Popen(["bash", "-c", "sleep 60", f"rig-suite-{nonce}"], start_new_session=True)
    try:
        (rig.logs / "busy.log").write_text(f"rig-suite-run={nonce}\nsuite at x\n")
        (rig.logs / "busy.pid").write_text(f"{live.pid}\n")
        r = rig.run(str(rig.wt), "busy")
    finally:
        os.killpg(live.pid, signal.SIGKILL)
        live.wait()
    assert r.returncode == 2, r.stdout + r.stderr
    assert "still going" in r.stderr


def test_g6_a_reused_pid_is_not_a_running_label(rig: _Rig) -> None:
    other = subprocess.Popen(["sleep", "60"], start_new_session=True)  # alive, but not that label's run
    try:
        (rig.logs / "reused.log").write_text("rig-suite-run=feed00000000-1-1-1\nsuite at x\n")
        (rig.logs / "reused.pid").write_text(f"{other.pid}\n")
        r = rig.run(str(rig.wt), "reused")
    finally:
        os.killpg(other.pid, signal.SIGKILL)
        other.wait()
    assert r.returncode == 0, r.stdout + r.stderr


def test_g7_a_transport_that_loses_output_never_launches_or_passes(tmp_path: Path) -> None:
    rig = _Rig(tmp_path, "drop")
    r = rig.run(str(rig.wt), "lossy")
    assert r.returncode == 2, r.stdout + r.stderr
    assert not (rig.logs / "lossy.log").exists()
