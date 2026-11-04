"""scripts/box_quiet.py: the shared box-slot waiter (two suites at most, no prepr/mypy beside any other run).

Each "how could this input lie" line in the module docstring has a test here: a waiter never counts itself, one run
is counted once, a narrow pytest is not a suite, the owner's priority rulings (critical tie-break, drain for a
critical or long-waiting check), pid reuse, and the lock/claim pair that stops two waiters starting into one slot.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from scripts import box_quiet as B

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "box_quiet.py"
FRAMEWORK_PY = "/opt/homebrew/Cellar/python@3.14/3.14.6/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python"


def P(pid: int, ppid: int, cmd: str, started: str = "t0") -> B.Proc:
    return B.Proc(pid, ppid, started, tuple(cmd.split()))


@pytest.mark.parametrize(
    "cmd,kind",
    [
        ("bash scripts/suite_at_commit.sh 0df45262", B.SUITE),
        ("/bin/bash /Users/x/Maxim/scripts/suite_at_commit.sh --no-cov abc", B.SUITE),
        ("python -m pytest tests/ -q -m not slow --ignore=tests/integration/test_memory_hub.py", B.SUITE),
        ("/usr/bin/python3.12 -m pytest tests -q", B.SUITE),
        ("python -m pytest /Users/x/Maxim/.worktrees/a/tests/ -q", B.SUITE),
        ("python -m pytest ./tests", B.SUITE),
        ("python -X dev -W error -m pytest tests/", B.SUITE),
        ("env PYTHONPATH=/x/src python -m pytest tests/", B.SUITE),
        ("pytest -q", B.SUITE),  # testpaths = tests
        # ps drops quoting: -m "not slow" arrives as three words, and "slow" is not a path.
        ("python -m pytest -q -m not slow", B.SUITE),
        ("python -m pytest -k not slow and not integration tests/", B.SUITE),
        ("python -m pytest .", B.SUITE),
        ("bash -o pipefail scripts/suite_at_commit.sh abc", B.SUITE),
        ("env -u MAXIM_LOG_FILE PYTHONPATH=/x/src python -m pytest tests/", B.SUITE),
        (f"{FRAMEWORK_PY} -m pytest tests/", B.SUITE),
        ("python3 /Users/x/Maxim/scripts/prepr.py --no-pr", B.CHECK),
        (f"{FRAMEWORK_PY} /Users/x/Maxim/scripts/prepr.py", B.CHECK),
        ("bash scripts/prepr.sh", B.CHECK),
        ("bash scripts/run_mypy_public.sh", B.CHECK),
        # Not a full suite: a narrow pytest (even with -k tests), an editor's mypy, inline code, unrelated programs.
        ("python -m pytest tests/unit/test_box_quiet.py -q", None),
        ("python -m pytest -m not slow tests/unit/test_x.py", None),
        ("python -m pytest tests/unit/test_x.py::test_y", None),
        ("python -m pytest -k tests tests/unit/test_x.py", None),
        ("/x/.venv/bin/mypy src/maxim/foo.py", None),
        ("/x/.venv/bin/mypy src/maxim/foo.py --warn-unused-ignores", None),
        ("python3 -c import pytest", None),
        ("grep suite_at_commit.sh log.txt", None),
        ("shellcheck scripts/prepr.sh", None),
        ("", None),
    ],
)
def test_a_process_is_classified_by_its_own_program(cmd, kind):
    assert B.classify(tuple(cmd.split())) == kind


@pytest.mark.parametrize(
    "cmd",
    [
        # The self-matching waiter bug class (2026-11-02/03): the command it WILL run is in its argv.
        "python3 scripts/box_quiet.py wait --critical -- bash scripts/suite_at_commit.sh abc",
        f"{FRAMEWORK_PY} scripts/box_quiet.py wait -- python -m pytest tests/",
        "bash -c while pgrep -f ^(python[0-9.]* -m pytest |bash scripts/suite_at_commit\\.sh ) ; do sleep 60; done",
        "zsh -c sleep 60; bash scripts/prepr.sh",
    ],
)
def test_a_waiter_whose_argv_names_a_runner_is_not_counted(cmd):
    assert B.classify(tuple(cmd.split())) is None


def test_one_run_is_counted_once_under_its_top_most_process():
    procs = [
        P(100, 1, "bash scripts/suite_at_commit.sh abc"),
        P(101, 100, "python -m pytest tests/ -q"),
        P(200, 1, "python3 scripts/prepr.py"),
        P(201, 200, "bash scripts/run_mypy_public.sh"),
        P(300, 1, "python -m pytest tests/ -q"),  # a bare full suite, no wrapper
    ]
    now = B.running(procs, claims={})
    assert sorted(now[B.SUITE]) == [100, 300]
    assert now[B.CHECK] == [200]


def test_a_claimed_child_counts_before_ps_shows_it_and_its_descendants_are_not_recounted():
    procs = [P(500, 1, "bash"), P(501, 500, "python -m pytest tests/ -q")]
    now = B.running(procs, claims={500: B.SUITE, 777: B.CHECK})
    assert now[B.SUITE] == [500]
    assert now[B.CHECK] == [777]


@pytest.mark.parametrize(
    "suites,checks,suite_fits,check_fits",
    [(0, 0, True, True), (1, 0, True, False), (2, 0, False, False), (0, 1, False, False)],
)
def test_the_box_rule(suites, checks, suite_fits, check_fits):
    now = {B.SUITE: list(range(suites)), B.CHECK: list(range(checks))}
    assert B.fits(B.SUITE, now) is suite_fits
    assert B.fits(B.CHECK, now) is check_fits


NOW = 10_000 * 10**9


def T(critical: bool, waited_s: float, pid: int, kind: str) -> B.Ticket:
    return B.Ticket(critical, NOW - int(waited_s * 1e9), pid, kind, "t0")


def _now(suites=0, checks=0):
    return {B.SUITE: list(range(suites)), B.CHECK: list(range(checks))}


def test_a_fitting_suite_goes_past_a_young_check_that_cannot_fit():
    # Efficiency: a non-critical check that has waited under 10 min does not idle the free suite slot.
    me, young = T(False, 1, 1, B.SUITE), T(False, 60, 2, B.CHECK)
    assert B.may_start(me, [me, young], _now(suites=1), NOW)


def test_a_check_waiting_ten_minutes_drains_the_box_of_new_non_critical_suites():
    me, old = T(False, 1, 1, B.SUITE), T(False, 601, 2, B.CHECK)
    assert not B.may_start(me, [me, old], _now(suites=1), NOW)
    # ... but a critical suite still goes past a non-critical draining check.
    crit = T(True, 1, 3, B.SUITE)
    assert B.may_start(crit, [crit, old], _now(suites=1), NOW)


def test_a_waiting_critical_check_drains_the_box_of_every_new_suite():
    crit_check = T(True, 1, 2, B.CHECK)
    for me in (T(False, 900, 1, B.SUITE), T(True, 900, 3, B.SUITE)):
        assert not B.may_start(me, [me, crit_check], _now(suites=1), NOW)


def test_an_older_suite_and_a_draining_check_never_hold_each_other_on_an_empty_box():
    """The hang the delta review found: an older suite waited for a draining check, which waited for the older suite."""
    cases = [
        (T(False, 1200, 1, B.SUITE), T(False, 660, 2, B.CHECK)),  # the check has waited 10 min, the suite longer
        (T(True, 60, 1, B.SUITE), T(True, 30, 2, B.CHECK)),  # a critical suite queued before a critical check
    ]
    for suite, check in cases:
        queue = [suite, check]
        assert B.may_start(check, queue, _now(), NOW), "the draining check is ranked first and starts"
        assert not B.may_start(suite, queue, _now(), NOW), "the suite waits for it"


def test_some_waiter_always_starts_on_an_empty_box():
    """Liveness over every mix of two or three waiters: the box is never idle while something waits."""
    import itertools

    kinds = [(c, w, k) for c in (False, True) for w in (5, 700) for k in (B.SUITE, B.CHECK)]
    for n in (1, 2, 3):
        for combo in itertools.permutations(kinds, n):
            queue = [T(c, w + i, 10 + i, k) for i, (c, w, k) in enumerate(combo)]
            assert any(B.may_start(t, queue, _now(), NOW) for t in queue), combo


def test_ctrl_c_to_the_waiter_keeps_waiting_for_its_child():
    class Child:
        calls = 0

        def wait(self):
            Child.calls += 1
            if Child.calls == 1:
                raise KeyboardInterrupt  # the terminal sent Ctrl-C to the process group: the child gets it too
            return 3

        def send_signal(self, _signum):
            pass

    assert B._run_child(Child()) == 3 and Child.calls == 2


def test_an_earlier_waiter_that_fits_goes_first_and_an_unfit_one_never_holds_a_slot():
    first, second = T(False, 50, 1, B.SUITE), T(False, 10, 2, B.SUITE)
    assert not B.may_start(second, [first, second], _now(), NOW)
    assert B.may_start(first, [first, second], _now(), NOW)
    crit_suite, normal_check = T(True, 1, 3, B.SUITE), T(False, 50, 4, B.CHECK)
    assert not B.may_start(normal_check, [normal_check, crit_suite], _now(), NOW)  # critical first on an empty box
    assert B.may_start(crit_suite, [normal_check, crit_suite], _now(), NOW)


def test_nothing_starts_past_the_cap():
    me = T(True, 1, 1, B.SUITE)
    assert not B.may_start(me, [me], _now(suites=2), NOW)


def _state(tmp_path: Path) -> Path:
    for sub in ("queue", "claims"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_a_ticket_or_claim_whose_pid_was_reused_is_dropped(tmp_path):
    s = _state(tmp_path)
    t = B.Ticket(False, 1, 4242, B.SUITE, "Mon Nov  3 10:00:00 2026")
    (s / "queue" / t.name).write_text(json.dumps({"critical": False, "queued_ns": 1, "pid": 4242, "kind": "suite",
                                                  "started": t.started}))  # fmt: skip
    (s / "claims" / "4242.json").write_text(json.dumps({"pid": 4242, "kind": "suite", "started": t.started}))
    reused = [P(4242, 1, "python3 scripts/box_quiet.py wait -- x", started="Mon Nov  3 17:00:00 2026")]
    # Reading without pruning (status) removes nothing; the pruning read (under the waiter's lock) does.
    assert B._load_queue(s, reused, prune=False) == [] and B._load_claims(s, reused, prune=False) == {}
    assert len(list((s / "queue").iterdir())) == 1 and len(list((s / "claims").iterdir())) == 1
    assert B._load_queue(s, reused, prune=True) == [] and B._load_claims(s, reused, prune=True) == {}
    assert list((s / "queue").iterdir()) == [] and list((s / "claims").iterdir()) == []


def test_a_live_claim_survives_an_exec_in_its_child(tmp_path):
    s = _state(tmp_path)
    (s / "claims" / "77.json").write_text(json.dumps({"pid": 77, "kind": "check", "started": "t0"}))
    after_exec = [P(77, 1, "python3 scripts/prepr.py")]  # was `bash scripts/prepr.sh`; same pid, same start
    assert B._load_claims(s, after_exec, prune=True) == {77: B.CHECK}


def test_a_half_written_file_is_never_read(tmp_path):
    s = _state(tmp_path)
    (s / "queue" / "1-00000000000000000001-9.json.tmp").write_text('{"crit')
    assert B._load_queue(s, [], prune=True) == []
    assert (s / "queue" / "1-00000000000000000001-9.json.tmp").exists()


def test_the_lock_is_exclusive_and_freed_when_its_holder_dies(tmp_path):
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl,os,sys,time;fd=os.open(sys.argv[1],os.O_RDWR|os.O_CREAT);fcntl.flock(fd,fcntl.LOCK_EX);"
         "print('held',flush=True);time.sleep(30)", str(tmp_path / "lock")],
        stdout=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        assert holder.stdout.readline().strip() == "held"
        got = threading.Event()

        def take():
            with B.Lock(tmp_path):
                got.set()

        th = threading.Thread(target=take, daemon=True)
        th.start()
        assert not got.wait(0.5)  # blocked while the holder lives
        holder.kill()
        assert got.wait(5)  # the kernel freed it: no stale-lock takeover to race
    finally:
        holder.kill()
        holder.wait()


def test_wait_runs_the_command_and_returns_its_status(tmp_path):
    rc = B.main(["--state-dir", str(tmp_path), "--only-matching", "SystemExit(7)", "wait", "--kind", "check", "--",
                 sys.executable, "-c", "raise SystemExit(7)"])  # fmt: skip
    assert rc == 7
    assert list((tmp_path / "queue").glob("*.json")) == [] and list((tmp_path / "claims").glob("*.json")) == []


def test_a_child_killed_by_a_signal_returns_128_plus_the_signal(tmp_path):
    rc = B.main(["--state-dir", str(tmp_path), "--only-matching", "signal.SIGTERM", "wait", "--kind", "check", "--",
                 sys.executable, "-c", "import os,signal; os.kill(os.getpid(), signal.SIGTERM)"])  # fmt: skip
    assert rc == 128 + 15


def test_two_waiters_for_one_slot_never_start_together(tmp_path):
    """The race the hand-written idle waiters had: both see a free slot in the same poll. The lock + claim serialise it.

    Two real waiter processes; each child marks itself present FIRST, then looks for another mark, so two children
    running at once always see each other (a deletion probe of the claim write fails this every time)."""
    marks = tmp_path / "marks"
    marks.mkdir()
    prog = (
        "import os,sys,time,pathlib;d=pathlib.Path(sys.argv[1]);(d/str(os.getpid())).write_text('x');time.sleep(0.8);"
        "n=len(list(d.iterdir()))-1;(d/str(os.getpid())).unlink();sys.exit(n)"
    )
    cmd = [sys.executable, str(SCRIPT), "--state-dir", str(tmp_path / "s"), "--only-matching", str(marks), "wait",
           "--kind", "check", "--poll-s", "0.05", "--", sys.executable, "-c", prog, str(marks)]  # fmt: skip
    waiters = [subprocess.Popen(cmd, stderr=subprocess.DEVNULL) for _ in range(2)]
    assert [w.wait(timeout=30) for w in waiters] == [0, 0]  # neither child ever saw the other running beside it


def test_wait_refuses_a_command_it_cannot_classify(tmp_path):
    with pytest.raises(SystemExit) as e:
        B.main(["--state-dir", str(tmp_path), "wait", "--", "echo", "hi"])
    assert e.value.code == 2


def test_a_real_waiter_process_is_not_counted_but_its_child_is(tmp_path):
    """End to end on the real ps: the waiter's argv names a check, yet only its claimed child counts."""
    waiter = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--state-dir", str(tmp_path), "--only-matching", "box-quiet-e2e", "wait", "--kind",
         "check", "--", sys.executable, "-c", "import time; time.sleep(4)", "box-quiet-e2e"],
        stderr=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        line = waiter.stderr.readline()
        assert "started check pid" in line, line
        child_pid = int(line.rsplit(" ", 1)[1])
        procs = B.read_ps()
        rows = {p.pid: p for p in procs}
        assert B.classify(rows[waiter.pid].argv) is None
        claims = B._load_claims(tmp_path, procs, prune=False)
        assert claims == {child_pid: B.CHECK}
    finally:
        waiter.terminate()  # forwarded to the child; the waiter then exits with its status
        assert waiter.wait(timeout=10) == 128 + 15
    assert list((tmp_path / "claims").glob("*.json")) == []


def test_status_prints_and_removes_nothing(tmp_path, capsys):
    s = _state(tmp_path)
    (s / "claims" / "1.json").write_text("{not json")
    assert B.main(["--state-dir", str(s), "--only-matching", "nothing-here", "status"]) == 0
    assert "suites running" in capsys.readouterr().out
    assert (s / "claims" / "1.json").exists()


def test_cli_status_runs(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT), "--state-dir", str(tmp_path), "status"],
                         capture_output=True, text=True, timeout=30)  # fmt: skip
    assert out.returncode == 0 and "a suite fits now" in out.stdout


def test_run_mypy_public_is_counted_after_it_execs_mypy():
    # The wrapper execs mypy, so ps shows only mypy's argv: classify the exact line the script runs.
    script = (Path(__file__).resolve().parents[2] / "scripts" / "run_mypy_public.sh").read_text()
    line = next(x for x in script.splitlines() if x.startswith("exec mypy "))
    argv = ("/x/.venv/bin/python3", "/x/.venv/bin/mypy", "src/maxim/api.py", *line.split()[3:])
    assert B.classify(argv) == B.CHECK


def test_only_matching_is_refused_on_the_shared_state_dir():
    with pytest.raises(SystemExit):
        B.main(["--only-matching", "x", "status"])
