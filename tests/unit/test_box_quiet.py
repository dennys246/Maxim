"""scripts/box_quiet.py: the shared box-slot waiter (two suites at most, no prepr/mypy beside anything).

Each "how could this input lie" line in the module docstring has a test here: a waiter never counts itself, one run
is counted once, a narrow pytest is not a suite, the release-priority rule only breaks ties (it never idles a slot),
and the lock/claim pair stops two waiters starting into one slot.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import pytest

from scripts import box_quiet as B


def P(pid: int, ppid: int, cmd: str) -> B.Proc:
    return B.Proc(pid, ppid, tuple(cmd.split()))


@pytest.mark.parametrize(
    "cmd,kind",
    [
        ("bash scripts/suite_at_commit.sh 0df45262", B.SUITE),
        ("/bin/bash /Users/x/Maxim/scripts/suite_at_commit.sh --no-cov abc", B.SUITE),
        ("python -m pytest tests/ -q -m not slow --ignore=tests/integration/test_memory_hub.py", B.SUITE),
        ("/usr/bin/python3.12 -m pytest tests -q", B.SUITE),
        ("python3 /Users/x/Maxim/scripts/prepr.py --no-pr", B.CHECK),
        ("bash scripts/prepr.sh", B.CHECK),
        ("bash scripts/run_mypy_public.sh", B.CHECK),
        # Not a full suite: a narrow pytest, an editor's mypy, unrelated programs.
        ("python -m pytest tests/unit/test_box_quiet.py -q", None),
        ("/x/.venv/bin/mypy src/maxim/foo.py", None),
        ("grep suite_at_commit.sh log.txt", None),
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


def T(critical: bool, at: int, pid: int, kind: str) -> B.Ticket:
    return B.Ticket(critical, at, pid, kind)


def _now(suites=0, checks=0):
    return {B.SUITE: list(range(suites)), B.CHECK: list(range(checks))}


def test_a_fitting_start_goes_when_nothing_critical_waits():
    me = T(False, 5, 1, B.SUITE)
    assert B.may_start(
        me, [me, T(False, 1, 2, B.CHECK)], _now(suites=1)
    )  # efficiency: an earlier blocked check does not idle the slot


def test_a_non_critical_suite_yields_to_a_waiting_critical_check():
    me = T(False, 5, 1, B.SUITE)
    assert not B.may_start(me, [me, T(True, 9, 2, B.CHECK)], _now(suites=1))


def test_a_non_critical_check_never_idles_the_box_for_a_critical_suite_that_cannot_fit():
    # Box empty means the critical suite fits too and goes first; with a check running neither fits.
    me = T(False, 1, 1, B.CHECK)
    crit = T(True, 9, 2, B.SUITE)
    assert not B.may_start(me, [me, crit], _now())
    assert B.may_start(crit, [me, crit], _now())


def test_a_critical_waiter_that_fits_goes_first_and_a_later_one_is_not_blocked_by_an_unfit_critical():
    crit_check = T(True, 1, 1, B.CHECK)
    crit_suite = T(True, 2, 2, B.SUITE)
    q = [crit_check, crit_suite]
    # One suite running: the critical check cannot fit; the critical suite can and goes.
    assert not B.may_start(crit_check, q, _now(suites=1))
    assert B.may_start(crit_suite, q, _now(suites=1))


def test_nothing_starts_past_the_cap():
    me = T(True, 1, 1, B.SUITE)
    assert not B.may_start(me, [me], _now(suites=2))


def test_stale_tickets_and_claims_are_dropped(tmp_path):
    (tmp_path / "queue").mkdir()
    (tmp_path / "claims").mkdir()
    dead = 2**22 + 12345  # beyond any real pid on the CI and dev boxes
    t = T(False, 1, dead, B.SUITE)
    (tmp_path / "queue" / t.name).write_text(
        json.dumps({"critical": False, "queued_ns": 1, "pid": dead, "kind": "suite"})
    )
    (tmp_path / "claims" / f"{dead}.json").write_text(json.dumps({"pid": dead, "kind": "suite", "program": "bash"}))
    # A claim whose pid is alive but now runs a different program is pid reuse: dropped too.
    me = os.getpid()
    (tmp_path / "claims" / f"{me}.json").write_text(json.dumps({"pid": me, "kind": "suite", "program": "not-this"}))
    procs = [P(me, 1, f"{sys.executable} -m pytest")]
    assert B._load_queue(tmp_path) == []
    assert B._load_claims(tmp_path, procs) == {}
    assert list((tmp_path / "queue").iterdir()) == [] and list((tmp_path / "claims").iterdir()) == []


def test_a_lock_held_by_a_dead_pid_is_taken_over(tmp_path):
    lock = tmp_path / "lock"
    lock.mkdir()
    (lock / "pid").write_text(str(2**22 + 54321))
    with B.Lock(tmp_path):
        assert (lock / "pid").read_text() == str(os.getpid())
    assert not lock.exists()


def test_wait_runs_the_command_and_returns_its_status(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "read_ps", lambda: [])
    rc = B.main(
        ["--state-dir", str(tmp_path), "wait", "--kind", "check", "--", sys.executable, "-c", "raise SystemExit(7)"]
    )
    assert rc == 7
    assert list((tmp_path / "queue").iterdir()) == [] and list((tmp_path / "claims").iterdir()) == []


def test_two_waiters_for_one_slot_never_start_together(tmp_path, monkeypatch):
    """The race the hand-written idle waiters had: both see a free slot in the same poll. The lock + claim serialise it."""
    monkeypatch.setattr(B, "read_ps", lambda: [])  # no foreign runs; only the waiters' own claims count
    marks = tmp_path / "marks"
    marks.mkdir()
    prog = (
        "import os,sys,time,pathlib;d=pathlib.Path(sys.argv[1]);"
        "n=len(list(d.iterdir()));(d/str(os.getpid())).write_text(str(n));time.sleep(0.6);(d/str(os.getpid())).unlink()"
    )
    seen: list[int] = []

    def run():
        seen.append(
            B.main(
                [
                    "--state-dir",
                    str(tmp_path / "s"),
                    "wait",
                    "--kind",
                    "check",
                    "--poll-s",
                    "0.05",
                    "--",
                    sys.executable,
                    "-c",
                    prog + ";sys.exit(int(n))",
                    str(marks),
                ]
            )
        )

    # _load_claims drops a claim unless ps shows its pid running the claimed program: let ps report the claims' own
    # children as alive so the second waiter sees the first's slot as taken.
    real_load = B._load_claims
    monkeypatch.setattr(
        B,
        "_load_claims",
        lambda state, procs: {
            int(f.stem): json.loads(f.read_text())["kind"]
            for f in (state / "claims").glob("*.json")
            if B._alive(int(f.stem))
        }
        if (state / "claims").exists()
        else real_load(state, procs),
    )
    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert seen == [0, 0]  # each child saw no other child running beside it


def test_wait_refuses_a_command_it_cannot_classify(tmp_path):
    with pytest.raises(SystemExit) as e:
        B.main(["--state-dir", str(tmp_path), "wait", "--", "echo", "hi"])
    assert e.value.code == 2


def test_the_waiter_does_not_count_itself_in_a_real_ps(tmp_path):
    """End to end on the real ps: a box_quiet waiter process, its argv naming a suite, adds nothing to the count."""
    import subprocess

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(5)",
            "box_quiet.py",
            "wait",
            "--",
            "bash",
            "scripts/suite_at_commit.sh",
        ]
    )
    try:
        time.sleep(0.2)
        procs = B.read_ps()
        assert holder.pid in {p.pid for p in procs}
        assert B.classify(next(p for p in procs if p.pid == holder.pid).argv) is None
    finally:
        holder.kill()
        holder.wait()
