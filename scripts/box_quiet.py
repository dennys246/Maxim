#!/usr/bin/env python3
"""One shared waiter for the development box's test slots: start a full suite or a prepr/mypy run when it fits.

    python3 scripts/box_quiet.py status
    python3 scripts/box_quiet.py wait [--critical] [--kind suite|check] -- <command...>

The box rule (docs/agents/ci-testing.md §4, owner 2026-10-27): at most TWO full suites at once, and no prepr or mypy
beside any other counted suite or check. So a SUITE fits when fewer than two suites and no check are running, and a
CHECK (prepr, mypy) fits only when nothing counted is running. ``wait`` blocks until its command fits, starts it, and
exits with its status (128+N if a signal N killed it). The box is the only machine counted; the rig's own cap is
not.

Priority (owner 2026-11-03): use the box EFFICIENTLY for the next release. A waiter starts the moment its command
fits; nothing running is ever stopped. ``--critical`` marks work the next release waits on. Two owner rulings bound
"never idle a slot":
- while a critical CHECK waits, no new suite starts: the box drains for it (a new suite would add a whole ~40 min
  run to the critical check's wait, and a prepr takes minutes);
- a check that has waited ``DRAIN_AFTER_S`` (10 min) also drains the box, so prepr is never starved by steady suite
  traffic. A non-critical draining check holds back non-critical suites only; a critical suite still goes.
Both rulings and "critical first" are ONE rank, so they can never block each other (two separate rules once hung
an empty box: an older suite waited for a draining check that waited for the older suite): critical checks, then
critical suites, then non-critical draining checks, then the rest, each in queue order. A waiter waits for every
higher-ranked waiter that fits, and a suite also for every higher-ranked draining check; the top-ranked waiter
therefore starts the moment it fits, and no slot is idled except to drain. Among the rest, whichever fits goes.
The same drain covers a heavier suite (#1645): a ``--shards N`` suite ranked ahead of a lighter suite, and critical
or waiting ``DRAIN_AFTER_S``, holds the lighter one back until the box has room for it (this is a reading of the
owner's rulings for weights, not a ruling of its own).

Why a script of its own: the state is shared by every session and worktree on the box, and today's callers launch
``suite_at_commit.sh`` and ``prepr.sh`` directly, so neither script can see the other sessions' waiters. Backlog M85
moves the call inside them. Four hand-written ``pgrep -f`` waiters matched their own command line or each other's
(2026-11-02/03, one mutual deadlock of about 4 h), and several "wait for an idle box" waiters could see idle in the
same poll and all start.

Inputs and how each could lie:

- ``ps`` argv: a waiter's own argv CONTAINS the command it will run (``... wait -- bash scripts/suite_at_commit.sh``).
  Only a process's own program classifies it (argv[0], or the script/module after one interpreter, ``env NAME=value``
  words and interpreter flags), never a later word. Interpreter names are case-folded: macOS framework Pythons show
  as ``.../MacOS/Python``. ``prepr.sh`` execs ``python3 .../prepr.py``, so a running prepr is matched by
  ``prepr.py``.
- One run, many processes: a suite is ``suite_at_commit.sh`` plus its pytest child, and prepr runs mypy. A classified
  process with a classified (or claimed) ancestor is counted once, under the top-most.
- A sharded suite (#1645): ``scripts/run_fast_suite.py --shards N`` runs N pytest processes, so it WEIGHS N against
  ``MAX_SUITES``, not 1 (its top-most process) and not N + 1 (the runner plus its children). The weight is the
  largest ``--shards N`` in the counted process's own argv or any descendant's (``suite_at_commit.sh --shards N``
  passes it down), else 1; a queued waiter's weight is read from the command it will run. A suite heavier than the
  cap starts only on a box with nothing else counted.
- What a full suite is: pytest whose positional arguments include a path whose last part is ``tests`` (or ``.``), or
  none at all (``testpaths``). Values of pytest's value-taking flags (``-k tests``) are not positional, and since
  ``ps`` drops quoting, a bare word (``slow`` from ``-m "not slow"``) is not a path either; a path must contain ``/``
  or ``::``, end in ``.py``, or be ``tests`` or ``.``. (``pytest src`` therefore counts as a suite: it errs toward
  waiting.) A narrow
  ``pytest tests/unit/x.py`` is not counted, and neither is a standalone mypy (an editor's), unless it carries
  ``run_mypy_public.sh``'s flags (that script execs mypy, so its own name never stays in ``ps``) or runs under a
  counted run.
- Two waiters deciding at once: the decision and the launch happen under an exclusive ``flock`` (the kernel drops it
  when its holder dies, so there is no stale-lock takeover to race), and the launched child is written as a CLAIM
  before the lock is released, so the next waiter counts it before ``ps`` shows its argv.
- Pid reuse: tickets and claims record their process's START TIME from ``ps``; one whose pid is gone, or now has a
  different start time, is dropped. An exec in the child (``prepr.sh`` -> ``prepr.py``) keeps the pid and start time,
  so the claim survives it.
- Half-written state: files are written to a ``.tmp`` name and ``os.replace``d, and readers see only ``*.json``.
  Tickets are written and stale files removed under the lock; ``status`` takes the lock and never removes anything.
- Signals: SIGTERM/SIGHUP to the waiter are forwarded to its child, and the waiter waits for the child before it
  removes the claim, so a stopped waiter neither orphans a running suite nor uncounts one still shutting down.
- Waiters that do not use this script are still counted while running (by argv); they just do not queue.
- Truncated argv: Linux procps cuts ``args`` to a terminal width when it finds one (a waiter's inherited stderr), which
  hid a claimed child's argv and let two checks start together on Linux; ``ps -ww`` (procps and BSD) never truncates.
- The waiter's own process tree: it and its ancestors are never counted (``running(own_pid=)``), or a waiter run
  from inside a full suite (its tests in CI) counts that suite and waits for itself forever.

State lives in ``~/.cache/maxim-box-quiet`` (shared by every worktree and session of this user), or ``--state-dir``.
These files are not persisted data (throwaway coordination state; CC3 and ``_format_version`` do not apply).
Standard library only.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MAX_SUITES = 2
DRAIN_AFTER_S = 600.0
SUITE, CHECK = "suite", "check"
DEFAULT_STATE = Path.home() / ".cache" / "maxim-box-quiet"
SELF = "box_quiet.py"
_INTERPRETER = re.compile(r"^(python[0-9.]*|bash|sh|zsh|dash)$")
_PY_FLAGS_WITH_VALUE = {"-X", "-W", "-Q"}
_SHELL_FLAGS_WITH_VALUE = {"-o", "-O"}
_ENV_FLAGS_WITH_VALUE = {"-u", "-C"}
_PYTEST_FLAGS_WITH_VALUE = {
    "-k", "-m", "-p", "-c", "-o", "-W", "-n", "--rootdir", "--confcutdir", "--basetemp", "--ignore", "--ignore-glob",
    "--deselect", "--junitxml", "--cov", "--cov-report", "--cov-config", "--maxfail", "--durations", "--timeout",
    "--log-level", "--log-file", "--tb", "-r", "--capture", "--import-mode", "--override-ini",
}  # fmt: skip


@dataclass(frozen=True)
class Proc:
    """One ``ps`` row; not persisted (CC3 n/a)."""

    pid: int
    ppid: int
    started: str
    argv: tuple[str, ...]


# The flags scripts/run_mypy_public.sh's ``exec mypy`` passes (test_box_quiet reads them from the script).
_MYPY_PUBLIC_SIGNATURE = frozenset({"--follow-imports=silent", "--warn-unused-ignores"})


def _program(argv: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """The program a process runs and its own arguments: past ``env NAME=value`` and one interpreter with its flags."""
    rest = list(argv)
    if rest and os.path.basename(rest[0]) == "env":
        rest = rest[1:]
        while rest and ("=" in rest[0] or rest[0].startswith("-")):
            flag = rest.pop(0)
            if flag in _ENV_FLAGS_WITH_VALUE and rest:
                rest.pop(0)
    if not rest:
        return "", ()
    head = os.path.basename(rest[0])
    if not _INTERPRETER.match(head.lower()):
        return head, tuple(rest[1:])
    rest = rest[1:]
    with_value = _PY_FLAGS_WITH_VALUE if head.lower().startswith("python") else _SHELL_FLAGS_WITH_VALUE
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag == "-m" and rest:
            return rest[0], tuple(rest[1:])
        if flag == "-c":
            return "", ()  # inline code: nothing we classify
        if flag in with_value and rest:
            rest.pop(0)
    if rest:
        return os.path.basename(rest[0]), tuple(rest[1:])
    return head, ()


def _pytest_positionals(args: tuple[str, ...]) -> list[str]:
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a.startswith("-"):
            skip = a in _PYTEST_FLAGS_WITH_VALUE
        else:
            out.append(a)
    return out


def _looks_like_path(word: str) -> bool:
    """``ps`` drops quoting, so ``-m "not slow"`` arrives as ``-m not slow``: a bare word is an expression's, not a path."""
    return "/" in word or "::" in word or word.endswith(".py") or word in ("tests", ".")


def classify(argv: tuple[str, ...]) -> str | None:
    """SUITE, CHECK or None for one process, from its own program only (never a later word of a waiter's argv)."""
    prog, args = _program(argv)
    if prog == "suite_at_commit.sh":
        return SUITE
    if prog == "run_fast_suite.py":
        return None if "--check-map" in args else SUITE  # --check-map is a lint (the lint job's), not a suite
    if prog in ("pytest", "py.test"):
        pos = [a for a in _pytest_positionals(args) if _looks_like_path(a)]
        if not pos or any(os.path.basename(os.path.normpath(a)) in ("tests", ".") for a in pos):
            return SUITE
        return None
    if prog in ("prepr.py", "prepr.sh", "run_mypy_public.sh"):
        return CHECK
    if prog == "mypy" and _MYPY_PUBLIC_SIGNATURE <= set(args):
        return CHECK  # run_mypy_public.sh execs mypy, so the wrapper's own name is gone within milliseconds
    return None


def weight(argv: tuple[str, ...]) -> int:
    """How many pytest processes one process's own run is: ``--shards N`` for the sharded runner or the suite script
    that passes it down, else 1. A malformed value weighs 1 (the runner and the script refuse it before any pytest)."""
    prog, args = _program(argv)
    if prog not in ("suite_at_commit.sh", "run_fast_suite.py"):
        return 1
    for i, a in enumerate(args):
        if a == "--shards" and i + 1 < len(args):
            value = args[i + 1]
        elif a.startswith("--shards="):
            value = a.split("=", 1)[1]
        else:
            continue
        return int(value) if value.isdigit() and int(value) >= 1 else 1
    return 1


def read_ps() -> list[Proc]:
    """Every process with its start time. ``lstart`` is five words (``Mon Nov  3 16:00:00 2026``)."""
    out = subprocess.run(
        ["ps", "-ww", "-axo", "pid=,ppid=,lstart=,args="],  # -ww: Linux procps cuts args to a width otherwise
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    ).stdout
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        procs.append(Proc(int(parts[0]), int(parts[1]), " ".join(parts[2:7]), tuple(parts[7].split())))
    return procs


def scoped(procs: list[Proc], only: str | None) -> list[Proc]:
    """Tests run inside a full suite on a shared box: ``--only-matching`` counts only their own processes."""
    if only is None:
        return procs
    me = os.getpid()
    return [p for p in procs if p.pid == me or only in " ".join(p.argv)]


def running(procs: list[Proc], claims: dict[int, str], *, own_pid: int) -> dict[str, dict[int, int]]:
    """The runs in progress by kind, each counted once (``{pid: weight}``): claims first, then argv-classified top-most
    processes. A run's weight is the largest :func:`weight` in its own process or any descendant (a sharded suite).

    ``own_pid`` (the caller, required so no call site can forget it) and its ancestors are never counted: a waiter
    started inside a run (a test of this script inside a full suite) cannot wait for that run to end."""
    by_pid = {p.pid: p for p in procs}
    children: dict[int, list[int]] = {}
    for p in procs:
        if p.ppid != p.pid:
            children.setdefault(p.ppid, []).append(p.pid)

    def run_weight(pid: int) -> int:
        best, todo, seen = 1, [pid], set()
        while todo:
            cur = todo.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur in by_pid:
                best = max(best, weight(by_pid[cur].argv))
            todo += children.get(cur, [])
        return best

    def ancestors(pid: int):
        seen = set()
        cur = by_pid.get(pid)
        while cur is not None and cur.ppid not in seen and cur.ppid != cur.pid:
            seen.add(cur.ppid)
            yield cur.ppid
            cur = by_pid.get(cur.ppid)

    mine = {own_pid, *ancestors(own_pid)}
    kinds = {p.pid: (None if p.pid in mine else classify(p.argv)) for p in procs}
    claims = {pid: kind for pid, kind in claims.items() if pid not in mine}  # a waiter nested in a claimed run
    out: dict[str, dict[int, int]] = {SUITE: {}, CHECK: {}}
    for pid, kind in claims.items():
        out[kind][pid] = run_weight(pid)
    for p in procs:
        kind = kinds[p.pid]
        if kind is None or p.pid in claims:
            continue
        if any(a in claims or kinds.get(a) is not None for a in ancestors(p.pid)):
            continue
        out[kind][p.pid] = run_weight(p.pid)
    return out


def suite_load(now: dict[str, dict[int, int]]) -> int:
    """The running suites' total weight (a ``--shards N`` suite counts N)."""
    return sum(now[SUITE].values())


def fits(kind: str, now: dict[str, dict[int, int]], need: int = 1) -> bool:
    """The box rule in weights: a suite of weight ``need`` fits while the load stays within MAX_SUITES (one heavier
    than the cap only on an otherwise empty box) and no check runs; a check fits only on an empty box."""
    load, checks = suite_load(now), len(now[CHECK])
    if kind == SUITE:
        return checks == 0 and (load + need <= MAX_SUITES or load == 0)
    return load == 0 and checks == 0


@dataclass(frozen=True)
class Ticket:
    """A queued waiter; a throwaway coordination file, not persisted data (CC3 n/a)."""

    critical: bool
    queued_ns: int
    pid: int
    kind: str
    started: str = ""
    weight: int = 1  # the command's own weight (a ``--shards N`` suite); a ticket written before #1645 reads 1

    @property
    def order(self) -> tuple[int, int, int]:
        return (0 if self.critical else 1, self.queued_ns, self.pid)

    @property
    def name(self) -> str:
        return f"{self.order[0]}-{self.queued_ns:020d}-{self.pid}.json"


def _draining(t: Ticket, now_ns: int) -> bool:
    return t.kind == CHECK and (t.critical or now_ns - t.queued_ns >= DRAIN_AFTER_S * 1e9)


def _drains_lighter_suites(t: Ticket, me: Ticket, now_ns: int) -> bool:
    """#1645 review: a heavier suite waiting for room (a ``--shards N`` suite needs an emptier box) drains it of new
    LIGHTER suites once it is critical or has waited ``DRAIN_AFTER_S``, or a stream of weight-1 suites starves it."""
    return (
        me.kind == SUITE
        and t.kind == SUITE
        and t.weight > me.weight
        and (t.critical or now_ns - t.queued_ns >= DRAIN_AFTER_S * 1e9)
    )


def _rank(t: Ticket, now_ns: int) -> tuple[int, int, int]:
    """The single precedence order (module docstring): critical checks, critical suites, draining checks, the rest."""
    draining = _draining(t, now_ns)
    return (0 if t.critical and draining else 1 if t.critical else 2 if draining else 3, t.queued_ns, t.pid)


def may_start(me: Ticket, queue: list[Ticket], now: dict[str, dict[int, int]], now_ns: int | None = None) -> bool:
    """Whether ``me`` starts now: it fits, no waiter ordered ahead of it fits and has precedence, and starting would
    not keep a draining check waiting (the owner's two rulings in the module docstring)."""
    if not fits(me.kind, now, me.weight):
        return False
    now_ns = time.time_ns() if now_ns is None else now_ns
    mine = _rank(me, now_ns)
    for t in queue:
        if t.pid == me.pid or t is me or _rank(t, now_ns) >= mine:
            continue
        if fits(t.kind, now, t.weight):
            return False  # it goes first; we re-check after it has claimed its slot
        if me.kind == SUITE and _draining(t, now_ns):
            return False  # the box drains for it
        if _drains_lighter_suites(t, me, now_ns):
            return False  # the box drains for a heavier suite ranked ahead
    return True


class Lock:
    """An exclusive ``flock`` on ``<state>/lock``: held for one decide-and-launch step; the kernel frees it on death."""

    def __init__(self, state: Path):
        self.path = state / "lock"
        self.fd = -1

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def _write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _started(procs: list[Proc]) -> dict[int, str]:
    return {p.pid: p.started for p in procs}


def _load_queue(state: Path, procs: list[Proc], prune: bool) -> list[Ticket]:
    """Live tickets: the pid still runs with the recorded start time. ``prune`` (under the lock) removes the rest."""
    started = _started(procs)
    out = []
    for f in sorted((state / "queue").glob("*.json")):
        try:
            d = json.loads(f.read_text())
            t = Ticket(bool(d["critical"]), int(d["queued_ns"]), int(d["pid"]), str(d["kind"]), str(d["started"]),
                       max(1, int(d.get("weight", 1))))  # fmt: skip
        except (OSError, ValueError, KeyError) as e:
            print(f"box_quiet: unreadable ticket {f.name}: {e}", file=sys.stderr)
            if prune:
                f.unlink(missing_ok=True)
            continue
        if started.get(t.pid) == t.started:
            out.append(t)
        elif prune:
            f.unlink(missing_ok=True)
    return out


def _load_claims(state: Path, procs: list[Proc], prune: bool) -> dict[int, str]:
    started = _started(procs)
    out = {}
    for f in (state / "claims").glob("*.json"):
        try:
            d = json.loads(f.read_text())
            pid, kind, when = int(d["pid"]), str(d["kind"]), str(d["started"])
        except (OSError, ValueError, KeyError) as e:
            print(f"box_quiet: unreadable claim {f.name}: {e}", file=sys.stderr)
            if prune:
                f.unlink(missing_ok=True)
            continue
        if started.get(pid) == when:
            out[pid] = kind
        elif prune:
            f.unlink(missing_ok=True)
    return out


def infer_kind(cmd: list[str]) -> str | None:
    return classify(tuple(cmd))


def cmd_status(state: Path, only: str | None = None) -> int:
    for sub in ("queue", "claims"):
        (state / sub).mkdir(parents=True, exist_ok=True)
    with Lock(state):
        procs = scoped(read_ps(), only)
        now = running(procs, _load_claims(state, procs, prune=False), own_pid=os.getpid())
        queue = _load_queue(state, procs, prune=False)
    print(
        f"suites running: load {suite_load(now)}/{MAX_SUITES} {now[SUITE]} (pid: weight)   "
        f"checks running: {len(now[CHECK])} {list(now[CHECK])}"
    )
    now_ns = time.time_ns()
    for t in sorted(queue, key=lambda t: t.order):
        waited = (now_ns - t.queued_ns) / 1e9
        flags = ("  CRITICAL" if t.critical else "") + ("  DRAINING" if _draining(t, now_ns) else "")
        print(f"  queued: {t.kind:5} pid {t.pid}  weight {t.weight}  waited {waited:.0f}s{flags}")
    print(f"a suite fits now: {fits(SUITE, now)}   a check fits now: {fits(CHECK, now)}")
    return 0


def _run_child(child: subprocess.Popen) -> int:
    """Wait for the child, forwarding SIGTERM/SIGHUP to it; a Ctrl-C (which the child also gets) keeps waiting."""

    def forward(signum, _frame):
        child.send_signal(signum)

    old = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        while True:
            try:
                rc = child.wait()
                break
            except KeyboardInterrupt:
                continue
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    return 128 - rc if rc < 0 else rc


def cmd_wait(state: Path, kind: str, critical: bool, cmd: list[str], poll_s: float, only: str | None = None) -> int:
    for sub in ("queue", "claims"):
        (state / sub).mkdir(parents=True, exist_ok=True)
    child = None
    with Lock(state):
        procs = read_ps()
        me = Ticket(critical, time.time_ns(), os.getpid(), kind, _started(procs).get(os.getpid(), ""),
                    weight(tuple(cmd)))  # fmt: skip
        ticket = state / "queue" / me.name
        _write_atomic(ticket, {"critical": critical, "queued_ns": me.queued_ns, "pid": me.pid, "kind": kind,
                               "started": me.started, "weight": me.weight})  # fmt: skip
    announced = False
    try:
        while child is None:
            with Lock(state):
                procs = scoped(read_ps(), only)
                now = running(procs, _load_claims(state, procs, prune=True), own_pid=os.getpid())
                if may_start(me, _load_queue(state, procs, prune=True), now):
                    child = subprocess.Popen(cmd)
                    when = ""
                    for _ in range(50):  # ps lists the new pid within milliseconds; bounded, never a hang
                        when = _started(read_ps()).get(child.pid, "")
                        if when:
                            break
                        time.sleep(0.02)
                    claim = state / "claims" / f"{child.pid}.json"
                    _write_atomic(claim, {"pid": child.pid, "kind": kind, "started": when})
                    ticket.unlink(missing_ok=True)
                    break
            if not announced:
                print(f"box_quiet: waiting to start a {kind}{' (critical)' if critical else ''}: {' '.join(cmd)}",
                      file=sys.stderr, flush=True)  # fmt: skip
                announced = True
            time.sleep(poll_s)
    finally:
        ticket.unlink(missing_ok=True)
    print(f"box_quiet: started {kind} pid {child.pid}", file=sys.stderr, flush=True)
    try:
        return _run_child(child)
    finally:
        claim.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    sub = ap.add_subparsers(dest="verb", required=True)
    sub.add_parser("status", help="what is running and queued, and what fits now")
    w = sub.add_parser("wait", help="start <command> when it fits; exit with its status")
    w.add_argument("--critical", action="store_true", help="work the next release waits on")
    w.add_argument("--kind", choices=(SUITE, CHECK), help="default: inferred from the command")
    w.add_argument("--poll-s", type=float, default=20.0)
    w.add_argument("cmd", nargs=argparse.REMAINDER)
    ap.add_argument("--only-matching", default=None, help=argparse.SUPPRESS)  # tests: count only their processes
    args = ap.parse_args(argv)
    if args.only_matching is not None and args.state_dir == DEFAULT_STATE:
        ap.error("--only-matching blinds the waiter to the box; it is for tests with their own --state-dir")
    if args.verb == "status":
        return cmd_status(args.state_dir, args.only_matching)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("wait needs a command after --")
    kind = args.kind or infer_kind(cmd)
    if kind is None:
        ap.error(f"cannot tell whether {cmd[0]!r} is a suite or a check: pass --kind")
    return cmd_wait(args.state_dir, kind, args.critical, cmd, args.poll_s, args.only_matching)


if __name__ == "__main__":
    sys.exit(main())
