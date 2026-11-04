#!/usr/bin/env python3
"""One shared waiter for the development box's test slots: start a full suite or a prepr/mypy run when it fits.

    python3 scripts/box_quiet.py status
    python3 scripts/box_quiet.py wait [--critical] [--kind suite|check] -- <command...>

The box rule (docs/agents/ci-testing.md §4, owner 2026-10-27): at most TWO full suites at once, and no prepr or mypy
beside a suite (or beside each other). So a SUITE fits when fewer than two suites and no check are running, and a
CHECK (prepr, mypy) fits only when nothing else is. ``wait`` blocks until its command fits, starts it, and exits with
its status.

Release priority (owner 2026-11-03): the point is to use the box EFFICIENTLY for the next release, not to reserve it.
A waiter starts the moment its command fits; nothing runs is ever stopped and no slot is held idle. ``--critical``
(work the next release waits on) only breaks ties: a non-critical waiter does not start when doing so would delay a
critical waiter that is already queued. Concretely that is one case, a non-critical suite while a critical check
waits for the suites to drain; every other fitting start cannot delay a critical waiter and goes ahead.

Why this exists: four sessions hand-wrote ``pgrep -f`` waiters whose pattern matched their own command line or other
sessions' waiters (2026-11-02 and -03: a 4 h mutual deadlock), and three "wait for an idle box" waiters could all see
idle in the same poll and start together.

Inputs and how each could lie:

- ``ps`` argv: a waiter's own argv CONTAINS the command it will run (``... wait -- bash scripts/suite_at_commit.sh``).
  Only a process's own program (argv[0], or argv[1] after an interpreter) classifies it, never a later word.
  ``prepr.sh`` execs ``python3 .../prepr.py``, so the running prepr is matched by ``prepr.py``; a pattern on
  ``bash scripts/prepr.sh`` never saw a running prepr at all.
- One run, many processes: a suite is ``suite_at_commit.sh`` plus its pytest child, and prepr runs mypy. A classified
  process with a classified (or claimed) ancestor is counted once, under the top-most.
- A narrow ``pytest tests/unit/test_x.py`` is not a full suite and is not counted; a full suite is pytest with ``tests``
  or ``tests/`` as a positional argument. A standalone mypy (an editor's) is not counted unless it is
  ``run_mypy_public.sh`` or runs under a counted run.
- Two waiters deciding at once: the decision and the launch happen under one lock (an ``mkdir`` directory, stale when
  its holder pid is dead), and the launched child is written as a CLAIM before the lock is released, so the next
  waiter counts it even before ``ps`` shows its argv.
- Dead waiters and finished runs: queue tickets and claims carry a pid and are dropped once that pid is gone (a claim
  also when the pid now runs a different program).
- Waiters that do not use this script are still counted while running (by argv); they just do not queue.

State lives in ``~/.cache/maxim-box-quiet`` (shared by every worktree and session of this user), or ``--state-dir``.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MAX_SUITES = 2
SUITE, CHECK = "suite", "check"
DEFAULT_STATE = Path.home() / ".cache" / "maxim-box-quiet"
LOCK_STALE_S = 120.0
INTERPRETERS = ("python", "bash", "sh", "zsh", "env")


@dataclass(frozen=True)
class Proc:
    """One ``ps`` row. SHAPE-FROZEN at 1.0 (CC3): never persisted or sent; a script-local view of ps."""

    pid: int
    ppid: int
    argv: tuple[str, ...]


def _program(argv: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """The program a process runs and its own arguments, skipping one interpreter (``python3 -m pytest``, ``bash x.sh``)."""
    if not argv:
        return "", ()
    head = os.path.basename(argv[0])
    rest = argv[1:]
    if head.startswith(INTERPRETERS):
        # Skip interpreter flags (-u, -I, -c is not a program we classify).
        while rest and rest[0].startswith("-") and rest[0] != "-m":
            rest = rest[1:]
        if rest[:1] == ("-m",) and len(rest) >= 2:
            return rest[1], rest[2:]
        if rest:
            return os.path.basename(rest[0]), rest[1:]
        return head, ()
    return head, rest


def classify(argv: tuple[str, ...]) -> str | None:
    """SUITE, CHECK or None for one process, from its own program only (never a later word of a waiter's argv)."""
    prog, args = _program(argv)
    if prog == "suite_at_commit.sh":
        return SUITE
    if prog in ("pytest", "py.test"):
        positional = [a for a in args if not a.startswith("-")]
        return SUITE if any(a.rstrip("/") == "tests" for a in positional) else None
    if prog in ("prepr.py", "prepr.sh", "run_mypy_public.sh"):
        return CHECK
    return None


def read_ps() -> list[Proc]:
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,args="], capture_output=True, text=True, check=True).stdout
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        procs.append(Proc(int(parts[0]), int(parts[1]), tuple(parts[2].split())))
    return procs


def running(procs: list[Proc], claims: dict[int, str]) -> dict[str, list[int]]:
    """The runs in progress by kind, each counted once: claims first, then argv-classified top-most processes."""
    by_pid = {p.pid: p for p in procs}

    def ancestors(pid: int):
        seen = set()
        cur = by_pid.get(pid)
        while cur is not None and cur.ppid not in seen and cur.ppid != cur.pid:
            seen.add(cur.ppid)
            yield cur.ppid
            cur = by_pid.get(cur.ppid)

    kinds = {p.pid: classify(p.argv) for p in procs}
    out: dict[str, list[int]] = {SUITE: [], CHECK: []}
    for pid, kind in claims.items():
        out[kind].append(pid)
    for p in procs:
        kind = kinds[p.pid]
        if kind is None or p.pid in claims:
            continue
        if any(a in claims or kinds.get(a) is not None for a in ancestors(p.pid)):
            continue
        out[kind].append(p.pid)
    return out


def fits(kind: str, now: dict[str, list[int]]) -> bool:
    suites, checks = len(now[SUITE]), len(now[CHECK])
    if kind == SUITE:
        return suites < MAX_SUITES and checks == 0
    return suites == 0 and checks == 0


@dataclass(frozen=True)
class Ticket:
    """A queued waiter. SHAPE-FROZEN at 1.0 (CC3): a script-local file in the state dir, read only by this script."""

    critical: bool
    queued_ns: int
    pid: int
    kind: str

    @property
    def name(self) -> str:
        return f"{0 if self.critical else 1}-{self.queued_ns:020d}-{self.pid}.json"


def may_start(me: Ticket, queue: list[Ticket], now: dict[str, list[int]]) -> bool:
    """Whether ``me`` starts now: it fits, and starting cannot delay a critical waiter queued ahead of it.

    Efficiency first (owner 2026-11-03): any fitting start goes, except a non-critical SUITE while a critical CHECK
    waits (another suite would push the check's zero-suites point further out). A non-critical CHECK that fits means
    the box is empty, so a waiting critical suite would also fit and is ordered first.
    """
    if not fits(me.kind, now):
        return False
    ahead = [t for t in sorted(queue, key=lambda t: t.name) if t.name < me.name and t.critical]
    for t in ahead:
        if fits(t.kind, now):
            return False  # it goes first; we re-check after it has claimed its slot
        if not me.critical and me.kind == SUITE and t.kind == CHECK:
            return False
    return True


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Lock:
    """``mkdir`` lock with the holder's pid inside; taken over when the holder is dead or the lock is stale."""

    def __init__(self, state: Path):
        self.dir = state / "lock"

    def __enter__(self):
        while True:
            try:
                self.dir.mkdir()
                (self.dir / "pid").write_text(str(os.getpid()))
                return self
            except FileExistsError:
                try:
                    holder = int((self.dir / "pid").read_text())
                    age = time.time() - self.dir.stat().st_mtime
                except (OSError, ValueError):
                    holder, age = -1, LOCK_STALE_S + 1  # half-written lock: stale once old
                    try:
                        age = time.time() - self.dir.stat().st_mtime
                    except OSError:
                        continue
                if (holder > 0 and not _alive(holder)) or age > LOCK_STALE_S:
                    shutil.rmtree(self.dir, ignore_errors=True)
                    continue
                time.sleep(0.2)

    def __exit__(self, *exc):
        shutil.rmtree(self.dir, ignore_errors=True)


def _load_queue(state: Path) -> list[Ticket]:
    out = []
    for f in sorted((state / "queue").glob("*.json")):
        try:
            d = json.loads(f.read_text())
            t = Ticket(bool(d["critical"]), int(d["queued_ns"]), int(d["pid"]), str(d["kind"]))
        except (OSError, ValueError, KeyError) as e:
            print(f"box_quiet: dropping unreadable ticket {f.name}: {e}", file=sys.stderr)
            f.unlink(missing_ok=True)
            continue
        if _alive(t.pid):
            out.append(t)
        else:
            f.unlink(missing_ok=True)
    return out


def _load_claims(state: Path, procs: list[Proc]) -> dict[int, str]:
    progs = {p.pid: _program(p.argv)[0] for p in procs}
    out = {}
    for f in (state / "claims").glob("*.json"):
        try:
            d = json.loads(f.read_text())
            pid, kind, prog = int(d["pid"]), str(d["kind"]), str(d["program"])
        except (OSError, ValueError, KeyError) as e:
            print(f"box_quiet: dropping unreadable claim {f.name}: {e}", file=sys.stderr)
            f.unlink(missing_ok=True)
            continue
        if pid in progs and progs[pid] == prog:
            out[pid] = kind
        else:
            f.unlink(missing_ok=True)
    return out


def infer_kind(cmd: list[str]) -> str | None:
    return classify(tuple(cmd))


def cmd_status(state: Path) -> int:
    procs = read_ps()
    now = running(procs, _load_claims(state, procs))
    queue = _load_queue(state)
    print(
        f"suites running: {len(now[SUITE])}/{MAX_SUITES} {now[SUITE]}   checks running: {len(now[CHECK])} {now[CHECK]}"
    )
    for t in sorted(queue, key=lambda t: t.name):
        print(f"  queued: {t.kind:5} pid {t.pid}{'  CRITICAL' if t.critical else ''}")
    print(f"a suite fits now: {fits(SUITE, now)}   a check fits now: {fits(CHECK, now)}")
    return 0


def cmd_wait(state: Path, kind: str, critical: bool, cmd: list[str], poll_s: float) -> int:
    for sub in ("queue", "claims"):
        (state / sub).mkdir(parents=True, exist_ok=True)
    me = Ticket(critical, time.time_ns(), os.getpid(), kind)
    ticket = state / "queue" / me.name
    ticket.write_text(json.dumps({"critical": critical, "queued_ns": me.queued_ns, "pid": me.pid, "kind": kind}))
    announced = False
    try:
        while True:
            with Lock(state):
                procs = read_ps()
                now = running(procs, _load_claims(state, procs))
                if may_start(me, _load_queue(state), now):
                    child = subprocess.Popen(cmd)
                    claim = state / "claims" / f"{child.pid}.json"
                    program = _program(tuple(cmd))[0]
                    claim.write_text(json.dumps({"pid": child.pid, "kind": kind, "program": program}))
                    ticket.unlink(missing_ok=True)
                    break
            if not announced:
                print(
                    f"box_quiet: waiting to start a {kind}{' (critical)' if critical else ''}: {' '.join(cmd)}",
                    file=sys.stderr,
                    flush=True,
                )
                announced = True
            time.sleep(poll_s)
    finally:
        ticket.unlink(missing_ok=True)
    print(f"box_quiet: started {kind} pid {child.pid}", file=sys.stderr, flush=True)
    try:
        return child.wait()
    finally:
        claim.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    sub = ap.add_subparsers(dest="verb", required=True)
    sub.add_parser("status", help="what is running and queued, and what fits now")
    w = sub.add_parser("wait", help="start <command> when it fits; exit with its status")
    w.add_argument("--critical", action="store_true", help="work the next release waits on (breaks ties only)")
    w.add_argument("--kind", choices=(SUITE, CHECK), help="default: inferred from the command")
    w.add_argument("--poll-s", type=float, default=20.0)
    w.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if args.verb == "status":
        return cmd_status(args.state_dir)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("wait needs a command after --")
    kind = args.kind or infer_kind(cmd)
    if kind is None:
        ap.error(f"cannot tell whether {cmd[0]!r} is a suite or a check: pass --kind")
    return cmd_wait(args.state_dir, kind, args.critical, cmd, args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
