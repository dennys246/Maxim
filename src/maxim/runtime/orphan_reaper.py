"""Detect and optionally kill stale maxim sim processes.

A safety net for the "runaway process" failure mode. Normal shutdown
(graceful Ctrl+C + orchestrator cleanup + shared LLM cancellation event)
handles the common case, but edge cases can still leave zombie processes:

- kill -9 on the main process bypasses all cleanup
- Parent shell closes, detaching background runs
- Python crashes mid-sim leave daemon threads alive until the interpreter exits
- Uncaught exceptions in shutdown handlers skip the stop signalling

Each of those can leave a python process hitting the LLM backend until
OS reaps it. On cloud models that means real money. This module enumerates
candidate orphans at sim startup so the user can see and reap them before
launching a new run.

Design:
  - ``find_orphans()`` enumerates maxim sim processes owned by the current
    user, excluding the calling process. Uses psutil if available, falls
    back to ``ps``. Returns an empty list rather than raising on any error.
  - ``warn_about_orphans()`` prints a clear one-liner per orphan with age
    and pid. No I/O if the list is empty.
  - ``reap_orphans()`` sends SIGTERM then SIGKILL after a short grace
    period. Opt-in — only called when the user or config requests it.

The module is deliberately narrow-scope: it only targets processes whose
command line looks like a maxim sim run (``maxim --sim`` or the equivalent
entrypoint form). It will not touch ``maxim peer``, ``maxim doctor``,
leader proxies, or the user's other python processes.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrphanProcess:
    """A candidate stale maxim sim process."""

    pid: int
    age_seconds: float
    cmdline: str


def _looks_like_maxim_sim(cmdline: str) -> bool:
    """Heuristic: does this command line look like an active maxim sim run?

    Matches ``maxim --sim ...`` invocations from the CLI entrypoint. Does
    NOT match ``maxim peer``, ``maxim doctor``, ``maxim tunnel``, or any
    non-sim subcommand — those are legitimate long-running tools the user
    may be running intentionally.
    """
    if "maxim" not in cmdline:
        return False
    # Must be a python process running maxim code (entrypoint or -m)
    if "python" not in cmdline.lower() and "/maxim" not in cmdline:
        return False
    # Must have --sim somewhere in the argv
    return "--sim" in cmdline


def _find_orphans_psutil() -> list[OrphanProcess]:
    """psutil implementation — preferred when available."""
    try:
        import psutil
    except ImportError:
        return []

    self_pid = os.getpid()
    self_uid = os.getuid() if hasattr(os, "getuid") else None
    now = time.time()
    orphans: list[OrphanProcess] = []

    for proc in psutil.process_iter(attrs=["pid", "uids", "create_time", "cmdline"]):
        try:
            info = proc.info
            pid = info.get("pid")
            if pid is None or pid == self_pid:
                continue
            if self_uid is not None:
                uids = info.get("uids")
                if uids is not None and getattr(uids, "real", None) != self_uid:
                    continue
            cmdline_parts = info.get("cmdline") or []
            if not cmdline_parts:
                continue
            cmdline = " ".join(cmdline_parts)
            if not _looks_like_maxim_sim(cmdline):
                continue
            create_time = info.get("create_time") or now
            orphans.append(
                OrphanProcess(
                    pid=pid,
                    age_seconds=max(0.0, now - create_time),
                    cmdline=cmdline,
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception as e:
            logger.debug("psutil iteration error on pid %s: %s", getattr(proc, "pid", "?"), e)
            continue

    return orphans


def _find_orphans_ps() -> list[OrphanProcess]:
    """ps fallback — used when psutil is unavailable.

    Parses ``ps -eo pid,etime,command`` for the current user. Less precise
    on age (ps etime is a human-formatted string), but sufficient to
    surface candidates for manual review.
    """
    self_pid = os.getpid()
    try:
        # -u $USER limits to current user; etime gives elapsed time
        result = subprocess.run(
            ["ps", "-u", str(os.getuid()) if hasattr(os, "getuid") else "0", "-o", "pid=,etime=,command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug("ps fallback failed: %s", e)
        return []

    orphans: list[OrphanProcess] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_str, etime_str, cmdline = parts
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        if not _looks_like_maxim_sim(cmdline):
            continue
        orphans.append(
            OrphanProcess(
                pid=pid,
                age_seconds=_parse_ps_etime(etime_str),
                cmdline=cmdline,
            )
        )
    return orphans


def _parse_ps_etime(etime: str) -> float:
    """Parse ps etime format: ``[[DD-]HH:]MM:SS`` into seconds.

    Returns 0.0 on any parse failure rather than raising — age is
    informational and a missing value shouldn't block reaping.
    """
    try:
        days = 0
        if "-" in etime:
            day_part, etime = etime.split("-", 1)
            days = int(day_part)
        parts = etime.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return days * 86400 + int(h) * 3600 + int(m) * 60 + int(s)
        if len(parts) == 2:
            m, s = parts
            return days * 86400 + int(m) * 60 + int(s)
        return float(parts[0])
    except (ValueError, IndexError):
        return 0.0


def find_orphans() -> list[OrphanProcess]:
    """Return candidate stale maxim sim processes owned by this user.

    Best-effort: returns an empty list on any failure rather than raising.
    Excludes the calling process. Only matches sim-subcommand runs, not
    other maxim tools (peer, doctor, tunnel).
    """
    try:
        orphans = _find_orphans_psutil()
        if orphans:
            return orphans
    except Exception as e:
        logger.debug("psutil orphan scan failed, falling back to ps: %s", e)
    try:
        return _find_orphans_ps()
    except Exception as e:
        logger.debug("ps orphan scan failed: %s", e)
        return []


def warn_about_orphans(orphans: list[OrphanProcess]) -> None:
    """Print a one-line warning per orphan. No-op on empty list.

    Message is terse and points the user at the explicit reap option.
    Goes to stderr so it doesn't interleave with sim display output.
    """
    if not orphans:
        return
    print(
        f"\n  WARNING: {len(orphans)} stale maxim sim process(es) detected.",
        file=sys.stderr,
    )
    for orphan in orphans:
        age_min = orphan.age_seconds / 60.0
        # Truncate cmdline to keep the line readable
        short_cmd = orphan.cmdline if len(orphan.cmdline) <= 80 else orphan.cmdline[:77] + "..."
        print(
            f"    pid={orphan.pid}  age={age_min:.0f}m  {short_cmd}",
            file=sys.stderr,
        )
    print(
        "    These may still be hitting the LLM backend. "
        "Pass --reap-orphans or set MAXIM_REAP_ORPHANS=1 to kill them.\n",
        file=sys.stderr,
    )


def reap_orphans(orphans: list[OrphanProcess], grace_seconds: float = 2.0) -> int:
    """Send SIGTERM then SIGKILL to each orphan. Returns number killed.

    Gives each process ``grace_seconds`` to exit after SIGTERM before
    escalating to SIGKILL. Swallows per-process errors (already dead,
    permission denied) and continues to the next.
    """
    if not orphans:
        return 0

    killed = 0
    # Phase 1: polite SIGTERM
    for orphan in orphans:
        try:
            os.kill(orphan.pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to stale maxim sim process pid=%d", orphan.pid)
        except (ProcessLookupError, PermissionError) as e:
            logger.debug("SIGTERM to pid=%d failed: %s", orphan.pid, e)

    # Phase 2: wait briefly then escalate
    time.sleep(grace_seconds)

    for orphan in orphans:
        try:
            # Probe: kill with signal 0 tests liveness without sending
            os.kill(orphan.pid, 0)
            # Still alive — escalate
            os.kill(orphan.pid, signal.SIGKILL)
            logger.warning("Escalated to SIGKILL for pid=%d (did not exit on SIGTERM)", orphan.pid)
            killed += 1
        except ProcessLookupError:
            # Already dead from SIGTERM — count as killed
            killed += 1
        except PermissionError as e:
            logger.debug("SIGKILL to pid=%d failed: %s", orphan.pid, e)

    return killed
