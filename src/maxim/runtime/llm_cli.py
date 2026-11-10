"""``maxim llm``: operator commands for the local LLM server (#1344 PR 3b).

Only ``reap`` today: report the maxim-recorded server on a port and, at a terminal, kill it once the operator types
its pid. This module is the command's operator side (the report, the TTY gate, the typed pid, the messages); the
records, the identity and listener checks and the kill itself are ``runtime/llm_spawn_record.py``, which nothing
here bypasses. Not ``maxim peer llm``, which switches a leader's model.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import TextIO

from maxim._offline import OfflineRefusal
from maxim.runtime import llm_spawn_record as _rec
from maxim.utils import paths as _paths

CONNECTIONS_ARE_A_HINT = "connections are a hint: a client between calls holds none"


def _default_port() -> int:
    """The port the spawner uses: ``MAXIM_AUTO_SPAWN_PORT``, else 8100 (``lane_backends.py`` reads exactly this, and
    falls back to 8100 on a non-number too). Not ``config.json::auto_spawn.port``: no spawner reads that key today
    (#1618), so honouring it here would send the reap to a port the server is not on."""
    try:
        return int(os.environ.get("MAXIM_AUTO_SPAWN_PORT", "8100"))
    except ValueError:
        return 8100


def run_llm_subcommand(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="maxim llm", description="Operator commands for the local LLM server.")
    sub = parser.add_subparsers(dest="action", required=True)
    reap_p = sub.add_parser(
        "reap",
        help="clean up a leftover maxim-spawned LLM server (asks you to type its pid; refuses outside a terminal)",
        description=(
            "Report the llama_cpp.server on PORT. If a maxim process recorded spawning it and its identity still "
            "matches, and stdin and stdout are a terminal, type its pid to stop it. Never touches a server maxim did "
            "not record. Exit 0 reaped or nothing recorded, 1 cancelled, 2 refused."
        ),
    )
    reap_p.add_argument(
        "--port", type=int, default=None, help="the server's port (default: MAXIM_AUTO_SPAWN_PORT, else 8100)"
    )
    args = parser.parse_args(list(argv))

    port = args.port if args.port is not None else _default_port()
    if not 1 <= port <= 65535:
        parser.error(f"--port {port} is not a TCP port")
    return reap(port)


# ─── maxim llm reap ──────────────────────────────────────────────────────────────────────────────


def reap(port: int, *, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    """``maxim llm reap [--port N]``: report the maxim-recorded LLM server on ``port`` and, at a terminal only, kill it
    once the operator types its pid. Exit 0 reaped or nothing recorded, 1 cancelled, 2 refused."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    def say(msg: str) -> None:
        print(msg, file=stdout, flush=True)

    try:
        return _reap(port, stdin, stdout, say)
    except OfflineRefusal as e:  # macOS reads identities and listeners through ps/lsof, which the latch refuses
        say(f"refusing: MAXIM_OFFLINE refuses the ps/lsof this needs ({e}). Nothing touched.")
        return 2


def _describe(pid: int) -> str:
    argv = _rec.process_argv(pid)
    return f"pid {pid} ({' '.join(argv)[:160]})" if argv else f"pid {pid}"


def _age(rec: _rec.SpawnRecord) -> str:
    try:
        created = datetime.fromisoformat(str(rec.data.get("created_at")))
    except ValueError:
        return "unknown age"
    secs = int((datetime.now(timezone.utc) - created).total_seconds())
    return f"started {secs // 3600}h{(secs % 3600) // 60:02d}m ago"


def _spawner_state(rec: _rec.SpawnRecord) -> str:
    ppid = rec.data.get("spawned_by_pid")
    alive = _rec.spawner_alive(rec)
    if alive is None:
        return f"its spawner (pid {ppid}) cannot be checked"
    if alive:
        return f"its spawner (pid {ppid}) is ALIVE: this server may still be in use"
    return f"its spawner (pid {ppid}) is gone"


def _reap(port: int, stdin: TextIO, stdout: TextIO, say: Callable[[str], None]) -> int:
    if sys.platform == "win32":
        say("`maxim llm reap` needs Linux or macOS: native Windows writes no spawn records. Nothing touched.")
        return 2
    if _paths.llm_spawn_record_dir() is None:
        say("no usable spawn-record directory (no passwd home; set MAXIM_LLM_SPAWN_RECORD_DIR). Nothing touched.")
        return 2
    try:
        records = _rec.read_records(port)
        alive, mismatched, stale = _rec.classify_records(records, port)
        scan = _rec.listening_pids(port)
    except _rec.RecordsNotTrusted as e:
        say(f"spawn records not trusted: {e}. Nothing touched.")
        return 2
    except _rec.ListenerScanRefused as e:
        say(f"refusing: {e}. Nothing touched.")
        return 2
    stale_note = f" ({stale} stale record{'s' if stale != 1 else ''} left for the next spawn to prune)" if stale else ""

    if not scan.pids and not scan.unattributed:
        if alive:
            for live in alive.values():
                say(
                    f"recorded server pid {live.pid} is alive but not listening on port {port} (loading or hung); "
                    f"{_spawner_state(live)}. Not touched: wait for it, or stop it by hand."
                )
            return 2
        say(f"no maxim-recorded server on port {port}{stale_note}.")
        return 0
    if scan.unattributed or len(scan.pids) != 1:
        seen = ", ".join(_describe(p) for p in sorted(scan.pids)) or "none visible"
        hidden = f" and {scan.unattributed} socket(s) whose owner this user cannot see" if scan.unattributed else ""
        say(f"refusing: port {port} has listeners {seen}{hidden}; expected exactly one. Nothing touched.")
        return 2
    (pid,) = scan.pids
    rec = alive.get(pid)
    if rec is None:
        if pid in mismatched:
            say(f"{_describe(pid)} listens on port {port} and has a record, but {mismatched[pid]}. Not touched.")
        else:
            say(
                f"{_describe(pid)} listens on port {port}: not started by a maxim that records spawns; not touched."
                + stale_note
            )
        return 2

    shown = _rec.established_connections(port, pid)
    say(f"port {port}: maxim-recorded llama_cpp.server {_describe(pid)}")
    say(f"  model: {rec.data.get('model_path')}; {_age(rec)}; {_spawner_state(rec)}")
    say(f"  established connections: {len(shown)}" + "".join(f"\n    {c}" for c in shown))
    say(f"  {CONNECTIONS_ARE_A_HINT}")
    if not (stdin.isatty() and stdout.isatty()):
        say(
            f"Not at a terminal, so nothing was touched: run `maxim llm reap --port {port}` yourself in a terminal "
            "to clean it up."
        )
        return 2
    while True:
        stdout.write(f"type the pid ({pid}) to kill it (anything else cancels): ")
        stdout.flush()
        if stdin.readline().strip() != str(pid):
            say("cancelled. Nothing touched.")
            return 1
        now = _rec.established_connections(port, pid)
        new = [c for c in now if c not in shown]
        if not new:
            break
        say("new connections since the report:" + "".join(f"\n    {c}" for c in new))
        shown = now
    return _rec.kill_recorded(rec, port, say)
