"""Spawn records and ``maxim llm reap``: an operator-run cleanup of a leftover LLM server (#1344 PR 3b).

PR 3a's promise holds: no maxim code kills a process it did not start, and nothing here kills automatically. The one
kill path is :func:`reap`, run by an operator at a terminal (stdin AND stdout TTYs, the exact pid typed), acting only
on a server whose spawn record says a maxim process started it and whose identity still matches at signal time
(docs/plans/bounded_stop.md §7b).

The TTY gate stops an accidental or scripted kill (a pipe, CI, a script, an agent's ordinary shell). A pty wrapper
(``script``, ``expect``, tmux send-keys) defeats it: nothing tells a human from an agent at a pty, so agents are kept
off this command by policy (CLAUDE.md's agent command list).

The record: ``<llm_spawn_record_dir()>/<port>.<pid>.json``, written by ``LocalServerSpawner.start`` right after the
launch and unlinked by its ``stop()`` once the server is reaped. It carries the server's start time (Linux
``/proc/<pid>/stat`` field 22 plus the boot id; macOS ``ps -o lstart``), so a reused pid never matches. A record is
advisory: no record (a pre-3b orphan, a hand-started server) means no reap, which is PR 3a's behaviour.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import select
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, TextIO

from maxim._offline import OfflineRefusal
from maxim.utils import paths as _paths

logger = logging.getLogger(__name__)

FILE_TYPE = "llm_spawn_record"
# PIDFD_SIGNAL_PROCESS_GROUP, ``1 << 2`` in uapi linux/pidfd.h (kernel 6.9+); Python has no name for it.
PIDFD_SIGNAL_PROCESS_GROUP = 4
TERM_GRACE_S = 5.0
_SERVER_MODULE = "llama_cpp.server"
CONNECTIONS_ARE_A_HINT = "connections are a hint: a client between calls holds none"


class _NotTrusted(Exception):
    """The record dir or a record file is not safe to act on (a symlink, another owner, group/world-writable)."""


class _Refused(Exception):
    """The command cannot decide safely (a tool it needs is missing); the message says why."""


class ProcIdentity(NamedTuple):
    start_time: str
    source: str  # "proc_stat" (Linux) or "ps_lstart" (macOS)
    boot_id: str | None


class ListenerScan(NamedTuple):
    """Who listens on a port: the pids that own a listening socket, and how many listening sockets have no visible
    owner (another user's process, on Linux)."""

    pids: frozenset[int]
    unattributed: int


@dataclass
class SpawnRecord:
    """One parsed record file (in memory only; the file itself is a plain dict)."""

    path: Path
    data: dict[str, Any]

    @property
    def pid(self) -> int:
        return int(self.data["pid"])


# ─── process identity ────────────────────────────────────────────────────────────────────────────


def _linux() -> bool:
    return sys.platform.startswith("linux")


def process_identity(pid: int) -> ProcIdentity | None:
    """The process's start time (and boot id on Linux), or None when no such process exists.

    Raises :class:`OSError` when it exists but cannot be read: not the same as gone, so never "provably dead".
    """
    if _linux():
        try:
            raw = Path(f"/proc/{int(pid)}/stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return None
        # comm (field 2) may hold spaces and ")": the fields after the LAST ")" start at field 3 (state).
        fields = raw[raw.rindex(")") + 2 :].split()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return ProcIdentity(start_time=fields[22 - 3], source="proc_stat", boot_id=boot_id)
    out = _ps(pid, "lstart=")
    return None if out is None else ProcIdentity(start_time=out, source="ps_lstart", boot_id=None)


def _ps(pid: int, field: str) -> str | None:
    """``ps -o <field> -p <pid>`` under the C locale and UTC (a stable lstart), or None when ps lists no such pid."""
    try:
        res = subprocess.run(  # noqa: S603, S607
            ["ps", "-ww", "-o", field, "-p", str(int(pid))],
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
        )
    except subprocess.SubprocessError as e:
        raise OSError(errno.EIO, f"ps did not answer for pid {pid}: {e}") from e
    text = res.stdout.strip()
    if res.returncode != 0 and not text:
        return None
    return text or None


def process_argv(pid: int) -> list[str] | None:
    """The process's argv tokens (Linux ``/proc/<pid>/cmdline``; macOS ``ps -ww -o command=``, split on spaces), or
    None when it is gone or unreadable."""
    try:
        if _linux():
            raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
            return [tok.decode(errors="replace") for tok in raw.split(b"\0") if tok]
        out = _ps(pid, "command=")
        return None if out is None else out.split()
    except OSError as e:
        logger.debug("cannot read pid %d's command line: %s", pid, e)
        return None


def cmdline_names_server_on_port(tokens: list[str], port: int) -> bool:
    """True iff ``tokens`` run ``llama_cpp.server`` with the exact argv pair ``--port <port>``: ``--port 81000`` is
    not port 8100, and ``--port=8100`` is not a form the spawner writes."""
    if _SERVER_MODULE not in tokens:
        return False
    want = str(int(port))
    return any(tokens[i] == "--port" and tokens[i + 1] == want for i in range(len(tokens) - 1))


# ─── the record directory and files ──────────────────────────────────────────────────────────────


def _check_trusted(path: Path, *, is_dir: bool) -> None:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise _NotTrusted(f"{path} is a symlink")
    if is_dir and not stat.S_ISDIR(st.st_mode):
        raise _NotTrusted(f"{path} is not a directory")
    if not is_dir and not stat.S_ISREG(st.st_mode):
        raise _NotTrusted(f"{path} is not a regular file")
    if st.st_uid != os.getuid():
        raise _NotTrusted(f"{path} is owned by uid {st.st_uid}, not this user ({os.getuid()})")
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise _NotTrusted(f"{path} is group- or world-writable (mode {stat.S_IMODE(st.st_mode):o})")


def _writable_record_dir() -> Path | None:
    """The record dir, created 0700 and checked, or None (logged) when records are off or the dir is not trusted."""
    if sys.platform == "win32":
        logger.debug("LLM spawn records are POSIX only; none written on native Windows")
        return None
    d = _paths.llm_spawn_record_dir()
    if d is None:
        return None  # llm_spawn_record_dir logged the ERROR once
    try:
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        _check_trusted(d, is_dir=True)
    except (OSError, _NotTrusted) as e:
        logger.error("LLM spawn record dir unusable, no record written (`maxim llm reap` cannot clean up): %s", e)
        return None
    return d


def write_spawn_record(*, pid: int, port: int, model_path: str) -> Path | None:
    """Record that this process spawned the LLM server ``pid`` on ``port``. Returns the file, or None (logged).

    Called right after the launch. First prunes the provably dead records for this port (files only; nothing is
    signalled). A failure never blocks the spawn: no record only means no reap.
    """
    if not isinstance(pid, int) or pid <= 1:  # a test double's pid (int(MagicMock()) == 1 is init), never our child
        logger.debug("no spawn record for pid %r: not a spawned child's pid", pid)
        return None
    d = _writable_record_dir()
    if d is None:
        return None
    try:
        ident = process_identity(pid)
        parent = process_identity(os.getpid())
        if ident is not None:
            prune_dead_records(port, d)
    except (OSError, ValueError, IndexError, OfflineRefusal) as e:
        logger.error("cannot read the start time of pid %d, no spawn record written: %s", pid, e)
        return None
    if ident is None:
        logger.warning("LLM server pid %d exited before its spawn record was written", pid)
        return None
    from maxim import __version__
    from maxim.utils.atomic_io import atomic_write_text
    from maxim.utils.format_version import with_format_version

    payload = with_format_version(
        {
            "pid": int(pid),
            "port": int(port),
            "start_time": ident.start_time,
            "start_time_source": ident.source,
            "boot_id": ident.boot_id,
            "model_path": str(model_path),
            "spawned_by_pid": os.getpid(),
            "spawned_by_start_time": parent.start_time if parent else None,
            "maxim_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    path = d / f"{int(port)}.{int(pid)}.json"
    try:
        atomic_write_text(str(path), json.dumps(payload, indent=2), initial_mode=0o600)
    except OSError as e:
        logger.error("cannot write the LLM spawn record %s: %s", path, e)
        return None
    return path


def remove_spawn_record(port: int, pid: int) -> None:
    """Unlink THIS spawner's own record (``<port>.<pid>.json``), once its server is reaped."""
    d = _paths.llm_spawn_record_dir()
    if d is None or sys.platform == "win32" or not isinstance(pid, int):
        return
    try:
        (d / f"{int(port)}.{int(pid)}.json").unlink(missing_ok=True)
    except OSError as e:
        logger.warning("cannot remove the LLM spawn record for pid %d on port %d: %s", pid, port, e)


def _record_pid_from_name(path: Path, port: int) -> int | None:
    parts = path.name.split(".")
    if len(parts) != 3 or parts[0] != str(port) or not parts[1].isdigit():
        return None
    return int(parts[1])


def _load(path: Path, port: int) -> SpawnRecord | None:
    """One record file, or None when it is not a record for ``port`` (its name and its pid must agree)."""
    from maxim.utils.format_version import check_format_version

    name_pid = _record_pid_from_name(path, port)
    if name_pid is None:
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        return None
    check_format_version(data, FILE_TYPE, log=logger)
    try:
        if int(data["pid"]) != name_pid or int(data.get("port", port)) != port:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return SpawnRecord(path=path, data=data)


def _provably_dead(rec: SpawnRecord) -> bool:
    """The recorded process is gone, or the pid now names another process (start time or boot id differs). An
    unreadable identity is NOT proof."""
    try:
        ident = process_identity(rec.pid)
    except (OSError, ValueError, IndexError):
        return False
    if ident is None:
        return True
    if rec.data.get("boot_id") and ident.boot_id and rec.data["boot_id"] != ident.boot_id:
        return True
    return str(rec.data.get("start_time")) != ident.start_time


def prune_dead_records(port: int, d: Path | None = None) -> list[Path]:
    """Delete the records for ``port`` whose process is provably dead. Files only: nothing is signalled."""
    d = d or _paths.llm_spawn_record_dir()
    removed: list[Path] = []
    if d is None or not d.is_dir():
        return removed
    for path in sorted(d.glob(f"{int(port)}.*.json")):
        try:
            _check_trusted(path, is_dir=False)
            rec = _load(path, port)
        except (OSError, ValueError, _NotTrusted) as e:
            logger.warning("skipping the LLM spawn record %s while pruning: %s", path, e)
            continue
        if rec is not None and _provably_dead(rec):
            try:
                path.unlink()
                removed.append(path)
            except OSError as e:
                logger.warning("cannot prune the dead LLM spawn record %s: %s", path, e)
    # A writer that died mid-write leaves `<port>.<pid>.json.tmp` (the pid is the SERVER's), which the `*.json`
    # glob never sees: drop it once that server is gone (no record is needed for it then).
    for tmp in sorted(d.glob(f"{int(port)}.*.json.tmp")):
        parts = tmp.name.split(".")
        try:
            gone = parts[1].isdigit() and process_identity(int(parts[1])) is None
        except (OSError, ValueError, IndexError):
            gone = False
        if gone:
            try:
                tmp.unlink()
                removed.append(tmp)
            except OSError as e:
                logger.warning("cannot prune the partial LLM spawn record %s: %s", tmp, e)
    return removed


def read_records(port: int) -> list[SpawnRecord]:
    """Every record for ``port``. Raises :class:`_NotTrusted` when the dir or a record file is unsafe to act on."""
    d = _paths.llm_spawn_record_dir()
    if d is None or not os.path.lexists(d):
        return []
    try:
        _check_trusted(d, is_dir=True)
    except OSError as e:
        raise _NotTrusted(f"cannot stat {d}: {e}") from e
    records = []
    for path in sorted(d.glob(f"{int(port)}.*.json")):
        try:
            _check_trusted(path, is_dir=False)
            rec = _load(path, port)
        except OSError as e:
            raise _NotTrusted(f"cannot read {path}: {e}") from e
        except ValueError as e:
            raise _NotTrusted(f"{path} is not a valid record: {e}") from e
        if rec is not None:
            records.append(rec)
    return records


def identity_mismatch(rec: SpawnRecord, port: int) -> str | None:
    """Why ``rec`` no longer names a live llama_cpp.server on ``port`` (pid reused, exited, another command), or None
    when start time, boot id and the exact ``--port`` token pair all still match."""
    try:
        ident = process_identity(rec.pid)
    except (OSError, ValueError, IndexError) as e:
        return f"its identity cannot be read ({e})"
    if ident is None:
        return "it has exited"
    if rec.data.get("boot_id") != ident.boot_id:
        return "the boot id differs (the record is from an earlier boot)"
    if str(rec.data.get("start_time")) != ident.start_time:
        return "its start time differs (the pid now names another process)"
    argv = process_argv(rec.pid)
    if argv is None or not cmdline_names_server_on_port(argv, port):
        return f"its command line is not a {_SERVER_MODULE} with --port {port}"
    return None


# ─── listeners and connections ───────────────────────────────────────────────────────────────────


def _proc_net_tcp(port: int, state: str) -> list[tuple[str, str, str]]:
    """``(local, remote, inode)`` for every /proc/net/tcp{,6} socket in ``state`` whose LOCAL port is ``port``."""
    rows = []
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(name).read_text().splitlines()[1:]
        except FileNotFoundError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10 or parts[3] != state:
                continue
            if int(parts[1].rsplit(":", 1)[1], 16) == port:
                rows.append((parts[1], parts[2], parts[9]))
    return rows


def _socket_owners(inodes: set[str]) -> dict[str, set[int]]:
    """Map socket inodes to the pids holding them, for the processes this user can inspect."""
    owners: dict[str, set[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue  # another user's process, or gone: its sockets stay unattributed
        for fd in fds:
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                owners.setdefault(target[8:-1], set()).add(int(entry))
    return owners


def listening_pids(port: int) -> ListenerScan:
    """Who owns a listening TCP socket on ``port``: Linux reads every uid's sockets in this netns from /proc/net/tcp*
    and maps the ones it can to pids; macOS asks ``lsof`` (missing: :class:`_Refused`)."""
    if _linux():
        inodes = {inode for _, _, inode in _proc_net_tcp(port, "0A")}
        owners = _socket_owners(inodes)
        pids = frozenset(pid for pids in owners.values() for pid in pids)
        return ListenerScan(pids=pids, unattributed=len(inodes - set(owners)))
    try:
        res = subprocess.run(  # noqa: S603, S607
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True, timeout=10
        )
    except FileNotFoundError as e:
        raise _Refused("lsof is not installed, so the listener on the port cannot be named") from e
    except subprocess.SubprocessError as e:
        raise _Refused(f"lsof did not answer: {e}") from e
    if res.returncode not in (0, 1):
        raise _Refused(f"lsof failed (exit {res.returncode}): {res.stderr.strip()}")
    return ListenerScan(pids=frozenset(int(t) for t in res.stdout.split() if t.isdigit()), unattributed=0)


def _hex_addr(raw: str) -> str:
    host, port = raw.rsplit(":", 1)
    if len(host) == 8:
        ip = ".".join(str(b) for b in reversed(bytes.fromhex(host)))
    else:
        ip = f"[ipv6 {host}]"
    return f"{ip}:{int(port, 16)}"


def established_connections(port: int, pid: int) -> list[str]:
    """The server's own ESTABLISHED connections on ``port``, as phrases naming the client (pid and command where this
    user can see them). A hint only: a client between calls holds none."""
    if _linux():
        rows = _proc_net_tcp(port, "01")
        mine = _socket_owners({inode for _, _, inode in rows})
        server_rows = [(lo, rem) for lo, rem, inode in rows if pid in mine.get(inode, set())]
        out = []
        for _local, remote in server_rows:
            client = _client_of(remote, port)
            out.append(f"from {_hex_addr(remote)}" + (f" ({client})" if client else ""))
        return out
    try:
        res = subprocess.run(  # noqa: S603, S607
            ["lsof", "-a", "-nP", "-p", str(pid), f"-iTCP:{port}", "-sTCP:ESTABLISHED", "-Fn"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return [f"(lsof could not list connections: {e})"]
    return [f"from {line[1:].split('->', 1)[-1]}" for line in res.stdout.splitlines() if line.startswith("n")]


def _client_of(remote_hex: str, server_port: int) -> str:
    """The pid/command of the local client whose socket's LOCAL end is ``remote_hex``, when this user can see it."""
    client_port = int(remote_hex.rsplit(":", 1)[1], 16)
    inodes = {
        inode for _, rem, inode in _proc_net_tcp(client_port, "01") if int(rem.rsplit(":", 1)[1], 16) == server_port
    }
    for pids in _socket_owners(inodes).values():
        for cpid in sorted(pids):
            argv = process_argv(cpid) or []
            return f"pid {cpid}: {' '.join(argv)[:120]}"
    return ""


def describe_records(port: int) -> list[str]:
    """One read-only phrase per record for ``port``, for ``maxim doctor``: a stale record, or a recorded server whose
    spawner is gone. Never claims a server is idle: connections are not inspected here."""
    try:
        return _describe_records(port)
    except OfflineRefusal as e:
        return [f"not checked: MAXIM_OFFLINE refuses the ps this needs ({e})"]


def _describe_records(port: int) -> list[str]:
    try:
        records = read_records(port)
    except _NotTrusted as e:
        return [f"spawn records not trusted: {e}"]
    phrases = []
    for rec in records:
        if identity_mismatch(rec, port) is None:
            if _spawner_alive(rec) is False:
                phrases.append(
                    f"a recorded LLM server whose spawner is gone (pid {rec.pid}); "
                    f"`maxim llm reap --port {port}` in a terminal reports it"
                )
        elif _provably_dead(rec):
            phrases.append(f"stale spawn record (pid {rec.pid} gone); the next spawn on port {port} prunes it")
    return phrases


# ─── the command ─────────────────────────────────────────────────────────────────────────────────


def _describe(pid: int) -> str:
    argv = process_argv(pid)
    return f"pid {pid} ({' '.join(argv)[:160]})" if argv else f"pid {pid}"


def _age(rec: SpawnRecord) -> str:
    try:
        created = datetime.fromisoformat(str(rec.data.get("created_at")))
    except ValueError:
        return "unknown age"
    secs = int((datetime.now(timezone.utc) - created).total_seconds())
    return f"started {secs // 3600}h{(secs % 3600) // 60:02d}m ago"


def _spawner_alive(rec: SpawnRecord) -> bool | None:
    """Whether the process that spawned ``rec`` still runs (same pid AND start time); None when it cannot be told."""
    ppid = rec.data.get("spawned_by_pid")
    if not isinstance(ppid, int):
        return None
    try:
        ident = process_identity(ppid)
    except (OSError, ValueError, IndexError):
        return None
    return ident is not None and ident.start_time == str(rec.data.get("spawned_by_start_time"))


def _spawner_state(rec: SpawnRecord) -> str:
    ppid = rec.data.get("spawned_by_pid")
    alive = _spawner_alive(rec)
    if alive is None:
        return f"its spawner (pid {ppid}) cannot be checked"
    if alive:
        return f"its spawner (pid {ppid}) is ALIVE: this server may still be in use"
    return f"its spawner (pid {ppid}) is gone"


def _classify(records: list[SpawnRecord], port: int) -> tuple[dict[int, SpawnRecord], dict[int, str], int]:
    """``(alive by pid, mismatch reason by pid, stale count)``."""
    alive: dict[int, SpawnRecord] = {}
    mismatched: dict[int, str] = {}
    stale = 0
    for rec in records:
        why = identity_mismatch(rec, port)
        if why is None:
            alive[rec.pid] = rec
        elif _provably_dead(rec):
            stale += 1
        else:
            mismatched[rec.pid] = why
    return alive, mismatched, stale


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


def _reap(port: int, stdin: TextIO, stdout: TextIO, say: Any) -> int:
    if sys.platform == "win32":
        say("`maxim llm reap` needs Linux or macOS: native Windows writes no spawn records. Nothing touched.")
        return 2
    if _paths.llm_spawn_record_dir() is None:
        say("no usable spawn-record directory (no passwd home; set MAXIM_LLM_SPAWN_RECORD_DIR). Nothing touched.")
        return 2
    try:
        records = read_records(port)
        alive, mismatched, stale = _classify(records, port)
        scan = listening_pids(port)
    except _NotTrusted as e:
        say(f"spawn records not trusted: {e}. Nothing touched.")
        return 2
    except _Refused as e:
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

    shown = established_connections(port, pid)
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
        now = established_connections(port, pid)
        new = [c for c in now if c not in shown]
        if not new:
            break
        say("new connections since the report:" + "".join(f"\n    {c}" for c in new))
        shown = now
    return _kill(rec, port, say)


# ─── the kill ────────────────────────────────────────────────────────────────────────────────────


def _kill(rec: SpawnRecord, port: int, say: Any) -> int:
    exited = _kill_linux(rec, port, say) if _linux() else _kill_macos(rec, port, say)
    if exited is None:
        return 2
    if not exited:
        say(f"pid {rec.pid} has not exited; its record is kept.")
        return 2
    try:
        rec.path.unlink(missing_ok=True)
    except OSError as e:
        say(f"pid {rec.pid} stopped, but its record {rec.path} could not be removed: {e}")
        return 0
    say(f"pid {rec.pid} stopped; port {port}'s record removed.")
    return 0


def _readable(fd: int, timeout_s: float) -> bool:
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    return bool(poller.poll(int(timeout_s * 1000)))


def _kill_linux(rec: SpawnRecord, port: int, say: Any) -> bool | None:
    """SIGTERM, 5 s, SIGKILL through a pidfd that pins the process. None: refused before any signal."""
    pid = rec.pid
    try:
        fd = os.pidfd_open(pid)
    except OSError as e:
        name = errno.errorcode.get(e.errno or 0, "?")
        say(f"refusing: pidfd_open({pid}) failed ({name}: {os.strerror(e.errno or 0)}); nothing signalled.")
        return None
    try:
        why = identity_mismatch(rec, port)  # re-verified while the pidfd pins the process
        if why is not None:
            say(f"refusing: pid {pid} changed since the report ({why}); nothing signalled.")
            return None
        try:
            group = os.getpgid(pid) == pid  # the spawner starts a new session: pgid == pid
        except ProcessLookupError:
            return True
        except OSError as e:
            say(f"refusing: cannot read pid {pid}'s process group ({e}); nothing signalled.")
            return None
        flagged = False
        if group:
            try:
                signal.pidfd_send_signal(fd, 0, None, PIDFD_SIGNAL_PROCESS_GROUP)
                flagged = True
            except ProcessLookupError:
                return True
            except OSError as e:
                if e.errno != errno.EINVAL:  # EINVAL: a kernel before 6.9, which has no group flag
                    say(f"refusing: probing pid {pid} through its pidfd failed ({e}); nothing signalled.")
                    return None

        def send(sig: int) -> None:
            if flagged:
                signal.pidfd_send_signal(fd, sig, None, PIDFD_SIGNAL_PROCESS_GROUP)
                return
            signal.pidfd_send_signal(fd, sig)
            # Before 6.9 the group goes by number: only while the leader has not exited does it still hold the pgid.
            if group and not _readable(fd, 0):
                os.killpg(pid, sig)

        try:
            send(signal.SIGTERM)
            if not _readable(fd, TERM_GRACE_S):
                send(signal.SIGKILL)
                return _readable(fd, TERM_GRACE_S)
        except ProcessLookupError:
            return True
        except OSError as e:
            say(f"signalling pid {pid} failed: {e}")
            return _readable(fd, 0)
        return True
    finally:
        os.close(fd)


def _gone_or_zombie(pid: int) -> bool:
    try:
        state = _ps(pid, "stat=")
    except OSError:
        return False
    return state is None or state.startswith("Z")


def _kill_macos(rec: SpawnRecord, port: int, say: Any) -> bool | None:
    """macOS has no pidfd: re-verify the identity immediately before EACH signal. None: refused before any signal."""
    pid = rec.pid
    for n, sig in enumerate((signal.SIGTERM, signal.SIGKILL)):
        if _gone_or_zombie(pid):
            return True
        why = identity_mismatch(rec, port)
        if why is not None:
            if n == 0:
                say(f"refusing: pid {pid} changed since the report ({why}); nothing signalled.")
                return None
            say(f"pid {pid} changed before SIGKILL ({why}); stopping here.")
            return _gone_or_zombie(pid)
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, sig)
            else:
                os.kill(pid, sig)
        except ProcessLookupError:
            return True
        deadline = time.monotonic() + TERM_GRACE_S
        while time.monotonic() < deadline:
            if _gone_or_zombie(pid):
                return True
            time.sleep(0.1)
    return _gone_or_zombie(pid)
