"""``maxim llm reap``: an operator-run cleanup of a leftover LLM server (#1344 PR 3b, docs/plans/bounded_stop.md §7b).

The promise PR 3a made still holds: no maxim code kills a process it did not start. The one kill path this adds is run
by an operator at a terminal (stdin and stdout both TTYs, the exact pid typed) and acts only on a server whose spawn
record says a maxim process started it and whose identity still matches at signal time.

Real-process tests start a fake "server": a python one-liner whose argv mimics ``-m llama_cpp.server --port <p>``,
in its own session like the spawner's, listening on loopback. Never a real llama server. Each fixture kills its own
child at teardown.
"""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests._child_env import without_maxim_overrides

POSIX = pytest.mark.skipif(sys.platform == "win32", reason="spawn records are POSIX only")
LINUX = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the pidfd kill path is Linux only")


def red(n: int, why: str):
    return pytest.mark.xfail(strict=True, reason=f"PR 3b red gate {n}: {why}")


_FAKE_SERVER = (
    "import socket, sys, time\n"
    "flag = '--bind' if '--bind' in sys.argv else '--port'\n"
    "port = int(sys.argv[sys.argv.index(flag) + 1])\n"
    "listen = '--no-listen' not in sys.argv\n"
    "if listen:\n"
    "    s = socket.socket()\n"
    "    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "    s.bind(('127.0.0.1', port))\n"
    "    s.listen()\n"
    "print('ready', flush=True)\n"
    "time.sleep(600)\n"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Fake:
    def __init__(self, proc: subprocess.Popen, port: int) -> None:
        self.proc = proc
        self.port = port
        self.pid = proc.pid

    def alive(self) -> bool:
        return self.proc.poll() is None


@pytest.fixture
def record_dir(tmp_path, monkeypatch):
    d = tmp_path / "llm-spawn-records"
    monkeypatch.setenv("MAXIM_LLM_SPAWN_RECORD_DIR", str(d))
    return d


@pytest.fixture
def fake_server():
    started: list[subprocess.Popen] = []

    def start(port: int | None = None, *, listen: bool = True, argv_port: int | None = None) -> _Fake:
        port = port or _free_port()
        extra = [] if listen else ["--no-listen"]
        cmd = [
            sys.executable,
            "-c",
            _FAKE_SERVER,
            "-m",
            "llama_cpp.server",
            "--port",
            str(argv_port if argv_port is not None else port),
            *extra,
        ]
        if argv_port is not None:  # the socket binds `port`, the argv's --port names `argv_port`
            cmd += ["--bind", str(port)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=without_maxim_overrides(),
            start_new_session=True,
            text=True,
        )
        started.append(proc)
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "ready"
        return _Fake(proc, port)

    yield start
    for proc in started:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


class _Stream(io.StringIO):
    """A stdin/stdout stand-in whose isatty() the test chooses; ``reads`` counts every read."""

    def __init__(self, text: str = "", *, tty: bool) -> None:
        super().__init__(text)
        self._tty = tty
        self.reads = 0

    def isatty(self) -> bool:
        return self._tty

    def readline(self, *args):
        self.reads += 1
        return super().readline(*args)

    def read(self, *args):
        self.reads += 1
        return super().read(*args)


def _reap(port: int, *, typed: str | None = None, tty: bool = True) -> tuple[int, str, _Stream]:
    from maxim.runtime.llm_spawn_record import reap

    stdin = _Stream((typed or "") + "\n", tty=tty)
    stdout = _Stream(tty=tty)
    rc = reap(port, stdin=stdin, stdout=stdout)
    return rc, stdout.getvalue(), stdin


def _write(fake: _Fake) -> Path:
    from maxim.runtime.llm_spawn_record import write_spawn_record

    path = write_spawn_record(pid=fake.pid, port=fake.port, model_path="/models/fake.gguf")
    assert path is not None and path.is_file()
    return path


# ─── gate 1: no record ────────────────────────────────────────────────────────────────────────────


@POSIX
@red(1, "no record means no reap")
def test_a_listener_with_no_record_is_named_and_never_killed(record_dir, fake_server):
    fake = fake_server()
    rc, out, _ = _reap(fake.port, typed=str(fake.pid))
    assert rc == 2
    assert str(fake.pid) in out
    assert "not touched" in out
    time.sleep(0.2)
    assert fake.alive()


# ─── gate 2: identity mismatches ──────────────────────────────────────────────────────────────────


@POSIX
@red(2, "a record whose identity no longer matches is never acted on")
@pytest.mark.parametrize("field", ["pid", "start_time", "boot_id"])
def test_a_record_whose_identity_differs_never_kills(record_dir, fake_server, field):
    fake = fake_server()
    path = _write(fake)
    data = json.loads(path.read_text())
    if field == "pid":
        data["pid"] = fake.pid + 100_000
    elif field == "start_time":
        data["start_time"] = str(data["start_time"]) + "9"
    else:
        if not sys.platform.startswith("linux"):
            pytest.skip("boot_id is Linux only")
        data["boot_id"] = "00000000-0000-0000-0000-000000000000"
    path.write_text(json.dumps(data))
    rc, out, _ = _reap(fake.port, typed=str(fake.pid))
    assert rc == 2
    time.sleep(0.2)
    assert fake.alive()


# ─── gate 3: the exact --port token pair ──────────────────────────────────────────────────────────


@red(3, "the cmdline must name the port as an exact token pair, not a substring")
@pytest.mark.parametrize(
    ("tokens", "port", "matches"),
    [
        (["python", "-m", "llama_cpp.server", "--port", "8100"], 8100, True),
        (["python", "-m", "llama_cpp.server", "--port", "81000"], 8100, False),
        (["python", "-m", "llama_cpp.server", "--port", "8100"], 81000, False),
        (["python", "-m", "llama_cpp.server", "--port=8100"], 8100, False),
        (["python", "-m", "other.server", "--port", "8100"], 8100, False),
        (["python", "-m", "llama_cpp.server", "--n_ctx", "8100"], 8100, False),
    ],
)
def test_the_cmdline_identity_is_the_exact_port_token_pair(tokens, port, matches):
    from maxim.runtime.llm_spawn_record import cmdline_names_server_on_port

    assert cmdline_names_server_on_port(tokens, port) is matches


@POSIX
@red(3, "a recorded process whose argv names another port is not the server on this one")
def test_a_server_whose_argv_names_a_longer_port_is_not_matched(record_dir, fake_server):
    port = _free_port()
    fake = fake_server(port, argv_port=int(str(port) + "0"))
    from maxim.runtime.llm_spawn_record import write_spawn_record

    write_spawn_record(pid=fake.pid, port=port, model_path="/models/fake.gguf")
    rc, _, _ = _reap(port, typed=str(fake.pid))
    assert rc == 2
    time.sleep(0.2)
    assert fake.alive()


# ─── gate 4: exactly one listener, the recorded one ───────────────────────────────────────────────


@POSIX
@red(4, "two listeners, a listener that is not the recorded pid, or an unattributable socket refuses")
@pytest.mark.parametrize("seen", ["two", "stranger", "unattributed"])
def test_anything_but_one_visible_recorded_listener_refuses(record_dir, fake_server, monkeypatch, seen):
    import maxim.runtime.llm_spawn_record as rec

    fake = fake_server()
    _write(fake)
    other = os.getpid()
    scans = {
        "two": rec.ListenerScan(pids=frozenset({fake.pid, other}), unattributed=0),
        "stranger": rec.ListenerScan(pids=frozenset({other}), unattributed=0),
        "unattributed": rec.ListenerScan(pids=frozenset({fake.pid}), unattributed=1),
    }
    monkeypatch.setattr(rec, "listening_pids", lambda port: scans[seen])
    rc, _, _ = _reap(fake.port, typed=str(fake.pid))
    assert rc == 2
    time.sleep(0.2)
    assert fake.alive()


# ─── gate 5: alive but not listening ──────────────────────────────────────────────────────────────


@POSIX
@red(5, "a recorded server that is alive but not listening (loading or hung) is reported, never killed")
def test_an_alive_recorded_server_that_is_not_listening_is_reported_not_killed(record_dir, fake_server):
    fake = fake_server(listen=False)
    _write(fake)
    rc, out, _ = _reap(fake.port, typed=str(fake.pid))
    assert rc == 2
    assert "alive but not listening" in out
    time.sleep(0.2)
    assert fake.alive()


# ─── gate 6: a non-TTY run never kills ────────────────────────────────────────────────────────────


@POSIX
@red(6, "a non-TTY run prints the report and the command to run, never reads stdin, never prints a kill line")
@pytest.mark.parametrize("which", ["stdin", "stdout", "both"])
def test_a_non_tty_run_never_kills_never_reads_stdin_and_prints_no_kill_line(record_dir, fake_server, which):
    from maxim.runtime.llm_spawn_record import reap

    fake = fake_server()
    _write(fake)
    stdin = _Stream(f"{fake.pid}\n", tty=which == "stdout")
    stdout = _Stream(tty=which == "stdin")
    rc = reap(fake.port, stdin=stdin, stdout=stdout)
    out = stdout.getvalue()
    assert rc == 2
    assert stdin.reads == 0
    assert f"run `maxim llm reap --port {fake.port}` yourself in a terminal" in out
    assert f"kill {fake.pid}" not in out and "kill -" not in out and "killpg" not in out
    time.sleep(0.2)
    assert fake.alive()


# ─── gate 7: pidfd errors refuse ──────────────────────────────────────────────────────────────────


@LINUX
@red(7, "on Linux a pidfd_open that fails refuses; the kill never falls back to os.kill by number")
@pytest.mark.parametrize("errno_name", ["ENOSYS", "EPERM"])
def test_a_failed_pidfd_open_refuses_without_signalling_by_number(record_dir, fake_server, monkeypatch, errno_name):
    import errno

    import maxim.runtime.llm_spawn_record as rec

    fake = fake_server()
    _write(fake)
    signalled: list[tuple] = []

    def pidfd_open(pid, flags=0):
        raise OSError(getattr(errno, errno_name), errno_name)

    monkeypatch.setattr(rec.os, "pidfd_open", pidfd_open)
    monkeypatch.setattr(rec.os, "kill", lambda *a: signalled.append(("kill", *a)))
    monkeypatch.setattr(rec.os, "killpg", lambda *a: signalled.append(("killpg", *a)))
    rc, out, _ = _reap(fake.port, typed=str(fake.pid))
    assert rc == 2
    assert errno_name in out or os.strerror(getattr(errno, errno_name)) in out
    assert signalled == []
    assert fake.alive()


# ─── gate 8: stop() unlinks only its own record; start() prunes only provably dead ones ──────────


@POSIX
@red(8, "the spawner writes its record, prunes only provably dead records for its own port, unlinks only its own")
def test_the_spawner_writes_prunes_and_unlinks_only_what_it_should(record_dir, fake_server, tmp_path, monkeypatch):
    import maxim.runtime.local_server_spawner as LSS
    from maxim.runtime.llm_spawn_record import write_spawn_record

    port = _free_port()
    other = fake_server(port, listen=False)  # alive, matching, same port: not provably dead
    alive_path = write_spawn_record(pid=other.pid, port=port, model_path="/models/a.gguf")
    record_dir.mkdir(mode=0o700, exist_ok=True)
    gone = subprocess.Popen([sys.executable, "-c", "pass"], env=without_maxim_overrides())
    gone.wait(timeout=30)
    dead_same_port = record_dir / f"{port}.{gone.pid}.json"
    dead_same_port.write_text(json.dumps({"_format_version": "1.0", "pid": gone.pid, "port": port, "start_time": "1"}))
    dead_other_port = record_dir / f"{port + 1}.{gone.pid}.json"
    dead_other_port.write_text(
        json.dumps({"_format_version": "1.0", "pid": gone.pid, "port": port + 1, "start_time": "1"})
    )
    unrelated = record_dir / "notes.txt"
    unrelated.write_text("not a record")
    for p in (dead_same_port, dead_other_port, unrelated):
        p.chmod(0o600)

    model = tmp_path / "fake.gguf"
    model.write_bytes(b"GGUF")
    spawner = LSS.LocalServerSpawner(model_path=str(model), port=port)
    fake_cmd = [sys.executable, "-c", _FAKE_SERVER, "-m", "llama_cpp.server", "--port", str(port), "--no-listen"]
    monkeypatch.setattr(spawner, "_build_cmd", lambda: fake_cmd)
    monkeypatch.setattr(spawner, "_wait_ready", lambda timeout_s: True)
    monkeypatch.setattr(LSS, "verify_llm_server_exposure", lambda port, allow_authenticated_lan: None)
    try:
        assert spawner.start() is not None
        own = record_dir / f"{port}.{spawner._process.pid}.json"
        assert own.is_file()
        data = json.loads(own.read_text())
        assert data["_format_version"] == "1.0"
        assert data["pid"] == spawner._process.pid and data["port"] == port
        assert data["spawned_by_pid"] == os.getpid()
        assert not dead_same_port.exists()
        assert dead_other_port.exists() and unrelated.exists() and alive_path.exists()
    finally:
        spawner.stop()
    assert not own.exists()
    assert alive_path.exists() and dead_other_port.exists()


# ─── gate 10: insecure record dir or file ─────────────────────────────────────────────────────────


@POSIX
@red(10, "a group/world-writable or foreign-owned record dir or file is never written to or trusted")
@pytest.mark.parametrize("what", ["dir_mode", "file_mode", "foreign_owner"])
def test_an_insecure_record_dir_or_file_is_refused(record_dir, fake_server, monkeypatch, what):
    import maxim.runtime.llm_spawn_record as rec

    fake = fake_server()
    path = _write(fake)
    if what == "dir_mode":
        record_dir.chmod(0o777)
    elif what == "file_mode":
        path.chmod(0o666)
    else:
        real_uid = os.getuid()
        monkeypatch.setattr(rec.os, "getuid", lambda: real_uid + 1)
    rc, out, _ = _reap(fake.port, typed=str(fake.pid))
    assert rc == 2
    assert "not trusted" in out
    time.sleep(0.2)
    assert fake.alive()
    if what != "file_mode":
        assert rec.write_spawn_record(pid=fake.pid, port=fake.port + 1, model_path="x") is None


# ─── gate 11: native Windows ──────────────────────────────────────────────────────────────────────


@red(11, "native Windows writes no record and the command refuses")
def test_win32_writes_no_record_and_the_command_refuses(record_dir, monkeypatch):
    import maxim.runtime.llm_spawn_record as rec

    monkeypatch.setattr(rec.sys, "platform", "win32")
    assert rec.write_spawn_record(pid=os.getpid(), port=8100, model_path="x") is None
    assert not record_dir.exists() or not any(record_dir.iterdir())
    rc, out, _ = _reap(8100, typed="1")
    assert rc == 2
    assert "Windows" in out


# ─── gate 12: where the hint is surfaced ──────────────────────────────────────────────────────────


@red(12, "the reap hint goes on the wrong-model and key-rejected refusals, never on the busy one")
def test_the_reap_hint_is_on_wrong_model_and_key_rejected_never_on_port_busy(monkeypatch):
    import maxim.runtime.local_server_spawner as LSS
    from maxim.exceptions import LLMServerPortBusy
    from maxim.runtime import llm_server

    monkeypatch.setattr(LSS, "describe_port_listener", lambda port: f"pid 4242 on {port}")
    hint = "maxim llm reap --port 8123"
    assert hint in str(llm_server._wrong_model_error("http://127.0.0.1:8123/v1", "a.gguf", "b"))
    assert hint in str(LSS.LLMServerKeyRejected(8123))
    busy = LLMServerPortBusy(8123, "pid 4242", "stayed busy for the whole spawn budget")
    assert "reap" not in str(busy) and "reap" not in busy.fix_hint


# ─── gate 13: the positive path ───────────────────────────────────────────────────────────────────


@POSIX
@red(13, "at a TTY, a recorded matching server whose pid is typed is reaped and its record unlinked")
def test_a_typed_pid_at_a_tty_reaps_the_recorded_server(record_dir, fake_server):
    fake = fake_server()
    path = _write(fake)
    rc, out, _ = _reap(fake.port, typed=str(fake.pid))
    assert rc == 0, out
    assert fake.proc.wait(timeout=10) is not None
    assert not path.exists()
    assert "connections are a hint: a client between calls holds none" in out


@POSIX
@red(13, "anything but the exact pid cancels with exit 1")
@pytest.mark.parametrize("typed", ["", "y", "yes", "PID"])
def test_anything_but_the_pid_cancels(record_dir, fake_server, typed):
    fake = fake_server()
    path = _write(fake)
    rc, _, _ = _reap(fake.port, typed=typed)
    assert rc == 1
    time.sleep(0.2)
    assert fake.alive() and path.exists()


@POSIX
@red(13, "no listener and no live record is nothing to do: exit 0, stale records counted")
def test_nothing_recorded_is_exit_0(record_dir):
    rc, out, _ = _reap(_free_port())
    assert rc == 0
    assert "no maxim-recorded server" in out


# ─── gate 14: the CLI route ───────────────────────────────────────────────────────────────────────


@red(14, "`maxim llm reap` is a non-sim subcommand routed to the reap command")
def test_maxim_llm_reap_is_routed(monkeypatch):
    import maxim.cli as cli
    import maxim.runtime.llm_spawn_record as rec

    assert "llm" in cli._NON_SIM_SUBCOMMANDS
    seen: list[int] = []
    monkeypatch.setattr(rec, "reap", lambda port, **kw: seen.append(port) or 2)
    assert cli.main(["llm", "reap", "--port", "8123"]) == 2
    assert seen == [8123]
