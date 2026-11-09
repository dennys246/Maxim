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


_FAKE_SERVER = (
    "import os, signal, socket, sys, time\n"
    "if '--ignore-term' in sys.argv:\n"
    "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "child = os.fork() if '--fork-child' in sys.argv else None\n"  # before the bind: the child holds no socket
    "if child == 0:\n"
    "    time.sleep(600)\n"
    "    os._exit(0)\n"
    "flag = '--bind' if '--bind' in sys.argv else '--port'\n"
    "port = int(sys.argv[sys.argv.index(flag) + 1])\n"
    "listen = '--no-listen' not in sys.argv\n"
    "if listen:\n"
    "    s = socket.socket()\n"
    "    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "    s.bind(('127.0.0.1', port))\n"
    "    s.listen()\n"
    "    import threading\n"  # accept like a real server: a backlog connection has no fd to count server-side
    "    held = []\n"
    "    threading.Thread(target=lambda: [held.append(s.accept()) for _ in iter(int, 1)], daemon=True).start()\n"
    "print('ready', flush=True)\n"
    "if child:\n"
    "    print(child, flush=True)\n"
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

    def start(
        port: int | None = None,
        *,
        listen: bool = True,
        argv_port: int | None = None,
        flags: tuple[str, ...] = (),
    ) -> _Fake:
        port = port or _free_port()
        extra = ([] if listen else ["--no-listen"]) + list(flags)
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
    from maxim.runtime.llm_cli import reap

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
@pytest.mark.parametrize("which", ["stdin", "stdout", "both"])
def test_a_non_tty_run_never_kills_never_reads_stdin_and_prints_no_kill_line(record_dir, fake_server, which):
    from maxim.runtime.llm_cli import reap

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
@pytest.mark.parametrize("errno_name", ["ENOSYS", "EPERM"])
def test_a_failed_pidfd_open_refuses_without_signalling_by_number(record_dir, fake_server, monkeypatch, errno_name):
    import errno

    import maxim.runtime.llm_spawn_record as rec

    fake = fake_server()
    _write(fake)
    signalled: list[tuple] = []

    def pidfd_open(pid, flags=0):
        raise OSError(getattr(errno, errno_name), errno_name)

    with monkeypatch.context() as m:  # scoped: the fixture's teardown must still kill its own child
        m.setattr(rec.os, "pidfd_open", pidfd_open)
        m.setattr(rec.os, "kill", lambda *a: signalled.append(("kill", *a)))
        m.setattr(rec.os, "killpg", lambda *a: signalled.append(("killpg", *a)))
        m.setattr(rec.signal, "pidfd_send_signal", lambda *a: signalled.append(("pidfd_send_signal", *a)))
        rc, out, _ = _reap(fake.port, typed=str(fake.pid))
    assert rc == 2
    assert errno_name in out or os.strerror(getattr(errno, errno_name)) in out
    assert signalled == []
    assert fake.alive()


# ─── gate 8: stop() unlinks only its own record; start() prunes only provably dead ones ──────────


@POSIX
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


def test_win32_writes_no_record_and_the_command_refuses(record_dir, monkeypatch):
    import maxim.runtime.llm_spawn_record as rec

    monkeypatch.setattr(rec.sys, "platform", "win32")
    assert rec.write_spawn_record(pid=os.getpid(), port=8100, model_path="x") is None
    assert not record_dir.exists() or not any(record_dir.iterdir())
    rc, out, _ = _reap(8100, typed="1")
    assert rc == 2
    assert "Windows" in out


# ─── gate 12: where the hint is surfaced ──────────────────────────────────────────────────────────


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
def test_a_typed_pid_at_a_tty_reaps_the_recorded_server(record_dir, fake_server):
    fake = fake_server()
    path = _write(fake)
    rc, out, _ = _reap(fake.port, typed=str(fake.pid))
    assert rc == 0, out
    assert fake.proc.wait(timeout=10) is not None
    assert not path.exists()
    assert "connections are a hint: a client between calls holds none" in out


@POSIX
@pytest.mark.parametrize("typed", ["", "y", "yes", "PID"])
def test_anything_but_the_pid_cancels(record_dir, fake_server, typed):
    fake = fake_server()
    path = _write(fake)
    rc, _, _ = _reap(fake.port, typed=typed)
    assert rc == 1
    time.sleep(0.2)
    assert fake.alive() and path.exists()


@POSIX
def test_nothing_recorded_is_exit_0(record_dir):
    rc, out, _ = _reap(_free_port())
    assert rc == 0
    assert "no maxim-recorded server" in out


# ─── gate 14: the CLI route ───────────────────────────────────────────────────────────────────────


def test_maxim_llm_reap_is_routed(monkeypatch):
    import maxim.cli as cli
    import maxim.runtime.llm_cli as llm_cli

    assert "llm" in cli._NON_SIM_SUBCOMMANDS
    seen: list[int] = []
    monkeypatch.setattr(llm_cli, "reap", lambda port, **kw: seen.append(port) or 2)
    assert cli.main(["llm", "reap", "--port", "8123"]) == 2
    assert seen == [8123]


# ─── the re-snapshot after the typed pid (R7-N-d) ─────────────────────────────────────────────────


@POSIX
@pytest.mark.parametrize(("second", "rc_expected"), [("pid", 0), ("no", 1)])
def test_a_connection_that_appears_after_the_report_is_shown_and_asked_again(
    record_dir, fake_server, monkeypatch, second, rc_expected
):
    import maxim.runtime.llm_cli as llm_cli
    import maxim.runtime.llm_spawn_record as rec

    fake = fake_server()
    _write(fake)
    snapshots = iter([[], ["from 127.0.0.1:40000 (pid 1: a client)"], ["from 127.0.0.1:40000 (pid 1: a client)"]])
    monkeypatch.setattr(rec, "established_connections", lambda port, pid: next(snapshots))
    typed = f"{fake.pid}\n" + (f"{fake.pid}\n" if second == "pid" else "no\n")
    stdin, stdout = _Stream(typed, tty=True), _Stream(tty=True)
    assert llm_cli.reap(fake.port, stdin=stdin, stdout=stdout) == rc_expected
    out = stdout.getvalue()
    assert "new connections since the report" in out and "40000" in out
    assert stdin.reads == 2
    if rc_expected == 0:
        assert fake.proc.wait(timeout=10) is not None
    else:
        assert fake.alive()


# ─── maxim doctor's read-only line ────────────────────────────────────────────────────────────────


@POSIX
def test_doctor_reports_stale_records_and_orphaned_servers_never_idleness(record_dir, fake_server):
    from maxim.doctor.checks import check_llm_spawn_records

    fake = fake_server()
    path = _write(fake)
    result = check_llm_spawn_records(fake.port)
    assert result.status == "ok" and result.message.startswith("nothing to report")  # its spawner (us) is alive

    gone = subprocess.Popen([sys.executable, "-c", "pass"], env=without_maxim_overrides())
    gone.wait(timeout=30)
    data = json.loads(path.read_text())
    data["spawned_by_pid"] = gone.pid
    path.write_text(json.dumps(data))
    stale = path.with_name(f"{fake.port}.{gone.pid}.json")
    stale.write_text(json.dumps({"_format_version": "1.0", "pid": gone.pid, "port": fake.port, "start_time": "1"}))
    stale.chmod(0o600)
    message = check_llm_spawn_records(fake.port).message
    assert f"a recorded LLM server whose spawner is gone (pid {fake.pid})" in message
    assert f"stale spawn record (pid {gone.pid} gone)" in message
    assert "connection" not in message and "idle" not in message
    assert fake.alive() and stale.exists()  # read-only


@pytest.mark.parametrize(("env", "port"), [(None, 8100), ("8231", 8231), ("not-a-port", 8100)])
def test_the_reap_defaults_to_the_spawners_port(monkeypatch, env, port):
    """The spawner (lane_backends) reads MAXIM_AUTO_SPAWN_PORT else 8100, and never config.json::auto_spawn.port; the
    reap must look where the server is (Architecture review 1)."""
    from maxim.runtime import llm_cli

    if env is None:
        monkeypatch.delenv("MAXIM_AUTO_SPAWN_PORT", raising=False)
    else:
        monkeypatch.setenv("MAXIM_AUTO_SPAWN_PORT", env)
    assert llm_cli._default_port() == port


# ─── review folds: the checks the deletion probes showed no test caught (Executor 3, 4, 6) ───────


def _state(pid: int) -> str | None:
    """A process's state letter from ps (Z for a zombie), or None when it is gone."""
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return out or None


def _dead(pid: int, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:  # at least one look, even with no timeout
        state = _state(pid)
        if state is None or state.startswith("Z"):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


@POSIX
def test_the_identity_is_re_verified_at_signal_time(record_dir, fake_server, monkeypatch):
    """The pid's start time changing between the report and the kill (a reuse) refuses, before any signal."""
    import maxim.runtime.llm_cli as llm_cli
    import maxim.runtime.llm_spawn_record as rec

    fake = fake_server()
    _write(fake)
    real = rec.process_identity

    def changed(pid):
        ident = real(pid)
        return ident._replace(start_time="0") if pid == fake.pid and ident is not None else ident

    class _Swap(_Stream):
        def readline(self, *args):
            line = super().readline(*args)
            monkeypatch.setattr(rec, "process_identity", changed)
            return line

    stdin, stdout = _Swap(f"{fake.pid}\n", tty=True), _Stream(tty=True)
    assert llm_cli.reap(fake.port, stdin=stdin, stdout=stdout) == 2
    assert "changed since the report" in stdout.getvalue()
    time.sleep(0.2)
    assert fake.alive()


@POSIX
def test_a_group_that_ignores_sigterm_is_sigkilled_whole(record_dir, fake_server, monkeypatch):
    """SIGTERM reaches the whole group, and SIGKILL follows after the grace for the leader AND its group member."""
    import maxim.runtime.llm_spawn_record as rec

    monkeypatch.setattr(rec, "TERM_GRACE_S", 1.0)
    fake = fake_server(flags=("--ignore-term", "--fork-child"))
    child = int(fake.proc.stdout.readline())
    try:
        _write(fake)
        rc, out, _ = _reap(fake.port, typed=str(fake.pid))
        assert rc == 0, out
        assert _dead(fake.pid) and _dead(child)
    finally:
        if not _dead(child, timeout_s=0):
            os.kill(child, 9)  # the fake's own forked child, if the reap left it


@LINUX
def test_a_kernel_with_the_group_flag_signals_through_the_pidfd_only(record_dir, fake_server, monkeypatch):
    """On 6.9+ (PIDFD_SIGNAL_PROCESS_GROUP accepted) every signal goes through the pinned pidfd with the flag, and
    killpg by number is never used."""
    import maxim.runtime.llm_spawn_record as rec

    fake = fake_server()
    _write(fake)
    real = rec.signal.pidfd_send_signal
    calls: list[tuple[int, int]] = []
    killpg: list[tuple] = []

    def pidfd_send_signal(fd, sig, siginfo=None, flags=0):
        calls.append((sig, flags))
        if sig:
            real(fd, sig)  # stands in for the flag on an older kernel: the fake has no group members

    with monkeypatch.context() as m:
        m.setattr(rec.signal, "pidfd_send_signal", pidfd_send_signal)
        m.setattr(rec.os, "killpg", lambda *a: killpg.append(a))
        rc, out, _ = _reap(fake.port, typed=str(fake.pid))
    assert rc == 0, out
    assert calls == [(0, rec.PIDFD_SIGNAL_PROCESS_GROUP), (rec.signal.SIGTERM, rec.PIDFD_SIGNAL_PROCESS_GROUP)]
    assert killpg == []


@POSIX
def test_a_symlinked_record_is_not_trusted(record_dir, fake_server, tmp_path):
    fake = fake_server()
    path = _write(fake)
    elsewhere = tmp_path / "elsewhere.json"
    path.rename(elsewhere)
    path.symlink_to(elsewhere)
    rc, out, _ = _reap(fake.port, typed=str(fake.pid))
    assert rc == 2 and "is a symlink" in out
    time.sleep(0.2)
    assert fake.alive()


@POSIX
def test_stop_keeps_the_record_while_the_server_is_unreaped(record_dir, fake_server, monkeypatch):
    """A stop that did not reap its server (a failed kill) leaves the record: it is what lets an operator reap it."""
    import maxim.runtime.local_server_spawner as LSS

    fake = fake_server()
    path = _write(fake)
    spawner = LSS.LocalServerSpawner(model_path="/models/fake.gguf", port=fake.port)
    spawner._process = fake.proc
    monkeypatch.setattr(LSS, "stop_child_group", lambda proc: None)  # the kill failed: the leader stays unreaped
    monkeypatch.setattr(LSS, "unregister_child", lambda proc: None)
    spawner.stop()
    assert fake.alive() and path.exists()


@POSIX
def test_prune_deletes_a_reused_pid_but_never_an_unreadable_one(record_dir, fake_server, monkeypatch):
    import maxim.runtime.llm_spawn_record as rec

    fake = fake_server(listen=False)
    path = _write(fake)

    def unreadable(pid):
        raise PermissionError(13, "denied")

    with monkeypatch.context() as m:
        m.setattr(rec, "process_identity", unreadable)
        assert rec.prune_dead_records(fake.port) == [] and path.exists()  # unreadable is not proof of death
    data = json.loads(path.read_text())
    data["start_time"] = "0"  # the pid is alive, but it names another process now
    path.write_text(json.dumps(data))
    assert rec.prune_dead_records(fake.port) == [path] and not path.exists()


@POSIX
def test_a_record_whose_name_and_pid_disagree_is_ignored(record_dir, fake_server):
    import maxim.runtime.llm_spawn_record as rec

    fake = fake_server()
    path = _write(fake)
    path.rename(path.with_name(f"{fake.port}.{fake.pid + 1}.json"))
    assert rec.read_records(fake.port) == []


@POSIX
def test_prune_drops_a_partial_record_only_once_its_writer_is_gone(record_dir, fake_server):
    import maxim.runtime.llm_spawn_record as rec

    fake = fake_server()
    _write(fake)
    gone = subprocess.Popen([sys.executable, "-c", "pass"], env=without_maxim_overrides())
    gone.wait(timeout=30)
    dead_tmp = record_dir / f"{fake.port}.{gone.pid}.json.tmp"
    live_tmp = record_dir / f"{fake.port}.{fake.pid}.json.tmp"
    for t in (dead_tmp, live_tmp):
        t.write_text("{")
    assert rec.prune_dead_records(fake.port) == [dead_tmp]
    assert live_tmp.exists()


# ─── review folds: the macOS branch, run on any POSIX host whose ps has stat and lstart (Executor 5, 7) ───


def _ps_has_lstart() -> bool:
    res = subprocess.run(["ps", "-o", "lstart=", "-p", str(os.getpid())], capture_output=True, text=True)
    return bool(res.stdout.strip()) and _state(os.getpid()) is not None


@POSIX
def test_the_ps_branch_reaps_with_identity_rechecked_before_each_signal(record_dir, fake_server, monkeypatch):
    """The macOS path (no /proc, no pidfd): identity from `ps -o lstart`, argv from `ps -o command`, the kill by
    killpg after a re-verify, SIGKILL after the grace. The listener scan is lsof there; it is stubbed."""
    import maxim.runtime.llm_spawn_record as rec

    if not _ps_has_lstart():
        pytest.skip("ps here has no stat/lstart")
    monkeypatch.setattr(rec, "_linux", lambda: False)
    monkeypatch.setattr(rec, "TERM_GRACE_S", 1.0)
    fake = fake_server(flags=("--ignore-term",))
    path = _write(fake)
    data = json.loads(path.read_text())
    assert data["start_time_source"] == "ps_lstart" and data["boot_id"] is None
    scan = rec.ListenerScan(pids=frozenset({fake.pid}), unattributed=0)
    monkeypatch.setattr(rec, "listening_pids", lambda port: scan)
    monkeypatch.setattr(rec, "established_connections", lambda port, pid: [])
    rc, out, _ = _reap(fake.port, typed=str(fake.pid))
    assert rc == 0, out
    assert _dead(fake.pid) and not path.exists()


@POSIX
def test_an_offline_latch_refusing_ps_fails_closed_without_a_traceback(record_dir, fake_server, monkeypatch):
    import maxim.runtime.llm_spawn_record as rec
    from maxim._offline import OfflineRefusal
    from maxim.doctor.checks import check_llm_spawn_records

    fake = fake_server()
    _write(fake)
    monkeypatch.setattr(rec, "_linux", lambda: False)

    def refused(*a, **k):
        raise OfflineRefusal("subprocess.Popen refused")

    monkeypatch.setattr(rec.subprocess, "run", refused)
    rc, out, _ = _reap(fake.port, typed=str(fake.pid))
    assert rc == 2 and "MAXIM_OFFLINE" in out
    assert "MAXIM_OFFLINE" in check_llm_spawn_records(fake.port).message
    assert fake.alive()


# ─── review folds: the record never raises into the spawner (Wire F2, F3) ─────────────────────────


@POSIX
def test_an_unexpandable_override_turns_records_off_instead_of_raising(monkeypatch):
    import maxim.runtime.llm_spawn_record as rec
    from maxim.utils.paths import llm_spawn_record_dir

    monkeypatch.setenv("MAXIM_LLM_SPAWN_RECORD_DIR", "~no-such-user-pr3b/records")
    assert llm_spawn_record_dir() is None
    assert rec.write_spawn_record(pid=os.getpid(), port=8100, model_path="x") is None
    rec.remove_spawn_record(8100, os.getpid())  # no raise


def test_a_test_double_pid_writes_no_record(record_dir):
    from unittest.mock import MagicMock

    import maxim.runtime.llm_spawn_record as rec

    assert rec.write_spawn_record(pid=MagicMock(), port=8100, model_path="x") is None  # int(MagicMock()) is 1: init
    assert rec.write_spawn_record(pid=1, port=8100, model_path="x") is None
    rec.remove_spawn_record(8100, MagicMock())  # no raise
    assert not record_dir.exists() or not any(record_dir.iterdir())


@POSIX
def test_the_ps_branch_re_verifies_before_its_first_signal(record_dir, fake_server, monkeypatch):
    """macOS has no pidfd: a start time that changes between the report and the kill refuses before SIGTERM."""
    import maxim.runtime.llm_cli as llm_cli
    import maxim.runtime.llm_spawn_record as rec

    if not _ps_has_lstart():
        pytest.skip("ps here has no stat/lstart")
    monkeypatch.setattr(rec, "_linux", lambda: False)
    fake = fake_server()
    _write(fake)
    scan = rec.ListenerScan(pids=frozenset({fake.pid}), unattributed=0)
    monkeypatch.setattr(rec, "listening_pids", lambda port: scan)
    monkeypatch.setattr(rec, "established_connections", lambda port, pid: [])
    real = rec.process_identity

    def changed(pid):
        ident = real(pid)
        return ident._replace(start_time="0") if pid == fake.pid and ident is not None else ident

    class _Swap(_Stream):
        def readline(self, *args):
            line = super().readline(*args)
            monkeypatch.setattr(rec, "process_identity", changed)
            return line

    stdin, stdout = _Swap(f"{fake.pid}\n", tty=True), _Stream(tty=True)
    assert llm_cli.reap(fake.port, stdin=stdin, stdout=stdout) == 2
    assert "changed since the report" in stdout.getvalue()
    time.sleep(0.2)
    assert fake.alive()


# ─── the scanners: a real client on Linux, lsof's output and failures on the macOS branch ─────────


@LINUX
def test_a_real_client_connection_is_counted_server_side_and_named(record_dir, fake_server):
    import maxim.runtime.llm_spawn_record as rec

    fake = fake_server()
    with socket.create_connection(("127.0.0.1", fake.port), timeout=5):
        deadline = time.monotonic() + 5
        conns: list[str] = []
        while time.monotonic() < deadline and not conns:
            conns = rec.established_connections(fake.port, fake.pid)
            time.sleep(0.05)
    assert len(conns) == 1
    assert conns[0].startswith("from 127.0.0.1:") and f"pid {os.getpid()}" in conns[0]
    assert rec.listening_pids(fake.port) == rec.ListenerScan(pids=frozenset({fake.pid}), unattributed=0)


def _completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (_completed("123\n456\n"), frozenset({123, 456})),
        (_completed("", returncode=1), frozenset()),  # lsof's "no match"
    ],
)
def test_the_lsof_listener_scan_parses_pids(monkeypatch, outcome, expected):
    import maxim.runtime.llm_spawn_record as rec

    monkeypatch.setattr(rec, "_linux", lambda: False)
    monkeypatch.setattr(rec.subprocess, "run", lambda *a, **k: outcome)
    assert rec.listening_pids(8100) == rec.ListenerScan(pids=expected, unattributed=0)


@pytest.mark.parametrize(
    ("raises", "outcome", "why"),
    [
        (FileNotFoundError(2, "lsof"), None, "not installed"),
        (subprocess.TimeoutExpired("lsof", 10), None, "did not answer"),
        (None, _completed("", returncode=2, stderr="bad"), "exit 2"),
    ],
)
def test_an_lsof_that_cannot_answer_refuses(monkeypatch, raises, outcome, why):
    import maxim.runtime.llm_spawn_record as rec

    def run(*a, **k):
        if raises is not None:
            raise raises
        return outcome

    monkeypatch.setattr(rec, "_linux", lambda: False)
    monkeypatch.setattr(rec.subprocess, "run", run)
    with pytest.raises(rec.ListenerScanRefused, match=why):
        rec.listening_pids(8100)


def test_the_lsof_connection_scan_names_the_remote_end(monkeypatch):
    import maxim.runtime.llm_spawn_record as rec

    monkeypatch.setattr(rec, "_linux", lambda: False)
    out = "p4242\nf7\nn127.0.0.1:8100->127.0.0.1:50000\n"
    monkeypatch.setattr(rec.subprocess, "run", lambda *a, **k: _completed(out))
    assert rec.established_connections(8100, 4242) == ["from 127.0.0.1:50000"]

    def broken(*a, **k):
        raise FileNotFoundError(2, "lsof")

    monkeypatch.setattr(rec.subprocess, "run", broken)
    (hint,) = rec.established_connections(8100, 4242)
    assert "could not list connections" in hint
