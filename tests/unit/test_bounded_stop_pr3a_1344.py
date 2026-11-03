"""Bounded stop PR 3a (#1344, plan §7, owner split 2026-11-03): no maxim code kills a process it did not start, and
every LLM server and tunnel daemon it spawns dies on a forced stop.

The gates drive the shipped spawners and the shipped auto-spawn path against real child processes and a real local
HTTP listener on a free port (never 8100). The box-wide ``kill_stale_llm_servers`` is replaced by a recorder wherever
the pre-fix code would reach it, so a red run on main cannot kill another session's server.
"""

from __future__ import annotations

import dataclasses
import http.server
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from maxim.runtime import lane_backends, llm_server
from maxim.runtime import local_server_spawner as lss
from maxim.utils import cancellation

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="process groups and killpg are POSIX")

KEY = "k" * 32
RED = dict(strict=True, raises=AssertionError)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_dead(pid: int, limit_s: float = 5.0) -> bool:
    end = time.monotonic() + limit_s
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _kill_group_quietly(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass


# ── A llama-cpp-server stand-in: keyless /v1/models answers 401 at once (auth runs before the generation lock), a
# keyed one waits while the server is "busy" (llama_cpp's /v1/models depends on that lock) ─────────────────────────


class _FakeLlamaServer:
    def __init__(self, *, busy_s: float, model_id: str) -> None:
        self.busy_until = time.monotonic() + busy_s
        self.model_id = model_id
        self.keyed_requests = 0
        outer = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                if self.headers.get("Authorization") != f"Bearer {KEY}":
                    self.send_response(401)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                outer.keyed_requests += 1
                while time.monotonic() < outer.busy_until:
                    time.sleep(0.05)
                body = json.dumps({"object": "list", "data": [{"id": outer.model_id}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.daemon_threads = True
        self.port = int(self.server.server_address[1])
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@dataclasses.dataclass
class _Lane:
    name: str = "large"
    remote_url: str | None = None
    model_profile: str = "m"
    placement: tuple = ()
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    kv_quant_mode: str = "f16"
    served_n_ctx: int | None = None
    remote_model: str | None = None
    remote_api_key: str | None = None
    remote_api_key_is_server_key: bool = False


@pytest.fixture
def auto_spawn_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """``_maybe_auto_spawn_server`` up to its singleton check, with the box-wide killer and the spawner recorded."""
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(server=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "llama_cpp.server", SimpleNamespace())
    monkeypatch.setattr(lane_backends, "_read_persisted_model", lambda: None)
    monkeypatch.setattr(lane_backends, "_server_key_for_role", lambda: KEY)
    monkeypatch.setattr(lane_backends, "_lane_profile_is_cloud", lambda *a: False)
    monkeypatch.setattr(
        "maxim.models.language.config.load_llm_config", lambda **_: SimpleNamespace(model_path=str(gguf))
    )
    monkeypatch.setattr(
        "maxim.runtime.leader_mode.detect_role", lambda: SimpleNamespace(role="solo", bind_host="127.0.0.1", reason="t")
    )
    monkeypatch.setattr("maxim.utils.net.primary_ipv4_address", lambda: None)
    monkeypatch.setattr(lane_backends, "_check_vram_spillover_risk", lambda **_: None)
    monkeypatch.setenv("MAXIM_AUTO_SPAWN_TIMEOUT_S", "6")
    killed: list[Any] = []
    built: list[dict] = []
    monkeypatch.setattr(lss, "kill_stale_llm_servers", lambda *a, **k: killed.append(a) or 0, raising=False)

    class _Spawner:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

        def start(self, timeout_s: float) -> None:
            return None

    monkeypatch.setattr(lss, "LocalServerSpawner", _Spawner)
    monkeypatch.setattr(llm_server, "_active_spawner", None)
    lanes = {"large": _Lane()}

    def run(port: int) -> Any:
        monkeypatch.setenv("MAXIM_AUTO_SPAWN_PORT", str(port))
        return lane_backends._maybe_auto_spawn_server(SimpleNamespace(has_gpu=True, vram_gb=16.0), lanes, None)

    return SimpleNamespace(run=run, killed=killed, built=built, gguf=gguf)


# ── G1/G2: every server and tunnel this process spawns dies on a forced stop, the readiness wait included ─────────


_DRIVER = textwrap.dedent(
    """
    import os, sys, threading, time
    from pathlib import Path
    kind, scratch = sys.argv[1], Path(sys.argv[2])
    from maxim import cli_signals
    if kind == "server":
        from maxim.runtime.local_server_spawner import LocalServerSpawner
        gguf = scratch / "m.gguf"
        gguf.write_bytes(b"GGUF")
        sp = LocalServerSpawner(model_path=str(gguf), port=1, api_key="k")
        sp._build_cmd = lambda: [sys.executable, "-c", "import time; time.sleep(120)"]
        sp._health_check = lambda: False  # never ready: the readiness wait runs for the whole test
        threading.Thread(target=lambda: sp.start(timeout_s=60), daemon=True).start()
    else:
        from maxim.tunnel import daemon_spawner
        script = scratch / "cloudflared"
        script.write_text("#!/bin/sh\\nexec sleep 120\\n")
        script.chmod(0o755)
        config = scratch / "config.yml"
        config.write_text("tunnel: t\\n")
        daemon_spawner.find_cloudflared = lambda: str(script)
        sp = daemon_spawner.TunnelDaemonSpawner(config_path=config)
        threading.Thread(target=sp.start, daemon=True).start()
    end = time.monotonic() + 20
    while sp._process is None and time.monotonic() < end:
        time.sleep(0.02)
    print(f"CHILD {sp._process.pid}", flush=True)
    cli_signals._die(75, "forced stop (test)")
    """
)


@pytest.mark.parametrize("kind", ["server", "tunnel"])
@pytest.mark.xfail(**RED)
def test_g1_a_spawned_child_dies_with_a_forced_stop(kind: str, tmp_path: Path) -> None:
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER)
    env = {**os.environ, "PYTHONPATH": str(Path(lss.__file__).resolve().parents[2])}
    out = subprocess.run(
        [sys.executable, str(driver), kind, str(tmp_path)], capture_output=True, text=True, timeout=60, env=env
    )
    pids = [int(line.split()[1]) for line in out.stdout.splitlines() if line.startswith("CHILD ")]
    assert pids, out.stdout + out.stderr
    try:
        assert out.returncode == 75, out.stderr
        assert _wait_dead(pids[0]), f"the spawned {kind} survived the forced stop"
    finally:
        _kill_group_quietly(pids[0])


# ── G3: the box-wide killer is gone from both callers, and from the module ───────────────────────────────────────


@pytest.mark.xfail(**RED)
def test_g3_nothing_kills_a_server_it_did_not_start() -> None:
    root = Path(lss.__file__).resolve().parents[1]
    for rel in ("runtime/lane_backends.py", "runtime/leader_proxy.py", "runtime/local_server_spawner.py"):
        text = (root / rel).read_text()
        assert "kill_stale_llm_servers" not in text and "find_stale_llm_servers" not in text, rel
        assert "_kill_process_tree" not in text, rel


# ── G4/G5: a keyed listener that is busy is waited for; one that never answers is refused, never killed ──────────


@pytest.mark.xfail(**RED)
def test_g4_a_busy_server_is_waited_for_then_reused(auto_spawn_env: SimpleNamespace) -> None:
    server = _FakeLlamaServer(busy_s=4.0, model_id=str(auto_spawn_env.gguf))  # longer than the 0.8 s + 2 s probe
    try:
        out = auto_spawn_env.run(server.port)
    finally:
        server.close()
    assert auto_spawn_env.killed == [], "the box-wide killer ran"
    assert auto_spawn_env.built == [], "a second server was spawned beside a live one"
    assert out["large"].remote_url == f"http://127.0.0.1:{server.port}/v1"


@pytest.mark.xfail(**RED)
def test_g5_a_listener_that_never_answers_is_refused_with_its_port(auto_spawn_env: SimpleNamespace) -> None:
    busy = getattr(lss, "LLMServerPortBusy", None)
    assert busy is not None, "no typed refusal for an occupied port"
    server = _FakeLlamaServer(busy_s=60.0, model_id=str(auto_spawn_env.gguf))
    try:
        with pytest.raises(busy) as caught:
            auto_spawn_env.run(server.port)
    finally:
        server.busy_until = 0.0
        server.close()
    assert str(server.port) in str(caught.value)
    assert getattr(caught.value, "fix_hint", "")
    assert auto_spawn_env.killed == [] and auto_spawn_env.built == []


# ── G6: our own busy server is reused on a model match, and refused (loud, the existing text) on a mismatch ─────


class _OwnSpawner:
    def __init__(self, port: int, model_path: str) -> None:
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.model_path = model_path
        self.n_ctx = 4096
        self.is_running = True


@pytest.mark.xfail(**RED)
def test_g6_our_own_busy_server_is_reused_only_on_a_model_match(
    auto_spawn_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _FakeLlamaServer(busy_s=60.0, model_id=str(auto_spawn_env.gguf))
    try:
        monkeypatch.setattr(llm_server, "_active_spawner", _OwnSpawner(server.port, str(auto_spawn_env.gguf)))
        out = auto_spawn_env.run(server.port)
        assert out["large"].remote_url == f"http://127.0.0.1:{server.port}/v1"
        monkeypatch.setattr(llm_server, "_active_spawner", _OwnSpawner(server.port, "/elsewhere/other.gguf"))
        with pytest.raises(RuntimeError, match="serving model"):
            auto_spawn_env.run(server.port)
    finally:
        server.busy_until = 0.0
        server.close()
    assert auto_spawn_env.killed == [] and auto_spawn_env.built == []


# ── G7: a stop pending at the spawn decision starts no model load ─────────────────────────────────────────────────


@pytest.mark.xfail(**RED)
def test_g7_a_pending_stop_starts_no_spawn(auto_spawn_env: SimpleNamespace) -> None:
    from maxim.exceptions import StopStillPending
    from maxim.utils.deadline import Deadline

    request = cancellation.request_stop("test", Deadline.after(5))
    try:
        try:
            auto_spawn_env.run(_free_port())  # nothing listens: on main this goes straight to a spawn
        except StopStillPending:
            pass
        else:
            raise AssertionError("a spawn decision ran under a pending stop")
    finally:
        cancellation.clear_stop_request(request)
    assert auto_spawn_env.built == []


# ── G8: an occupied port is a typed refusal at the CLI: exit 2 with its hint, not a traceback ────────────────────


@pytest.mark.xfail(**RED)
def test_g8_an_occupied_port_exits_2_with_its_hint(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    from maxim import cli

    busy = getattr(lss, "LLMServerPortBusy", None)
    assert busy is not None, "no typed refusal for an occupied port"

    def _raise(argv: Any = None) -> int:
        raise busy(4321, "a listener (pid 1234)")

    monkeypatch.setattr(cli, "_main_impl", _raise)
    assert cli._main_guarded([]) == 2
    err = capsys.readouterr().err
    assert "4321" in err


# ── G9: stop() signals the server's whole group (a grandchild dies too), and never after the reap ───────────────


@pytest.mark.xfail(**RED)
def test_g9_stop_ends_the_whole_group_and_never_signals_after_the_reap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    code = (
        "import subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "print(g.pid, flush=True)\n"
        "time.sleep(120)\n"
    )
    spawner = lss.LocalServerSpawner(model_path=str(gguf), port=_free_port(), api_key=KEY)
    monkeypatch.setattr(spawner, "_build_cmd", lambda: [sys.executable, "-c", code])
    monkeypatch.setattr(spawner, "_wait_ready", lambda _t: True)
    monkeypatch.setattr(lss, "verify_llm_server_exposure", lambda *a, **k: None)
    real_popen = subprocess.Popen

    def _piped(*a: Any, **k: Any) -> Any:
        k["stdout"] = subprocess.PIPE
        return real_popen(*a, **k)

    monkeypatch.setattr(lss.subprocess, "Popen", _piped)
    signals: list[tuple[int, Any]] = []
    real_killpg = os.killpg
    proc_box: list[Any] = []

    def _recording_killpg(pid: int, sig: int) -> None:
        signals.append((sig, proc_box[0].returncode if proc_box else "?"))
        real_killpg(pid, sig)

    monkeypatch.setattr(os, "killpg", _recording_killpg)
    spawner.start(timeout_s=5)
    proc = spawner._process
    assert proc is not None
    proc_box.append(proc)
    grandchild = int(proc.stdout.readline())
    try:
        spawner.stop()
        assert _wait_dead(grandchild), "the server's grandchild survived stop()"
        assert signals, "stop() never signalled the server's group"
        assert all(rc is None for _sig, rc in signals), f"a group signal after the reap: {signals}"
    finally:
        _kill_group_quietly(proc.pid)
        if _alive(grandchild):
            os.kill(grandchild, signal.SIGKILL)


# ── G10/G11: a closed registry launches nothing; a fake proc is never signalled ─────────────────────────────────


@pytest.mark.xfail(**RED)
def test_g10_a_closed_registry_launches_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert hasattr(cancellation, "spawn_registered") and hasattr(cancellation, "close_child_registry")
    launched: list[bool] = []
    cancellation.close_child_registry()
    assert cancellation.spawn_registered(lambda: launched.append(True), kind="llm_server", stop=lambda: None) is None
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr(lss.subprocess, "Popen", lambda *a, **k: launched.append(True))
    assert lss.LocalServerSpawner(model_path=str(gguf), api_key=KEY).start(timeout_s=1) is None
    assert launched == []


@pytest.mark.xfail(**RED)
def test_g11_a_fake_proc_is_never_signalled(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim import cli_signals

    assert hasattr(cancellation, "spawn_registered")
    sent: list[int] = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: sent.append(pid))
    fake = SimpleNamespace(pid=99999, returncode=None, poll=lambda: None)
    cancellation.spawn_registered(lambda: fake, kind="llm_server", stop=lambda: None)
    cli_signals._kill_spawned_server()
    assert sent == []


# ── G12: a restart closes the registry and stops our own servers, through each owner's stop ─────────────────────


@pytest.mark.xfail(**RED)
def test_g12_a_restart_stops_only_what_this_process_started() -> None:
    assert hasattr(cancellation, "stop_children")
    stopped: list[str] = []
    cancellation.spawn_registered(lambda: SimpleNamespace(pid=1), kind="llm_server", stop=lambda: stopped.append("s"))
    cancellation.spawn_registered(lambda: SimpleNamespace(pid=2), kind="tunnel", stop=lambda: stopped.append("t"))
    cancellation.close_child_registry()
    cancellation.stop_children("llm_server")
    assert stopped == ["s"]
    import inspect

    from maxim.runtime import leader_proxy

    src = inspect.getsource(leader_proxy)
    restart = src[src.index("def _do_restart") : src.index("os.execv(")]
    assert "close_child_registry()" in restart and 'stop_children("llm_server")' in restart
