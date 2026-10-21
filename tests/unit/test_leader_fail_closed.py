"""A leader fails closed (2026-10-14).

Before this, a machine with no role signal became a leader (bound to every
interface), a leader with no key file served every route unauthenticated
(``_check_auth``: ``if not self.api_key: return True``), every leader switched
remote update/install/restart/llm-swap on with ``os.environ.setdefault``, the
``/v1/debug/*`` routes skipped auth for a loopback peer, which through a
cloudflared tunnel (ingress ``http://localhost:<port>``) is the internet, and
the leader's llama-server listened on 0.0.0.0 with its key in argv.

Each test below pins one of those closed, and each was checked by deleting
the mechanism it guards.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from maxim.runtime import leader_proxy as lp
from maxim.runtime.leader_proxy import ProxyAuth, ProxyAuthError, start_leader_proxy
from maxim.tunnel import keys
from maxim.tunnel.keys import LeaderKeyError, require_leader_key
from maxim.utils import http

REPO = Path(__file__).resolve().parents[2]
GOOD_KEY = "k" * 43


def _write_key(content: str, mode: int = 0o600) -> Path:
    path = keys.key_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.chmod(path, mode)
    return path


@pytest.fixture(autouse=True)
def _no_key_file():
    path = keys.key_file_path()
    path.unlink(missing_ok=True)
    yield
    path.unlink(missing_ok=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(auth: ProxyAuth):
    port = _free_port()
    server = start_leader_proxy(auth=auth, bind_host="127.0.0.1", proxy_port=port, upstream_port=_free_port())
    assert server is not None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.02)
    return server, f"http://127.0.0.1:{port}"


def _status(method: str, url: str, *, key: str | None = None, body: dict | None = None) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        resp = http.fetch_url(
            url, method=method, headers=headers, json=body, timeout=http.TimeoutPolicy(connect_s=2, read_s=5, total_s=6)
        )
        return resp.status, resp.json()
    except http.HTTPError as e:
        data = e.response.json() if e.response is not None else {}
        return int(e.status or 0), data


# ── the key, read fail-closed ────────────────────────────────────────────────


def test_a_leader_key_that_is_missing_or_unusable_raises_naming_the_path() -> None:
    with pytest.raises(LeaderKeyError, match="no key file") as exc:
        require_leader_key()
    assert exc.value.path == keys.key_file_path()
    assert "maxim tunnel key rotate" in exc.value.fix_hint

    _write_key("")
    with pytest.raises(LeaderKeyError, match="empty"):
        require_leader_key()
    _write_key("short")
    with pytest.raises(LeaderKeyError, match="shorter"):
        require_leader_key()
    _write_key("k" * 20 + " " + "k" * 20)
    with pytest.raises(LeaderKeyError, match="whitespace"):
        require_leader_key()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_a_leader_key_readable_by_other_users_is_refused() -> None:
    _write_key(GOOD_KEY, mode=0o644)
    with pytest.raises(LeaderKeyError, match="readable by other users") as exc:
        require_leader_key()
    assert "chmod 600" in exc.value.fix_hint


def test_a_good_key_is_returned_with_a_bom_and_newline_stripped() -> None:
    _write_key("\ufeff" + GOOD_KEY + "\n")
    assert require_leader_key() == GOOD_KEY


def test_the_cli_refuses_to_start_a_leader_without_a_key(tmp_path) -> None:
    """End to end through ``python -m maxim``: exit 2, the reason and the key path on stderr, no proxy."""
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO / "src"),
        "MAXIM_ROLE": "leader",
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
        "MAXIM_DATA_HOME": str(tmp_path / "data"),
        "MAXIM_LEADER_PROXY_PORT": str(_free_port()),
        "MAXIM_AUTO_SPAWN_TUNNEL": "0",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "maxim", "--list-models"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stderr[-2000:]
    assert "Refusing to start as leader" in proc.stderr
    assert str(tmp_path / "xdg" / "maxim" / "api_key") in proc.stderr
    assert "LeaderProxy listening" not in proc.stdout + proc.stderr


# ── the proxy's auth is a type, not an Optional key ──────────────────────────


def test_start_leader_proxy_has_no_unauthenticated_default() -> None:
    with pytest.raises(TypeError):
        start_leader_proxy(api_key=None, bind_host="127.0.0.1")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        start_leader_proxy(bind_host="127.0.0.1")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        start_leader_proxy(auth=None, bind_host="127.0.0.1")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ProxyAuth.bearer("")
    with pytest.raises(ValueError):
        ProxyAuth(kind="insecure", key="x")
    assert GOOD_KEY not in repr(ProxyAuth.bearer(GOOD_KEY))


def test_the_in_process_singleton_is_not_handed_out_under_different_auth(monkeypatch) -> None:
    held = SimpleNamespace(RequestHandlerClass=SimpleNamespace(auth=ProxyAuth.insecure_no_auth()))
    monkeypatch.setattr(lp, "_leader_proxy_server", held)
    with pytest.raises(ProxyAuthError):
        start_leader_proxy(auth=ProxyAuth.bearer(GOOD_KEY), bind_host="127.0.0.1", proxy_port=lp.DEFAULT_PROXY_PORT)


# ── loopback is not authentication ───────────────────────────────────────────


def test_debug_routes_need_the_key_even_from_loopback() -> None:
    """A tunnel delivers internet requests from 127.0.0.1, so the old localhost bypass served
    /v1/debug/logs to anyone who knew the hostname."""
    server, base = _serve(ProxyAuth.bearer(GOOD_KEY))
    try:
        for route in ("/v1/debug/logs", "/v1/debug/last-requests", "/v1/debug/deps", "/v1/debug/ping"):
            assert _status("GET", base + route)[0] == 401, route
        assert _status("GET", base + "/v1/debug/ping", key=GOOD_KEY)[0] == 200
    finally:
        server.shutdown()
        server.server_close()


# ── remote admin: bearer AND an explicit opt-in ──────────────────────────────


def test_remote_admin_is_refused_until_the_operator_opts_in(monkeypatch) -> None:
    server, base = _serve(ProxyAuth.bearer(GOOD_KEY))
    try:
        for route in ("update", "install", "restart", "llm-swap"):
            code, body = _status("POST", f"{base}/v1/admin/{route}", key=GOOD_KEY, body={})
            assert code == 403, route
            assert "proxy.allow_remote_update" in body["error"], route
        monkeypatch.setenv("MAXIM_PROXY_ALLOW_REMOTE_UPDATE", "1")
        # Past the gate: llm-swap with no model is a 400 from the handler, not the 403 from the gate.
        assert _status("POST", f"{base}/v1/admin/llm-swap", key=GOOD_KEY, body={})[0] == 400
    finally:
        server.shutdown()
        server.server_close()


def test_the_opt_in_reads_config_json(monkeypatch) -> None:
    from maxim.runtime.config_loader import config_path, reset_config_cache

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"_format_version": "1.3", "proxy": {"allow_remote_update": True}}))
    reset_config_cache()
    try:
        assert lp._remote_update_allowed() is True
    finally:
        path.unlink()
        reset_config_cache()
    assert lp._remote_update_allowed() is False


def test_an_insecure_proxy_never_exposes_admin_even_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("MAXIM_PROXY_ALLOW_REMOTE_UPDATE", "1")
    server, base = _serve(ProxyAuth.insecure_no_auth())
    try:
        code, body = _status("POST", f"{base}/v1/admin/restart", body={})
        assert code == 403
        assert "bearer" in body["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_nothing_switches_remote_update_on_for_leaders() -> None:
    for rel in ("src/maxim/cli.py", "src/maxim/runtime/lane_backends.py"):
        assert 'setdefault("MAXIM_PROXY_ALLOW_REMOTE_UPDATE"' not in (REPO / rel).read_text(), rel


# ── the leader entry point: a held port must be an authenticated LeaderProxy ──


def test_a_port_held_by_something_else_is_refused(monkeypatch) -> None:
    _write_key(GOOD_KEY)
    squatter = socket.socket()
    squatter.bind(("127.0.0.1", 0))
    squatter.listen()
    monkeypatch.setenv("MAXIM_LEADER_PROXY_PORT", str(squatter.getsockname()[1]))
    try:
        with pytest.raises(ProxyAuthError):
            lp.start_leader_proxy_for_role("127.0.0.1")
    finally:
        squatter.close()


def test_a_port_held_by_an_authenticated_leader_proxy_is_shared(monkeypatch) -> None:
    _write_key(GOOD_KEY)
    server, base = _serve(ProxyAuth.bearer(GOOD_KEY))
    monkeypatch.setenv("MAXIM_LEADER_PROXY_PORT", base.rsplit(":", 1)[1])
    try:
        assert lp.start_leader_proxy_for_role("127.0.0.1") is None
    finally:
        server.shutdown()
        server.server_close()


def test_the_leader_entry_point_reads_the_key_fail_closed() -> None:
    with pytest.raises(LeaderKeyError):
        lp.start_leader_proxy_for_role("127.0.0.1")


# ── role: leader is never a default ──────────────────────────────────────────


def test_a_box_with_a_mesh_key_that_lands_on_the_default_role_is_warned(caplog) -> None:
    from maxim.runtime import role

    _write_key(GOOD_KEY)
    with caplog.at_level("WARNING", logger="maxim.runtime.role"):
        role.apply_role("solo", "default")
    assert any("maxim config set role leader" in r.getMessage() for r in caplog.records) or any(
        getattr(r, "event", "") == "role_default_with_mesh_key" for r in caplog.records
    )


# ── the leader's llama-server: loopback, key in env, checked from outside ────


def test_auto_spawn_for_a_leader_without_a_key_raises(monkeypatch, tmp_path) -> None:
    from maxim.runtime import lane_backends

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setenv("MAXIM_ROLE", "leader")
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(server=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "llama_cpp.server", SimpleNamespace())
    monkeypatch.setattr(lane_backends, "_read_persisted_model", lambda: None)
    monkeypatch.setattr(
        "maxim.models.language.config.load_llm_config", lambda **_: SimpleNamespace(model_path=str(gguf))
    )
    lanes = {"large": SimpleNamespace(remote_url=None, model_profile="mistral-7b", placement=())}
    with pytest.raises(LeaderKeyError):
        lane_backends._maybe_auto_spawn_server(SimpleNamespace(has_gpu=True), lanes, None)


def test_the_swap_server_binds_loopback_only(monkeypatch) -> None:
    from maxim.runtime import lane_backends
    from maxim.runtime import local_server_spawner as lss

    built: list[dict] = []

    class _Spawner:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def start(self, timeout_s):
            return "http://127.0.0.1:8100/v1"

    monkeypatch.setattr(lss, "LocalServerSpawner", _Spawner)
    monkeypatch.setenv("MAXIM_LLM_N_CTX", "4096")
    monkeypatch.setenv("MAXIM_ROLE", "leader")
    cfg = SimpleNamespace(n_gpu_layers=-1, kv_quant_mode="f16")
    lane_backends._start_swap_server(cfg=cfg, model_path="/m.gguf", api_key=GOOD_KEY, port=8100, logger_obj=None)
    assert built[0]["bind_host"] == "127.0.0.1"
    assert built[0]["api_key"] == GOOD_KEY


def test_spawn_and_swap_decide_the_key_through_one_role_source(monkeypatch, tmp_path) -> None:
    """A Python-API leader (nothing exported MAXIM_ROLE) must key a hot-swapped server like the first one."""
    from maxim.runtime import lane_backends

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))  # the solo half creates a key file

    monkeypatch.setattr("maxim.runtime.role.resolved_role", lambda: ("leader", "config_json"))
    with pytest.raises(LeaderKeyError):
        lane_backends._server_key_for_role()
    _write_key(GOOD_KEY)
    assert lane_backends._server_key_for_role() == GOOD_KEY
    monkeypatch.setattr("maxim.runtime.role.resolved_role", lambda: ("solo", "default"))
    local = lane_backends._server_key_for_role()  # #1314: every role's server is keyed; a solo's has its own key
    assert local and local != GOOD_KEY
    src = (REPO / "src/maxim/runtime/lane_backends.py").read_text()
    assert src.count("api_key = _server_key_for_role()") == 1  # the first spawn; a swap reuses its key (_swap_key)
    assert src.count("api_key = _swap_key()") == 1
    assert "require_leader_key()" not in src.replace("    return require_leader_key()", "")


def test_exposure_check_refuses_a_keyed_server_that_answers_without_the_key(monkeypatch) -> None:
    from maxim.runtime import local_server_spawner as lss

    monkeypatch.setattr(http, "fetch_url", lambda *a, **k: SimpleNamespace(status=200))
    monkeypatch.setattr("maxim.utils.net.primary_ipv4_address", lambda: None)
    with pytest.raises(lss.LLMServerExposureError, match="without auth"):
        lss.verify_llm_server_exposure(8100, allow_authenticated_lan=False)


def _refused(*a, **k):
    raise http.HTTPAuthError("401", status=401)


def test_exposure_check_refuses_a_server_reachable_on_the_lan(monkeypatch) -> None:
    from maxim.runtime import local_server_spawner as lss

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    monkeypatch.setattr("maxim.utils.net.primary_ipv4_address", lambda: "127.0.0.1")  # stands in for the LAN address
    try:
        monkeypatch.setattr(http, "fetch_url", lambda *a, **k: SimpleNamespace(status=200))
        with pytest.raises(lss.LLMServerExposureError, match="answered an unauthenticated request"):
            lss.verify_llm_server_exposure(port, allow_authenticated_lan=True)
        monkeypatch.setattr(http, "fetch_url", _refused)
        with pytest.raises(lss.LLMServerExposureError, match="loopback only"):
            lss.verify_llm_server_exposure(port, allow_authenticated_lan=False)
        lss.verify_llm_server_exposure(port, allow_authenticated_lan=True)  # reuse: warned
    finally:
        listener.close()


def test_a_fresh_spawn_that_fails_the_exposure_check_is_stopped_and_raises(monkeypatch, tmp_path) -> None:
    from maxim.runtime import local_server_spawner as lss

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    spawner = lss.LocalServerSpawner(model_path=str(gguf), api_key=GOOD_KEY)
    stopped: list[bool] = []
    monkeypatch.setattr(lss.subprocess, "Popen", lambda *a, **k: SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(spawner, "_wait_ready", lambda _t: True)
    monkeypatch.setattr(spawner, "stop", lambda: stopped.append(True))

    def _exposed(*a, **k):
        raise lss.LLMServerExposureError("exposed")

    monkeypatch.setattr(lss, "verify_llm_server_exposure", _exposed)
    with pytest.raises(lss.LLMServerExposureError):
        spawner.start(timeout_s=1)
    assert stopped == [True]


def test_an_error_response_closes_the_connection_so_an_unread_body_cannot_become_a_request() -> None:
    """A refusal is sent before the body is read; on a kept-alive connection that body would be parsed as
    the next request line (seen as a 501 on a pooled client). The server must close after an error."""
    server, base = _serve(ProxyAuth.bearer(GOOD_KEY))
    port = int(base.rsplit(":", 1)[1])
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
            body = b"{}"
            conn.sendall(
                b"POST /v1/admin/restart HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer "
                + GOOD_KEY.encode()
                + b"\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n"
                + body
            )
            received = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break  # the server closed: what this test asserts
                received += chunk
        assert received.startswith(b"HTTP/1.1 403")
        assert b"connection: close" in received.lower()
    finally:
        server.shutdown()
        server.server_close()


def test_doctor_fails_a_leader_that_could_not_start_and_reports_remote_admin(monkeypatch) -> None:
    from maxim.doctor.checks import check_api_key, check_remote_admin

    monkeypatch.setenv("MAXIM_ROLE", "leader")
    result = check_api_key()
    assert result.status == "fail" and "no key file" in result.message
    assert GOOD_KEY not in (result.message + (result.fix or ""))
    _write_key(GOOD_KEY)
    assert check_api_key().status == "ok"
    assert "disabled" in check_remote_admin().message
    monkeypatch.setenv("MAXIM_PROXY_ALLOW_REMOTE_UPDATE", "1")
    assert "enabled (source=env)" in check_remote_admin().message


# ── review folds ─────────────────────────────────────────────────────────────


def test_a_held_port_is_probed_without_the_key(monkeypatch) -> None:
    """A local user who binds the proxy port first must not receive the mesh key."""
    import http.server
    import threading

    seen: list[str | None] = []

    class _Squatter(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen.append(self.headers.get("Authorization"))
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Squatter)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _write_key(GOOD_KEY)
    monkeypatch.setenv("MAXIM_LEADER_PROXY_PORT", str(srv.server_address[1]))
    try:
        assert lp.start_leader_proxy_for_role("127.0.0.1") is None  # demands auth: shared, with a warning
    finally:
        srv.shutdown()
        srv.server_close()
    assert seen and all(h is None for h in seen)


def test_the_legacy_opt_in_env_is_ignored(monkeypatch, caplog) -> None:
    """Older leaders setdefault-ed MAXIM_ALLOW_REMOTE_UPDATE=1 and a remote restart (os.execv) inherits it."""
    monkeypatch.setenv("MAXIM_ALLOW_REMOTE_UPDATE", "1")
    assert lp._remote_update_allowed() is False
    with caplog.at_level("WARNING", logger="maxim.runtime.leader_proxy"):
        lp._warn_legacy_remote_update_env()
    assert any("IGNORED" in r.getMessage() for r in caplog.records)
    monkeypatch.setenv("MAXIM_PROXY_ALLOW_REMOTE_UPDATE", "1")
    assert lp._remote_update_allowed() is True


def test_security_refusals_are_not_degraded_to_running_without_the_llm(monkeypatch) -> None:
    from maxim.exceptions import LeaderRefusal
    from maxim.utils.logging import warn_unless_refusal
    from maxim.runtime import lane_backends
    from maxim.runtime.local_server_spawner import LLMServerExposureError
    from maxim.simulation import research_orchestrator as ro

    for exc_type in (LeaderKeyError, ProxyAuthError, LLMServerExposureError):
        assert issubclass(exc_type, LeaderRefusal) and issubclass(exc_type, RuntimeError)
    with pytest.raises(LeaderRefusal):
        warn_unless_refusal("x %s", LeaderKeyError(path=Path("/k"), reason="no key file"))
    warn_unless_refusal("x %s", ValueError("ordinary"))  # an ordinary failure is reported, not raised
    for rel in ("src/maxim/embodied_runtime/media_loop.py", "src/maxim/embodied_runtime/agentic_runtime.py"):
        text = (REPO / rel).read_text()
        assert 'warn_unless_refusal("Failed to' in text, rel  # the robot runtime's LLM/agentic starts

    def _refuse(**_):
        raise LeaderKeyError(path=Path("/k"), reason="no key file")

    monkeypatch.setattr(lane_backends, "build_primary_router", _refuse)
    with pytest.raises(LeaderKeyError):
        ro._build_research_llm()
    cli_src = (REPO / "src/maxim/cli.py").read_text()
    server_mode = cli_src[cli_src.index('if mode == "server":') :]
    assert server_mode.index("except LeaderRefusal") < server_mode.index("except Exception as _srv_err")


def test_oasis_serve_reads_an_existing_key_strictly(tmp_path, capsys, monkeypatch) -> None:
    from maxim.hivemind.oasis_cli import run_oasis_subcommand

    started: list = []
    monkeypatch.setattr("maxim.runtime.leader_proxy.start_leader_proxy", lambda **k: started.append(k) or None)
    _write_key(GOOD_KEY, mode=0o644)
    assert run_oasis_subcommand(["serve", "--root", str(tmp_path / "s"), "--bind-host", "127.0.0.1"]) == 2
    assert started == []
    assert "readable by other users" in capsys.readouterr().err


def test_doctor_runs_the_same_exposure_check_as_startup(monkeypatch) -> None:
    from maxim.doctor import checks
    from maxim.runtime import local_server_spawner as lss

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    calls: list[tuple] = []

    def _exposed(p, *, allow_authenticated_lan):
        calls.append((p, allow_authenticated_lan))
        raise lss.LLMServerExposureError("answered an unauthenticated request")

    monkeypatch.setattr(lss, "verify_llm_server_exposure", _exposed)
    try:
        result = checks.check_llm_server_exposure(port)
    finally:
        listener.close()
    assert result.status == "fail" and calls == [(port, False)]


def test_only_oasis_serve_builds_an_insecure_proxy_and_nothing_writes_the_opt_in() -> None:
    import re

    insecure, writes = [], []
    write_re = re.compile(
        r"(environ\[\s*[\"']MAXIM_(PROXY_)?ALLOW_REMOTE_UPDATE|setdefault\(\s*[\"']MAXIM_(PROXY_)?ALLOW_REMOTE_UPDATE"
        r"|putenv\(\s*[\"']MAXIM_(PROXY_)?ALLOW_REMOTE_UPDATE)"
    )
    for path in (REPO / "src" / "maxim").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO).as_posix()
        if "insecure_no_auth()" in text and rel not in (
            "src/maxim/runtime/leader_proxy.py",
            "src/maxim/hivemind/oasis_cli.py",
        ):
            insecure.append(rel)
        writes += [f"{rel}: {m.group(0)}" for m in write_re.finditer(text)]
    assert insecure == []
    assert writes == []


def test_peer_llm_status_sends_the_key(monkeypatch) -> None:
    from maxim.peer import cli as peer_cli

    sent: list[dict] = []
    monkeypatch.setattr(
        peer_cli, "read_peer_config", lambda: SimpleNamespace(url="https://leader.example/v1", api_key=GOOD_KEY)
    )

    def _fetch(url, **kw):
        sent.append(kw.get("headers") or {})
        raise http.HTTPConnectionError("x")

    monkeypatch.setattr(http, "fetch_url", _fetch)
    peer_cli._cmd_llm_status([])
    assert sent and sent[0].get("Authorization") == f"Bearer {GOOD_KEY}"


def test_doctor_exposure_and_remote_admin_edges(monkeypatch) -> None:
    from maxim.doctor import checks
    from maxim.runtime import local_server_spawner as lss

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    monkeypatch.setattr(lss, "verify_llm_server_exposure", lambda *a, **k: None)
    monkeypatch.setenv("MAXIM_AUTO_SPAWN_PORT", "not-a-port")
    try:
        assert checks.check_llm_server_exposure(port).status == "ok"
        assert checks.check_llm_server_exposure().name == "LLM server exposure"  # bad env port falls back to 8100
    finally:
        listener.close()
    monkeypatch.setenv("MAXIM_ROLE", "leader")
    monkeypatch.setenv("MAXIM_PROXY_ALLOW_REMOTE_UPDATE", "maybe")
    assert checks.check_remote_admin().status == "warn"  # malformed opt-in: refused, said so
    monkeypatch.delenv("MAXIM_PROXY_ALLOW_REMOTE_UPDATE")
    monkeypatch.setenv("MAXIM_ALLOW_REMOTE_UPDATE", "1")
    row = checks.check_remote_admin()
    assert row.status == "warn" and "IGNORED" in row.message and "disabled" in row.message


def test_primary_ipv4_address_never_reports_loopback_or_raises(monkeypatch) -> None:
    from maxim.utils import net

    class _Sock:
        def __init__(self, *a, addr="127.0.0.1", fail=False):
            self._addr, self._fail = addr, fail

        def connect(self, _):
            if self._fail:
                raise OSError("no route")

        def getsockname(self):
            return (self._addr, 0)

        def close(self):
            pass

    monkeypatch.setattr(net.socket, "socket", lambda *a: _Sock())
    assert net.primary_ipv4_address() is None  # loopback is not a LAN address
    monkeypatch.setattr(net.socket, "socket", lambda *a: _Sock(addr="192.168.1.20"))
    assert net.primary_ipv4_address() == "192.168.1.20"
    monkeypatch.setattr(net.socket, "socket", lambda *a: _Sock(fail=True))
    assert net.primary_ipv4_address() is None

    def _no_socket(*a):
        raise OSError("no sockets")

    monkeypatch.setattr(net.socket, "socket", _no_socket)
    assert net.primary_ipv4_address() is None


def test_an_unreadable_leader_key_is_refused_not_treated_as_absent(monkeypatch) -> None:
    path = _write_key(GOOD_KEY)

    def _boom(*a, **k):
        raise PermissionError("denied")

    monkeypatch.setattr(type(path), "read_text", _boom)
    monkeypatch.setattr(keys.os, "read", _boom)  # #1311: the reader reads its one descriptor
    with pytest.raises(LeaderKeyError, match="cannot read key file"):
        require_leader_key()


def test_key_file_failures_are_typed_or_none_never_a_crash(monkeypatch, tmp_path) -> None:
    """The leader reader refuses an unstattable key; the lenient reader (non-leader callers) returns None."""
    path = _write_key(GOOD_KEY)
    real_stat = type(path).stat

    def _stat_fails(self, *a, **k):
        if self == path:
            raise PermissionError("denied")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(type(path), "stat", _stat_fails)
    # #1312: the reader opens and fstats one descriptor (never Path.is_file, which swallows the error on 3.14), so the
    # unstat-able path is denied at the os layer too, as an unreadable parent directory does.
    from tests.unit._key_file_helpers import deny_os_access

    deny_os_access(monkeypatch, path)
    with pytest.raises(LeaderKeyError, match="cannot stat key file"):
        require_leader_key()
    monkeypatch.undo()

    def _read_fails(*a, **k):
        raise OSError("io")

    monkeypatch.setattr(type(path), "read_text", _read_fails)
    monkeypatch.setattr(keys.os, "read", _read_fails)  # #1311: the reader reads its one descriptor
    assert keys.read_key() is None
    monkeypatch.undo()

    def _chmod_fails(*a, **k):
        raise OSError("ro fs")

    monkeypatch.setattr(keys.os, "chmod", _chmod_fails)
    assert keys.write_key(GOOD_KEY) == path  # written; the best-effort chmod failing is not fatal


def test_key_file_path_on_windows_uses_appdata(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(keys.platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert keys.key_file_path() == tmp_path / "maxim" / "api_key"
    monkeypatch.delenv("APPDATA")
    assert keys.key_file_path().parts[-3:] == ("Roaming", "maxim", "api_key")
