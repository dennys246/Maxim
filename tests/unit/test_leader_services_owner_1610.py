"""#1610: a leader's services have ONE owner: the ``maxim`` CLI leader process that binds the default-port proxy.

Owner decisions (2026-11-09):
- One role-keyed owner starts the cloudflared tunnel daemon, the LeaderProxy and the leader heartbeat, whatever the
  LLM path decides: a fresh spawn, a reused server, an opt-out, or a build that raises.
- The LLM spawn path starts none of them.
- Only the CLI leader owns them; a Python-API process and ``maxim serve`` never do.

Before this, auto-spawn started them only after a FRESH spawn. So a leader that reused the server already on its port
came up without its tunnel and heartbeat (#1610), and any process that fresh-spawned as a leader (a Python-API one, or
the console) started them.

Offline throughout: the proxy, the tunnel and the heartbeat are recorded, never started.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from maxim.runtime import lane_backends
from maxim.runtime import leader_mode
from maxim.runtime import leader_proxy
from maxim.runtime import llm_server
from maxim.runtime import local_server_spawner as lss
from maxim.runtime.worker_pool import LaneConfig
from maxim.tunnel import daemon_spawner

KEY = "test-server-key-0123456789"
ROOT = Path(__file__).resolve().parents[2]
LEADER = SimpleNamespace(role="leader", bind_host="127.0.0.1", reason="test")


@pytest.fixture
def services(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """No claim yet; the tunnel and the heartbeat only record."""
    started: list[str] = []
    monkeypatch.setattr(leader_mode, "_claimed_by_pid", None)
    monkeypatch.setattr(leader_mode, "_started_by_pid", None)
    monkeypatch.setattr(daemon_spawner, "auto_spawn_tunnel_daemon", lambda log: started.append("tunnel"))
    monkeypatch.setattr(
        "maxim.runtime.heartbeat.get_heartbeat_monitor",
        lambda: SimpleNamespace(start=lambda: started.append("heartbeat")),
    )
    return started


def _proxy(monkeypatch: pytest.MonkeyPatch, *, bound_here: bool, started: list[str]) -> None:
    """``start_leader_proxy_for_role`` as it behaves when this process binds the default port (it records the
    singleton), or when an authenticated proxy elsewhere holds it (None, no singleton)."""
    monkeypatch.setattr(leader_proxy, "_leader_proxy_server", None)

    def start(bind_host: str) -> Any:
        started.append(f"proxy@{bind_host}")
        if bound_here:
            server = SimpleNamespace(name="proxy")
            monkeypatch.setattr(leader_proxy, "_leader_proxy_server", server)
            return server
        return None

    monkeypatch.setattr(leader_proxy, "start_leader_proxy_for_role", start)


def _claimed(monkeypatch: pytest.MonkeyPatch, services: list[str]) -> None:
    _proxy(monkeypatch, bound_here=True, started=services)
    leader_mode.claim_leader_services("127.0.0.1")
    assert leader_mode.owns_leader_services()


# ─── the claim ────────────────────────────────────────────────────────────────────────────────────


def test_binding_the_default_port_proxy_claims_the_services(monkeypatch, services):
    _claimed(monkeypatch, services)


def test_sharing_another_processes_proxy_claims_nothing(monkeypatch, services):
    """A sub-sim on the leader box finds the leader's proxy on the port: the leader keeps its tunnel."""
    _proxy(monkeypatch, bound_here=False, started=services)
    leader_mode.claim_leader_services("127.0.0.1")
    assert not leader_mode.owns_leader_services()
    leader_mode.start_claimed_leader_services(None)
    assert services == ["proxy@127.0.0.1"]


def test_a_proxy_on_a_non_default_port_claims_nothing(monkeypatch, services):
    """``start_leader_proxy_for_role`` returns a server but records no singleton (a harness child's own port)."""
    monkeypatch.setattr(leader_proxy, "_leader_proxy_server", None)
    monkeypatch.setattr(leader_proxy, "start_leader_proxy_for_role", lambda bind_host: SimpleNamespace(name="p"))
    leader_mode.claim_leader_services("127.0.0.1")
    assert not leader_mode.owns_leader_services()


def test_a_forked_child_does_not_inherit_the_claim(monkeypatch, services):
    _claimed(monkeypatch, services)
    real_pid = os.getpid()
    monkeypatch.setattr(leader_mode.os, "getpid", lambda: real_pid + 1)
    assert not leader_mode.owns_leader_services()
    leader_mode.start_claimed_leader_services(None)
    assert "tunnel" not in services


def test_the_owner_starts_the_tunnel_and_heartbeat_once(monkeypatch, services):
    _claimed(monkeypatch, services)
    for _ in range(3):  # the orchestrator builds its routers more than once per process
        leader_mode.start_claimed_leader_services(None)
    assert services == ["proxy@127.0.0.1", "tunnel", "heartbeat"]


def test_a_tunnel_start_that_raises_is_logged_and_the_heartbeat_still_starts(monkeypatch, services):
    """E.g. a PermissionError reading /etc/cloudflared: the owner's router build must not fail on its tunnel."""
    _claimed(monkeypatch, services)

    def denied(log: Any) -> None:
        raise PermissionError(13, "Permission denied", "/etc/cloudflared/config.yml")

    monkeypatch.setattr(daemon_spawner, "auto_spawn_tunnel_daemon", denied)
    leader_mode.start_claimed_leader_services(None)
    assert services == ["proxy@127.0.0.1", "heartbeat"]


def test_the_early_leader_boot_claims(monkeypatch, services):
    import maxim.cli as cli

    _proxy(monkeypatch, bound_here=True, started=services)
    monkeypatch.setattr("maxim.runtime.leader_mode.detect_role", lambda: LEADER)
    assert cli._early_leader_boot() is None
    assert leader_mode.owns_leader_services()


def test_maxim_serve_on_a_leader_never_claims_or_binds_the_proxy(monkeypatch, services):
    """Owner decision: only the CLI leader owns a leader's services; the console shares, never binds :8099."""
    from maxim.console import server

    _proxy(monkeypatch, bound_here=True, started=services)
    monkeypatch.setattr("maxim.runtime.leader_mode.detect_role", lambda: LEADER)
    monkeypatch.setenv("MAXIM_ROLE", "leader")

    def fake_build_app(*a: Any, **k: Any) -> Any:
        raise SystemExit(0)  # stop before uvicorn

    monkeypatch.setattr(server, "build_app", fake_build_app)
    monkeypatch.setattr("maxim.tunnel.keys.ensure_console_token", lambda: "console-token-0123456789")
    with pytest.raises(SystemExit):
        server.run_serve(["--port", "8799"])
    assert services == []
    assert not leader_mode.owns_leader_services()


# ─── the tunnel starter ──────────────────────────────────────────────────────────────────────────


def test_the_tunnel_opt_out_starts_nothing(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(daemon_spawner, "cloudflared_already_running", lambda: seen.append("pgrep") or False)
    monkeypatch.setattr(daemon_spawner, "resolve_config_path", lambda: seen.append("config") or None)
    monkeypatch.setenv("MAXIM_AUTO_SPAWN_TUNNEL", "0")
    daemon_spawner.auto_spawn_tunnel_daemon(None)
    assert seen == []
    monkeypatch.delenv("MAXIM_AUTO_SPAWN_TUNNEL")
    daemon_spawner.auto_spawn_tunnel_daemon(None)
    assert seen == ["pgrep", "config"]


# ─── the router build starts them for the owner, whatever the LLM path decides ────────────────────


@pytest.fixture
def auto_spawn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """``_maybe_auto_spawn_server`` offline, with the singleton decision chosen by the test."""
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(server=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "llama_cpp.server", SimpleNamespace())
    for var in ("MAXIM_AUTO_SPAWN_LLM_SERVER", "MAXIM_LLM_PROFILE", "MAXIM_LANE_LARGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MAXIM_AUTO_SPAWN_PORT", "8100")
    monkeypatch.setattr(lane_backends, "_read_persisted_model", lambda: None)
    monkeypatch.setattr(lane_backends, "_server_key_for_role", lambda: KEY)
    monkeypatch.setattr(lane_backends, "_lane_profile_is_cloud", lambda *a: False)
    monkeypatch.setattr(lane_backends, "_check_vram_spillover_risk", lambda **_: None)
    monkeypatch.setattr(
        "maxim.models.language.config.load_llm_config", lambda **_: SimpleNamespace(model_path=str(gguf))
    )
    monkeypatch.setattr(lss, "verify_llm_server_exposure", lambda *a, **k: None)
    monkeypatch.setattr(llm_server, "_active_spawner", None)
    monkeypatch.setattr(llm_server, "set_active_spawner", lambda s: None)
    monkeypatch.setattr("maxim.runtime.leader_mode.detect_role", lambda: LEADER)

    class _Spawner:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def start(self, timeout_s: float) -> str:
            return "http://127.0.0.1:8100/v1"

    monkeypatch.setattr(lss, "LocalServerSpawner", _Spawner)

    def run(decision: str) -> dict[str, Any]:
        monkeypatch.setattr(lane_backends, "_existing_server_decision", lambda *a, **k: decision)
        lanes = {"large": LaneConfig(name="large", max_workers=1, model_profile="mistral-7b")}
        return lane_backends._maybe_auto_spawn_server(SimpleNamespace(has_gpu=True, vram_gb=16.0), lanes, None)

    return run


@pytest.mark.parametrize("decision", ["reuse", "spawn"])
def test_auto_spawn_starts_no_leader_service_on_any_path(monkeypatch, services, auto_spawn, decision):
    """Even for a claimed leader: the LLM path decides only the LLM (the reuse arm skipped the tunnel, #1610)."""
    _claimed(monkeypatch, services)
    services.clear()
    out = auto_spawn(decision)
    assert out["large"].remote_url == "http://127.0.0.1:8100/v1"
    assert services == []


@pytest.mark.parametrize("fails_at", ["_require_valid_user_profiles", "_project_lane_env_for_build"])
def test_the_build_starts_the_owners_services_first_even_when_it_raises(monkeypatch, services, fails_at):
    """A server-mode leader stays up behind its proxy when its LLM fails (#1338), including on an invalid profile;
    its tunnel must be up for that remote recovery."""
    _claimed(monkeypatch, services)

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("the LLM path failed")

    monkeypatch.setattr(lane_backends, "_require_valid_user_profiles", lambda: None)
    monkeypatch.setattr(lane_backends, fails_at, boom)
    with pytest.raises(RuntimeError, match="the LLM path failed"):
        lane_backends.build_primary_router(SimpleNamespace(has_gpu=True), logger=None)
    assert services == ["proxy@127.0.0.1", "tunnel", "heartbeat"]


def test_a_python_api_build_starts_no_leader_service(monkeypatch, services):
    """No claim (the API never runs the early boot): the router build starts nothing for a leader role."""

    def boom(*a: Any, **k: Any) -> Any:
        raise KeyError("stop")

    monkeypatch.setattr(lane_backends, "_require_valid_user_profiles", boom)
    monkeypatch.setenv("MAXIM_ROLE", "leader")
    with pytest.raises(KeyError):
        lane_backends.build_primary_router(SimpleNamespace(has_gpu=True), logger=None)
    assert services == []


# ─── structure: one owner, enforced ──────────────────────────────────────────────────────────────


def _names_outside(name: str, allowed: set[str]) -> list[str]:
    """Every Name / Attribute / import alias spelling ``name`` in src/, outside ``allowed`` files."""
    found: list[str] = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=rel)):
            if (
                (isinstance(node, ast.Name) and node.id == name)
                or (isinstance(node, ast.Attribute) and node.attr == name)
                or (isinstance(node, ast.alias) and node.name == name)
            ):
                found.append(f"{rel}:{getattr(node, 'lineno', '?')}")
    return found


def test_only_the_tunnel_starter_builds_a_tunnel_daemon():
    """``TunnelDaemonSpawner`` is built in one module; auto-spawn and the router build never name it."""
    assert _names_outside("TunnelDaemonSpawner", {"src/maxim/tunnel/daemon_spawner.py"}) == []


def test_only_the_owner_starts_the_tunnel():
    allowed = {"src/maxim/tunnel/daemon_spawner.py", "src/maxim/runtime/leader_mode.py"}
    assert _names_outside("auto_spawn_tunnel_daemon", allowed) == []


def test_only_the_owner_starts_the_leader_role_proxy():
    """``start_leader_proxy_for_role`` is reached only through ``leader_mode.claim_leader_services``. The sanctioned
    exception is ``maxim oasis serve``, which calls ``start_leader_proxy`` itself with its own auth (not this name)."""
    allowed = {"src/maxim/runtime/leader_proxy.py", "src/maxim/runtime/leader_mode.py"}
    assert _names_outside("start_leader_proxy_for_role", allowed) == []


def _starts_the_heartbeat(node: ast.AST) -> bool:
    """``get_heartbeat_monitor().start()`` (bare or module-qualified)."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "start"):
        return False
    inner = node.func.value
    if not isinstance(inner, ast.Call):
        return False
    fn = inner.func
    return (isinstance(fn, ast.Name) and fn.id == "get_heartbeat_monitor") or (
        isinstance(fn, ast.Attribute) and fn.attr == "get_heartbeat_monitor"
    )


def test_the_heartbeat_is_started_from_two_places_only():
    """The chained ``get_heartbeat_monitor().start()`` form: the leader owner's (``leader_mode``) and the env-enabled
    one (``lane_backends._start_heartbeat_if_enabled``). The console's own sampling start
    (``console/health.py::_acquire_monitor``, ``monitor.start()`` on a bound name) is outside this scan and outside
    #1610: it starts no leader service."""
    found = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        found += [rel for node in ast.walk(tree) if _starts_the_heartbeat(node)]
    assert sorted(found) == ["src/maxim/runtime/lane_backends.py", "src/maxim/runtime/leader_mode.py"]
