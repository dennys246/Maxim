"""#1314: every spawned llama-cpp-server requires a key, solo included.

Since 1.3.2 a leader's server is keyed with the mesh key (#1307). A solo's server bound loopback with no key, so any
local process or user on a shared host could use its inference, and a stale tunnel ingress pointing at :8100 would
expose it unauthenticated. A non-leader now gets a per-host key file (``llm_server_key``, 0600), created only when
missing, under a lock, and read back from disk; a file that exists but is unusable is refused, never overwritten.
Owner decisions 2026-10-20: key every spawned server; a 401 to our own key is NOT ready (fail loud); a read-only
config dir is a typed refusal naming the fix; an old unkeyed server is refused, not killed.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

POSIX = pytest.mark.skipif(platform.system() == "Windows", reason="POSIX file modes")


@pytest.fixture
def solo(tmp_path: Path, monkeypatch) -> Path:
    """A solo process with its own config dir (conftest's session-wide XDG dir would carry a key between tests)."""
    from maxim.runtime import role

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("MAXIM_ROLE", "solo")
    monkeypatch.setattr(role, "_APPLIED", None)
    return tmp_path / "xdg" / "maxim"


@POSIX
def test_a_solo_spawn_gets_a_key_file_created_0600(solo: Path) -> None:
    from maxim.runtime.lane_backends import _server_key_for_role

    key = _server_key_for_role()
    assert isinstance(key, str) and len(key) >= 16
    path = solo / "llm_server_key"
    assert path.read_text().strip() == key
    assert (path.stat().st_mode & 0o777) == 0o600
    assert _server_key_for_role() == key  # stable: read back, not regenerated


def test_a_reader_never_creates_the_key(solo: Path) -> None:
    """Doctor and the o19 harness read the key; creating it would leave a key no running server holds."""
    from maxim.tunnel import keys

    assert keys.read_local_server_key() is None
    assert not (solo / "llm_server_key").exists()


def test_concurrent_first_starts_agree_on_one_key(solo: Path, tmp_path: Path) -> None:
    """Two processes starting together must not each generate a key (the server would hold one, the file another)."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from maxim.tunnel.keys import ensure_local_server_key; print(ensure_local_server_key())"
    )
    src = str(Path(__file__).resolve().parents[2] / "src")
    env = {**os.environ}
    procs = [
        subprocess.Popen([sys.executable, "-c", code, src], env=env, stdout=subprocess.PIPE, text=True)
        for _ in range(8)
    ]
    keys_seen = {p.communicate(timeout=60)[0].strip() for p in procs}
    assert all(p.returncode == 0 for p in procs)
    assert len(keys_seen) == 1
    assert (solo / "llm_server_key").read_text().strip() == keys_seen.pop()


@POSIX
def test_an_unusable_key_file_is_refused_not_overwritten(solo: Path) -> None:
    from maxim.tunnel import keys

    solo.mkdir(parents=True)
    path = solo / "llm_server_key"
    path.write_text("x" * 43)
    path.chmod(0o644)
    before = path.read_bytes()
    with pytest.raises(keys.LeaderKeyError) as err:
        keys.ensure_local_server_key()
    assert "chmod 600" in err.value.fix_hint and "leader" not in str(err.value).lower()
    assert path.read_bytes() == before


@POSIX
def test_an_unwritable_config_dir_is_a_typed_refusal(solo: Path) -> None:
    from maxim.tunnel import keys

    if os.geteuid() == 0:
        pytest.skip("root ignores directory modes")
    solo.mkdir(parents=True)
    solo.chmod(0o500)
    try:
        with pytest.raises(keys.LeaderKeyError) as err:
            keys.ensure_local_server_key()
        assert "XDG_CONFIG_HOME" in err.value.fix_hint
    finally:
        solo.chmod(0o700)


def test_a_401_to_our_own_key_is_not_ready() -> None:
    """A 401 to the key we spawned with means another server holds the port, or the key never reached the child."""
    from unittest.mock import patch

    from maxim.runtime.local_server_spawner import LocalServerSpawner
    from maxim.utils import http as _http

    s = LocalServerSpawner(model_path="/tmp/fake.gguf", api_key="our-key-0123456789")
    with patch(
        "maxim.utils.http.fetch_url",
        side_effect=_http.HTTPAuthError("_external", status=401, fix_hint="Auth rejected (401)"),
    ):
        assert s._health_check() is not True


def test_start_fails_at_once_on_a_401_and_stops_the_child(tmp_path: Path, monkeypatch) -> None:
    """Fail loud, not after the readiness timeout: the first 401 raises out of ``start``, which stops its child."""
    import time
    from unittest.mock import MagicMock, patch

    from maxim.runtime import local_server_spawner as lss
    from maxim.utils import http as _http

    model = tmp_path / "fake.gguf"
    model.write_bytes(b"GGUF")
    s = lss.LocalServerSpawner(model_path=str(model), api_key="our-key-0123456789")
    stopped: list[bool] = []
    monkeypatch.setattr(s, "stop", lambda: stopped.append(True))
    proc = MagicMock(pid=99999)
    proc.poll.return_value = None  # the child stays up
    with (
        patch("subprocess.Popen", return_value=proc),
        patch(
            "maxim.utils.http.fetch_url",
            side_effect=_http.HTTPAuthError("_external", status=401, fix_hint="Auth rejected (401)"),
        ),
    ):
        began = time.monotonic()
        with pytest.raises(lss.LLMServerKeyRejected, match="MAXIM_ROLE"):
            s.start(timeout_s=60.0)
    assert time.monotonic() - began < 5.0
    assert stopped


# ── the client side: a lane pointed at our own server sends that server's key ──────────────────────────


def test_an_operator_lane_key_cannot_override_our_servers_key(caplog) -> None:
    """A lane key set with no URL described no server; sending it to the server we started would 401 every call."""
    import logging

    from maxim.runtime.lane_backends import _server_key_fields
    from maxim.runtime.worker_pool import LaneConfig

    cfg = LaneConfig(name="large", max_workers=1, remote_api_key="operator-key")
    with caplog.at_level(logging.WARNING):
        fields = _server_key_fields(cfg, "server-key-0123456789", None)
    assert fields == {"remote_api_key": "server-key-0123456789", "remote_api_key_is_server_key": True}
    assert any("ignoring the configured lane API key" in r.getMessage() for r in caplog.records)
    assert "operator-key" not in caplog.text and "server-key" not in caplog.text  # never the values


@pytest.mark.parametrize("server_owned", [True, False])
def test_our_servers_key_rides_on_the_provider_entry_never_the_env(monkeypatch, server_owned: bool) -> None:
    """For our own server the key is the truth: it goes on the lane's provider entry, so an earlier value of the env
    var (a stale "not-needed", a peer-setup export) cannot win, and the key never enters the process env that child
    processes inherit. Any other lane keeps today's behaviour: the operator's export wins (setdefault)."""
    from unittest.mock import patch

    from maxim.models.language.config import LLMConfig
    from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
    from maxim.runtime.lane_backends import LaneBackendManager
    from maxim.runtime.worker_pool import LaneConfig

    monkeypatch.setenv("MAXIM_LANE_LARGE_API_KEY", "stale-or-exported")
    cfg = LaneConfig(
        name="large",
        max_workers=1,
        remote_url="http://127.0.0.1:8100/v1",
        remote_api_key="server-key-0123456789",
        remote_model="m",
        remote_api_key_is_server_key=server_owned,
    )
    mgr = LaneBackendManager({"large": cfg}, max_cloud_lanes=1, max_backends=5)
    with (
        patch("maxim.models.language.config.load_llm_config", return_value=LLMConfig()),
        patch("maxim.models.language.router.LLMRouter") as router,
    ):
        mgr.get_backend("large")
    entry = router.call_args[0][0].providers["lane-large"]
    assert os.environ["MAXIM_LANE_LARGE_API_KEY"] == "stale-or-exported"  # never overwritten
    backend = _MaximPeerBackend.__new__(_MaximPeerBackend)
    backend._api_key_override = None
    monkeypatch.setattr(backend, "_provider_cfg", lambda: entry, raising=False)
    expected = "server-key-0123456789" if server_owned else "stale-or-exported"
    assert backend._get_api_key() == expected
    assert ("api_key" in entry) is server_owned


def test_a_lane_config_repr_never_carries_the_key() -> None:
    from maxim.runtime.worker_pool import LaneConfig

    assert "secret-key-0123456789" not in repr(
        LaneConfig(name="large", max_workers=1, remote_api_key="secret-key-0123456789")
    )


# ── doctor: reads every key on the host, creates none, prints none ─────────────────────────────────────


def _fake_server(monkeypatch, accepts: str | None, seen: list | None = None):
    """``maxim.utils.http.fetch_url`` standing in for a server that accepts only the key ``accepts``."""
    from types import SimpleNamespace

    from maxim.utils import http as _http

    def fetch(url, *, headers=None, **_kw):
        if seen is not None:
            seen.append((url, dict(headers or {})))
        if (headers or {}).get("Authorization") == f"Bearer {accepts}":
            return SimpleNamespace(status=200, content=b'{"data": [{"id": "m"}]}')
        raise _http.HTTPAuthError("_external", status=401, fix_hint="")

    monkeypatch.setattr(_http, "fetch_url", fetch)


def test_doctor_finds_which_host_key_the_server_accepts_without_creating_one(solo: Path, monkeypatch) -> None:
    from maxim.doctor import checks
    from maxim.tunnel import keys

    assert checks._accepted_server_key(8100) == (None, None, "none-configured")
    assert not (solo / "llm_server_key").exists()  # a reader never creates the key
    local = keys.ensure_local_server_key()
    _fake_server(monkeypatch, accepts=local)
    assert checks._accepted_server_key(8100) == ("local", local, "accepted")
    smoke = checks.check_key_auth_smoke(8100)
    assert smoke.status == "ok" and "local key accepted" in smoke.message and local not in smoke.message
    _fake_server(monkeypatch, accepts="someone-elses-key")
    rejected = checks.check_key_auth_smoke(8100)
    assert rejected.status == "fail" and "MAXIM_ROLE" in rejected.fix


def test_doctor_coherence_probe_sends_the_servers_key(solo: Path, monkeypatch) -> None:
    from maxim.doctor import checks
    from maxim.tunnel import keys

    local = keys.ensure_local_server_key()
    seen: list = []
    _fake_server(monkeypatch, accepts=local, seen=seen)
    checks.check_inference_coherence(8100)
    posts = [h for url, h in seen if url.endswith("/chat/completions")]
    assert posts and posts[0]["Authorization"] == f"Bearer {local}"


def test_a_loopback_lane_answering_401_prints_the_fix_with_the_key_file_never_the_key(solo: Path, monkeypatch) -> None:
    from maxim.doctor import checks
    from maxim.tunnel import keys

    monkeypatch.setattr("maxim.runtime.config_loader.load_config", lambda *a, **k: None)
    assert checks.check_loopback_lane_auth().status == "info"  # no lane points at this host
    local = keys.ensure_local_server_key()
    monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://127.0.0.1:8100/v1")
    monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", raising=False)
    _fake_server(monkeypatch, accepts=local)
    result = checks.check_loopback_lane_auth()
    assert result.status == "fail"
    assert result.fix.startswith('export MAXIM_LANE_LARGE_REMOTE_API_KEY="$(cat ')
    assert str(solo / "llm_server_key") in result.fix
    assert local not in result.fix and local not in result.message
    monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", local)
    assert checks.check_loopback_lane_auth().status == "ok"


def test_key_creation_takes_the_lock_and_rereads_inside_it(solo: Path, monkeypatch) -> None:
    """Deterministic half of the race test: creation runs under ``file_lock`` with a real wait (not the non-blocking
    default), and a key another process wrote between the first read and the lock is read back, never overwritten."""
    import contextlib

    from maxim.tunnel import keys
    from maxim.utils import process_lock

    taken: list[tuple[str, float]] = []

    @contextlib.contextmanager
    def spy_lock(path, *, timeout_s: float = 0.0):
        taken.append((str(path), timeout_s))
        # Another process wins the race while we wait for the lock.
        keys._write_key_secret("the-other-process-key-0123456789", keys.LOCAL_SERVER_KEY_NAME)
        yield

    monkeypatch.setattr(process_lock, "file_lock", spy_lock)
    assert keys.ensure_local_server_key() == "the-other-process-key-0123456789"
    assert (solo / "llm_server_key").read_text().strip() == "the-other-process-key-0123456789"
    assert taken == [(str(solo / "llm_server_key.lock"), keys._LOCAL_KEY_LOCK_TIMEOUT_S)]
    assert keys._LOCAL_KEY_LOCK_TIMEOUT_S > 0


# ── review-round folds ─────────────────────────────────────────────────────────────────────────────────


def test_a_swap_reuses_the_running_servers_key_and_refuses_without_one(monkeypatch) -> None:
    """The proxy's, the lane client's and every peer's key were fixed when the server started: a swap after a key
    rotation must not come up under the new key. With no server this process started there is nothing to swap."""
    from types import SimpleNamespace

    from maxim.runtime import lane_backends

    monkeypatch.setattr(
        lane_backends._server_mod, "_active_spawner", SimpleNamespace(_api_key="running-key-0123456789")
    )
    monkeypatch.setattr(lane_backends, "_server_key_for_role", lambda: pytest.fail("a swap must not re-read the key"))
    assert lane_backends._swap_key() == "running-key-0123456789"
    monkeypatch.setattr(lane_backends._server_mod, "_active_spawner", None)
    with pytest.raises(RuntimeError, match="no llama-cpp-server started by this process"):
        lane_backends._swap_key()


@POSIX
def test_doctor_names_an_unusable_local_key_instead_of_no_key(solo: Path) -> None:
    from maxim.doctor import checks

    solo.mkdir(parents=True)
    path = solo / "llm_server_key"
    path.write_text("x" * 43)
    path.chmod(0o644)
    for result in (checks.check_api_key(), checks.check_key_auth_smoke(8100)):
        assert result.status == "fail" and "local server key unusable" in result.message, result
        assert str(path) in result.message and "chmod 600" in result.fix


def test_a_leader_whose_server_accepts_only_the_local_key_fails_the_smoke(solo: Path, monkeypatch) -> None:
    from maxim.doctor import checks
    from maxim.tunnel import keys

    local = keys.ensure_local_server_key()
    _fake_server(monkeypatch, accepts=local)
    monkeypatch.setattr("maxim.runtime.role.resolved_role", lambda: ("leader", "env_var"))
    result = checks.check_key_auth_smoke(8100)
    assert result.status == "fail" and "every peer request" in result.message


def test_a_key_path_that_is_a_directory_or_a_held_lock_is_a_typed_refusal(solo: Path, monkeypatch) -> None:
    import contextlib

    from maxim.tunnel import keys
    from maxim.utils import process_lock

    (solo / "llm_server_key").mkdir(parents=True)
    with pytest.raises(keys.LocalServerKeyError, match="not a regular file"):
        keys.ensure_local_server_key()
    (solo / "llm_server_key").rmdir()

    @contextlib.contextmanager
    def held(path, *, timeout_s: float = 0.0):
        raise process_lock.LockContended(f"lock held after {timeout_s:.1f}s: {path}")
        yield  # pragma: no cover

    monkeypatch.setattr(process_lock, "file_lock", held)
    with pytest.raises(keys.LocalServerKeyError, match="key-creation lock"):
        keys.ensure_local_server_key()
