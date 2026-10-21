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

RED = pytest.mark.xfail(strict=True, reason="#1314 red gate: a non-leader's server is spawned without a key")
POSIX = pytest.mark.skipif(platform.system() == "Windows", reason="POSIX file modes")


@pytest.fixture
def solo(tmp_path: Path, monkeypatch) -> Path:
    """A solo process with its own config dir (conftest's session-wide XDG dir would carry a key between tests)."""
    from maxim.runtime import role

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("MAXIM_ROLE", "solo")
    monkeypatch.setattr(role, "_APPLIED", None)
    return tmp_path / "xdg" / "maxim"


@RED
@POSIX
def test_a_solo_spawn_gets_a_key_file_created_0600(solo: Path) -> None:
    from maxim.runtime.lane_backends import _server_key_for_role

    key = _server_key_for_role()
    assert isinstance(key, str) and len(key) >= 16
    path = solo / "llm_server_key"
    assert path.read_text().strip() == key
    assert (path.stat().st_mode & 0o777) == 0o600
    assert _server_key_for_role() == key  # stable: read back, not regenerated


@RED
def test_a_reader_never_creates_the_key(solo: Path) -> None:
    """Doctor and the o19 harness read the key; creating it would leave a key no running server holds."""
    from maxim.tunnel import keys

    assert keys.read_local_server_key() is None
    assert not (solo / "llm_server_key").exists()


@RED
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
        for _ in range(4)
    ]
    keys_seen = {p.communicate(timeout=60)[0].strip() for p in procs}
    assert all(p.returncode == 0 for p in procs)
    assert len(keys_seen) == 1
    assert (solo / "llm_server_key").read_text().strip() == keys_seen.pop()


@RED
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


@RED
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


@RED
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
