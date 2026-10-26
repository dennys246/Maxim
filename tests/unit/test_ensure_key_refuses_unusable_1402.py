"""#1402: an ensure_* function never overwrites a key file it cannot read.

``tunnel/keys.py::ensure_key`` and ``ensure_console_token`` read through the lenient readers, which return None
for a missing file AND for one that exists but is unusable (mode 000, a directory, non-UTF-8, oversized), then
wrote a fresh key over it. That silently rotated the mesh key or the console token: every peer or device 401'd and
nothing named why. Now only a missing (or empty, owner decision 2026-10-25) file is generated; any other unusable
file is refused with a typed :class:`maxim.exceptions.KeyFileUnusable` naming the path, and every caller presents
the refusal instead of crashing.
"""

from __future__ import annotations

import os
import stat
import sys

import pytest
from maxim.exceptions import LeaderRefusal
from maxim.tunnel import keys


def _kfu():
    from maxim.exceptions import KeyFileUnusable

    return KeyFileUnusable


_POSIX_PERMS = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX permission bits; root ignores them",
)


@pytest.fixture
def key_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(keys, "key_file_path", lambda name=keys.MESH_KEY_NAME: tmp_path / name)
    yield tmp_path
    for p in tmp_path.iterdir():
        if p.is_file():
            os.chmod(p, 0o600)


def _unreadable(path, content="mxc_" + "a" * 43 + "\n"):
    path.write_text(content)
    os.chmod(path, 0)
    return path


_EMPTY_CASES = [
    pytest.param(keys.ensure_key, keys.MESH_KEY_NAME, id="mesh"),  # write_key already chmods 600: a pin, not a gate
    pytest.param(
        keys.ensure_console_token,
        keys.CONSOLE_KEY_NAME,
        id="console",
    ),
]

_ENSURES = [
    pytest.param(keys.ensure_key, keys.MESH_KEY_NAME, id="mesh"),
    pytest.param(keys.ensure_console_token, keys.CONSOLE_KEY_NAME, id="console"),
]


@_POSIX_PERMS
@pytest.mark.parametrize(("ensure", "name"), _ENSURES)
def test_an_unreadable_key_survives_ensure_byte_identical(key_dir, ensure, name):
    path = _unreadable(key_dir / name)
    with pytest.raises(_kfu()) as exc:
        ensure()
    os.chmod(path, 0o600)
    assert path.read_text() == "mxc_" + "a" * 43 + "\n"
    assert exc.value.path == path
    assert str(path) in str(exc.value)
    assert "rotate" not in exc.value.fix_hint  # rotating re-applies the unusable mode (atomic_write_secret)


@pytest.mark.parametrize(("ensure", "name"), _ENSURES)
def test_a_directory_at_the_key_path_is_refused_not_replaced(key_dir, ensure, name):
    (key_dir / name).mkdir()
    with pytest.raises(_kfu()):
        ensure()
    assert (key_dir / name).is_dir()


@_POSIX_PERMS
def test_the_console_refusal_is_not_a_leader_refusal(key_dir):
    """A solo console's token is not the leader key: no "leader key unusable", no LeaderRefusal handling."""
    _unreadable(key_dir / keys.CONSOLE_KEY_NAME)
    with pytest.raises(_kfu()) as exc:
        keys.ensure_console_token()
    assert type(exc.value) is _kfu()
    assert not isinstance(exc.value, LeaderRefusal)
    assert "leader" not in str(exc.value).lower()


@pytest.mark.parametrize(("ensure", "name"), _EMPTY_CASES)
def test_an_empty_key_file_is_regenerated_at_0600(key_dir, ensure, name):
    """Owner decision: empty = missing. The new key must not inherit a world-readable mode from the empty file."""
    path = key_dir / name
    path.write_text("  \n")
    os.chmod(path, 0o644)
    key = ensure()
    assert key and path.read_text().strip() == key
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize(("ensure", "name"), _ENSURES)
def test_a_missing_key_is_generated_and_then_stable(key_dir, ensure, name):
    first = ensure()
    assert first and ensure() == first
    assert (key_dir / name).read_text().strip() == first


def test_leader_key_errors_share_the_key_neutral_base():
    assert issubclass(keys.LeaderKeyError, _kfu())
    assert issubclass(keys.LeaderKeyError, LeaderRefusal)
    err = keys.LeaderKeyError(path=keys.key_file_path(), reason="cannot read key file (PermissionError)")
    assert "leader key unusable (cannot read key file (PermissionError))" in str(err)


@_POSIX_PERMS
def test_tunnel_setup_key_step_presents_the_refusal(key_dir, capsys):
    from maxim.tunnel import cli as tunnel_cli

    _unreadable(key_dir / keys.MESH_KEY_NAME)
    assert tunnel_cli._setup_api_key() is None
    out = capsys.readouterr().out
    assert "✗" in out and str(key_dir / keys.MESH_KEY_NAME) in out


@_POSIX_PERMS
def test_serve_show_token_exits_2_naming_the_file(key_dir, capsys):
    from maxim.console import server

    _unreadable(key_dir / keys.CONSOLE_KEY_NAME)
    assert server.run_serve(["--show-token"]) == 2
    out = capsys.readouterr().out
    assert str(key_dir / keys.CONSOLE_KEY_NAME) in out
    assert "mxc_" not in out


@_POSIX_PERMS
def test_serve_startup_refuses_before_building_the_app(key_dir, monkeypatch, capsys):
    from maxim.console import server

    monkeypatch.delenv("MAXIM_CONSOLE_SANDBOX", raising=False)
    monkeypatch.setattr(server, "build_app", lambda *a, **k: pytest.fail("app built over an unusable token"))
    _unreadable(key_dir / keys.CONSOLE_KEY_NAME)
    assert server.run_serve([]) == 2
    assert str(key_dir / keys.CONSOLE_KEY_NAME) in capsys.readouterr().out


@_POSIX_PERMS
def test_device_handoff_raises_typed(key_dir):
    from maxim.console.server import device_console_handoff

    _unreadable(key_dir / keys.CONSOLE_KEY_NAME)
    with pytest.raises(_kfu()):
        device_console_handoff("reachy.local")


def test_a_held_creation_lock_is_a_typed_refusal(key_dir, monkeypatch):
    import contextlib

    from maxim.utils import process_lock

    @contextlib.contextmanager
    def _held(*a, **k):
        raise process_lock.LockContended("held")
        yield

    monkeypatch.setattr(process_lock, "file_lock", _held)
    with pytest.raises(_kfu(), match="key-creation lock"):
        keys.ensure_console_token()
    assert not (key_dir / keys.CONSOLE_KEY_NAME).exists()


def test_a_key_that_cannot_be_created_is_a_typed_refusal(key_dir, monkeypatch):
    def _denied(key, name):
        raise PermissionError("read-only config dir")

    monkeypatch.setattr(keys, "_write_key_secret", _denied)
    with pytest.raises(_kfu(), match="cannot create key file"):
        keys.ensure_console_token()


def test_a_writer_that_leaves_nothing_is_a_typed_refusal(key_dir, monkeypatch):
    monkeypatch.setattr(keys, "_write_key_secret", lambda key, name: None)
    with pytest.raises(_kfu(), match="empty after it was written"):
        keys.ensure_console_token()


@pytest.mark.parametrize(
    ("rotate", "name"),
    [(keys.rotate_key, keys.MESH_KEY_NAME), (keys.rotate_console_token, keys.CONSOLE_KEY_NAME)],
)
def test_rotation_writes_under_the_creation_lock(key_dir, monkeypatch, rotate, name):
    import contextlib

    from maxim.utils import process_lock

    held = []
    real = process_lock.file_lock

    @contextlib.contextmanager
    def _spy(path, **k):
        with real(path, **k):
            held.append(str(path))
            yield

    monkeypatch.setattr(process_lock, "file_lock", _spy)
    new = rotate()
    assert held == [str(key_dir / f"{name}.lock")]
    assert (key_dir / name).read_text().strip() == new


@pytest.mark.parametrize(("ensure", "name"), _ENSURES)
def test_a_key_created_while_waiting_for_the_lock_is_returned_not_overwritten(key_dir, monkeypatch, ensure, name):
    """The re-read inside the lock: another process creates the key (and hands it to a device) between this
    process's first read and its lock; regenerating would 401 that device with nothing naming why."""
    import contextlib

    from maxim.utils import process_lock

    other = "mxc_" + "o" * 43

    @contextlib.contextmanager
    def _race(path, **k):
        (key_dir / name).write_text(other + "\n")
        os.chmod(key_dir / name, 0o600)
        yield

    monkeypatch.setattr(process_lock, "file_lock", _race)
    assert ensure() == other
    assert (key_dir / name).read_text().strip() == other


def test_an_empty_local_server_key_is_regenerated_too(tmp_path, monkeypatch):
    """Owner decision 2026-10-25 covers every ensure: the spawn key's empty file is missing, not a refusal."""
    monkeypatch.setattr(keys, "key_file_path", lambda name=keys.MESH_KEY_NAME: tmp_path / name)
    (tmp_path / keys.LOCAL_SERVER_KEY_NAME).write_text("\n")
    key = keys.ensure_local_server_key()
    assert key and (tmp_path / keys.LOCAL_SERVER_KEY_NAME).read_text().strip() == key


def test_tunnel_setup_key_step_reports_a_ready_key(key_dir, capsys):
    from maxim.tunnel import cli as tunnel_cli

    key = tunnel_cli._setup_api_key()
    out = capsys.readouterr().out
    assert key and "API key ready" in out
    assert key[:6] not in out  # no key material on the terminal, not even a prefix (#1490 CodeQL)


def test_rotate_commands_report_a_held_lock(key_dir, monkeypatch, capsys):
    import contextlib

    from maxim.console import server
    from maxim.tunnel import cli as tunnel_cli
    from maxim.utils import process_lock

    @contextlib.contextmanager
    def _held(*a, **k):
        raise process_lock.LockContended("held")
        yield

    monkeypatch.setattr(process_lock, "file_lock", _held)
    assert tunnel_cli._cmd_key_rotate() == 1
    assert server.run_serve(["--rotate-token"]) == 2
    out = capsys.readouterr().out
    assert out.count("key-creation lock") == 2
