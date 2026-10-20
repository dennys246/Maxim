"""#1311 + #1312: an unusable leader key path is a typed refusal on every Python, never a raw traceback.

Both rest on ``Path.is_file()``. On Python 3.10–3.13 it RAISES the stat error of an unreadable parent directory, so
``maxim oasis serve``'s ``key_exists()`` pre-check crashed with a raw ``PermissionError`` before the typed
``LeaderKeyError`` could name the path and the fix (#1311). On Python 3.14 it SWALLOWS every ``OSError`` and returns
False, so ``require_leader_key``'s "cannot stat key file" became "no key file" (#1312). The fix stats the key path
directly, and a missing key is its own type, ``LeaderKeyMissing``, so callers branch on type, not on text.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _unstatable_key(tmp_path: Path, monkeypatch, *, is_file_raises: bool):
    """A key file whose stat fails, as under an unreadable parent directory. ``is_file_raises`` picks the Python
    behaviour: 3.10–3.13 raise from ``Path.is_file``, 3.14 swallows and returns False."""
    from maxim.tunnel import keys

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    path = keys.key_file_path(keys.MESH_KEY_NAME)
    path.parent.mkdir(parents=True)
    path.write_text("x" * 43)
    path.chmod(0o600)
    real_stat = type(path).stat

    def _stat_denied(self, *a, **k):
        if self == path:
            raise PermissionError(13, "Permission denied", str(self))
        return real_stat(self, *a, **k)

    def _is_file(self):
        if self == path:
            if is_file_raises:
                raise PermissionError(13, "Permission denied", str(self))
            return False
        return real_stat(self).st_mode & 0o170000 == 0o100000 if os.path.lexists(self) else False

    monkeypatch.setattr(type(path), "stat", _stat_denied)
    monkeypatch.setattr(type(path), "is_file", _is_file)
    deny_os_access(monkeypatch, path)
    return path


def deny_os_access(monkeypatch, path: Path) -> None:
    """An unreadable parent directory denies ``open()`` and ``stat()`` on the path too, not only ``Path.stat``."""
    real_open, real_stat = os.open, os.stat

    def _open(p, *a, **k):
        if os.fspath(p) == os.fspath(path):
            raise PermissionError(13, "Permission denied", os.fspath(p))
        return real_open(p, *a, **k)

    def _stat(p, *a, **k):
        if not isinstance(p, int) and os.fspath(p) == os.fspath(path):
            raise PermissionError(13, "Permission denied", os.fspath(p))
        return real_stat(p, *a, **k)

    monkeypatch.setattr(os, "open", _open)
    monkeypatch.setattr(os, "stat", _stat)


def test_oasis_serve_refuses_an_unstatable_key_with_the_typed_message(tmp_path: Path, monkeypatch, capsys) -> None:
    from maxim.hivemind.oasis_cli import run_oasis_subcommand

    path = _unstatable_key(tmp_path, monkeypatch, is_file_raises=True)
    rc = run_oasis_subcommand(["serve", "--root", str(tmp_path / "store"), "--port", "1"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "cannot stat key file" in err and str(path) in err


def test_an_unstatable_key_is_named_as_such_whatever_is_file_does(tmp_path: Path, monkeypatch) -> None:
    """Python 3.14's ``is_file()`` swallows the error and returns False: the reader must not depend on it."""
    from maxim.tunnel import keys

    _unstatable_key(tmp_path, monkeypatch, is_file_raises=False)
    with pytest.raises(keys.LeaderKeyError, match="cannot stat key file"):
        keys.require_leader_key()


# ── Design-pass folds: only a missing file is "no key"; everything else unusable is refused ────────


@pytest.fixture
def xdg(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return tmp_path / "xdg"


def test_a_missing_key_is_its_own_type(xdg) -> None:
    from maxim.tunnel import keys

    with pytest.raises(keys.LeaderKeyMissing, match="no key file"):
        keys.require_leader_key()
    assert keys.key_exists() is False and keys.read_key() is None


def test_a_config_path_that_is_a_file_is_unusable_not_missing(xdg, tmp_path, capsys) -> None:
    """ENOTDIR (``~/.config/maxim`` is a regular file) must not let ``--insecure`` serve with auth off."""
    from maxim.hivemind.oasis_cli import run_oasis_subcommand
    from maxim.tunnel import keys

    xdg.mkdir()
    (xdg / "maxim").write_text("not a directory")
    with pytest.raises(keys.LeaderKeyError) as exc:
        keys.require_leader_key()
    assert not isinstance(exc.value, keys.LeaderKeyMissing)
    assert keys.key_exists() is True and keys.read_key() is None
    rc = run_oasis_subcommand(["serve", "--root", str(tmp_path / "store"), "--port", "1", "--insecure"])
    assert rc == 2 and "leader key unusable" in capsys.readouterr().err


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
@pytest.mark.parametrize("kind", ["directory", "fifo", "symlink-to-fifo"])
def test_a_key_path_that_is_not_a_regular_file_is_refused_without_hanging(xdg, kind) -> None:
    from maxim.tunnel import keys

    path = keys.key_file_path(keys.MESH_KEY_NAME)
    path.parent.mkdir(parents=True)
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        os.mkfifo(path.parent / "pipe")
        path.symlink_to(path.parent / "pipe")
    with pytest.raises(keys.LeaderKeyError, match="not a regular file"):
        keys.require_leader_key()
    assert keys.read_key() is None


def test_a_symlink_to_a_good_key_is_read(xdg) -> None:
    """Dotfile managers symlink keys; the target's mode is the one checked."""
    from maxim.tunnel import keys

    path = keys.key_file_path(keys.MESH_KEY_NAME)
    path.parent.mkdir(parents=True)
    real = path.parent / "real_key"
    real.write_text("k" * 43)
    real.chmod(0o600)
    path.symlink_to(real)
    assert keys.require_leader_key() == "k" * 43


def test_an_unencodable_path_is_unusable_not_a_raw_error(monkeypatch) -> None:
    from maxim.tunnel import keys

    monkeypatch.setattr(keys, "key_file_path", lambda name=keys.MESH_KEY_NAME: Path("/tmp/nul\x00here") / name)
    with pytest.raises(keys.LeaderKeyError, match="cannot stat key file"):
        keys.require_leader_key()
    assert keys.key_exists() is True and keys.read_key() is None


def test_the_lenient_readers_never_raise_on_an_unstatable_key(tmp_path, monkeypatch, caplog) -> None:
    """``read_key`` documented "None for every failure", but ``is_file()`` raised on <=3.13; the console token reader
    is on the per-request auth path."""
    from maxim.tunnel import keys

    _unstatable_key(tmp_path, monkeypatch, is_file_raises=True)
    assert keys.read_key() is None and keys.key_exists() is True
