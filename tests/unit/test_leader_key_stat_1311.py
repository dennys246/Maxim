"""#1311 + #1312: an unusable leader key path is a typed refusal on every Python, never a raw traceback.

Both rest on ``Path.is_file()``. On Python 3.10–3.13 it RAISES the stat error of an unreadable parent directory, so
``maxim oasis serve``'s ``key_exists()`` pre-check crashed with a raw ``PermissionError`` before the typed
``LeaderKeyError`` could name the path and the fix (#1311). On Python 3.14 it SWALLOWS every ``OSError`` and returns
False, so ``require_leader_key``'s "cannot stat key file" became "no key file" (#1312). The fix opens, fstats and
reads one descriptor, and a missing key is its own type, ``LeaderKeyMissing``, so callers branch on type, not on text.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.unit._key_file_helpers import deny_os_access


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


# ── Review folds ───────────────────────────────────────────────────────────────────────────────


def _key_path(xdg: Path, name: str | None = None) -> Path:
    from maxim.tunnel import keys

    path = keys.key_file_path(name or keys.MESH_KEY_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


posix_non_root = pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions enforced (non-root)"
)


@posix_non_root
def test_a_mode_000_key_reads_as_cannot_read(xdg) -> None:
    """open() is denied but stat() succeeds: the file itself refused, not the directory."""
    from maxim.tunnel import keys

    path = _key_path(xdg)
    path.write_text("k" * 43)
    path.chmod(0o000)
    try:
        with pytest.raises(keys.LeaderKeyError, match="cannot read key file"):
            keys.require_leader_key()
        assert keys.key_file_state() == ("unusable", "cannot read key file (PermissionError)")
    finally:
        path.chmod(0o600)


def test_an_oversized_or_undecodable_key_is_unusable(xdg) -> None:
    from maxim.tunnel import keys

    path = _key_path(xdg)
    path.write_bytes(b"k" * (64 * 1024 + 1))
    path.chmod(0o600)
    with pytest.raises(keys.LeaderKeyError, match="too large"):
        keys.require_leader_key()
    path.write_bytes(b"\xff\xfe" + b"k" * 43)
    with pytest.raises(keys.LeaderKeyError, match="UnicodeDecodeError"):
        keys.require_leader_key()
    assert keys.key_file_state()[0] == "unusable" and keys.read_key() is None


def test_a_short_read_is_completed(xdg, monkeypatch) -> None:
    """A network filesystem may return fewer bytes than asked; the reader loops to EOF."""
    from maxim.tunnel import keys

    path = _key_path(xdg)
    path.write_text("k" * 43)
    path.chmod(0o600)
    real_read = os.read
    monkeypatch.setattr(os, "read", lambda fd, n: real_read(fd, min(n, 5)))
    assert keys.require_leader_key() == "k" * 43


def test_read_key_strips_a_bom_like_the_leader(xdg) -> None:
    from maxim.tunnel import keys

    path = _key_path(xdg)
    path.write_text("﻿" + "k" * 43 + "\n", encoding="utf-8")
    path.chmod(0o600)
    assert keys.read_key() == keys.require_leader_key() == "k" * 43


def test_key_file_state_names_each_state(xdg) -> None:
    from maxim.tunnel import keys

    assert keys.key_file_state() == ("missing", None)
    path = _key_path(xdg)
    path.write_text("k" * 43)
    assert keys.key_file_state() == ("ok", None)
    path.unlink()
    path.mkdir()
    assert keys.key_file_state() == ("unusable", "key path is not a regular file")


@pytest.mark.parametrize("check", ["check_api_key", "check_key_age", "check_key_permissions", "check_key_auth_smoke"])
def test_doctor_reports_an_unusable_key_not_a_missing_or_mismatched_one(xdg, monkeypatch, check) -> None:
    """Not "no key set", not a directory's mode, never "rotate" (which would discard a key that may be fine)."""
    from maxim.doctor import checks

    monkeypatch.setenv("MAXIM_ROLE", "peer")
    _key_path(xdg).mkdir()
    result = getattr(checks, check)()
    assert result.status == "fail" and "unusable" in result.message
    assert "rotate" not in (result.fix or "").replace("do not rotate", "")


def test_doctor_tells_an_empty_key_file_to_generate_one(xdg, monkeypatch) -> None:
    from maxim.doctor import checks

    monkeypatch.setenv("MAXIM_ROLE", "peer")
    _key_path(xdg).write_text("")
    result = checks.check_api_key()
    assert result.status == "fail" and "empty" in result.message and "rotate" in result.fix
