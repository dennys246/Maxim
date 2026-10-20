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
    return path


@pytest.mark.xfail(strict=True, reason="#1311: key_exists() raises a raw PermissionError before the typed refusal")
def test_oasis_serve_refuses_an_unstatable_key_with_the_typed_message(tmp_path: Path, monkeypatch, capsys) -> None:
    from maxim.hivemind.oasis_cli import run_oasis_subcommand

    path = _unstatable_key(tmp_path, monkeypatch, is_file_raises=True)
    rc = run_oasis_subcommand(["serve", "--root", str(tmp_path / "store"), "--port", "1"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "cannot stat key file" in err and str(path) in err


@pytest.mark.xfail(strict=True, reason="#1312: require_leader_key reads existence through Path.is_file()")
def test_an_unstatable_key_is_named_as_such_whatever_is_file_does(tmp_path: Path, monkeypatch) -> None:
    """Python 3.14's ``is_file()`` swallows the error and returns False: the reader must not depend on it."""
    from maxim.tunnel import keys

    _unstatable_key(tmp_path, monkeypatch, is_file_raises=False)
    with pytest.raises(keys.LeaderKeyError, match="cannot stat key file"):
        keys.require_leader_key()
