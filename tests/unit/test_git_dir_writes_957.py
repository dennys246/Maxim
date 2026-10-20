"""#957: the file tools refuse to write under a ``.git/`` directory.

Any git call over a repository the model can write is code execution: ``git diff`` and ``git add`` run clean/process
filters set by ``.gitattributes`` plus ``filter.<name>.clean=<cmd>`` in ``.git/config``, and git has no switch that
disables filters (#949). ``write_file`` and ``edit_file`` had no deny for ``.git/``. They now refuse any path whose
resolved form has a ``.git`` component (case-folded: macOS and Windows filesystems are case-insensitive). This
narrows the risk; it does not close it, because ``bash`` can still write there (owner decision 2026-09-28: contain,
give an honest claim, keep both git tools opt-in).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    from maxim.tools.filesystem import EditFileTool, WriteFileTool

    # tmp_path is under /private/var on macOS, a FORBIDDEN_PATHS root (test_edit_file_tool.py's pattern).
    monkeypatch.setattr(WriteFileTool, "FORBIDDEN_PATHS", ())
    monkeypatch.setattr(EditFileTool, "FORBIDDEN_PATHS", ())
    root = tmp_path / "work"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "config").write_text("[core]\n")
    (root / "src.py").write_text("x = 1\n")
    return root


def _write(root: Path, path: str):
    from maxim.tools.filesystem import WriteFileTool

    return WriteFileTool(allowed_dirs=[str(root)]).execute(
        path=path, content='[filter "x"]\nclean = touch pwned\n', overwrite=True
    )


def _edit(root: Path, path: str):
    from maxim.tools.filesystem import EditFileTool

    return EditFileTool(allowed_dirs=[str(root)]).execute(
        path=path, old_text="[core]", new_text="[core]\nfsmonitor = x"
    )


@pytest.mark.xfail(strict=True, reason="#957: write_file writes into .git/")
@pytest.mark.parametrize("target", [".git/config", ".GIT/config", ".git/hooks/pre-commit"])
def test_write_file_refuses_the_git_dir(repo: Path, target: str) -> None:
    before = (repo / ".git" / "config").read_bytes()
    result = _write(repo, str(repo / target))
    assert result.success is False and ".git" in (result.error or "")
    assert (repo / ".git" / "config").read_bytes() == before
    assert not (repo / ".git" / "hooks").exists()


@pytest.mark.xfail(strict=True, reason="#957: edit_file edits .git/config")
def test_edit_file_refuses_the_git_dir(repo: Path) -> None:
    before = (repo / ".git" / "config").read_bytes()
    result = _edit(repo, str(repo / ".git" / "config"))
    assert result.success is False and ".git" in (result.error or "")
    assert (repo / ".git" / "config").read_bytes() == before


@pytest.mark.xfail(strict=True, reason="#957: a symlink into .git/ is written through")
def test_a_symlink_into_the_git_dir_is_refused(repo: Path) -> None:
    os.symlink(repo / ".git" / "config", repo / "innocent.cfg")
    before = (repo / ".git" / "config").read_bytes()
    result = _write(repo, str(repo / "innocent.cfg"))
    assert result.success is False
    assert (repo / ".git" / "config").read_bytes() == before


def test_a_name_that_only_contains_git_is_still_writable(repo: Path) -> None:
    """A component match, not a substring: ``my.git.repo/`` and ``.gitignore`` are ordinary files."""
    (repo / "my.git.repo").mkdir()
    assert _write(repo, str(repo / "my.git.repo" / "notes.txt")).success is True
    assert _write(repo, str(repo / ".gitignore")).success is True
