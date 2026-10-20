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


@pytest.mark.parametrize("target", [".git/config", ".GIT/config", ".git/hooks/pre-commit"])
def test_write_file_refuses_the_git_dir(repo: Path, target: str) -> None:
    before = (repo / ".git" / "config").read_bytes()
    result = _write(repo, str(repo / target))
    assert result.success is False and ".git" in (result.error or "")
    assert (repo / ".git" / "config").read_bytes() == before
    assert not (repo / ".git" / "hooks").exists()


def test_edit_file_refuses_the_git_dir(repo: Path) -> None:
    before = (repo / ".git" / "config").read_bytes()
    result = _edit(repo, str(repo / ".git" / "config"))
    assert result.success is False and ".git" in (result.error or "")
    assert (repo / ".git" / "config").read_bytes() == before


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


# ── Design-pass folds ──────────────────────────────────────────────────────────────────────────


def test_the_users_global_git_config_is_refused(repo: Path, tmp_path: Path, monkeypatch) -> None:
    """Host git reads ~/.gitconfig and $XDG_CONFIG_HOME/git/ (host_tool_env passes HOME through): the same
    filter exploit by another file."""
    from maxim.tools.base import git_dir_write_refusal

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for target in ("home/.gitconfig", "xdg/git/config", "xdg/git/attributes", "home/.config/git/config"):
        assert git_dir_write_refusal(str(tmp_path / target)) is not None, target
    assert git_dir_write_refusal(str(tmp_path / "home" / "notes.txt")) is None


def test_a_separate_git_dir_behind_a_gitfile_is_refused(tmp_path: Path) -> None:
    """``git init --separate-git-dir``: the worktree's .git is a FILE pointing at a git dir not named .git."""
    from maxim.tools.base import git_dir_write_refusal

    work, gitdir = tmp_path / "work", tmp_path / "work" / "meta"
    gitdir.mkdir(parents=True)
    (gitdir / "config").write_text("[core]\n")
    (work / ".git").write_text("gitdir: meta\n")
    assert git_dir_write_refusal(str(gitdir / "config")) is not None
    assert git_dir_write_refusal(str(work / ".git")) is not None  # the gitfile itself
    assert git_dir_write_refusal(str(work / "src.py")) is None


def test_another_spelling_of_the_git_dir_is_refused_by_identity(tmp_path: Path, monkeypatch) -> None:
    """A name the filesystem maps onto .git (an ignorable character, an 8.3 name) is caught by stat identity."""
    from maxim.tools import base

    work = tmp_path / "work"
    (work / ".git").mkdir(parents=True)
    alias = work / "G1T~1"
    alias.mkdir()
    real_samefile = os.path.samefile
    monkeypatch.setattr(
        base.os.path,
        "samefile",
        lambda a, b: True if {os.fspath(a), os.fspath(b)} == {str(alias), str(work / ".git")} else real_samefile(a, b),
    )
    assert base.git_dir_write_refusal(str(alias / "config")) is not None


def test_the_sandbox_writers_refuse_the_git_dir(tmp_path: Path) -> None:
    from maxim.tools.sandbox import WriteSandboxFileTool, WriteToSharedOutputsTool

    sandbox = tmp_path / "sandbox"
    (sandbox / ".git").mkdir(parents=True)
    result = WriteSandboxFileTool(sandbox_dir=str(sandbox)).execute(path=".git/config", content="x")
    assert result.success is False and ".git" in result.error
    outputs = tmp_path / "outputs"
    result = WriteToSharedOutputsTool(instance_outputs_dir=str(outputs)).execute(filename=".git", content="gitdir: /x")
    assert result.success is False
    assert not (outputs / ".git").exists()


def test_a_refused_write_creates_no_directories(repo: Path) -> None:
    """The refusal comes before ``mkdir``: ``.git/hooks/`` is never created."""
    _write(repo, str(repo / ".git" / "hooks" / "post-checkout"))
    assert not (repo / ".git" / "hooks").exists()


def test_the_name_check_is_case_folded_where_the_path_does_not_exist_yet() -> None:
    """On a case-sensitive filesystem ``.GIT`` is a different directory, but a case-insensitive one maps it onto
    ``.git``; the name check refuses it either way (and covers paths that do not exist, which identity cannot)."""
    from maxim.tools.base import git_dir_write_refusal

    assert git_dir_write_refusal("/nonexistent-957/.GIT/config") is not None


# ── Review folds ───────────────────────────────────────────────────────────────────────────────


def test_a_sibling_separate_git_dir_is_refused(tmp_path: Path) -> None:
    """The usual ``git init --separate-git-dir <gd> work`` layout: the git dir is a SIBLING, not inside the worktree."""
    import shutil
    import subprocess

    from maxim.tools.base import git_dir_write_refusal

    if shutil.which("git") is None:
        pytest.skip("needs git")
    gd, work = tmp_path / "gd", tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--separate-git-dir", str(gd), str(work)], check=True)
    for target in (gd / "config", gd / "hooks" / "x", gd / "info" / "attributes"):
        assert git_dir_write_refusal(str(target)) is not None, target


def test_git_refuses_a_bare_repository_assembled_with_plain_writes(tmp_path: Path) -> None:
    """No single write looks like a git dir, in some order (config, then HEAD, then objects/); host git must refuse to
    use the result. Control: without ``safe.bareRepository=explicit`` git runs the planted filter."""
    import shutil
    import subprocess

    from maxim.tools.base import GIT_HARDENING_ARGS, git_env

    if shutil.which("git") is None:
        pytest.skip("needs git")
    d = tmp_path / "d"
    for sub in ("objects", "refs/heads"):
        (d / sub).mkdir(parents=True)
    pwned = tmp_path / "PWNED"
    (d / "config").write_text(f'[core]\n\tbare = true\n[filter "x"]\n\tclean = touch {pwned}\n')
    (d / "HEAD").write_text("ref: refs/heads/main\n")
    env = git_env(str(d))
    hardened = subprocess.run(
        ["git", *GIT_HARDENING_ARGS, "rev-parse", "--git-dir"], cwd=d, env=env, capture_output=True, text=True
    )
    assert hardened.returncode != 0 and "bare" in hardened.stderr.lower()
    control = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "rev-parse", "--git-dir"],
        cwd=d,
        env=env,
        capture_output=True,
        text=True,
    )
    assert control.returncode == 0  # the directory IS usable as a repository without the new switch
    assert "safe.bareRepository=explicit" in GIT_HARDENING_ARGS


def test_a_case_variant_of_the_global_git_config_is_refused_before_it_exists(tmp_path: Path, monkeypatch) -> None:
    from maxim.tools.base import git_dir_write_refusal

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert git_dir_write_refusal(str(tmp_path / "home" / ".GitConfig")) is not None
    assert git_dir_write_refusal(str(tmp_path / "home" / ".config" / "GIT" / "config")) is not None


def test_homebrews_system_gitconfig_is_refused() -> None:
    from maxim.tools.base import git_dir_write_refusal

    for target in ("/usr/local/etc/gitconfig", "/opt/homebrew/etc/gitconfig"):
        assert git_dir_write_refusal(target) is not None
