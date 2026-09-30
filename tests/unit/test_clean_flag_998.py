"""#998 -- the clean-tree flag is decided by content, not by ``git status``.

``working_tree_dirty_src_scripts`` gates every committed experiment record (``scripts/_provenance.py``'s
refuse path) and is stamped into every sim report. It asked ``git status``, which git config and index
state could make report a dirty tree as clean. It now lives in ``src/maxim/utils/code_tree.py`` (loaded by
path by ``scripts/_provenance.py``) and compares every code path on disk with HEAD: blob, executable bit,
symlink target.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

import maxim
from maxim.utils import code_tree

REPO = Path(maxim.__file__).resolve().parents[2]
CODE_TREE = REPO / "src" / "maxim" / "utils" / "code_tree.py"


def _provenance():
    spec = importlib.util.spec_from_file_location("_provenance", REPO / "scripts" / "_provenance.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_provenance"] = module
    spec.loader.exec_module(module)
    return module


P = _provenance()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_repo(path: Path, *, object_format: str = "sha1") -> Path:
    for d in ("src", "scripts", "docs"):
        (path / d).mkdir(parents=True)
    (path / ".gitignore").write_text("__pycache__/\n.pytest_cache/\nnode_modules/\n.venv/\n")
    (path / "src" / "a.py").write_text("x = 1\n")
    (path / "scripts" / "run.sh").write_text("echo hi\n")
    os.chmod(path / "scripts" / "run.sh", 0o755)
    (path / "scripts" / "link").symlink_to("run.sh")
    _git(path, "init", "-q", f"--object-format={object_format}")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "init")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _make_repo(tmp_path / "repo")


def _case_insensitive(where: Path) -> bool:
    probe = where / "CaseProbe"
    probe.write_text("")
    try:
        return (where / "caseprobe").exists()
    finally:
        probe.unlink()


# ── clean stays clean ────────────────────────────────────────────────────


def test_a_clean_tree_reads_clean_and_harmless_changes_keep_it_clean(repo: Path) -> None:
    assert code_tree.tree_difference(repo) is None
    later = os.stat(repo / "src" / "a.py").st_mtime + 5
    os.utime(repo / "src" / "a.py", (later, later))  # a touch is not a change
    (repo / "src" / "__pycache__").mkdir()
    (repo / "src" / "__pycache__" / "a.pyc").write_bytes(b"\x00")  # ignored by the repo's own rules
    (repo / "docs" / "notes.md").write_text("outside the scope\n")
    (repo / "src" / "a.py").write_text("x = 9\n")
    (repo / "src" / "a.py").write_text("x = 1\n")  # edited and restored
    os.chmod(repo / "src" / "a.py", 0o654)  # group exec only: git records 100644 (owner bit)
    assert code_tree.tree_difference(repo) is None


@pytest.mark.parametrize(
    "where", ["scripts/.pytest_cache", "scripts/node_modules/pkg", "scripts/node_modules/pkg/sub/deeper", "src/.venv"]
)
def test_tooling_that_ships_its_own_gitignore_inside_an_ignored_directory_stays_clean(repo: Path, where: str) -> None:
    """A pytest cache, a venv or an npm package writes ``.gitignore: *``; its directory is already excluded
    by the TRACKED root rules, so nothing there is code (review finding: these refused clean runs)."""
    (repo / where).mkdir(parents=True)
    (repo / where / ".gitignore").write_text("*\n")
    (repo / where / "module.py").write_text("print('tooling')\n")
    assert code_tree.tree_difference(repo) is None


def test_a_staged_only_change_with_the_disk_equal_to_head_is_clean(repo: Path) -> None:
    """The flag judges the code on disk, not the index."""
    (repo / "src" / "a.py").write_text("x = 2\n")
    _git(repo, "add", "src/a.py")
    (repo / "src" / "a.py").write_text("x = 1\n")
    assert code_tree.tree_difference(repo) is None


@pytest.mark.parametrize("rule", ["/scripts/cache/", "**/cache/", "cache"])
def test_anchored_and_recursive_tracked_rules_excuse_tooling_too(repo: Path, rule: str) -> None:
    (repo / ".gitignore").write_text(f"__pycache__/\n{rule}\n")
    _git(repo, "commit", "-q", "-am", "rule")
    (repo / "scripts" / "cache" / "x").mkdir(parents=True)
    (repo / "scripts" / "cache" / "x" / ".gitignore").write_text("*\n")
    assert code_tree.tree_difference(repo) is None


def test_index_state_cannot_change_which_rules_count_as_tracked(repo: Path) -> None:
    """Round 3: the tracked rules came from the index, so ``git rm --cached`` of a rule file -- nothing on disk
    changed -- flipped the verdict and moved the digest. HEAD decides which rules are tracked."""
    (repo / "src" / ".gitignore").write_text("vend/\n")
    _git(repo, "add", "src/.gitignore")
    _git(repo, "commit", "-q", "-m", "rule")
    (repo / "src" / "vend").mkdir()
    (repo / "src" / "vend" / ".gitignore").write_text("*\n")  # tooling in an excused directory
    before = (code_tree.tree_difference(repo), code_tree.code_tree_sha256(repo))
    assert before[0] is None
    _git(repo, "rm", "-q", "--cached", "src/.gitignore")
    assert (code_tree.tree_difference(repo), code_tree.code_tree_sha256(repo)) == before


def test_a_sha256_repository_is_judged_in_its_own_object_format(tmp_path: Path) -> None:
    try:
        repo = _make_repo(tmp_path / "repo", object_format="sha256")
    except subprocess.CalledProcessError:
        pytest.skip("this git has no sha256 object format")
    assert code_tree.tree_difference(repo) is None
    (repo / "src" / "a.py").write_text("x = 2\n")
    assert code_tree.tree_difference(repo) == "src/a.py: content differs from HEAD"


# ── every change reads dirty, and says which ─────────────────────────────


@pytest.mark.parametrize(
    "change, expected",
    [
        ("edit", "src/a.py: content differs from HEAD"),
        ("delete", "src/a.py: deleted"),
        ("untracked", "scripts/new.py: not in HEAD"),
        ("exec_bit", "src/a.py: is executable, HEAD has file"),
        ("symlink_target", "scripts/link: content differs from HEAD"),
        ("root_gitignore", ".gitignore: content differs from HEAD"),
        ("self_ignoring_gitignore", "src/pkg/.gitignore: not in HEAD"),
        ("info_exclude_only", "scripts/pkg/.gitignore: not in HEAD"),
        ("excludes_file", "src/new.py: not in HEAD"),
        ("assume_unchanged", "src/a.py: content differs from HEAD"),
        ("skip_worktree", "src/a.py: content differs from HEAD"),
        ("file_mode_off", "src/a.py: is executable, HEAD has file"),
    ],
)
def test_every_change_to_the_code_reads_dirty_and_is_named(repo: Path, tmp_path: Path, change, expected) -> None:
    """From ``excludes_file`` on, these were invisible to ``git status`` (#998)."""
    if change == "edit":
        (repo / "src" / "a.py").write_text("x = 2\n")
    elif change == "delete":
        (repo / "src" / "a.py").unlink()
    elif change == "untracked":
        (repo / "scripts" / "new.py").write_text("print(1)\n")
    elif change == "exec_bit":
        os.chmod(repo / "src" / "a.py", 0o755)
    elif change == "symlink_target":
        (repo / "scripts" / "link").unlink()
        (repo / "scripts" / "link").symlink_to("elsewhere.sh")
    elif change == "root_gitignore":
        (repo / ".gitignore").write_text("__pycache__/\nsrc/evil.py\n")
        (repo / "src" / "evil.py").write_text("print('hidden')\n")
    elif change == "self_ignoring_gitignore":
        (repo / "src" / "pkg").mkdir()
        (repo / "src" / "pkg" / ".gitignore").write_text("*\n")
        (repo / "src" / "pkg" / "evil.py").write_text("print('hidden')\n")
    elif change == "info_exclude_only":  # only the tracked rules may excuse a directory
        (repo / ".git" / "info").mkdir(exist_ok=True)
        (repo / ".git" / "info" / "exclude").write_text("scripts/pkg/\n")
        (repo / "scripts" / "pkg").mkdir()
        (repo / "scripts" / "pkg" / ".gitignore").write_text("*\n")
        (repo / "scripts" / "pkg" / "evil.py").write_text("print('hidden')\n")
    elif change == "excludes_file":
        (tmp_path / "ignore").write_text("*.py\n")
        _git(repo, "config", "core.excludesFile", str(tmp_path / "ignore"))
        (repo / "src" / "new.py").write_text("print('hidden')\n")
    elif change in ("assume_unchanged", "skip_worktree"):
        _git(repo, "update-index", f"--{change.replace('_', '-')}", "src/a.py")
        (repo / "src" / "a.py").write_text("x = 3\n")
    else:
        _git(repo, "config", "core.fileMode", "false")
        os.chmod(repo / "src" / "a.py", 0o755)
    assert code_tree.tree_difference(repo) == expected
    assert P.working_tree_dirty(repo) is True


def test_a_removed_file_reads_deleted_even_when_the_index_forgot_it(repo: Path) -> None:
    """``git rm`` takes it out of the index; only HEAD's listing still names it."""
    _git(repo, "rm", "-q", "src/a.py")
    assert code_tree.tree_difference(repo) == "src/a.py: deleted"


def test_a_tracked_re_include_rule_is_honoured(repo: Path) -> None:
    """Round 2: ``cache/*`` with ``!cache/*.py`` makes ``cache/*.py`` code. An untracked
    ``cache/.gitignore`` hiding them must not be excused because the directory's CHILDREN are ignored."""
    (repo / "scripts" / "mb").mkdir()
    (repo / "scripts" / "mb" / ".gitignore").write_text("cache/*\n!cache/*.py\n")
    _git(repo, "add", "scripts/mb/.gitignore")
    _git(repo, "commit", "-q", "-m", "rules")
    (repo / "scripts" / "mb" / "cache").mkdir()
    (repo / "scripts" / "mb" / "cache" / ".gitignore").write_text("*.py\n.gitignore\n")
    (repo / "scripts" / "mb" / "cache" / "evil.py").write_text("print('hidden')\n")
    assert code_tree.tree_difference(repo) == "scripts/mb/cache/.gitignore: not in HEAD"


def test_a_negated_tracked_rule_does_not_excuse_a_directory(repo: Path) -> None:
    (repo / "scripts" / "mb").mkdir()
    (repo / "scripts" / "mb" / ".gitignore").write_text("cache/\n!cache/\n")
    _git(repo, "add", "scripts/mb/.gitignore")
    _git(repo, "commit", "-q", "-m", "rules")
    (repo / "scripts" / "mb" / "cache").mkdir()
    (repo / "scripts" / "mb" / "cache" / ".gitignore").write_text("*\n")
    (repo / "scripts" / "mb" / "cache" / "evil.py").write_text("print('hidden')\n")
    assert code_tree.tree_difference(repo) == "scripts/mb/cache/.gitignore: not in HEAD"


@pytest.mark.parametrize("how", ["repo_config", "environment"])
def test_a_per_user_excludes_file_cannot_pose_as_a_tracked_rule(repo: Path, monkeypatch, how) -> None:
    """Round 2: ``core.excludesFile`` naming a tracked ``.gitignore`` by its relative path made check-ignore
    report that tracked file as the source, while its rules applied repo-wide."""
    (repo / "scripts" / "mb").mkdir()
    (repo / "scripts" / "mb" / ".gitignore").write_text("vendored/\n")
    _git(repo, "add", "scripts/mb/.gitignore")
    _git(repo, "commit", "-q", "-m", "rules")
    if how == "repo_config":
        _git(repo, "config", "core.excludesFile", "scripts/mb/.gitignore")
    else:
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.excludesFile")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", "scripts/mb/.gitignore")
    (repo / "src" / "vendored").mkdir()
    (repo / "src" / "vendored" / ".gitignore").write_text("*\n")
    (repo / "src" / "vendored" / "evil.py").write_text("print('hidden')\n")
    assert code_tree.tree_difference(repo) == "src/vendored/.gitignore: not in HEAD"


def test_a_pathspec_variable_alone_cannot_blind_the_listing(repo: Path, monkeypatch) -> None:
    (repo / "src" / "pkg").mkdir()
    (repo / "src" / "pkg" / ".gitignore").write_text("*\n")
    (repo / "src" / "pkg" / "evil.py").write_text("print('hidden')\n")
    monkeypatch.setenv("GIT_LITERAL_PATHSPECS", "1")
    assert code_tree.tree_difference(repo) == "src/pkg/.gitignore: not in HEAD"


def test_an_untracked_root_gitignore_that_ignores_itself_reads_dirty(tmp_path: Path) -> None:
    repo = tmp_path / "bare"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")  # no root .gitignore in HEAD
    (repo / ".gitignore").write_text("*\n")
    (repo / "src" / "evil.py").write_text("print('hidden')\n")
    assert code_tree.tree_difference(repo) == ".gitignore: not in HEAD"


def test_only_the_top_of_the_work_tree_can_be_judged(repo: Path, tmp_path: Path) -> None:
    """Round 2: ``core.worktree`` (or a subdirectory) made every listing describe another tree."""
    assert code_tree.tree_difference(repo / "src").startswith(f"({repo / 'src'}): not the top of its git work tree")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _git(repo, "config", "core.worktree", str(elsewhere))
    (repo / "src" / "evil.py").write_text("print('hidden')\n")
    assert code_tree.tree_difference(repo) is not None
    assert code_tree.code_tree_sha256(repo) == "unknown"


def test_a_replace_ref_cannot_make_a_dirty_tree_read_clean(repo: Path) -> None:
    """``git replace`` swaps HEAD's tree for another in every ordinary read (review finding: this read
    clean while ``git status`` had caught it)."""
    head_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    (repo / "src" / "a.py").write_text("x = EVIL\n")
    _git(repo, "add", "src/a.py")
    evil_tree = _git(repo, "write-tree")
    _git(repo, "reset", "-q")
    _git(repo, "replace", head_tree, evil_tree)
    assert code_tree.tree_difference(repo) == "src/a.py: content differs from HEAD"


def test_case_insensitive_matching_cannot_hide_code(repo: Path) -> None:
    """``core.ignorecase`` let a case-only rename, or an untracked ``A.py`` beside ``a.py``, slip past."""
    if _case_insensitive(repo):
        (repo / "src" / "a.py").rename(repo / "src" / "A.py")
        expected = "src/A.py: not in HEAD"
    else:
        _git(repo, "config", "core.ignorecase", "true")
        (repo / "src" / "A.py").write_text("print('hidden')\n")
        expected = "src/A.py: not in HEAD"
    assert code_tree.tree_difference(repo) == expected


def test_a_leaked_git_variable_cannot_point_the_flag_or_the_hash_elsewhere(
    repo: Path, tmp_path: Path, monkeypatch
) -> None:
    """The other repo is clean AND holds exactly the edited content, so only the dropped variables keep the
    verdict honest (the first version of this test passed without the drop)."""
    (repo / "src" / "a.py").write_text("x = 2\n")
    other = _make_repo(tmp_path / "other")
    (other / "src" / "a.py").write_text("x = 2\n")
    _git(other, "commit", "-q", "-am", "same content as the dirty tree")
    own_hash = code_tree.head_commit(repo)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    monkeypatch.setenv("GIT_LITERAL_PATHSPECS", "1")
    assert code_tree.tree_difference(repo) == "src/a.py: content differs from HEAD"
    assert code_tree.head_commit(repo) == own_hash


def test_code_git_cannot_map_is_unknown_and_reads_dirty(repo: Path, tmp_path: Path) -> None:
    (repo / "src" / "sub").mkdir()
    oid = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{oid},src/sub")
    _git(repo, "commit", "-q", "-m", "submodule")
    assert code_tree.tree_difference(repo) == "src/sub: HEAD mode 160000 (a submodule or unknown entry)"
    assert "git cannot read it as a repository" in code_tree.tree_difference(tmp_path / "not-a-repo")


# ── the gate ─────────────────────────────────────────────────────────────


def test_the_harness_gate_refuses_and_names_what_git_status_could_not_see(repo: Path) -> None:
    _git(repo, "update-index", "--assume-unchanged", "src/a.py")
    (repo / "src" / "a.py").write_text("x = 3\n")
    with pytest.raises(P.DirtyTreeError, match=r"src/a\.py: content differs from HEAD"):
        P.preflight_gated_record(repo, repo / "docs" / "experiments" / "data" / "x.jsonl")


def test_a_clean_checkout_of_this_repo_reads_clean() -> None:
    """Known answer on the real tree: where ``git status`` sees nothing (a default-config checkout, as in
    CI), the content flag must agree. Skipped in a working tree with changes in scope."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", ".gitignore", "src", "scripts"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0 or status.stdout.strip():
        pytest.skip("this checkout has changes in scope")
    assert code_tree.tree_difference(REPO) is None


# ── one implementation, loaded the right way ─────────────────────────────


def test_the_harness_loads_this_file_from_its_own_tree() -> None:
    assert Path(P._code_tree.__file__).resolve() == CODE_TREE.resolve()
    assert "maxim" not in sys.modules.get("_provenance").__dict__  # it never imports the package


def test_code_tree_is_stdlib_only() -> None:
    """``scripts/_provenance.py`` loads it by path and must never import ``maxim`` (the installed package can
    be a different checkout), so the file itself may import only the standard library."""
    imported = set()
    for node in ast.walk(ast.parse(CODE_TREE.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported - set(sys.stdlib_module_names)
