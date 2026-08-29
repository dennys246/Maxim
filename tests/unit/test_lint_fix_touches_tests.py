"""Fixture-repo tests for scripts/lint_fix_touches_tests.py (roadmap 1.1.x item 16.2).

Positive control for the CI step: a `fix(...)` commit touching src/ without tests/ must be a
violation. Verified to fail 3/3 order-sensitive cases if the src/tests predicate is inverted.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts import lint_fix_touches_tests as L


def _git(root: Path, *args: str) -> str:
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/a.py").write_text("x = 1\n")
    (tmp_path / "tests/test_a.py").write_text("def test_a(): pass\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD").strip()
    return tmp_path, base


def _commit(root: Path, subject: str, files: dict[str, str], body: str = "") -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", subject, *(["-m", body] if body else []))


def test_fix_touching_src_without_tests_is_a_violation(repo) -> None:
    root, base = repo
    _commit(root, "fix(loop): abort path", {"src/a.py": "x = 2\n"})
    fails = L.violations(root, base)
    assert len(fails) == 1 and "without touching tests/" in fails[0]


def test_fix_with_tests_passes(repo) -> None:
    root, base = repo
    _commit(root, "fix(loop): abort path", {"src/a.py": "x = 2\n", "tests/test_a.py": "def test_a(): assert 1\n"})
    assert L.violations(root, base) == []


def test_non_fix_commits_and_non_src_fixes_are_not_judged(repo) -> None:
    root, base = repo
    _commit(root, "feat: thing", {"src/a.py": "x = 3\n"})
    _commit(root, "fix(docs): typo", {"README.md": "hi\n"})
    _commit(root, "refactor: prefix trap", {"src/a.py": "x = 4\n"})
    assert L.violations(root, base) == []


def test_opt_out_trailer_is_accepted_and_echoed(repo, capsys) -> None:
    root, base = repo
    _commit(root, "fix(loop): log line", {"src/a.py": "x = 5\n"}, body="No-Tests-Reason: log-message wording only")
    assert L.violations(root, base) == []
    assert "No-Tests-Reason: log-message wording only" in capsys.readouterr().out


def test_subject_match_is_case_insensitive_prefix(repo) -> None:
    root, base = repo
    _commit(root, "Fix: crash", {"src/a.py": "x = 6\n"})
    assert len(L.violations(root, base)) == 1
