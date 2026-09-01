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
    assert len(fails) == 1 and "without tests/" in fails[0]


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
    assert "declared: log-message wording only" in capsys.readouterr().out


def test_subject_match_is_case_insensitive_prefix(repo) -> None:
    root, base = repo
    _commit(root, "Fix: crash", {"src/a.py": "x = 6\n"})
    assert len(L.violations(root, base)) == 1


# ── the PR title is the subject that squash-merges onto main (review-round BLOCKER) ──


def test_pr_title_is_checked_against_the_aggregate_diff(repo) -> None:
    """A branch whose commits say `wip` but whose PR title says `fix(...)`: main gets the
    TITLE as its subject, so the first draft (branch commits only) gated nothing."""
    root, base = repo
    _commit(root, "wip", {"src/a.py": "x = 9\n"})
    assert L.violations(root, base) == []
    fails = L.violations(root, base, pr_title="fix(api): honor the contract")
    assert len(fails) == 1 and "squash-merged subject on main" in fails[0]


def test_pr_title_with_tests_in_the_aggregate_diff_passes(repo) -> None:
    root, base = repo
    _commit(root, "wip", {"src/a.py": "x = 9\n"})
    _commit(root, "wip 2", {"tests/test_a.py": "def test_a(): assert 1\n"})
    assert L.violations(root, base, pr_title="fix(api): honor the contract") == []


def test_pr_title_opt_out_marker(repo, capsys) -> None:
    root, base = repo
    _commit(root, "wip", {"src/a.py": "x = 9\n"})
    assert L.violations(root, base, pr_title="fix(api): typo [no-tests: comment only]") == []
    assert "declared: comment only" in capsys.readouterr().out
    assert L.violations(root, base, pr_title="fix(api): typo", pr_body="body [no-tests: log wording]") == []


def test_non_fix_pr_title_is_not_judged(repo) -> None:
    root, base = repo
    _commit(root, "wip", {"src/a.py": "x = 9\n"})
    assert L.violations(root, base, pr_title="feat(api): new verb") == []


def test_merge_commits_on_the_branch_are_skipped_with_a_note(repo, capsys) -> None:
    root, base = repo
    _git(root, "checkout", "-q", "-b", "side")
    _commit(root, "chore: side", {"src/b.py": "y = 1\n"})
    _git(root, "checkout", "-q", "main")
    _commit(root, "chore: main", {"src/c.py": "z = 1\n"})
    _git(root, "merge", "-q", "--no-ff", "-m", "fix: merge branches", "side")
    assert L.violations(root, base) == []
    assert "is a merge commit — skipped" in capsys.readouterr().out


def test_step_summary_receives_declared_exemptions(repo, tmp_path, monkeypatch) -> None:
    root, base = repo
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    _commit(root, "fix(loop): log line", {"src/a.py": "x = 5\n"}, body="No-Tests-Reason: wording only")
    assert L.violations(root, base) == []
    assert "wording only" in summary.read_text()


# ── the documented PR-body escape must cover BRANCH COMMITS too ──────────────
#
# The module docstring has always promised "a `No-Tests-Reason: <why>` trailer
# in the commit body, or `[no-tests: <why>]` in the PR title/body". Until
# 2026-08-31 the marker was only read for the PR-TITLE population, so the
# documented escape did nothing for the per-commit population — which is the one
# that actually fails while a branch is being written (found on PR #579: a
# docstring-only correction to `similarity/ec.py` in a `fix(...)` commit).
#
# A promised escape that silently does not apply is worse than no escape: the
# author reads the advice, follows it, and the gate stays red with the same
# message.


def test_pr_body_marker_exempts_a_branch_commit(repo, capsys) -> None:
    root, base = repo
    _commit(root, "fix(ec): correct a docstring claim", {"src/a.py": "x = 2\n"})
    out = L.violations(root, base, pr_body="[no-tests: docstring-only correction]")
    assert out == []
    assert "docstring-only correction" in capsys.readouterr().out


def test_pr_title_marker_exempts_a_branch_commit(repo, capsys) -> None:
    root, base = repo
    _commit(root, "fix(ec): correct a docstring claim", {"src/a.py": "x = 2\n"})
    assert L.violations(root, base, pr_title="chore: docs [no-tests: prose only]") == []


def test_the_rule_still_bites_without_any_marker(repo) -> None:
    """The escape must not become the default. Same commit, no marker, fails."""
    root, base = repo
    _commit(root, "fix(ec): correct a docstring claim", {"src/a.py": "x = 2\n"})
    out = L.violations(root, base)
    assert len(out) == 1
    assert "without tests/" in out[0]


def test_an_unrelated_pr_body_does_not_exempt(repo) -> None:
    """Only the marker exempts — arbitrary prose must not."""
    root, base = repo
    _commit(root, "fix(ec): correct a docstring claim", {"src/a.py": "x = 2\n"})
    out = L.violations(root, base, pr_body="This PR has no tests because it is docs.")
    assert len(out) == 1


def test_marker_exempts_every_offending_commit_on_the_branch(repo, capsys) -> None:
    """One PR-body marker covers the branch — that is the intent, and it is
    stated so a reviewer can see the scope rather than infer it."""
    root, base = repo
    _commit(root, "fix(a): one", {"src/a.py": "x = 2\n"})
    _commit(root, "fix(b): two", {"src/b.py": "y = 1\n"})
    assert L.violations(root, base, pr_body="[no-tests: prose]") == []
    assert capsys.readouterr().out.count("declared: prose") == 2
