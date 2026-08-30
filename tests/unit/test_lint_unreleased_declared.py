"""scripts/lint_unreleased_declared.py — roadmap 1.1.x item 16.10.

Verified to fail on the history that motivated it: #561/#562 changed
`runtime/agent_loop.py` and `motor_backend.py` one day after the v1.1.0 tag
with no bump and an empty `[Unreleased]`. ``test_src_change_without_bump_or_entry_fails``
is that shape.

The tests build real temporary git repositories rather than mocking git,
because the thing being checked IS a two-commit diff — a mocked diff would
test the parser and leave the plumbing (which is where the D37 silent-no-op
lived) unexercised.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import lint_unreleased_declared as L

CHANGELOG_EMPTY = "# Changelog\n\n## [Unreleased]\n\n## [1.1.1] - 2026-08-30\n\n- shipped\n"
CHANGELOG_FILLED = "# Changelog\n\n## [Unreleased]\n\n- Fixed the thing.\n\n## [1.1.1] - 2026-08-30\n\n- shipped\n"
PYPROJECT = '[project]\nname = "pymaxim"\nversion = "{v}"\n'


def _run(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "maxim").mkdir(parents=True)
    _run(repo.parent, "init", "-q", str(repo))
    _run(repo, "config", "user.email", "t@example.com")
    _run(repo, "config", "user.name", "T")
    _run(repo, "config", "commit.gpgsign", "false")
    (repo / "CHANGELOG.md").write_text(CHANGELOG_EMPTY)
    (repo / "pyproject.toml").write_text(PYPROJECT.format(v="1.1.1"))
    (repo / "src" / "maxim" / "mod.py").write_text("x = 1\n")
    (repo / "notes.md").write_text("hi\n")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-q", "-m", "base")
    return repo


def _commit(repo: Path, msg: str = "change") -> None:
    _run(repo, "add", "-A")
    _run(repo, "commit", "-q", "-m", msg)


def _base(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD~1"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


# ── the failing shape (the #561/#562 history) ────────────────────────────────


def test_src_change_without_bump_or_entry_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "src" / "maxim" / "mod.py").write_text("x = 2\n")
    _commit(repo)
    fails = L.violations(repo, _base(repo))
    assert len(fails) == 1
    assert "src/" in fails[0] and "[Unreleased]" in fails[0]


# ── the two ways to satisfy it ───────────────────────────────────────────────


def test_src_change_with_unreleased_entry_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "src" / "maxim" / "mod.py").write_text("x = 2\n")
    (repo / "CHANGELOG.md").write_text(CHANGELOG_FILLED)
    _commit(repo)
    assert L.violations(repo, _base(repo)) == []


def test_src_change_with_version_bump_passes(tmp_path: Path) -> None:
    """A release transaction; lint_version_sync.py owns the rest of it."""
    repo = _repo(tmp_path)
    (repo / "src" / "maxim" / "mod.py").write_text("x = 2\n")
    (repo / "pyproject.toml").write_text(PYPROJECT.format(v="1.1.2"))
    _commit(repo)
    assert L.violations(repo, _base(repo)) == []


def test_pyproject_touched_without_a_version_change_does_not_excuse(tmp_path: Path) -> None:
    """Editing pyproject is not a bump. The escape hatch is the VERSION changing."""
    repo = _repo(tmp_path)
    (repo / "src" / "maxim" / "mod.py").write_text("x = 2\n")
    (repo / "pyproject.toml").write_text(PYPROJECT.format(v="1.1.1") + "\n# a comment\n")
    _commit(repo)
    assert len(L.violations(repo, _base(repo))) == 1


# ── what the lint must NOT fire on ───────────────────────────────────────────


def test_docs_only_change_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "notes.md").write_text("changed\n")
    _commit(repo)
    assert L.violations(repo, _base(repo)) == []


def test_blank_line_added_to_unreleased_is_not_content(tmp_path: Path) -> None:
    """An empty section and one with a blank line make the same claim."""
    repo = _repo(tmp_path)
    (repo / "src" / "maxim" / "mod.py").write_text("x = 2\n")
    (repo / "CHANGELOG.md").write_text(CHANGELOG_EMPTY.replace("## [Unreleased]\n", "## [Unreleased]\n\n   \n"))
    _commit(repo)
    assert len(L.violations(repo, _base(repo))) == 1


# ── the section parser ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        (CHANGELOG_EMPTY, []),
        (CHANGELOG_FILLED, ["- Fixed the thing."]),
        ("## [Unreleased]\n- a\n- b\n## [1.0.0] - x\n- c\n", ["- a", "- b"]),
        ("# Changelog\n\n## [1.0.0] - x\n- c\n", []),
    ],
)
def test_unreleased_lines(text, expected):
    assert L.unreleased_lines(text) == expected


def test_unreleased_section_stops_at_the_next_heading():
    """A released section's bullets must never be counted as Unreleased content."""
    text = "## [Unreleased]\n\n## [1.1.1] - 2026-08-30\n\n- lots\n- of\n- entries\n"
    assert L.unreleased_lines(text) == []
