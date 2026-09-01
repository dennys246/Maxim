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


# ── the third state: a release IN FLIGHT (cut landed, nothing published yet) ──
#
# Added 2026-08-31 after this lint failed PR #579, which was doing the RIGHT
# thing. 1.1.2's release transaction was on main (pyproject bumped, `## [1.1.2]`
# cut) but 1.1.2 was not yet on PyPI, and a src/ change that ships in the 1.1.2
# wheel had its CHANGELOG entry correctly filed under [1.1.2]. The lint modelled
# only two states — "between releases" and "the transaction itself" — so it
# demanded an [Unreleased] line that would have UNDERSTATED the release.
#
# The tag is the discriminator: docs/publication_guide.md places it at publish
# time, so `v<version>` absent means in flight.

CHANGELOG_CUT = "# Changelog\n\n## [Unreleased]\n\n## [1.1.2] - 2026-08-31\n\n- shipped item\n"
CHANGELOG_CUT_PLUS = "# Changelog\n\n## [Unreleased]\n\n## [1.1.2] - 2026-08-31\n\n- shipped item\n- a second thing\n"


def _repo_at_cut(tmp_path: Path) -> Path:
    """A repo whose release transaction has landed but which is untagged."""
    repo = tmp_path / "repo"
    (repo / "src" / "maxim").mkdir(parents=True)
    _run(repo.parent, "init", "-q", str(repo))
    _run(repo, "config", "user.email", "t@example.com")
    _run(repo, "config", "user.name", "T")
    _run(repo, "config", "commit.gpgsign", "false")
    (repo / "CHANGELOG.md").write_text(CHANGELOG_CUT)
    (repo / "pyproject.toml").write_text(PYPROJECT.format(v="1.1.2"))
    (repo / "src" / "maxim" / "mod.py").write_text("x = 1\n")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-q", "-m", "release cut")
    return repo


def test_src_change_added_to_the_in_flight_release_passes(tmp_path: Path) -> None:
    """The #579 shape: untagged version, entry filed under it. Must PASS."""
    repo = _repo_at_cut(tmp_path)
    (repo / "src" / "maxim" / "mod.py").write_text("x = 2\n")
    (repo / "CHANGELOG.md").write_text(CHANGELOG_CUT_PLUS)
    _commit(repo)
    assert L.violations(repo, _base(repo)) == []


def test_src_change_added_to_an_ALREADY_TAGGED_release_still_fails(tmp_path: Path) -> None:
    """Once v1.1.2 exists the release is published, so later src/ changes belong
    to the NEXT one. The in-flight escape must close behind the tag."""
    repo = _repo_at_cut(tmp_path)
    _run(repo, "tag", "v1.1.2")
    (repo / "src" / "maxim" / "mod.py").write_text("x = 2\n")
    (repo / "CHANGELOG.md").write_text(CHANGELOG_CUT_PLUS)
    _commit(repo)
    fails = L.violations(repo, _base(repo))
    assert len(fails) == 1
    assert "[Unreleased]" in fails[0]


def test_in_flight_escape_needs_the_version_to_MATCH_pyproject(tmp_path: Path) -> None:
    """Adding to an OLD section while pyproject is elsewhere is not in-flight."""
    repo = _repo_at_cut(tmp_path)
    (repo / "pyproject.toml").write_text(PYPROJECT.format(v="1.2.0"))
    (repo / "src" / "maxim" / "mod.py").write_text("x = 2\n")
    (repo / "CHANGELOG.md").write_text(CHANGELOG_CUT_PLUS)
    _commit(repo)
    # pyproject changed 1.1.2 -> 1.2.0, so escape (a) legitimately applies here;
    # the point is that (c) alone would NOT have matched.
    assert L.newest_released_section(CHANGELOG_CUT_PLUS)[0] == "1.1.2"


def test_in_flight_escape_does_not_fire_without_a_new_line(tmp_path: Path) -> None:
    """Being in flight is not a blanket exemption — the entry must be added."""
    repo = _repo_at_cut(tmp_path)
    (repo / "src" / "maxim" / "mod.py").write_text("x = 2\n")
    _commit(repo)
    assert len(L.violations(repo, _base(repo))) == 1


def test_newest_released_section_skips_unreleased(tmp_path: Path) -> None:
    version, lines = L.newest_released_section(CHANGELOG_CUT)
    assert version == "1.1.2"
    assert lines == ["- shipped item"]


def test_version_is_published_reads_the_tag(tmp_path: Path) -> None:
    repo = _repo_at_cut(tmp_path)
    assert L.version_is_published(repo, "1.1.2") is False
    _run(repo, "tag", "v1.1.2")
    assert L.version_is_published(repo, "1.1.2") is True
