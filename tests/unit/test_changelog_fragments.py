"""scripts/changelog_fragments.py: M83 step 3 (CHANGELOG fragments in ``changelog.d/``).

Design and its four adversarial rounds: docs/plans/changelog_fragments_m83_approach.md. Every rule there has a
positive control here, and every round finding that asked for a named test has one, tagged with its id (S1-S5,
R1-R5, F1-F3, N*). The #561/#562 shape that motivated the old ``lint_unreleased_declared`` stays as
``test_src_change_without_fragment_fails``.

Like the lints this replaces, the tests build real temporary git repositories rather than mocking git: the
thing being checked IS a diff, and the plumbing (rename and copy detection, first-parent units) is where the
fail-open holes the rounds found lived.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

try:
    from scripts import changelog_fragments as C
except ImportError:  # red gate: the script is not built yet
    C = None  # type: ignore[assignment]

pytestmark = pytest.mark.xfail(strict=True, reason="M83 step 3: scripts/changelog_fragments.py not built yet")

CHANGELOG = "# Changelog\n\n## [Unreleased]\n\n{unreleased}## [1.1.1] - 2026-08-30\n\n- shipped\n"
PYPROJECT = '[project]\nname = "pymaxim"\nversion = "{v}"\n'
README = "# changelog.d\n\nOne file per change: `<slug>.<kind>.md`.\n"
LONG = (
    "- **Talk remembers what you told it in an earlier session** (#1444). Six defects composed into that one "
    "symptom: your words were never memorized, a recalled memory showed only its tool, stored tool results were "
    "their Python repr, recall ignored your question, and the first question after a restart was flaky.\n"
)


def _run(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def _repo(tmp_path: Path, unreleased: str = "", fragments: dict[str, str] | None = None) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "maxim").mkdir(parents=True)
    (repo / "changelog.d").mkdir()
    _run(repo.parent, "init", "-q", "-b", "main", str(repo))
    for k, v in (("user.email", "t@example.com"), ("user.name", "T"), ("commit.gpgsign", "false")):
        _run(repo, "config", k, v)
    (repo / "CHANGELOG.md").write_text(CHANGELOG.format(unreleased=unreleased))
    (repo / "pyproject.toml").write_text(PYPROJECT.format(v="1.1.1"))
    (repo / "src" / "maxim" / "mod.py").write_text("x = 1\n")
    (repo / "changelog.d" / "README.md").write_text(README)
    (repo / "notes.md").write_text("hi\n")
    for name, body in (fragments or {}).items():
        (repo / "changelog.d" / name).write_text(body)
    _commit(repo, "base")
    _run(repo, "tag", "v1.1.1")
    return repo


def _commit(repo: Path, msg: str = "change") -> str:
    _run(repo, "add", "-A")
    _run(repo, "commit", "-q", "--allow-empty", "-m", msg)
    return _run(repo, "rev-parse", "HEAD").strip()


def _head(repo: Path) -> str:
    return _run(repo, "rev-parse", "HEAD").strip()


def _src_change(repo: Path, value: str = "2") -> None:
    (repo / "src" / "maxim" / "mod.py").write_text(f"x = {value}\n")


def _frag(repo: Path, name: str, body: str = "- Fixed the thing.\n") -> None:
    (repo / "changelog.d" / name).write_text(body)


def _bump(repo: Path, version: str = "1.2.0") -> None:
    (repo / "pyproject.toml").write_text(PYPROJECT.format(v=version))
    C.assemble(repo, version=version, date="2026-10-28", name="Test")


def _violations(repo: Path, base: str) -> list[str]:
    return C.violations(repo, base, "HEAD")


# ── rule 1: a src/ change carries a NEW fragment ─────────────────────────────


def test_src_change_without_fragment_fails(tmp_path: Path) -> None:
    """The #561/#562 shape: src changed, nothing declared."""
    repo = _repo(tmp_path)
    base = _head(repo)
    _src_change(repo)
    _commit(repo)
    fails = _violations(repo, base)
    assert fails and any("changelog.d" in f for f in fails)


def test_src_change_with_new_fragment_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _head(repo)
    _src_change(repo)
    _frag(repo, "mod-fix.fixed.md")
    _commit(repo)
    assert _violations(repo, base) == []


def test_non_python_src_change_needs_a_fragment_too(tmp_path: Path) -> None:
    """S5: the old declared lint's scope was any src/ path; YAML and packaged data ship in the wheel."""
    repo = _repo(tmp_path)
    base = _head(repo)
    (repo / "src" / "maxim" / "body.yaml").write_text("a: 1\n")
    _commit(repo)
    assert _violations(repo, base)


def test_docs_only_change_needs_no_fragment(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _head(repo)
    (repo / "notes.md").write_text("changed\n")
    _commit(repo)
    assert _violations(repo, base) == []


def test_renaming_another_prs_fragment_does_not_count(tmp_path: Path) -> None:
    """S1: under --no-renames a rename listed as an ADD; it must neither satisfy rule 1 nor pass rule 2."""
    repo = _repo(tmp_path, fragments={"other.fixed.md": "- Another PR's entry.\n"})
    base = _head(repo)
    _src_change(repo)
    _run(repo, "mv", "changelog.d/other.fixed.md", "changelog.d/mine.fixed.md")
    _commit(repo)
    fails = _violations(repo, base)
    assert any("new fragment" in f for f in fails)
    assert any("other.fixed.md" in f for f in fails)


def test_exact_copy_of_an_untouched_fragment_is_refused(tmp_path: Path) -> None:
    """R2: -C without --find-copies-harder lists a copy of an untouched file as A."""
    repo = _repo(tmp_path, fragments={"other.fixed.md": LONG})
    base = _head(repo)
    _src_change(repo)
    _frag(repo, "mine.fixed.md", LONG)
    _commit(repo)
    assert any("other.fixed.md" in f and "mine.fixed.md" in f for f in _violations(repo, base))


def test_long_copy_with_one_word_changed_is_refused(tmp_path: Path) -> None:
    """F2: difflib's default autojunk scored a long one-word-changed copy 0.595; autojunk=False is required."""
    assert len(LONG) >= 200
    repo = _repo(tmp_path, fragments={"other.fixed.md": LONG})
    base = _head(repo)
    _src_change(repo)
    _frag(repo, "mine.fixed.md", LONG.replace("flaky", "racy"))
    _commit(repo)
    assert any("similar" in f for f in _violations(repo, base))


def test_two_short_typo_fixes_are_not_copies(tmp_path: Path) -> None:
    """F2's floor: short entries score about 0.95 against each other without being copies."""
    repo = _repo(tmp_path, fragments={"typo-x.fixed.md": "- Fixed a typo in X.\n"})
    base = _head(repo)
    _src_change(repo)
    _frag(repo, "typo-y.fixed.md", "- Fixed a typo in Y.\n")
    _commit(repo)
    assert _violations(repo, base) == []


# ── rule 2: no fragment leaves outside a release ─────────────────────────────


def test_a_docs_only_pr_deleting_a_fragment_fails(tmp_path: Path) -> None:
    """S2: the deletion is checked whether or not src changed."""
    repo = _repo(tmp_path, fragments={"other.fixed.md": "- Another PR's entry.\n"})
    base = _head(repo)
    _run(repo, "rm", "-q", "changelog.d/other.fixed.md")
    _commit(repo)
    assert any("other.fixed.md" in f for f in _violations(repo, base))


def test_rewording_a_fragment_is_allowed(tmp_path: Path) -> None:
    repo = _repo(tmp_path, fragments={"other.fixed.md": "- Another PR's entry.\n"})
    base = _head(repo)
    _frag(repo, "other.fixed.md", "- Another PR's entry, reworded.\n")
    _commit(repo)
    assert _violations(repo, base) == []


# ── rule 3: a bump consumes every fragment exactly once ──────────────────────


def test_a_bump_with_assemble_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n", "b.added.md": "- B.\n"})
    base = _head(repo)
    _bump(repo)
    _commit(repo, "bump")
    assert _violations(repo, base) == []
    assert not (repo / "changelog.d" / "a.fixed.md").exists()


def test_a_bump_that_deletes_a_fragment_without_its_marker_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n", "b.added.md": "- B.\n"})
    base = _head(repo)
    _bump(repo)
    text = (repo / "CHANGELOG.md").read_text().replace("<!-- changelog.d/b.added.md -->\n", "")
    (repo / "CHANGELOG.md").write_text(text)
    _commit(repo, "bump")
    assert any("b.added.md" in f for f in _violations(repo, base))


def test_a_bump_that_leaves_a_fragment_fails(tmp_path: Path) -> None:
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    base = _head(repo)
    _bump(repo)
    _frag(repo, "late.fixed.md", "- Late.\n")
    _commit(repo, "bump")
    assert any("late.fixed.md" in f for f in _violations(repo, base))


def test_a_marker_whose_entry_was_deleted_fails_even_after_moved_bullets(tmp_path: Path) -> None:
    """R1/F3: the fixture fragment is LAST in a kind block that also holds moved [Unreleased] bullets, so a
    marker left behind by the reword pass would sit right before someone else's unmarked bullet if assemble
    ever put fragments first."""
    repo = _repo(
        tmp_path,
        unreleased="### Fixed\n\n- Moved one.\n- Moved two.\n\n",
        fragments={"z.fixed.md": "- Z.\n"},
    )
    base = _head(repo)
    _bump(repo)
    text = (repo / "CHANGELOG.md").read_text()
    assert text.index("- Moved two.") < text.index("<!-- changelog.d/z.fixed.md -->")
    (repo / "CHANGELOG.md").write_text(text.replace("- Z.\n", ""))
    _commit(repo, "bump")
    assert any("z.fixed.md" in f and "entry" in f for f in _violations(repo, base))


def test_a_stray_comment_in_the_new_section_fails(tmp_path: Path) -> None:
    """R1: a stray <!-- ... --> around the markers would hide the whole section."""
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    base = _head(repo)
    _bump(repo)
    text = (repo / "CHANGELOG.md").read_text().replace("- A.\n", "- A.\n<!-- hidden -->\n")
    (repo / "CHANGELOG.md").write_text(text)
    _commit(repo, "bump")
    assert any("<!--" in f for f in _violations(repo, base))


def test_duplicate_and_forged_markers_fail(tmp_path: Path) -> None:
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    base = _head(repo)
    _bump(repo)
    text = (repo / "CHANGELOG.md").read_text()
    text = text.replace("- A.\n", "- A.\n\n<!-- changelog.d/a.fixed.md -->\n- Again.\n\n<!-- changelog.d/ghost.fixed.md -->\n- Ghost.\n")
    (repo / "CHANGELOG.md").write_text(text)
    _commit(repo, "bump")
    fails = _violations(repo, base)
    assert any("a.fixed.md" in f and "once" in f for f in fails)
    assert any("ghost.fixed.md" in f for f in fails)


def test_a_fragment_added_and_assembled_inside_the_bump_branch_passes(tmp_path: Path) -> None:
    """R4: absent at both ends of the net diff, but added by a commit in the range."""
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    base = _head(repo)
    _frag(repo, "review-fold.fixed.md", "- A review fold.\n")
    _commit(repo, "fold")
    _bump(repo)
    _commit(repo, "bump")
    assert _violations(repo, base) == []


def test_a_fragment_renamed_during_review_inside_the_bump_branch_passes(tmp_path: Path) -> None:
    """F1: the permitted set uses --no-renames, so both names are known."""
    repo = _repo(tmp_path)
    base = _head(repo)
    _frag(repo, "tmp-name.fixed.md", "- Renamed later.\n")
    _commit(repo, "add")
    _run(repo, "mv", "changelog.d/tmp-name.fixed.md", "changelog.d/final-name.fixed.md")
    _commit(repo, "rename")
    _bump(repo)
    _commit(repo, "bump")
    assert _violations(repo, base) == []


def test_two_headings_for_one_version_fail(tmp_path: Path) -> None:
    """R3: a re-run must merge into the existing section, never add a second heading."""
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    base = _head(repo)
    _bump(repo)
    text = (repo / "CHANGELOG.md").read_text()
    text = text.replace("## [1.1.1]", '## [1.2.0] - 2026-10-28 — "Dup"\n\n- x\n\n## [1.1.1]')
    (repo / "CHANGELOG.md").write_text(text)
    _commit(repo, "bump")
    assert any("1.2.0" in f and "heading" in f for f in _violations(repo, base))


# ── rule 4: [Unreleased] does not grow; no pre-written sections ──────────────


def test_unreleased_growth_fails_outside_a_bump(tmp_path: Path) -> None:
    repo = _repo(tmp_path, unreleased="- Old entry.\n\n")
    base = _head(repo)
    _src_change(repo)
    _frag(repo, "x.fixed.md")
    (repo / "CHANGELOG.md").write_text(CHANGELOG.format(unreleased="- Old entry.\n- New entry.\n\n"))
    _commit(repo)
    assert any("[Unreleased]" in f for f in _violations(repo, base))


def test_rewording_unreleased_without_growth_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path, unreleased="- Old entry.\n\n")
    base = _head(repo)
    (repo / "CHANGELOG.md").write_text(CHANGELOG.format(unreleased="- Old entry, reworded.\n\n"))
    _commit(repo)
    assert _violations(repo, base) == []


def test_a_comment_line_counts_as_unreleased_growth(tmp_path: Path) -> None:
    """n8: HTML-comment lines are content for rule 4."""
    repo = _repo(tmp_path)
    base = _head(repo)
    (repo / "CHANGELOG.md").write_text(CHANGELOG.format(unreleased="<!-- note -->\n\n"))
    _commit(repo)
    assert _violations(repo, base)


def test_a_pre_written_version_section_fails_outside_a_bump(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _head(repo)
    text = (repo / "CHANGELOG.md").read_text().replace("## [1.1.1]", "## [1.2.0] - TBD\n\n- soon\n\n## [1.1.1]")
    (repo / "CHANGELOG.md").write_text(text)
    _commit(repo)
    assert any("## [" in f for f in _violations(repo, base))


# ── rule 5: every fragment is valid ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "body", "needle"),
    [
        ("bad.typo.md", "- x\n", "kind"),
        ("Bad_Name.fixed.md", "- x\n", "name"),
        ("ok.fixed.md", "", "empty"),
        ("ok.fixed.md", "Not a bullet.\n", "- "),
        ("ok.fixed.md", "- x\n\n## Heading\n", "heading"),
        ("ok.fixed.md", "- x <!-- changelog.d/other.fixed.md -->\n", "<!--"),
    ],
)
def test_an_invalid_fragment_fails(tmp_path: Path, name: str, body: str, needle: str) -> None:
    repo = _repo(tmp_path)
    base = _head(repo)
    _frag(repo, name, body)
    _commit(repo)
    assert any(needle in f for f in _violations(repo, base)), _violations(repo, base)


def test_a_fragment_in_a_subdirectory_or_a_symlink_fails(tmp_path: Path) -> None:
    """n6."""
    repo = _repo(tmp_path)
    base = _head(repo)
    (repo / "changelog.d" / "sub").mkdir()
    _frag(repo, "sub/x.fixed.md")
    os.symlink("../notes.md", repo / "changelog.d" / "link.fixed.md")
    _commit(repo)
    fails = _violations(repo, base)
    assert any("sub/x.fixed.md" in f for f in fails)
    assert any("link.fixed.md" in f for f in fails)


# ── assemble ─────────────────────────────────────────────────────────────────


def test_assemble_merges_headings_by_kind_and_keeps_subtitles(tmp_path: Path) -> None:
    """N3/n3: one plain ### Fixed, a subtitle heading kept verbatim after its kind, Breaking first, no line lost."""
    unreleased = (
        "### Fixed\n\n- F one.\n\n### Changed\n\n- C one.\n\n### Changed — release integrity\n\n- RI one.\n\n"
        "### Fixed\n\n- F two.\n\n### Oddities\n\n- O one.\n\n"
    )
    repo = _repo(tmp_path, unreleased=unreleased, fragments={"brk.breaking.md": "- B.\n", "f3.fixed.md": "- F3.\n"})
    before = (repo / "CHANGELOG.md").read_text()
    _bump(repo)
    text = (repo / "CHANGELOG.md").read_text()
    section = text.split('## [1.2.0] - 2026-10-28 — "Test"', 1)[1].split("## [1.1.1]", 1)[0]
    headings = [line for line in section.splitlines() if line.startswith("### ")]
    assert headings == ["### Breaking", "### Changed", "### Changed — release integrity", "### Fixed", "### Oddities"]
    assert section.index("- F two.") < section.index("<!-- changelog.d/f3.fixed.md -->")
    old_bullets = sorted(line for line in before.splitlines() if line.startswith("- ") and line != "- shipped")
    new_bullets = sorted(line for line in section.splitlines() if line.startswith("- "))
    assert new_bullets == sorted(old_bullets + ["- B.", "- F3."])
    assert "## [Unreleased]" in text and not C.unreleased_lines(text)


def test_assemble_reruns_into_the_existing_section(tmp_path: Path) -> None:
    """N2/R3: assemble -> reword -> a new fragment arrives -> assemble again: one heading, the reword kept."""
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    base = _head(repo)
    _bump(repo)
    _commit(repo, "bump")
    text = (repo / "CHANGELOG.md").read_text().replace("- A.\n", "- A, reworded.\n")
    (repo / "CHANGELOG.md").write_text(text)
    _frag(repo, "b.fixed.md", "- B.\n")
    _commit(repo, "main brought b")
    C.assemble(repo, version="1.2.0", date="2026-10-28", name="Test")
    _commit(repo, "re-assemble")
    text = (repo / "CHANGELOG.md").read_text()
    assert text.count("## [1.2.0]") == 1
    assert "- A, reworded." in text and "- B." in text
    assert text.index("<!-- changelog.d/a.fixed.md -->") < text.index("<!-- changelog.d/b.fixed.md -->")
    assert _violations(repo, base) == []


# ── a push range is judged one first-parent unit at a time ───────────────────


def _merge_branch(repo: Path, branch: str, work) -> None:
    _run(repo, "checkout", "-q", "-b", branch)
    work()
    _commit(repo, branch)
    _run(repo, "checkout", "-q", "main")
    _run(repo, "merge", "-q", "--no-ff", "-m", f"Merge {branch}", branch)


def test_a_src_pr_merging_after_the_bump_in_one_push_range_passes(tmp_path: Path) -> None:
    """F1/N6: a cancelled push run puts the bump and a later PR in one range. Each is judged as its own unit, so
    the later PR's fragment is not "left over" by the bump."""
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    base = _head(repo)
    _merge_branch(repo, "bump", lambda: _bump(repo))

    def later() -> None:
        _src_change(repo, "3")
        _frag(repo, "later.fixed.md", "- Later.\n")

    _merge_branch(repo, "later", later)
    assert C.range_violations(repo, base) == []


def test_a_bad_unit_in_a_push_range_is_still_caught(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _head(repo)
    _merge_branch(repo, "ok", lambda: (_src_change(repo, "2"), _frag(repo, "ok.fixed.md")))
    _merge_branch(repo, "bad", lambda: _src_change(repo, "3"))
    assert C.range_violations(repo, base)


# ── the tag-time check (check_release_tag.py notes) ──────────────────────────


def test_tag_time_check_only_bites_when_untagged_or_on_a_tag_push(tmp_path: Path, monkeypatch) -> None:
    """N4/R5: a pre-bump rehearsal (pyproject names a tagged release) must not fail on the cycle's fragments."""
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    monkeypatch.delenv("GITHUB_REF", raising=False)
    assert C.leftover_fragment_problems(repo, "1.1.1") == []  # tagged, a rehearsal
    assert C.leftover_fragment_problems(repo, "1.2.0")  # untagged: the post-bump state
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v1.1.1")
    assert C.leftover_fragment_problems(repo, "1.1.1")  # a tag push always checks


# ── the D37 rule: on a pull request a missing base is a hard error ───────────


def test_no_base_ref_on_a_pull_request_is_a_hard_error(monkeypatch) -> None:
    def _no_base(_root):
        raise C.GitUnavailable("no merge-base")

    monkeypatch.setattr(C, "base_ref", _no_base)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    assert C.main(["check"]) == 2
