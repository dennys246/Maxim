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

from scripts import changelog_fragments as C

CHANGELOG = "# Changelog\n\n## [Unreleased]\n\n{unreleased}## [1.1.1] - 2026-08-30\n\n- shipped\n"
PYPROJECT = '[project]\nname = "pymaxim"\nversion = "{v}"\n'
README = "# changelog.d\n\nOne file per change: `<slug>.<kind>.md`.\n"
# A real 1.3.x bullet (245 characters). With difflib's default autojunk, the copy below with ONE word changed
# ("never" -> "racy") scores 0.491 and would pass the near-duplicate check; with autojunk=False it scores 0.985.
# Found by searching every 200+ character CHANGELOG bullet; it pins F2, which the deletion probe proved a shorter
# synthetic fixture did not.
LONG = (
    "- **`simulation/sources.py` no longer implies live percept sharing shipped** (D46): it names the remote adapter as\n"
    "  never written, the mesh percept message types as having no producer, and `deferred/mesh_perception_transport.md`\n"
    "  as the owner.\n"
)
LONG_ONE_WORD_CHANGED = LONG.replace("never written", "racy written")


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
    assert C.similarity(LONG, LONG_ONE_WORD_CHANGED) >= C.SIMILARITY
    repo = _repo(tmp_path, fragments={"other.fixed.md": LONG})
    base = _head(repo)
    _src_change(repo)
    _frag(repo, "mine.fixed.md", LONG_ONE_WORD_CHANGED)
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
    text = text.replace(
        "- A.\n", "- A.\n\n<!-- changelog.d/a.fixed.md -->\n- Again.\n\n<!-- changelog.d/ghost.fixed.md -->\n- Ghost.\n"
    )
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


def test_a_rerun_takes_the_heading_from_its_arguments(tmp_path: Path) -> None:
    """Code round SF2: a corrected --date or --name must not be silently dropped on a re-run."""
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    _bump(repo)
    C.assemble(repo, version="1.2.0", date="2026-10-29", name="Renamed")
    text = (repo / "CHANGELOG.md").read_text()
    assert '## [1.2.0] - 2026-10-29 — "Renamed"' in text and text.count("## [1.2.0]") == 1


def test_assemble_never_touches_released_history(tmp_path: Path) -> None:
    """Code round SF2: the blank-line collapse once rewrote 1.0.x-era sections on the real CHANGELOG."""
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    text = (repo / "CHANGELOG.md").read_text().replace("- shipped\n", "- shipped\n\n\n\n- odd old spacing\n")
    (repo / "CHANGELOG.md").write_text(text)
    old_tail = text.split("## [1.1.1]", 1)[1]
    _bump(repo)
    assert (repo / "CHANGELOG.md").read_text().split("## [1.1.1]", 1)[1] == old_tail


def test_assemble_refuses_a_version_that_is_not_pyprojects(tmp_path: Path) -> None:
    """Code round N5: ``--version v1.2.0`` (or a version not yet bumped) would write a heading no gate reads."""
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    with pytest.raises(ValueError, match="pyproject"):
        C.assemble(repo, version="1.2.0", date="2026-10-28", name="Test")
    (repo / "pyproject.toml").write_text(PYPROJECT.format(v="1.2.0"))
    with pytest.raises(ValueError, match="X.Y.Z"):
        C.assemble(repo, version="v1.2.0", date="2026-10-28", name="Test")


def test_a_src_path_git_would_quote_still_needs_a_fragment(tmp_path: Path) -> None:
    """Code round N2: without -z, git quotes a path holding a double quote, and the quoted form did not start with
    src/, so the change passed with no fragment."""
    repo = _repo(tmp_path)
    base = _head(repo)
    (repo / "src" / "maxim" / 'we"ird.yaml').write_text("a: 1\n")
    _commit(repo)
    assert any("new fragment" in f for f in _violations(repo, base))


# ── a push range is judged one first-parent unit at a time ───────────────────


def _api(prs: dict[str, int] | None = None):
    """A stand-in for ``gh api``: ``prs`` maps a first-parent sha to the merged PR it belongs to."""
    prs = prs or {}

    def api(path: str):
        if path.endswith("/pulls"):
            sha = path.split("/commits/")[1].split("/")[0]
            if sha in prs:
                return [{"number": prs[sha], "merged_at": "2026-10-28T00:00:00Z", "base": {"ref": "main"}}]
            return []
        return [{"commit": {"committer": {"date": "2026-10-28T00:00:00Z"}}}]

    return api


@pytest.fixture
def _repo_env(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")


def _merge_branch(repo: Path, branch: str, work) -> None:
    _run(repo, "checkout", "-q", "-b", branch)
    work()
    _commit(repo, branch)
    _run(repo, "checkout", "-q", "main")
    _run(repo, "merge", "-q", "--no-ff", "-m", f"Merge {branch}", branch)


def test_a_src_pr_merging_after_the_bump_in_one_push_range_passes(tmp_path: Path, _repo_env) -> None:
    """F1/N6: a cancelled push run puts the bump and a later PR in one range. Each is judged as its own unit, so
    the later PR's fragment is not "left over" by the bump."""
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    base = _head(repo)
    _merge_branch(repo, "bump", lambda: _bump(repo))

    def later() -> None:
        _src_change(repo, "3")
        _frag(repo, "later.fixed.md", "- Later.\n")

    _merge_branch(repo, "later", later)
    assert C.range_violations(repo, base, api=_api()) == []


def test_a_bad_unit_in_a_push_range_is_still_caught(tmp_path: Path, _repo_env) -> None:
    repo = _repo(tmp_path)
    base = _head(repo)
    _merge_branch(repo, "ok", lambda: (_src_change(repo, "2"), _frag(repo, "ok.fixed.md")))
    _merge_branch(repo, "bad", lambda: _src_change(repo, "3"))
    assert C.range_violations(repo, base, api=_api())


def test_a_rebase_merged_pr_is_judged_as_one_unit(tmp_path: Path, _repo_env) -> None:
    """Code round SF1: "change src/" then "add the fragment" as two first-parent commits of ONE merged PR passed on
    the PR; judged commit by commit, its push would turn main red."""
    repo = _repo(tmp_path)
    base = _head(repo)
    _src_change(repo)
    first = _commit(repo, "change src")
    _frag(repo, "rebased.fixed.md")
    second = _commit(repo, "add the fragment")
    assert C.range_violations(repo, base, api=_api({first: 7, second: 7})) == []
    assert C.range_violations(repo, base, api=_api())  # as two direct pushes, the first really is bad


# ── the tag-time check (check_release_tag.py notes) ──────────────────────────


def test_tag_time_check_only_bites_when_untagged_or_on_a_tag_push(tmp_path: Path, monkeypatch) -> None:
    """N4/R5: a pre-bump rehearsal (pyproject names a tagged release) must not fail on the cycle's fragments."""
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    monkeypatch.delenv("GITHUB_REF", raising=False)
    assert C.leftover_fragment_problems(repo, "1.1.1") == []  # tagged, a rehearsal
    assert C.leftover_fragment_problems(repo, "1.2.0")  # untagged: the post-bump state
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v1.1.1")
    assert C.leftover_fragment_problems(repo, "1.1.1")  # a tag push always checks
    _run(repo, "rm", "-q", "changelog.d/a.fixed.md")
    (repo / "changelog.d" / "stray.fixed.txt").write_text("- x\n")
    assert C.leftover_fragment_problems(repo, "1.1.1")  # any file left counts, not only *.md (code round N8)


def test_the_release_job_fetches_tags_and_runs_the_belt() -> None:
    """R5's "tagged" is read with ``git tag -l``, which needs the tags fetched: the build job's checkout pins
    ``fetch-depth: 0``, and its notes step is where the belt runs."""
    root = Path(__file__).resolve().parents[2]
    job = (
        (root / ".github" / "workflows" / "release.yml").read_text().split("\n  build:", 1)[1].split("\n  smoke:", 1)[0]
    )
    assert "fetch-depth: 0" in job
    assert "check_release_tag.py notes" in job
    assert "leftover_fragment_problems(root, version)" in (root / "scripts" / "check_release_tag.py").read_text()


# ── the D37 rule: on a pull request a missing base is a hard error ───────────


def test_no_base_ref_on_a_pull_request_is_a_hard_error(monkeypatch) -> None:
    def _no_base(_root):
        raise C.GitUnavailable("no merge-base")

    monkeypatch.setattr(C, "base_ref", _no_base)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    assert C.main(["check"]) == 2


# ── cases carried over from the two lints this replaces ──────────────────────


def test_a_pyproject_edit_without_a_version_change_is_not_a_bump(tmp_path: Path) -> None:
    """From lint_unreleased_declared: editing pyproject (a dependency, say) must not excuse a src change."""
    repo = _repo(tmp_path)
    base = _head(repo)
    _src_change(repo)
    (repo / "pyproject.toml").write_text(PYPROJECT.format(v="1.1.1") + 'dependencies = ["x"]\n')
    _commit(repo)
    assert any("new fragment" in f for f in _violations(repo, base))


def test_a_blank_line_added_to_unreleased_is_not_growth(tmp_path: Path) -> None:
    repo = _repo(tmp_path, unreleased="- Old entry.\n\n")
    base = _head(repo)
    (repo / "CHANGELOG.md").write_text(CHANGELOG.format(unreleased="- Old entry.\n\n\n\n"))
    _commit(repo)
    assert _violations(repo, base) == []


def test_unreleased_lines_stop_at_the_next_section() -> None:
    text = "# C\n\n## [Unreleased]\n\n### Fixed\n\n- a\n<!-- c -->\n\n## [1.0.0] - x\n\n- shipped\n"
    assert C.unreleased_lines(text) == ["### Fixed", "- a", "<!-- c -->"]


def test_an_unreadable_base_changelog_is_not_an_empty_one(tmp_path: Path) -> None:
    """#1098: a failed read of the base must raise (and reach must_not_skip), never read as "" and pass."""
    repo = _repo(tmp_path)
    _src_change(repo)
    _frag(repo, "x.fixed.md")
    _commit(repo)
    with pytest.raises(C.GitUnavailable):
        C.violations(repo, "0" * 40, "HEAD")


def test_an_entry_under_the_in_flight_version_no_longer_passes(tmp_path: Path) -> None:
    """Q2: the in-flight path is dropped. The tag sits on the bump merge commit, so a src change merged during the
    freeze ships in the NEXT release, and its record is a fragment, not a line under the frozen version."""
    repo = _repo(tmp_path)
    (repo / "pyproject.toml").write_text(PYPROJECT.format(v="1.2.0"))
    text = (repo / "CHANGELOG.md").read_text().replace("## [1.1.1]", "## [1.2.0] - 2026-10-28\n\n- cut\n\n## [1.1.1]")
    (repo / "CHANGELOG.md").write_text(text)
    base = _commit(repo, "the bump, merged, not yet tagged")
    _src_change(repo)
    (repo / "CHANGELOG.md").write_text(text.replace("- cut\n", "- cut\n- a freeze-time fix\n"))
    _commit(repo)
    assert any("new fragment" in f for f in _violations(repo, base))


def test_assemble_refuses_uncommitted_fragment_edits_before_writing(tmp_path: Path) -> None:
    """Delta round: git rm refused an edited fragment only AFTER CHANGELOG.md was written, leaving a half-made cut."""
    repo = _repo(tmp_path, fragments={"a.fixed.md": "- A.\n"})
    (repo / "pyproject.toml").write_text(PYPROJECT.format(v="1.2.0"))
    _frag(repo, "a.fixed.md", "- A, edited but not committed.\n")
    before = (repo / "CHANGELOG.md").read_text()
    with pytest.raises(ValueError, match="uncommitted"):
        C.assemble(repo, version="1.2.0", date="2026-10-28", name="Test")
    assert (repo / "CHANGELOG.md").read_text() == before


@pytest.mark.parametrize("exc", [C.GitUnavailable("git died"), UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")])
def test_a_mid_run_failure_on_a_push_is_exit_2(monkeypatch, exc) -> None:
    """Fold 7: on a push, git dying mid-run or an undecodable fragment is "could not check", never a pass."""
    monkeypatch.setattr(C, "base_ref", lambda _root: "base")

    def _boom(*_a, **_k):
        raise exc

    monkeypatch.setattr(C, "range_violations", _boom)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    assert C.main(["check"]) == 2
