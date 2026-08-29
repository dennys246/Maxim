"""scripts/lint_version_sync.py — fixture-tree tests (roadmap 1.1.x item 16.1).

Verified to fail on the pre-policy tree: the three version lines carried "PyPI serves 1.1.0"
prose and docs/index.md had no `**Version:**` line."""

from __future__ import annotations

from pathlib import Path

from scripts import lint_version_sync as V

LINE = "Current version: **{v}** (`pyproject.toml`; PyPI: https://pypi.org/project/pymaxim/)"


def _tree(
    root: Path, *, version="1.2.3", init="1.2.3", changelog_v="1.2.3", claude=None, plans=None, index=None
) -> Path:
    (root / "src/maxim").mkdir(parents=True)
    (root / "docs/plans").mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{version}"\n')
    (root / "src/maxim/__init__.py").write_text(f'__version__ = "{init}"\n')
    (root / "CHANGELOG.md").write_text(f"# Changelog\n\n## [Unreleased]\n\n## [{changelog_v}] - 2026-01-01\n")
    (root / "CLAUDE.md").write_text(claude if claude is not None else LINE.format(v=version) + "\n")
    (root / "docs/plans/README.md").write_text(plans if plans is not None else LINE.format(v=version) + "\n")
    (root / "docs/index.md").write_text(
        index if index is not None else f"**Package version:** {version} (PyPI: https://pypi.org/project/pymaxim/)\n"
    )
    return root


def test_consistent_tree_passes(tmp_path: Path) -> None:
    assert V.violations(_tree(tmp_path)) == []


def test_init_drift_fails(tmp_path: Path) -> None:
    fails = V.violations(_tree(tmp_path, init="1.2.4"))
    assert any("version drift" in f for f in fails)


def test_bump_without_changelog_section_fails(tmp_path: Path) -> None:
    fails = V.violations(_tree(tmp_path, changelog_v="1.2.2"))
    assert any("newest released section is `## [1.2.2]`" in f for f in fails)


def test_a_prewritten_future_section_does_not_satisfy_the_check(tmp_path: Path) -> None:
    """`## [1.3.0]` drafted above the current section must not pass while pyproject says 1.2.3."""
    root = _tree(tmp_path)
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [1.3.0] - 2026-02-01\n\n## [1.2.3] - 2026-01-01\n"
    )
    fails = V.violations(root)
    assert any("newest released section is `## [1.3.0]`" in f for f in fails)


def test_unreleased_section_is_ignored(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n\n- a thing\n\n## [1.2.3] - 2026-01-01\n")
    assert V.violations(root) == []


def test_prose_is_judged_only_in_the_version_claim(tmp_path: Path) -> None:
    """CLAUDE.md's line is a long paragraph; 'published' in a later sentence is innocent."""
    ok = LINE.format(v="1.2.3") + ". The 1.1 line: items 17 and 19 published as planned.\n"
    assert V.violations(_tree(tmp_path / "ok", claude=ok)) == []
    bad = "Current version: **1.2.3** — PyPI serves 1.2.2 (https://pypi.org/project/pymaxim/)\n"
    assert any("PyPI-state prose" in f for f in V.violations(_tree(tmp_path / "bad", claude=bad)))


def test_version_line_naming_another_version_fails(tmp_path: Path) -> None:
    fails = V.violations(_tree(tmp_path, plans=LINE.format(v="1.2.2") + "\n"))
    assert any("docs/plans/README.md" in f and "says '1.2.2'" in f for f in fails)


def test_pypi_state_prose_fails(tmp_path: Path) -> None:
    line = "Current version: **1.2.3** — **PyPI serves 1.2.2** (release pending) https://pypi.org/project/pymaxim/\n"
    fails = V.violations(_tree(tmp_path, claude=line))
    assert any("PyPI-state prose" in f for f in fails)


def test_missing_version_line_fails_not_passes(tmp_path: Path) -> None:
    fails = V.violations(_tree(tmp_path, index="# Docs\n"))
    assert any("docs/index.md" in f and "missing" in f for f in fails)


def test_real_tree_is_consistent() -> None:
    assert V.violations() == []
