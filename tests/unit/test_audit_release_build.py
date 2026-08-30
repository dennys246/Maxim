"""scripts/audit_release_build.py — the two release-build near-misses of 2026-08-30.

Bugs ledger D47 (wheel with no Console UI) and D48 (wheel at the wrong version).
`twine check` PASSED both real artifacts, which is why this exists.

These tests build synthetic wheels carrying each defect and assert the auditor
catches it. That matters more than usual here: the CI build job runs with
`--allow-missing-ui-dist` (a CI checkout has no vendored bundle), so the D47
assertion is exercised ONLY by these negative controls. Without them the CI job
would be a guard that has never been shown to fail — the vacuous-guard shape the
model-cache lane's `executed > 0` assert exists to prevent.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts import audit_release_build as A

VERSION = "1.1.2"
DIST_INFO = f"pymaxim-{VERSION}.dist-info"


def _wheel(
    tmp_path: Path,
    *,
    filename_version: str = VERSION,
    metadata_version: str = VERSION,
    ui_dist: bool = True,
    py_typed: bool = True,
    main_module: bool = True,
    data_files: int = 30,
) -> Path:
    path = tmp_path / f"pymaxim-{filename_version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            f"{DIST_INFO}/METADATA",
            f"Metadata-Version: 2.1\nName: pymaxim\nVersion: {metadata_version}\n",
        )
        zf.writestr("maxim/__init__.py", "")
        if py_typed:
            zf.writestr("maxim/py.typed", "")
        if main_module:
            zf.writestr("maxim/__main__.py", "")
        if ui_dist:
            zf.writestr("maxim/console/ui_dist/index.html", "<html></html>")
            zf.writestr("maxim/console/ui_dist/assets/index-abc.js", "//")
            zf.writestr("maxim/console/ui_dist/maxim-ui.json", '{"target":"console"}')
        for i in range(data_files):
            zf.writestr(f"maxim/_data/thing_{i}.yaml", "a: 1\n")
    return path


def test_a_good_wheel_passes(tmp_path: Path) -> None:
    assert A.audit_wheel(_wheel(tmp_path), VERSION) == []


# ── D47 — the Console UI bundle ──────────────────────────────────────────────


def test_missing_ui_dist_fails(tmp_path: Path) -> None:
    """The exact worktree-build defect: 0 files under console/ui_dist/."""
    problems = A.audit_wheel(_wheel(tmp_path, ui_dist=False), VERSION)
    assert len(problems) == 1
    assert "D47" in problems[0]
    assert "ui_dist/index.html is MISSING" in problems[0]


def test_missing_ui_dist_can_be_waived_explicitly(tmp_path: Path, capsys) -> None:
    """The CI build path. The waiver must be LOUD, not silent."""
    problems = A.audit_wheel(_wheel(tmp_path, ui_dist=False), VERSION, require_ui_dist=False)
    assert problems == []
    assert "WAIVED" in capsys.readouterr().out


# ── D48 — the version ────────────────────────────────────────────────────────


def test_wrong_filename_version_fails(tmp_path: Path) -> None:
    """The stale-branch build: a well-formed wheel of the wrong code."""
    problems = A.audit_wheel(_wheel(tmp_path, filename_version="1.1.0", metadata_version="1.1.0"), VERSION)
    assert len(problems) == 2, problems
    assert all("D48" in p for p in problems)


def test_metadata_version_drift_fails(tmp_path: Path) -> None:
    """Filename right, METADATA wrong — filename alone is not enough."""
    problems = A.audit_wheel(_wheel(tmp_path, metadata_version="9.9.9"), VERSION)
    assert len(problems) == 1
    assert "D48" in problems[0] and "METADATA" in problems[0]


def test_version_check_is_not_waivable_by_the_ui_flag(tmp_path: Path) -> None:
    """--allow-missing-ui-dist waives D47 ONLY. A wrong version still fails."""
    problems = A.audit_wheel(
        _wheel(tmp_path, filename_version="1.1.0", metadata_version="1.1.0"), VERSION, require_ui_dist=False
    )
    assert problems and all("D48" in p for p in problems)


# ── the package data the guide's manual checks covered ───────────────────────


@pytest.mark.parametrize("missing", ["py_typed", "main_module"])
def test_missing_required_package_file_fails(tmp_path: Path, missing: str) -> None:
    problems = A.audit_wheel(_wheel(tmp_path, **{missing: False}), VERSION)
    assert len(problems) == 1
    assert "required package file missing" in problems[0]


def test_stripped_data_tree_fails(tmp_path: Path) -> None:
    problems = A.audit_wheel(_wheel(tmp_path, data_files=3), VERSION)
    assert len(problems) == 1
    assert "_data/" in problems[0]


def test_several_defects_are_all_reported(tmp_path: Path) -> None:
    """The report must not stop at the first problem."""
    problems = A.audit_wheel(_wheel(tmp_path, ui_dist=False, py_typed=False, data_files=0), VERSION)
    assert len(problems) == 3


# ── the repo's own version is readable (the auditor's default) ───────────────


def test_pyproject_version_is_readable() -> None:
    version = A.pyproject_version()
    assert version and version[0].isdigit()
