"""scripts/audit_release_build.py — the two release-build near-misses of 2026-08-30.

Bugs ledger D47 (wheel with no Console UI) and D48 (wheel at the wrong version).
`twine check` PASSED both real artifacts, which is why this exists.

These tests build synthetic wheels carrying each defect and assert the auditor
catches it, so every guard here has been shown to FAIL, not merely to pass. Since
R4 (#1333) CI vendors the real pinned pulse bundle and the audit checks it against
`scripts/console_ui_pin.json`; `--allow-missing-ui-dist` lost its only caller and
was removed, so the D47 and R4 checks have no waiver left to test.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts import audit_release_build as A
from tests.unit import _console_bundles as cb

cp = A.cp

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
    constitution: bool = True,
    data_files: int = 30,
    console: dict[str, bytes] | None = None,
    contract: str | None = None,
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
        if constitution:  # D32: gates the agent's foundational preamble
            zf.writestr("maxim/_data/CONSTITUTION.md", "# Constitution\n")
        zf.writestr("maxim/console/ui_bundle.py", f'CONSOLE_CONTRACT_VERSION = "{contract or cp.backend_contract()}"\n')
        if ui_dist:
            for rel, data in (console or cb.bundle(cp)).items():
                zf.writestr(f"maxim/console/ui_dist/{rel}", data)
        for i in range(data_files):
            zf.writestr(f"maxim/_data/thing_{i}.yaml", "a: 1\n")
    return path


PIN = cb.pin_for(cp, cb.bundle(cp))


def _audit(wheel: Path, **kw) -> list[str]:
    return A.audit_wheel(wheel, VERSION, pin=kw.pop("pin", PIN), **kw)[0]


def test_a_good_wheel_passes(tmp_path: Path) -> None:
    problems, line = A.audit_wheel(_wheel(tmp_path), VERSION, pin=PIN)
    assert problems == []
    assert line == f"vendored console: pulse v0.4.1 (da20a72, contract {cp.backend_contract()})"


# ── D47 — the Console UI bundle ──────────────────────────────────────────────


def test_missing_ui_dist_fails(tmp_path: Path) -> None:
    """The exact worktree-build defect: 0 files under console/ui_dist/."""
    problems = _audit(_wheel(tmp_path, ui_dist=False))
    assert len(problems) == 1
    assert "D47" in problems[0]
    assert "ui_dist/index.html is MISSING" in problems[0]


def test_the_waiver_is_gone() -> None:
    """R4 fold 8: CI vendors the real bundle, so nothing may skip the console checks."""
    with pytest.raises(SystemExit):
        A.main(["--allow-missing-ui-dist"])


# ── R4 — the vendored Console is the pinned pulse bundle (#1333) ────────────


def test_the_1_3_2_wheel_fails(tmp_path: Path) -> None:
    """pymaxim 1.3.2 shipped v0.4.0's build with v0.4.1 tagged (the real artifact fails the same way)."""
    stale = cb.bundle(cp, commit="b39ca63", describe="v0.4.0", app_version="0.4.0")
    problems = _audit(_wheel(tmp_path, console=stale))
    assert any("differ from the pinned v0.4.1" in p for p in problems), problems
    assert any("not the pinned v0.4.1" in p for p in problems), problems


def test_an_edited_stamp_over_other_files_fails(tmp_path: Path) -> None:
    forged = {**cb.bundle(cp), "assets/index-abc.js": b"// other code"}
    assert any("differ from the pinned" in p for p in _audit(_wheel(tmp_path, console=forged)))


def test_the_contract_is_read_from_the_artifacts_own_ui_bundle(tmp_path: Path) -> None:
    """A wheel built from another tree: its ui_bundle.py, not the repo's, must match the stamp."""
    problems = _audit(_wheel(tmp_path, contract="9.9.9"), releasing=True)
    assert any("backend is '9.9.9'" in p for p in problems), problems


def test_a_contract_drift_is_reported_not_failed_on_a_tree_that_is_not_releasing(tmp_path: Path, capsys) -> None:
    assert _audit(_wheel(tmp_path, contract="9.9.9"), releasing=False) == []
    assert "CONTRACT DRIFT" in capsys.readouterr().out


def test_a_dotfile_in_the_shipped_bundle_fails(tmp_path: Path) -> None:
    files = {**cb.bundle(cp), ".vite/manifest.json": b"{}"}
    assert any("dotfile" in p for p in _audit(_wheel(tmp_path, console=files)))


def test_main_refuses_an_unreadable_pin(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path)
    (tmp_path / "pin.json").write_text("{}")
    assert A.main(["--wheel", str(wheel), "--pin", str(tmp_path / "pin.json")]) == 2


# ── D48 — the version ────────────────────────────────────────────────────────


def test_wrong_filename_version_fails(tmp_path: Path) -> None:
    """The stale-branch build: a well-formed wheel of the wrong code."""
    problems = _audit(_wheel(tmp_path, filename_version="1.1.0", metadata_version="1.1.0"))
    assert len(problems) == 2, problems
    assert all("D48" in p for p in problems)


def test_metadata_version_drift_fails(tmp_path: Path) -> None:
    """Filename right, METADATA wrong — filename alone is not enough."""
    problems = _audit(_wheel(tmp_path, metadata_version="9.9.9"))
    assert len(problems) == 1
    assert "D48" in problems[0] and "METADATA" in problems[0]


# ── the package data the guide's manual checks covered ───────────────────────


@pytest.mark.parametrize("missing", ["py_typed", "main_module", "constitution"])
def test_missing_required_package_file_fails(tmp_path: Path, missing: str) -> None:
    problems = _audit(_wheel(tmp_path, **{missing: False}))
    assert len(problems) == 1
    assert "required package file missing" in problems[0]


def test_stripped_data_tree_fails(tmp_path: Path) -> None:
    problems = _audit(_wheel(tmp_path, data_files=3))
    assert len(problems) == 1
    assert "_data/" in problems[0]


def test_several_defects_are_all_reported(tmp_path: Path) -> None:
    """The report must not stop at the first problem."""
    problems = _audit(_wheel(tmp_path, ui_dist=False, py_typed=False, data_files=0))
    assert len(problems) == 3


# ── the repo's own version is readable (the auditor's default) ───────────────


def test_pyproject_version_is_readable() -> None:
    version = A.pyproject_version()
    assert version and version[0].isdigit()


# ── the sdist (review fold, 2026-08-30) ──────────────────────────────────────
#
# `twine upload dist/pymaxim-*` uploads BOTH artifacts and
# `pip install --no-binary :all:` consumes the sdist, so a wheel-only audit
# leaves half the release unchecked. D47 and D48 apply to it verbatim.


def _sdist(
    tmp_path: Path, *, version: str = VERSION, ui_dist: bool = True, console: dict[str, bytes] | None = None
) -> Path:
    import io
    import tarfile

    path = tmp_path / f"pymaxim-{version}.tar.gz"
    root = f"pymaxim-{version}"

    def add(tf, name: str, body: bytes = b"x") -> None:
        info = tarfile.TarInfo(f"{root}/{name}")
        info.size = len(body)
        tf.addfile(info, io.BytesIO(body))

    with tarfile.open(path, "w:gz") as tf:
        add(tf, "PKG-INFO", f"Name: pymaxim\nVersion: {version}\n".encode())
        add(tf, "src/maxim/__init__.py")
        add(tf, "src/maxim/console/ui_bundle.py", f'CONSOLE_CONTRACT_VERSION = "{cp.backend_contract()}"\n'.encode())
        if ui_dist:
            for rel, data in (console or cb.bundle(cp)).items():
                add(tf, f"src/maxim/console/ui_dist/{rel}", data)
    return path


def test_good_sdist_passes(tmp_path: Path) -> None:
    assert A.audit_sdist(_sdist(tmp_path), VERSION, pin=PIN) == []


def test_sdist_missing_ui_dist_fails(tmp_path: Path) -> None:
    problems = A.audit_sdist(_sdist(tmp_path, ui_dist=False), VERSION, pin=PIN)
    assert len(problems) == 1
    assert "D47" in problems[0] and "sdist" in problems[0]


def test_sdist_wrong_version_fails(tmp_path: Path) -> None:
    problems = A.audit_sdist(_sdist(tmp_path, version="1.1.0"), VERSION, pin=PIN)
    assert len(problems) == 1
    assert "D48" in problems[0]


def test_sdist_with_a_stale_console_fails(tmp_path: Path) -> None:
    """R4 applies to the sdist verbatim: `pip install --no-binary :all:` installs its bundle."""
    stale = cb.bundle(cp, commit="b39ca63", describe="v0.4.0")
    problems = A.audit_sdist(_sdist(tmp_path, console=stale), VERSION, pin=PIN)
    assert any("R4: sdist" in p for p in problems), problems


def test_newest_sdist_returns_none_when_absent(tmp_path: Path) -> None:
    assert A.newest_sdist(tmp_path) is None
