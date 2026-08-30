#!/usr/bin/env python
"""Audit a BUILT wheel before it is published. Bugs ledger D47 + D48.

WHY THIS EXISTS — two near-misses on 2026-08-30, both of which `twine check` PASSED
-----------------------------------------------------------------------------------
`twine check` validates that a distribution's *metadata* renders. It says nothing
about whether the artifact contains what it is supposed to contain, or whether it
is the version you meant to build. Both of those went wrong on one afternoon:

**D47 — the wheel with no Console UI.** `src/maxim/console/ui_dist/` is
`.gitignore`d with ZERO tracked files: it is copied in at release time by
`scripts/vendor_console_ui.py` and exists only in whatever checkout the operator
vendored into. A wheel built from a *worktree* therefore silently ships no Console
UI at all — `maxim serve` degrades to the "no UI installed" page for every user.
1.1.0 shipped 6 files under that path; the worktree build shipped 0, and nothing
complained. The failure is invisible locally too, because a dev box sets
`config.json::console.ui_dist` to an external path and serves a working Console
regardless of what is in the wheel.

**D48 — the wheel with the wrong version.** A build run from a stale branch
produced a `1.1.0`-versioned wheel while `pyproject.toml` on the intended commit
said something else. `twine check` passed that too — it is a perfectly well-formed
distribution of the wrong code.

Neither is caught by any existing check: `lint_version_sync.py` compares the repo's
version strings *to each other* (a stale branch is internally consistent, just
wrong), and `audit_release_tags.py` runs POST-publication against PyPI.

WHAT IT ASSERTS
---------------
Given a built wheel (default: the newest under `dist/`):

1. the version in the wheel filename == the version in its `.dist-info` METADATA
   == `pyproject.toml`'s version   (D48)
2. `maxim/console/ui_dist/index.html` is present  (D47)
3. the package data the guide's manual checks cover is present: `maxim/py.typed`,
   `maxim/__main__.py`, and a non-trivial `maxim/_data/` tree

`--allow-missing-ui-dist` exists for exactly one caller — a CI build on a checkout
that legitimately has no vendored bundle — and it is NOT the release path. When it
is passed the report says so out loud, because a check that can be silently waived
is not a check.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

UI_DIST_INDEX = "maxim/console/ui_dist/index.html"
REQUIRED_FILES = ("maxim/py.typed", "maxim/__main__.py")
MIN_DATA_FILES = 25  # the guide's manual check; a stripped _data/ is a broken wheel

_PYPROJECT_VERSION = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
_WHEEL_NAME = re.compile(r"^(?P<dist>[A-Za-z0-9_.]+)-(?P<version>[^-]+)-")
_METADATA_VERSION = re.compile(r"^Version:\s*(.+)$", re.MULTILINE)


def pyproject_version(repo_root: Path = REPO_ROOT) -> str:
    m = _PYPROJECT_VERSION.search((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    if not m:
        raise SystemExit("ERROR: could not read version from pyproject.toml")
    return m.group(1)


def newest_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("pymaxim-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise SystemExit(f"ERROR: no pymaxim-*.whl under {dist_dir} — build first (`python -m build`)")
    return wheels[-1]


def newest_sdist(dist_dir: Path) -> Path | None:
    sdists = sorted(dist_dir.glob("pymaxim-*.tar.gz"), key=lambda p: p.stat().st_mtime)
    return sdists[-1] if sdists else None


def audit_sdist(sdist: Path, expect_version: str, *, require_ui_dist: bool = True) -> list[str]:
    """The sdist ships too, and D47/D48 apply to it verbatim.

    `twine upload dist/pymaxim-*` uploads BOTH artifacts, and
    `pip install --no-binary :all: pymaxim` consumes the sdist — so auditing
    only the wheel leaves half the release unchecked. v1.1.0's sdist carried
    the same 6 `console/ui_dist/` files as its wheel; a worktree build's would
    carry none.
    """
    problems: list[str] = []

    name_match = re.match(r"^(?P<dist>[A-Za-z0-9_.]+)-(?P<version>.+)\.tar\.gz$", sdist.name)
    filename_version = name_match.group("version") if name_match else None
    if filename_version is None:
        problems.append(f"cannot parse a version out of the sdist filename {sdist.name!r}")
    elif filename_version != expect_version:
        problems.append(
            f"D48: sdist filename says version {filename_version!r} but pyproject.toml says {expect_version!r}"
        )

    with tarfile.open(sdist, "r:gz") as tf:
        names = tf.getnames()
    # sdist members are prefixed with `pymaxim-<version>/`.
    suffixes = {n.split("/", 1)[1] for n in names if "/" in n}

    if "src/maxim/console/ui_dist/index.html" not in suffixes:
        ui_files = [s for s in suffixes if s.startswith("src/maxim/console/ui_dist/")]
        message = (
            f"D47: src/maxim/console/ui_dist/index.html is MISSING from the sdist "
            f"({len(ui_files)} file(s) under it). The sdist ships and installs too."
        )
        if require_ui_dist:
            problems.append(message)
        else:
            print(f"WAIVED (--allow-missing-ui-dist): {message}")

    return problems


def audit_wheel(wheel: Path, expect_version: str, *, require_ui_dist: bool = True) -> list[str]:
    """Return a list of problems; empty means the wheel is publishable."""
    problems: list[str] = []

    name_match = _WHEEL_NAME.match(wheel.name)
    filename_version = name_match.group("version") if name_match else None
    if filename_version is None:
        problems.append(f"cannot parse a version out of the wheel filename {wheel.name!r}")
    elif filename_version != expect_version:
        problems.append(
            f"D48: wheel filename says version {filename_version!r} but pyproject.toml says "
            f"{expect_version!r} — this wheel was built from a different tree than you think"
        )

    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()

        metadata_names = [n for n in names if n.endswith(".dist-info/METADATA")]
        if not metadata_names:
            problems.append("no .dist-info/METADATA in the wheel")
        else:
            meta = zf.read(metadata_names[0]).decode("utf-8", "replace")
            mm = _METADATA_VERSION.search(meta)
            metadata_version = mm.group(1).strip() if mm else None
            if metadata_version is None:
                problems.append("no Version: field in the wheel METADATA")
            elif metadata_version != expect_version:
                problems.append(
                    f"D48: wheel METADATA says version {metadata_version!r} but pyproject.toml says {expect_version!r}"
                )

        if UI_DIST_INDEX not in names:
            ui_files = [n for n in names if n.startswith("maxim/console/ui_dist/")]
            message = (
                f"D47: {UI_DIST_INDEX} is MISSING from the wheel "
                f"({len(ui_files)} file(s) under console/ui_dist/). The Console UI bundle is "
                f"gitignored and vendored at release time — a worktree build has none. "
                f"Run `python scripts/vendor_console_ui.py <maxim-pulse>/apps/console/dist` "
                f"and rebuild."
            )
            if require_ui_dist:
                problems.append(message)
            else:
                print(f"WAIVED (--allow-missing-ui-dist): {message}")

        for required in REQUIRED_FILES:
            if required not in names:
                problems.append(f"required package file missing from the wheel: {required}")

        data_files = [n for n in names if n.startswith("maxim/_data/") and not n.endswith("/")]
        if len(data_files) < MIN_DATA_FILES:
            problems.append(
                f"only {len(data_files)} file(s) under maxim/_data/ (expected >= {MIN_DATA_FILES}) — "
                f"package data looks stripped"
            )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", type=Path, default=None, help="wheel to audit (default: newest in --dist-dir)")
    parser.add_argument("--dist-dir", type=Path, default=REPO_ROOT / "dist")
    parser.add_argument("--expect-version", default=None, help="default: pyproject.toml's version")
    parser.add_argument(
        "--allow-missing-ui-dist",
        action="store_true",
        help="CI-only: a checkout with no vendored Console bundle. NOT the release path.",
    )
    args = parser.parse_args(argv)

    wheel = args.wheel or newest_wheel(args.dist_dir)
    expect = args.expect_version or pyproject_version()

    print(f"auditing {wheel.name} against pyproject version {expect}")
    problems = audit_wheel(wheel, expect, require_ui_dist=not args.allow_missing_ui_dist)

    # The sdist is uploaded alongside the wheel and installs via
    # `pip install --no-binary :all:`, so leaving it unaudited leaves half the
    # release unchecked. Absent only when --wheel names a file directly.
    sdist = None if args.wheel else newest_sdist(args.dist_dir)
    if sdist is not None:
        print(f"auditing {sdist.name}")
        problems += audit_sdist(sdist, expect, require_ui_dist=not args.allow_missing_ui_dist)
    elif not args.wheel:
        problems.append(f"no pymaxim-*.tar.gz under {args.dist_dir} — `python -m build` produces both")

    if problems:
        print("release-build audit FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("\nDo NOT upload this artifact.", file=sys.stderr)
        return 1

    suffix = " (ui_dist check waived)" if args.allow_missing_ui_dist else ""
    print(f"release-build audit: clean{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
