#!/usr/bin/env python
"""Audit the BUILT artifacts (wheel + sdist) before they are published. Bugs ledger D47 + D48.

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
Given the newest wheel AND sdist under `dist/`:

1. the version in the wheel filename == the version in its `.dist-info` METADATA
   == `pyproject.toml`'s version, and the same for the sdist filename   (D48)
2. `console/ui_dist/index.html` is present in BOTH artifacts  (D47)
3. the package data the guide's manual checks cover is present in the wheel:
   `maxim/py.typed`, `maxim/__main__.py`, and a non-trivial `maxim/_data/` tree
4. the vendored Console in BOTH artifacts is the pinned pulse bundle (R4, #1333): its
   files digest to `scripts/console_ui_pin.json`'s `tree_sha256`, its `maxim-ui.json`
   names the pinned commit, is clean, and was built for the contract the ARTIFACT's own
   `console/ui_bundle.py` declares (read as text, not the repo's). pymaxim 1.3.2 shipped
   pulse v0.4.0 two days after v0.4.1 was tagged, and nothing read the stamp.

The sdist is audited because `twine upload dist/pymaxim-*` ships it too and
`pip install --no-binary :all: pymaxim` installs from it — auditing only the
wheel leaves half the release unchecked.

There is no waiver for any of it. `--allow-missing-ui-dist` existed for CI, which
used to stub the bundle; CI now vendors the real pinned asset (`vendor_console_ui.py
--from-pin`), so the flag lost its only caller and was removed (R4 design pass, fold 8).
`--contract-drift-ok-unless-releasing` is not a waiver of the pin: on a tree that is
not cutting a release it reports a contract drift instead of failing, so a PR that bumps
the contract can merge before pulse tags a bundle for it; on a releasing tree it fails.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _console_pin as cp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

UI_DIST_INDEX = "maxim/console/ui_dist/index.html"
WHEEL_UI_PREFIX = "maxim/console/ui_dist/"
WHEEL_UI_BUNDLE = "maxim/console/ui_bundle.py"
REQUIRED_FILES = ("maxim/py.typed", "maxim/__main__.py", "maxim/_data/CONSTITUTION.md")  # D32: gates the agent preamble
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


def audit_console(
    members: Iterable[tuple[str, bytes | None]],
    prefix: str,
    ui_bundle_name: str,
    *,
    artifact: str,
    pin: cp.Pin,
    releasing: bool = True,
) -> tuple[list[str], str | None]:
    """R4: is the vendored Console in this artifact the pinned pulse bundle? Returns (problems, summary line).

    `members` are (name, bytes-or-None-for-a-non-file) for the whole artifact; the console files are those under
    `prefix`, and the backend contract is read from the artifact's own `ui_bundle_name`.
    """
    files = dict(members)
    problems = list(cp.pin_identity_problems(pin))
    try:
        digest = cp.tree_digest(cp.prefixed_entries(files.items(), prefix))
    except cp.DigestError as exc:
        return [*problems, f"R4: {artifact} console bundle: {exc}"], None
    if digest != pin.tree_sha256:
        problems.append(
            f"R4: {artifact} ships a console bundle whose files differ from the pinned {pin.tag} "
            f"(tree digest {digest[:12]} != {pin.tree_sha256[:12]}). Vendor with "
            f"`python scripts/vendor_console_ui.py --from-pin` and rebuild."
        )
    try:
        stamp = json.loads(files.get(prefix + cp.MANIFEST_NAME) or b"")
    except json.JSONDecodeError as exc:
        return [*problems, f"R4: {artifact} console {cp.MANIFEST_NAME} is unreadable: {exc}"], None
    contract = cp.contract_from_source((files.get(ui_bundle_name) or b"").decode("utf-8", "replace"))
    if not isinstance(stamp, dict) or contract is None:
        return [*problems, f"R4: {artifact} has no readable console stamp or {ui_bundle_name} contract"], None
    stamp_problems, notes = cp.stamp_problems(stamp, pin, contract, releasing=releasing)
    problems += [f"R4: {artifact}: {p}" for p in stamp_problems]
    for note in notes:
        print(f"  {artifact} console: {note}")
    summary = f"vendored console: pulse {pin.tag} ({pin.commit[:7]}, contract {stamp.get('contract_version')})"
    return problems, summary


def _sdist_members(tf: tarfile.TarFile) -> list[tuple[str, bytes | None]]:
    out: list[tuple[str, bytes | None]] = []
    for m in tf.getmembers():
        if m.isdir():
            continue
        f = tf.extractfile(m) if m.isfile() else None
        out.append((m.name, f.read() if f is not None else None))
    return out


def audit_sdist(sdist: Path, expect_version: str, *, pin: cp.Pin, releasing: bool = True) -> list[str]:
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

    try:
        with tarfile.open(sdist, "r:gz") as tf:
            members = _sdist_members(tf)
            names = tf.getnames()
    except (tarfile.TarError, OSError) as exc:
        # Keep the file's collect-all-problems-then-report contract: an
        # unreadable artifact is a finding, not a traceback.
        problems.append(f"sdist {sdist.name} could not be read as a gzip tar: {exc}")
        return problems
    # sdist members are prefixed with `pymaxim-<version>/`.
    suffixes = {n.split("/", 1)[1] for n in names if "/" in n}

    if "src/maxim/console/ui_dist/index.html" not in suffixes:
        ui_files = [s for s in suffixes if s.startswith("src/maxim/console/ui_dist/")]
        problems.append(
            f"D47: src/maxim/console/ui_dist/index.html is MISSING from the sdist "
            f"({len(ui_files)} file(s) under it). The sdist ships and installs too."
        )
        return problems

    root = names[0].split("/", 1)[0] + "/" if names else ""
    console_problems, _ = audit_console(
        members,
        f"{root}src/maxim/console/ui_dist/",
        f"{root}src/maxim/console/ui_bundle.py",
        artifact="sdist",
        pin=pin,
        releasing=releasing,
    )
    return problems + console_problems


def audit_wheel(
    wheel: Path, expect_version: str, *, pin: cp.Pin, releasing: bool = True
) -> tuple[list[str], str | None]:
    """Return (problems, the vendored-console summary line); no problems means the wheel is publishable."""
    problems: list[str] = []
    console_line: str | None = None

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
            ui_files = [n for n in names if n.startswith(WHEEL_UI_PREFIX)]
            problems.append(
                f"D47: {UI_DIST_INDEX} is MISSING from the wheel "
                f"({len(ui_files)} file(s) under console/ui_dist/). The Console UI bundle is "
                f"gitignored and vendored at release time — a worktree build has none. "
                f"Run `python scripts/vendor_console_ui.py --from-pin` and rebuild."
            )
        else:
            members = [(n, None if n.endswith("/") else zf.read(n)) for n in names if not n.endswith("/")]
            console_problems, console_line = audit_console(
                members, WHEEL_UI_PREFIX, WHEEL_UI_BUNDLE, artifact="wheel", pin=pin, releasing=releasing
            )
            problems += console_problems

        for required in REQUIRED_FILES:
            if required not in names:
                problems.append(f"required package file missing from the wheel: {required}")

        data_files = [n for n in names if n.startswith("maxim/_data/") and not n.endswith("/")]
        if len(data_files) < MIN_DATA_FILES:
            problems.append(
                f"only {len(data_files)} file(s) under maxim/_data/ (expected >= {MIN_DATA_FILES}) — "
                f"package data looks stripped"
            )

    return problems, console_line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", type=Path, default=None, help="wheel to audit (default: newest in --dist-dir)")
    parser.add_argument("--dist-dir", type=Path, default=REPO_ROOT / "dist")
    parser.add_argument("--sdist", type=Path, default=None, help="sdist to audit (default: newest in --dist-dir)")
    parser.add_argument("--expect-version", default=None, help="default: pyproject.toml's version")
    parser.add_argument("--pin", type=Path, default=cp.PIN_PATH, help="the console pin (default: the committed one)")
    parser.add_argument(
        "--contract-drift-ok-unless-releasing",
        action="store_true",
        help="CI only: report (not fail) a console contract drift on a tree that is not cutting a release",
    )
    args = parser.parse_args(argv)

    wheel = args.wheel or newest_wheel(args.dist_dir)
    expect = args.expect_version or pyproject_version()
    try:
        pin = cp.load_pin(args.pin)
    except cp.PinError as exc:
        print(f"UNVERIFIED: {exc}", file=sys.stderr)
        return 2
    releasing = True
    if args.contract_drift_ok_unless_releasing:
        releasing, why = cp.releasing()
        print(why)

    print(f"auditing {wheel.name} against pyproject version {expect}")
    problems, console_line = audit_wheel(wheel, expect, pin=pin, releasing=releasing)

    # The sdist is uploaded alongside the wheel and installs via
    # `pip install --no-binary :all:`, so leaving it unaudited leaves half the
    # release unchecked. Absent only when --wheel names a file directly.
    sdist = args.sdist or (None if args.wheel else newest_sdist(args.dist_dir))
    if sdist is not None:
        print(f"auditing {sdist.name}")
        problems += audit_sdist(sdist, expect, pin=pin, releasing=releasing)
    elif args.wheel:
        # Say so. "A check that can be silently waived is not a check" applies
        # to this script's own output too: --wheel names one file, so the sdist
        # half of the release is simply not covered by this invocation.
        print("NOTE: sdist NOT audited — --wheel names a single file. Pass --sdist to audit it too.")
    else:
        problems.append(f"no pymaxim-*.tar.gz under {args.dist_dir} — `python -m build` produces both")

    if problems:
        print("release-build audit FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("\nDo NOT upload this artifact.", file=sys.stderr)
        return 1

    if console_line:
        print(console_line)  # the release notes name this tag (backlog M71 until the audit reads the notes)
    print("release-build audit: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
