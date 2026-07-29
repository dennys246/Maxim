#!/usr/bin/env python3
"""Vendor a built Console UI bundle into the package for release.

The Console UI is built in **maxim-pulse**; the wheel ships a copy so that a
plain ``pip install pymaxim[console] && maxim serve`` serves a working Console
with no flag and no config. This script is the copy step — run it BEFORE
``python -m build`` when cutting a release.

Sources (any of):

  * a local pulse checkout::

        python scripts/vendor_console_ui.py ~/Scripts/Maxim-pulse/apps/console/dist

  * a CI artifact downloaded from a maxim-pulse run (every push to main
    uploads ``ui-dist``), unzipped anywhere::

        python scripts/vendor_console_ui.py ~/Downloads/ui-dist

  * ``--clean`` to remove a previously vendored bundle (back to a source-
    checkout state).

The destination (``src/maxim/console/ui_dist/``) is .gitignore'd: vendoring is
a RELEASE step, not a commit. Verify with ``--check`` in CI or by hand.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "src" / "maxim" / "console" / "ui_dist"
MANIFEST_NAME = "maxim-ui.json"


def _contract_version() -> str:
    """The backend's facade contract version (imported without side effects)."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from maxim.console.ui_bundle import CONSOLE_CONTRACT_VERSION

    return CONSOLE_CONTRACT_VERSION


def validate(source: Path) -> tuple[bool, list[str]]:
    """Is ``source`` a plausible Console bundle? Returns (ok, problems)."""
    problems: list[str] = []
    if not source.is_dir():
        return False, [f"{source} is not a directory"]
    if not (source / "index.html").is_file():
        problems.append("no index.html — is this really a built bundle?")
    if not (source / "assets").is_dir():
        problems.append("no assets/ directory")

    manifest_path = source / MANIFEST_NAME
    if not manifest_path.is_file():
        problems.append(f"no {MANIFEST_NAME} — cannot verify the facade contract (build with a current maxim-pulse)")
        return (not problems), problems

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        problems.append(f"{MANIFEST_NAME} is unreadable: {e}")
        return False, problems

    target = manifest.get("target")
    if target != "console":
        problems.append(f"bundle target is {target!r}, expected 'console' (did you point at the reachy build?)")
    bundle_contract = str(manifest.get("contract_version") or "")
    backend_contract = _contract_version()
    if bundle_contract and bundle_contract != backend_contract:
        problems.append(
            f"facade contract mismatch: bundle {bundle_contract!r} vs backend {backend_contract!r} — "
            f"regenerate the client (pnpm gen:facade) against this pymaxim's openapi.json"
        )
    return (not problems), problems


def vendor(source: Path, *, force: bool = False) -> int:
    ok, problems = validate(source)
    if not ok and not force:
        print(f"REFUSING to vendor {source}:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("\nPass --force to vendor anyway (you probably don't want to).", file=sys.stderr)
        return 1
    for p in problems:  # forced: still say what's wrong
        print(f"WARNING: {p}", file=sys.stderr)

    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(source, DEST)
    files = sum(1 for _ in DEST.rglob("*") if _.is_file())
    size_mb = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file()) / 1e6
    print(f"vendored {files} files ({size_mb:.1f} MB) → {DEST.relative_to(REPO_ROOT)}")
    print("This is a RELEASE step; ui_dist/ is .gitignore'd. Build the wheel now (python -m build).")
    return 0


def check() -> int:
    """Is a usable bundle vendored right now?"""
    if not (DEST / "index.html").is_file():
        print(f"no vendored bundle at {DEST.relative_to(REPO_ROOT)} (source checkout state)")
        return 1
    ok, problems = validate(DEST)
    for p in problems:
        print(f"WARNING: {p}", file=sys.stderr)
    print(f"vendored bundle present at {DEST.relative_to(REPO_ROOT)}{' (with warnings)' if problems else ''}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "source", nargs="?", help="Path to a built Console bundle (apps/console/dist or an unzipped artifact)"
    )
    ap.add_argument("--clean", action="store_true", help="Remove the vendored bundle and exit")
    ap.add_argument("--check", action="store_true", help="Report whether a usable bundle is vendored, and exit")
    ap.add_argument("--force", action="store_true", help="Vendor even if validation fails")
    args = ap.parse_args(argv)

    if args.clean:
        if DEST.exists():
            shutil.rmtree(DEST)
            print(f"removed {DEST.relative_to(REPO_ROOT)}")
        else:
            print("nothing to clean")
        return 0
    if args.check:
        return check()
    if not args.source:
        ap.error("a source path is required (or use --clean / --check)")
    return vendor(Path(args.source).expanduser().resolve(), force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
