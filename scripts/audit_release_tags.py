#!/usr/bin/env python3
"""Audit — and optionally reconstruct — the git tags for released versions.

Every version with a ``## [X.Y.Z]`` CHANGELOG section should have a matching
annotated ``vX.Y.Z`` tag on the commit that introduced the version bump. PyPI is
immutable but carries no git history, so an untagged release cannot be traced
back to the code that produced it.

This exists because that failure already happened twice: 1.0.1-1.0.6 were
reconstructed months later during the 1.1 release-truth pass, and the 2026-08-19
pre-publish review found the gap was wider still (pre-1.0 versions plus 1.0.7+).
The publication guide now tags at publish time; this script closes the backlog
and lets anyone re-check in one command.

Usage:
    python scripts/audit_release_tags.py                 # report drift only
    python scripts/audit_release_tags.py --write-tags    # create missing tags LOCALLY

``--write-tags`` never pushes. Review with ``git tag -n99`` and push deliberately
(``git push origin --tags``) — a published tag is effectively permanent.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHANGELOG = REPO / "CHANGELOG.md"
VERSION_FILES = ("pyproject.toml", "src/maxim/__init__.py")

_HEADING = re.compile(r"^## \[(?P<version>\d+\.\d+\.\d+)\](?:\s*-\s*(?P<date>\d{4}-\d{2}-\d{2}))?", re.M)


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def changelog_releases() -> list[tuple[str, str | None, str]]:
    """(version, date, first summary line) for each released section, newest first."""
    text = CHANGELOG.read_text()
    out: list[tuple[str, str | None, str]] = []
    matches = list(_HEADING.finditer(text))
    for i, m in enumerate(matches):
        body = text[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        summary = ""
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("- **"):
                summary = line.lstrip("- ").strip()
                summary = re.sub(r"\*\*(.+?)\*\*", r"\1", summary).split(" — ")[0].split(". ")[0]
                break
        out.append((m.group("version"), m.group("date"), summary))
    return out


def version_bump_commit(version: str) -> str | None:
    """The earliest commit that introduced this version string into a version file."""
    result = _git("log", "--reverse", "--format=%H", "-S", f'"{version}"', "--", *VERSION_FILES)
    return result.splitlines()[0] if result else None


def existing_tags() -> set[str]:
    return {t.lstrip("v") for t in _git("tag", "-l", "v*").splitlines()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-tags", action="store_true", help="create the missing tags locally (never pushes)")
    ap.add_argument(
        "--include-current",
        action="store_true",
        help="also tag the version in pyproject.toml — only correct AT publish time",
    )
    args = ap.parse_args(argv)

    current = ""
    pyproject = (REPO / "pyproject.toml").read_text()
    if m := re.search(r'^version = "(.+?)"', pyproject, re.M):
        current = m.group(1)

    tagged = existing_tags()
    missing: list[tuple[str, str | None, str, str]] = []
    unresolved: list[str] = []
    seen: set[str] = set()

    for version, released, summary in changelog_releases():
        if version in tagged:
            continue
        if version in seen:
            # A version with two CHANGELOG sections is history, not a typo:
            # 0.2.0 was drafted as 1.0.0, relabelled to a research preview, and
            # its PyPI slot was later burned by an upload+delete cycle. Tag the
            # NEWEST section (sections arrive newest-first) and say so.
            print(f"  {version:<8} DUPLICATE CHANGELOG section ignored — tagging the newest one only")
            continue
        seen.add(version)
        if version == current and not args.include_current:
            print(f"  {version:<8} SKIPPED — current in-development version; tag it at publish time")
            continue
        commit = version_bump_commit(version)
        if commit is None:
            unresolved.append(version)
            continue
        missing.append((version, released, summary, commit))

    if not missing and not unresolved:
        print("All released CHANGELOG versions have tags.")
        return 0

    print(f"\n{len(missing)} released version(s) without a tag:\n")
    for version, released, summary, commit in missing:
        subject = _git("log", "-1", "--format=%s", commit)
        print(f"  v{version:<8} {commit[:8]}  {released or '????-??-??'}  {subject[:58]}")

    if unresolved:
        print(f"\n  Could not locate a version-bump commit for: {', '.join(unresolved)}")
        print("  (version string may predate the current version files — tag by hand)")

    if not args.write_tags:
        print("\nRe-run with --write-tags to create these locally, then review with `git tag -n99`.")
        return 1

    stamp = date.today().isoformat()
    for version, released, summary, commit in missing:
        message = f"pymaxim {version}"
        if released:
            message += f" ({released})"
        if summary:
            message += f" — {summary}"
        message += f". Tag reconstructed {stamp} from the version-bump commit."
        proc = subprocess.run(
            ("git", "tag", "-a", f"v{version}", commit, "-m", message),
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"  FAILED v{version}: {proc.stderr.strip()}", file=sys.stderr)
            return 2
        print(f"  created v{version} -> {commit[:8]}")

    print("\nTags created LOCALLY. Review (`git tag -n99 | sort -V`), then push deliberately:")
    print("  git push origin --tags")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
