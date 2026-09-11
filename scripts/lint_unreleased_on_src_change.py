#!/usr/bin/env python3
"""Diff-scoped lint: a src/ change must declare a CHANGELOG ``[Unreleased]`` entry.

The owed half of the versioning policy (CLAUDE.md §Versioning / roadmap item 16.10):
``lint_version_sync.py`` enforces "main is ahead of PyPI", but the paired rule — "a
post-tag ``src/`` commit adds an ``[Unreleased]`` line" — was "convention, not yet
enforced … A policy half-enforced is the divergence 16.1 exists to end." This is the
enforcement.

Rule (diff-scoped against the merge-base with origin/main): if the diff touches
``src/maxim/**/*.py``, the ``## [Unreleased]`` section must have GROWN in the same diff —
unless the diff is a release transaction (it adds a new ``## [X.Y.Z]`` header, which is
exactly when ``[Unreleased]`` legitimately empties). Docs/tests/scripts-only diffs don't
trigger it. On a pull request a skip becomes a hard error (the check IS the gate).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_git import (  # noqa: E402
    GitUnavailable,
    base_ref,
    changed_files,
    must_not_skip,
    show,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
_VERSION_HEADER = re.compile(r"^## \[\d+\.\d+\.\d+\]", re.MULTILINE)


def _unreleased_lines(changelog: str) -> set[str]:
    """The stripped, non-empty content lines under ``## [Unreleased]`` (excl. headings)."""
    lines = changelog.splitlines()
    out: set[str] = set()
    inside = False
    for line in lines:
        if line.startswith("## [Unreleased]"):
            inside = True
            continue
        if inside and line.startswith("## ["):
            break
        if inside:
            s = line.strip()
            if s and not s.startswith("###"):
                out.add(s)
    return out


def verdict(src_changed: bool, base_changelog: str, head_changelog: str) -> tuple[bool, str]:
    """Pure decision (git-free, so it is unit-testable): (ok, message).

    Fails only when a src change ships with neither a grown ``[Unreleased]`` nor a
    release transaction.
    """
    if not src_changed:
        return True, "no src/maxim/*.py change in the diff — N/A"
    # Release-transaction exemption: a new dated version header appeared (the
    # [Unreleased] content moved under it, which is when it legitimately empties).
    if len(_VERSION_HEADER.findall(head_changelog)) > len(_VERSION_HEADER.findall(base_changelog)):
        return True, "release transaction (new version header) — exempt"
    added = _unreleased_lines(head_changelog) - _unreleased_lines(base_changelog)
    if added:
        return True, f"[Unreleased] grew by {len(added)} line(s)"
    return False, (
        "a src/maxim/*.py change did not grow CHANGELOG '## [Unreleased]'. A runtime/CLI/"
        "protocol change owes an [Unreleased] entry (CLAUDE.md §Versioning; roadmap 16.10)."
    )


def check() -> int:
    try:
        base = base_ref(REPO_ROOT)
    except GitUnavailable as exc:
        if must_not_skip(str(exc)):
            return 1
        print(f"INFO: no base ref available; skipping [Unreleased]-on-src check ({exc})")
        return 0

    src_changed = changed_files(REPO_ROOT, base, "src/maxim", suffix=".py")
    base_changelog = show(REPO_ROOT, base, "CHANGELOG.md")
    head_changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    ok, msg = verdict(bool(src_changed), base_changelog, head_changelog)
    print(f"[Unreleased]-on-src: {'clean' if ok else 'FAIL'} — {msg}")
    if not ok:
        for _status, rel in src_changed[:10]:
            print(f"    - {rel}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check())
