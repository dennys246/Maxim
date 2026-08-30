#!/usr/bin/env python
"""A post-tag `src/` change must declare itself: bump, or add an `[Unreleased]` line.

Roadmap 1.1.x item 16.10; score card 2026-08-27 Release-governance "Upgrade to
B−" condition (4).

WHY
---
The versioning policy (item 16.1) has two halves. "`main` is ahead of PyPI, the
bump happens in the release transaction" is enforced by
`scripts/lint_version_sync.py`. The other half — "`CHANGELOG.md` accumulates
under `## [Unreleased]`" — was convention only, and convention lost: one day
after the v1.1.0 tag, #561 and #562 changed `runtime/agent_loop.py` and
`motor_backend.py` with no bump and an empty `[Unreleased]`. A release cut then
has to reconstruct from git log what the CHANGELOG was supposed to record.

**A policy half-enforced is the divergence item 16.1 exists to end.** This is
the other half.

THE RULE
--------
If a branch's diff touches `src/`, then at least one of:

  (a) the version in `pyproject.toml` changed  — this is a release transaction,
      and `lint_version_sync.py` owns the rest of it; or
  (b) `CHANGELOG.md`'s `## [Unreleased]` section GAINED at least one content
      line relative to the base.

Docs-only, tests-only and scripts-only branches are untouched by this lint —
the trigger for a release cut is a `src/` change, so that is the trigger here.

WHAT THIS DELIBERATELY DOES NOT DO
-----------------------------------
It does not check that the `[Unreleased]` text is *good*, or that it describes
the change. It checks that the author was made to write something. A lint that
tried to judge the prose would be unfalsifiable; this one is mechanical and its
failure message says exactly what to add.

Skip semantics follow the shared helper: git failing to answer is a SKIP, never
a pass-by-accident — except on a pull request, where `must_not_skip` turns it
into a hard error. That asymmetry is deliberate and is the D37 lesson (every
diff-scoped lint was a silent no-op on PRs for weeks because a shallow clone had
no merge-base).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lint_git import GitUnavailable, base_ref, git, must_not_skip, show  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

CHANGELOG = "CHANGELOG.md"
PYPROJECT = "pyproject.toml"
SRC_PREFIX = "src/"

_VERSION_RE = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
# The heading that opens the accumulator, e.g. "## [Unreleased]".
_UNRELEASED_RE = re.compile(r"^##\s*\[Unreleased\]\s*$", re.IGNORECASE)
_ANY_H2_RE = re.compile(r"^##\s")


def unreleased_lines(changelog_text: str) -> list[str]:
    """Content lines under `## [Unreleased]`, up to the next `##` heading.

    Blank lines and pure-comment lines do not count as content — an empty
    section and a section containing one blank line are the same claim.
    """
    lines = changelog_text.splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if _UNRELEASED_RE.match(line):
            inside = True
            continue
        if inside:
            if _ANY_H2_RE.match(line):
                break
            if line.strip():
                out.append(line.rstrip())
    return out


def _version_of(text: str) -> str | None:
    m = _VERSION_RE.search(text)
    return m.group(1) if m else None


def violations(repo_root: Path, base: str) -> list[str]:
    changed = git(repo_root, "diff", "--name-only", f"{base}...HEAD").split("\n")
    changed = [c.strip() for c in changed if c.strip()]

    src_changed = sorted(c for c in changed if c.startswith(SRC_PREFIX))
    if not src_changed:
        return []

    # (a) a version bump — the release transaction. lint_version_sync.py owns
    #     the rest; this lint steps aside.
    if PYPROJECT in changed:
        before = _version_of(show(repo_root, base, PYPROJECT))
        after = _version_of((repo_root / PYPROJECT).read_text(encoding="utf-8"))
        if before is not None and after is not None and before != after:
            return []

    # (b) the [Unreleased] accumulator gained a content line.
    head_text = (repo_root / CHANGELOG).read_text(encoding="utf-8")
    try:
        base_text = show(repo_root, base, CHANGELOG)
    except GitUnavailable:
        base_text = ""
    before_lines = unreleased_lines(base_text)
    after_lines = unreleased_lines(head_text)
    if len(after_lines) > len(before_lines):
        return []

    shown = ", ".join(src_changed[:5]) + (f" (+{len(src_changed) - 5} more)" if len(src_changed) > 5 else "")
    return [
        f"{len(src_changed)} file(s) under src/ changed ({shown}) but this branch neither "
        f"bumps the version in {PYPROJECT} nor adds a line under '## [Unreleased]' in "
        f"{CHANGELOG}. Per the versioning policy main accumulates under [Unreleased] between "
        f"releases — add the entry, or make this a release transaction. "
        f"([Unreleased] content lines: base {len(before_lines)} -> head {len(after_lines)})"
    ]


def main() -> int:
    try:
        base = base_ref(REPO_ROOT)
    except GitUnavailable as exc:
        if must_not_skip(str(exc)):
            return 2
        print(f"INFO: no base ref (origin/main) available; skipping [Unreleased]-declared lint ({exc})")
        return 0
    try:
        fails = violations(REPO_ROOT, base)
    except GitUnavailable as exc:
        # The mid-run path gets the SAME must_not_skip treatment as base_ref.
        # Unlike the other diff-scoped lints — which still print their
        # non-diff findings when git dies here — this lint IS entirely the
        # diff check, so a silent `return 0` on a pull request would discard
        # the whole gate. That is the D37 shape this file exists to end, and
        # the docstring promises otherwise.
        if must_not_skip(str(exc)):
            return 2
        print(f"INFO: [Unreleased]-declared lint skipped mid-run ({exc})")
        return 0
    except OSError as exc:
        print(f"ERROR: [Unreleased]-declared lint could not run: {exc}", file=sys.stderr)
        return 2

    if fails:
        print("[Unreleased]-declared lint FAILED:", file=sys.stderr)
        for f in fails:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("[Unreleased]-declared lint: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
