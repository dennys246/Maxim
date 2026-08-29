#!/usr/bin/env python3
"""Diff-scoped lint: a `fix` commit that touches src/ must touch tests/ (roadmap 1.1.x item 16.2).

Score card 2026-08-27, Test quantity "Upgrade to A": a CI step failing any commit whose
subject matches ``^fix`` and touches ``src/`` without ``tests/``. The incident is #519 —
a behavioural fix to an abort path with zero test changes; every fix since the cards has
complied, so the lint ratifies practice and makes "90 days clean" measurable instead of
asserted.

Scope: every commit on the current branch relative to ``origin/main`` (merge-base), the
same diff-scoping as lint_multi_agent_marker.py / lint_no_silent_swallows.py. On a push
to main the range is empty and the lint passes — it is a PR gate. The squash-merge
subject is the PR title, which is not visible here; the branch commits are.

Opt-out (catches FORGETTING, not evasion — house convention): a commit whose body
carries a ``No-Tests-Reason: <why>`` trailer is accepted with the reason echoed. A
``fix`` that touches ``src/`` and only docs/scripts is still a fix to src.

Exits: 0 clean; 1 violations (stderr); INFO + 0 when no base ref is available (shallow
clone without origin/main — the graceful-skip pattern shared by the other diff lints).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIX_SUBJECT = re.compile(r"^fix\b", re.IGNORECASE)
OPT_OUT_TRAILER = re.compile(r"^No-Tests-Reason:\s*(\S.*)$", re.IGNORECASE | re.MULTILINE)


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def _base(cwd: Path) -> str | None:
    for ref in ("origin/main", "main"):
        try:
            return _git("merge-base", ref, "HEAD", cwd=cwd).strip()
        except RuntimeError:
            continue
    return None


def violations(cwd: Path = REPO_ROOT, base: str | None = None) -> list[str]:
    """Violation messages for the commits in base..HEAD (base defaults to the merge-base)."""
    base = base or _base(cwd)
    if base is None:
        return []
    out: list[str] = []
    shas = _git("rev-list", "--reverse", f"{base}..HEAD", cwd=cwd).split()
    for sha in shas:
        subject = _git("log", "-1", "--format=%s", sha, cwd=cwd).strip()
        if not FIX_SUBJECT.match(subject):
            continue
        files = _git("diff-tree", "--no-commit-id", "--name-only", "-r", sha, cwd=cwd).split()
        touches_src = any(f.startswith("src/") for f in files)
        touches_tests = any(f.startswith("tests/") for f in files)
        if not touches_src or touches_tests:
            continue
        body = _git("log", "-1", "--format=%b", sha, cwd=cwd)
        m = OPT_OUT_TRAILER.search(body)
        if m:
            print(f"INFO: {sha[:8]} `{subject[:60]}` touches src/ without tests/ — No-Tests-Reason: {m.group(1)}")
            continue
        out.append(
            f"{sha[:8]} `{subject[:72]}` touches src/ ({sum(f.startswith('src/') for f in files)} file(s)) "
            "without touching tests/ — a fix ships with the test that would have caught it (#519 lesson), "
            "or carries a `No-Tests-Reason: <why>` trailer"
        )
    return out


def main() -> int:
    base = _base(REPO_ROOT)
    if base is None:
        print("INFO: no base ref (origin/main) available; skipping fix→tests lint")
        return 0
    fails = violations(REPO_ROOT, base)
    if fails:
        print("fix→tests lint FAILED:", file=sys.stderr)
        for f in fails:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"fix→tests lint: clean ({len(_git('rev-list', f'{base}..HEAD').split())} commit(s) on this branch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
