#!/usr/bin/env python3
"""A `fix` that touches src/ ships with a test (roadmap 1.1.x item 16.2).

Score card 2026-08-27, Test quantity "Upgrade to A": a CI step failing any commit whose
subject matches ``^fix`` and touches ``src/`` without ``tests/``. The incident is #519 —
a behavioural fix to an abort path with zero test changes.

**Two populations, because `main` is squash-merged** (the review-round correction,
2026-08-29): the subject that ends up on ``main`` is the PULL REQUEST TITLE, not any
branch commit's subject. A first draft that read only branch commits gated a population
the score card never counts — a PR titled ``fix(...)`` whose branch commits read ``wip``
would have sailed through, and "90 days clean" would have stayed unmeasurable. So:

1. **PR title vs the aggregate diff** — when ``PR_TITLE`` is set (CI passes
   ``github.event.pull_request.title``), the title is matched against ``^fix`` and the
   whole ``base...HEAD`` diff must touch ``tests/`` if it touches ``src/``. This is the
   commit that will exist on ``main``.
2. **Per-branch-commit** — every commit in ``base..HEAD`` is checked the same way, so
   the rule is visible while the branch is being written, before the squash exists.

Opt-out (catches FORGETTING, not evasion — house convention): a ``No-Tests-Reason:
<why>`` trailer in the commit body, or ``[no-tests: <why>]`` in the PR title/body, is
accepted with the reason echoed to stdout and to ``$GITHUB_STEP_SUMMARY`` when set — an
escape hatch must not be quieter than the rule it exempts. Merge commits on the branch
are skipped with a printed note (their ``diff-tree`` output is empty by default).

Exits: 0 clean; 1 violations (stderr); 2 unexpected error. No base ref (a shallow clone
without origin/main) SKIPS with an INFO — this is a PR gate, not a fail-closed guard.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_git import GitUnavailable, base_ref, git, must_not_skip  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FIX_SUBJECT = re.compile(r"^fix\b", re.IGNORECASE)
OPT_OUT_TRAILER = re.compile(r"^No-Tests-Reason:\s*(\S.*)$", re.IGNORECASE | re.MULTILINE)
OPT_OUT_INLINE = re.compile(r"\[no-tests:\s*([^\]]+)\]", re.IGNORECASE)
ADVICE = (
    "a fix ships with the test that would have caught it (#519 lesson) — add it, squash it into "
    "the fix commit, or declare a `No-Tests-Reason: <why>` trailer (PR title/body: `[no-tests: <why>]`)"
)


def _note(message: str) -> None:
    print(message)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(f"- fix→tests lint: {message}\n")
        except OSError:
            pass


def _split(files: list[str]) -> tuple[bool, bool]:
    return any(f.startswith("src/") for f in files), any(f.startswith("tests/") for f in files)


def violations(
    cwd: Path = REPO_ROOT, base: str | None = None, *, pr_title: str | None = None, pr_body: str = ""
) -> list[str]:
    """Violation messages for the PR title (when given) and for each branch commit."""
    base = base or base_ref(cwd)
    out: list[str] = []

    if pr_title and FIX_SUBJECT.match(pr_title.strip()):
        files = git(cwd, "diff", "--name-only", f"{base}...HEAD").split()
        touches_src, touches_tests = _split(files)
        inline = OPT_OUT_INLINE.search(pr_title) or OPT_OUT_INLINE.search(pr_body)
        if touches_src and not touches_tests:
            if inline:
                _note(f"PR title `{pr_title[:60]}` touches src/ without tests/ — declared: {inline.group(1).strip()}")
            else:
                out.append(
                    f"PR title `{pr_title[:72]}` — the squash-merged subject on main — touches src/ "
                    f"({sum(f.startswith('src/') for f in files)} file(s)) without tests/: {ADVICE}"
                )

    for sha in git(cwd, "rev-list", "--reverse", f"{base}..HEAD").split():
        if len(git(cwd, "rev-list", "--parents", "-n", "1", sha).split()) > 2:
            _note(f"{sha[:8]} is a merge commit — skipped (its own diff-tree is empty)")
            continue
        subject = git(cwd, "log", "-1", "--format=%s", sha).strip()
        if not FIX_SUBJECT.match(subject):
            continue
        files = git(cwd, "diff-tree", "--no-commit-id", "--name-only", "-r", sha).split()
        touches_src, touches_tests = _split(files)
        if not touches_src or touches_tests:
            continue
        body = git(cwd, "log", "-1", "--format=%b", sha)
        # The PR title/body counts here too. The docstring above has always
        # promised "a `No-Tests-Reason:` trailer in the commit body, OR
        # `[no-tests: <why>]` in the PR title/body" — but the marker was only
        # ever read for the PR-title population, so the documented escape did
        # not work for the population that actually fails (found 2026-08-31,
        # PR #579). A promised escape that silently does not apply is worse
        # than no escape: the author reads the advice, follows it, and the gate
        # stays red with the same message.
        #
        # This does NOT weaken the rule. The marker still demands a written
        # reason and is still echoed to stdout and $GITHUB_STEP_SUMMARY, and the
        # PR body is a REVIEWED artifact — more visible to a reviewer than a
        # trailer buried in one commit of a stack. House convention stands: this
        # lint catches forgetting, not evasion.
        m = (
            OPT_OUT_TRAILER.search(body)
            or OPT_OUT_INLINE.search(body)
            or OPT_OUT_INLINE.search(pr_title or "")
            or OPT_OUT_INLINE.search(pr_body)
        )
        if m:
            _note(f"{sha[:8]} `{subject[:60]}` touches src/ without tests/ — declared: {m.group(1).strip()}")
            continue
        out.append(
            f"{sha[:8]} `{subject[:72]}` touches src/ ({sum(f.startswith('src/') for f in files)} file(s)) "
            f"without tests/: {ADVICE}"
        )
    return out


def main() -> int:
    try:
        base = base_ref(REPO_ROOT)
    except GitUnavailable as exc:
        if must_not_skip(str(exc)):
            return 2
        print(f"INFO: no base ref (origin/main) available; skipping fix→tests lint ({exc})")
        return 0
    try:
        fails = violations(
            REPO_ROOT,
            base,
            pr_title=os.environ.get("PR_TITLE") or None,
            pr_body=os.environ.get("PR_BODY", ""),
        )
        n_commits = len(git(REPO_ROOT, "rev-list", f"{base}..HEAD").split())
    except GitUnavailable as exc:
        print(f"INFO: fix→tests lint skipped mid-run ({exc})")
        return 0
    except OSError as exc:
        print(f"ERROR: fix→tests lint could not run: {exc}", file=sys.stderr)
        return 2
    if fails:
        print("fix→tests lint FAILED:", file=sys.stderr)
        for f in fails:
            print(f"  {f}", file=sys.stderr)
        return 1
    scope = "PR title + " if os.environ.get("PR_TITLE") else ""
    print(f"fix→tests lint: clean ({scope}{n_commits} commit(s) on this branch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
