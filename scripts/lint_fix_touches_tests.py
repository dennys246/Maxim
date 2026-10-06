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

**On a push to main** (#1089) there is no PR title or body: each unit that landed is judged
against its own merged PR, read through the GitHub API (``_lint_git.push_units``).

Exits: 0 clean; 1 violations (stderr); 2 unexpected error, or no base / a mid-run git failure on
a pull request or push. Locally, no base ref (a clone without origin/main) SKIPS with an INFO.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_git import GitUnavailable, base_ref, git, must_not_skip, push_units  # noqa: E402

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


def _title_violation(cwd: Path, diff_range: str, pr_title: str | None, pr_body: str) -> str | None:
    """The PR-title rule: a ``fix`` title whose aggregate ``diff_range`` touches src/ without tests/."""
    if not (pr_title and FIX_SUBJECT.match(pr_title.strip())):
        return None
    files = git(cwd, "diff", "--name-only", diff_range).split()
    touches_src, touches_tests = _split(files)
    if not touches_src or touches_tests:
        return None
    inline = OPT_OUT_INLINE.search(pr_title) or OPT_OUT_INLINE.search(pr_body)
    if inline:
        _note(f"PR title `{pr_title[:60]}` touches src/ without tests/ — declared: {inline.group(1).strip()}")
        return None
    return (
        f"PR title `{pr_title[:72]}` — the squash-merged subject on main — touches src/ "
        f"({sum(f.startswith('src/') for f in files)} file(s)) without tests/: {ADVICE}"
    )


def _commit_violation(cwd: Path, sha: str, pr_title: str | None, pr_body: str) -> str | None:
    """The per-commit rule for one commit; merge commits are skipped with a note."""
    if len(git(cwd, "rev-list", "--parents", "-n", "1", sha).split()) > 2:
        _note(f"{sha[:8]} is a merge commit — skipped (its own diff-tree is empty)")
        return None
    subject = git(cwd, "log", "-1", "--format=%s", sha).strip()
    if not FIX_SUBJECT.match(subject):
        return None
    files = git(cwd, "diff-tree", "--no-commit-id", "--name-only", "-r", sha).split()
    touches_src, touches_tests = _split(files)
    if not touches_src or touches_tests:
        return None
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
        return None
    return (
        f"{sha[:8]} `{subject[:72]}` touches src/ ({sum(f.startswith('src/') for f in files)} file(s)) "
        f"without tests/: {ADVICE}"
    )


def violations(
    cwd: Path = REPO_ROOT, base: str | None = None, *, pr_title: str | None = None, pr_body: str = ""
) -> list[str]:
    """Violation messages for the PR title (when given) and for each branch commit."""
    base = base or base_ref(cwd)
    found = [_title_violation(cwd, f"{base}...HEAD", pr_title, pr_body)]
    found += [
        _commit_violation(cwd, sha, pr_title, pr_body)
        for sha in git(cwd, "rev-list", "--reverse", f"{base}..HEAD").split()
    ]
    return [v for v in found if v]


def push_violations(cwd: Path, base: str) -> list[str]:
    """On a push to main there is no PR_TITLE/PR_BODY: each unit that landed is judged against ITS merged PR's
    title and body (#1089, owner decision 2026-10-04), so the ``[no-tests]`` opt-out keeps working; a direct push
    has no PR and so no opt-out but a commit's own trailer. A rebase-merged PR lands as several first-parent
    commits, so the title rule reads that PR's aggregate diff."""
    units = push_units(cwd, base)
    found: list[str | None] = []
    seen: set[object] = set()
    for i, unit in enumerate(units):
        title = unit.pr["title"] if unit.pr else None
        body = unit.pr["body"] if unit.pr else ""
        number = unit.pr["number"] if unit.pr else None
        if number is not None and number not in seen:
            seen.add(number)
            mine = [u for u in units if u.pr and u.pr["number"] == number]
            first_parent = git(cwd, "rev-parse", f"{mine[0].sha}^1").strip()
            found.append(_title_violation(cwd, f"{first_parent}..{mine[-1].sha}", title, body))
        found += [_commit_violation(cwd, sha, title, body) for sha in unit.commits]
    return [v for v in found if v]


def main() -> int:
    try:
        base = base_ref(REPO_ROOT)
    except GitUnavailable as exc:
        if must_not_skip(str(exc)):
            return 2
        print(f"INFO: no base ref (origin/main) available; skipping fix→tests lint ({exc})")
        return 0
    try:
        if os.environ.get("GITHUB_EVENT_NAME") == "push":
            fails = push_violations(REPO_ROOT, base)
        else:
            fails = violations(
                REPO_ROOT,
                base,
                pr_title=os.environ.get("PR_TITLE") or None,
                pr_body=os.environ.get("PR_BODY", ""),
            )
        n_commits = len(git(REPO_ROOT, "rev-list", f"{base}..HEAD").split())
    except GitUnavailable as exc:
        # Was a silent `return 0`: a fail-open path that a push gate would make reachable (#1089 review).
        if must_not_skip(f"git failed mid-run: {exc}"):
            return 2
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
